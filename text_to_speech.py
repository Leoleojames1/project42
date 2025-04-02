#!/usr/bin/env python3
"""
Text to speech module for Project 42
Handles sentence splitting and speech synthesis
"""

import re
import io
import os
import tempfile
import threading
import queue
import pygame
from gtts import gTTS
import time
import concurrent.futures
from collections import deque
import numpy as np

class TextToSpeech:
    """
    Advanced text-to-speech manager with parallel processing pipeline
    
    Features:
    - Parallel generation of speech chunks while playing
    - Sophisticated sentence boundary detection
    - Support for speech interruption
    - Special handling of phonetic markers and abbreviations
    - Dynamic speech rate adjustment
    """
    
    def __init__(self):
        """Initialize advanced text to speech manager"""
        # Initialize pygame mixer for audio playback
        pygame.mixer.init(frequency=24000)
        
        # Speech settings
        self.speech_interrupted = False
        self.lang = 'en'
        self.speech_queue = queue.Queue()
        self.audio_buffer = deque()
        
        # Concurrency management
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=3)
        self.audio_buffer_lock = threading.RLock()
        self.generation_futures = []
        
        # Status flags
        self.is_generating = False
        self.is_playing = False
        self.active_future = None
        
        # Start speech processing thread
        self.processing_thread = threading.Thread(
            target=self.process_speech_queue, 
            daemon=True
        )
        self.processing_thread.start()
        
        # Start audio playback thread
        self.playback_thread = threading.Thread(
            target=self.audio_playback_loop,
            daemon=True
        )
        self.playback_thread.start()
        
        # Performance metrics
        self.generation_times = []
        self.playback_times = []
        
        # Add visualization support
        self.visualizer_samples = []
        self.is_visualizing = False
        
    def split_into_sentences(self, text):
        """
        Advanced sentence splitting with phonetic markers and abbreviation handling
        
        Args:
            text (str): The input text to split
            
        Returns:
            list: List of well-formed sentences for natural TTS
        """
        if not text or not text.strip():
            return []
            
        # Add spaces around punctuation marks for consistent splitting
        text = " " + text + " "
        text = text.replace("\n", " ")

        # Handle common abbreviations and special cases
        text = re.sub(r"(Mr|Mrs|Ms|Dr|Prof|Gen|Rep|Sen|St|Jr|Sr|Ph\.D|i\.e|e\.g)\.", r"\1<prd>", text)
        text = re.sub(r"\.\.\.", r"<prd><prd><prd>", text)
        text = re.sub(r"\s+", " ", text)  # Normalize whitespace
        
        # Handle special markers for emotional content
        # These would be handled differently in Coqui TTS, but we'll preserve them for Google TTS
        text = re.sub(r"\*([^*]+)\*", r" <break time='500ms'/> \1 <break time='500ms'/> ", text)

        # Split on period, question mark, exclamation mark, or colon followed by spaces
        sentences = re.split(r"(?<=\d\.)\s+|(?<=[.!?:])\s+", text)

        # Remove empty sentences and restore periods
        sentences = [s.strip().replace("<prd>", ".") for s in sentences if s.strip()]

        # Combine numbered list items with their content
        combined_sentences = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and re.match(r"^\d+\.$", sentences[i]):
                combined_sentences.append(f"{sentences[i]} {sentences[i + 1]}")
                i += 2
            else:
                combined_sentences.append(sentences[i])
                i += 1

        # Further split long sentences for better TTS flow
        result = []
        for sentence in combined_sentences:
            if len(sentence) > 150:
                # If sentence is too long, split by commas or other breaks
                parts = re.split(r'(?<=[,;])\s+', sentence)
                for part in parts:
                    if len(part) > 150:
                        # If part is still too long, force split
                        while len(part) > 150:
                            split_index = part.find(' ', 100, 150)
                            if split_index == -1:
                                split_index = 150
                            result.append(part[:split_index].strip())
                            part = part[split_index:].strip()
                        if part:
                            result.append(part)
                    else:
                        result.append(part)
            else:
                result.append(sentence)
                
        return result
    
    def speak_text(self, text, lang='en', wait_for_completion=True):
        """
        Convert text to speech using parallel processing pipeline
        
        Args:
            text (str): Text to speak
            lang (str): Language code
            wait_for_completion (bool): Whether to wait for speech to complete
            
        Returns:
            int: Number of sentences queued
        """
        # Reset interruption flag
        self.speech_interrupted = False
        
        # Split text into natural sentences
        sentences = self.split_into_sentences(text)
        if not sentences:
            return 0
            
        print(f"Processing {len(sentences)} sentences")
        
        # Cancel any pending generation tasks
        self.cancel_pending_generations()
        
        # Clear audio buffer
        with self.audio_buffer_lock:
            self.audio_buffer.clear()
        
        # Start parallel generation of all sentences
        self.is_generating = True
        self.generation_futures = []
        
        # Submit first sentence with priority
        first_future = self.executor.submit(self.generate_audio_chunk, sentences[0], lang, 0)
        self.generation_futures.append(first_future)
        
        # Submit remaining sentences
        for i, sentence in enumerate(sentences[1:], 1):
            future = self.executor.submit(self.generate_audio_chunk, sentence, lang, i)
            self.generation_futures.append(future)
        
        # If immediate return requested, don't wait
        if not wait_for_completion:
            return len(sentences)
            
        # Wait for all audio to finish playing
        while (self.is_generating or self.is_playing) and not self.speech_interrupted:
            time.sleep(0.1)
            
        return len(sentences)
    
    def generate_audio_chunk(self, text, lang, index):
        """
        Generate audio for a single chunk of text
        
        Args:
            text (str): Text to generate audio for
            lang (str): Language code
            index (int): Sentence index for ordering
            
        Returns:
            tuple: (index, BytesIO object containing audio data)
        """
        if self.speech_interrupted:
            return None
            
        start_time = time.time()
        
        try:
            # Generate speech with gTTS
            tts = gTTS(text=text, lang=lang, slow=False)
            
            # Save to BytesIO object
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            
            generation_time = time.time() - start_time
            self.generation_times.append(generation_time)
            print(f"Generated audio for chunk {index} in {generation_time:.2f}s: {text[:30]}...")
            
            # Add to audio buffer in correct order
            with self.audio_buffer_lock:
                # Insert maintaining order by index
                self.audio_buffer.append((index, fp))
                # Sort buffer by index
                self.audio_buffer = deque(sorted(self.audio_buffer, key=lambda x: x[0]))
            
            return index, fp
            
        except Exception as e:
            print(f"Error generating audio for chunk {index}: {e}")
            return None
    
    def audio_playback_loop(self):
        """Background thread that plays audio chunks in order"""
        last_played_index = -1
        
        while True:
            try:
                # Check if we have the next chunk to play
                next_chunk = None
                with self.audio_buffer_lock:
                    if self.audio_buffer and self.audio_buffer[0][0] == last_played_index + 1:
                        next_chunk = self.audio_buffer.popleft()
                
                if next_chunk:
                    index, audio_data = next_chunk
                    
                    if self.speech_interrupted:
                        continue
                    
                    # Play the audio
                    self.is_playing = True
                    start_time = time.time()
                    
                    # Stop any currently playing audio
                    pygame.mixer.music.stop()
                    
                    # Load and play
                    pygame.mixer.music.load(audio_data)
                    pygame.mixer.music.play()
                    
                    # Wait for it to finish
                    while pygame.mixer.music.get_busy():
                        if self.speech_interrupted:
                            pygame.mixer.music.stop()
                            break
                        pygame.time.Clock().tick(10)
                    
                    # Update last played index
                    last_played_index = index
                    
                    # Track performance
                    playback_time = time.time() - start_time
                    self.playback_times.append(playback_time)
                    print(f"Played chunk {index} in {playback_time:.2f}s")
                    
                    # Update state if we're done
                    if not self.audio_buffer and all(future.done() for future in self.generation_futures):
                        self.is_playing = False
                        last_played_index = -1  # Reset for next utterance
                else:
                    # No chunks ready yet, small wait
                    time.sleep(0.05)
                    
                    # Check if generation is complete but no more chunks
                    if (not self.audio_buffer and 
                        all(future.done() for future in self.generation_futures) and 
                        self.is_generating):
                        self.is_generating = False
                        
            except Exception as e:
                print(f"Error in audio playback: {e}")
                time.sleep(0.1)
    
    def process_speech_queue(self):
        """Process the speech queue for backward compatibility"""
        while True:
            try:
                sentence, lang = self.speech_queue.get(timeout=0.5)
                
                if self.speech_interrupted:
                    self.speech_queue.task_done()
                    continue
                    
                try:
                    # Process in the new way
                    self.speak_text(sentence, lang)
                except Exception as e:
                    print(f"Speech queue processing error: {str(e)}")
                    
                self.speech_queue.task_done()
                
            except queue.Empty:
                # No sentences to process
                pass
            except Exception as e:
                print(f"Queue processing error: {str(e)}")
                time.sleep(0.1)
    
    def chunk_sentence(self, sentence, max_chunk_length=100):
        """
        Break a long sentence into smaller chunks for more responsive TTS
        
        This method is enhanced with more natural breakpoint detection and
        improved handling of special text markers for prosody
        
        Args:
            sentence (str): Sentence to chunk
            max_chunk_length (int): Maximum chunk length
            
        Returns:
            list: List of sentence chunks
        """
        if len(sentence) <= max_chunk_length:
            return [sentence]
            
        chunks = []
        
        # Try to split on natural breakpoints in order of preference
        breakpoints = [
            ',', ';', ':', '—', '-', '(', ')', '[', ']', '{', '}',
            ' and ', ' but ', ' or ', ' nor ', ' yet ', ' so '
        ]
        
        while len(sentence) > max_chunk_length:
            # Find the last breakpoint before max_length
            split_idx = -1
            
            for bp in breakpoints:
                last_bp = sentence[:max_chunk_length].rfind(bp)
                if last_bp > split_idx:
                    split_idx = last_bp
                    if bp in [',', ';', ':', '—', '-']:
                        split_idx += 1  # Include the breakpoint in the chunk
                    elif bp in ['(', '[', '{']:
                        # Try to find matching closing bracket
                        open_char = bp
                        close_map = {'(': ')', '[': ']', '{': '}'}
                        close_char = close_map[open_char]
                        nesting = 1
                        for i in range(last_bp + 1, len(sentence)):
                            if sentence[i] == open_char:
                                nesting += 1
                            elif sentence[i] == close_char:
                                nesting -= 1
                                if nesting == 0:
                                    split_idx = i + 1
                                    break
                    elif ' ' in bp:  # Handle conjunctions
                        split_idx += len(bp)
                    
            # If no good breakpoint, just split on the last space
            if split_idx <= 0:
                split_idx = sentence[:max_chunk_length].rfind(' ')
                
            # If no space found, force split
            if split_idx <= 0:
                split_idx = max_chunk_length
                
            # Add the chunk and continue with remainder
            chunk = sentence[:split_idx].strip()
            if chunk:
                chunks.append(chunk)
            sentence = sentence[split_idx:].strip()
            
        # Add the last part
        if sentence:
            chunks.append(sentence)
            
        return chunks
    
    def interrupt_speech(self):
        """Interrupt current speech playback and generation"""
        print("Interrupting speech...")
        self.speech_interrupted = True
        
        # Stop playback
        pygame.mixer.music.stop()
        
        # Cancel pending generations
        self.cancel_pending_generations()
        
        # Reset state flags
        self.is_playing = False
        self.is_generating = False
        
        # Clear buffers
        with self.audio_buffer_lock:
            self.audio_buffer.clear()
        
        # Clear the queue
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
                self.speech_queue.task_done()
            except queue.Empty:
                break
    
    def cancel_pending_generations(self):
        """Cancel any pending generation tasks"""
        for future in self.generation_futures:
            if not future.done():
                future.cancel()
        self.generation_futures = []
    
    def is_speaking(self):
        """Check if speech is currently playing or processing"""
        return self.is_playing or self.is_generating or not self.speech_queue.empty()
        
    def stop(self):
        """Stop all speech and clean up resources"""
        self.interrupt_speech()
        # Shutdown executor - comment out if you want to keep it alive for future use
        # self.executor.shutdown(wait=False)
    
    def get_metrics(self):
        """Return performance metrics"""
        avg_gen_time = sum(self.generation_times) / len(self.generation_times) if self.generation_times else 0
        avg_play_time = sum(self.playback_times) / len(self.playback_times) if self.playback_times else 0
        
        return {
            "avg_generation_time": avg_gen_time,
            "avg_playback_time": avg_play_time,
            "generation_samples": len(self.generation_times),
            "playback_samples": len(self.playback_times)
        }
    
    def get_audio_samples(self):
        """Return audio samples for visualization"""
        if not self.is_playing:
            return np.zeros(100)
        
        # Return the current audio samples if available
        if hasattr(self, "visualizer_samples") and len(self.visualizer_samples) > 0:
            return np.array(self.visualizer_samples)
        
        # Fill visualizer_samples with simulated audio when nothing else is available
        self.visualizer_samples = np.sin(np.linspace(0, 3*np.pi, 100)) * 0.5 * self.is_playing
        return self.visualizer_samples

    def set_output_device(self, device_index, volume=0.8):
        """Set the output device for TTS playback
        
        Args:
            device_index: PyAudio device index for output
            volume: Volume level from 0.0 to 1.0
        """
        self.output_device_index = device_index
        self.output_volume = volume
        
        try:
            # Update pygame mixer if that's what you're using
            import pygame
            pygame.mixer.quit()
            pygame.mixer.init(devicename=device_index)
            pygame.mixer.music.set_volume(volume)
            print(f"TTS output device set to {device_index} with volume {volume}")
        except Exception as e:
            print(f"Error setting TTS output device: {e}")

    def setup_continuous_mode(self):
        """Setup continuous conversation mode without wake word"""
        # Set continuous mode flag
        self.continuous_mode = True
        self.in_conversation = True
        
        # Start listening immediately in a separate thread
        threading.Thread(target=self.continuous_listening_loop, daemon=True).start()
        
    def continuous_listening_loop(self):
        """Continuous listening loop that doesn't require wake word"""
        self.update_signal.emit("status", "Continuous mode: I'm listening...")
        
        while self.running and hasattr(self, "continuous_mode") and self.continuous_mode:
            # Only listen if not currently processing a response
            if not self.thinking:
                # Start listening with live transcription
                self.live_listening_active = True
                
                # Start background thread for live results
                live_thread = threading.Thread(target=self.live_transcription_preview)
                live_thread.daemon = True
                live_thread.start()
                
                # Listen for speech
                speech_text = self.stt.start_listening_session(
                    non_blocking=False,
                    max_listen_time=10
                )
                
                # Stop live preview
                self.live_listening_active = False
                
                # Process speech if meaningful
                if speech_text and len(speech_text.strip()) > 2:
                    self.update_signal.emit("user", speech_text)
                    self.process_speech_input(speech_text)
                
            # Small pause to prevent CPU overutilization
            time.sleep(0.5)

# Modify SpeechWorker.run method to handle continuous mode

def run(self):
    """Main worker loop for speech interaction with enhanced TTS"""
    # Set default attributes
    self.continuous_mode = False
    self.push_to_talk_active = False
    self.live_listening_active = False
    
    self.update_signal.emit("status", "Waiting for wake word or command...")
    
    # Start the interrupt listener thread
    self.setup_interrupt_listener()
    
    while self.running:
        # If continuous mode is enabled, skip wake word detection
        if hasattr(self, "continuous_mode") and self.continuous_mode:
            # Just check running state and sleep
            time.sleep(0.5)
            continue
            
        # Initial wake word to start conversation mode
        if not self.in_conversation:
            wake_word_detected = self.stt.wait_for_wake_word()
            if wake_word_detected:
                self.in_conversation = True
                self.update_signal.emit("status", "Wake word detected! Listening...")
            else:
                # If we're not running anymore, exit
                if not self.running:
                    break
                continue
        
        # In conversation mode: continuously listen and respond
        while self.in_conversation and self.running:
            # Listen for user speech
            self.update_signal.emit("status", "Listening...")
            
            # Start listening with live transcription
            self.live_listening_active = True
            
            # Start background thread for live results
            live_thread = threading.Thread(target=self.live_transcription_preview)
            live_thread.daemon = True
            live_thread.start()
            
            # Listen for speech
            speech_text = self.stt.start_listening_session(non_blocking=False)
            
            # Stop live preview
            self.live_listening_active = False
            
            # Process recognized speech
            if speech_text and speech_text.strip() != "":
                # Check for exit command
                if speech_text.lower().strip() in ["exit conversation", "exit", "quit", "stop conversation"]:
                    self.in_conversation = False
                    self.update_signal.emit("status", "Conversation ended. Say wake word to start again.")
                    break
                
                # Update UI with recognized text
                self.update_signal.emit("user", speech_text)
                self.update_signal.emit("status", "Processing your request...")
                
                # Process speech
                self.process_speech_input(speech_text)
            else:
                self.update_signal.emit("status", "Didn't catch that. Try again?")
        
        # Add this to emit input audio samples periodically
        if hasattr(self, "stt") and hasattr(self.stt, "samples_buffer") and len(self.stt.samples_buffer) > 0:
            input_samples = np.array(self.stt.samples_buffer)
            self.stt_audio_signal.emit(input_samples)
    
    self.finished_signal.emit()