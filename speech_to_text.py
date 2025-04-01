#!/usr/bin/env python3
"""
Speech recognition module for Project 42
Handles silence detection, wake words and speech-to-text functionality
"""

import os
import pyaudio
import wave
import audioop
import tempfile
import threading
import queue
import speech_recognition as sr
import numpy as np
from datetime import datetime
import time

class SpeechToText:
    """
    Manages speech recognition with wake word detection and
    silence-based recording control.
    
    Features:
    - Wake word detection
    - Silence detection for natural recording boundaries
    - Google Speech API integration
    - Automatic speech chunking
    """
    
    def __init__(self, wake_word="Eddie"):
        """
        Initialize speech recognition manager
        
        Args:
            wake_word (str): The wake word to activate the assistant
        """
        # Core settings
        self.wake_word = wake_word
        self.is_listening = False
        self.is_active = True
        self.speech_interrupted = False
        
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.energy_threshold = 450  # Slightly lower threshold for better sensitivity
        
        # Configure audio settings
        self.FORMAT = pyaudio.paInt32
        self.CHANNELS = 1
        self.RATE = 44100
        self.CHUNK = 2
        self.SILENCE_THRESHOLD = 550  # Adjustable silence threshold
        self.SILENCE_DURATION = 0.8  # Longer duration for more natural pauses
        
        # Queue system for processing
        self.text_queue = queue.Queue()
        self.audio_queue = queue.Queue()
        
        # Audio buffers
        self.frames = []
        
        # Add dynamic silence detection
        self.rms_history = []
        self.rms_history_max_size = 20
        self.background_noise_level = 0
        self.in_speech = False
        self.speech_start_time = 0

        # Add whisper detection settings
        self.whisper_mode = False
        self.whisper_threshold = 300  # Lower threshold for whispers
        self.normal_threshold = 550   # Regular threshold
        self.current_threshold = self.normal_threshold
        self.samples_buffer = []      # Buffer for audio visualization

        # Add this line:
        self.current_partial_result = ""
        
        # Add support for partial results
        self.partial_results_enabled = True
        
    def listen(self, threshold=None, silence_duration=0.8, non_blocking=False, max_listen_time=5):
        """
        Listen for audio with improved whisper detection
        
        Args:
            threshold (int, optional): RMS threshold for silence detection
            silence_duration (float): Duration of silence to mark end of speech
            non_blocking (bool): Whether to return quickly if no speech is detected
            max_listen_time (int): Maximum listening time in seconds for non-blocking mode
            
        Returns:
            str: Path to temporary WAV file containing recorded audio or None
        """
        # Use provided threshold or current mode's threshold
        used_threshold = threshold if threshold is not None else self.current_threshold
        
        audio = pyaudio.PyAudio()
        try:
            stream = audio.open(
                format=self.FORMAT, 
                channels=self.CHANNELS,
                rate=self.RATE, 
                input=True, 
                frames_per_buffer=self.CHUNK
            )
        except IOError:
            print("Error: Could not access the microphone.")
            audio.terminate()
            return None

        frames = []
        silent_frames = 0
        sound_detected = False
        start_time = time.time()
        dynamic_threshold = used_threshold
        
        # Clear samples buffer for visualization
        self.samples_buffer = []
        
        # Timing variables for speech pattern recognition
        self.in_speech = False
        speech_duration = 0
        
        print(f"Listening with threshold {dynamic_threshold}...")

        while self.is_active:
            try:
                data = stream.read(self.CHUNK, exception_on_overflow=False)
                frames.append(data)

                # Calculate RMS to detect silence
                rms = audioop.rms(data, 2)
                
                # Store RMS for visualization - MAKE SURE THIS IS WORKING
                sample_value = rms / 1000.0  # Normalize for visualization
                self.samples_buffer.append(sample_value)  
                if len(self.samples_buffer) > 100:
                    self.samples_buffer = self.samples_buffer[-100:]  # Keep last 100 samples
                
                # Update dynamic threshold based on ambient noise
                dynamic_threshold = self.update_silence_threshold(rms)
                
                # Apply whisper mode adjustment if needed
                if self.whisper_mode and dynamic_threshold > self.whisper_threshold:
                    dynamic_threshold = max(self.whisper_threshold, dynamic_threshold * 0.6)

                # Speech state machine
                if rms > dynamic_threshold:
                    # Sound detected
                    if not self.in_speech:
                        # Transition to speech
                        self.in_speech = True
                        self.speech_start_time = time.time()
                        print(f"Speech detected with RMS {rms} (threshold: {dynamic_threshold})")
                    
                    silent_frames = 0
                    sound_detected = True
                    speech_duration = time.time() - self.speech_start_time
                    
                    # Add this section for live transcription:
                    if len(frames) % 10 == 0 and len(frames) > 20 and self.partial_results_enabled:
                        # Every 10 frames, try to get a partial transcription
                        self.update_partial_result(frames)
                else:
                    # Silence detected
                    silent_frames += 1
                    
                    # If in speech and silent for too long, transition out
                    if self.in_speech and (silent_frames * (self.CHUNK / self.RATE) > silence_duration):
                        self.in_speech = False
                        print(f"Speech ended after {speech_duration:.2f}s")
                        
                        # If we had a meaningful speech segment, stop recording
                        if speech_duration > 0.3:  # Shorter minimum for whispers
                            break

                # Check if we've been listening too long with no meaningful input
                elapsed_time = time.time() - start_time
                
                # For non-blocking mode, return quicker if nothing is heard
                if non_blocking and elapsed_time > max_listen_time and not sound_detected:
                    print("No speech detected in non-blocking mode")
                    stream.stop_stream()
                    stream.close()
                    audio.terminate()
                    return None
                    
                # Check for wake word while listening (for interruption)
                # This check is optimized to run less frequently to reduce processing load
                if sound_detected and len(frames) % 10 == 0 and self.check_for_wake_word(frames[-50:]):
                    print("Wake word detected during listening!")
                    self.speech_interrupted = True
                    break
                    
                # Prevent recording forever if no significant sound is detected
                if len(frames) > int(15 * self.RATE / self.CHUNK):  # 15 seconds max
                    if not sound_detected:
                        print("No significant speech detected within time limit.")
                        audio.terminate()
                        return None
                    break
                    
            except KeyboardInterrupt:
                print("Listening interrupted by user.")
                break
            except Exception as e:
                print(f"Error during recording: {e}")
                break

        stream.stop_stream()
        stream.close()
        audio.terminate()

        if sound_detected and len(frames) > 10:  # At least some frames with sound
            # Save to temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames))
            
            return temp_file.name
        return None

    def recognize_speech(self, audio_file):
        """
        Recognize speech from audio file with improved error handling
        """
        if not audio_file:
            return ""
        
        try:
            with sr.AudioFile(audio_file) as source:
                audio_data = self.recognizer.record(source)
                
            # Try to recognize with Google Speech API
            result = self.recognizer.recognize_google(audio_data)
            print(f"Recognized text: '{result}'")  # Add clear debug output
            return result
            
        except sr.UnknownValueError:
            print("RECOGNITION ERROR: Could not understand audio")
            return "Sorry, I couldn't understand what you said."

    def wait_for_wake_word(self):
        """
        Wait for wake word activation
        
        Returns:
            bool: True if wake word detected, False otherwise
        """
        print(f"Waiting for wake word: '{self.wake_word}'")
        while self.is_active:
            temp_file = self.listen()
            if temp_file:
                speech_text = self.recognize_speech(temp_file).lower()
                print(f"Heard: {speech_text}")
                
                if self.wake_word.lower() in speech_text:
                    print("Wake word detected!")
                    return True
        return False
                    
    def start_listening_session(self, non_blocking=False):
        """
        Start a listening session, optionally in non-blocking mode
        
        Args:
            non_blocking (bool): Whether to return immediately if no speech is detected
            
        Returns:
            str: Recognized speech
        """
        self.is_listening = True
        temp_file = self.listen(non_blocking=non_blocking)
        if temp_file:
            speech_text = self.recognize_speech(temp_file)
            return speech_text
        self.is_listening = False
        return ""
    
    def set_wake_word(self, wake_word):
        """
        Set a new wake word
        
        Args:
            wake_word (str): New wake word
        """
        self.wake_word = wake_word.strip()
        print(f"Wake word set to: '{self.wake_word}'")
        
    def interrupt_listening(self):
        """Interrupt current listening session"""
        self.speech_interrupted = True
        self.is_listening = False
    
    def stop(self):
        """Stop all listening activities"""
        self.is_active = False
        self.is_listening = False
        self.speech_interrupted = True

    def check_for_wake_word(self, frames):
        """
        Check if the wake word is present in the current audio frames
        
        Args:
            frames (list): List of audio frames to check
            
        Returns:
            bool: True if wake word detected, False otherwise
        """
        # Only check if we have enough frames to make it worthwhile
        if len(frames) < 20:
            return False
            
        try:
            # Save frames to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio = pyaudio.PyAudio()
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames[-50:]))  # Check last ~1 second
                
            audio.terminate()
            
            # Recognize speech in the temporary file
            with sr.AudioFile(temp_file.name) as source:
                audio_data = self.recognizer.record(source)
                
            try:
                # Use Google Speech Recognition with shorter timeout
                result = self.recognizer.recognize_google(audio_data)
                
                # Check if wake word is in the result
                if self.wake_word.lower() in result.lower():
                    print(f"Wake word '{self.wake_word}' detected in: {result}")
                    return True
                    
            except sr.UnknownValueError:
                # No speech detected or couldn't understand
                pass
            except sr.RequestError:
                # API error
                pass
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_file.name)
                except:
                    pass
                    
        except Exception as e:
            print(f"Error checking for wake word: {e}")
            
        return False

    def update_silence_threshold(self, rms):
        """
        Dynamically adjust silence threshold based on ambient noise
        
        Args:
            rms (int): Current RMS value
            
        Returns:
            int: Updated silence threshold
        """
        # Keep history of RMS values for adaptive thresholding
        self.rms_history.append(rms)
        if len(self.rms_history) > self.rms_history_max_size:
            self.rms_history.pop(0)
        
        # Calculate noise floor
        if not self.in_speech:
            # Update background noise when not in speech
            noise_samples = self.rms_history[-5:] if len(self.rms_history) >= 5 else self.rms_history
            self.background_noise_level = sum(noise_samples) / len(noise_samples)
        
        # Adaptive threshold - higher when background noise is higher
        dynamic_threshold = max(500, self.background_noise_level * 1.5)
        
        # Add hysteresis - once speech is detected, make it easier to continue detecting
        if self.in_speech:
            # Lower threshold during active speech to avoid cutting off quiet parts
            return dynamic_threshold * 0.7
        else:
            # Higher threshold when not in speech to avoid triggering on noise
            return dynamic_threshold * 1.2

    def toggle_whisper_mode(self, enabled=None):
        """
        Toggle between whisper and normal detection modes
        
        Args:
            enabled (bool, optional): Set to True for whisper mode, False for normal
                or None to toggle current state
        """
        if enabled is None:
            self.whisper_mode = not self.whisper_mode
        else:
            self.whisper_mode = enabled
            
        # Update threshold based on mode
        if self.whisper_mode:
            self.current_threshold = self.whisper_threshold
            print("Whisper mode enabled - listening for quiet speech")
        else:
            self.current_threshold = self.normal_threshold
            print("Normal speech mode enabled")
            
        return self.whisper_mode

    def get_audio_samples(self):
        """Return audio samples for visualization"""
        if len(self.samples_buffer) > 0:
            return np.array(self.samples_buffer)
        return np.zeros(100)

    def update_partial_result(self, frames):
        """Attempt to get partial transcription from current audio buffer"""
        try:
            # Save frames to a temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio = pyaudio.PyAudio()
            
            with wave.open(temp_file.name, 'wb') as wf:
                wf.setnchannels(self.CHANNELS)
                wf.setsampwidth(audio.get_sample_size(self.FORMAT))
                wf.setframerate(self.RATE)
                wf.writeframes(b''.join(frames[-50:]))  # Use last second of audio
                    
            audio.terminate()
            
            # Try to recognize the speech
            with sr.AudioFile(temp_file.name) as source:
                audio_data = self.recognizer.record(source)
                
            try:
                result = self.recognizer.recognize_google(audio_data)
                if result:
                    self.current_partial_result = result
                    print(f"Partial result: {result}")
            except sr.UnknownValueError:
                # No recognizable speech
                pass
            except sr.RequestError:
                # API error
                pass
            finally:
                # Clean up temp file
                try:
                    os.remove(temp_file.name)
                except:
                    pass
        except Exception as e:
            print(f"Error getting partial result: {e}")

class SpeechWorker:
    def __init__(self, stt):
        self.stt = stt
        self.live_listening_active = False

    def run(self):
        # Listen for user speech
        self.update_signal.emit("status", "Listening...")

        # Start listening but show interim results
        self.live_listening_active = True
        self.stt.partial_results_enabled = True  # Enable partial results

        # Start background listening thread to show live results
        live_thread = threading.Thread(target=self.live_transcription_preview)
        live_thread.daemon = True
        live_thread.start()

        # Do the actual listening
        speech_text = self.stt.start_listening_session(non_blocking=False)

        # Stop live preview
        self.live_listening_active = False
        self.stt.partial_results_enabled = False  # Disable partial results

        wake_word_detected = self.stt.wait_for_wake_word()
        if wake_word_detected:
            self.in_conversation = True
            # Update UI with clear wake word detection message
            self.update_signal.emit("status", "Wake word detected! I'm listening...")
            # Add visual feedback
            self.update_signal.emit("wake_word_detected", True)

class MainWindow:
    def update_from_worker(self, update_type, content):
        """
        Update the UI based on worker signals
        
        Args:
            update_type (str): Type of update (e.g., 'status', 'transcript', 'partial_transcript')
            content (str): Content of the update
        """
        if update_type == "status":
            self.status_display.setText(content)
        elif update_type == "transcript":
            self.output_display.append(content)
        elif update_type == "partial_transcript":
            # Show partial transcript in status bar with different style
            self.status_display.setText(f"Recognizing: {content}")
            self.statusBar().showMessage(f"Hearing: {content}", 2000)
            
            # Also add it to the output display with distinguishing style
            timestamp = datetime.now().strftime("%H:%M:%S")
            partial_html = f"<p><span style='color: #AAAAAA;'>[{timestamp}] Recognizing:</span> <i>{content}</i></p>"
            
            # Update or add partial text
            current_html = self.output_display.toHtml()
            if "Recognizing:</span>" in current_html:
                # Replace existing partial text
                parts = current_html.split("Recognizing:</span>")
                before = parts[0] + "Recognizing:</span>"
                after = parts[1].split("</p>", 1)[1] if "</p>" in parts[1] else ""
                self.output_display.setHtml(before + f" <i>{content}</i></p>" + after)
            else:
                # Add new partial text
                self.output_display.append(partial_html)

    def init_ui(self):
        # In MainWindow.init_ui where you create the visualizers
        # Create input visualizer with settings button
        self.input_visualizer = AudioWaveformVisualizer(mode="input", show_settings=False)  # No settings here
        self.input_visualizer.setMinimumHeight(80)
        visualizer_container.addWidget(self.input_visualizer)

        # Create output visualizer without settings button (or the reverse if you prefer)
        self.output_visualizer = AudioWaveformVisualizer(mode="output", show_settings=True)  # Only show settings here
        self.output_visualizer.setMinimumHeight(80)
        visualizer_container.addWidget(self.output_visualizer)

        # In the MainWindow.init_ui method, add clear wake word indicator
        self.statusBar().showMessage(f"Say wake word: '{self.stt.wake_word}' to begin")

class AudioWaveformVisualizer:
    def __init__(self, parent=None, mode="input", show_settings=True):
        # Existing initialization...
        
        # Change this line to use the parameter
        self.show_settings_button = show_settings