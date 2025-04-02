#!/usr/bin/env python3
"""
Project 42 - Speech-to-Speech Digital Assistant
A dark-themed digital assistant that uses Google speech recognition and text-to-speech,
integrated with Ollama for LLM inference and pandas for conversation history management.
"""

import sys
import os
import threading
import time
import queue
import json
import pandas as pd
from datetime import datetime
import tempfile
import io
import asyncio
import uuid
import wave

# Add these imports near the top with your other imports (around line 20)
import re
import queue
import uuid
import tempfile
import os
import pandas as pd
import io
import pyaudio
import wave
import audioop
import numpy as np
import pygame
import pyaudio
import audioop
import numpy as np

# Ollama import
import ollama
# Google Speech Recognition STT
import speech_recognition as sr
# Google Text-to-Speech TTS
from gtts import gTTS

from PyQt6 import sip

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QComboBox, QLabel, 
                           QTextEdit, QTabWidget, QSplitter, QTableWidget, 
                           QTableWidgetItem, QHeaderView, QScrollArea, QFrame,
                           QDialog, QFileDialog, QListWidget, QListWidgetItem,
                           QMessageBox, QMenu, QInputDialog, QLineEdit, QProgressBar, QCheckBox)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QUrl
from PyQt6.QtGui import QColor, QPalette, QFont, QIcon, QAction

# Speech imports
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech

# First, add the import for our new component
from audio_visualizer import AudioWaveformVisualizer

# Add this import near the top with your other imports
from conversation_manager import ConversationManager

class StyleSheet:
    """Style constants for dark theme UI"""
    
    DARK_THEME = """
    QMainWindow, QDialog {
        background-color: #0a0a0a;
        color: #f0f0f0;
    }
    QWidget {
        background-color: #0a0a0a;
        color: #f0f0f0;
    }
    QSplitter::handle {
        background-color: #333333;
    }
    QTextEdit, QTableWidget {
        background-color: #121212;
        color: #e0e0e0;
        border: 1px solid #333333;
        border-radius: 8px;
        padding: 6px;
        selection-background-color: #2d5cbd;
    }
    QPushButton {
        background-color: #1e1e1e;
        color: #f0f0f0;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 8px 16px;
        min-height: 24px;
        font-weight: bold;
    }
    QPushButton:hover {
        background-color: #2e2e2e;
        border: 1px solid #555555;
        color: #3d87ff;
    }
    QPushButton:pressed {
        background-color: #404040;
        color: #6ba5ff;
    }
    QPushButton:disabled {
        background-color: #1a1a1a;
        color: #666666;
        border: 1px solid #333333;
    }
    QPushButton#start_button {
        background-color: #1e3a6d;
        color: #ffffff;
    }
    QPushButton#start_button:hover {
        background-color: #254b8a;
        color: #ffffff;
        border: 1px solid #3d6bbd;
    }
    QPushButton#stop_button {
        background-color: #6d1e1e;
        color: #ffffff;
    }
    QPushButton#stop_button:hover {
        background-color: #8a2525;
        color: #ffffff;
        border: 1px solid #bd3d3d;
    }
    QComboBox {
        background-color: #1e1e1e;
        color: #f0f0f0;
        border: 1px solid #444444;
        border-radius: 8px;
        padding: 6px 12px;
        min-height: 24px;
    }
    QComboBox::drop-down {
        border: 0;
    }
    QComboBox::down-arrow {
        width: 16px;
        height: 16px;
    }
    QComboBox QAbstractItemView {
        background-color: #1e1e1e;
        color: #f0f0f0;
        selection-background-color: #2a2a2a;
        border: 1px solid #444444;
        border-radius: 4px;
    }
    QHeaderView::section {
        background-color: #1e1e1e;
        color: #f0f0f0;
        padding: 6px;
        border: 1px solid #444444;
    }
    QTabWidget::pane {
        border: 1px solid #444444;
        border-radius: 8px;
    }
    QTabBar::tab {
        background-color: #1e1e1e;
        color: #f0f0f0;
        border: 1px solid #444444;
        border-bottom-color: #444444;
        border-top-left-radius: 8px;
        border-top-right-radius: 8px;
        min-width: 10ex;
        padding: 8px 16px;
    }
    QTabBar::tab:selected {
        background-color: #2a2a2a;
        border-bottom-color: #2a2a2a;
        color: #3d87ff;
    }
    QTabBar::tab:!selected {
        margin-top: 2px;
    }
    QScrollBar:vertical {
        border: 0;
        background: #121212;
        width: 14px;
        margin: 0;
    }
    QScrollBar::handle:vertical {
        background: #424242;
        min-height: 20px;
        border-radius: 7px;
    }
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
        border: 0;
        background: none;
    }
    QScrollBar:horizontal {
        border: 0;
        background: #121212;
        height: 14px;
        margin: 0;
    }
    QScrollBar::handle:horizontal {
        background: #424242;
        min-width: 20px;
        border-radius: 7px;
    }
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
        border: 0;
        background: none;
    }
    QLabel {
        color: #f0f0f0;
        font-weight: 500;
    }
    QTableWidget {
        gridline-color: #2a2a2a;
    }
    QStatusBar {
        background-color: #0a0a0a;
        color: #f0f0f0;
    }
    
    /* Glow effects for specific elements without box-shadow */
    #output_display {
        border: 2px solid #3d87ff;
        border-radius: 10px;
        background-color: #0d0d15;
    }
    #status_display {
        border: 2px solid #3d87ff;
        border-radius: 8px;
        background-color: #0d0d15;
    }
    #conversation_display {
        border: 2px solid #3d87ff;
        border-radius: 10px;
        background-color: #0d0d15;
    }

    /* Glow effects now use border-width and border-color */
    #output_display:focus, #status_display:focus, #conversation_display:focus {
        border: 3px solid #5d97ff;
        background-color: #0d1020;
    }

    /* Holographic-like effect for buttons */
    QPushButton#start_button, QPushButton#stop_button {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #1e3a6d, stop:1 #254680);
        border: 2px solid #3d6bbd;
        border-radius: 8px;
        color: #ffffff;
        padding: 8px 16px;
        font-weight: bold;
    }

    QPushButton#start_button:hover, QPushButton#stop_button:hover {
        background: qlineargradient(x1:0, y1:0, x2:1, y2:1, stop:0 #254680, stop:1 #2d5499);
        border: 2px solid #4d7bd7;
    }
    """


class ConversationManager:
    """Manage conversation history and user memory using pandas"""
    
    def __init__(self, storage_path="conversations"):
        """Initialize conversation manager with storage path"""
        self.storage_path = storage_path
        
        # Create storage directory if it doesn't exist
        if not os.path.exists(storage_path):
            os.makedirs(storage_path)
            
        # Current conversation
        self.current_conversation = {
            'id': str(uuid.uuid4()),
            'name': f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'messages': []
        }
        
        # Memory dataframe
        self.memory_df = pd.DataFrame(columns=['timestamp', 'source', 'content', 'conversation_id'])
        
        # Load existing memory if available
        self.memory_file = os.path.join(storage_path, "memory.csv")
        if os.path.exists(self.memory_file):
            try:
                self.memory_df = pd.read_csv(self.memory_file)
            except Exception as e:
                print(f"Error loading memory: {e}")
    
    def add_message(self, role, content):
        """Add a message to the current conversation"""
        timestamp = datetime.now()
        
        # Add to current conversation
        self.current_conversation['messages'].append({
            'timestamp': timestamp.isoformat(),
            'role': role,
            'content': content
        })
        
        # Add to memory dataframe
        new_row = {
            'timestamp': timestamp,
            'source': role,
            'content': content,
            'conversation_id': self.current_conversation['id']
        }
        
        self.memory_df = pd.concat([self.memory_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save memory
        self.save_memory()
        
        return timestamp
    
    def save_memory(self):
        """Save memory dataframe to disk"""
        try:
            self.memory_df.to_csv(self.memory_file, index=False)
        except Exception as e:
            print(f"Error saving memory: {e}")
    
    def save_conversation(self):
        """Save the current conversation to disk"""
        if not self.current_conversation['messages']:
            return False
            
        filename = os.path.join(
            self.storage_path, 
            f"{self.current_conversation['id']}.json"
        )
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.current_conversation, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving conversation: {e}")
            return False
    
    def load_conversation(self, conversation_id):
        """Load a conversation from disk"""
        filename = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        if not os.path.exists(filename):
            return False
            
        try:
            with open(filename, 'r') as f:
                self.current_conversation = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading conversation: {e}")
            return False
    
    def get_all_conversations(self):
        """Get list of all saved conversations"""
        conversations = []
        
        for filename in os.listdir(self.storage_path):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.storage_path, filename), 'r') as f:
                        conversation = json.load(f)
                        conversations.append({
                            'id': conversation['id'],
                            'name': conversation.get('name', 'Unnamed conversation'),
                            'timestamp': conversation['messages'][0]['timestamp'] if conversation['messages'] else 'Unknown',
                            'message_count': len(conversation['messages'])
                        })
                except Exception as e:
                    print(f"Error loading conversation {filename}: {e}")
                    
        return sorted(conversations, key=lambda x: x['timestamp'], reverse=True)
    
    def delete_conversation(self, conversation_id):
        """Delete a conversation from disk and memory"""
        filename = os.path.join(self.storage_path, f"{conversation_id}.json")
        
        if not os.path.exists(filename):
            return False
            
        try:
            # Remove from disk
            os.remove(filename)
            
            # Remove from memory dataframe
            self.memory_df = self.memory_df[self.memory_df['conversation_id'] != conversation_id]
            
            # Save updated memory
            self.save_memory()
            
            return True
        except Exception as e:
            print(f"Error deleting conversation: {e}")
            return False
    
    def new_conversation(self):
        """Start a new conversation"""
        # Save current conversation if it has messages
        if self.current_conversation['messages']:
            self.save_conversation()
            
        # Create new conversation
        self.current_conversation = {
            'id': str(uuid.uuid4()),
            'name': f"Conversation {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            'messages': []
        }
        
        return self.current_conversation
    
    def get_conversation_messages(self, format_for_ollama=False):
        """Get conversation messages, optionally formatted for Ollama"""
        if not format_for_ollama:
            return self.current_conversation['messages']
            
        # Format for Ollama API
        ollama_messages = []
        
        for msg in self.current_conversation['messages']:
            role = msg['role']
            # Map 'user' and 'assistant' roles to what Ollama expects
            if role == 'user':
                ollama_role = 'user'
            elif role == 'assistant':
                ollama_role = 'assistant'
            else:
                ollama_role = 'system'
                
            ollama_messages.append({
                'role': ollama_role,
                'content': msg['content']
            })
            
        return ollama_messages
    
    def search_memory(self, query):
        """Search memory for relevant content"""
        # Basic search implementation - can be enhanced with better matching
        if query.strip() == "":
            return []
            
        # Convert query to lowercase for case-insensitive search
        query_lower = query.lower()
        
        # Search in content column
        results = self.memory_df[self.memory_df['content'].str.lower().str.contains(query_lower, na=False)]
        
        # Return as list of dictionaries
        return results.to_dict('records')


class OllamaClient:
    """Client for interacting with Ollama LLM API"""
    
    def __init__(self):
        """Initialize Ollama client"""
        pass
    
    def get_available_models(self):
        """Get list of available models from Ollama"""
        try:
            result = ollama.list()
            print(f"Ollama response: {result}")  # Debug print
            
            # Handle the new API format where models are objects
            if hasattr(result, 'models'):
                return [model.model for model in result.models]
            
            # Handle dictionary format with 'models' key
            elif isinstance(result, dict) and 'models' in result:
                return [model['name'] for model in result['models']]
                
            # Handle list format (older API)
            elif isinstance(result, list):
                return [model.get('name') for model in result if 'name' in model]
                
            # Handle direct dictionary format
            elif isinstance(result, dict):
                return list(result.keys())
                
            return []
        except Exception as e:
            print(f"Error getting models: {str(e)}")
            return []
    
    async def chat(self, model, messages, stream=False):
        """Send a chat request to Ollama"""
        try:
            return ollama.chat(
                model=model,
                messages=messages,
                stream=stream
            )
        except Exception as e:
            print(f"Error in Ollama chat: {e}")
            if stream:
                # Return generator with error
                async def error_generator():
                    yield {"message": {"content": f"Error: {str(e)}"}}
                return error_generator()
            else:
                return {"message": {"content": f"Error: {str(e)}"}}


class SpeechWorker(QThread):
    """Worker thread for speech processing with continuous operation and interruptions"""
    
    update_signal = pyqtSignal(str, str)  # (type, content)
    finished_signal = pyqtSignal()
    stt_audio_signal = pyqtSignal(np.ndarray)  # Audio input samples
    tts_audio_signal = pyqtSignal(np.ndarray)  # Audio output samples
    
    def __init__(self, stt, tts, ollama_client, model_name):
        super().__init__()
        self.stt = stt
        self.tts = tts
        self.ollama_client = ollama_client
        self.model_name = model_name
        self.running = True
        self.messages = []
        self.listening_for_interrupt = False
        self.in_conversation = False
        self.thinking = False
        self.response_buffer = ""
        
    def run(self):
        """Main worker loop for speech interaction with enhanced TTS"""
        self.update_signal.emit("status", "Waiting for wake word to start conversation...")
        
        # Start the interrupt listener thread
        self.setup_interrupt_listener()
        
        while self.running:
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
                
                # If speech was interrupted, reset and continue
                if self.stt.speech_interrupted:
                    self.stt.speech_interrupted = False
                    self.update_signal.emit("status", "Speech interrupted. Listening again...")
                    continue
                    
                # Process recognized speech
                if speech_text and speech_text.strip() != "":
                    # Update UI with recognized text
                    self.update_signal.emit("user", speech_text)
                    self.update_signal.emit("status", "Processing your request...")
                    
                    # Start thinking mode
                    self.thinking = True
                    
                    try:
                        # Prepare messages for LLM
                        messages = [
                            {"role": "user", "content": speech_text}
                        ]
                        
                        # Generate LLM response with streaming
                        self.update_signal.emit("status", "Assistant is thinking...")
                        self.response_buffer = ""
                        
                        # Get streaming response
                        stream = self.ollama_client.chat(
                            self.model_name,
                            messages,
                            stream=True
                        )
                        
                        # Process streaming chunks
                        full_response = ""
                        for chunk in stream:
                            if not self.thinking or not self.running:
                                break
                                
                            if 'message' in chunk and 'content' in chunk['message']:
                                response_text = chunk['message']['content']
                                full_response += response_text
                                self.response_buffer += response_text
                                
                                # Emit streaming update
                                self.update_signal.emit("assistant_stream", response_text)
                                
                                # Create audio in the background for smoother interaction
                                if self.detect_sentence_boundary(self.response_buffer):
                                    # Speak the buffered text
                                    self.tts.speak_text(self.response_buffer)
                                    self.response_buffer = ""
                        
                        # Emit the complete response once done
                        if full_response and self.running:
                            self.update_signal.emit("assistant", full_response)
                        
                    except Exception as e:
                        self.update_signal.emit("error", f"Error generating response: {str(e)}")
                    finally:
                        # Clear thinking state
                        self.thinking = False
                else:
                    self.update_signal.emit("status", "Didn't catch that. Try again?")
                        
                # Check for exit conditions
                if not self.running:
                    break
            
            # Reset for the next conversation cycle
            if not self.in_conversation:
                self.update_signal.emit("status", "Waiting for wake word...")
            
            # Add this to emit input audio samples periodically
            if hasattr(self, "stt") and hasattr(self.stt, "samples_buffer") and len(self.stt.samples_buffer) > 0:
                input_samples = np.array(self.stt.samples_buffer)
                self.stt_audio_signal.emit(input_samples)
        
        self.finished_signal.emit()
        
    def setup_interrupt_listener(self):
        """Set up a separate thread to listen for interrupt keys and wake words"""
        def check_for_interrupt():
            import keyboard
            while self.running:
                # Check for ESC key interrupt
                if keyboard.is_pressed('esc') and (self.in_conversation or self.thinking):
                    print("Interrupt key pressed!")
                    self.interrupt_current_action()
                    
                # Wait a bit to prevent busy-waiting
                time.sleep(0.1)
                
        interrupt_thread = threading.Thread(target=check_for_interrupt, daemon=True)
        interrupt_thread.start()
        
    def interrupt_current_action(self):
        """Interrupt current action (speaking or generating)"""
        print("Interrupting current action")
        
        # Stop TTS
        self.tts.interrupt_speech()
        
        # Signal that speech was interrupted
        self.stt.speech_interrupted = True
        
        # Clear response buffer if thinking
        if self.thinking:
            self.response_buffer = ""
            self.thinking = False
            
        # Update UI
        self.update_signal.emit("status", "Interrupted! Listening...")
        
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.in_conversation = False
        self.thinking = False
        self.stt.stop()
        self.tts.interrupt_speech()

    # Add this helper method to SpeechWorker class

    def detect_sentence_boundary(self, text):
        """
        Detect if text contains a natural sentence boundary
        
        Args:
            text (str): Text to check
            
        Returns:
            bool: True if text contains a sentence boundary
        """
        # Check for explicit sentence terminators
        has_terminator = any(char in text for char in '.!?:')
        
        # Check for natural pauses that don't end sentences
        trailing_comma = text.rstrip().endswith(',')
        
        # Check for quote terminators
        quote_terminators = [
            '." ', '!" ', '?" ', '": ',  # Quotes with space after
            '."', '!"', '?"', '":',      # Quotes at end of text
            '" ', "' "                    # Standalone quotes with space
        ]
        has_quote_end = any(qt in text for qt in quote_terminators)
        
        # Check for list item markers
        list_markers = re.search(r'\d+\.\s+\w+', text)
        
        # Natural clause boundaries
        clause_markers = [
            ' and ', ' but ', ' or ', ' nor ', ' yet ', ' so ', ' for ', ' because '
        ]
        has_clause = any(cm in text for cm in clause_markers)
        
        # Add this after detecting a sentence boundary:
        if has_terminator or has_quote_end:
            if hasattr(self.tts, "speak_text") and self.tts:
                # This sentence is complete, speak it and show visualization
                self.tts.speak_text(self.response_buffer)
                self.visualize_tts_chunk(self.response_buffer)
                self.response_buffer = ""
                return True
        
        return False

    def visualize_tts_chunk(self, text):
        """Show visual indicator for TTS chunk processing"""
        # Create a pulse pattern for the output visualizer based on text length
        chunk_length = len(text)
        pulse_intensity = min(1.0, chunk_length / 100.0) * 0.8 + 0.2  # Scale between 0.2-1.0
        
        # Generate sample pattern
        pulse_pattern = np.sin(np.linspace(0, np.pi*2, 100)) * pulse_intensity
        
        # Use signal instead of direct access - this was causing issues
        self.tts_audio_signal.emit(pulse_pattern)
        
        # Send status update instead of direct UI manipulation
        status_message = f"Speaking: {text[:30]}..."
        self.update_signal.emit("status", status_message)

    # Fix in SpeechWorker.live_transcription_preview
    def live_transcription_preview(self):
        """Show live transcription preview as audio is processed"""
        last_result = ""
        print("Starting live transcription preview")
        while self.live_listening_active:
            try:
                if hasattr(self.stt, "current_partial_result") and self.stt.current_partial_result:
                    # Only emit when the result changes
                    current = self.stt.current_partial_result
                    if current and current != last_result:
                        self.update_signal.emit("partial_transcript", current)
                        last_result = current
                        print(f"Live transcript: {current}")
            except Exception as e:
                print(f"Error in live transcription: {e}")
            time.sleep(0.2)  # Reduced polling frequency
        print("Live transcription preview stopped")

    # Add to SpeechWorker class
    def start_push_to_talk_listening(self):
        """Start direct listening bypassing wake word"""
        def listen_and_process():
            # Show active listening status
            self.update_signal.emit("status", "Listening (Push to Talk)...")
            
            # Start listening with live transcription
            self.live_listening_active = True
            
            # Start background thread for live results
            live_thread = threading.Thread(target=self.live_transcription_preview)
            live_thread.daemon = True
            live_thread.start()
            
            # Listen until push-to-talk button released or max duration
            speech_text = self.stt.start_listening_session(
                non_blocking=False,
                max_listen_time=10,  # Longer timeout for push-to-talk
                wait_for_release=True,  # Will stop when push_to_talk_active becomes False
            )
            
            # Stop live preview
            self.live_listening_active = False
            
            # Process speech if we got something
            if speech_text and speech_text.strip():
                self.update_signal.emit("user", speech_text)
                self.process_speech_input(speech_text)
            else:
                self.update_signal.emit("status", "No speech detected")
        
        # Run in background thread
        threading.Thread(target=listen_and_process, daemon=True).start()

    def process_speech_input(self, speech_text):
        """Process recognized speech input"""
        self.update_signal.emit("status", "Processing your request...")
        
        # Start thinking mode
        self.thinking = True
        
        try:
            # Prepare messages for LLM
            messages = self.ollama_client.get_conversation_messages()
            messages.append({"role": "user", "content": speech_text})
            
            # Generate LLM response with streaming
            self.update_signal.emit("status", "Assistant is thinking...")
            self.response_buffer = ""
            
            # Get streaming response
            stream = self.ollama_client.chat(
                self.model_name,
                messages,
                stream=True
            )
            
            # Process streaming chunks with visualization
            self.process_llm_response(stream)
            
        except Exception as e:
            self.update_signal.emit("error", f"Error generating response: {str(e)}")
        finally:
            self.thinking = False


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window title and size
        self.setWindowTitle("Project 42 - Speech Assistant")
        self.resize(1200, 800)
        
        # Set debug mode BEFORE initializing UI
        self.debug_mode = True  # Set to False in production
        
        # Initialize components
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.ollama_client = OllamaClient()
        self.conversation_manager = ConversationManager(storage_path="conversations")
        
        # Default audio settings
        self.input_device_index = None
        self.input_gain = 1.0
        
        # Initialize UI
        self.init_ui()
        
        # Apply dark theme
        self.setStyleSheet(StyleSheet.DARK_THEME)
        
        # Load audio settings BEFORE refreshing models
        self.load_audio_settings()
        
        # Show available models
        self.refresh_models()
        
        # Load conversations
        self.load_conversation_list()
        
        
        # Main layout
        main_layout = QVBoxLayout(central_widget)
        
        # Top control panel
        control_panel = QWidget()
        control_layout = QHBoxLayout(control_panel)
        control_layout.setContentsMargins(10, 10, 10, 10)
        
        # Model selection
        model_label = QLabel("LLM Model:")
        self.model_combo = QComboBox()
        self.model_combo.setMinimumWidth(200)
        
        # Control buttons
        self.refresh_button = QPushButton("Refresh Models")
        self.refresh_button.clicked.connect(self.refresh_models)

        self.whisper_mode_button = QPushButton("Whisper Mode: OFF")
        self.whisper_mode_button.setCheckable(True)
        self.whisper_mode_button.clicked.connect(self.toggle_whisper_mode)

        self.start_button = QPushButton("Start Assistant")
        self.start_button.clicked.connect(self.start_assistant)
        
        self.stop_button = QPushButton("Stop Assistant")
        self.stop_button.clicked.connect(self.stop_assistant)
        self.stop_button.setEnabled(False)
        
        # Add widgets to control panel
        control_layout.addWidget(model_label)
        control_layout.addWidget(self.model_combo)
        control_layout.addWidget(self.refresh_button)
        control_layout.addWidget(self.whisper_mode_button)
        control_layout.addWidget(self.start_button)
        control_layout.addWidget(self.stop_button)
        control_layout.addStretch(1)
        
        # Add control panel to main layout
        main_layout.addWidget(control_panel)
        
        # Main content splitter
        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left panel (conversation history and memory)
        left_panel = QTabWidget()
        
        # Conversation tab
        self.conversation_tab = QWidget()
        conversation_layout = QVBoxLayout(self.conversation_tab)
        
        # Conversation controls
        conv_control_layout = QHBoxLayout()
        self.new_conv_button = QPushButton("New Conversation")
        self.new_conv_button.clicked.connect(self.new_conversation)
        self.load_conv_button = QPushButton("Load Conversation")
        self.load_conv_button.clicked.connect(self.show_load_conversation_dialog)
        conv_control_layout.addWidget(self.new_conv_button)
        conv_control_layout.addWidget(self.load_conv_button)
        conversation_layout.addLayout(conv_control_layout)
        
        # Conversation display
        self.conversation_display = QTextEdit()
        self.conversation_display.setReadOnly(True)
        conversation_layout.addWidget(self.conversation_display)
        
        # Memory tab
        self.memory_tab = QWidget()
        memory_layout = QVBoxLayout(self.memory_tab)
        
        # Memory search
        search_layout = QHBoxLayout()
        search_label = QLabel("Search Memory:")
        self.memory_search = QTextEdit()
        self.memory_search.setMaximumHeight(28)
        self.search_button = QPushButton("Search")
        self.search_button.clicked.connect(self.search_memory)
        search_layout.addWidget(search_label)
        search_layout.addWidget(self.memory_search)
        search_layout.addWidget(self.search_button)
        memory_layout.addLayout(search_layout)
        
        # Memory table
        self.memory_table = QTableWidget()
        self.memory_table.setColumnCount(3)
        self.memory_table.setHorizontalHeaderLabels(["Timestamp", "Source", "Content"])
        self.memory_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        memory_layout.addWidget(self.memory_table)
        
        # Add tabs to left panel
        left_panel.addTab(self.conversation_tab, "Conversation")
        left_panel.addTab(self.memory_tab, "Memory")
        
        # Right panel (live output)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Status display
        status_layout = QHBoxLayout()
        status_label = QLabel("Status:")
        self.status_display = QTextEdit()
        self.status_display.setReadOnly(True)
        self.status_display.setMaximumHeight(28)
        status_layout.addWidget(status_label)
        status_layout.addWidget(self.status_display)
        right_layout.addLayout(status_layout)
        
        # Live output
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        right_layout.addWidget(self.output_display)
        
        # Add panels to splitter
        self.content_splitter.addWidget(left_panel)
        self.content_splitter.addWidget(right_panel)
        self.content_splitter.setSizes([400, 800])
        
        # Add splitter to main layout
        main_layout.addWidget(self.content_splitter)
        
        # Initialize worker
        self.speech_worker = None
        
        # After creating the status_display
        self.status_display.setObjectName("status_display")
        
        # After creating the output_display
        self.output_display.setObjectName("output_display")
        
        # After creating the conversation_display
        self.conversation_display.setObjectName("conversation_display")
        
        # Set IDs for styled buttons
        self.start_button.setObjectName("start_button")
        self.stop_button.setObjectName("stop_button")
        
        # After adding control panel to main layout
        # Add audio visualizers
        visualizer_panel = QWidget()
        self.audio_layout = QHBoxLayout(visualizer_panel)  # Define audio_layout as a class attribute
        self.audio_layout.setContentsMargins(10, 5, 10, 5)
        
        # Input visualizer
        input_visualizer_widget = QWidget()
        input_visualizer_layout = QVBoxLayout(input_visualizer_widget)
        input_label = QLabel("Microphone Input")
        input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.input_visualizer = AudioWaveformVisualizer(mode="input")
        input_visualizer_layout.addWidget(input_label)
        input_visualizer_layout.addWidget(self.input_visualizer)
        
        # Output visualizer
        output_visualizer_widget = QWidget()
        output_visualizer_layout = QVBoxLayout(output_visualizer_widget)
        output_label = QLabel("Speech Output")
        output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.output_visualizer = AudioWaveformVisualizer(mode="output")
        output_visualizer_layout.addWidget(output_label)
        output_visualizer_layout.addWidget(self.output_visualizer)
        
        # Add to visualizer panel
        self.audio_layout.addWidget(input_visualizer_widget)
        self.audio_layout.addWidget(output_visualizer_widget)
        
        # Add visualizers to main layout
        main_layout.addWidget(visualizer_panel)
        
        # Add visualizer update timer
        self.visualizer_timer = QTimer(self)
        self.visualizer_timer.timeout.connect(self.update_visualizers)
        self.visualizer_timer.start(50)  # Update every 50ms
        
        # Add a status bar for important notifications
        self.statusBar().showMessage("Ready - Say the wake word to begin")
        
        # Add this line near the end of init_ui
        self.stt.wake_word = "Alexa"  # Make sure wake word is explicitly set
        
        # Add the audio visualizers to the UI

        def init_ui(self):
            # After adding control panel to main layout
            # Add audio visualizers
            visualizer_panel = QWidget()
            self.audio_layout = QVBoxLayout(visualizer_panel)
            self.audio_layout.setContentsMargins(10, 5, 10, 5)
            
            # Create input/output visualizer headers
            header_layout = QHBoxLayout()
            
            # Input side
            input_label = QLabel("Voice Input")
            input_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(input_label)
            
            # Output side
            output_label = QLabel("Speech Output")
            output_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            header_layout.addWidget(output_label)
            
            self.audio_layout.addLayout(header_layout)
            
            # Create visualizer container
            visualizer_container = QHBoxLayout()
            
            # Create input visualizer
            self.input_visualizer = AudioWaveformVisualizer(mode="input")
            self.input_visualizer.setMinimumHeight(80)
            visualizer_container.addWidget(self.input_visualizer)
            
            # Create output visualizer
            self.output_visualizer = AudioWaveformVisualizer(mode="output")
            self.output_visualizer.setMinimumHeight(80)
            visualizer_container.addWidget(self.output_visualizer)
            
            self.audio_layout.addLayout(visualizer_container)
            
            # Add visualizer panel to main layout
            main_layout.addWidget(visualizer_panel)
            
            # Start update timer for visualizers
            self.visualizer_timer = QTimer(self)
            self.visualizer_timer.timeout.connect(self.update_visualizers)
            self.visualizer_timer.start(50)  # Update every 50ms

        # After creating control panel, add this:
        
        # Wake word controls panel
        wake_word_panel = QWidget()
        wake_word_layout = QHBoxLayout(wake_word_panel)
        
        # Wake word input
        wake_label = QLabel("Wake Word:")
        self.wake_word_input = QLineEdit(self.stt.wake_word)
        self.wake_word_input.setMaximumWidth(120)
        self.wake_word_input.returnPressed.connect(self.update_wake_word)
        
        # Apply button
        self.apply_wake_button = QPushButton("Apply")
        self.apply_wake_button.clicked.connect(self.update_wake_word)
        
        # Wake word enable toggle
        self.wake_word_toggle = QPushButton("Wake Word: ON")
        self.wake_word_toggle.setCheckable(True)
        self.wake_word_toggle.setChecked(True)
        self.wake_word_toggle.clicked.connect(self.toggle_wake_word)
        
        # Push-to-talk button
        self.push_to_talk = QPushButton("Push to Talk")
        self.push_to_talk.setToolTip("Hold to speak without wake word")
        self.push_to_talk.pressed.connect(self.start_push_to_talk)
        self.push_to_talk.released.connect(self.stop_push_to_talk)
        
        # Add to layout
        wake_word_layout.addWidget(wake_label)
        wake_word_layout.addWidget(self.wake_word_input)
        wake_word_layout.addWidget(self.apply_wake_button)
        wake_word_layout.addWidget(self.wake_word_toggle)
        wake_word_layout.addWidget(self.push_to_talk)
        
        # Add to main layout (after control panel)
        main_layout.addWidget(wake_word_panel)
        
        # Add to MainWindow.init_ui after audio visualizer setup
        # Add audio level meter
        level_meter_panel = QWidget()
        level_meter_layout = QHBoxLayout(level_meter_panel)

        # Current threshold indicator
        self.threshold_label = QLabel("Threshold: 550")
        level_meter_layout.addWidget(self.threshold_label)

        # Audio level meter
        self.level_meter = QProgressBar()
        self.level_meter.setMinimum(0)
        self.level_meter.setMaximum(1000)
        self.level_meter.setFormat("Input Level: %v")
        self.level_meter.setTextVisible(True)
        self.level_meter.setStyleSheet("""
            QProgressBar {
                border: 1px solid #444;
                border-radius: 5px;
                text-align: center;
                color: white;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #375e9e, stop:0.5 #376e9e, stop:1 #5e9c37);
                border-radius: 5px;
            }
        """)
        level_meter_layout.addWidget(self.level_meter, 1)  # 1 = stretch factor

        # Threshold adjustment
        thresh_down_btn = QPushButton("-")
        thresh_down_btn.setFixedSize(30, 30)
        thresh_down_btn.clicked.connect(self.decrease_threshold)
        level_meter_layout.addWidget(thresh_down_btn)

        thresh_up_btn = QPushButton("+")
        thresh_up_btn.setFixedSize(30, 30)
        thresh_up_btn.clicked.connect(self.increase_threshold)
        level_meter_layout.addWidget(thresh_up_btn)

        # Add to main layout
        main_layout.addWidget(level_meter_panel)
        
        # Add this to MainWindow.init_ui after tabs setup

        # Add Debug tab
        self.debug_tab = QWidget()
        debug_layout = QVBoxLayout(self.debug_tab)

        # Debug output
        self.debug_output = QTextEdit()
        self.debug_output.setReadOnly(True)
        debug_layout.addWidget(QLabel("Debug Log:"))
        debug_layout.addWidget(self.debug_output)

        # Debug controls
        debug_control_layout = QHBoxLayout()
        self.debug_checkbox = QCheckBox("Enable Debug Mode")
        self.debug_checkbox.setChecked(self.debug_mode)
        self.debug_checkbox.toggled.connect(self.toggle_debug_mode)

        self.clear_log_button = QPushButton("Clear Log")
        self.clear_log_button.clicked.connect(self.debug_output.clear)

        debug_control_layout.addWidget(self.debug_checkbox)
        debug_control_layout.addWidget(self.clear_log_button)
        debug_layout.addLayout(debug_control_layout)

        # Add debug tab to left panel
        left_panel.addTab(self.debug_tab, "Debug")

    def refresh_models(self):
        """Refresh available Ollama models"""
        self.model_combo.clear()
        
        # Instead of using speech_worker (which might be None), 
        # use the ollama_client directly
        try:
            models = self.ollama_client.get_available_models()
            if models:
                for model in models:
                    self.model_combo.addItem(model)
                self.status_display.setText(f"Found {len(models)} models")
            else:
                self.status_display.setText("No models found. Is Ollama running?")
        except Exception as e:
            self.status_display.setText(f"Error loading models: {e}")
            print(f"Error loading models: {e}")
        
    def stop_assistant(self):
        """Stop the speech assistant"""
        if self.speech_worker is not None:
            self.status_display.setText("Stopping assistant...")
            self.speech_worker.stop()
        
    def worker_finished(self):
        """Handle worker thread finishing"""
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.refresh_button.setEnabled(True)
        self.model_combo.setEnabled(True)
        
        self.status_display.setText("Assistant stopped")
        
    def update_from_worker(self, update_type, content):
        """Update UI based on worker signals"""
        
        # Add this case for partial transcription
        if update_type == "partial_transcript":
            # Show partial transcript in status bar with different style
            self.status_display.setText(f"Recognizing: {content}")
            self.statusBar().showMessage(f"Hearing: {content}", 1000)
        
        if update_type == "status":
            self.status_display.setText(content)
            self.statusBar().showMessage(content, 3000)  # Show in status bar for 3 seconds
            
        elif update_type == "user":
            # Format and display user speech
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"<p><span style='color: #66CDAA;'>[{timestamp}] <b>You:</b></span> {content}</p>"
            self.output_display.append(formatted_text)
            self.conversation_display.append(formatted_text)
            
            # Add to conversation history
            self.conversation_manager.add_message("user", content)
            
        elif update_type == "assistant":
            # Format and display complete assistant response
            timestamp = datetime.now().strftime("%H:%M:%S")
            formatted_text = f"<p><span style='color: #9370DB;'>[{timestamp}] <b>Assistant:</b></span> {content}</p>"
            
            # Replace any existing streaming response with the complete one
            self.replace_streaming_response(formatted_text)
            self.conversation_display.append(formatted_text)
            
            # Add to conversation history
            self.conversation_manager.add_message("assistant", content)
            
        elif update_type == "assistant_stream":
            # For streaming updates, continuously update the current assistant response
            self.update_streaming_response(content)
            
        elif update_type == "error":
            # Show errors in red
            self.output_display.append(f"<p style='color: #ff6666;'><b>Error:</b> {content}</p>")
            self.status_display.setText(f"Error: {content}")

    def update_streaming_response(self, new_content):
        """Update the streaming response in real-time"""
        # Get current content
        html = self.output_display.toHtml()
        
        # Check if we already have a streaming message
        if "<span class='streaming'>" in html:
            # Update existing streaming message
            html = html.split("<span class='streaming'>")[0] + "<span class='streaming'>" + new_content + "</span>"
            self.output_display.setHtml(html)
        else:
            # Add new streaming message
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.output_display.append(f"<p><span style='color: #9370DB;'>[{timestamp}] <b>Assistant:</b></span> <span class='streaming'>{new_content}</span></p>")

    def replace_streaming_response(self, complete_response):
        """Replace streaming response with complete response"""
        html = self.output_display.toHtml()
        if "<span class='streaming'>" in html:
            parts = html.split("<span class='streaming'>")
            before = parts[0]
            after = parts[1].split("</span>", 1)[1] if "</span>" in parts[1] else ""
            self.output_display.setHtml(before + complete_response + after)
        else:
            # Fallback if no streaming response found
            self.output_display.append(complete_response)
            
    def new_conversation(self):
        """Start a new conversation"""
        # Clear displays
        self.conversation_display.clear()
        self.output_display.clear()
        
        # Create new conversation in manager
        self.conversation_manager.new_conversation()
        
        self.status_display.setText("New conversation started")
        
    def show_load_conversation_dialog(self):
        """Show dialog to load a conversation"""
        dialog = QDialog(self)
        dialog.setWindowTitle("Load Conversation")
        dialog.setMinimumSize(500, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Conversation list
        self.conv_list = QListWidget()
        self.conv_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.populate_conversation_list(self.conv_list)
        
        # Buttons
        button_layout = QHBoxLayout()
        load_button = QPushButton("Load")
        load_button.clicked.connect(lambda: self.load_selected_conversation(dialog))
        delete_button = QPushButton("Delete")
        delete_button.clicked.connect(lambda: self.delete_selected_conversation())
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(dialog.reject)
        
        button_layout.addWidget(load_button)
        button_layout.addWidget(delete_button)
        button_layout.addWidget(cancel_button)
        
        layout.addWidget(QLabel("Select a conversation to load:"))
        layout.addWidget(self.conv_list)
        layout.addLayout(button_layout)
        
        dialog.exec()
        
    def populate_conversation_list(self, list_widget):
        """Populate list with available conversations"""
        list_widget.clear()
        
        conversations = self.conversation_manager.get_all_conversations()
        
        for conv in conversations:
            item = QListWidgetItem(f"{conv['name']} - {conv['message_count']} messages")
            item.setData(Qt.ItemDataRole.UserRole, conv['id'])
            list_widget.addItem(item)
            
    def load_selected_conversation(self, dialog):
        """Load the selected conversation"""
        selected_items = self.conv_list.selectedItems()
        
        if not selected_items:
            return
            
        conv_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        if self.conversation_manager.load_conversation(conv_id):
            # Update display
            self.conversation_display.clear()
            
            for msg in self.conversation_manager.get_conversation_messages():
                role = msg['role']
                content = msg['content']
                
                if role == 'user':
                    self.conversation_display.append(f"<b>You:</b> {content}")
                elif role == 'assistant':
                    self.conversation_display.append(f"<b>Assistant:</b> {content}")
                else:
                    self.conversation_display.append(f"<b>{role}:</b> {content}")
            
            self.status_display.setText("Conversation loaded")
            dialog.accept()
        else:
            QMessageBox.warning(self, "Error", "Failed to load conversation")
            
    def delete_selected_conversation(self):
        """Delete the selected conversation"""
        selected_items = self.conv_list.selectedItems()
        
        if not selected_items:
            return
            
        conv_id = selected_items[0].data(Qt.ItemDataRole.UserRole)
        
        reply = QMessageBox.question(
            self, 
            "Confirm Delete", 
            "Are you sure you want to delete this conversation?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            if self.conversation_manager.delete_conversation(conv_id):
                self.populate_conversation_list(self.conv_list)
                self.status_display.setText("Conversation deleted")
            else:
                QMessageBox.warning(self, "Error", "Failed to delete conversation")
                
    def load_conversation_list(self):
        """Load the list of conversations for the memory view"""
        # Clear the table
        self.memory_table.setRowCount(0)
        
        # Load memory from manager
        memory_data = self.conversation_manager.memory_df
        
        if memory_data.empty:
            return
            
        # Sort by timestamp descending
        memory_data = memory_data.sort_values(by='timestamp', ascending=False)
        
        # Populate table
        self.memory_table.setRowCount(len(memory_data))
        
        for i, (_, row) in enumerate(memory_data.iterrows()):
            # Timestamp
            timestamp_item = QTableWidgetItem(str(row['timestamp']))
            self.memory_table.setItem(i, 0, timestamp_item)
            
            # Source
            source_item = QTableWidgetItem(row['source'])
            self.memory_table.setItem(i, 1, source_item)
            
            # Content (shorten if too long)
            content = row['content']
            if len(content) > 100:
                content = content[:100] + "..."
            content_item = QTableWidgetItem(content)
            content_item.setData(Qt.ItemDataRole.UserRole, row['content'])  # Store full content
            self.memory_table.setItem(i, 2, content_item)
            
        self.memory_table.resizeColumnsToContents()
        self.memory_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
        # Connect double-click to show full content
        self.memory_table.cellDoubleClicked.connect(self.show_full_memory_content)
        
    def show_full_memory_content(self, row, column):
        """Show full content of memory entry in a dialog"""
        if column != 2:  # Only for content column
            return
            
        item = self.memory_table.item(row, column)
        if not item:
            return
            
        full_content = item.data(Qt.ItemDataRole.UserRole)
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Memory Content")
        dialog.setMinimumSize(600, 400)
        
        layout = QVBoxLayout(dialog)
        
        # Source and timestamp
        source_item = self.memory_table.item(row, 1)
        timestamp_item = self.memory_table.item(row, 0)
        
        header_label = QLabel(f"<b>{source_item.text()}</b> - {timestamp_item.text()}")
        layout.addWidget(header_label)
        
        # Content
        content_text = QTextEdit()
        content_text.setReadOnly(True)
        content_text.setText(full_content)
        layout.addWidget(content_text)
        
        # Close button
        close_button = QPushButton("Close")
        close_button.clicked.connect(dialog.accept)
        layout.addWidget(close_button)
        
        dialog.exec()
        
    def search_memory(self):
        """Search memory for query"""
        query = self.memory_search.toPlainText().strip()
        
        if not query:
            self.load_conversation_list()  # Reset to full list
            return
            
        results = self.conversation_manager.search_memory(query)
        
        # Clear the table
        self.memory_table.setRowCount(0)
        
        if not results:
            self.memory_table.setRowCount(1)
            no_results = QTableWidgetItem("No results found")
            self.memory_table.setItem(0, 0, no_results)
            return
            
        # Populate table with results
        self.memory_table.setRowCount(len(results))
        
        for i, row in enumerate(results):
            # Timestamp
            timestamp_item = QTableWidgetItem(str(row['timestamp']))
            self.memory_table.setItem(i, 0, timestamp_item)
            
            # Source
            source_item = QTableWidgetItem(row['source'])
            self.memory_table.setItem(i, 1, source_item)
            
            # Content (shorten if too long)
            content = row['content']
            if len(content) > 100:
                content = content[:100] + "..."
            content_item = QTableWidgetItem(content)
            content_item.setData(Qt.ItemDataRole.UserRole, row['content'])  # Store full content
            self.memory_table.setItem(i, 2, content_item)
            
        self.memory_table.resizeColumnsToContents()
        self.memory_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        
    def keyPressEvent(self, event):
        """Handle key press events with enhanced interruption support"""
        from PyQt6.QtCore import Qt
        
        # Check for ESC key to interrupt current action
        if event.key() == Qt.Key.Key_Escape:
            if self.speech_worker and hasattr(self.speech_worker, 'interrupt_current_action'):
                self.speech_worker.interrupt_current_action()
                self.status_display.setText("Interrupted current action")
        
        # Check for F2 key to change wake word
        elif event.key() == Qt.Key.Key_F2:
            self.show_wake_word_dialog()
            
        # Check for F3 key to toggle continuous listening mode
        elif event.key() == Qt.Key.Key_F3:
            if self.speech_worker and hasattr(self.speech_worker, 'in_conversation'):
                self.speech_worker.in_conversation = not self.speech_worker.in_conversation
                status = "Continuous conversation mode: " + ("ON" if self.speech_worker.in_conversation else "OFF")
                self.status_display.setText(status)
        
        # Check for F4 key to toggle ultra-responsive mode
        elif event.key() == Qt.Key.Key_F4:
            if hasattr(self, 'ultra_responsive_mode'):
                self.ultra_responsive_mode = not self.ultra_responsive_mode
                status = "Ultra-responsive mode: " + ("ON" if self.ultra_responsive_mode else "OFF")
                self.status_display.setText(status)
            else:
                self.ultra_responsive_mode = True
                self.status_display.setText("Ultra-responsive mode: ON")
            
            # Apply settings to the worker
            if self.speech_worker:
                self.speech_worker.ultra_responsive_mode = self.ultra_responsive_mode
        
        # Allow default event processing
        super().keyPressEvent(event)

    def show_wake_word_dialog(self):
        """Show dialog to change wake word with better feedback"""
        current_wake_word = self.stt.wake_word
        new_wake_word, ok = QInputDialog.getText(
            self, 
            "Change Wake Word",
            "Enter new wake word (used to activate assistant and interrupt speech):",
            text=current_wake_word
        )
        
        if ok and new_wake_word:
            self.stt.set_wake_word(new_wake_word)
            self.status_display.setText(f"Wake word changed to: '{new_wake_word}'")
            
            # Show a message box with instructions
            QMessageBox.information(
                self,
                "Wake Word Changed",
                f"The wake word has been changed to '{new_wake_word}'.\n\n"
                f"How to use the wake word:\n"
                f"• Say '{new_wake_word}' to start a conversation\n"
                f"• Say '{new_wake_word}' during AI speech to interrupt\n"
                f"• Press ESC to manually interrupt at any time\n"
                f"• Say 'exit conversation' to end conversation mode"
            )

    def process_tts_responses(self, response, voice_name):
        # Process text in parallel with audio generation
        sentences = self.split_into_sentences(response)
        
        # Generate audio for each sentence in parallel
        audio_futures = []
        for sentence in sentences:
            # Process next sentence while current one is playing
            audio_futures.append(self.executor.submit(self.generate_sentence_audio, sentence))

    def update_visualizers(self):
        """Update audio visualizers with latest samples"""
        # If speech worker is active and we have a speech-to-text instance
        if hasattr(self, "stt") and self.stt:
            # Get audio samples from STT
            if hasattr(self.stt, "samples_buffer"):  # Fixed attribute check
                input_samples = self.stt.get_audio_samples()
                if len(input_samples) > 0:
                    self.input_visualizer.update_samples(input_samples)
                    
                    # Update level meter
                    avg_level = np.mean(input_samples)
                    self.update_level_meter(avg_level)
                    
                    # Show recognized speech status in real time
                    if hasattr(self.stt, "in_speech") and self.stt.in_speech:
                        if avg_level > 0.1:  # Only show for significant sound
                            self.status_display.setText("Listening: Speech detected...")
                            
                            # Add rms value to debug if enabled
                            if self.debug_mode and hasattr(self.stt, "last_rms"):
                                self.add_debug_log(f"RMS: {self.stt.last_rms}, Threshold: {self.stt.current_threshold}")
    
        # If we have text-to-speech instance
        if hasattr(self, "tts") and self.tts:
            # Get audio samples from TTS
            if hasattr(self.tts, "get_audio_samples"):
                output_samples = self.tts.get_audio_samples()
                if output_samples is not None and len(output_samples) > 0:
                    self.output_visualizer.update_samples(output_samples)
                    
                    # Show TTS active status
                    if self.tts.is_playing:
                        self.status_display.setText("Speaking...")

    def toggle_whisper_mode(self):
        """Toggle between whisper and normal detection modes"""
        if self.whisper_mode_button.isChecked():
            self.stt.toggle_whisper_mode(True)
            self.whisper_mode_button.setText("Whisper Mode: ON")
            self.status_display.setText("Whisper mode enabled - can detect quiet speech")
        else:
            self.stt.toggle_whisper_mode(False)
            self.whisper_mode_button.setText("Whisper Mode: OFF")
            self.status_display.setText("Normal speech mode enabled")

    def start_assistant(self):
        """Start the speech assistant"""
        # Get selected model
        model_name = self.model_combo.currentText()
        
        if not model_name:
            self.status_display.setText("Please select a model")
            return
        
        # Disable UI elements
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.refresh_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Create and start worker
        self.speech_worker = SpeechWorker(self.stt, self.tts, self.ollama_client, model_name)
        
        # Connect signals
        self.speech_worker.update_signal.connect(self.update_from_worker)
        self.speech_worker.finished_signal.connect(self.worker_finished)
        self.speech_worker.stt_audio_signal.connect(self.input_visualizer.update_samples)
        self.speech_worker.tts_audio_signal.connect(self.output_visualizer.update_samples)
        
        # Check if continuous mode is enabled
        continuous_mode = self.continuous_mode_button.isChecked() if hasattr(self, "continuous_mode_button") else False
        
        # Start the worker
        self.speech_worker.start()
        
        # Enable continuous mode if needed
        if continuous_mode:
            # Let the thread start first
            QTimer.singleShot(500, self.speech_worker.setup_continuous_mode)
            self.status_display.setText("Starting continuous conversation mode...")
        
        # Debug info
        if self.debug_mode:
            print("\n=== DEBUG INFO ===")
            print(f"Wake word: '{self.stt.wake_word}'")
            print(f"Using model: {model_name}")
            print(f"Whisper mode: {'ON' if hasattr(self.stt, 'whisper_mode') and self.stt.whisper_mode else 'OFF'}")
            print(f"Current threshold: {self.stt.current_threshold}")
            print(f"Continuous mode: {'ON' if continuous_mode else 'OFF'}")
            print("=================\n")

    # Enable query interface (add this in the MainWindow class)
    def show_memory_query_dialog(self):
        """Show dialog for querying conversation memory using natural language"""
        query, ok = QInputDialog.getText(
            self,
            "Memory Query",
            "Enter your question about conversation history:",
            text="Show me conversations about weather"
        )
        
        if ok and query:
            self.status_display.setText(f"Querying memory: {query}")
            
            # Run query
            result = self.conversation_manager.search_memory(query)
            
            if self.conversation_manager.query_engine:
                # Show the result in a dialog if using the query engine
                dialog = QDialog(self)
                dialog.setWindowTitle("Memory Query Result")
                dialog.setMinimumSize(600, 400)
                
                layout = QVBoxLayout(dialog)
                
                # Query text
                query_label = QLabel(f"<b>Query:</b> {query}")
                layout.addWidget(query_label)
                
                # Result
                result_text = QTextEdit()
                result_text.setReadOnly(True)
                
                if isinstance(result, dict) and 'response' in result:
                    result_text.setText(result['response'])
                    
                    # Show pandas code if available
                    if 'code' in result:
                        result_text.append(f"\n\n<b>Pandas code:</b>\n{result['code']}")
                else:
                    # Format the results
                    result_text.setText(f"Found {len(result)} matching messages:\n\n")
                    
                    for item in result[:20]:  # Limit to 20
                        timestamp = item.get('timestamp', 'Unknown')
                        source = item.get('source', 'Unknown')
                        content = item.get('content', '')
                        
                        result_text.append(f"[{timestamp}] <b>{source}:</b> {content}\n")
                        
                    if len(result) > 20:
                        result_text.append(f"\n... and {len(result) - 20} more results")
                        
                layout.addWidget(result_text)
                
                # Close button
                close_button = QPushButton("Close")
                close_button.clicked.connect(dialog.accept)
                layout.addWidget(close_button)
                
                dialog.exec()
            else:
                # Use standard table display without query engine
                self.memory_tab.setCurrentIndex(1)  # Switch to memory tab
                self.load_memory_search_results(result)
                
    # Add this method to the memory tab (add a button)
    def add_memory_query_button(self):
        """Add a natural language query button to memory tab"""
        # In your init_ui method, add this to the memory_layout:
        nl_query_button = QPushButton("Natural Language Query")
        nl_query_button.setToolTip("Query memory using natural language")
        nl_query_button.clicked.connect(self.show_memory_query_dialog)
        memory_layout.addWidget(nl_query_button)

    def update_wake_word(self):
        """Update wake word from input field"""
        new_word = self.wake_word_input.text().strip()
        if new_word:
            self.stt.set_wake_word(new_word)
            self.statusBar().showMessage(f"Wake word set to: '{new_word}'", 3000)
            self.status_display.setText(f"Wake word set to: '{new_word}'")

    def toggle_wake_word(self):
        """Enable or disable wake word detection"""
        if self.wake_word_toggle.isChecked():
            self.stt.wake_word_enabled = True
            self.wake_word_toggle.setText("Wake Word: ON")
            self.statusBar().showMessage("Wake word detection enabled", 3000)
        else:
            self.stt.wake_word_enabled = False
            self.wake_word_toggle.setText("Wake Word: OFF") 
            self.statusBar().showMessage("Wake word detection disabled - Push to Talk only", 3000)

    def start_push_to_talk(self):
        """Start push-to-talk listening"""
        if self.speech_worker:
            self.speech_worker.push_to_talk_active = True
            self.speech_worker.start_push_to_talk_listening()
            self.status_display.setText("Listening (Push to Talk)...")

    def stop_push_to_talk(self):
        """Stop push-to-talk listening"""
        if self.speech_worker:
            self.speech_worker.push_to_talk_active = False
            self.status_display.setText("Processing...")

    def update_level_meter(self, level):
        """Update audio level meter with current input level"""
        meter_value = int(level * 1000)
        self.level_meter.setValue(meter_value)
        
        # Change color based on threshold
        if self.stt and hasattr(self.stt, "current_threshold"):
            threshold = int(self.stt.current_threshold)
            if meter_value > threshold:
                self.level_meter.setStyleSheet("""
                    QProgressBar::chunk { background-color: #5e9c37; }
                """)
            else:
                self.level_meter.setStyleSheet("""
                    QProgressBar::chunk { background-color: #376e9e; }
                """)

    def increase_threshold(self):
        """Increase speech detection threshold"""
        if self.stt:
            self.stt.current_threshold += 50
            self.threshold_label.setText(f"Threshold: {int(self.stt.current_threshold)}")
            self.statusBar().showMessage(f"Speech threshold increased to {int(self.stt.current_threshold)}", 2000)

    def decrease_threshold(self):
        """Decrease speech detection threshold"""
        if self.stt:
            self.stt.current_threshold = max(100, self.stt.current_threshold - 50)
            self.threshold_label.setText(f"Threshold: {int(self.stt.current_threshold)}")
            self.statusBar().showMessage(f"Speech threshold decreased to {int(self.stt.current_threshold)}", 2000)

    # Add this method to MainWindow
    def add_debug_log(self, message):
        """Add message to debug log with timestamp"""
        if self.debug_mode:
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            self.debug_output.append(f"[{timestamp}] {message}")

    def toggle_debug_mode(self, enabled):
        """Toggle debug mode"""
        self.debug_mode = enabled
        if enabled:
            self.add_debug_log("Debug mode enabled")
        else:
            self.add_debug_log("Debug mode disabled")

    # Add this method to MainWindow class

    def save_audio_settings(self):
        """Save audio settings to a JSON file for persistence"""
        import json
        
        settings = {
            "input_device_index": self.input_device_index if hasattr(self, "input_device_index") else None,
            "input_gain": self.input_gain if hasattr(self, "input_gain") else 1.0,
            "threshold": self.stt.current_threshold if hasattr(self.stt, "current_threshold") else 550,
            "whisper_mode": self.stt.whisper_mode if hasattr(self.stt, "whisper_mode") else False,
            "wake_word": self.stt.wake_word,
            "wake_word_enabled": self.stt.wake_word_enabled if hasattr(self.stt, "wake_word_enabled") else True
        }
        
        try:
            with open("audio_settings.json", "w") as f:
                json.dump(settings, f)
            self.add_debug_log("Audio settings saved")
        except Exception as e:
            self.add_debug_log(f"Error saving settings: {e}")

    def load_audio_settings(self):
        """Load audio settings from a JSON file"""
        import json
        import os
        
        if not os.path.exists("audio_settings.json"):
            self.add_debug_log("No saved settings found")
            return
            
        try:
            with open("audio_settings.json", "r") as f:
                settings = json.load(f)
                
            # Apply loaded settings
            if "input_device_index" in settings and settings["input_device_index"] is not None:
                self.input_device_index = settings["input_device_index"]
                
            if "input_gain" in settings:
                self.input_gain = settings["input_gain"]
                
            if "threshold" in settings and hasattr(self.stt, "current_threshold"):
                self.stt.current_threshold = settings["threshold"]
                self.threshold_label.setText(f"Threshold: {int(self.stt.current_threshold)}")
                
            if "whisper_mode" in settings and hasattr(self.stt, "whisper_mode"):
                self.stt.toggle_whisper_mode(settings["whisper_mode"])
                self.whisper_mode_button.setChecked(settings["whisper_mode"])
                self.whisper_mode_button.setText(f"Whisper Mode: {'ON' if settings['whisper_mode'] else 'OFF'}")
                
            if "wake_word" in settings:
                self.stt.set_wake_word(settings["wake_word"])
                self.wake_word_input.setText(settings["wake_word"])
                
            if "wake_word_enabled" in settings:
                self.stt.wake_word_enabled = settings["wake_word_enabled"]
                self.wake_word_toggle.setChecked(settings["wake_word_enabled"])
                self.wake_word_toggle.setText(f"Wake Word: {'ON' if settings['wake_word_enabled'] else 'OFF'}")
                
            self.add_debug_log("Audio settings loaded")
        except Exception as e:
            self.add_debug_log(f"Error loading settings: {e}")

    def open_audio_settings(self):
        """Open unified audio settings dialog"""
        dialog = AudioSettingsDialog(self)
        
        # Pre-select current device if set
        if hasattr(self, "input_device_index") and self.input_device_index is not None:
            index = dialog.input_device_combo.findData(self.input_device_index)
            if index >= 0:
                dialog.input_device_combo.setCurrentIndex(index)
        
        # Set current gain
        if hasattr(self, "input_gain"):
            dialog.gain_slider.setValue(int(self.input_gain * 100))
        
        if dialog.exec():
            # Apply settings if OK was clicked
            device_index = dialog.input_device_combo.currentData()
            gain_value = dialog.gain_slider.value() / 100.0
            
            # Log the change
            print(f"Selected device: {device_index}, Gain: {gain_value}")
            
            # Save settings to instance
            self.input_device_index = device_index
            self.input_gain = gain_value
            
            # Apply to speech recognition component
            if self.stt:
                self.stt.input_device_index = device_index
                self.stt.input_gain = gain_value
                
                # Apply changes immediately if possible
                if hasattr(self.stt, "update_audio_device") and callable(self.stt.update_audio_device):
                    self.stt.update_audio_device(device_index, gain_value)
                
                # Update threshold display if it exists
                if hasattr(self, "threshold_label"):
                    self.threshold_label.setText(f"Threshold: {int(self.stt.current_threshold)}")
            
            # Visual confirmation
            self.status_display.setText(f"Audio settings updated: Device {device_index}, Gain {gain_value:.2f}")
            
            # Save settings to file
            self.save_audio_settings()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.load_audio_settings()  # Load settings on startup
    sys.exit(app.exec())