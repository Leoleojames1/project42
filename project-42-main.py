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

# PyQt6 imports
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QComboBox, QLabel, 
                           QTextEdit, QTabWidget, QSplitter, QTableWidget, 
                           QTableWidgetItem, QHeaderView, QScrollArea, QFrame,
                           QDialog, QFileDialog, QListWidget, QListWidgetItem,
                           QMessageBox, QMenu, QInputDialog)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QSize, QTimer, QUrl
from PyQt6.QtGui import QColor, QPalette, QFont, QIcon, QAction

# Speech imports
from speech_to_text import SpeechToText
from text_to_speech import TextToSpeech

# First, add the import for our new component
from audio_visualizer import AudioWaveformVisualizer

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
                if not self.stt.wait_for_wake_word():
                    if not self.running:
                        break
                    continue
                
                # Enter conversation mode
                self.in_conversation = True
                self.update_signal.emit("status", "Conversation started! Listening...")
                self.tts.speak_text("I'm listening", wait_for_completion=False)  # Non-blocking
            
            # In conversation mode: continuously listen and respond
            while self.in_conversation and self.running:
                # Listen for user input with non-blocking approach
                self.update_signal.emit("status", "Listening...")
                user_input = self.stt.start_listening_session(non_blocking=True)
                
                if not user_input or not user_input.strip():
                    # No clear speech detected, keep trying
                    continue
                
                # Check for exit commands
                if any(phrase in user_input.lower() for phrase in ["exit conversation", "end conversation", "stop conversation"]):
                    self.update_signal.emit("status", "Ending conversation...")
                    self.tts.speak_text("Ending our conversation. Say the wake word when you want to chat again.")
                    self.in_conversation = False
                    break
                
                # Process normal input
                self.update_signal.emit("user", user_input)
                self.update_signal.emit("status", "Processing your request...")
                
                # Add user message to context
                self.messages.append({"role": "user", "content": user_input})
                
                # Prepare for streaming response
                self.update_signal.emit("status", "Thinking...")
                self.thinking = True
                self.response_buffer = ""
                
                # Get response from Ollama with streaming
                try:
                    stream = ollama.chat(
                        model=self.model_name,
                        messages=self.messages,
                        stream=True
                    )
                    
                    sentence_buffer = ""
                    full_response = ""
                    last_chunk_time = time.time()
                    sentence_complete = False
                    
                    # Process streaming response with parallel TTS generation
                    for chunk in stream:
                        if not self.running or not self.in_conversation:
                            break
                            
                        if self.stt.speech_interrupted:
                            print("Response generation interrupted by wake word")
                            self.stt.speech_interrupted = False
                            break
                            
                        if 'message' in chunk:
                            content = chunk['message']['content']
                            sentence_buffer += content
                            full_response += content
                            self.update_signal.emit("assistant_stream", content)
                            
                            # Check if we have sentence terminators
                            if any(char in content for char in '.!?:'):
                                sentence_complete = True
                            
                            # Process in sentence chunks for more natural speech
                            current_time = time.time()
                            chunk_interval = current_time - last_chunk_time
                            
                            # Send to TTS if we have a complete sentence or enough content
                            if (sentence_complete or len(sentence_buffer) > 50 or chunk_interval > 0.8) and sentence_buffer.strip():
                                # We have a chunk to process - speak it
                                self.update_signal.emit("status", "Speaking while continuing to think...")
                                
                                # Speak this chunk without waiting for completion
                                print(f"Processing TTS chunk: {sentence_buffer[:30]}...")
                                self.tts.speak_text(sentence_buffer, wait_for_completion=False)
                                
                                # Reset for next chunk
                                sentence_buffer = ""
                                sentence_complete = False
                                last_chunk_time = current_time
                    
                    # Process any remaining text
                    if sentence_buffer.strip():
                        self.tts.speak_text(sentence_buffer, wait_for_completion=False)
                    
                    # Add complete response to context
                    if full_response:
                        self.messages.append({"role": "assistant", "content": full_response})
                        self.update_signal.emit("assistant", full_response)
                        
                    # Wait for TTS to finish
                    while self.tts.is_speaking() and not self.stt.speech_interrupted:
                        time.sleep(0.1)
                        
                    self.thinking = False
                    
                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    self.update_signal.emit("error", error_msg)
                    self.tts.speak_text("Sorry, I encountered an error while processing your request.")
                    self.thinking = False
            
            # Reset for the next conversation cycle
            if not self.in_conversation:
                self.update_signal.emit("status", "Waiting for wake word to start conversation...")
        
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
        
        # Return true if we have a terminator that's not just a trailing comma
        return has_terminator or has_quote_end or bool(list_markers) or has_clause


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        # Set window title and size
        self.setWindowTitle("Project 42 - Speech Assistant")
        self.resize(1200, 800)
        
        # Initialize components
        self.stt = SpeechToText()
        self.tts = TextToSpeech()
        self.ollama_client = OllamaClient()
        self.conversation_manager = ConversationManager()
        
        # Initialize UI
        self.init_ui()
        
        # Apply dark theme
        self.setStyleSheet(StyleSheet.DARK_THEME)
        
        # Show available models
        self.refresh_models()
        
        # Load conversations
        self.load_conversation_list()
        
    def init_ui(self):
        """Initialize UI components"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
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
        visualizer_layout = QHBoxLayout(visualizer_panel)
        visualizer_layout.setContentsMargins(10, 5, 10, 5)
        
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
        visualizer_layout.addWidget(input_visualizer_widget)
        visualizer_layout.addWidget(output_visualizer_widget)
        
        # Add visualizers to main layout
        main_layout.addWidget(visualizer_panel)
        
        # Add visualizer update timer
        self.visualizer_timer = QTimer(self)
        self.visualizer_timer.timeout.connect(self.update_visualizers)
        self.visualizer_timer.start(50)  # Update every 50ms
        
    def refresh_models(self):
        """Refresh available Ollama models"""
        self.model_combo.clear()
        models = self.ollama_client.get_available_models()
        
        if not models:
            self.status_display.setText("No models found. Is Ollama running?")
            return
            
        for model in models:
            self.model_combo.addItem(model)
            
        self.status_display.setText(f"Found {len(models)} models")
        
    def start_assistant(self):
        """Start the speech assistant"""
        if not self.model_combo.currentText():
            self.status_display.setText("Please select a model first")
            return
            
        model_name = self.model_combo.currentText()
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.refresh_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        
        # Clear output display
        self.output_display.clear()
        self.status_display.setText(f"Starting assistant with model: {model_name}")
        
        # Start worker thread
        self.speech_worker = SpeechWorker(
            self.stt, 
            self.tts,
            self.ollama_client,
            model_name
        )
        self.speech_worker.update_signal.connect(self.update_from_worker)
        self.speech_worker.finished_signal.connect(self.worker_finished)
        self.speech_worker.start()
        
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
        """Handle updates from worker thread"""
        print(f"Worker update: {update_type} - {content[:50]}...")  # Debug print
        
        if update_type == "status":
            self.status_display.setText(content)
        elif update_type == "user":
            # Add user message to conversation display
            self.conversation_display.append(f"<b>You:</b> {content}")
            # Add to conversation manager
            self.conversation_manager.add_message("user", content)
        elif update_type == "assistant_stream":
            # Update live output display
            current_text = self.output_display.toPlainText()
            self.output_display.setText(current_text + content)
            # Make sure to scroll to the bottom
            self.output_display.verticalScrollBar().setValue(
                self.output_display.verticalScrollBar().maximum())
        elif update_type == "assistant":
            # Add complete assistant message to conversation display
            self.conversation_display.append(f"<b>Assistant:</b> {content}")
            # Make sure to scroll to the bottom
            self.conversation_display.verticalScrollBar().setValue(
                self.conversation_display.verticalScrollBar().maximum())
            # Add to conversation manager
            self.conversation_manager.add_message("assistant", content)
            # Clear streaming output
            self.output_display.clear()
        elif update_type == "error":
            self.status_display.setText("Error")
            self.output_display.append(f"<span style='color:red'>{content}</span>")
            
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
        """Update audio visualizers with current samples"""
        # Update input visualizer
        if hasattr(self, 'stt') and hasattr(self.stt, 'get_audio_samples'):
            samples = self.stt.get_audio_samples()
            if len(samples) > 0:
                self.input_visualizer.update_samples(samples)
        
        # Update output visualizer
        if hasattr(self, 'tts') and hasattr(self.tts, 'get_audio_samples'):
            samples = self.tts.get_audio_samples()
            if len(samples) > 0:
                self.output_visualizer.update_samples(samples)

    # Add this method to the MainWindow class

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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())