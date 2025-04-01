#!/usr/bin/env python3
"""
Audio settings dialog for Project 42
Handles audio device selection and testing
"""

import pyaudio
import numpy as np
import threading
import time

from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                           QPushButton, QComboBox, QGroupBox, QCheckBox,
                           QSlider, QFrame, QListWidget)
from PyQt6.QtGui import QIcon, QPixmap, QFont
from PyQt6.QtCore import Qt, pyqtSignal, QTimer

class AudioSettingsDialog(QDialog):
    """Dialog for audio device settings and testing"""
    
    # Signal to update test visualizer
    audio_level_signal = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.setWindowTitle("Audio Settings")
        self.setMinimumWidth(400)
        self.setMinimumHeight(300)
        
        # Audio test variables
        self.is_testing = False
        self.test_stream = None
        self.p = pyaudio.PyAudio()
        
        # Create UI
        self.setup_ui()
        
        # Populate devices
        self.refresh_devices()
        
    def setup_ui(self):
        """Set up the dialog UI components"""
        layout = QVBoxLayout(self)
        
        # Input devices section
        input_group = QGroupBox("Microphone Input")
        input_layout = QVBoxLayout(input_group)
        
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Input Device:"))
        
        self.input_device_combo = QComboBox()
        device_layout.addWidget(self.input_device_combo)
        
        refresh_button = QPushButton("Refresh")
        refresh_button.clicked.connect(self.refresh_devices)
        device_layout.addWidget(refresh_button)
        
        input_layout.addLayout(device_layout)
        
        # Test section
        test_layout = QHBoxLayout()
        self.test_button = QPushButton("Test Microphone")
        self.test_button.clicked.connect(self.toggle_mic_test)
        test_layout.addWidget(self.test_button)
        
        # Add indicator light
        self.indicator_frame = QFrame()
        self.indicator_frame.setFixedSize(30, 30)
        self.indicator_frame.setStyleSheet("background-color: #333; border-radius: 15px;")
        test_layout.addWidget(self.indicator_frame)
        
        input_layout.addLayout(test_layout)
        
        # Input gain slider
        gain_layout = QHBoxLayout()
        gain_layout.addWidget(QLabel("Input Gain:"))
        
        self.gain_slider = QSlider(Qt.Orientation.Horizontal)
        self.gain_slider.setMinimum(50)
        self.gain_slider.setMaximum(200)
        self.gain_slider.setValue(100)
        self.gain_slider.setTickInterval(25)
        self.gain_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        gain_layout.addWidget(self.gain_slider)
        
        self.gain_label = QLabel("100%")
        gain_layout.addWidget(self.gain_label)
        
        self.gain_slider.valueChanged.connect(self.update_gain_label)
        input_layout.addLayout(gain_layout)
        
        layout.addWidget(input_group)
        
        # Virtual Audio Option
        self.virtual_audio_check = QCheckBox("Enable Virtual Audio Input (capture system audio)")
        layout.addWidget(self.virtual_audio_check)
        # Enable this now that we're implementing it
        self.virtual_audio_check.setEnabled(True)
        self.virtual_audio_check.setToolTip("Capture audio from other applications")
        self.virtual_audio_check.stateChanged.connect(self.toggle_virtual_audio)
        
        # Buttons
        button_layout = QHBoxLayout()
        cancel_button = QPushButton("Cancel")
        cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(cancel_button)
        
        ok_button = QPushButton("OK")
        ok_button.clicked.connect(self.accept)
        button_layout.addWidget(ok_button)
        
        layout.addLayout(button_layout)
        
        # Update timer for indicator
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self.update_indicator)
        
    def refresh_devices(self):
        """Refresh the list of available audio devices"""
        self.input_device_combo.clear()
        
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            
            # Only add input devices
            if device_info['maxInputChannels'] > 0:
                name = device_info['name']
                self.input_device_combo.addItem(f"{name}", i)
                
    def toggle_mic_test(self):
        """Start or stop microphone testing"""
        if not self.is_testing:
            self.start_mic_test()
        else:
            self.stop_mic_test()
            
    def start_mic_test(self):
        """Start testing the selected microphone"""
        try:
            device_idx = self.input_device_combo.currentData()
            
            if device_idx is None:
                return
                
            # Configure PyAudio stream
            self.test_stream = self.p.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=44100,
                input=True,
                input_device_index=device_idx,
                frames_per_buffer=1024,
                stream_callback=self.audio_callback
            )
            
            self.is_testing = True
            self.test_button.setText("Stop Test")
            self.update_timer.start(50)
            
        except Exception as e:
            print(f"Error starting microphone test: {e}")
            self.indicator_frame.setStyleSheet("background-color: red; border-radius: 15px;")
            
    def stop_mic_test(self):
        """Stop testing the microphone"""
        if self.test_stream:
            self.test_stream.stop_stream()
            self.test_stream.close()
            self.test_stream = None
            
        self.is_testing = False
        self.test_button.setText("Test Microphone")
        self.update_timer.stop()
        self.indicator_frame.setStyleSheet("background-color: #333; border-radius: 15px;")
        
        # Reset level indicator
        self.audio_level_signal.emit(0.0)
        
    def audio_callback(self, in_data, frame_count, time_info, status):
        """Process audio data from mic test"""
        if self.is_testing:
            # Calculate audio level
            data = np.frombuffer(in_data, dtype=np.int16)
            level = np.abs(data).mean() / 32768.0  # Normalize to 0.0-1.0
            
            # Apply gain
            gain = self.gain_slider.value() / 100.0
            adjusted_level = min(1.0, level * gain)
            
            # Emit signal for visualization
            self.audio_level_signal.emit(adjusted_level)
            
            # Update indicator light based on audio level
            if adjusted_level > 0.05:
                color = f"background-color: rgb({min(255, int(adjusted_level * 255))}, 200, 0); border-radius: 15px;"
                self.indicator_frame.setStyleSheet(color)
            else:
                self.indicator_frame.setStyleSheet("background-color: #333; border-radius: 15px;")
                
        # Continue with PyAudio callback requirements
        return (in_data, pyaudio.paContinue)
        
    def update_indicator(self):
        """Update the indicator appearance based on current level"""
        # This is called by timer to handle UI updates outside the audio callback
        pass

    def update_gain_label(self):
        """Update the gain label when slider changes"""
        value = self.gain_slider.value()
        self.gain_label.setText(f"{value}%")

    def toggle_virtual_audio(self, state):
        """Toggle virtual audio input mode"""
        if state == Qt.CheckState.Checked.value:
            # Show system audio output selection
            sources_dialog = QDialog(self)
            sources_dialog.setWindowTitle("Select Audio Source")
            sources_layout = QVBoxLayout(sources_dialog)
            
            sources_list = QListWidget()
            # Populate with available audio output devices
            for i in range(self.p.get_device_count()):
                device_info = self.p.get_device_info_by_index(i)
                if device_info['maxOutputChannels'] > 0:
                    sources_list.addItem(f"{device_info['name']}")
                    
            sources_layout.addWidget(QLabel("Select audio source to capture:"))
            sources_layout.addWidget(sources_list)
            
            button_layout = QHBoxLayout()
            cancel_btn = QPushButton("Cancel")
            cancel_btn.clicked.connect(sources_dialog.reject)
            select_btn = QPushButton("Select")
            select_btn.clicked.connect(sources_dialog.accept)
            
            button_layout.addWidget(cancel_btn)
            button_layout.addWidget(select_btn)
            
            sources_layout.addLayout(button_layout)
            
            # Show dialog and process result
            if sources_dialog.exec():
                selected = sources_list.currentRow()
                if selected >= 0:
                    print(f"Selected virtual audio source: {sources_list.item(selected).text()}")
                    # Here you would set up virtual audio routing
                    # This requires platform-specific implementation
                else:
                    self.virtual_audio_check.setChecked(False)
            else:
                self.virtual_audio_check.setChecked(False)
        else:
            # Disable virtual audio
            pass