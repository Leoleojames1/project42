#!/usr/bin/env python3
"""
Audio settings dialog for Project 42
Handles audio device selection and testing
"""

import pyaudio
import numpy as np
import threading
import time
from PyQt6 import sip

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
        """Set up the dialog UI components with separate input and output sections"""
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
        
        # ADD THIS SECTION: Output devices section
        output_group = QGroupBox("Audio Output")
        output_layout = QVBoxLayout(output_group)
        
        output_device_layout = QHBoxLayout()
        output_device_layout.addWidget(QLabel("Output Device:"))
        
        self.output_device_combo = QComboBox()
        output_device_layout.addWidget(self.output_device_combo)
        
        output_layout.addLayout(output_device_layout)
        
        # Add playback test button
        test_output_layout = QHBoxLayout()
        self.test_output_button = QPushButton("Test Speaker")
        self.test_output_button.clicked.connect(self.test_output_device)
        test_output_layout.addWidget(self.test_output_button)
        
        # Output volume slider
        output_volume_layout = QHBoxLayout()
        output_volume_layout.addWidget(QLabel("Output Volume:"))
        
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(80)
        self.volume_slider.setTickInterval(10)
        self.volume_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        output_volume_layout.addWidget(self.volume_slider)
        
        self.volume_label = QLabel("80%")
        output_volume_layout.addWidget(self.volume_label)
        
        self.volume_slider.valueChanged.connect(self.update_volume_label)
        
        output_layout.addLayout(test_output_layout)
        output_layout.addLayout(output_volume_layout)
        
        layout.addWidget(output_group)
        
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
        self.output_device_combo.clear()
        
        # Add system default options
        self.input_device_combo.addItem("Default System Microphone", None)
        self.output_device_combo.addItem("Default System Speakers", None)
        
        # Add a line to show which device is currently selected
        if hasattr(self.parent(), 'input_device_index') and self.parent().input_device_index is not None:
            current_input = self.parent().input_device_index
        else:
            current_input = None
            
        if hasattr(self.parent(), 'output_device_index') and self.parent().output_device_index is not None:
            current_output = self.parent().output_device_index
        else:
            current_output = None
        
        # Find all input and output devices
        for i in range(self.p.get_device_count()):
            device_info = self.p.get_device_info_by_index(i)
            
            # Handle input devices
            if device_info['maxInputChannels'] > 0:
                name = device_info['name']
                is_current = current_input == i
                self.input_device_combo.addItem(
                    f"{name}{' (CURRENT)' if is_current else ''}", 
                    i
                )
                
                # Select this device if it's the current one
                if is_current:
                    self.input_device_combo.setCurrentIndex(self.input_device_combo.count() - 1)
            
            # Handle output devices
            if device_info['maxOutputChannels'] > 0:
                name = device_info['name']
                is_current = current_output == i
                self.output_device_combo.addItem(
                    f"{name}{' (CURRENT)' if is_current else ''}",
                    i
                )
                
                # Select this device if it's the current one
                if is_current:
                    self.output_device_combo.setCurrentIndex(self.output_device_combo.count() - 1)
                    
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
            try:
                # Calculate audio level
                data = np.frombuffer(in_data, dtype=np.int16)
                level = np.abs(data).mean() / 32768.0  # Normalize to 0.0-1.0
                
                # Apply gain - SAFETY CHECK ADDED
                if hasattr(self, 'gain_slider') and not sip.isdeleted(self.gain_slider):
                    gain = self.gain_slider.value() / 100.0
                else:
                    gain = 1.0  # Default gain if slider unavailable
                
                adjusted_level = min(1.0, level * gain)
                
                # Emit signal for visualization - SAFETY CHECK
                if hasattr(self, 'audio_level_signal') and not self.audio_level_signal.destroyed:
                    self.audio_level_signal.emit(adjusted_level)
                
                # Update indicator light - SAFETY CHECK
                if hasattr(self, 'indicator_frame') and not sip.isdeleted(self.indicator_frame):
                    if adjusted_level > 0.05:
                        color = f"background-color: rgb({min(255, int(adjusted_level * 255))}, 200, 0); border-radius: 15px;"
                        self.indicator_frame.setStyleSheet(color)
                    else:
                        self.indicator_frame.setStyleSheet("background-color: #333; border-radius: 15px;")
            except Exception as e:
                # Silent fail for callback
                pass
        
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

    def update_volume_label(self):
        """Update the volume label when slider changes"""
        value = self.volume_slider.value()
        self.volume_label.setText(f"{value}%")

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

    def test_output_device(self):
        """Test the selected output device with a short sound"""
        from pygame import mixer
        import os
        import sys
        import tempfile
        
        try:
            device_idx = self.output_device_combo.currentData()
            
            # Generate a simple test tone using PyAudio
            def generate_tone(frequency=440, duration=1, volume=0.5):
                sample_rate = 44100
                samples = int(duration * sample_rate)
                
                # Generate a sine wave
                import numpy as np
                buf = np.sin(2 * np.pi * np.arange(samples) * frequency / sample_rate)
                # Apply volume
                buf = buf * volume
                # Convert to 16-bit PCM
                buf = (buf * 32767).astype(np.int16)
                
                return buf
                
            # Create a temp file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            temp_file.close()
            
            # Write the tone to a WAV file
            import wave
            sampleRate = 44100
            duration = 0.5  # half second
            frequency = 440  # A4
            
            wf = wave.open(temp_file.name, 'wb')
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sampleRate)
            
            # Generate and write the tone
            tone = generate_tone(frequency, duration, self.volume_slider.value() / 100.0)
            wf.writeframes(tone.tobytes())
            wf.close()
            
            # Play the tone
            try:
                mixer.init(devicename=device_idx)
                mixer.music.load(temp_file.name)
                mixer.music.play()
                
                # Change button text temporarily
                original_text = self.test_output_button.text()
                self.test_output_button.setText("Playing...")
                self.test_output_button.setEnabled(False)
                
                # Use timer to restore button
                QTimer.singleShot(1000, lambda: self.restore_output_button(original_text))
                
            except Exception as e:
                print(f"Error playing test tone: {e}")
                self.test_output_button.setText("Error")
                QTimer.singleShot(1000, lambda: self.test_output_button.setText("Test Speaker"))
                
            # Clean up the temp file
            try:
                os.unlink(temp_file.name)
            except:
                pass
                
        except Exception as e:
            print(f"Error in output device test: {e}")
            self.test_output_button.setText("Error")
            QTimer.singleShot(1000, lambda: self.test_output_button.setText("Test Speaker"))

    def restore_output_button(self, text):
        """Restore the output test button state"""
        self.test_output_button.setText(text)
        self.test_output_button.setEnabled(True)

    def closeEvent(self, event):
        """Clean up resources when dialog is closed"""
        self.stop_mic_test()
        
        # Clean up PyAudio instance
        if hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except:
                pass
                
        super().closeEvent(event)

# Add to MainWindow class

def save_audio_settings(self):
    """Save audio settings to a JSON file for persistence"""
    import json
    
    settings = {
        "input_device_index": self.input_device_index if hasattr(self, "input_device_index") else None,
        "input_gain": self.input_gain if hasattr(self, "input_gain") else 1.0,
        "output_device_index": self.output_device_index if hasattr(self, "output_device_index") else None,
        "output_volume": self.output_volume if hasattr(self, "output_volume") else 80,
        "threshold": self.stt.current_threshold if hasattr(self.stt, "current_threshold") else 550,
        "whisper_mode": self.stt.whisper_mode if hasattr(self.stt, "whisper_mode") else False,
        "wake_word": self.stt.wake_word,
        "wake_word_enabled": self.stt.wake_word_enabled if hasattr(self.stt, "wake_word_enabled") else True
    }
    
    try:
        with open("audio_settings.json", "w") as f:
            json.dump(settings, f)
        print("Audio settings saved")
    except Exception as e:
        print(f"Error saving settings: {e}")

def open_audio_settings(self):
    """Open unified audio settings dialog"""
    dialog = AudioSettingsDialog(self)
    
    if dialog.exec():
        # Apply settings if OK was clicked
        
        # Input device settings
        input_device_index = dialog.input_device_combo.currentData()
        input_gain_value = dialog.gain_slider.value() / 100.0
        
        # Output device settings
        output_device_index = dialog.output_device_combo.currentData()
        output_volume = dialog.volume_slider.value()
        
        # Store settings
        self.input_device_index = input_device_index
        self.input_gain = input_gain_value
        self.output_device_index = output_device_index
        self.output_volume = output_volume
        
        # Apply to speech components
        if self.stt:
            self.stt.input_device_index = input_device_index
            self.stt.input_gain = input_gain_value
            
            # If method exists, apply device change
            if hasattr(self.stt, "update_audio_device") and callable(self.stt.update_audio_device):
                self.stt.update_audio_device(input_device_index, input_gain_value)
        
        if self.tts:
            # If TTS supports output device selection
            if hasattr(self.tts, "set_output_device"):
                self.tts.set_output_device(output_device_index, output_volume / 100.0)
        
        # Show confirmation
        self.status_display.setText(f"Audio settings updated")
        
        # Save settings
        self.save_audio_settings()