#!/usr/bin/env python3
"""
Audio visualizer component for Project 42
Provides real-time waveform visualization for speech input and output
"""
import numpy as np
import pyaudio
from PyQt6.QtWidgets import QWidget, QMenu, QDialog
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient, QRadialGradient, QIcon
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF, QSize  

from audio_settings import AudioSettingsDialog

# Make sure we're using the correct enum references
# You can add these constants at the top of your file after imports
PEN_STYLE_NOPEN = Qt.PenStyle.NoPen
PEN_STYLE_DOTLINE = Qt.PenStyle.DotLine
PEN_CAP_ROUND = Qt.PenCapStyle.RoundCap

class AudioWaveformVisualizer(QWidget):
    """
    Widget for visualizing audio waveforms with neon glow effects
    
    Features:
    - Real-time waveform display
    - Neon color themes with glow effects
    - Separate visualization for input and output audio
    - Audio activity indicator light
    - Settings access for audio devices
    """
    
    def __init__(self, parent=None, mode="input"):
        """
        Initialize audio visualizer widget
        
        Args:
            parent: Parent widget
            mode (str): 'input' for microphone, 'output' for TTS
        """
        super().__init__(parent)
        
        # Set minimum size
        self.setMinimumHeight(60)
        self.setMinimumWidth(100)
        
        # Visualization data
        self.samples = np.zeros(100)
        self.mode = mode
        
        # Visual settings
        if mode == "input":
            # Cyan neon for input (microphone)
            self.base_color = QColor(0, 200, 255)
            self.glow_color = QColor(0, 230, 255, 150)
        else:
            # Purple neon for output (TTS)
            self.base_color = QColor(200, 0, 255)
            self.glow_color = QColor(210, 0, 255, 150)
        
        # Animation settings
        self.decay_factor = 0.8  # Decay rate for smooth transitions
        self.max_amplitude = 1.0
        self.smooth_factor = 0.3
        
        # Add indicator light settings
        self.audio_active = False
        self.activity_threshold = 0.1
        self.light_intensity = 0.0
        
        # Settings button
        self.settings_icon = QIcon("icons/settings.png")  # You'll need to add this icon
        self.show_settings_button = True
        
        # Enable context menu for right-click
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        
        # Update timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.decay_samples)
        self.timer.start(50)  # Update every 50ms
        
        # Set background to transparent
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    
    def update_samples(self, new_samples):
        """
        Update visualization with new audio samples
        
        Args:
            new_samples (np.ndarray): New audio samples to visualize
        """
        if len(new_samples) == 0:
            return
            
        # Normalize and process input samples
        normalized = np.abs(new_samples)
        if normalized.max() > 0:
            normalized = normalized / normalized.max()
        
        # Resize to our display size if needed
        if len(normalized) > len(self.samples):
            normalized = normalized[:len(self.samples)]
        elif len(normalized) < len(self.samples):
            # Pad with zeros
            padding = np.zeros(len(self.samples) - len(normalized))
            normalized = np.concatenate((normalized, padding))
        
        # Apply smoothing for transitions
        self.samples = self.samples * self.smooth_factor + normalized * (1 - self.smooth_factor)
        
        # Update maximum amplitude for scaling
        current_max = self.samples.max()
        if current_max > self.max_amplitude:
            self.max_amplitude = current_max
        
        # Update audio activity state for indicator light
        avg_level = np.mean(self.samples)
        if avg_level > self.activity_threshold:
            self.audio_active = True
            self.light_intensity = min(1.0, avg_level * 4)  # Scale up for better visibility
        else:
            self.audio_active = False
            self.light_intensity = max(0.1, self.light_intensity * 0.9)  # Smooth fadeout
        
        # Trigger repaint
        self.update()
    
    def set_samples_from_level(self, level):
        """
        Set visualization based on audio level
        
        Args:
            level (float): Audio level from 0.0 to 1.0
        """
        if level > 0:
            # Create a pulse effect based on the level
            self.audio_active = level > self.activity_threshold
            self.light_intensity = level * 2  # Scale up for visibility
            
            # Generate samples that respond to the level
            center_pulse = np.sin(np.linspace(0, np.pi, len(self.samples))) * level
            self.samples = self.samples * 0.3 + center_pulse * 0.7
            self.update()
    
    def decay_samples(self):
        """Gradually decay samples for smooth animation when audio stops"""
        if np.sum(self.samples) < 0.01:
            # If nearly silent, reset everything
            self.samples = np.zeros_like(self.samples)
            self.light_intensity *= 0.7  # Fade out light
        else:
            # Apply decay
            self.samples = self.samples * self.decay_factor
            
        # Slowly reduce max amplitude to adjust sensitivity
        self.max_amplitude = max(1.0, self.max_amplitude * 0.95)
        
        # Update display
        self.update()
    
    def show_context_menu(self, position):
        """Show a context menu with settings options"""
        menu = QMenu(self)
        settings_action = menu.addAction("Audio Settings...")
        action = menu.exec(self.mapToGlobal(position))
        
        if action == settings_action:
            self.open_settings_dialog()
            
    def open_settings_dialog(self):
        """Open audio settings dialog"""
        dialog = AudioSettingsDialog(self)
        if dialog.exec():
            # Apply settings if OK was clicked
            selected_device = dialog.input_device_combo.currentData()
            gain_value = dialog.gain_slider.value() / 100.0  # Convert to float factor
            # You would pass these values to your audio handler
            print(f"Selected device: {selected_device}, Gain: {gain_value}")
    
    def paintEvent(self, event):
        """Draw the waveform with neon glow effect"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Get widget dimensions
        width = self.width()
        height = self.height()
        center_y = height / 2
        
        # Calculate scaling
        x_scale = width / len(self.samples)
        y_scale = height / 2.5  # Scale to 80% of half-height
        
        # Draw indicator light (first, so it's in the background)
        light_radius = height / 5
        light_x = light_radius + 5
        light_y = center_y
        
        # Create gradient for the light
        if self.audio_active:
            # Active light
            light_gradient = QRadialGradient(light_x, light_y, light_radius)
            light_gradient.setColorAt(0, QColor(self.base_color.red(), 
                                              self.base_color.green(),
                                              self.base_color.blue(),
                                              255))
            light_gradient.setColorAt(0.7, QColor(self.base_color.red(), 
                                               self.base_color.green(),
                                               self.base_color.blue(),
                                               180))
            light_gradient.setColorAt(1, QColor(self.base_color.red(), 
                                             self.base_color.green(),
                                             self.base_color.blue(),
                                             0))
        else:
            # Inactive light
            light_gradient = QRadialGradient(light_x, light_y, light_radius)
            light_gradient.setColorAt(0, QColor(80, 80, 80, 100))
            light_gradient.setColorAt(0.7, QColor(60, 60, 60, 70))
            light_gradient.setColorAt(1, QColor(40, 40, 40, 0))
        
        # Draw the light
        painter.setPen(PEN_STYLE_NOPEN)
        painter.setBrush(light_gradient)
        painter.drawEllipse(QPointF(light_x, light_y), 
                           light_radius * self.light_intensity, 
                           light_radius * self.light_intensity)
        
        # Draw "light bulb" indicator on the right side
        bulb_x = width - light_radius - 5
        bulb_y = center_y
        
        # Create gradient for the bulb
        if self.audio_active:
            # Active bulb
            bulb_gradient = QRadialGradient(bulb_x, bulb_y, light_radius)
            bulb_gradient.setColorAt(0, QColor(self.base_color.red(), 
                                             self.base_color.green(),
                                             self.base_color.blue(),
                                             255))
            bulb_gradient.setColorAt(0.7, QColor(self.base_color.red(), 
                                              self.base_color.green(),
                                              self.base_color.blue(),
                                              180))
            bulb_gradient.setColorAt(1, QColor(self.base_color.red(), 
                                            self.base_color.green(),
                                            self.base_color.blue(),
                                            0))
        else:
            # Inactive bulb
            bulb_gradient = QRadialGradient(bulb_x, bulb_y, light_radius)
            bulb_gradient.setColorAt(0, QColor(80, 80, 80, 100))
            bulb_gradient.setColorAt(0.7, QColor(60, 60, 60, 70))
            bulb_gradient.setColorAt(1, QColor(40, 40, 40, 0))
        
        # Draw the bulb
        painter.setPen(PEN_STYLE_NOPEN)
        painter.setBrush(bulb_gradient)
        painter.drawEllipse(QPointF(bulb_x, bulb_y), 
                           light_radius * self.light_intensity, 
                           light_radius * self.light_intensity)
        
        # Adjust layout to make room for the indicator lights
        waveform_start_x = light_x + light_radius + 10
        waveform_width = width - (light_radius * 2) - 30
        
        # Create gradient for glow effect
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, self.glow_color.lighter(150))
        gradient.setColorAt(0.5, self.base_color)
        gradient.setColorAt(1, self.glow_color.darker(150))
        
        # Draw glow background (blurry wider line)
        glow_pen = QPen(self.glow_color, 4.0)
        glow_pen.setCapStyle(PEN_CAP_ROUND)  # Correct enum reference
        painter.setPen(glow_pen)
        
        # Draw glow path
        path = []
        for i, sample in enumerate(self.samples):
            x = waveform_start_x + (i * waveform_width / len(self.samples))
            y = center_y - (sample * y_scale)
            path.append((x, y))
            
        for i in range(len(path) - 1):
            # Use QPointF objects for floating-point coordinates
            painter.drawLine(
                QPointF(path[i][0], path[i][1]),
                QPointF(path[i+1][0], path[i+1][1])
            )
            
        # Draw main waveform line
        main_pen = QPen(self.base_color, 2.0)
        main_pen.setCapStyle(PEN_CAP_ROUND)  # Correct enum reference
        painter.setPen(main_pen)
        
        for i in range(len(path) - 1):
            # Use QPointF objects here too
            painter.drawLine(
                QPointF(path[i][0], path[i][1]),
                QPointF(path[i+1][0], path[i+1][1])
            )
            
        # Draw mirror effect (bottom half)
        mirror_pen = QPen(self.base_color, 1.0)
        mirror_pen.setCapStyle(PEN_CAP_ROUND)  # Correct enum reference
        painter.setPen(mirror_pen)
        
        mirror_path = []
        for i, sample in enumerate(self.samples):
            x = waveform_start_x + (i * waveform_width / len(self.samples))
            y = center_y + (sample * y_scale * 0.7)  # Slightly smaller for mirror
            mirror_path.append((x, y))
            
        for i in range(len(mirror_path) - 1):
            # Use QPointF objects here too
            painter.drawLine(
                QPointF(mirror_path[i][0], mirror_path[i][1]),
                QPointF(mirror_path[i+1][0], mirror_path[i+1][1])
            )
            
        # Draw center line for reference
        center_pen = QPen(self.base_color.lighter(120), 1.0)
        center_pen.setStyle(PEN_STYLE_DOTLINE)  # This is correct as is
        painter.setPen(center_pen)
        painter.drawLine(
            QPointF(waveform_start_x, center_y), 
            QPointF(waveform_start_x + waveform_width, center_y)
        )
        
        # Draw settings gear icon in top right corner
        if self.show_settings_button:
            if not self.settings_icon.isNull():
                self.settings_icon.paint(painter, width - 24, 5, 20, 20)
            else:
                # Fallback if icon is missing - draw a simple gear shape
                gear_x = width - 15
                gear_y = 15
                gear_radius = 8
                painter.setPen(QPen(QColor(200, 200, 200), 1.5))
                painter.drawEllipse(QPointF(gear_x, gear_y), gear_radius, gear_radius)
                painter.drawEllipse(QPointF(gear_x, gear_y), gear_radius/2, gear_radius/2)
                
                # Draw gear teeth
                for angle in range(0, 360, 45):
                    rad = angle * 3.14159 / 180
                    x1 = gear_x + (gear_radius * 0.8) * np.cos(rad)
                    y1 = gear_y + (gear_radius * 0.8) * np.sin(rad)
                    x2 = gear_x + (gear_radius * 1.3) * np.cos(rad)
                    y2 = gear_y + (gear_radius * 1.3) * np.sin(rad)
                    painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))
    
    def mousePressEvent(self, event):
        """Handle mouse clicks on settings icon"""
        # Check if click is on settings icon
        gear_x = self.width() - 15
        gear_y = 15
        gear_radius = 12  # Slightly larger hit area
        
        dx = event.position().x() - gear_x
        dy = event.position().y() - gear_y
        
        if (dx*dx + dy*dy) <= gear_radius*gear_radius:
            self.open_settings_dialog()