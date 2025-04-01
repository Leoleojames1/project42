#!/usr/bin/env python3
"""
Audio visualizer component for Project 42
Provides real-time waveform visualization for speech input and output
"""

import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtGui import QPainter, QColor, QPen, QLinearGradient
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QRectF, QPointF  # Add QPointF import

class AudioWaveformVisualizer(QWidget):
    """
    Widget for visualizing audio waveforms with neon glow effects
    
    Features:
    - Real-time waveform display
    - Neon color themes with glow effects
    - Separate visualization for input and output audio
    - Smooth transitions and decay effects
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
            # Downsample
            factor = len(normalized) // len(self.samples)
            normalized = normalized[::factor][:len(self.samples)]
        elif len(normalized) < len(self.samples):
            # Upsample with padding
            padding = np.zeros(len(self.samples) - len(normalized))
            normalized = np.concatenate([normalized, padding])
        
        # Apply smoothing for transitions
        self.samples = self.samples * self.smooth_factor + normalized * (1 - self.smooth_factor)
        
        # Update maximum amplitude for scaling
        current_max = self.samples.max()
        if current_max > self.max_amplitude:
            self.max_amplitude = current_max
        
        # Trigger repaint
        self.update()
    
    def set_samples_from_level(self, level):
        """
        Set visualization based on audio level (for simpler integration)
        
        Args:
            level (float): Audio level from 0.0 to 1.0
        """
        # Create a simple waveform pattern based on the level
        samples = np.zeros(100)
        
        # Generate a simple sine wave scaled by the level
        x = np.linspace(0, 3*np.pi, 100)
        samples = np.sin(x) * level * 0.8 + np.random.random(100) * level * 0.2
        
        self.update_samples(samples)
    
    def decay_samples(self):
        """Gradually decay samples for smooth animation when audio stops"""
        if np.sum(self.samples) < 0.01:
            # Almost no signal, keep minimal noise
            self.samples = np.random.random(len(self.samples)) * 0.05
        else:
            # Apply decay
            self.samples = self.samples * self.decay_factor
            
        # Slowly reduce max amplitude to adjust sensitivity
        self.max_amplitude = max(1.0, self.max_amplitude * 0.95)
        
        # Update display
        self.update()
    
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
        
        # Create gradient for glow effect
        gradient = QLinearGradient(0, 0, 0, height)
        gradient.setColorAt(0, self.glow_color.lighter(150))
        gradient.setColorAt(0.5, self.base_color)
        gradient.setColorAt(1, self.glow_color.darker(150))
        
        # Draw glow background (blurry wider line)
        glow_pen = QPen(self.glow_color, 4.0)
        glow_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(glow_pen)
        
        # Draw glow path
        path = []
        for i, sample in enumerate(self.samples):
            x = i * x_scale
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
        main_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(main_pen)
        
        for i in range(len(path) - 1):
            # Use QPointF objects here too
            painter.drawLine(
                QPointF(path[i][0], path[i][1]),
                QPointF(path[i+1][0], path[i+1][1])
            )
            
        # Draw mirror effect (bottom half)
        mirror_pen = QPen(self.base_color, 1.0)
        mirror_pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(mirror_pen)
        
        mirror_path = []
        for i, sample in enumerate(self.samples):
            x = i * x_scale
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
        center_pen.setStyle(Qt.PenStyle.DotLine)
        painter.setPen(center_pen)
        painter.drawLine(QPointF(0, center_y), QPointF(width, center_y))