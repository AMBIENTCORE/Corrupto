#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Glitchy Image Editor — Corrupto Edition

This script implements a simple image editing application using PySide6 and OpenCV.
It features a dark themed GUI inspired by the Phonodex project with pixel art fonts,
multi-selection of effects, and basic file operations.  The first available filter
converts images to black & white with adjustable parameters.

Key features:
 - Dark theme with custom colors and optional pixel font loaded from local assets.
 - Open and Save buttons supporting common image formats (PNG, JPEG, BMP, WebP, TIFF).
 - List of available filters with multi‑selection; each selected effect exposes its own settings.
 - Real‑time preview; apply button commits the current selection to the working image; undo and reset support.

To run this application:
    pip install PySide6 opencv-python numpy
    python main.py

Note: If a custom pixel font is available in "D:\\Documents\\Projects\\Corrupto\\assets\\I-pixel-u.ttf"
or in the local "assets" folder relative to this script, it will be loaded and used globally.

"""

import os
import sys
import cv2
import numpy as np
import math
from dataclasses import dataclass
from typing import Optional, List

from PySide6.QtCore import Qt, QSize, Signal
from PySide6.QtGui import QPixmap, QColor, QPalette, QAction, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QMessageBox,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QListWidget,
    QListWidgetItem,
    QWidget,
    QPushButton,
    QLabel,
    QGroupBox,
    QScrollArea,
    QSlider,
    QSplitter,
    QSizePolicy,
    QStyleFactory,
    QAbstractItemView,
    QGridLayout,
    QDial,
    QLineEdit,
)

# Import from effects module
from effects import (
    np_to_qimage, clamp_img, BaseFilter, KnobControl,
    BlackAndWhiteFilter, RGBShiftFilter, WaveDistortionFilter,
    BlockifyFilter, PixelSortFilter, NoiseInjectionFilter,
    SliceShiftFilter, CRTScanFilter, PixelSmashFilter,
    GlitchBlocksFilter, ASCIIArtFilter, MeshFilter
)

# ---------------------------------------------------------------------
# Theme constants
# ---------------------------------------------------------------------

# Colour palette drawn from the Phonodex project's config. These constants
# define the primary and secondary backgrounds as well as success and error
# accents. Feel free to adjust these values for different themes.
COLORS = {
    "SUCCESS": "#4CAF50",
    "ERROR": "#F44336",
    "BACKGROUND": "#1e1e1e",
    "SECONDARY_BACKGROUND": "#252526",
    "TEXT": "#ffffff",
    "SECONDARY_TEXT": "#cccccc",
    "PROGRESS_TROUGH": "#2d2d2d",
}

# Candidate font locations. The application will attempt to load the first
# existing font file in this list. The 'I pixel u' font gives a retro
# pixelated look.  If not found, the default system font is used.
PIXEL_FONT_PATHS = [
    # Attempt to load a font from an absolute Windows path; slash-forward avoids escape issues
    os.path.join("D:/Documents/Projects/Corrupto/assets", "I-pixel-u.ttf"),
    os.path.join(os.path.dirname(__file__), "assets", "I-pixel-u.ttf"),
]
PIXEL_FONT_FAMILY = "I pixel u"





























# ---------------------------------------------------------------------
# Noise Injection effect
# ---------------------------------------------------------------------

class NoiseInjectionFilter(BaseFilter):
    """
    Add various types of noise to the image to create glitch artifacts.
    Supports different noise types: gaussian, salt & pepper, and structured noise.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: intensity, noise type, seed
        self.intensity = 0      # [0, 100] noise strength
        self.noise_type = 0     # [0, 2] 0=gaussian, 1=salt&pepper, 2=structured
        self.seed = 0           # [0, 100] random seed for consistent results

    @property
    def name(self) -> str:
        return "NOISE INJECT"

    def reset_params(self) -> None:
        self.intensity = 0
        self.noise_type = 0
        self.seed = 0
        self._ui = None

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        from PySide6.QtWidgets import QFrame
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#FF5722"  # deep orange accent
        panel.setStyleSheet(
            f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        params = [
            ("Intensity", "intensity", 0, 100, 0),
            ("Type", "noise_type", 0, 2, 0),
            ("Seed", "seed", 0, 100, 0),
        ]
        
        # Calculate and set fixed height based on content
        rows_needed = (len(params) + 2) // 3  # 3 knobs per row, round up
        panel_height = rows_needed * 60 + 20  # 20px for margins
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self,
                attr,
                disp,
                min_a,
                max_a,
                current,
                panel,
                default_actual=neutral,
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
        if self.intensity <= 0:
            return src_bgr
            
        # Set random seed for consistent results
        if self.seed > 0:
            np.random.seed(self.seed)
            
        intensity = self.intensity / 100.0
        h, w, c = src_bgr.shape
        
        if self.noise_type == 0:  # Gaussian noise
            # Add gaussian noise to all channels
            noise = np.random.normal(0, intensity * 50, (h, w, c)).astype(np.float32)
            result = src_bgr.astype(np.float32) + noise
            return clamp_img(result)
            
        elif self.noise_type == 1:  # Salt & Pepper noise
            # Create salt & pepper noise mask
            noise_mask = np.random.random((h, w, c)) < (intensity * 0.3)
            salt_mask = np.random.random((h, w, c)) < 0.5
            pepper_mask = ~salt_mask
            
            result = src_bgr.copy().astype(np.float32)
            # Add salt (white pixels)
            result[salt_mask & noise_mask] = 255
            # Add pepper (black pixels)
            result[pepper_mask & noise_mask] = 0
            return clamp_img(result)
            
        else:  # Structured noise (self.noise_type == 2)
            # Create structured noise pattern
            x_coords = np.arange(w)
            y_coords = np.arange(h)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            # Create sine wave interference pattern
            freq_x = 0.1 + intensity * 0.2
            freq_y = 0.05 + intensity * 0.15
            noise = np.sin(X * freq_x) * np.cos(Y * freq_y) * intensity * 100
            
            # Apply to all channels
            noise_3d = np.stack([noise] * c, axis=2)
            result = src_bgr.astype(np.float32) + noise_3d
            return clamp_img(result)


# ---------------------------------------------------------------------
# Slice Shifting effect
# ---------------------------------------------------------------------

class SliceShiftFilter(BaseFilter):
    """
    Cut the image into horizontal strips and randomly shift them left/right
    to create VHS tracking error effects.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: strip height, shift amount, frequency
        self.strip_height = 0   # [0, 50] height of each strip in pixels
        self.shift_amount = 0   # [0, 100] maximum shift amount
        self.frequency = 0      # [0, 100] how often strips are shifted

    @property
    def name(self) -> str:
        return "SLICE SHIFT"

    def reset_params(self) -> None:
        self.strip_height = 0
        self.shift_amount = 0
        self.frequency = 0
        self._ui = None

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        from PySide6.QtWidgets import QFrame
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#E91E63"  # pink accent
        panel.setStyleSheet(
            f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        params = [
            ("Freq", "frequency", 0, 100, 0),
            ("Height", "strip_height", 0, 50, 0),
            ("Shift", "shift_amount", 0, 100, 0),
        ]
        
        # Calculate and set fixed height based on content
        rows_needed = (len(params) + 2) // 3  # 3 knobs per row, round up
        panel_height = rows_needed * 60 + 20  # 20px for margins
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self,
                attr,
                disp,
                min_a,
                max_a,
                current,
                panel,
                default_actual=neutral,
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
        if self.strip_height <= 0 or self.shift_amount <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy()
        
        # Calculate strip height (minimum 1 pixel)
        strip_h = max(1, int(self.strip_height))
        max_shift = int(self.shift_amount * w / 100)  # Convert percentage to pixels
        frequency = self.frequency / 100.0  # Convert to 0-1 range
        
        # Process each strip
        for y_start in range(0, h, strip_h):
            y_end = min(y_start + strip_h, h)
            
            # Randomly decide if this strip should be shifted
            if np.random.random() < frequency:
                # Random shift amount (positive or negative)
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift != 0:
                    # Shift the strip horizontally
                    strip = result[y_start:y_end, :, :]
                    shifted_strip = np.roll(strip, shift, axis=1)
                    result[y_start:y_end, :, :] = shifted_strip
        
        return result


# ---------------------------------------------------------------------
# CRT Scan effect
# ---------------------------------------------------------------------

class CRTScanFilter(BaseFilter):
    """
    Simulates old CRT monitor scan lines and curvature.
    Creates curved edges and horizontal scan lines for retro aesthetic.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: intensity, curvature, scan thickness
        self.intensity = 0      # [0, 100] overall effect strength
        self.curvature = 0      # [0, 100] screen curvature amount
        self.scan_thickness = 0 # [0, 20] thickness of scan lines

    @property
    def name(self) -> str:
        return "CRT SCAN"

    def reset_params(self) -> None:
        self.intensity = 0
        self.curvature = 0
        self.scan_thickness = 0
        self._ui = None

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        from PySide6.QtWidgets import QFrame
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#4CAF50"  # green accent
        panel.setStyleSheet(
            f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        params = [
            ("Intensity", "intensity", 0, 100, 0),
            ("Curvature", "curvature", 0, 100, 0),
            ("Scan", "scan_thickness", 0, 20, 0),
        ]
        
        # Calculate and set fixed height based on content
        rows_needed = (len(params) + 2) // 3  # 3 knobs per row, round up
        panel_height = rows_needed * 60 + 20  # 20px for margins
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self,
                attr,
                disp,
                min_a,
                max_a,
                current,
                panel,
                default_actual=neutral,
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
        if self.intensity <= 0 and self.curvature <= 0 and self.scan_thickness <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy().astype(np.float32)
        
        intensity = self.intensity / 100.0
        curvature = self.curvature / 100.0
        scan_thickness = max(1, int(self.scan_thickness))
        
        # Apply scan lines (vectorized)
        if scan_thickness > 0:
            # Create scan line mask
            scan_mask = np.ones((h, w, 1), dtype=np.float32)
            for y in range(0, h, scan_thickness * 2):
                y_end = min(y + scan_thickness, h)
                # Make scan lines more visible - darker lines
                scan_mask[y:y_end, :, :] = 0.3 + (0.7 * (1.0 - intensity))
            # Apply mask to all channels
            result *= scan_mask
        
        # Apply curvature distortion (vectorized)
        if curvature > 0:
            # Create coordinate grids
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            center_x, center_y = w // 2, h // 2
            
            # Calculate distances from center
            dx = x_coords - center_x
            dy = y_coords - center_y
            dist = np.sqrt(dx**2 + dy**2)
            
            # Avoid division by zero
            dist = np.maximum(dist, 1)
            
            # Calculate distortion
            max_dist = math.sqrt(center_x**2 + center_y**2)
            distortion = (dist / max_dist) ** 2 * curvature * 0.5
            
            # Calculate new coordinates
            new_x = np.clip(x_coords + dx * distortion, 0, w - 1).astype(int)
            new_y = np.clip(y_coords + dy * distortion, 0, h - 1).astype(int)
            
            # Apply distortion using advanced indexing
            distorted = src_bgr[new_y, new_x, :].astype(np.float32)
            
            # Blend with current result based on intensity
            if intensity > 0:
                result = cv2.addWeighted(result, 1.0 - intensity, distorted, intensity, 0.0)
            else:
                result = distorted
        
        # Apply overall intensity scaling if no other effects are active
        if intensity > 0 and curvature <= 0 and scan_thickness <= 0:
            # Just apply a subtle overall effect
            result = cv2.addWeighted(result, 1.0 - intensity * 0.3, src_bgr.astype(np.float32), intensity * 0.3, 0.0)
        
        return clamp_img(result)


# ---------------------------------------------------------------------
# Pixel Smash effect
# ---------------------------------------------------------------------

class PixelSmashFilter(BaseFilter):
    """
    Randomly "smashes" pixels by moving them to nearby positions.
    Creates digital distortion and glitch artifacts.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: smash radius, frequency, intensity
        self.smash_radius = 0   # [0, 20] maximum pixel displacement
        self.frequency = 0      # [0, 100] how many pixels get smashed
        self.intensity = 0      # [0, 100] strength of the effect

    @property
    def name(self) -> str:
        return "PIXEL SMASH"

    def reset_params(self) -> None:
        self.smash_radius = 0
        self.frequency = 0
        self.intensity = 0
        self._ui = None

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        from PySide6.QtWidgets import QFrame
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#FF9800"  # orange accent
        panel.setStyleSheet(
            f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        params = [
            ("Radius", "smash_radius", 0, 20, 0),
            ("Freq", "frequency", 0, 100, 0),
            ("Intensity", "intensity", 0, 100, 0),
        ]
        
        # Calculate and set fixed height based on content
        rows_needed = (len(params) + 2) // 3  # 3 knobs per row, round up
        panel_height = rows_needed * 60 + 20  # 20px for margins
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self,
                attr,
                disp,
                min_a,
                max_a,
                current,
                panel,
                default_actual=neutral,
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
        if self.smash_radius <= 0 or self.frequency <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy()
        
        radius = int(self.smash_radius)
        frequency = self.frequency / 100.0
        intensity = self.intensity / 100.0
        
        # Create random displacement map (vectorized)
        # Generate random mask for pixels to smash
        smash_mask = np.random.random((h, w)) < frequency
        
        if np.any(smash_mask):
            # Generate random displacements for all pixels at once
            dx = np.random.randint(-radius, radius + 1, (h, w))
            dy = np.random.randint(-radius, radius + 1, (h, w))
            
            # Calculate source positions
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            src_x = np.clip(x_coords - dx, 0, w - 1)
            src_y = np.clip(y_coords - dy, 0, h - 1)
            
            # Only apply to pixels that are marked for smashing
            smash_indices = np.where(smash_mask)
            
            # Get source pixels for smashed positions
            src_pixels = src_bgr[src_y[smash_indices], src_x[smash_indices], :]
            orig_pixels = result[smash_indices]
            
            # Blend original and smashed pixels
            blended = (orig_pixels.astype(np.float32) * (1.0 - intensity) + 
                      src_pixels.astype(np.float32) * intensity)
            
            # Update result
            result[smash_indices] = blended.astype(np.uint8)
        
        return result


# ---------------------------------------------------------------------
# Glitch Blocks effect
# ---------------------------------------------------------------------

class GlitchBlocksFilter(BaseFilter):
    """
    Replaces random rectangular areas with glitched versions.
    Creates "corrupted data block" effects.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: block size, frequency, glitch intensity
        self.block_size = 0     # [0, 100] size of glitch blocks
        self.frequency = 0      # [0, 100] how many blocks appear
        self.glitch_intensity = 0 # [0, 100] strength of glitch effect

    @property
    def name(self) -> str:
        return "GLITCH BLOCKS"

    def reset_params(self) -> None:
        self.block_size = 0
        self.frequency = 0
        self.glitch_intensity = 0
        self._ui = None

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        from PySide6.QtWidgets import QFrame
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#9C27B0"  # purple accent
        panel.setStyleSheet(
            f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        params = [
            ("Size", "block_size", 0, 100, 0),
            ("Freq", "frequency", 0, 100, 0),
            ("Glitch", "glitch_intensity", 0, 100, 0),
        ]
        
        # Calculate and set fixed height based on content
        rows_needed = (len(params) + 2) // 3  # 3 knobs per row, round up
        panel_height = rows_needed * 60 + 20  # 20px for margins
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self,
                attr,
                disp,
                min_a,
                max_a,
                current,
                panel,
                default_actual=neutral,
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
        if self.block_size <= 0 or self.frequency <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy()
        
        # Calculate block dimensions
        max_block_size = min(h, w) // 4  # Maximum 1/4 of image dimensions
        block_dim = max(10, int(self.block_size * max_block_size / 100))
        frequency = self.frequency / 100.0
        glitch_strength = self.glitch_intensity / 100.0
        
        # Calculate number of blocks to create
        num_blocks = int(frequency * 20)  # Up to 20 blocks at max frequency
        
        for _ in range(num_blocks):
            # Random block position
            x = np.random.randint(0, w - block_dim)
            y = np.random.randint(0, h - block_dim)
            
            # Random glitch effect
            effect_type = np.random.randint(0, 4)
            
            if effect_type == 0:  # Color shift
                # Shift color channels randomly
                shift_x = np.random.randint(-block_dim//4, block_dim//4)
                shift_y = np.random.randint(-block_dim//4, block_dim//4)
                
                for cy in range(block_dim):
                    for cx in range(block_dim):
                        src_y = min(h-1, max(0, y + cy + shift_y))
                        src_x = min(w-1, max(0, x + cx + shift_x))
                        result[y+cy, x+cx, :] = src_bgr[src_y, src_x, :]
                        
            elif effect_type == 1:  # Invert colors
                result[y:y+block_dim, x:x+block_dim, :] = 255 - result[y:y+block_dim, x:x+block_dim, :]
                
            elif effect_type == 2:  # Noise injection
                noise = np.random.randint(0, 256, (block_dim, block_dim, c))
                result[y:y+block_dim, x:x+block_dim, :] = noise
                
            elif effect_type == 3:  # Pixel scramble
                # Randomly rearrange pixels within the block
                block_pixels = result[y:y+block_dim, x:x+block_dim, :].reshape(-1, c)
                np.random.shuffle(block_pixels)
                result[y:y+block_dim, x:x+block_dim, :] = block_pixels.reshape(block_dim, block_dim, c)
            
            # Blend with original based on glitch intensity
            if glitch_strength < 1.0:
                original_block = src_bgr[y:y+block_dim, x:x+block_dim, :]
                glitched_block = result[y:y+block_dim, x:x+block_dim, :]
                blended = cv2.addWeighted(original_block, 1.0 - glitch_strength, glitched_block, glitch_strength, 0.0)
                result[y:y+block_dim, x:x+block_dim, :] = blended
        
        return result


# ---------------------------------------------------------------------
# ASCII Art effect
# ---------------------------------------------------------------------

class ASCIIArtFilter(BaseFilter):
    """
    Convert the image to ASCII art characters.
    Creates a retro terminal/typewriter aesthetic.
    """

    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: character size, contrast
        self.char_size = 8      # [4, 20] size of ASCII characters
        self.contrast = 0       # [0, 100] contrast adjustment

    @property
    def name(self) -> str:
        return "ASCII ART"

    def reset_params(self) -> None:
        self.char_size = 8
        self.contrast = 0
        self._ui = None

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        from PySide6.QtWidgets import QFrame
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#00BCD4"  # cyan accent
        panel.setStyleSheet(
            f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        params = [
            ("Size", "char_size", 4, 20, 8),
            ("Contrast", "contrast", 0, 100, 0),
        ]
        
        # Calculate and set fixed height based on content
        rows_needed = (len(params) + 2) // 3  # 3 knobs per row, round up
        panel_height = rows_needed * 60 + 20  # 20px for margins
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self,
                attr,
                disp,
                min_a,
                max_a,
                current,
                panel,
                default_actual=neutral,
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
            
        h, w, c = src_bgr.shape
        
        # Convert to grayscale
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply contrast adjustment
        if self.contrast > 0:
            contrast_factor = 1.0 + (self.contrast / 100.0) * 2.0
            gray = cv2.convertScaleAbs(gray, alpha=contrast_factor, beta=0)
        
        # Character size from parameter
        char_size = max(4, min(20, int(self.char_size)))
        
        # Downsample image to character grid
        grid_h = max(1, h // char_size)
        grid_w = max(1, w // char_size)
        
        # Resize to character grid
        small_gray = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        
        # Extended ASCII character set (from dark to light)
        ascii_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        
        # Create ASCII art mapping
        ascii_art = []
        for row in small_gray:
            ascii_row = []
            for pixel in row:
                # Map brightness to ASCII character
                char_idx = int((pixel / 255.0) * (len(ascii_chars) - 1))
                char_idx = max(0, min(len(ascii_chars) - 1, char_idx))
                ascii_row.append(ascii_chars[char_idx])
            ascii_art.append(ascii_row)
        
        # Create output image with black background
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Draw actual ASCII characters using OpenCV text rendering
        for y, ascii_row in enumerate(ascii_art):
            for x, char in enumerate(ascii_row):
                if char != ' ':
                    # Calculate position
                    pos_x = x * char_size
                    pos_y = y * char_size + char_size  # Adjust for text baseline
                    
                    # Calculate color based on character darkness
                    char_idx = ascii_chars.find(char)
                    brightness = char_idx / (len(ascii_chars) - 1)
                    color = int(brightness * 255)
                    
                    # Calculate font scale based on character size
                    font_scale = char_size / 20.0  # Scale font relative to character size
                    
                    # Draw the actual character using OpenCV putText
                    cv2.putText(result, char, (pos_x, pos_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (color, color, color), 1)
        
        return result


# ---------------------------------------------------------------------
# Image view widget
# ---------------------------------------------------------------------

class ImageView(QScrollArea):
    """
    A scrollable area that displays the current image scaled to fit the viewport.
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.label = QLabel()
        self.label.setBackgroundRole(QPalette.Base)
        self.label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.label.setScaledContents(False)
        self.setWidget(self.label)
        self.setWidgetResizable(True)
        self._pixmap: Optional[QPixmap] = None

    def set_image(self, img_bgr: Optional[np.ndarray]) -> None:
        if img_bgr is None:
            self._pixmap = None
            self.label.clear()
            return
        pix = QPixmap.fromImage(np_to_qimage(img_bgr))
        self._pixmap = pix
        self._update_scaled()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._update_scaled()

    def _update_scaled(self) -> None:
        if not self._pixmap:
            return
        viewport = self.viewport().size()
        scaled = self._pixmap.scaled(
            viewport, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.label.setPixmap(scaled)
        self.label.resize(scaled.size())


# ---------------------------------------------------------------------
# Main application window
# ---------------------------------------------------------------------

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Corrupto")
        self.resize(1280, 800)
        self._image_loaded: Optional[np.ndarray] = None
        self._image_committed: Optional[np.ndarray] = None
        self._history: List[np.ndarray] = []
        self._connected_filters: List[BaseFilter] = []
        # Track previously selected filters to avoid resetting parameters unnecessarily
        self._last_selected_filters: List[BaseFilter] = []

        # Setup UI and theme
        self._setup_dark_mode()
        self._init_ui()

    # -----------------------------------------------------------------
    # Theme configuration
    # -----------------------------------------------------------------

    def _setup_dark_mode(self) -> None:
        """
        Apply a dark theme inspired by the Phonodex project and load the pixel font.
        """
        QApplication.setStyle(QStyleFactory.create("Fusion"))

        # Load pixel font globally if available
        try:
            from PySide6.QtGui import QFontDatabase, QFont

            for path in PIXEL_FONT_PATHS:
                if os.path.exists(path):
                    QFontDatabase.addApplicationFont(path)
                    break
            app_font = QFont(PIXEL_FONT_FAMILY)
            app_font.setPointSize(10)
            QApplication.setFont(app_font)
        except Exception:
            pass

        palette = QPalette()
        palette.setColor(QPalette.Window, QColor(COLORS["BACKGROUND"]))
        palette.setColor(QPalette.WindowText, QColor(COLORS["TEXT"]))
        palette.setColor(QPalette.Base, QColor(COLORS["SECONDARY_BACKGROUND"]))
        palette.setColor(QPalette.AlternateBase, QColor(COLORS["BACKGROUND"]))
        palette.setColor(QPalette.ToolTipBase, QColor(COLORS["BACKGROUND"]))
        palette.setColor(QPalette.ToolTipText, QColor(COLORS["TEXT"]))
        palette.setColor(QPalette.Text, QColor(COLORS["TEXT"]))
        palette.setColor(QPalette.Button, QColor(COLORS["SECONDARY_BACKGROUND"]))
        palette.setColor(QPalette.ButtonText, QColor(COLORS["TEXT"]))
        palette.setColor(QPalette.BrightText, QColor(COLORS["ERROR"]))
        palette.setColor(QPalette.Highlight, QColor(COLORS["SUCCESS"]))
        palette.setColor(QPalette.HighlightedText, QColor("#000000"))
        self.setPalette(palette)

        # Global stylesheet for widgets
        self.setStyleSheet(
            f"""
            QWidget {{ color: {COLORS['TEXT']}; background-color: {COLORS['BACKGROUND']}; }}
            QGroupBox {{
                border: 1px solid #333333;
                border-radius: 6px;
                margin-top: 8px;
                padding: 6px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
                color: {COLORS['SECONDARY_TEXT']};
            }}
            QListWidget {{
                background: {COLORS['SECONDARY_BACKGROUND']};
                border: 1px solid #333333;
            }}
            QListWidget::item:selected {{
                background: {COLORS['SUCCESS']};
                color: #000000;
            }}
            QPushButton {{
                background-color: {COLORS['SECONDARY_BACKGROUND']};
                border: 1px solid #3a3a3a;
                border-radius: 6px;
                padding: 6px 10px;
            }}
            QPushButton:hover {{
                border-color: #5a5a5a;
            }}
            QPushButton:pressed {{
                background-color: #1b1b1b;
            }}
            QSlider::groove:horizontal {{
                height: 6px;
                background: {COLORS['SECONDARY_BACKGROUND']};
                border: 1px solid #333333;
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                width: 14px;
                height: 14px;
                margin: -5px 0;
                border-radius: 7px;
                background: {COLORS['SUCCESS']};
            }}
            QScrollArea {{
                background: {COLORS['SECONDARY_BACKGROUND']};
                border: none;
            }}
            QScrollBar:vertical {{
                background: {COLORS['SECONDARY_BACKGROUND']};
                width: 12px;
                border-radius: 6px;
            }}
            QScrollBar::handle:vertical {{
                background: {COLORS['SUCCESS']};
                border-radius: 6px;
                min-height: 20px;
            }}
            QScrollBar::handle:vertical:hover {{
                background: #66bb6a;
            }}
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
                height: 0px;
            }}
            QLabel {{
                background: transparent;
            }}
            /* Effect panel styling for VST-like look */
            QFrame#effectPanel {{
                background: {COLORS['SECONDARY_BACKGROUND']};
                border: 2px solid white;
                border-radius: 6px;
                padding: 4px;
            }}
            /* Alternative styling for effect panels */
            QFrame[objectName="effectPanel"] {{
                background: {COLORS['SECONDARY_BACKGROUND']};
                border: 2px solid white;
                border-radius: 6px;
                padding: 4px;
            }}
            /* Knob styling: the dial is circular with accent border.  The
               size is enforced in code via setFixedSize on each dial,
               so we do not specify min-width or min-height here. */
            KnobControl QDial {{
                background-color: #333333;
                border: 2px solid {COLORS['SUCCESS']};
                border-radius: 20px;
            }}
            KnobControl QLabel {{
                color: {COLORS['SECONDARY_TEXT']};
                font-size: 9px;
            }}
            """
        )

    # -----------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------

    def _init_ui(self) -> None:
        """
        Construct the main interface: left panel for file and filter selection,
        right panel for filter settings and image preview.
        """
        splitter = QSplitter(Qt.Horizontal, self)
        # Expose splitter on self so we can reapply the sizing ratio later if needed
        self.splitter = splitter
        self.setCentralWidget(splitter)

        # Left panel: file buttons and filter list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        self.btn_open = QPushButton("OPEN IMAGE…")
        self.btn_open.clicked.connect(self.open_image_dialog)
        left_layout.addWidget(self.btn_open)

        self.btn_save = QPushButton("SAVE IMAGE…")
        self.btn_save.clicked.connect(self.save_image_dialog)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)

        # Apply button placed directly under Save
        self.btn_apply = QPushButton("APPLY")
        self.btn_apply.clicked.connect(self.apply_current_preview)
        # Apply button uses the positive accent class
        self.btn_apply.setProperty("class", "positive")
        left_layout.addWidget(self.btn_apply)
        # Accent style for the Apply button
        self.btn_apply.setStyleSheet(
            f"QPushButton[class='positive'] {{ background: {COLORS['SUCCESS']}; color: #000000; border: 1px solid #2e7d32; }}"
        )

        self.filter_list = QListWidget()
        # Allow selecting none, one, or multiple filters
        self.filter_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.filter_list.itemSelectionChanged.connect(self._on_filter_selection_changed)
        left_layout.addWidget(self.filter_list, 1)

        # Right panel: settings and image preview
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.setSpacing(6)

        # Container for filter settings: effect panels will stack on top of each other
        # below the filter list.  A vertical layout is used so the panels appear
        # stacked rather than side by side.
        self.settings_container = QWidget()
        self.settings_container.setStyleSheet(f"background-color: {COLORS['SECONDARY_BACKGROUND']};")
        self.settings_layout = QVBoxLayout(self.settings_container)
        self.settings_layout.setContentsMargins(0, 0, 0, 0)
        self.settings_layout.setSpacing(4)
        self.settings_layout.setAlignment(Qt.AlignTop)
        
        # Add RANDOMIZE button above the PARAMS label
        self.btn_randomize = QPushButton("RANDOMIZE")
        self.btn_randomize.clicked.connect(self.randomize_effects)
        self.btn_randomize.setProperty("class", "randomize")
        left_layout.addWidget(self.btn_randomize)
        # Style for the randomize button
        self.btn_randomize.setStyleSheet(
            f"QPushButton[class='randomize'] {{ background: {COLORS['ERROR']}; color: #000000; border: 1px solid #d32f2f; }}"
        )
        

        
        # Create a scroll area for the settings container to make it scrollable
        self.settings_scroll_area = QScrollArea()
        self.settings_scroll_area.setWidget(self.settings_container)
        self.settings_scroll_area.setWidgetResizable(True)
        self.settings_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.settings_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        # Set the same background color as the filters section
        self.settings_scroll_area.setStyleSheet(f"background-color: {COLORS['SECONDARY_BACKGROUND']}; border: none;")
        
        # Set fixed height for the scroll area (50% of window height)
        # We'll update this in resizeEvent
        self.settings_scroll_area.setFixedHeight(400)  # Default height, will be updated
        
        # Add the scroll area to the left panel beneath the filter list
        left_layout.addWidget(self.settings_scroll_area, 0)

        # Image preview area
        self.image_view = ImageView()
        right_layout.addWidget(self.image_view, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        # left panel proportion
        splitter.setSizes([int(self.width() * 0.15), int(self.width() * 0.85)])
        # Maintain a 1:9 stretch ratio so that adding or removing effects does not
        # change the relative width of the panels unless the user adjusts it
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 9)

        # Register filters
        self.filters: List[BaseFilter] = []
        self._register_filters()

        # Keyboard shortcuts
        self._create_actions()
        
        # Set initial height for settings scroll area
        self._update_settings_height()

    def _create_actions(self) -> None:
        """
        Register keyboard shortcuts for common actions.  Undo and reset
        actions are deliberately omitted since those controls were removed
        from the UI at the user's request.
        """
        open_act = QAction("Open…", self)
        open_act.setShortcut(QKeySequence.Open)
        open_act.triggered.connect(self.open_image_dialog)

        save_act = QAction("Save…", self)
        save_act.setShortcut(QKeySequence.Save)
        save_act.triggered.connect(self.save_image_dialog)

        # Apply action via Ctrl+Enter
        apply_act = QAction("Apply", self)
        apply_act.setShortcut(QKeySequence(Qt.CTRL | Qt.Key_Return))
        apply_act.triggered.connect(self.apply_current_preview)

        # Register only the actions we still support
        self.addAction(open_act)
        self.addAction(save_act)
        self.addAction(apply_act)
    
    def resizeEvent(self, event) -> None:
        """
        Handle window resize events to update the settings scroll area height.
        """
        super().resizeEvent(event)
        self._update_settings_height()
    
    def _update_settings_height(self) -> None:
        """
        Update the settings scroll area height to be 50% of the window height.
        """
        if hasattr(self, 'settings_scroll_area'):
            height = int(self.height() * 0.5)
            self.settings_scroll_area.setFixedHeight(height)

    def _register_filters(self) -> None:
        """
        Instantiate and add filters to the list widget.  New filters can be
        registered here by appending instances to self.filters.
        """
        def add_filter(f: BaseFilter) -> None:
            self.filters.append(f)
            item = QListWidgetItem(f.name)
            self.filter_list.addItem(item)

        add_filter(BlackAndWhiteFilter())
        # Glitch effects
        add_filter(RGBShiftFilter())
        add_filter(WaveDistortionFilter())
        add_filter(BlockifyFilter())
        add_filter(PixelSortFilter())
        add_filter(NoiseInjectionFilter())
        add_filter(SliceShiftFilter())
        add_filter(CRTScanFilter())
        add_filter(PixelSmashFilter())
        add_filter(GlitchBlocksFilter())
        add_filter(ASCIIArtFilter())
        add_filter(MeshFilter())

    # -----------------------------------------------------------------
    # File operations
    # -----------------------------------------------------------------

    def open_image_dialog(self) -> None:
        """
        Prompt the user to select an image file and load it into the editor.
        """
        file_types = (
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All Files (*)"
        )
        path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", file_types)
        if not path:
            return
        # Use cv2.imdecode with fromfile to handle unicode paths on Windows
        try:
            img_array = np.fromfile(path, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception:
            img = None
        if img is None:
            QMessageBox.warning(self, "Open Image", "Failed to open image.")
            return
        self._image_loaded = img
        self._image_committed = img.copy()
        self._history.clear()
        # Enable save button now that an image is loaded
        self.btn_save.setEnabled(True)
        self._update_preview()

    def save_image_dialog(self) -> None:
        """
        Prompt the user to choose a filename and save the current committed image.
        Supports PNG, JPEG, BMP, WebP, and TIFF formats.
        """
        if self._image_committed is None:
            QMessageBox.information(self, "Save", "No image to save yet.")
            return
        filters = (
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp);;WebP (*.webp);;TIFF (*.tif *.tiff)"
        )
        out_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Save Image", "output.png", filters
        )
        if not out_path:
            return

        ext = os.path.splitext(out_path)[1].lower()
        # If no extension or unsupported, infer from selected filter
        if ext not in [".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"]:
            if "PNG" in selected_filter:
                ext = ".png"
            elif "JPEG" in selected_filter:
                ext = ".jpg"
            elif "BMP" in selected_filter:
                ext = ".bmp"
            elif "WebP" in selected_filter:
                ext = ".webp"
            elif "TIFF" in selected_filter:
                ext = ".tif"
            out_path = out_path + ext

        img = self._image_committed
        params = []
        if ext in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        success, buf = cv2.imencode(ext, img, params)
        if not success:
            QMessageBox.warning(self, "Save", "Failed to encode image.")
            return
        try:
            buf.tofile(out_path)
        except Exception as e:
            QMessageBox.warning(self, "Save", f"Failed to save image: {e}")
            return

        QMessageBox.information(self, "Save", f"Saved to:\n{out_path}")

    # -----------------------------------------------------------------
    # Editing workflow
    # -----------------------------------------------------------------

    def _on_filter_selection_changed(self) -> None:
        """
        Respond to changes in the filter selection.  Supports zero, one, or
        multiple selected filters.  Resets parameters on selection and rebuilds
        the UI for each selected effect.  Manages signal connections to avoid
        duplicate connections and runtime warnings.
        """
        # Determine newly selected filters in the order they appear in the list
        selected_filters: List[BaseFilter] = []
        for index in self.filter_list.selectedIndexes():
            row = index.row()
            if 0 <= row < len(self.filters):
                selected_filters.append(self.filters[row])



        # Identify newly selected filters and deselected filters compared to the previous selection
        previously_selected = list(self._last_selected_filters)
        newly_selected = [f for f in selected_filters if f not in previously_selected]
        deselected = [f for f in previously_selected if f not in selected_filters]

        # Disconnect and remove any filters that are no longer selected
        for f in deselected:
            if f in self._connected_filters:
                try:
                    f.params_changed.disconnect(self._update_preview)
                except Exception:
                    pass
                self._connected_filters.remove(f)

        # Clear the settings container UI
        while self.settings_layout.count():
            item = self.settings_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.setParent(None)

        # Build UI panels for each selected filter in order
        for f in selected_filters:
            # If this filter has just been selected (not previously selected), reset its params
            if f in newly_selected:
                f.reset_params()
            # Ensure the params_changed signal is connected once
            if f not in self._connected_filters:
                try:
                    f.params_changed.connect(self._update_preview)
                except Exception:
                    pass
                self._connected_filters.append(f)
            # Create UI and add to layout
            ui = f.build_ui(self.settings_container)
            self.settings_layout.addWidget(ui)

        # Store current selection for next comparison
        self._last_selected_filters = selected_filters
        # Update preview to reflect the new selection
        self._update_preview()

    def _update_preview(self) -> None:
        """
        Generate and display a preview image.  If no filters are selected,
        display the committed image.  Otherwise, apply each selected filter
        sequentially to the committed image.
        """
        if self._image_committed is None:
            self.image_view.set_image(None)
            return

        # Determine current selected filters (in list order)
        selected_filters: List[BaseFilter] = []
        for index in self.filter_list.selectedIndexes():
            row = index.row()
            if 0 <= row < len(self.filters):
                selected_filters.append(self.filters[row])

        if not selected_filters:
            # No filter selected; show committed image
            self.image_view.set_image(self._image_committed)
            return

        # Start with the committed image
        preview = self._image_committed.copy()
        # Apply each filter in order
        for f in selected_filters:
            try:
                preview = f.apply(preview)
            except Exception:
                # If a filter fails, skip it
                pass

        self.image_view.set_image(preview)

    def randomize_effects(self) -> None:
        """
        Apply 5 random effects with random parameters to the current image.
        Each click will generate a new random combination.
        """
        if self._image_committed is None:
            return
            
        # Clear current filter selection
        self.filter_list.clearSelection()
        
        # Randomly select 5 different filters
        available_filters = self.filters.copy()
        selected_filters = []
        
        # Ensure we don't select the same filter twice
        for _ in range(min(5, len(available_filters))):
            if available_filters:
                # Pick a random filter
                filter_idx = np.random.randint(0, len(available_filters))
                selected_filter = available_filters.pop(filter_idx)
                selected_filters.append(selected_filter)
        
        # Select the filters in the UI
        for filter_obj in selected_filters:
            # Find the item in the list widget
            for i in range(self.filter_list.count()):
                if self.filter_list.item(i).text() == filter_obj.name:
                    self.filter_list.item(i).setSelected(True)
                    break
        
        # Randomize parameters for each selected filter
        for filter_obj in selected_filters:
            # Reset to default first
            filter_obj.reset_params()
            
            # Get parameter ranges from the filter's build_ui method
            # We need to temporarily build the UI to get the parameter definitions
            temp_parent = QWidget()
            temp_ui = filter_obj.build_ui(temp_parent)
            
            # Extract parameter definitions from the UI building process
            # This is a bit hacky but works with the current architecture
            param_ranges = {}
            
            # For each filter type, define the parameter ranges manually
            if isinstance(filter_obj, NoiseInjectionFilter):
                param_ranges = {
                    "intensity": (0, 100),
                    "noise_type": (0, 2),
                    "seed": (0, 100)
                }
            elif isinstance(filter_obj, SliceShiftFilter):
                param_ranges = {
                    "frequency": (0, 100),
                    "strip_height": (0, 50),
                    "shift_amount": (0, 100)
                }
            elif isinstance(filter_obj, RGBShiftFilter):
                param_ranges = {
                    "shift_r": (-50, 50),
                    "shift_g": (-50, 50),
                    "shift_b": (-50, 50)
                }
            elif isinstance(filter_obj, WaveDistortionFilter):
                param_ranges = {
                    "frequency": (0, 50),
                    "amplitude": (0, 50)
                }
            elif isinstance(filter_obj, BlockifyFilter):
                param_ranges = {
                    "alpha": (0, 100),
                    "tile": (4, 64),
                    "palette": (2, 9)
                }
            elif isinstance(filter_obj, PixelSortFilter):
                param_ranges = {
                    "intensity": (0, 100),
                    "direction": (0, 1),
                    "threshold": (0, 100)
                }
            elif isinstance(filter_obj, BlackAndWhiteFilter):
                param_ranges = {
                    "strength": (0, 100),
                    "brightness": (-100, 100),
                    "contrast": (-100, 100),
                    "gamma": (10, 300),
                    "grain": (0, 100),
                    "vignette": (0, 100)
                }
            elif isinstance(filter_obj, CRTScanFilter):
                param_ranges = {
                    "intensity": (0, 100),
                    "curvature": (0, 100),
                    "scan_thickness": (0, 20)
                }
            elif isinstance(filter_obj, PixelSmashFilter):
                param_ranges = {
                    "smash_radius": (0, 20),
                    "frequency": (0, 100),
                    "intensity": (0, 100)
                }
            elif isinstance(filter_obj, GlitchBlocksFilter):
                param_ranges = {
                    "block_size": (0, 100),
                    "frequency": (0, 100),
                    "glitch_intensity": (0, 100)
                }
            elif isinstance(filter_obj, ASCIIArtFilter):
                param_ranges = {
                    "char_size": (4, 20),
                    "contrast": (0, 100)
                }
            elif isinstance(filter_obj, MeshFilter):
                param_ranges = {
                    "grid_size": (0, 50),
                    "distortion": (0, 100),
                    "intensity": (0, 100),
                    "perspective": (0, 100),
                    "wireframe": (0, 100)
                }
            
            # Randomize each parameter using its actual range
            for param_name, (min_val, max_val) in param_ranges.items():
                if hasattr(filter_obj, param_name):
                    # Generate random value within the parameter's range
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        # Integer parameter
                        random_value = np.random.randint(min_val, max_val + 1)
                    else:
                        # Float parameter
                        random_value = np.random.uniform(min_val, max_val)
                        if isinstance(min_val, int) and isinstance(max_val, int):
                            random_value = int(random_value)
                    
                    setattr(filter_obj, param_name, random_value)
            
            # Clean up temporary UI
            temp_ui.setParent(None)
            
            # Clear the cached UI so it will be rebuilt with new values
            filter_obj._ui = None
        
        # Force rebuild of the UI to show the new parameter values
        self._on_filter_selection_changed()
        
        # Update the preview to show the randomized effects
        self._update_preview()

    def apply_current_preview(self) -> None:
        """
        Commit the current preview image to the working image.  Applies all
        selected filters in sequence to the original loaded image (not the
        previously committed image) so that effects do not stack.  After
        applying, reset parameters on the selected filters to avoid
        reapplying the effect multiple times.
        """
        if self._image_committed is None:
            return

        # Determine selected filters; if none, nothing to apply
        selected_filters: List[BaseFilter] = []
        for index in self.filter_list.selectedIndexes():
            row = index.row()
            if 0 <= row < len(self.filters):
                selected_filters.append(self.filters[row])

        if not selected_filters:
            return

        # Use the original loaded image as the base, so effects are applied
        # only to the unmodified source.  If the original image is not
        # available (should not happen), fall back to the current commit.
        base_img = self._image_loaded.copy() if self._image_loaded is not None else self._image_committed.copy()

        new_img = base_img
        for f in selected_filters:
            try:
                new_img = f.apply(new_img)
            except Exception:
                pass

        # Push current committed image to history and update committed image
        # Although undo is not exposed in the UI, history is maintained for
        # potential future features.
        self._history.append(self._image_committed)
        self._image_committed = new_img

        # After committing the changes, reset parameters for all selected filters
        for f in selected_filters:
            f.reset_params()

        # Rebuild the UI for the selected filters without altering the selection
        # This will cause their knobs to reflect the neutral (default) values
        self._on_filter_selection_changed()

    def undo(self) -> None:
        """
        Undo the last committed change if available.
        """
        if not self._history:
            return
        self._image_committed = self._history.pop()
        self._update_preview()

    def reset_to_original(self) -> None:
        """
        Revert to the originally loaded image and clear history.
        """
        if self._image_loaded is None:
            return
        self._image_committed = self._image_loaded.copy()
        self._history.clear()
        self._update_preview()


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

def main() -> None:
    app = QApplication(sys.argv)
    win = MainWindow()
    # Launch the window maximized to utilize the full screen
    win.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()