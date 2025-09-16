#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image Effects Module for Corrupto

This module contains all the image processing filters and effects used by the main application.
"""

import cv2
import numpy as np
import math
from typing import Optional, List

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QImage
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QDial, QLineEdit, QFrame, QGridLayout, QSizePolicy
)

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def np_to_qimage(img_bgr: np.ndarray) -> QImage:
    """Convert a BGR image in NumPy format to a QImage in RGB888 format."""
    if img_bgr is None:
        return QImage()
    if img_bgr.ndim == 2:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_GRAY2RGB)
    else:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = img_rgb.shape
    bytes_per_line = ch * w
    return QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888).copy()

def clamp_img(img: np.ndarray) -> np.ndarray:
    """Clamp image values to [0, 255] and ensure dtype uint8."""
    return np.clip(img, 0, 255).astype(np.uint8)

# ---------------------------------------------------------------------
# Base Filter Class
# ---------------------------------------------------------------------

class BaseFilter(QWidget):
    """Abstract base class for all image filters."""
    params_changed = Signal()

    @property
    def name(self) -> str:
        raise NotImplementedError

    def build_ui(self, parent: QWidget) -> QWidget:
        raise NotImplementedError

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def reset_params(self) -> None:
        pass

# ---------------------------------------------------------------------
# Knob Control Widget
# ---------------------------------------------------------------------

class KnobControl(QWidget):
    """Custom knob control for filter parameters."""
    
    def __init__(
        self,
        filter_obj: BaseFilter,
        property_name: str,
        display_name: str,
        min_actual: int,
        max_actual: int,
        current_actual: int,
        parent: Optional[QWidget] = None,
        default_actual: Optional[int] = None,
    ) -> None:
        super().__init__(parent)
        self.filter_obj = filter_obj
        self.property_name = property_name
        self.min_actual = min_actual
        self.max_actual = max_actual
        self.default_actual = default_actual if default_actual is not None else min_actual
        self.offset = -min_actual
        dial_range = max_actual - min_actual

        # Layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)

        # Title label
        self.name_label = QLabel(display_name.upper())
        self.name_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.name_label)

        # Value edit and dial
        value_dial_layout = QHBoxLayout()
        value_dial_layout.setContentsMargins(0, 0, 0, 0)
        value_dial_layout.setSpacing(2)

        self.value_edit = QLineEdit(str(current_actual))
        self.value_edit.setAlignment(Qt.AlignCenter)
        self.value_edit.setFrame(False)
        self.value_edit.setFixedWidth(45)
        self.value_edit.setStyleSheet("background: transparent; border: none;")
        value_dial_layout.addWidget(self.value_edit)

        self.dial = QDial()
        value_dial_layout.addWidget(self.dial, alignment=Qt.AlignCenter)
        layout.addLayout(value_dial_layout)
        
        # Configure dial
        self.dial.setNotchesVisible(True)
        self.dial.setMinimum(0)
        self.dial.setMaximum(dial_range)
        self.dial.setValue(current_actual + self.offset)
        self.dial.setToolTip(f"{display_name.upper()}: {min_actual}..{max_actual}")
        self.dial.setFixedSize(30, 30)

        # State tracking
        self._press_value = current_actual + self.offset
        self._dragged = False

        # Connect signals
        self.dial.valueChanged.connect(self._on_dial_change)
        self.dial.installEventFilter(self)
        self.value_edit.editingFinished.connect(self._on_edit_finished)

    def _on_dial_change(self, val: int) -> None:
        actual_value = val - self.offset
        if actual_value < self.min_actual:
            actual_value = self.min_actual
        if actual_value > self.max_actual:
            actual_value = self.max_actual
        self.value_edit.setText(str(actual_value))
        if self._dragged:
            setattr(self.filter_obj, self.property_name, actual_value)
            self.filter_obj.params_changed.emit()

    def eventFilter(self, obj, event):
        from PySide6.QtCore import QEvent
        if obj is self.dial:
            if event.type() == QEvent.MouseButtonPress:
                if event.button() == Qt.MouseButton.RightButton:
                    self._dragged = True
                    default_val = self.default_actual
                    try:
                        self.dial.blockSignals(True)
                        self.dial.setValue(default_val + self.offset)
                    finally:
                        self.dial.blockSignals(False)
                    self.value_edit.setText(str(default_val))
                    setattr(self.filter_obj, self.property_name, default_val)
                    self.filter_obj.params_changed.emit()
                    return True
                elif event.button() == Qt.MouseButton.LeftButton:
                    self._press_value = self.dial.value()
                    self._dragged = False
            elif event.type() == QEvent.MouseMove:
                if event.buttons() & Qt.MouseButton.LeftButton:
                    self._dragged = True
            elif event.type() == QEvent.MouseButtonRelease:
                if event.button() == Qt.MouseButton.RightButton:
                    return True
                if event.button() == Qt.MouseButton.LeftButton:
                    if not self._dragged:
                        try:
                            self.dial.blockSignals(True)
                            self.dial.setValue(self._press_value)
                        finally:
                            self.dial.blockSignals(False)
                        original_actual = self._press_value - self.offset
                        self.value_edit.setText(str(original_actual))
                        return True
                    else:
                        actual_value = self.dial.value() - self.offset
                        self.value_edit.setText(str(actual_value))
                        setattr(self.filter_obj, self.property_name, actual_value)
                        self.filter_obj.params_changed.emit()
                        return True
        return super().eventFilter(obj, event)

    def _on_edit_finished(self) -> None:
        text = self.value_edit.text().strip()
        try:
            val = int(text)
        except ValueError:
            val = self.default_actual
        if val < self.min_actual:
            val = self.min_actual
        if val > self.max_actual:
            val = self.max_actual
        self.value_edit.setText(str(val))
        self.dial.blockSignals(True)
        self.dial.setValue(val + self.offset)
        self.dial.blockSignals(False)
        setattr(self.filter_obj, self.property_name, val)
        self.filter_obj.params_changed.emit()

# ---------------------------------------------------------------------
# Filter Implementations
# ---------------------------------------------------------------------

class BlackAndWhiteFilter(BaseFilter):
    """Black & white filter with adjustable parameters."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.brightness = 0
        self.contrast = 0
        self.gamma = 100
        self.grain = 0
        self.vignette = 0
        self.strength = 0

    @property
    def name(self) -> str:
        return "B&W"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui

        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        panel.setStyleSheet("border: 1px solid white; border-radius: 6px; padding: 4px;")
        grid = QGridLayout(panel)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)

        params = [
            ("Strength", "strength", 0, 100, 0),
            ("Brightness", "brightness", -100, 100, 0),
            ("Contrast", "contrast", -100, 100, 0),
            ("Gamma", "gamma", 10, 300, 100),
            ("Grain", "grain", 0, 100, 0),
            ("Vignette", "vignette", 0, 100, 0),
        ]
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current_val = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current_val, panel, default_actual=neutral
            )
            row = idx // 3
            col = idx % 3
            grid.addWidget(knob, row, col, alignment=Qt.AlignCenter)

        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr

        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        
        # Apply brightness/contrast
        alpha = max(0.0, 1.0 + (self.contrast / 100.0) * 1.5)
        beta = float(self.brightness)
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        
        # Apply gamma
        gamma = max(0.01, self.gamma / 100.0)
        inv = 1.0 / gamma
        lut = np.array([((i / 255.0) ** inv) * 255 for i in range(256)]).astype("uint8")
        gray = cv2.LUT(gray, lut)
        
        # Apply grain
        if self.grain > 0:
            sigma = self.grain * 0.6
            noise = np.random.normal(0, sigma, gray.shape).astype(np.float32)
            gray = gray.astype(np.float32) + noise
            gray = clamp_img(gray)
        
        # Apply vignette
        if self.vignette > 0:
            h, w = gray.shape[:2]
            kernel_x = cv2.getGaussianKernel(w, w * 0.5)
            kernel_y = cv2.getGaussianKernel(h, h * 0.5)
            mask = kernel_y * kernel_x.T
            mask = mask / mask.max()
            strength = self.vignette / 100.0
            vignette_mask = (1.0 - strength) + (strength * mask)
            gray = gray.astype(np.float32) * vignette_mask
            gray = clamp_img(gray)
        
        # Convert back to BGR and blend
        bw_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        t = self.strength / 100.0
        out = cv2.addWeighted(bw_bgr, t, src_bgr, 1.0 - t, 0.0)
        return out

    def reset_params(self) -> None:
        self.brightness = 0
        self.contrast = 0
        self.gamma = 100
        self.grain = 0
        self.vignette = 0
        self.strength = 0
        self._ui = None

class RGBShiftFilter(BaseFilter):
    """Shift RGB channels independently."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.shift_r = 0
        self.shift_g = 0
        self.shift_b = 0

    @property
    def name(self) -> str:
        return "RGB_SHFT"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#F44336"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Shift R", "shift_r", -50, 50, 0),
            ("Shift G", "shift_g", -50, 50, 0),
            ("Shift B", "shift_b", -50, 50, 0),
        ]
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
            )
            knob.dial.setStyleSheet(
                f"QDial {{ background-color: #333333; border: 2px solid {accent}; border-radius: 20px; }}"
            )
            knob.name_label.setStyleSheet(f"color: {accent};")
            row = idx // 3
            col = idx % 3
            layout.addWidget(knob, row, col, alignment=Qt.AlignCenter)
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        self._ui = panel
        return panel

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
        if self.shift_r == 0 and self.shift_g == 0 and self.shift_b == 0:
            return src_bgr
        
        b, g, r = cv2.split(src_bgr)
        if self.shift_b != 0:
            b = np.roll(b, self.shift_b, axis=1)
        if self.shift_g != 0:
            g = np.roll(g, self.shift_g, axis=1)
        if self.shift_r != 0:
            r = np.roll(r, self.shift_r, axis=1)
        return cv2.merge([b, g, r])

    def reset_params(self) -> None:
        self.shift_r = 0
        self.shift_g = 0
        self.shift_b = 0
        self._ui = None

class WaveDistortionFilter(BaseFilter):
    """Apply sine-wave horizontal distortion across the image."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.amplitude = 0
        self.frequency = 0

    @property
    def name(self) -> str:
        return "WAVES"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#9C27B0"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Frequency", "frequency", 0, 50, 0),
            ("Amplitude", "amplitude", 0, 50, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if self.amplitude <= 0 or self.frequency <= 0:
            return src_bgr
        
        h, w, c = src_bgr.shape
        result = np.zeros_like(src_bgr)
        for y in range(h):
            offset = int(self.amplitude * math.sin(2.0 * math.pi * self.frequency * y / h))
            result[y] = np.roll(src_bgr[y], offset, axis=0)
        return result

    def reset_params(self) -> None:
        self.amplitude = 0
        self.frequency = 0
        self._ui = None

class BlockifyFilter(BaseFilter):
    """Turn image into mosaic made of colored squares."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.tile = 16
        self.palette = 9
        self.alpha = 0
        self._palette = np.array([
            [255, 255, 255], [0, 0, 0], [244, 67, 54], [255, 152, 0],
            [255, 235, 59], [76, 175, 80], [33, 150, 243], [156, 39, 176], [121, 85, 72]
        ], dtype=np.float32)

    @property
    def name(self) -> str:
        return "BLOCKS"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#FFC107"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("ALPHA", "alpha", 0, 100, 0),
            ("TILE", "tile", 4, 64, 16),
            ("PALETTE", "palette", 2, 9, 9),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, mn, mx, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, mn, mx, current, panel, default_actual=neutral
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
        
        tile = max(4, min(64, int(self.tile)))
        pal_n = max(2, min(9, int(self.palette)))
        alpha = max(0.0, min(1.0, self.alpha / 100.0))
        
        h, w = src_bgr.shape[:2]
        gh = max(1, h // tile)
        gw = max(1, w // tile)
        small = cv2.resize(src_bgr, (gw, gh), interpolation=cv2.INTER_AREA)
        avg = small.reshape(-1, 3).astype(np.float32)
        pal = self._palette[:pal_n]
        
        d = ((avg[:, None, :] - pal[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(d, axis=1)
        mapped = pal[idx].reshape(gh, gw, 3).astype(np.uint8)
        mosaic = cv2.resize(mapped, (w, h), interpolation=cv2.INTER_NEAREST)
        
        out = cv2.addWeighted(mosaic, alpha, src_bgr, 1.0 - alpha, 0.0)
        return out

    def reset_params(self) -> None:
        self.tile = 16
        self.palette = 9
        self.alpha = 0
        self._ui = None

class PixelSortFilter(BaseFilter):
    """Sort pixels in rows/columns based on brightness."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.direction = 0
        self.threshold = 0
        self.intensity = 0

    @property
    def name(self) -> str:
        return "PIXEL_SRT"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#00BCD4"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Intensity", "intensity", 0, 100, 0),
            ("Direction", "direction", 0, 1, 0),
            ("Threshold", "threshold", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if src_bgr is None or self.intensity <= 0:
            return src_bgr
            
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        result = src_bgr.copy()
        intensity = self.intensity / 100.0
        actual_threshold = int(self.threshold * 2.55)
        
        if self.direction == 0:  # Horizontal
            for y in range(h):
                row_bgr = src_bgr[y, :, :]
                row_gray = gray[y, :]
                mask = row_gray >= actual_threshold
                if not np.any(mask):
                    continue
                above_threshold = np.where(mask)[0]
                if len(above_threshold) < 2:
                    continue
                sorted_indices = above_threshold[np.argsort(row_gray[above_threshold])]
                sorted_row = row_bgr.copy()
                sorted_row[above_threshold] = row_bgr[sorted_indices]
                result[y, :, :] = cv2.addWeighted(sorted_row, intensity, row_bgr, 1.0 - intensity, 0.0)
        else:  # Vertical
            for x in range(w):
                col_bgr = src_bgr[:, x, :]
                col_gray = gray[:, x]
                mask = col_gray >= actual_threshold
                if not np.any(mask):
                    continue
                above_threshold = np.where(mask)[0]
                if len(above_threshold) < 2:
                    continue
                sorted_indices = above_threshold[np.argsort(col_gray[above_threshold])]
                sorted_col = col_bgr.copy()
                sorted_col[above_threshold] = col_bgr[sorted_indices]
                result[:, x, :] = cv2.addWeighted(sorted_col, intensity, col_bgr, 1.0 - intensity, 0.0)
        
        return result

    def reset_params(self) -> None:
        self.direction = 0
        self.threshold = 0
        self.intensity = 0
        self._ui = None

class NoiseInjectionFilter(BaseFilter):
    """Add various types of noise to create glitch artifacts."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.intensity = 0
        self.noise_type = 0
        self.seed = 0

    @property
    def name(self) -> str:
        return "NOISE_INJCT"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#FF5722"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Intensity", "intensity", 0, 100, 0),
            ("Type", "noise_type", 0, 2, 0),
            ("Seed", "seed", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if src_bgr is None or self.intensity <= 0:
            return src_bgr
            
        if self.seed > 0:
            np.random.seed(self.seed)
            
        intensity = self.intensity / 100.0
        h, w, c = src_bgr.shape
        
        if self.noise_type == 0:  # Gaussian noise
            noise = np.random.normal(0, intensity * 50, (h, w, c)).astype(np.float32)
            result = src_bgr.astype(np.float32) + noise
            return clamp_img(result)
            
        elif self.noise_type == 1:  # Salt & Pepper noise
            noise_mask = np.random.random((h, w, c)) < (intensity * 0.3)
            salt_mask = np.random.random((h, w, c)) < 0.5
            pepper_mask = ~salt_mask
            
            result = src_bgr.copy().astype(np.float32)
            result[salt_mask & noise_mask] = 255
            result[pepper_mask & noise_mask] = 0
            return clamp_img(result)
            
        else:  # Structured noise
            x_coords = np.arange(w)
            y_coords = np.arange(h)
            X, Y = np.meshgrid(x_coords, y_coords)
            
            freq_x = 0.1 + intensity * 0.2
            freq_y = 0.05 + intensity * 0.15
            noise = np.sin(X * freq_x) * np.cos(Y * freq_y) * intensity * 100
            
            noise_3d = np.stack([noise] * c, axis=2)
            result = src_bgr.astype(np.float32) + noise_3d
            return clamp_img(result)

    def reset_params(self) -> None:
        self.intensity = 0
        self.noise_type = 0
        self.seed = 0
        self._ui = None

class SliceShiftFilter(BaseFilter):
    """Cut image into horizontal strips and randomly shift them."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.strip_height = 0
        self.shift_amount = 0
        self.frequency = 0

    @property
    def name(self) -> str:
        return "SLICE_SHFT"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#E91E63"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Freq", "frequency", 0, 100, 0),
            ("Height", "strip_height", 0, 50, 0),
            ("Shift", "shift_amount", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if src_bgr is None or self.strip_height <= 0 or self.shift_amount <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy()
        
        strip_h = max(1, int(self.strip_height))
        max_shift = int(self.shift_amount * w / 100)
        frequency = self.frequency / 100.0
        
        for y_start in range(0, h, strip_h):
            y_end = min(y_start + strip_h, h)
            
            if np.random.random() < frequency:
                shift = np.random.randint(-max_shift, max_shift + 1)
                if shift != 0:
                    strip = result[y_start:y_end, :, :]
                    shifted_strip = np.roll(strip, shift, axis=1)
                    result[y_start:y_end, :, :] = shifted_strip
        
        return result

    def reset_params(self) -> None:
        self.strip_height = 0
        self.shift_amount = 0
        self.frequency = 0
        self._ui = None

class CRTScanFilter(BaseFilter):
    """Simulates old CRT monitor scan lines and curvature."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.intensity = 0
        self.curvature = 0
        self.scan_thickness = 0

    @property
    def name(self) -> str:
        return "CRT"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#4CAF50"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Intensity", "intensity", 0, 100, 0),
            ("Curvature", "curvature", 0, 100, 0),
            ("Scan", "scan_thickness", 0, 20, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if src_bgr is None or (self.intensity <= 0 and self.curvature <= 0 and self.scan_thickness <= 0):
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy().astype(np.float32)
        
        intensity = self.intensity / 100.0
        curvature = self.curvature / 100.0
        scan_thickness = max(1, int(self.scan_thickness))
        
        # Apply scan lines
        if scan_thickness > 0:
            scan_mask = np.ones((h, w, 1), dtype=np.float32)
            for y in range(0, h, scan_thickness * 2):
                y_end = min(y + scan_thickness, h)
                scan_mask[y:y_end, :, :] = 0.3 + (0.7 * (1.0 - intensity))
            result *= scan_mask
        
        # Apply curvature distortion
        if curvature > 0:
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            center_x, center_y = w // 2, h // 2
            
            dx = x_coords - center_x
            dy = y_coords - center_y
            dist = np.maximum(np.sqrt(dx**2 + dy**2), 1)
            
            max_dist = math.sqrt(center_x**2 + center_y**2)
            distortion = (dist / max_dist) ** 2 * curvature * 0.5
            
            new_x = np.clip(x_coords + dx * distortion, 0, w - 1).astype(int)
            new_y = np.clip(y_coords + dy * distortion, 0, h - 1).astype(int)
            
            distorted = src_bgr[new_y, new_x, :].astype(np.float32)
            
            if intensity > 0:
                result = cv2.addWeighted(result, 1.0 - intensity, distorted, intensity, 0.0)
            else:
                result = distorted
        
        if intensity > 0 and curvature <= 0 and scan_thickness <= 0:
            result = cv2.addWeighted(result, 1.0 - intensity * 0.3, src_bgr.astype(np.float32), intensity * 0.3, 0.0)
        
        return clamp_img(result)

    def reset_params(self) -> None:
        self.intensity = 0
        self.curvature = 0
        self.scan_thickness = 0
        self._ui = None

class PixelSmashFilter(BaseFilter):
    """Randomly 'smashes' pixels by moving them to nearby positions."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.smash_radius = 0
        self.frequency = 0
        self.intensity = 0

    @property
    def name(self) -> str:
        return "PIXEL_SMSH"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#FF9800"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Radius", "smash_radius", 0, 20, 0),
            ("Freq", "frequency", 0, 100, 0),
            ("Intensity", "intensity", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if src_bgr is None or self.smash_radius <= 0 or self.frequency <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy()
        
        radius = int(self.smash_radius)
        frequency = self.frequency / 100.0
        intensity = self.intensity / 100.0
        
        smash_mask = np.random.random((h, w)) < frequency
        
        if np.any(smash_mask):
            dx = np.random.randint(-radius, radius + 1, (h, w))
            dy = np.random.randint(-radius, radius + 1, (h, w))
            
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            src_x = np.clip(x_coords - dx, 0, w - 1)
            src_y = np.clip(y_coords - dy, 0, h - 1)
            
            smash_indices = np.where(smash_mask)
            src_pixels = src_bgr[src_y[smash_indices], src_x[smash_indices], :]
            orig_pixels = result[smash_indices]
            
            blended = (orig_pixels.astype(np.float32) * (1.0 - intensity) + 
                      src_pixels.astype(np.float32) * intensity)
            
            result[smash_indices] = blended.astype(np.uint8)
        
        return result

    def reset_params(self) -> None:
        self.smash_radius = 0
        self.frequency = 0
        self.intensity = 0
        self._ui = None

class GlitchBlocksFilter(BaseFilter):
    """Replaces random rectangular areas with glitched versions."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.block_size = 0
        self.frequency = 0
        self.glitch_intensity = 0

    @property
    def name(self) -> str:
        return "GLITCH_BLOCKS"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#9C27B0"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Size", "block_size", 0, 100, 0),
            ("Freq", "frequency", 0, 100, 0),
            ("Glitch", "glitch_intensity", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        if src_bgr is None or self.block_size <= 0 or self.frequency <= 0:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy()
        
        max_block_size = min(h, w) // 4
        block_dim = max(10, int(self.block_size * max_block_size / 100))
        frequency = self.frequency / 100.0
        glitch_strength = self.glitch_intensity / 100.0
        
        num_blocks = int(frequency * 20)
        
        for _ in range(num_blocks):
            x = np.random.randint(0, w - block_dim)
            y = np.random.randint(0, h - block_dim)
            
            effect_type = np.random.randint(0, 4)
            
            if effect_type == 0:  # Color shift
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
                block_pixels = result[y:y+block_dim, x:x+block_dim, :].reshape(-1, c)
                np.random.shuffle(block_pixels)
                result[y:y+block_dim, x:x+block_dim, :] = block_pixels.reshape(block_dim, block_dim, c)
            
            if glitch_strength < 1.0:
                original_block = src_bgr[y:y+block_dim, x:x+block_dim, :]
                glitched_block = result[y:y+block_dim, x:x+block_dim, :]
                blended = cv2.addWeighted(original_block, 1.0 - glitch_strength, glitched_block, glitch_strength, 0.0)
                result[y:y+block_dim, x:x+block_dim, :] = blended
        
        return result

    def reset_params(self) -> None:
        self.block_size = 0
        self.frequency = 0
        self.glitch_intensity = 0
        self._ui = None

class ASCIIArtFilter(BaseFilter):
    """Convert the image to ASCII art characters."""
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.char_size = 8
        self.contrast = 0

    @property
    def name(self) -> str:
        return "ASCII"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#00BCD4"
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Size", "char_size", 4, 20, 8),
            ("Contrast", "contrast", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
        gray = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2GRAY)
        
        if self.contrast > 0:
            contrast_factor = 1.0 + (self.contrast / 100.0) * 2.0
            gray = cv2.convertScaleAbs(gray, alpha=contrast_factor, beta=0)
        
        char_size = max(4, min(20, int(self.char_size)))
        grid_h = max(1, h // char_size)
        grid_w = max(1, w // char_size)
        
        small_gray = cv2.resize(gray, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        
        ascii_chars = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        
        ascii_art = []
        for row in small_gray:
            ascii_row = []
            for pixel in row:
                char_idx = int((pixel / 255.0) * (len(ascii_chars) - 1))
                char_idx = max(0, min(len(ascii_chars) - 1, char_idx))
                ascii_row.append(ascii_chars[char_idx])
            ascii_art.append(ascii_row)
        
        result = np.zeros((h, w, 3), dtype=np.uint8)
        
        for y, ascii_row in enumerate(ascii_art):
            for x, char in enumerate(ascii_row):
                if char != ' ':
                    pos_x = x * char_size
                    pos_y = y * char_size + char_size
                    
                    char_idx = ascii_chars.find(char)
                    brightness = char_idx / (len(ascii_chars) - 1)
                    color = int(brightness * 255)
                    
                    font_scale = char_size / 20.0
                    
                    cv2.putText(result, char, (pos_x, pos_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (color, color, color), 1)
        
        return result

    def reset_params(self) -> None:
        self.char_size = 8
        self.contrast = 0
        self._ui = None

class MeshFilter(BaseFilter):
    """
    Creates a 3D mesh-like distortion effect similar to Glitché's Mesh filter.
    Transforms the image into a grid-based mesh with vertex displacement and perspective distortion.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        # Parameters: grid size, distortion amount, intensity, perspective
        self.grid_size = 0      # [0, 50] size of mesh grid cells
        self.distortion = 0     # [0, 100] amount of vertex displacement
        self.intensity = 0      # [0, 100] overall effect strength
        self.perspective = 0    # [0, 100] 3D perspective depth
        self.wireframe = 0      # [0, 100] wireframe overlay strength

    @property
    def name(self) -> str:
        return "MESH"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#673AB7"  # deep purple accent
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Grid", "grid_size", 0, 50, 0),
            ("Distort", "distortion", 0, 100, 0),
            ("Intensity", "intensity", 0, 100, 0),
            ("Perspective", "perspective", 0, 100, 0),
            ("Wireframe", "wireframe", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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
            
        h, w, c = src_bgr.shape
        result = src_bgr.copy().astype(np.float32)
        
        # Convert parameters to usable values
        grid_size = max(5, int(self.grid_size * min(h, w) / 100)) if self.grid_size > 0 else 20
        distortion = self.distortion / 100.0
        intensity = self.intensity / 100.0
        perspective = self.perspective / 100.0
        wireframe = self.wireframe / 100.0
        
        # Create mesh grid
        grid_h = max(2, h // grid_size)
        grid_w = max(2, w // grid_size)
        
        # Generate mesh vertices with displacement
        vertices = []
        for gy in range(grid_h + 1):
            row = []
            for gx in range(grid_w + 1):
                # Base position
                x = gx * grid_size
                y = gy * grid_size
                
                # Add random displacement for mesh distortion
                if distortion > 0:
                    disp_x = (np.random.random() - 0.5) * distortion * grid_size * 0.5
                    disp_y = (np.random.random() - 0.5) * distortion * grid_size * 0.5
                    x += disp_x
                    y += disp_y
                
                # Add perspective distortion
                if perspective > 0:
                    center_x, center_y = w // 2, h // 2
                    dist_from_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
                    max_dist = math.sqrt(center_x**2 + center_y**2)
                    perspective_factor = 1.0 + (dist_from_center / max_dist) * perspective * 0.3
                    x = center_x + (x - center_x) * perspective_factor
                    y = center_y + (y - center_y) * perspective_factor
                
                # Clamp to image bounds
                x = max(0, min(w - 1, x))
                y = max(0, min(h - 1, y))
                
                row.append((int(x), int(y)))
            vertices.append(row)
        
        # Create mesh by sampling from original image
        mesh_result = np.zeros_like(src_bgr, dtype=np.float32)
        
        for gy in range(grid_h):
            for gx in range(grid_w):
                # Get quad vertices
                v1 = vertices[gy][gx]
                v2 = vertices[gy][gx + 1]
                v3 = vertices[gy + 1][gx]
                v4 = vertices[gy + 1][gx + 1]
                
                # Calculate quad bounds
                min_x = min(v1[0], v2[0], v3[0], v4[0])
                max_x = max(v1[0], v2[0], v3[0], v4[0])
                min_y = min(v1[1], v2[1], v3[1], v4[1])
                max_y = max(v1[1], v2[1], v3[1], v4[1])
                
                # Sample from original image for this quad
                if max_x > min_x and max_y > min_y:
                    # Get the corresponding area from original image
                    orig_x1 = gx * grid_size
                    orig_x2 = min(w, (gx + 1) * grid_size)
                    orig_y1 = gy * grid_size
                    orig_y2 = min(h, (gy + 1) * grid_size)
                    
                    if orig_x2 > orig_x1 and orig_y2 > orig_y1:
                        orig_quad = src_bgr[orig_y1:orig_y2, orig_x1:orig_x2, :]
                        if orig_quad.size > 0:
                            # Resize to fit the distorted quad
                            resized_quad = cv2.resize(orig_quad, (max_x - min_x + 1, max_y - min_y + 1))
                            
                            # Place in result
                            mesh_result[min_y:max_y + 1, min_x:max_x + 1, :] = resized_quad
        
        # Add wireframe overlay if requested
        if wireframe > 0:
            wireframe_overlay = np.zeros_like(src_bgr, dtype=np.float32)
            
            # Draw mesh lines
            for gy in range(grid_h + 1):
                for gx in range(grid_w):
                    if gx < len(vertices[gy]) - 1:
                        v1 = vertices[gy][gx]
                        v2 = vertices[gy][gx + 1]
                        cv2.line(wireframe_overlay, v1, v2, (255, 255, 255), 1)
            
            for gy in range(grid_h):
                for gx in range(grid_w + 1):
                    if gy < len(vertices) - 1 and gx < len(vertices[gy]):
                        v1 = vertices[gy][gx]
                        v2 = vertices[gy + 1][gx]
                        cv2.line(wireframe_overlay, v1, v2, (255, 255, 255), 1)
            
            # Blend wireframe with mesh
            mesh_result = cv2.addWeighted(mesh_result, 1.0 - wireframe, wireframe_overlay, wireframe, 0.0)
        
        # Blend with original image based on intensity
        if intensity < 1.0:
            result = cv2.addWeighted(src_bgr.astype(np.float32), 1.0 - intensity, mesh_result, intensity, 0.0)
        else:
            result = mesh_result
        
        return clamp_img(result)

    def reset_params(self) -> None:
        self.grid_size = 0
        self.distortion = 0
        self.intensity = 0
        self.perspective = 0
        self.wireframe = 0
        self._ui = None


class FragmentationFilter(BaseFilter):
    """
    Breaks the image into fragments and scatters them randomly.
    Creates a shattered glass effect with configurable fragment count, scatter radius, rotation, and gaps.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.fragment_count = 0      # [10, 200] number of fragments to create
        self.scatter_radius = 0      # [0, 100] how far fragments scatter from original position
        self.rotation = 0            # [0, 360] maximum rotation angle for fragments
        self.gap_size = 0            # [0, 50] size of gaps between fragments
        self.intensity = 0           # [0, 100] overall effect strength

    @property
    def name(self) -> str:
        return "FRGMNT"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#E91E63"  # pink accent
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Fragments", "fragment_count", 10, 200, 0),
            ("Scatter", "scatter_radius", 0, 100, 0),
            ("Rotation", "rotation", 0, 360, 0),
            ("Gap", "gap_size", 0, 50, 0),
            ("Intensity", "intensity", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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

    def _generate_fragments(self, h: int, w: int, fragment_count: int) -> List[tuple]:
        """Generate random fragment boundaries within the image."""
        fragments = []
        
        for _ in range(fragment_count):
            # Generate random fragment size (minimum 20x20 pixels)
            min_size = 20
            max_size = min(h, w) // 3
            fragment_w = np.random.randint(min_size, max_size + 1)
            fragment_h = np.random.randint(min_size, max_size + 1)
            
            # Generate random position
            x = np.random.randint(0, max(1, w - fragment_w))
            y = np.random.randint(0, max(1, h - fragment_h))
            
            fragments.append((x, y, fragment_w, fragment_h))
        
        return fragments

    def _scatter_fragment(self, fragment: tuple, scatter_radius: float, rotation: float, h: int, w: int) -> tuple:
        """Apply scattering and rotation to a fragment (optimized)."""
        x, y, fragment_w, fragment_h = fragment
        
        # Early return if no scattering needed
        if scatter_radius <= 0 and rotation <= 0:
            return (x, y, fragment_w, fragment_h, 0.0)
        
        # Calculate scatter distance
        max_scatter = min(h, w) * (scatter_radius / 100.0)
        scatter_x = np.random.uniform(-max_scatter, max_scatter)
        scatter_y = np.random.uniform(-max_scatter, max_scatter)
        
        # Apply scatter
        new_x = int(x + scatter_x)
        new_y = int(y + scatter_y)
        
        # Clamp to image bounds (more efficient)
        new_x = np.clip(new_x, 0, w - fragment_w)
        new_y = np.clip(new_y, 0, h - fragment_h)
        
        # Apply rotation (in degrees) - only if needed
        rotation_angle = np.random.uniform(-rotation, rotation) if rotation > 0 else 0.0
        
        return (new_x, new_y, fragment_w, fragment_h, rotation_angle)

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
            
        h, w, c = src_bgr.shape
        result = np.zeros_like(src_bgr)
        
        # Convert parameters to usable values
        fragment_count = max(10, min(200, int(self.fragment_count)))
        scatter_radius = self.scatter_radius
        rotation = self.rotation
        gap_size = int(self.gap_size * min(h, w) / 200)  # Scale gap size to image size
        intensity = self.intensity / 100.0
        
        # Ensure we always generate some fragments if the filter is active
        if fragment_count <= 0:
            fragment_count = 20  # Default fragment count
        
        # Generate fragments
        fragments = self._generate_fragments(h, w, fragment_count)
        
        # Create a mask to track which pixels have been placed (more efficient than boolean)
        placed_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Process fragments in batches for better performance
        batch_size = min(20, len(fragments))
        
        for i in range(0, len(fragments), batch_size):
            batch_fragments = fragments[i:i + batch_size]
            
            for fragment in batch_fragments:
                scattered = self._scatter_fragment(fragment, scatter_radius, rotation, h, w)
                new_x, new_y, fragment_w, fragment_h, rotation_angle = scattered
                
                # Extract original fragment
                orig_x, orig_y, _, _ = fragment
                original_fragment = src_bgr[orig_y:orig_y+fragment_h, orig_x:orig_x+fragment_w, :].copy()
                
                # Apply rotation if specified (only when needed)
                if abs(rotation_angle) > 0.1:
                    # Get rotation matrix
                    center = (fragment_w // 2, fragment_h // 2)
                    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
                    
                    # Rotate the fragment
                    rotated_fragment = cv2.warpAffine(original_fragment, rotation_matrix, 
                                                    (fragment_w, fragment_h), 
                                                    borderMode=cv2.BORDER_CONSTANT, 
                                                    borderValue=(0, 0, 0))
                else:
                    rotated_fragment = original_fragment
                
                # Apply gap effect by creating a mask
                if gap_size > 0:
                    # Create gap mask more efficiently
                    gap_mask = np.ones((fragment_h, fragment_w), dtype=bool)
                    gap_mask[:gap_size, :] = False
                    gap_mask[-gap_size:, :] = False
                    gap_mask[:, :gap_size] = False
                    gap_mask[:, -gap_size:] = False
                    
                    # Apply gap mask
                    rotated_fragment = rotated_fragment * gap_mask[:, :, np.newaxis]
                
                # Place fragment using vectorized operations
                self._place_fragment_vectorized(result, placed_mask, rotated_fragment, 
                                              new_x, new_y, fragment_w, fragment_h, h, w)
        
        # Fill remaining empty areas with a dark background (vectorized)
        empty_mask = placed_mask == 0
        if np.any(empty_mask):
            result[empty_mask] = [20, 20, 20]  # Dark background
        
        # Blend with original image based on intensity
        if intensity < 1.0:
            result = cv2.addWeighted(src_bgr, 1.0 - intensity, result, intensity, 0.0)
        
        return result

    def _place_fragment_vectorized(self, result: np.ndarray, placed_mask: np.ndarray, 
                                 fragment: np.ndarray, new_x: int, new_y: int, 
                                 fragment_w: int, fragment_h: int, h: int, w: int) -> None:
        """Efficiently place a fragment using vectorized operations."""
        # Calculate valid bounds
        start_x = max(0, new_x)
        end_x = min(w, new_x + fragment_w)
        start_y = max(0, new_y)
        end_y = min(h, new_y + fragment_h)
        
        # Calculate fragment slice bounds
        frag_start_x = start_x - new_x
        frag_end_x = frag_start_x + (end_x - start_x)
        frag_start_y = start_y - new_y
        frag_end_y = frag_start_y + (end_y - start_y)
        
        # Extract the valid region
        if end_x > start_x and end_y > start_y:
            valid_region = fragment[frag_start_y:frag_end_y, frag_start_x:frag_end_x, :]
            valid_mask = placed_mask[start_y:end_y, start_x:end_x] == 0
            
            # Only place pixels where mask is empty
            if np.any(valid_mask):
                # Use advanced indexing for efficient placement
                result[start_y:end_y, start_x:end_x, :][valid_mask] = valid_region[valid_mask]
                placed_mask[start_y:end_y, start_x:end_x][valid_mask] = 1

    def reset_params(self) -> None:
        self.fragment_count = 0
        self.scatter_radius = 0
        self.rotation = 0
        self.gap_size = 0
        self.intensity = 0
        self._ui = None


class LiquidMorphingFilter(BaseFilter):
    """
    Creates a liquid morphing effect that makes the image appear to flow and distort
    like liquid. Uses flow fields generated from noise to create organic, flowing
    distortions that completely reshape the image content.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self._ui = None
        self.flow_intensity = 0      # [0, 100] - how strong the liquid effect is
        self.viscosity = 50          # [0, 100] - how thick/slow the liquid flows
        self.turbulence = 0          # [0, 100] - chaotic flow patterns
        self.morph_speed = 0         # [0, 100] - how fast the morphing happens
        self.wave_frequency = 0      # [0, 100] - frequency of wave patterns
        self._flow_field_x = None    # Cached flow field for performance
        self._flow_field_y = None    # Cached flow field for performance
        self._last_params = None     # Track parameter changes

    @property
    def name(self) -> str:
        return "LIQUID_MORPH"

    def build_ui(self, parent: QWidget) -> QWidget:
        if self._ui is not None:
            return self._ui
        
        panel = QFrame(parent)
        panel.setObjectName("effectPanel")
        accent = "#00E676"  # bright green accent
        panel.setStyleSheet(
            f"background-color: #252526; border: 1px solid {accent}; border-radius: 8px; padding: 6px;"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)
        
        params = [
            ("Flow", "flow_intensity", 0, 100, 0),
            ("Viscosity", "viscosity", 0, 100, 50),
            ("Turbulence", "turbulence", 0, 100, 0),
            ("Speed", "morph_speed", 0, 100, 0),
            ("Waves", "wave_frequency", 0, 100, 0),
        ]
        
        rows_needed = (len(params) + 2) // 3
        panel_height = rows_needed * 60 + 20
        panel.setFixedHeight(panel_height)
        
        for idx, (disp, attr, min_a, max_a, neutral) in enumerate(params):
            current = getattr(self, attr)
            knob = KnobControl(
                self, attr, disp, min_a, max_a, current, panel, default_actual=neutral
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

    def _generate_flow_field(self, h: int, w: int) -> tuple:
        """Generate flow field using noise and wave patterns for liquid morphing."""
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Convert to normalized coordinates
        x_norm = x_coords / w
        y_norm = y_coords / h
        
        # Generate multiple noise layers for complex flow
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        
        # Base flow parameters
        flow_strength = self.flow_intensity / 100.0
        viscosity = max(0.1, self.viscosity / 100.0)
        turbulence = self.turbulence / 100.0
        wave_freq = max(0.01, self.wave_frequency / 100.0) * 10.0
        
        if flow_strength <= 0:
            return flow_x, flow_y
        
        # Generate multiple noise octaves for realistic flow
        for octave in range(3):
            scale = 2 ** octave
            freq = 0.1 * scale * wave_freq
            amplitude = flow_strength / (2 ** octave)
            
            # Perlin-like noise using sine waves
            noise_x = (np.sin(x_norm * freq * np.pi * 2) * 
                      np.cos(y_norm * freq * np.pi * 2) * 
                      np.sin((x_norm + y_norm) * freq * np.pi))
            
            noise_y = (np.cos(x_norm * freq * np.pi * 2) * 
                      np.sin(y_norm * freq * np.pi * 2) * 
                      np.cos((x_norm - y_norm) * freq * np.pi))
            
            flow_x += noise_x * amplitude
            flow_y += noise_y * amplitude
        
        # Add turbulence for chaotic flow
        if turbulence > 0:
            # High-frequency noise for turbulence
            turb_freq = wave_freq * 5.0
            turb_x = (np.sin(x_norm * turb_freq * np.pi * 3) * 
                     np.cos(y_norm * turb_freq * np.pi * 3))
            turb_y = (np.cos(x_norm * turb_freq * np.pi * 3) * 
                     np.sin(y_norm * turb_freq * np.pi * 3))
            
            flow_x += turb_x * turbulence * flow_strength * 0.3
            flow_y += turb_y * turbulence * flow_strength * 0.3
        
        # Apply viscosity (smooth the flow field)
        if viscosity < 1.0:
            kernel_size = int((1.0 - viscosity) * 10) + 1
            if kernel_size > 1:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                flow_x = cv2.filter2D(flow_x, -1, kernel)
                flow_y = cv2.filter2D(flow_y, -1, kernel)
        
        # Scale flow field to pixel displacement
        max_displacement = min(h, w) * 0.3 * flow_strength
        flow_x *= max_displacement
        flow_y *= max_displacement
        
        return flow_x, flow_y

    def _apply_liquid_morphing(self, src_bgr: np.ndarray, flow_x: np.ndarray, flow_y: np.ndarray) -> np.ndarray:
        """Apply liquid morphing using flow field displacement."""
        h, w, c = src_bgr.shape
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        
        # Calculate new coordinates with flow displacement
        new_x = x_coords + flow_x
        new_y = y_coords + flow_y
        
        # Clamp coordinates to image bounds
        new_x = np.clip(new_x, 0, w - 1)
        new_y = np.clip(new_y, 0, h - 1)
        
        # Convert to float32 for remap
        map_x = new_x.astype(np.float32)
        map_y = new_y.astype(np.float32)
        
        # Apply the morphing using OpenCV's remap for efficiency
        result = cv2.remap(src_bgr, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return result

    def apply(self, src_bgr: np.ndarray) -> np.ndarray:
        if src_bgr is None:
            return src_bgr
            
        h, w, c = src_bgr.shape
        
        # Check if we need to regenerate flow field
        current_params = (self.flow_intensity, self.viscosity, self.turbulence, 
                         self.wave_frequency, h, w)
        
        if (self._flow_field_x is None or self._flow_field_y is None or 
            self._last_params != current_params):
            self._flow_field_x, self._flow_field_y = self._generate_flow_field(h, w)
            self._last_params = current_params
        
        # Apply liquid morphing
        result = self._apply_liquid_morphing(src_bgr, self._flow_field_x, self._flow_field_y)
        
        # Add morphing speed effect (temporal variation)
        if self.morph_speed > 0:
            # Create a time-based variation
            time_factor = self.morph_speed / 100.0
            # Add subtle time-based distortion
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            time_wave = np.sin(x_coords * 0.01 + y_coords * 0.01) * time_factor * 5
            
            # Apply time-based displacement
            time_flow_x = np.sin(time_wave) * time_factor * 10
            time_flow_y = np.cos(time_wave) * time_factor * 10
            
            # Combine with existing flow
            combined_flow_x = self._flow_field_x + time_flow_x
            combined_flow_y = self._flow_field_y + time_flow_y
            
            result = self._apply_liquid_morphing(src_bgr, combined_flow_x, combined_flow_y)
        
        return result

    def reset_params(self) -> None:
        self.flow_intensity = 0
        self.viscosity = 50
        self.turbulence = 0
        self.morph_speed = 0
        self.wave_frequency = 0
        self._flow_field_x = None
        self._flow_field_y = None
        self._last_params = None
        self._ui = None
