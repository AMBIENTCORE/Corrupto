#!/usr/bin/env python3
# -*- coding: utf-8 -*-


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
        # Create a horizontal layout instead of a splitter
        central_widget = QWidget()
        central_layout = QHBoxLayout(central_widget)
        central_layout.setContentsMargins(0, 0, 0, 0)
        central_layout.setSpacing(0)
        self.setCentralWidget(central_widget)

        # Left panel: file buttons and filter list
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(6, 6, 6, 6)
        left_layout.setSpacing(6)

        self.btn_open = QPushButton("OPEN")
        self.btn_open.clicked.connect(self.open_image_dialog)
        left_layout.addWidget(self.btn_open)

        self.btn_save = QPushButton("SAVE")
        self.btn_save.clicked.connect(self.save_image_dialog)
        self.btn_save.setEnabled(False)
        left_layout.addWidget(self.btn_save)

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

        # Clear button placed under RANDOMIZE
        self.btn_clear = QPushButton("CLEAR")
        self.btn_clear.clicked.connect(self.clear_all_effects)
        self.btn_clear.setProperty("class", "clear")
        left_layout.addWidget(self.btn_clear)
        # Style for the clear button (yellow)
        self.btn_clear.setStyleSheet(
            f"QPushButton[class='clear'] {{ background: #FFC107; color: #000000; border: 1px solid #F57F17; }}"
        )

        # Apply button placed under CLEAR
        self.btn_apply = QPushButton("APPLY")
        self.btn_apply.clicked.connect(self.apply_current_preview)
        # Apply button uses the positive accent class
        self.btn_apply.setProperty("class", "positive")
        left_layout.addWidget(self.btn_apply)
        # Accent style for the Apply button
        self.btn_apply.setStyleSheet(
            f"QPushButton[class='positive'] {{ background: {COLORS['SUCCESS']}; color: #000000; border: 1px solid #2e7d32; }}"
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

        # Add widgets to the central layout with fixed proportions
        central_layout.addWidget(left, 17)  # Left panel gets 1/10 of the space
        central_layout.addWidget(right, 83)  # Right panel gets 9/10 of the space

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
        Prompt the user to choose a filename and save the current preview image.
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

        # Get the current preview image (what the user is seeing)
        preview_img = self._get_current_preview_image()
        if preview_img is None:
            QMessageBox.warning(self, "Save", "No preview image to save.")
            return
            
        params = []
        if ext in [".jpg", ".jpeg"]:
            params = [cv2.IMWRITE_JPEG_QUALITY, 95]

        success, buf = cv2.imencode(ext, preview_img, params)
        if not success:
            QMessageBox.warning(self, "Save", "Failed to encode image.")
            return
        try:
            buf.tofile(out_path)
        except Exception as e:
            QMessageBox.warning(self, "Save", f"Failed to save image: {e}")
            return

        QMessageBox.information(self, "Save", f"SAVED TO:\n{out_path.upper()}")

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

    def _get_current_preview_image(self) -> Optional[np.ndarray]:
        """
        Generate and return the current preview image without displaying it.
        Used for saving the current preview state.
        """
        if self._image_committed is None:
            return None

        # Determine current selected filters (in list order)
        selected_filters: List[BaseFilter] = []
        for index in self.filter_list.selectedIndexes():
            row = index.row()
            if 0 <= row < len(self.filters):
                selected_filters.append(self.filters[row])

        if not selected_filters:
            # No filter selected; return committed image
            return self._image_committed.copy()

        # Start with the committed image
        preview = self._image_committed.copy()
        # Apply each filter in order
        for f in selected_filters:
            try:
                preview = f.apply(preview)
            except Exception:
                # If a filter fails, skip it
                pass

        return preview

    def _update_preview(self) -> None:
        """
        Generate and display a preview image.  If no filters are selected,
        display the committed image.  Otherwise, apply each selected filter
        sequentially to the committed image.
        """
        preview = self._get_current_preview_image()
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

    def clear_all_effects(self) -> None:
        """
        Clear all current effects and reset to the original loaded image.
        This resets both the committed image and clears all filter selections.
        """
        if self._image_loaded is None:
            return
            
        # Reset to the original loaded image
        self._image_committed = self._image_loaded.copy()
        self._history.clear()
        
        # Clear all filter selections
        self.filter_list.clearSelection()
        
        # Reset parameters for all filters
        for f in self.filters:
            f.reset_params()
        
        # Rebuild the UI (will be empty since no filters are selected)
        self._on_filter_selection_changed()
        
        # Update the preview to show the original image
        self._update_preview()

    def apply_current_preview(self) -> None:
        """
        Commit the current preview image to the working image.  Applies all
        selected filters in sequence to the current committed image, allowing
        effects to build on top of previously applied effects.  After
        applying, reset parameters and clear the filter selection.
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

        # Apply effects to the current committed image (not the original loaded image)
        # This allows effects to build on top of previously applied effects
        new_img = self._image_committed.copy()
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

        # Clear the filter selection
        self.filter_list.clearSelection()
        
        # Rebuild the UI (will be empty since no filters are selected)
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