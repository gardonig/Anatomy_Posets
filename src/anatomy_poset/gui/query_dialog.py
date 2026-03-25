import json
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QCloseEvent, QGuiApplication, QImage, QPainter, QPen, QPixmap, QColor
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core.builder import (
    MatrixBuilder,
    _parse_bilateral_core as parse_bilateral_core,
)
from ..core.config import ASSETS_DIR
from ..core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)
from .dialog_widgets import ClickableImageLabel
from .utils import _is_plural_structure, _relation_verb

# Map structure names to view (Front/Side/Rear) -> filename
_STRUCTURE_VIEWS: Dict[str, Dict[str, str]] = {
    "Skeleton": {
        "Front": "skeleton_front.png",
        "Side": "skeleton_side.png",
        "Rear": "skeleton_rear.png",
    },
    "muscles: sub-layer": {
        "Front": "mm_sub_front.png",
        "Side": "mm_sub_side.png",
        "Rear": "mm_sub_rear.png",
    },
    "muscles: superficial": {
        "Front": "mm_super_front.png",
        "Side": "mm_super_side.png",
        "Rear": "mm_super_rear.png",
    },
    "Gastrointestinal system": {
        "Front": "gastro_front.png",
        "Side": "gastro_side.png",
        "Rear": "gastro_rear.png",
    },
    "Urinary / Genital / Respiratory / Endocrine": {
        "Front": "ur_gen_resp_endocrin_front.png",
        "Side": "ur_gen_resp_endocrin_side.png",
        "Rear": "ur_gen_resp_endocrin_rear.png",
    },
}


def _create_anatomy_views_panel(parent: QDialog) -> QWidget:
    """Build the detailed anatomy views panel (structure tabs + rotations + image)."""
    panel = QWidget(parent)
    layout = QVBoxLayout(panel)

    tabs = QTabWidget(panel)
    images_dir = ASSETS_DIR / "images"
    # Make the front/side/rear anatomy views tall so they fill the column
    img_height = 460

    for structure_name, views in _STRUCTURE_VIEWS.items():
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)

        btn_row = QHBoxLayout()
        image_label = ClickableImageLabel(f"{structure_name} — full view", parent)
        image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Allow in-panel zoom and panning for these anatomy views.
        image_label.enable_interactive_view(True)
        # Keep the background white but remove the surrounding border for a cleaner look.
        image_label.setStyleSheet("background: #ffffff; padding: 0px; margin: 0px;")

        buttons: Dict[str, QPushButton] = {}
        group = QButtonGroup(panel)
        group.setExclusive(True)

        def _load_view(
            label: ClickableImageLabel,
            vdict: Dict[str, str],
            view_key: str,
        ) -> None:
            """Load the requested view image into the label."""
            filename = vdict.get(view_key)
            if not filename:
                label.setText(f"[No {view_key.lower()} view]")
                label.setPixmap(QPixmap())
                return
            path = images_dir / filename
            if not path.exists():
                label.setText(f"[Missing: {filename}]")
                label.setPixmap(QPixmap())
                return
            pix = QPixmap(str(path))
            if pix.isNull():
                label.setText(f"[Could not load: {filename}]")
                label.setPixmap(QPixmap())
                return
            label.set_full_pixmap(pix)
            label.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))

        # Center the view buttons horizontally
        btn_row.addStretch(1)

        for view_key in ("Front", "Side", "Rear"):
            btn = QPushButton(view_key)
            btn.setCheckable(True)
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 4px 12px;
                    border-radius: 4px;
                    border: 1px solid #c0c0c5;
                    background: #f2f2f7;
                    color: #1a1a1a;
                }
                QPushButton:hover {
                    background: #e0e0ea;
                }
                QPushButton:checked {
                    background: #007aff;
                    color: white;
                    border-color: #0051d5;
                }
                """
            )

            def on_clicked(
                checked: bool,  # noqa: ARG001
                k: str = view_key,
                vdict: Dict[str, str] = views,
                label: ClickableImageLabel = image_label,
            ) -> None:
                # QButtonGroup ensures only one button is checked at a time.
                _load_view(label, vdict, k)

            btn.clicked.connect(on_clicked)
            group.addButton(btn)
            buttons[view_key] = btn
            btn_row.addWidget(btn)

        btn_row.addStretch(1)
        tab_layout.addLayout(btn_row)
        tab_layout.addWidget(image_label)

        # Default view when the tab is opened
        default_view = "Front" if "Front" in buttons else next(iter(buttons))
        buttons[default_view].setChecked(True)
        _load_view(image_label, views, default_view)

        tabs.addTab(tab, structure_name)

    layout.addWidget(tabs)
    ca_note = QLabel("Structure view images are from Complete Anatomy.", panel)
    ca_note.setAlignment(Qt.AlignmentFlag.AlignCenter)
    ca_note.setStyleSheet("color: #666666; font-size: 11px; margin-top: 2px;")
    layout.addWidget(ca_note)
    return panel


class SliceLocationWidget(QWidget):
    """
    Small widget that shows a human outline and a red line indicating the current
    slice position along the body. Used next to the full-body slice slider to give
    a quick "where in the body" cue for axial (head→feet), coronal (front→back),
    and sagittal (left→right) planes.
    """

    def __init__(self, outline_path: Path | None = None, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Axial / sagittal use the standard outline; coronal uses a dedicated coronal outline.
        self._outline_axial_sagittal = outline_path or (ASSETS_DIR / "images" / "human_outline.png")
        self._outline_coronal = ASSETS_DIR / "images" / "human_outline_coronal.png"
        self._pixmap: QPixmap | None = None
        self._plane = "axial"
        self._min_val = 0
        self._max_val = 0
        self._value = 0
        # Make the outline large (but still shrinkable with the window).
        self.setMaximumWidth(280)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Preferred,
        )
        self.setToolTip("Slice position in body")

        self._reload_pixmap()

    def _reload_pixmap(self) -> None:
        if self._plane == "coronal" and self._outline_coronal.exists():
            path = self._outline_coronal
        else:
            path = self._outline_axial_sagittal
        self._pixmap = QPixmap(str(path)) if path.exists() else None

    def set_plane(self, plane: str) -> None:
        self._plane = plane.lower()
        self._reload_pixmap()
        # Let layouts recompute using the plane-specific sizeHint.
        self.updateGeometry()
        self.update()

    def set_range(self, min_val: int, max_val: int) -> None:
        self._min_val = min_val
        self._max_val = max_val
        self.update()

    def set_value(self, value: int) -> None:
        self._value = value
        self.update()

    def sizeHint(self) -> QSize:
        """Preferred size so layouts keep the outline large and stable across plane changes."""
        # The outline is rendered into a square box (based on min(width, height)).
        # For the axial plane, a tall widget just creates empty padding above/below,
        # so keep it closer to square to free vertical space for the slider.
        if self._plane == "axial":
            return QSize(220, 260)
        return QSize(220, 480)

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()
        if self._pixmap is None or self._pixmap.isNull():
            painter.fillRect(rect, QColor(240, 240, 245))
            painter.setPen(QPen(QColor(120, 120, 120)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No\noutline")
            return
        # Use a consistent content box for all planes so axial/sagittal/coronal outlines
        # appear the same size (coronal no longer dwarfs the others).
        px_rect = self._pixmap.rect()
        if px_rect.isEmpty():
            return
        box_side = min(rect.width(), rect.height())
        box_side = max(1, int(0.92 * box_side))
        box_x = (rect.width() - box_side) / 2.0
        box_y = (rect.height() - box_side) / 2.0
        content_rect = QRectF(box_x, box_y, box_side, box_side)
        scale = min(
            content_rect.width() / px_rect.width(),
            content_rect.height() / px_rect.height(),
        )
        w = max(1, int(px_rect.width() * scale))
        h = max(1, int(px_rect.height() * scale))
        x = content_rect.left() + (content_rect.width() - w) / 2.0
        y = content_rect.top() + (content_rect.height() - h) / 2.0
        target_rect = QRectF(x, y, w, h)
        painter.drawPixmap(target_rect, self._pixmap, QRectF(px_rect))

        # Red line position: t in [0, 1] from slice index.
        # For axial we map min->top of the outline and max->bottom of the outline.
        # For sagittal / coronal we map min->left edge of the outline and max->right edge.
        span = max(1, self._max_val - self._min_val)
        t = (self._value - self._min_val) / span if span else 0.5
        t = max(0.0, min(1.0, t))
        pen = QPen(QColor(220, 50, 50))
        pen.setWidth(3)
        painter.setPen(pen)
        if self._plane in ("sagittal", "coronal"):
            # Vertical line over the outline: left (min) to right (max).
            # For sagittal, keep the indicator away from the extreme edges:
            # map t∈[0,1] into [0.1,0.9] of the outline width.
            if self._plane == "sagittal":
                t = 0.1 + 0.8 * t
            x_line = target_rect.left() + t * max(1.0, target_rect.width() - 1.0)
            painter.drawLine(
                int(x_line),
                int(target_rect.top()),
                int(x_line),
                int(target_rect.bottom()),
            )
        else:
            # Axial: horizontal line, top = head (min), bottom = feet (max)
            y_line = target_rect.top() + t * max(1.0, target_rect.height() - 1.0)
            painter.drawLine(
                int(target_rect.left()),
                int(y_line),
                int(target_rect.right()),
                int(y_line),
            )


class FullBodyVolumePanel(QWidget):
    """
    Embedded viewer for the downsampled full-body volume stored as a NumPy tensor.
    Designed to live alongside the coronal slice stacks in the Query dialog.
    """

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)

        self._volume: np.ndarray | None = None  # shape (Z, Y, X)
        self._plane: str = "axial"  # axial / coronal / sagittal
        self._index: int = 0

        layout = QVBoxLayout(self)

        # Plane selector: centered in the top row, similar to the Anatomy Images window.
        plane_row = QHBoxLayout()
        plane_row.addStretch(1)
        plane_row.addWidget(QLabel("Plane:"))
        # Map human-readable label -> button (keys keep original casing so comparisons match).
        self._plane_buttons: Dict[str, QPushButton] = {}
        self._plane_button_group = QButtonGroup(self)
        self._plane_button_group.setExclusive(True)

        for label in ("Axial", "Coronal", "Sagittal"):
            btn = QPushButton(label)
            btn.setCheckable(True)
            btn.setStyleSheet(
                """
                QPushButton {
                    padding: 4px 12px;
                    border-radius: 4px;
                    border: 1px solid #c0c0c5;
                    background: #f2f2f7;
                    color: #1a1a1a;
                }
                QPushButton:hover {
                    background: #e0e0ea;
                }
                QPushButton:checked {
                    background: #007aff;
                    color: white;
                    border-color: #0051d5;
                }
                """
            )

            def on_clicked(
                checked: bool,  # noqa: ARG001
                name: str = label,
            ) -> None:
                # QButtonGroup enforces exclusivity; we only need to react when a button becomes checked.
                if checked:
                    self._on_plane_changed(name)

            btn.clicked.connect(on_clicked)
            self._plane_button_group.addButton(btn)
            self._plane_buttons[label] = btn
            plane_row.addWidget(btn)

        plane_row.addStretch(1)
        layout.addLayout(plane_row)

        # Second row: volume chooser aligned to the left.
        top_row = QHBoxLayout()
        layout.addLayout(top_row)

        self._choose_btn = QPushButton("Choose volume (.npy)…")
        self._choose_btn.setToolTip(
            "Select a NumPy tensor file containing the full-body volume (e.g. full_body_tensor.npy)."
        )
        self._choose_btn.clicked.connect(self._select_tensor_file)
        top_row.addWidget(self._choose_btn)
        top_row.addStretch(1)

        # Default to axial plane (internal state starts as "axial").
        self._plane_buttons["Axial"].setChecked(True)

        # Image + navigation controls: image on the left (with its own slider row below),
        # slice-location outline and buttons/slider column on the right.
        content_row = QHBoxLayout()
        layout.addLayout(content_row)

        left_col = QVBoxLayout()
        content_row.addLayout(left_col, 3)

        self._image_label = ClickableImageLabel("Full-body slice — full view", parent=parent)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.enable_interactive_view(True)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_label.setStyleSheet(
            "border: none; border-radius: 0px; "
            "background: #000000; padding: 0px; margin-top: 4px;"
        )
        left_col.addWidget(self._image_label)

        # Right side: compact slice outline ABOVE the slider controls
        right_widget = QWidget(self)
        right_col = QVBoxLayout(right_widget)
        right_col.setContentsMargins(0, 0, 0, 0)
        # Keep outline and slider snug so the slider can be as long as possible.
        right_col.setSpacing(2)
        content_row.addWidget(right_widget, 1)

        self._slice_location = SliceLocationWidget(parent=self)
        right_col.addWidget(self._slice_location, 0, Qt.AlignmentFlag.AlignHCenter)

        slider_widget = QWidget(self)
        slider_col = QVBoxLayout(slider_widget)
        slider_col.setContentsMargins(0, 0, 0, 0)
        right_col.addWidget(slider_widget, 1)

        self._prev_btn = QPushButton("▲")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.clicked.connect(self._step_prev)
        self._prev_btn.setEnabled(False)
        slider_col.addWidget(self._prev_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        # Vertical slider used for axial and coronal planes.
        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        # First slice at the bottom, last slice at the top.
        self._slider.setInvertedAppearance(False)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        # Keep the slider centered between the up/down buttons.
        slider_col.addWidget(self._slider, 1, Qt.AlignmentFlag.AlignHCenter)

        self._next_btn = QPushButton("▼")
        self._next_btn.setFixedWidth(32)
        self._next_btn.clicked.connect(self._step_next)
        self._next_btn.setEnabled(False)
        slider_col.addWidget(self._next_btn, 0, Qt.AlignmentFlag.AlignHCenter)

        self._index_label = QLabel("Slice: – / –")
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_col.addWidget(self._index_label)

        # Horizontal slider + fine-step arrows directly below the image on the left.
        # Used for coronal and sagittal planes so that left/right movement of the slider
        # matches left/right of the body.
        bottom_row = QHBoxLayout()
        left_col.addLayout(bottom_row)

        self._bottom_prev_btn = QPushButton("◀")
        self._bottom_prev_btn.setFixedWidth(32)
        self._bottom_prev_btn.setEnabled(False)
        self._bottom_prev_btn.clicked.connect(self._step_prev)
        self._bottom_prev_btn.setVisible(False)
        bottom_row.addWidget(self._bottom_prev_btn)

        self._bottom_slider = QSlider(Qt.Orientation.Horizontal)
        self._bottom_slider.setMinimum(0)
        self._bottom_slider.setMaximum(0)
        self._bottom_slider.setSingleStep(1)
        self._bottom_slider.setPageStep(10)
        self._bottom_slider.setEnabled(False)
        self._bottom_slider.valueChanged.connect(self._on_bottom_slider_changed)
        self._bottom_slider.setVisible(False)
        bottom_row.addWidget(self._bottom_slider, stretch=1)

        self._bottom_next_btn = QPushButton("▶")
        self._bottom_next_btn.setFixedWidth(32)
        self._bottom_next_btn.setEnabled(False)
        self._bottom_next_btn.clicked.connect(self._step_next)
        self._bottom_next_btn.setVisible(False)
        bottom_row.addWidget(self._bottom_next_btn)

        # Try to auto-load a default tensor: prefer assets/visible_human_tensors, then repo root.
        self._try_auto_load_tensor()

    # ---- Tensor loading ----
    def _select_tensor_file(self) -> None:
        base = QFileDialog.getOpenFileName(
            self,
            "Select NumPy volume file",
            str(ASSETS_DIR),
            "NumPy files (*.npy *.npz);;All files (*)",
        )[0]
        if not base:
            return
        self._load_tensor(Path(base))

    def _try_auto_load_tensor(self) -> None:
        candidates: List[Path] = []
        # Prefer the RGB tensor generated by view_full_body_male.py under visible_human_tensors,
        # then fall back to grayscale.
        vh_dir = ASSETS_DIR / "visible_human_tensors"
        candidates.append(vh_dir / "full_body_tensor_rgb.npy")
        candidates.append(vh_dir / "full_body_tensor.npy")
        try:
            repo_root = Path(__file__).resolve().parents[3]
            candidates.append(repo_root / "full_body_tensor_rgb.npy")
            candidates.append(repo_root / "full_body_tensor.npy")
        except Exception:
            pass

        for p in candidates:
            if not p.exists():
                continue
            # Never block dialog startup if a default tensor is missing/corrupt.
            try:
                self._load_tensor(p)
            except Exception:
                continue
            if self._volume is not None:
                return

        self._image_label.setText(
            "Full-body volume viewer\n\n"
            "Default tensor not found at assets/visible_human_tensors/full_body_tensor_rgb.npy.\n\n"
            "Click 'Choose volume (.npy)…' to load a NumPy tensor created from the\n"
            "downsampled full-body slices."
        )

    def _load_tensor(self, path: Path) -> None:
        try:
            arr = np.load(str(path))
        except Exception as exc:  # noqa: BLE001
            self._image_label.setText(f"[Could not load NumPy array: {exc}]")
            self._image_label.setPixmap(QPixmap())
            return

        if arr.ndim not in (3, 4):
            self._image_label.setText(f"[Expected a 3D or 4D RGB array, got shape {arr.shape!r}]")
            self._image_label.setPixmap(QPixmap())
            return

        self._volume = arr.astype(np.float32, copy=False)
        # Initialize index in the middle of the axial dimension
        z_dim = self._volume.shape[0]
        self._index = z_dim // 2

        # Enable all navigation controls; _reset_slider_for_plane will decide which slider is visible.
        self._slider.setEnabled(True)
        self._bottom_slider.setEnabled(True)
        self._prev_btn.setEnabled(True)
        self._next_btn.setEnabled(True)
        self._reset_slider_for_plane()
        self._update_image()

    # ---- Navigation / plane ----
    def _on_plane_changed(self, label: str) -> None:
        self._plane = label.lower()
        self._reset_slider_for_plane()
        self._update_image()

    def _reset_slider_for_plane(self) -> None:
        if self._volume is None:
            return
        if self._volume.ndim == 4:
            z_dim, y_dim, x_dim, _ = self._volume.shape
        else:
            z_dim, y_dim, x_dim = self._volume.shape
        if self._plane == "axial":
            min_idx, max_idx = 0, max(0, z_dim - 1)
        elif self._plane == "coronal":
            min_idx, max_idx = 0, max(0, y_dim - 1)
        else:  # sagittal
            min_idx, max_idx = 0, max(0, x_dim - 1)

        # Clamp current index into the valid range.
        self._index = max(min_idx, min(self._index, max_idx))

        # Only axial uses the vertical slider; coronal and sagittal use the bottom slider.
        use_vertical = self._plane == "axial"
        self._slider.setVisible(use_vertical)
        self._prev_btn.setVisible(use_vertical)
        self._next_btn.setVisible(use_vertical)

        self._bottom_slider.setVisible(not use_vertical)
        self._bottom_prev_btn.setVisible(not use_vertical)
        self._bottom_next_btn.setVisible(not use_vertical)

        self._bottom_prev_btn.setEnabled(not use_vertical and self._volume is not None)
        self._bottom_next_btn.setEnabled(not use_vertical and self._volume is not None)

        # Update ranges for both sliders.
        self._slider.blockSignals(True)
        # For axial: top = head (first slice), bottom = feet (last slice)
        self._slider.setInvertedAppearance(self._plane == "axial")
        self._slider.setMinimum(min_idx)
        self._slider.setMaximum(max_idx)
        self._slider.blockSignals(False)

        self._bottom_slider.blockSignals(True)
        self._bottom_slider.setMinimum(min_idx)
        self._bottom_slider.setMaximum(max_idx)
        self._bottom_slider.blockSignals(False)

        # Set the active slider's value.
        if use_vertical:
            self._slider.setValue(self._index)
        else:
            self._bottom_slider.setValue(self._index)

        self._update_slice_location()

    def _update_slice_location(self) -> None:
        """Update the slice location outline widget to match current plane and index."""
        if self._volume is None:
            return
        if self._volume.ndim == 4:
            z_dim, y_dim, x_dim, _ = self._volume.shape
        else:
            z_dim, y_dim, x_dim = self._volume.shape
        if self._plane == "axial":
            min_idx, max_idx = 0, max(0, z_dim - 1)
        elif self._plane == "coronal":
            min_idx, max_idx = 0, max(0, y_dim - 1)
        else:
            min_idx, max_idx = 0, max(0, x_dim - 1)
        self._slice_location.set_plane(self._plane)
        self._slice_location.set_range(min_idx, max_idx)
        self._slice_location.set_value(self._index)

    def _on_slider_changed(self, value: int) -> None:
        self._index = int(value)
        self._slice_location.set_value(self._index)
        self._update_image()

    def _on_bottom_slider_changed(self, value: int) -> None:
        self._index = int(value)
        self._slice_location.set_value(self._index)
        self._update_image()

    def _step_prev(self) -> None:
        if self._volume is None:
            return
        active = self._slider if self._plane == "axial" else self._bottom_slider
        if self._index <= active.minimum():
            return
        self._index -= 1
        active.setValue(self._index)

    def _step_next(self) -> None:
        if self._volume is None:
            return
        active = self._slider if self._plane == "axial" else self._bottom_slider
        if self._index >= active.maximum():
            return
        self._index += 1
        active.setValue(self._index)

    # ---- Rendering ----
    def _current_slice_array(self) -> np.ndarray | None:
        if self._volume is None:
            return None
        if self._volume.ndim == 4:
            z_dim, y_dim, x_dim, _ = self._volume.shape
        else:
            z_dim, y_dim, x_dim = self._volume.shape
        idx = int(max(self._slider.minimum(), min(self._index, self._slider.maximum())))

        if self._plane == "axial":
            # (Y, X) or (Y, X, 3)
            sl = self._volume[idx, ...]
            # Rotate 180° in-plane
            sl = np.rot90(sl, 2, axes=(0, 1))
            return sl

        if self._plane == "coronal":
            # (Z, X) or (Z, X, 3)
            sl = self._volume[:, idx, ...]
            # No additional rotation; show as-sliced (after cropping).
            return sl

        # Sagittal
        sl = self._volume[:, :, idx, ...] if self._volume.ndim == 4 else self._volume[:, :, idx]
        # No additional rotation; show as-sliced (after cropping).
        return sl

    def _update_image(self) -> None:
        sl = self._current_slice_array()
        if sl is None:
            self._image_label.setText("[No volume loaded]")
            self._image_label.setPixmap(QPixmap())
            self._index_label.setText("Slice: – / –")
            return

        # Normalize to [0, 255] uint8 for display
        sl = np.nan_to_num(sl, nan=0.0, posinf=0.0, neginf=0.0)
        if sl.ndim == 2:
            vmin = float(sl.min())
            vmax = float(sl.max())
        else:
            vmin = float(sl.min())
            vmax = float(sl.max())
        if vmax > vmin:
            sl_norm = (sl - vmin) / (vmax - vmin)
        else:
            sl_norm = np.zeros_like(sl, dtype=np.float32)
        img8 = (sl_norm * 255.0).clip(0, 255).astype(np.uint8)
        if img8.ndim == 2:
            h, w = img8.shape
            rgb = np.stack([img8, img8, img8], axis=-1)
        else:
            h, w, _ = img8.shape
            rgb = img8
        rgb = np.ascontiguousarray(rgb)

        qimg = QImage(
            rgb.data,
            w,
            h,
            3 * w,
            QImage.Format.Format_RGB888,
        )
        pix = QPixmap.fromImage(qimg)

        # Preserve the current zoom/offset when changing slices so that the
        # next slice is shown with the same view as the previous one.
        prev_zoom = getattr(self._image_label, "_zoom", 1.0)
        prev_off_x = getattr(self._image_label, "_offset_x", 0.0)
        prev_off_y = getattr(self._image_label, "_offset_y", 0.0)

        self._image_label.set_full_pixmap(pix)
        # Use the raw pixmap; ClickableImageLabel will handle fitting and zoom.
        self._image_label.setPixmap(pix)

        # Restore the previous interactive view state.
        self._image_label._zoom = prev_zoom
        self._image_label._offset_x = prev_off_x
        self._image_label._offset_y = prev_off_y
        self._image_label.update()

        max_idx = self._slider.maximum()
        self._index_label.setText(
            f"{self._plane.capitalize()} slice: {self._index + 1} / {max_idx + 1}"
        )

class QueryDialog(QDialog):
    """
    Standalone dialog for expert queries only.
    Clinicians focus on answering questions; no structure input.
    """

    def __init__(
        self,
        poset_builder: MatrixBuilder,
        autosave_path: Path,
        axis: str,
        save_callback: Callable[[str, List[Structure], List[List[int]]], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Expert Query")
        # Initial preferred size (user can resize freely).
        self.resize(1100, 580)
        self.setModal(False)
        self._clamping_geometry: bool = False

        self.poset_builder = poset_builder
        self._autosave_path = autosave_path
        self._axis = axis
        self._save_callback = save_callback
        self._feedback_log_path: Path | None = None
        self.pending_pair: Tuple[int, int] | None = None
        # answer is True (Yes), False (No), or None ("Not sure"/skipped)
        self._answer_history: List[Tuple[int, int, Optional[bool]]] = []
        # Snapshot of M before each user answer (enables correct undo for tri-valued matrix).
        self._matrix_snapshots: List[List[List[int]]] = []
        # Clamp initial size to screen, but do not lock min/max.
        self._clamp_to_current_screen()

        # For vertical axis, detect bilateral (Left/Right) cores to combine in question text
        self._bilateral_cores: Set[str] = set()
        # For vertical axis, store combined CoM for bilateral cores (mean of Left/Right)
        self._bilateral_core_com_vertical: Dict[str, float] = {}
        if self._axis == AXIS_VERTICAL:
            core_counts: Dict[str, int] = {}
            names = [s.name.strip() for s in self.poset_builder.structures]
            for name in names:
                side, core = parse_bilateral_core(name)
                if core is None:
                    continue
                core_counts[core] = core_counts.get(core, 0) + 1
            self._bilateral_cores = {c for c, cnt in core_counts.items() if cnt >= 2}
            # Precompute vertical CoM for bilateral cores as the mean of their sides.
            core_to_values: Dict[str, List[float]] = {}
            for s in self.poset_builder.structures:
                side, core = parse_bilateral_core(s.name)
                if core is None or core not in self._bilateral_cores:
                    continue
                core_to_values.setdefault(core, []).append(s.com_vertical)
            for core, vals in core_to_values.items():
                if vals:
                    self._bilateral_core_com_vertical[core] = sum(vals) / len(vals)

        main_layout = QHBoxLayout(self)

        # Use a horizontal splitter so the three main sections (anatomy images,
        # questions/overview, full-body volume) can be resized by the user.
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        # Make the splitter handles thicker and visually distinct so they are easy to grab.
        splitter.setHandleWidth(10)
        splitter.setStyleSheet(
            """
            QSplitter::handle {
                background-color: rgba(0, 0, 0, 18);  /* very subtle, almost transparent */
            }
            QSplitter::handle:hover {
                background-color: rgba(0, 0, 0, 40);  /* slightly stronger on hover */
            }
            """
        )
        main_layout.addWidget(splitter)

        # Middle column: questions and labels overview (inserted between anatomy and coronal panels)
        middle_widget = QWidget(self)
        left_col = QVBoxLayout(middle_widget)

        # --- Questions panel (placed above overview) ---
        questions_group = QGroupBox("Questions")
        questions_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 13px; }")
        questions_layout = QVBoxLayout(questions_group)

        # Question card (no border inside the questions panel)
        self.question_card = QFrame()
        self.question_card.setMinimumHeight(120)
        self.question_card.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border-radius: 12px;
                padding: 24px;
                margin: 8px 0 12px 0;
            }
            """
        )
        card_layout = QVBoxLayout(self.question_card)
        self.query_label = QLabel("")
        self.query_label.setWordWrap(True)
        self.query_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.query_label.setStyleSheet(
            "color: #1a1a1a; font-size: 22px; font-weight: 500; line-height: 1.4;"
        )
        card_layout.addWidget(self.query_label)

        # Center of mass info for the current pair (per axis + pair mean)
        self.com_label = QLabel("")
        self.com_label.setWordWrap(True)
        self.com_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.com_label.setStyleSheet(
            "color: #555555; font-size: 13px; margin-top: 4px;"
        )
        card_layout.addWidget(self.com_label)
        questions_layout.addWidget(self.question_card)

        self.feedback_box = QPlainTextEdit()
        self.feedback_box.setPlaceholderText(
            "Optional comment/feedback on this question / your answer (e.g. ambiguity, missing context, unsure anatomy, etc.)"
        )
        # Compact by default, but allow shrinking with the window.
        self.feedback_box.setMaximumHeight(90)
        self.feedback_box.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.feedback_box.setStyleSheet(
            "background: #ffffff; color: #1a1a1a; border: 1px solid #e0e0e0; border-radius: 8px; padding: 8px;"
        )
        questions_layout.addWidget(self.feedback_box)

        # Back, Yes, No
        btn_row = QHBoxLayout()
        self.back_btn = QPushButton("← Undo")
        self.back_btn.setEnabled(False)
        self.back_btn.setStyleSheet("padding: 10px 16px; font-size: 14px;")
        self.back_btn.clicked.connect(self.go_back_one_question)
        btn_row.addWidget(self.back_btn)

        self.yes_btn = QPushButton("Yes")
        self.yes_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #2e7d32; color: white; border: none; border-radius: 8px;
                padding: 14px 28px; font-size: 18px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #388e3c; }
            QPushButton:pressed:enabled { background-color: #1b5e20; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.yes_btn.clicked.connect(lambda: self.answer_query(True))

        self.no_btn = QPushButton("No")
        self.no_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #c62828; color: white; border: none; border-radius: 8px;
                padding: 14px 28px; font-size: 18px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #d32f2f; }
            QPushButton:pressed:enabled { background-color: #b71c1c; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.no_btn.clicked.connect(lambda: self.answer_query(False))
        
        self.not_sure_btn = QPushButton("Not sure")
        self.not_sure_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #f2f2f7; color: #1a1a1a; border: 1px solid #d1d1d6; border-radius: 8px;
                padding: 14px 18px; font-size: 16px; font-weight: 600;
            }
            QPushButton:hover:enabled { background-color: #e5e5ea; }
            QPushButton:pressed:enabled { background-color: #d1d1d6; }
            QPushButton:disabled { background-color: #f2f2f7; color: #757575; border: 1px solid #e0e0e0; }
            """
        )
        self.not_sure_btn.clicked.connect(lambda: self.answer_query(None))
        btn_row.addStretch()
        btn_row.addWidget(self.no_btn)
        btn_row.addWidget(self.not_sure_btn)
        btn_row.addWidget(self.yes_btn)
        btn_row.addStretch()
        questions_layout.addLayout(btn_row)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setFixedHeight(8)
        self.progress_bar.setStyleSheet(
            """
            QProgressBar {
                border: none;
                border-radius: 4px;
                background: #e0e0e0;
            }
            QProgressBar::chunk {
                border-radius: 4px;
                background: #007aff;
            }
            """
        )
        questions_layout.addWidget(self.progress_bar)

        # Finish and Close (shown when done)
        self.finish_btn = QPushButton("Done")
        self.finish_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        self.finish_btn.clicked.connect(self.accept)
        self.finish_btn.hide()
        questions_layout.addWidget(self.finish_btn)

        # Do not let the questions panel stretch to fill all remaining vertical space.
        questions_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        left_col.addWidget(questions_group)

        # --- Segmentation labels overview panel (below questions) ---
        overview_group = QGroupBox("Segmentation Labels Overview")
        overview_group.setStyleSheet("QGroupBox { font-weight: 600; font-size: 13px; }")
        overview_layout = QVBoxLayout(overview_group)

        # Segmentation classes overview image
        overview_label = ClickableImageLabel("Segmentation classes overview — full view", self)
        overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overview_label.enable_interactive_view(True)
        overview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        overview_label.setStyleSheet(
            "background: #ffffff; padding: 0px; margin: 8px 0 4px 0;"
        )
        overview_path = ASSETS_DIR / "definition_images" / "overview_classes_v2.png"
        if overview_path.exists():
            overview_pix = QPixmap(str(overview_path))
            if not overview_pix.isNull():
                overview_label.set_full_pixmap(overview_pix)
                overview_label.setPixmap(
                    overview_pix.scaledToHeight(220, Qt.SmoothTransformation)
                )
        if overview_label.pixmap() is None or overview_label.pixmap().isNull():
            overview_label.setText("[Segmentation classes overview image missing]")
        overview_layout.addWidget(overview_label)

        overview_link = QLabel(
            '<a href="https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/overview_classes_v2.png">'
            "Source: TotalSegmentator overview classes v2</a>",
            self,
        )
        overview_link.setOpenExternalLinks(True)
        overview_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overview_link.setStyleSheet("color: #0a66c2; font-size: 11px; margin-bottom: 4px;")
        overview_layout.addWidget(overview_link)

        # Likewise, keep the overview panel compact vertically.
        overview_group.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Maximum)

        left_col.addWidget(overview_group)

        # Add the middle column widget to the splitter
        splitter.addWidget(middle_widget)

        # Right column: Anatomy Images + labels overview below
        anatomy_group = QGroupBox("Anatomy Images")
        anatomy_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 13px; }"
        )
        anatomy_layout = QVBoxLayout(anatomy_group)
        anatomy_layout.addWidget(_create_anatomy_views_panel(self))

        # Give the anatomy panel a tall left column with the front/side/rear views.
        splitter.insertWidget(0, anatomy_group)

        # Imaging support: full-body volume viewer (coronal slice stacks are kept in code but hidden by default)
        coronal_group = QGroupBox("Full-Body Volume")
        coronal_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 13px; }"
        )
        coronal_layout = QVBoxLayout(coronal_group)

        # NOTE: CoronalSlicesPanel remains available in code but is not added to the UI by default.
        # If needed in the future, it can be re-added here alongside the volume panel.
        # coronal_panel = CoronalSlicesPanel(self)
        # coronal_layout.addWidget(coronal_panel)

        # Full-body volume viewer (NumPy tensor)
        volume_panel = FullBodyVolumePanel(self)
        coronal_layout.addWidget(volume_panel)

        # Data source note for the volume viewer
        vh_source = QLabel(
            '<a href="https://data.lhncbc.nlm.nih.gov/public/Visible-Human/Male-Images/PNG_format/index.html">'
            "Source: Visible Human Male full-body volume (NLM)</a>",
            self,
        )
        vh_source.setOpenExternalLinks(True)
        vh_source.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vh_source.setStyleSheet("color: #0a66c2; font-size: 11px; margin-top: 4px;")
        coronal_layout.addWidget(vh_source)

        splitter.addWidget(coronal_group)

        # Set initial relative sizes for the three panes (can be adjusted by user)
        splitter.setStretchFactor(0, 2)  # anatomy images
        splitter.setStretchFactor(1, 3)  # questions/overview
        splitter.setStretchFactor(2, 3)  # full-body volume

        self._advance_to_next_query()

    def _clamp_to_current_screen(self) -> None:
        """
        Clamp this window's size to the available geometry of the screen the
        window is currently on. We intentionally do not clamp position so the
        user can move the dialog partly off-screen, like a normal macOS window.
        """
        if self._clamping_geometry:
            return
        self._clamping_geometry = True
        try:
            # Prefer the screen under the window center; fall back to primary.
            center = self.frameGeometry().center()
            screen = QGuiApplication.screenAt(center) or QGuiApplication.primaryScreen()
            if screen is None:
                return
            avail = screen.availableGeometry()

            # Hard cap: never allow the *outer* window frame to exceed the current screen.
            # Qt's availableGeometry() is in the same coordinate system as the window, but
            # the window decorations (title bar/borders) add extra size beyond self.size().
            # If we don't account for that, the frame can spill off-screen even if the
            # content size is <= availableGeometry().
            frame_extra_w = max(0, self.frameGeometry().width() - self.size().width())
            frame_extra_h = max(0, self.frameGeometry().height() - self.size().height())
            max_w = max(200, avail.width() - frame_extra_w)
            max_h = max(200, avail.height() - frame_extra_h)
            self.setMaximumSize(max_w, max_h)

            # Clamp size (never exceed current screen once frame is included).
            new_w = min(self.width(), max_w)
            new_h = min(self.height(), max_h)
            if new_w != self.width() or new_h != self.height():
                self.resize(new_w, new_h)
        finally:
            self._clamping_geometry = False

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._clamp_to_current_screen()

    def moveEvent(self, event) -> None:  # type: ignore[override]
        # Update max size when the window is moved to another monitor.
        super().moveEvent(event)
        self._clamp_to_current_screen()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        # Ensure we never exceed the current monitor size even after UI-triggered resizes.
        super().resizeEvent(event)
        self._clamp_to_current_screen()

    def _autosave_poset(self, seal_lower_triangle: bool = False) -> None:
        """
        Persist the poset JSON (autosave after each answer / undo by default).

        When ``seal_lower_triangle`` is True (session complete or window closing), run
        :meth:`MatrixBuilder.seal_lower_triangle_com_prior` so the file does not keep
        spurious ``-2`` below the diagonal from partial NO answers. Intermediate saves
        skip sealing so progress is cheap to write.
        """
        if not self._autosave_path or not self._save_callback:
            return
        try:
            if seal_lower_triangle:
                self.poset_builder.seal_lower_triangle_com_prior()
            structures = self.poset_builder.structures
            matrix = self.poset_builder.M
            self._save_callback(self._axis, structures, matrix)
        except Exception:
            pass

    def closeEvent(self, event: QCloseEvent) -> None:  # type: ignore[override]
        """Final save with CoM lower-triangle seal so partial sessions persist cleanly."""
        try:
            self._autosave_poset(seal_lower_triangle=True)
        except Exception:
            pass
        super().closeEvent(event)

    def _append_feedback_report(
        self,
        *,
        question_text: str,
        answer: Optional[bool],
        feedback: str,
    ) -> None:
        """
        Append a feedback report entry to a .jsonl file next to the autosave JSON.
        A file is created the first time feedback is provided.
        """
        feedback = (feedback or "").strip()
        if not feedback:
            return
        try:
            if self._feedback_log_path is None:
                base_dir = self._autosave_path.parent
                stem = self._autosave_path.stem
                self._feedback_log_path = base_dir / f"{stem}.feedback.jsonl"

            entry = {
                "axis": self._axis,
                "question": question_text,
                "answer": (
                    "yes" if answer is True else
                    "no" if answer is False else
                    "not_sure"
                ),
                "feedback": feedback,
            }
            self._feedback_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._feedback_log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            # Feedback logging must never interrupt querying.
            pass

    def _advance_to_next_query(self) -> None:
        pair = self.poset_builder.next_pair()
        self.pending_pair = pair
        if pair is not None:
            i, j = pair
            # Only for vertical axis do we collapse bilateral same-core pairs.
            # For lateral/AP we intentionally ask left/right sides separately.
            if self._axis == AXIS_VERTICAL:
                # Never ask "is the lungs above the lungs" — skip same bilateral core (left vs right).
                core_i = self._bilateral_core_for_index(i)
                core_j = self._bilateral_core_for_index(j)
                if core_i is not None and core_i == core_j:
                    self.poset_builder.record_skip(i, j)
                    self._advance_to_next_query()
                    return
        if pair is None:
            self._autosave_poset(seal_lower_triangle=True)
            self.query_label.setText(
                "Thank you for your participation!\n\nEnjoy the pizza 🍕"
            )
            self.yes_btn.hide()
            self.no_btn.hide()
            self.not_sure_btn.hide()
            self.back_btn.hide()
            self.finish_btn.show()
            self.progress_bar.setValue(100)
            return
        i, j = pair
        si, sj = self.poset_builder.structures[i], self.poset_builder.structures[j]
        verb = _relation_verb(self._axis)
        name_i = self._display_name(i, si.name)
        name_j = self._display_name(j, sj.name)
        subj_verb = "Are" if _is_plural_structure(name_i) else "Is"
        self.query_label.setText(f"{subj_verb} the {name_i} {verb} the {name_j}?")
        # Center of mass information: individual values and mean for the current axis
        if self._axis == AXIS_VERTICAL:
            # If bilateral cores are merged (Left/Right), use their combined CoM (keyed by singular core).
            core_i = self._bilateral_core_for_index(i)
            core_j = self._bilateral_core_for_index(j)
            ci = (
                self._bilateral_core_com_vertical[core_i]
                if core_i is not None
                else si.com_vertical
            )
            cj = (
                self._bilateral_core_com_vertical[core_j]
                if core_j is not None
                else sj.com_vertical
            )
            axis_label = "vertical"
        elif self._axis == AXIS_MEDIOLATERAL:
            ci = si.com_lateral
            cj = sj.com_lateral
            axis_label = "lateral"
        else:
            ci = si.com_anteroposterior
            cj = sj.com_anteroposterior
            axis_label = "anteroposterior"
        # One line per structure. For vertical axis, ci/cj may already be
        # the mean of left/right when a bilateral core is combined.
        self.com_label.setText(
            f"{name_i}: CoM {axis_label} = {np.round(ci, 1)}\n"
            f"{name_j}: CoM {axis_label} = {np.round(cj, 1)}"
        )
        self._update_progress()
        # Progress autosave (no seal — seal on thank-you or window close).
        if len(self._answer_history) > 0:
            self._autosave_poset(seal_lower_triangle=False)

    def answer_query(self, is_above: Optional[bool]) -> None:
        if self.pending_pair is None:
            return
        # Save optional feedback (if any) along with the question and answer.
        try:
            feedback_text = self.feedback_box.toPlainText() if self.feedback_box is not None else ""
            self.feedback_box.clear()
        except Exception:
            feedback_text = ""
        self._append_feedback_report(
            question_text=self.query_label.text(),
            answer=is_above,
            feedback=feedback_text,
        )
        i, j = self.pending_pair
        self._matrix_snapshots.append([row[:] for row in self.poset_builder.M])
        self._answer_history.append((i, j, is_above))
        self.back_btn.setEnabled(True)

        if is_above is True:
            self.poset_builder.record_response_matrix(i, j, 1)
        elif is_above is False:
            self.poset_builder.record_response_matrix(i, j, -1)
        else:
            self.poset_builder.record_unknown(i, j)
        self._advance_to_next_query()

    def go_back_one_question(self) -> None:
        if not self._answer_history or not self._matrix_snapshots:
            return
        last_i, last_j, _last_answer = self._answer_history.pop()
        prev_m = self._matrix_snapshots.pop()
        self.poset_builder.restore_matrix(prev_m)
        self.poset_builder.finished = False
        self.poset_builder.current_gap = last_j - last_i
        self.poset_builder.current_i = last_i + 1
        self.pending_pair = (last_i, last_j)
        si, sj = self.poset_builder.structures[last_i], self.poset_builder.structures[last_j]
        verb = _relation_verb(self._axis)
        name_i = self._display_name(last_i, si.name)
        name_j = self._display_name(last_j, sj.name)
        subj_verb = "Are" if _is_plural_structure(name_i) else "Is"
        self.query_label.setText(f"(Correcting) {subj_verb} the {name_i} {verb} the {name_j}?")
        # Refresh CoM info for the corrected pair
        if self._axis == AXIS_VERTICAL:
            core_i = self._bilateral_core_for_index(last_i)
            core_j = self._bilateral_core_for_index(last_j)
            ci = (
                self._bilateral_core_com_vertical[core_i]
                if core_i is not None
                else si.com_vertical
            )
            cj = (
                self._bilateral_core_com_vertical[core_j]
                if core_j is not None
                else sj.com_vertical
            )
            axis_label = "vertical"
        elif self._axis == AXIS_MEDIOLATERAL:
            ci = si.com_lateral
            cj = sj.com_lateral
            axis_label = "lateral"
        else:
            ci = si.com_anteroposterior
            cj = sj.com_anteroposterior
            axis_label = "anteroposterior"
        self.com_label.setText(
            f"{name_i}: CoM {axis_label} = {np.round(ci, 1)}\n"
            f"{name_j}: CoM {axis_label} = {np.round(cj, 1)}"
        )
        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        self.not_sure_btn.setEnabled(True)
        if not self._answer_history:
            self.back_btn.setEnabled(False)
        self._update_progress()
        self._autosave_poset(seal_lower_triangle=False)

    def _update_progress(self) -> None:
        asked = len(self._answer_history)
        remaining = self.poset_builder.estimate_remaining_questions()
        total = asked + remaining
        if total == 0:
            value = 0
        else:
            value = int(100 * asked / total)
        self.progress_bar.setValue(value)

    def _pluralize_core(self, core: str) -> str:
        """One word for the query: e.g. Lung -> lungs, Kidney -> kidneys."""
        s = core.lower()
        if not s:
            return s
        if s.endswith("y") and len(s) > 1 and s[-2] not in "aeiou":
            return s[:-1] + "ies"
        if s.endswith("s"):
            return s
        return s + "s"

    def _bilateral_core_for_index(self, idx: int) -> Optional[str]:
        """Singular core name for this structure if it is a bilateral side, else None (for CoM lookup)."""
        if idx < 0 or idx >= len(self.poset_builder.structures):
            return None
        name = self.poset_builder.structures[idx].name.strip()
        _side, core = parse_bilateral_core(name)
        if core and core in self._bilateral_cores:
            return core
        return None

    def _display_name(self, idx: int, original: str) -> str:
        if self._axis != AXIS_VERTICAL:
            return original

        name = original.strip()
        _side, core = parse_bilateral_core(name)
        if core and core in self._bilateral_cores:
            return self._pluralize_core(core)
        return original