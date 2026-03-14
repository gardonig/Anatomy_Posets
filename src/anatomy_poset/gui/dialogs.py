from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QFileDialog,
    QSlider,
    QComboBox,
)

from ..core.builder import PosetBuilder
from ..core.config import ASSETS_DIR
from ..core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)
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
        # Allow the image to grow when the window is resized while keeping a sensible minimum.
        image_label.setMinimumHeight(img_height)
        image_label.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 8px; "
            "background: #ffffff; padding: 4px; margin: 4px;"
        )

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


class ImagePreviewDialog(QDialog):
    """
    Simple full-window image preview dialog used when the user clicks an image.
    """

    def __init__(self, pixmap: QPixmap, title: str | None = None, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title or "Image preview")
        self.setModal(True)
        layout = QVBoxLayout(self)

        label = ClickableImageLabel(preview_title=title or "Image preview", parent=self)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Full-screen preview should be zoomable/pannable but must not open itself again.
        label.enable_interactive_view(True)
        # In preview mode we want to fit as large as possible.
        label.set_fit_scale(1.0)
        label.set_preview_click_enabled(False)
        label.set_full_pixmap(pixmap)
        layout.addWidget(label)

        # Choose an initial window size that fits on screen.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            avail = screen.availableGeometry()
            max_w = int(avail.width() * 0.9)
            max_h = int(avail.height() * 0.9)
        else:
            max_w = pixmap.width()
            max_h = pixmap.height()

        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(120)
        close_btn.setStyleSheet(
            "QPushButton { margin-top: 8px; padding: 6px 16px; border-radius: 6px; "
            "border: 1px solid #c0c0c5; background: #f2f2f7; color: #1a1a1a; }"
            "QPushButton:hover { background: #e0e0ea; } QPushButton:pressed { background: #d0d0dd; }"
        )
        close_btn.clicked.connect(self.accept)
        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_row.addWidget(close_btn)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        target_w = min(pixmap.width(), max_w)
        target_h = min(pixmap.height(), max_h)
        self.resize(target_w + 40, target_h + 80)


class ClickableImageLabel(QLabel):
    """
    QLabel that opens its pixmap in a full-window preview when clicked.
    Shows "Click to enlarge" overlay on hover.
    """

    def __init__(self, preview_title: str | None = None, parent: QDialog | None = None) -> None:
        super().__init__(parent)
        self._preview_title = preview_title or "Image preview"
        self._full_pixmap: QPixmap | None = None
        self._hovered = False
        # Slightly shrink the fitted image so it never touches the border/padding.
        self._fit_scale: float = 0.94
        # Interactive zoom/pan is optional and disabled by default for embedded images.
        self._interactive: bool = False
        # Whether clicking should open a separate preview dialog.
        self._allow_preview_click: bool = True
        self._zoom: float = 1.0
        self._offset_x: float = 0.0
        self._offset_y: float = 0.0
        self._panning: bool = False
        self._last_pos = None
        self.setMouseTracking(True)

    def set_full_pixmap(self, pixmap: QPixmap) -> None:
        """Store the original-resolution pixmap for high-quality preview."""
        self._full_pixmap = pixmap
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

    def set_fit_scale(self, scale: float) -> None:
        """Controls how large the fitted image appears inside the label (1.0 = max fit)."""
        self._fit_scale = max(0.2, min(float(scale), 1.0))
        self.update()

    def enable_interactive_view(self, enabled: bool = True) -> None:
        """Enable or disable scroll-to-zoom and drag-to-pan for this label."""
        self._interactive = enabled
        self._zoom = 1.0
        self._offset_x = 0.0
        self._offset_y = 0.0

    def set_preview_click_enabled(self, enabled: bool = True) -> None:
        """Enable or disable opening a new preview dialog when clicked."""
        self._allow_preview_click = enabled

    def enterEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = True
        self.update()

    def leaveEvent(self, event) -> None:  # type: ignore[override]
        self._hovered = False
        self.update()

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        base = self._full_pixmap or self.pixmap()
        if base is not None and not base.isNull():
            w = base.width()
            h = base.height()
            if w > 0 and h > 0:
                # Compute a base scale that fits the full image into the label,
                # then apply interactive zoom on top.
                rect = self.rect()
                sx = rect.width() / w
                sy = rect.height() / h
                base_scale = min(sx, sy) * self._fit_scale
                scale = base_scale * self._zoom

                target_w = w * scale
                target_h = h * scale
                cx = rect.center().x() + self._offset_x
                cy = rect.center().y() + self._offset_y
                target_rect = QRectF(
                    cx - target_w / 2.0,
                    cy - target_h / 2.0,
                    target_w,
                    target_h,
                )
                painter.drawPixmap(target_rect, base, QRectF(0, 0, w, h))

        # Hover hint overlay
        if self._hovered and base is not None and not base.isNull():
            font = self.font()
            font.setPointSize(max(10, font.pointSize() + 1))
            painter.setFont(font)
            text = "Click to enlarge"
            fm = painter.fontMetrics()
            tw, th = fm.horizontalAdvance(text), fm.height()
            x, y = (self.rect().width() - tw) // 2, (self.rect().height() - th) // 2
            painter.setPen(QColor(0, 0, 0, 80))
            painter.drawText(x + 1, y + 1, text)
            painter.setPen(QColor(255, 255, 255, 220))
            painter.drawText(x, y, text)

        painter.end()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        # Right- or middle-button drag: pan the zoomed image (only when interactive).
        if self._interactive and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            base = self._full_pixmap or self.pixmap()
            if base is not None and not base.isNull():
                self._panning = True
                self._last_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        if event.button() == Qt.MouseButton.LeftButton and self._allow_preview_click:
            base = self._full_pixmap or self.pixmap()
            if base is None:
                return
            dlg = ImagePreviewDialog(base, self._preview_title, self.window())
            dlg.exec()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # type: ignore[override]
        if self._interactive and self._panning and self._last_pos is not None:
            delta = event.position() - self._last_pos
            self._last_pos = event.position()
            self._offset_x += delta.x()
            self._offset_y += delta.y()
            self.update()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # type: ignore[override]
        if self._interactive and self._panning and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            self._panning = False
            self._last_pos = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self._interactive:
            # Default QLabel behaviour when interactive zoom is disabled.
            super().wheelEvent(event)
            return
        base = self._full_pixmap or self.pixmap()
        if base is None or base.isNull():
            super().wheelEvent(event)
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 1.15 if delta > 0 else 1 / 1.15
        self._zoom *= factor
        # Clamp zoom to a reasonable range
        self._zoom = max(0.2, min(self._zoom, 10.0))
        self.update()


class CoronalSlicesPanel(QWidget):
    """
    Embedded viewer for coronal image stacks (e.g. Visible Human PNG slices).
    Designed to live in the right-hand side of the Query dialog.
    """

    def __init__(self, parent: QDialog | None = None) -> None:
        super().__init__(parent)

        self._base_dir: Path | None = None
        self._stacks: Dict[str, List[Path]] = {}
        self._current_region: str | None = None
        self._current_index: int = 0

        layout = QVBoxLayout(self)

        # Top controls: choose folder + region selector
        top_row = QHBoxLayout()
        layout.addLayout(top_row)

        self._choose_btn = QPushButton("Choose image folder…")
        self._choose_btn.setToolTip(
            "Select the folder that contains the subfolders with PNG slices "
            "(e.g. head, thorax, abdomen, pelvis, thighs, legs)."
        )
        self._choose_btn.clicked.connect(self._select_base_folder)
        top_row.addWidget(self._choose_btn)

        top_row.addStretch(1)

        self._region_combo = QComboBox()
        self._region_combo.setEnabled(False)
        self._region_combo.currentTextChanged.connect(self._on_region_changed)
        top_row.addWidget(QLabel("Region:"))
        top_row.addWidget(self._region_combo)

        # Image + vertical navigation controls on the right
        content_row = QHBoxLayout()
        layout.addLayout(content_row)

        self._image_label = ClickableImageLabel("Coronal slice — full view", parent=parent)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # Use a generous minimum height, but allow the label to grow/shrink
        # with the window so users can resize freely.
        self._image_label.setMinimumHeight(600)
        self._image_label.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 8px; "
            "background: #000000; padding: 4px; margin-top: 4px;"
        )
        content_row.addWidget(self._image_label, stretch=1)

        right_col = QVBoxLayout()
        content_row.addLayout(right_col)

        self._prev_btn = QPushButton("▲")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.clicked.connect(self._step_prev)
        self._prev_btn.setEnabled(False)
        right_col.addWidget(self._prev_btn)

        self._slider = QSlider(Qt.Orientation.Vertical)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.setInvertedAppearance(True)  # first slice at the top
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        right_col.addWidget(self._slider, stretch=1)

        self._next_btn = QPushButton("▼")
        self._next_btn.setFixedWidth(32)
        self._next_btn.clicked.connect(self._step_next)
        self._next_btn.setEnabled(False)
        right_col.addWidget(self._next_btn)

        self._index_label = QLabel("Slice: – / –")
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        right_col.addWidget(self._index_label)

        # Try to auto-load a default folder under assets:
        # - Prefer assets/visible_human_male if it exists,
        # - fall back to assets/visible_human.
        default_base = None
        vh_male = ASSETS_DIR / "visible_human_male"
        vh_generic = ASSETS_DIR / "visible_human"
        if vh_male.exists():
            default_base = vh_male
        elif vh_generic.exists():
            default_base = vh_generic

        if default_base is not None:
            self._load_base_folder(default_base)
            # If successful, we can hide the folder chooser so the user
            # only needs the Region selector inside the GUI.
            if self._stacks:
                self._choose_btn.hide()
        else:
            # Helpful placeholder text when no default folder is available.
            self._image_label.setText(
                "Coronal slice viewer\n\n"
                "Click 'Choose image folder…' and select the directory that contains\n"
                "subfolders such as head, thorax, abdomen, pelvis, thighs, legs.\n"
                "Then use the Region menu, slider, and arrows to browse slices."
            )

    # ---- Folder / stack handling ----
    def _select_base_folder(self) -> None:
        base = QFileDialog.getExistingDirectory(
            self,
            "Select base folder with coronal image stacks",
            str(self._base_dir or ASSETS_DIR),
        )
        if not base:
            return
        self._load_base_folder(Path(base))

    def _load_base_folder(self, base: Path) -> None:
        if not base.exists() or not base.is_dir():
            self._image_label.setText(
                f"The selected folder does not exist or is not a directory:\n{base}"
            )
            self._image_label.setPixmap(QPixmap())
            return

        self._base_dir = base
        self._stacks.clear()

        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            # Collect only PNG images (case-insensitive), ignore any .txt or other files.
            pngs = sorted(
                p
                for p in sub.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            )
            if not pngs:
                continue
            self._stacks[sub.name] = pngs

        self._region_combo.blockSignals(True)
        self._region_combo.clear()
        for region in sorted(self._stacks.keys()):
            self._region_combo.addItem(region)
        self._region_combo.blockSignals(False)

        has_any = bool(self._stacks)
        self._region_combo.setEnabled(has_any)
        self._slider.setEnabled(has_any)
        self._prev_btn.setEnabled(has_any)
        self._next_btn.setEnabled(has_any)

        if not has_any:
            self._image_label.setText(
                "No PNG image stacks were found.\n\n"
                "The selected folder should contain subfolders (e.g. head, thorax, abdomen,\n"
                "pelvis, thighs, legs) with .png slices inside each of them."
            )
            self._image_label.setPixmap(QPixmap())
            self._index_label.setText("Slice: – / –")
            return

        first_region = self._region_combo.currentText()
        if first_region:
            self._on_region_changed(first_region)

    # ---- Region / navigation ----
    def _on_region_changed(self, region: str) -> None:
        if not region or region not in self._stacks:
            return
        self._current_region = region
        self._current_index = 0
        num_slices = len(self._stacks[region])
        self._slider.blockSignals(True)
        self._slider.setMinimum(0)
        self._slider.setMaximum(max(0, num_slices - 1))
        self._slider.setValue(0)
        self._slider.blockSignals(False)
        self._update_image()

    def _on_slider_changed(self, value: int) -> None:
        self._current_index = int(value)
        self._update_image()

    def _step_prev(self) -> None:
        if self._current_region is None:
            return
        if self._current_index <= 0:
            return
        self._current_index -= 1
        self._slider.setValue(self._current_index)

    def _step_next(self) -> None:
        if self._current_region is None:
            return
        stack = self._stacks.get(self._current_region, [])
        if not stack:
            return
        if self._current_index >= len(stack) - 1:
            return
        self._current_index += 1
        self._slider.setValue(self._current_index)

    def _update_image(self) -> None:
        if self._current_region is None:
            return
        stack = self._stacks.get(self._current_region, [])
        if not stack:
            self._image_label.setText("[No slices found for this region]")
            self._image_label.setPixmap(QPixmap())
            self._index_label.setText("Slice: – / –")
            return

        idx = max(0, min(self._current_index, len(stack) - 1))
        self._current_index = idx
        path = stack[idx]
        pix = QPixmap(str(path))
        if pix.isNull():
            self._image_label.setText(f"[Could not load slice: {path.name}]")
            self._image_label.setPixmap(QPixmap())
        else:
            self._image_label.set_full_pixmap(pix)
            # Keep a stable, large display size across slices.
            target_h = self._image_label.height() or 360
            self._image_label.setPixmap(
                pix.scaledToHeight(
                    target_h,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self._index_label.setText(f"Slice: {idx + 1} / {len(stack)}")


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
        self.setFixedWidth(52)
        self.setMinimumHeight(140)
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
        self.update()

    def set_range(self, min_val: int, max_val: int) -> None:
        self._min_val = min_val
        self._max_val = max_val
        self.update()

    def set_value(self, value: int) -> None:
        self._value = value
        self.update()

    def paintEvent(self, event) -> None:  # noqa: ARG002
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        rect = self.rect()
        if self._pixmap is None or self._pixmap.isNull():
            painter.fillRect(rect, QColor(240, 240, 245))
            painter.setPen(QPen(QColor(120, 120, 120)))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "No\noutline")
            return
        # Scale outline to fit widget, preserving aspect ratio
        px_rect = self._pixmap.rect()
        if px_rect.isEmpty():
            return
        scale = min(rect.width() / px_rect.width(), rect.height() / px_rect.height())
        w = max(1, int(px_rect.width() * scale))
        h = max(1, int(px_rect.height() * scale))
        x = (rect.width() - w) // 2
        y = (rect.height() - h) // 2
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
            # Vertical line over the outline: left (min) to right (max)
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

        # Image + vertical navigation controls on the right (match CoronalSlicesPanel layout)
        content_row = QHBoxLayout()
        layout.addLayout(content_row)

        self._image_label = ClickableImageLabel("Full-body slice — full view", parent=parent)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumHeight(300)
        self._image_label.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 8px; "
            "background: #000000; padding: 4px; margin-top: 4px;"
        )
        content_row.addWidget(self._image_label, stretch=1)

        right_col = QHBoxLayout()
        content_row.addLayout(right_col)

        self._slice_location = SliceLocationWidget(
            ASSETS_DIR / "images" / "human_outline.png",
            parent=self,
        )
        right_col.addWidget(self._slice_location)

        slider_col = QVBoxLayout()
        right_col.addLayout(slider_col)

        self._prev_btn = QPushButton("▲")
        self._prev_btn.setFixedWidth(32)
        self._prev_btn.clicked.connect(self._step_prev)
        self._prev_btn.setEnabled(False)
        slider_col.addWidget(self._prev_btn)

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
        slider_col.addWidget(self._slider, stretch=1)

        self._next_btn = QPushButton("▼")
        self._next_btn.setFixedWidth(32)
        self._next_btn.clicked.connect(self._step_next)
        self._next_btn.setEnabled(False)
        slider_col.addWidget(self._next_btn)

        self._index_label = QLabel("Slice: – / –")
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        slider_col.addWidget(self._index_label)

        # Horizontal slider + fine-step arrows at the bottom, used only for sagittal plane so that
        # left/right movement of the slider matches left/right of the body.
        bottom_row = QHBoxLayout()
        layout.addLayout(bottom_row)

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

        # Try to auto-load a default tensor: prefer assets/full_body_tensor.npy, then repo root.
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
        # Prefer the RGB tensor generated by view_full_body_male.py, then fall back to grayscale.
        candidates.append(ASSETS_DIR / "full_body_tensor_rgb.npy")
        candidates.append(ASSETS_DIR / "full_body_tensor.npy")
        try:
            repo_root = Path(__file__).resolve().parents[3]
            candidates.append(repo_root / "full_body_tensor_rgb.npy")
            candidates.append(repo_root / "full_body_tensor.npy")
        except Exception:
            pass

        for p in candidates:
            if p.exists():
                self._load_tensor(p)
                return

        self._image_label.setText(
            "Full-body volume viewer\n\n"
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

        use_vertical = self._plane in ("axial", "coronal")
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
        if self._index <= self._slider.minimum():
            return
        self._index -= 1
        self._slider.setValue(self._index)

    def _step_next(self) -> None:
        if self._volume is None:
            return
        if self._index >= self._slider.maximum():
            return
        self._index += 1
        self._slider.setValue(self._index)

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
        self._image_label.set_full_pixmap(pix)
        target_h = self._image_label.height() or 300
        self._image_label.setPixmap(
            pix.scaledToHeight(
                target_h,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

        max_idx = self._slider.maximum()
        self._index_label.setText(
            f"{self._plane.capitalize()} slice: {self._index + 1} / {max_idx + 1}"
        )

class DefinitionDialog(QDialog):
    """
    Shown before the query window. Displays the definition (and examples for vertical).
    User presses "Proceed" to proceed to questions.
    """

    def __init__(self, axis: str = AXIS_VERTICAL) -> None:
        super().__init__()
        self.setWindowTitle("Instructions — Poset Construction")
        # Match the size of the axis-specific definition windows for consistency.
        self.resize(820, 520)
        self.setModal(True)
        self._axis = axis
        self.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout(self)

        # All text: dark on white (no grey backgrounds)
        _text_style = "color: #1a1a1a; font-size: 14px; padding: 6px 0;"

        # ---- 1. Welcome + anatomical position (side-by-side) ----
        welcome_heading = QLabel("Welcome to the Anatomical Structure Questionnaire")
        welcome_heading.setStyleSheet("color: #1a1a1a; font-weight: bold; font-size: 18px; padding: 0 0 12px 0;")
        layout.addWidget(welcome_heading)

        intro_text = (
            "Thank you for taking part. In this questionnaire you will be asked to compare pairs of "
            "anatomical structures and indicate whether one is strictly above the other (vertical axis) or "
            "strictly to the left of the other (mediolateral axis). Your answers help us build a spatial ordering "
            "that can be used to check and correct automatic segmentations. There are no wrong answers—we need "
            "your clinical judgement.\n\n"
            "We assume the patient is in standard anatomical position, as shown in the figure.\n\n"
            "If you have any questions, please do not hesitate to reach out to Gian or Güney."
        )
        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(_text_style)

        anatomy_img = ClickableImageLabel("Anatomical axes — full view")
        anatomy_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        anatomy_img.setFixedHeight(400)
        
        # Use updated example figure for anatomical axes.
        anatomy_path = ASSETS_DIR / "definition_images" / "Axes_example.png"
        
        if anatomy_path.exists():
            anatomy_pix = QPixmap(str(anatomy_path))
            if not anatomy_pix.isNull():
                anatomy_img.set_full_pixmap(anatomy_pix)
                anatomy_img.setPixmap(
                    anatomy_pix.scaledToHeight(400, Qt.SmoothTransformation)
                )
        if anatomy_img.pixmap() is None or anatomy_img.pixmap().isNull():
            anatomy_img.setText("[Anatomical position diagram missing]")
        
        # Put text and image side-by-side
        intro_row = QHBoxLayout()
        intro_row.addWidget(intro_label, stretch=3)
        intro_row.addWidget(anatomy_img, stretch=2)
        layout.addLayout(intro_row)

        layout.addStretch(1)

        # Proceed button + image source note in a bottom bar
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        button_layout = QHBoxLayout(button_box)

        anatomy_ref = QLabel("Images in this window are captured from Complete Anatomy.")
        anatomy_ref.setWordWrap(True)
        anatomy_ref.setStyleSheet("color: #555; font-size: 10px; margin-top: 2px;")
        button_layout.addWidget(anatomy_ref)

        button_layout.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        button_layout.addWidget(proceed_btn)
        layout.addWidget(button_box)


class VerticalDefinitionDialog(QDialog):
    """
    Dedicated window for the vertical 'strictly above' definition and examples.
    Shown only when the vertical axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Vertical "strictly above"')
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=3)

        heading = QLabel('Vertical relation: "strictly above"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to:\n'
            '  "Is the first structure strictly above the second along the superior–inferior (head–to–toes) axis?"'
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "One structure is strictly above another if the lowest point of the upper structure is higher "
            "than the highest point of the lower one."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        left_col.addStretch(1)

        # Right column: textual examples + images side by side (No and Yes)
        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=4)

        _img_dir = ASSETS_DIR / "definition_images"
        img_height = 320

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=2)

        # No example (Femur–Tibia)
        no_col = QVBoxLayout()
        no_col.setSpacing(4)
        examples_col.addLayout(no_col)
        no_text = QLabel('Example 1: "Is the Femur strictly above the Tibia?" → Answer: No.')
        no_text.setWordWrap(True)
        no_text.setStyleSheet(_text_style)
        no_col.addWidget(no_text)

        no_label = ClickableImageLabel("Vertical example 1 — full view")
        no_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        no_label.setFixedHeight(img_height)
        no_label.setStyleSheet(img_style)
        no_path = _img_dir / "example_vert_No.png"
        if no_path.exists():
            pix = QPixmap(str(no_path))
            if not pix.isNull():
                no_label.set_full_pixmap(pix)
                no_label.setPixmap(
                    pix.scaledToHeight(img_height, Qt.SmoothTransformation)
                )
        if no_label.pixmap() is None or no_label.pixmap().isNull():
            no_label.setText("[Add vertical No example image here]")
            no_label.setStyleSheet(placeholder_style)
        no_col.addWidget(no_label)

        # Yes example (Femur–Fibula)
        yes_col = QVBoxLayout()
        yes_col.setSpacing(4)
        examples_col.addLayout(yes_col)
        yes_text = QLabel('Example 2: "Is the Femur strictly above the Fibula?" → Answer: Yes.')
        yes_text.setWordWrap(True)
        yes_text.setStyleSheet(_text_style)
        yes_text.setContentsMargins(12, 0, 0, 0)
        yes_col.addWidget(yes_text)

        yes_label = ClickableImageLabel("Vertical example 2 — full view")
        yes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        yes_label.setFixedHeight(img_height)
        yes_label.setStyleSheet(img_style)
        yes_path = _img_dir / "eample_vert_Yes.png"
        if yes_path.exists():
            pix = QPixmap(str(yes_path))
            if not pix.isNull():
                yes_label.set_full_pixmap(pix)
                yes_label.setPixmap(
                    pix.scaledToHeight(img_height, Qt.SmoothTransformation)
                )
        if yes_label.pixmap() is None or yes_label.pixmap().isNull():
            yes_label.setText("[Add vertical Yes example image here]")
            yes_label.setStyleSheet(placeholder_style)
        yes_col.addWidget(yes_label)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=1)

        com_heading = QLabel("Center of mass (CoM)")
        com_heading.setStyleSheet(_heading_style)
        com_col.addWidget(com_heading)

        com_text = QLabel(
            "For the vertical axis, CoM is scaled from 0 (toes/feet, most inferior) "
            "to 100 (vertex/head, most superior)."
        )
        com_text.setWordWrap(True)
        com_text.setStyleSheet(_text_style)
        com_col.addWidget(com_text)

        com_img = ClickableImageLabel("Vertical CoM — full view")
        com_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        com_img.setFixedHeight(2 * img_height)
        com_img.setStyleSheet(img_style)
        # Updated labeled image for vertical CoM
        com_path = _img_dir / "Vertical_CoM_numbers.png"
        if com_path.exists():
            pix = QPixmap(str(com_path))
            if not pix.isNull():
                com_img.set_full_pixmap(pix)
                com_img.setPixmap(
                    pix.scaledToHeight(2 * img_height, Qt.SmoothTransformation)
                )
        if com_img.pixmap() is None or com_img.pixmap().isNull():
            com_img.setText("[Add vertical CoM image here]")
            com_img.setStyleSheet(placeholder_style)
        com_col.addWidget(com_img)

        # Proceed button + image source note below all content (bottom bar)
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        btn_row = QHBoxLayout(button_box)
        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        btn_row.addWidget(source)
        btn_row.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row.addWidget(proceed_btn)
        main.addWidget(button_box)


class MediolateralDefinitionDialog(QDialog):
    """
    Dedicated window for the lateral (right–left) 'strictly to the left of' definition and examples.
    Shown only when the lateral axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Lateral "strictly to the left of"')
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=3)

        heading = QLabel('Lateral relation: "strictly to the left of"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to: '
            '"Is the first structure strictly to the left of the second along the '
            "right–left (lateral) axis, from the patient's perspective?"
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "One structure is strictly to the left of another if the rightmost point of the first is to the left "
            "of the leftmost point of the second."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        patient_note = QLabel(
            "Left and right are always defined from the patient's view: the patient's right femur is to the right "
            "of the patient's left femur."
        )
        patient_note.setWordWrap(True)
        patient_note.setStyleSheet(_text_style)
        left_col.addWidget(patient_note)

        left_col.addStretch(1)

        # Right column: two examples side-by-side (text above image)
        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=4)

        _img_dir = ASSETS_DIR / "definition_images"
        img_height = 280

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=2)

        # Example (Yes)
        yes_col = QVBoxLayout()
        yes_col.setSpacing(4)
        examples_col.addLayout(yes_col)
        ex1_text = QLabel(
            'Example 1: "Is the Left femur strictly to the left of the Right femur?" '
            "→ Answer: Yes (from the patient's perspective)."
        )
        ex1_text.setWordWrap(True)
        ex1_text.setStyleSheet(_text_style)
        yes_col.addWidget(ex1_text)

        ex1_img = ClickableImageLabel("Lateral example 1 — full view")
        ex1_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex1_img.setFixedHeight(img_height)
        ex1_img.setStyleSheet(img_style)
        ex1_path = _img_dir / "example_lat_yes.png"
        if ex1_path.exists():
            pix = QPixmap(str(ex1_path))
            if not pix.isNull():
                ex1_img.set_full_pixmap(pix)
                ex1_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex1_img.pixmap() is None or ex1_img.pixmap().isNull():
            ex1_img.setText("[Add mediolateral Yes example image here]")
            ex1_img.setStyleSheet(placeholder_style)
        yes_col.addWidget(ex1_img)

        # Example (No)
        no_col = QVBoxLayout()
        no_col.setSpacing(4)
        examples_col.addLayout(no_col)
        ex2_text = QLabel(
            'Example 2: "Is the Left femur strictly to the left of the pelvis?" '
            "→ Answer: No (they overlap in the mediolateral direction)."
        )
        ex2_text.setWordWrap(True)
        ex2_text.setStyleSheet(_text_style)
        ex2_text.setContentsMargins(12, 0, 0, 0)
        no_col.addWidget(ex2_text)

        ex2_img = ClickableImageLabel("Lateral example 2 — full view")
        ex2_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex2_img.setFixedHeight(img_height)
        ex2_img.setStyleSheet(img_style)
        ex2_path = _img_dir / "example_lat_no.png"
        if ex2_path.exists():
            pix = QPixmap(str(ex2_path))
            if not pix.isNull():
                ex2_img.set_full_pixmap(pix)
                ex2_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex2_img.pixmap() is None or ex2_img.pixmap().isNull():
            ex2_img.setText("[Add mediolateral No example image here]")
            ex2_img.setStyleSheet(placeholder_style)
        no_col.addWidget(ex2_img)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=1)

        com_heading = QLabel("Center of mass (CoM)")
        com_heading.setStyleSheet(_heading_style)
        com_col.addWidget(com_heading)

        com_text = QLabel(
            "For the lateral axis, CoM is scaled from 0 (far right, e.g. right thumb) "
            "to 100 (far left, e.g. left thumb), from the patient's perspective."
        )
        com_text.setWordWrap(True)
        com_text.setStyleSheet(_text_style)
        com_col.addWidget(com_text)

        com_img = ClickableImageLabel("Lateral CoM — full view")
        com_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        com_img.setFixedHeight(2 * img_height)
        com_img.setStyleSheet(img_style)
        # Updated labeled image for lateral CoM
        com_path = _img_dir / "Lateral_CoM_numbers.png"
        if com_path.exists():
            pix = QPixmap(str(com_path))
            if not pix.isNull():
                com_img.set_full_pixmap(pix)
                com_img.setPixmap(pix.scaledToHeight(2 * img_height, Qt.SmoothTransformation))
        if com_img.pixmap() is None or com_img.pixmap().isNull():
            com_img.setText("[Add mediolateral CoM image here]")
            com_img.setStyleSheet(placeholder_style)
        com_col.addWidget(com_img)

        # Proceed button + image source note below all content
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        btn_row = QHBoxLayout(button_box)
        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        btn_row.addWidget(source)
        btn_row.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row.addWidget(proceed_btn)
        main.addWidget(button_box)


class AnteroposteriorDefinitionDialog(QDialog):
    """
    Dedicated window for the anteroposterior (front–back) 'strictly in front of' definition and examples.
    Shown only when the anteroposterior axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Anteroposterior "strictly in front of"')
        self.resize(820, 520)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=3)

        heading = QLabel('Anteroposterior relation: "strictly in front of"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to:\n'
            '  "Is the first structure strictly in front of the second along the front–back '
            "(anteroposterior) axis?"
        )
        text1.setWordWrap(True)
        text1.setStyleSheet(_text_style)
        left_col.addWidget(text1)

        text2 = QLabel(
            "One structure is strictly in front of another if the posterior-most point of the first is anterior "
            "to the anterior-most point of the second."
        )
        text2.setWordWrap(True)
        text2.setStyleSheet(_text_style)
        left_col.addWidget(text2)

        left_col.addStretch(1)

        # Right column: two examples side-by-side (text above image)
        right_col = QVBoxLayout()
        content.addLayout(right_col, stretch=4)

        img_height = 280

        img_style = (
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #ffffff;"
        )
        placeholder_style = (
            "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
            "color: #444; font-size: 13px; background: #ffffff;"
        )

        _img_dir = ASSETS_DIR / "definition_images"

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=2)

        # Example (Yes)
        yes_col = QVBoxLayout()
        yes_col.setSpacing(4)
        examples_col.addLayout(yes_col)
        ex1_text = QLabel(
            'Example 1 (Yes): "Is the sternum strictly in front of the thoracic spine?" → Answer: Yes.'
        )
        ex1_text.setWordWrap(True)
        ex1_text.setStyleSheet(_text_style)
        yes_col.addWidget(ex1_text)

        ex1_img = ClickableImageLabel("Anteroposterior example 1 — full view")
        ex1_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex1_img.setFixedHeight(img_height)
        ex1_img.setStyleSheet(img_style)
        ex1_path = _img_dir / "example_ap_yes.png"
        if ex1_path.exists():
            pix = QPixmap(str(ex1_path))
            if not pix.isNull():
                ex1_img.set_full_pixmap(pix)
                ex1_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex1_img.pixmap() is None or ex1_img.pixmap().isNull():
            ex1_img.setText("[Add anteroposterior Yes example image here]")
            ex1_img.setStyleSheet(placeholder_style)
        yes_col.addWidget(ex1_img)

        # Example (No)
        no_col = QVBoxLayout()
        no_col.setSpacing(4)
        examples_col.addLayout(no_col)
        ex2_text = QLabel(
            'Example 2 (No): "Is the clavicle strictly in front of the cervical spine?" → Answer: No.'
        )
        ex2_text.setWordWrap(True)
        ex2_text.setStyleSheet(_text_style)
        ex2_text.setContentsMargins(12, 0, 0, 0)
        no_col.addWidget(ex2_text)

        ex2_img = ClickableImageLabel("Anteroposterior example 2 — full view")
        ex2_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ex2_img.setFixedHeight(img_height)
        ex2_img.setStyleSheet(img_style)
        ex2_path = _img_dir / "example_ap_no.png"
        if ex2_path.exists():
            pix = QPixmap(str(ex2_path))
            if not pix.isNull():
                ex2_img.set_full_pixmap(pix)
                ex2_img.setPixmap(pix.scaledToHeight(img_height, Qt.SmoothTransformation))
        if ex2_img.pixmap() is None or ex2_img.pixmap().isNull():
            ex2_img.setText("[Add anteroposterior No example image here]")
            ex2_img.setStyleSheet(placeholder_style)
        no_col.addWidget(ex2_img)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=1)

        com_heading = QLabel("Center of mass (CoM)")
        com_heading.setStyleSheet(_heading_style)
        com_col.addWidget(com_heading)

        com_text = QLabel(
            "For the anteroposterior axis, CoM is scaled from 0 (back/dorsal side) "
            "to 100 (front/ventral side)."
        )
        com_text.setWordWrap(True)
        com_text.setStyleSheet(_text_style)
        com_col.addWidget(com_text)

        com_img = ClickableImageLabel("Anteroposterior CoM — full view")
        com_img.setAlignment(Qt.AlignmentFlag.AlignCenter)
        com_img.setFixedHeight(2 * img_height)
        com_img.setStyleSheet(img_style)
        # Updated labeled image for anteroposterior CoM
        com_path = _img_dir / "AP_CoM_numbers.png"
        if com_path.exists():
            pix = QPixmap(str(com_path))
            if not pix.isNull():
                com_img.set_full_pixmap(pix)
                com_img.setPixmap(pix.scaledToHeight(2 * img_height, Qt.SmoothTransformation))
        if com_img.pixmap() is None or com_img.pixmap().isNull():
            com_img.setText("[Add anteroposterior CoM image here]")
            com_img.setStyleSheet(placeholder_style)
        com_col.addWidget(com_img)

        # Proceed button + image source note below all content
        button_box = QFrame()
        button_box.setStyleSheet(
            "QFrame { border-top: 1px solid #e0e0e0; margin-top: 16px; padding-top: 8px; }"
        )
        btn_row = QHBoxLayout(button_box)
        source = QLabel("Images in this window are captured from Complete Anatomy.")
        source.setWordWrap(True)
        source.setStyleSheet("color: #555; font-size: 10px; margin-top: 4px;")
        btn_row.addWidget(source)
        btn_row.addStretch(1)
        proceed_btn = QPushButton("Proceed")
        proceed_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 10px 22px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        proceed_btn.clicked.connect(self.accept)
        btn_row.addWidget(proceed_btn)
        main.addWidget(button_box)


class QueryDialog(QDialog):
    """
    Standalone dialog for expert queries only.
    Clinicians focus on answering questions; no structure input.
    """

    def __init__(
        self,
        poset_builder: PosetBuilder,
        autosave_path: Path,
        axis: str,
        save_callback: Callable[[str, List[Structure], Set[Tuple[int, int]]], None],
    ) -> None:
        super().__init__()
        self.setWindowTitle("Expert Query")
        self.resize(1100, 580)
        self.setModal(False)

        self.poset_builder = poset_builder
        self._autosave_path = autosave_path
        self._axis = axis
        self._save_callback = save_callback
        self.pending_pair: Tuple[int, int] | None = None
        # answer is True (Yes), False (No), or None ("Not sure"/skipped)
        self._answer_history: List[Tuple[int, int, Optional[bool]]] = []

        # For vertical axis, detect bilateral (Left/Right) cores to combine in question text
        self._bilateral_cores: Set[str] = set()
        # For vertical axis, store combined CoM for bilateral cores (mean of Left/Right)
        self._bilateral_core_com_vertical: Dict[str, float] = {}
        if self._axis == AXIS_VERTICAL:
            core_counts: Dict[str, int] = {}
            names = [s.name.strip() for s in self.poset_builder.structures]
            for name in names:
                if name.startswith("Left "):
                    core = name[5:].strip()
                elif name.startswith("Right "):
                    core = name[6:].strip()
                else:
                    continue
                core_counts[core] = core_counts.get(core, 0) + 1
            self._bilateral_cores = {c for c, cnt in core_counts.items() if cnt >= 2}
            # Precompute vertical CoM for bilateral cores as the mean of their sides.
            core_to_values: Dict[str, List[float]] = {}
            for s in self.poset_builder.structures:
                nm = s.name.strip()
                if nm.startswith("Left "):
                    core = nm[5:].strip()
                elif nm.startswith("Right "):
                    core = nm[6:].strip()
                else:
                    continue
                if core in self._bilateral_cores:
                    core_to_values.setdefault(core, []).append(s.com_vertical)
            for core, vals in core_to_values.items():
                if vals:
                    self._bilateral_core_com_vertical[core] = sum(vals) / len(vals)

        main_layout = QHBoxLayout(self)

        # Middle column: questions (will be inserted between anatomy and coronal panels)
        left_col = QVBoxLayout()

        # Segmentation classes overview image above the question
        overview_label = ClickableImageLabel("Segmentation classes overview — full view", self)
        overview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overview_label.setFixedHeight(220)
        overview_label.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 8px; background: #ffffff; padding: 4px; margin: 8px 0 4px 0;"
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
        left_col.addWidget(overview_label)

        overview_link = QLabel(
            '<a href="https://github.com/wasserth/TotalSegmentator/blob/master/resources/imgs/overview_classes_v2.png">'
            "Source: TotalSegmentator overview classes v2</a>",
            self,
        )
        overview_link.setOpenExternalLinks(True)
        overview_link.setAlignment(Qt.AlignmentFlag.AlignCenter)
        overview_link.setStyleSheet("color: #0a66c2; font-size: 11px; margin-bottom: 4px;")
        left_col.addWidget(overview_link)

        # Question card (no border)
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
        left_col.addWidget(self.question_card)

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
        left_col.addLayout(btn_row)

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
        left_col.addWidget(self.progress_bar)

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
        left_col.addWidget(self.finish_btn)

        # Right column: Anatomy Images + labels overview below
        anatomy_group = QGroupBox("Anatomy Images")
        anatomy_group.setStyleSheet(
            "QGroupBox { font-weight: 600; font-size: 13px; }"
        )
        anatomy_layout = QVBoxLayout(anatomy_group)
        anatomy_layout.addWidget(_create_anatomy_views_panel(self))

        # Give the anatomy panel a tall left column with the front/side/rear views.
        main_layout.addWidget(anatomy_group, stretch=1)

        # Imaging support: full-body volume viewer (coronal slice stacks are kept in code but hidden by default)
        coronal_group = QGroupBox("Full-body volume")
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

        main_layout.addWidget(coronal_group, stretch=2)

        # Insert the question/CoM/controls column between anatomy and coronal panels
        main_layout.insertLayout(1, left_col)

        self._advance_to_next_query()

    def _autosave_poset(self) -> None:
        if not self._autosave_path or not self._save_callback:
            return
        try:
            structures, edges = self.poset_builder.get_final_relations()
            self._save_callback(self._axis, structures, edges)
        except Exception:
            pass

    def _advance_to_next_query(self) -> None:
        pair = self.poset_builder.next_pair()
        self.pending_pair = pair
        if pair is None:
            self._autosave_poset()
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
            # If bilateral cores are merged (Left/Right), use their combined CoM.
            core_i = name_i if name_i in self._bilateral_core_com_vertical else None
            core_j = name_j if name_j in self._bilateral_core_com_vertical else None
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
            f"{name_i}: CoM {axis_label} = {ci:.1f}\n"
            f"{name_j}: CoM {axis_label} = {cj:.1f}"
        )
        self._update_progress()

    def answer_query(self, is_above: Optional[bool]) -> None:
        if self.pending_pair is None:
            return
        i, j = self.pending_pair
        self._answer_history.append((i, j, is_above))
        self.back_btn.setEnabled(True)
        if is_above is True:
            self.poset_builder.record_response(i, j, True)
        elif is_above is None:
            self.poset_builder.record_skip(i, j)
        self._autosave_poset()
        self._advance_to_next_query()

    def go_back_one_question(self) -> None:
        if not self._answer_history:
            return
        last_i, last_j, last_answer = self._answer_history.pop()
        if last_answer is True:
            self.poset_builder.edges.discard((last_i, last_j))
        elif last_answer is None:
            self.poset_builder.unskip_pair(last_i, last_j)
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
            core_i = name_i if name_i in self._bilateral_core_com_vertical else None
            core_j = name_j if name_j in self._bilateral_core_com_vertical else None
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
            f"{name_i}: CoM {axis_label} = {ci:.1f}\n"
            f"{name_j}: CoM {axis_label} = {cj:.1f}"
        )
        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        self.not_sure_btn.setEnabled(True)
        if not self._answer_history:
            self.back_btn.setEnabled(False)
        self._update_progress()
        self._autosave_poset()

    def _update_progress(self) -> None:
        asked = len(self._answer_history)
        remaining = self.poset_builder.estimate_remaining_questions()
        total = asked + remaining
        if total == 0:
            value = 0
        else:
            value = int(100 * asked / total)
        self.progress_bar.setValue(value)

    def _display_name(self, idx: int, original: str) -> str:
        if self._axis != AXIS_VERTICAL:
            return original

        name = original.strip()
        core = None
        if name.startswith("Left "):
            core = name[5:].strip()
        elif name.startswith("Right "):
            core = name[6:].strip()

        if core and core in self._bilateral_cores:
            return core
        return original