from PySide6.QtCore import Qt, QRectF, QSize
from PySide6.QtGui import QColor, QGuiApplication, QPainter, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)


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
        self._preferred_size: QSize | None = None
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
        self._dragged = False
        self.setMouseTracking(True)

        # Important: QLabel's default sizeHint is based on the pixmap size.
        # For very tall slices (e.g. sagittal), this can make layouts expand the
        # window unexpectedly. We override size hints below to keep the label
        # responsive to the available layout space instead of the pixmap dimensions.

    def set_preferred_size(self, w: int, h: int) -> None:
        self._preferred_size = QSize(max(0, int(w)), max(0, int(h)))
        self.updateGeometry()

    def sizeHint(self) -> QSize:  # type: ignore[override]
        # A modest, stable preferred size. The label will still expand to fill
        # available space due to its size policy.
        return self._preferred_size or QSize(320, 320)

    def minimumSizeHint(self) -> QSize:  # type: ignore[override]
        # Allow shrinking as much as the layout requires.
        return QSize(0, 0)

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

                # Clamp offsets so the image cannot be dragged outside the label.
                cx_center = rect.center().x()
                cy_center = rect.center().y()
                max_dx = max(0.0, (target_w - rect.width()) / 2.0)
                max_dy = max(0.0, (target_h - rect.height()) / 2.0)
                dx = max(-max_dx, min(self._offset_x, max_dx))
                dy = max(-max_dy, min(self._offset_y, max_dy))
                cx = cx_center + dx
                cy = cy_center + dy

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
        # Right- or middle-button drag: pan the zoomed image when interactive.
        if self._interactive and event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ):
            base = self._full_pixmap or self.pixmap()
            if base is not None and not base.isNull():
                self._panning = True
                self._dragged = False
                self._last_pos = event.position()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return

        # Simple left-click opens the preview dialog.
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
            self._dragged = True
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
