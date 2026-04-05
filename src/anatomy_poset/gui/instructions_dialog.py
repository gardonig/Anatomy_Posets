from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
)

from ..core.config import ASSETS_DIR
from ..core.axis_models import AXIS_VERTICAL
from .dialog_widgets import ClickableImageLabel


class InstructionsDialog(QDialog):
    """
    Shown before the query window. Displays the instructions (and examples for vertical).
    User presses "Proceed" to proceed to questions.
    """

    def __init__(self, axis: str = AXIS_VERTICAL) -> None:
        super().__init__()
        self.setWindowTitle("Instructions — Poset Construction")
        # Use a slightly larger window than the definition dialogs so the image
        # has more room, but still clamp to the available screen geometry.
        self.resize(1200, 800)
        self.setModal(True)
        self._axis = axis
        self.setStyleSheet("background-color: #ffffff;")

        layout = QVBoxLayout(self)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(self.width(), geom.width())
            h = min(self.height(), geom.height())
            self.resize(w, h)
            self.setMaximumSize(geom.width(), geom.height())

        # Image height: fill most of the dialog height leaving room for heading and button bar
        _win_h = self.height()
        _anatomy_img_height = max(400, _win_h - 180)

        # All text: dark on white (no grey backgrounds)
        _text_style = "color: #1a1a1a; font-size: 14px; padding: 6px 0;"

        # ---- 1. Welcome + anatomical position (side-by-side) ----
        welcome_heading = QLabel("Welcome to the Anatomical Structure Questionnaire")
        welcome_heading.setStyleSheet("color: #1a1a1a; font-weight: bold; font-size: 18px; padding: 0 0 12px 0;")
        layout.addWidget(welcome_heading)

        intro_text = (
            "Thank you for taking part. In this questionnaire you will be asked to compare pairs of "
            "anatomical structures and indicate whether one is strictly above the other (vertical axis), "
            "strictly to the left of the other (lateral axis) or strictly in front of the other (anteroposterior axis). Your answers help us build a spatial ordering for segmentation models "
            "that can be used to check and correct automatic segmentations. There are no wrong answers—we need "
            "your clinical judgement.\n\n"
            "We assume the patient is in standard anatomical position, as shown in the figure.\n\n"
            "You can click any image to open a larger view. Inside any image, use the mouse wheel to zoom "
            "and hold the right mouse button to drag (pan) while zoomed.\n\n"
            "If you have any questions, please do not hesitate to reach out to Gian or Güney."
        )
        intro_label = QLabel(intro_text)
        intro_label.setWordWrap(True)
        intro_label.setStyleSheet(_text_style)
        intro_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)

        anatomy_img = ClickableImageLabel("Anatomical axes — full view")
        anatomy_img.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignHCenter)
        anatomy_img.enable_interactive_view(True)
        anatomy_img.set_fit_scale(1.0)
        anatomy_img.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        # Use updated example figure for anatomical axes.
        anatomy_path = ASSETS_DIR / "definition_images" / "Axes_example.png"

        if anatomy_path.exists():
            anatomy_pix = QPixmap(str(anatomy_path))
            if not anatomy_pix.isNull():
                anatomy_img.set_full_pixmap(anatomy_pix)
                scaled = anatomy_pix.scaledToHeight(_anatomy_img_height, Qt.SmoothTransformation)
                anatomy_img.setPixmap(scaled)
                # Fix both dimensions so the widget footprint == the image pixels exactly.
                anatomy_img.setFixedSize(scaled.width(), scaled.height())
        if anatomy_img.pixmap() is None or anatomy_img.pixmap().isNull():
            anatomy_img.setText("[Anatomical position diagram missing]")

        # Text expands to fill the left; image sits at its natural size on the right.
        intro_row = QHBoxLayout()
        intro_row.setSpacing(12)
        intro_row.addWidget(intro_label, stretch=1)
        intro_row.addWidget(anatomy_img, stretch=0)
        intro_row.setAlignment(intro_label, Qt.AlignmentFlag.AlignTop)
        intro_row.setAlignment(anatomy_img, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(intro_row)

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
