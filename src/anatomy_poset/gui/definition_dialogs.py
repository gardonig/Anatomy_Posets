from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
)

from ..core.config import ASSETS_DIR
from .dialog_widgets import ClickableImageLabel

IMG_STYLE_DEFAULT = "margin-top: 6px; background: #ffffff;"
PLACEHOLDER_STYLE_DEFAULT = (
    "margin-top: 6px; color: #444; font-size: 13px; background: #ffffff;"
)


def _configure_definition_image_label(
    label: ClickableImageLabel,
    img_path: Path,
    target_height: int,
    placeholder_text: str,
) -> None:
    """Helper to load a definition/example image into a ClickableImageLabel."""
    if img_path.exists():
        from PySide6.QtGui import QPixmap

        pix = QPixmap(str(img_path))
        if not pix.isNull():
            label.set_full_pixmap(pix)
            label.setPixmap(
                pix.scaledToHeight(target_height, Qt.SmoothTransformation)
            )
            return
    # Fallback if missing or invalid.
    label.setText(placeholder_text)
    label.setStyleSheet(PLACEHOLDER_STYLE_DEFAULT)


class VerticalDefinitionDialog(QDialog):
    """
    Dedicated window for the vertical 'strictly above' definition and examples.
    Shown only when the vertical axis is selected; user presses 'Proceed' to start questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle('Definition — Vertical "strictly above"')
        self.resize(1200, 800)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(self.width(), geom.width())
            h = min(self.height(), geom.height())
            self.resize(w, h)
            self.setMaximumSize(geom.width(), geom.height())
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=2)

        heading = QLabel('Vertical relation: "strictly above"')
        heading.setStyleSheet(_heading_style)
        left_col.addWidget(heading)

        text1 = QLabel(
            'For each pair of structures you will answer "Yes" or "No" to:\n'
            '  "Is the first structure strictly above the second along the vertical axis?"'
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
        content.addLayout(right_col, stretch=5)

        _img_dir = ASSETS_DIR / "definition_images"
        screen = QGuiApplication.primaryScreen()
        avail_h = screen.availableGeometry().height() if screen is not None else 900
        img_height = max(720, int(avail_h * 0.90))

        img_style = IMG_STYLE_DEFAULT

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=3)

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
        no_label.enable_interactive_view(True)
        no_label.setStyleSheet(img_style)
        _configure_definition_image_label(
            no_label,
            _img_dir / "example_vert_No.png",
            2 * img_height,
            "[Add vertical No example image here]",
        )
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
        yes_label.enable_interactive_view(True)
        yes_label.setStyleSheet(img_style)
        _configure_definition_image_label(
            yes_label,
            _img_dir / "eample_vert_Yes.png",
            2 * img_height,
            "[Add vertical Yes example image here]",
        )
        yes_col.addWidget(yes_label)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        # Give the CoM panel much more space than the examples panel by default.
        examples_and_com.addLayout(com_col, stretch=3)

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
        com_img.enable_interactive_view(True)
        com_img.setStyleSheet(img_style)
        # Make the third image (CoM) much larger by default.
        com_img.set_preferred_size(2200, 4 * img_height)
        _configure_definition_image_label(
            com_img,
            _img_dir / "Vertical_CoM_numbers.png",
            4 * img_height,
            "[Add vertical CoM image here]",
        )
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
        self.resize(1200, 800)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(self.width(), geom.width())
            h = min(self.height(), geom.height())
            self.resize(w, h)
            self.setMaximumSize(geom.width(), geom.height())
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=2)

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
        content.addLayout(right_col, stretch=5)

        _img_dir = ASSETS_DIR / "definition_images"
        screen = QGuiApplication.primaryScreen()
        avail_h = screen.availableGeometry().height() if screen is not None else 900
        img_height = max(720, int(avail_h * 0.90))

        img_style = IMG_STYLE_DEFAULT

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=3)

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
        ex1_img.enable_interactive_view(True)
        ex1_img.setStyleSheet(img_style)
        _configure_definition_image_label(
            ex1_img,
            _img_dir / "example_lat_yes.png",
            2 * img_height,
            "[Add mediolateral Yes example image here]",
        )
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
        ex2_img.enable_interactive_view(True)
        ex2_img.setStyleSheet(img_style)
        _configure_definition_image_label(
            ex2_img,
            _img_dir / "example_lat_no.png",
            2 * img_height,
            "[Add mediolateral No example image here]",
        )
        no_col.addWidget(ex2_img)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=3)

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
        com_img.enable_interactive_view(True)
        com_img.setStyleSheet(img_style)
        _configure_definition_image_label(
            com_img,
            _img_dir / "Lateral_CoM_numbers.png",
            4 * img_height,
            "[Add mediolateral CoM image here]",
        )
        # Match the vertical dialog: give CoM more horizontal room so it can
        # render at full scale (otherwise the label gets squeezed width-wise).
        com_img.set_preferred_size(2200, 4 * img_height)
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
        btn_row.addStretch(2)
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
        self.resize(1200, 800)
        self.setModal(True)
        self.setStyleSheet("background-color: #ffffff;")

        main = QVBoxLayout(self)

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(self.width(), geom.width())
            h = min(self.height(), geom.height())
            self.resize(w, h)
            self.setMaximumSize(geom.width(), geom.height())
        content = QHBoxLayout()
        main.addLayout(content)

        _text_style = "color: #1a1a1a; font-size: 14px; padding: 4px 0;"
        _heading_style = "color: #1a1a1a; font-weight: bold; font-size: 16px; padding: 4px 0 4px 0;"

        # Left column: text (task, definition, question form)
        left_col = QVBoxLayout()
        content.addLayout(left_col, stretch=2)

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
        content.addLayout(right_col, stretch=5)

        img_style = IMG_STYLE_DEFAULT

        _img_dir = ASSETS_DIR / "definition_images"
        screen = QGuiApplication.primaryScreen()
        avail_h = screen.availableGeometry().height() if screen is not None else 900
        img_height = max(720, int(avail_h * 0.90))

        # Left: examples stacked vertically, Right: CoM explanation + image
        examples_and_com = QHBoxLayout()
        right_col.addLayout(examples_and_com)

        examples_col = QVBoxLayout()
        examples_col.setSpacing(16)
        examples_and_com.addLayout(examples_col, stretch=3)

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
        ex1_img.enable_interactive_view(True)
        ex1_img.setStyleSheet(img_style)
        _configure_definition_image_label(
            ex1_img,
            _img_dir / "example_ap_yes.png",
            2 * img_height,
            "[Add anteroposterior Yes example image here]",
        )
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
        ex2_img.enable_interactive_view(True)
        ex2_img.setStyleSheet(img_style)
        _configure_definition_image_label(
            ex2_img,
            _img_dir / "example_ap_no.png",
            2 * img_height,
            "[Add anteroposterior No example image here]",
        )
        no_col.addWidget(ex2_img)

        # CoM explanation + image to the right of the examples
        com_col = QVBoxLayout()
        examples_and_com.addLayout(com_col, stretch=3)

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
        com_img.enable_interactive_view(True)
        com_img.setStyleSheet(img_style)
        _configure_definition_image_label(
            com_img,
            _img_dir / "AP_CoM_numbers.png",
            4 * img_height,
            "[Add anteroposterior CoM image here]",
        )
        # Match the vertical dialog: give CoM more horizontal room so it can
        # render at full scale (otherwise the label gets squeezed width-wise).
        com_img.set_preferred_size(2200, 4 * img_height)
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
