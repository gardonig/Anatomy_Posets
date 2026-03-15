from pathlib import Path
from typing import List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QFileDialog,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.builder import PosetBuilder
from ..core.config import INPUT_DIR, OUTPUT_DIR
from ..core.io import load_structures_from_json, save_poset_to_json
from ..core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)
from .dialogs import (
    AnteroposteriorDefinitionDialog,
    InstructionsDialog,
    MediolateralDefinitionDialog,
    QueryDialog,
    VerticalDefinitionDialog,
)
from .viewer import PosetViewerWindow


class MainWindow(QMainWindow):
    def __init__(self, input_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Builder")
        # Initial size; will be clamped to screen geometry below.
        self.resize(520, 500)

        # Remember optional input path for use during UI setup
        self._input_path: Optional[str] = input_path
        # Where to auto-save; set after we know the actual load path in _init_ui
        self._autosave_path: Optional[Path] = OUTPUT_DIR / "poset_autosave.json"

        self.poset_builder: PosetBuilder | None = None
        self._viewer_windows: List[QWidget] = []
        self._edges_vertical: Set[Tuple[int, int]] = set()
        self._edges_mediolateral: Set[Tuple[int, int]] = set()
        self._edges_anteroposterior: Set[Tuple[int, int]] = set()

        self._init_ui()

        # Never exceed available screen size.
        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(self.width(), geom.width())
            h = min(self.height(), geom.height())
            self.resize(w, h)
            self.setMaximumSize(geom.width(), geom.height())

    def _init_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        # Structure definition only (no query UI until Start is clicked)
        left_group = QGroupBox("Anatomical Structures (Input)")
        left_layout = QVBoxLayout(left_group)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels([
            "Name",
            "CoM vertical",
            "CoM lateral (right–left)",
            "CoM anteroposterior",
        ])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)

        btn_row = QHBoxLayout()
        load_btn = QPushButton("Load Structures")
        load_btn.clicked.connect(self.load_structures_dialog)
        add_row_btn = QPushButton("+ Add Structure")
        add_row_btn.clicked.connect(self.add_structure_row)
        remove_row_btn = QPushButton("− Remove Selected")
        remove_row_btn.clicked.connect(self.remove_selected_row)
        view_btn = QPushButton("View Posets")
        view_btn.clicked.connect(self._open_viewer)
        btn_row.addWidget(load_btn)
        btn_row.addWidget(add_row_btn)
        btn_row.addWidget(remove_row_btn)
        btn_row.addWidget(view_btn)
        left_layout.addLayout(btn_row)

        # Axis choice: run vertical, lateral, or anteroposterior poset construction
        axis_group = QGroupBox("Axis for This Run:")
        axis_layout = QVBoxLayout(axis_group)
        self.axis_vertical_rb = QRadioButton(
            'Vertical Axis'
        )
        self.axis_vertical_rb.setChecked(True)
        self.axis_frontal_rb = QRadioButton(
            'Lateral Axis'
        )
        self.axis_ap_rb = QRadioButton(
            'Anteroposterior Axis'
        )
        axis_layout.addWidget(self.axis_vertical_rb)
        axis_layout.addWidget(self.axis_frontal_rb)
        axis_layout.addWidget(self.axis_ap_rb)
        left_layout.addWidget(axis_group)

        self.start_btn = QPushButton("▶  Start Poset Construction")
        self.start_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            QPushButton:disabled { background-color: #bdbdbd; color: #757575; }
            """
        )
        self.start_btn.clicked.connect(self.start_poset_construction)
        left_layout.addWidget(self.start_btn)

        root_layout.addWidget(left_group)

        self.setCentralWidget(central)

        # Load structures from file: CLI arg, or default to the latest CoM set
        load_path = self._input_path
        if load_path is None:
            # Prefer the newer CoM structures file if present.
            default_file = INPUT_DIR / ".json"
            if default_file.exists():
                load_path = str(default_file)
        if load_path is not None:
            self._autosave_path = self._builtposet_output_path(Path(load_path))
            try:
                structures = load_structures_from_json(load_path)
                self._edges_vertical = set()
                self._edges_anteroposterior = set()
                self._edges_mediolateral = set()
                for s in structures:
                    self.add_structure_row(
                        s.name,
                        str(s.com_vertical),
                        str(s.com_lateral),
                        str(s.com_anteroposterior),
                    )
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Failed to load input",
                    f"Could not load structures from:\n{load_path}\n\n{exc}",
                )

    def _builtposet_output_path(self, input_path: Path) -> Path:
        """Autosave goes to Output_constructed_posets folder, not the input folder."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR / f"{input_path.stem}.poset_autosave.json"

    def _on_poset_autosave(
        self, axis: str, structures: List[Structure], edges: Set[Tuple[int, int]]
    ) -> None:
        """Called by QueryDialog on each answer; updates the correct edge set and saves all axes."""
        if not self._autosave_path:
            return
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            if axis == AXIS_VERTICAL:
                self._edges_vertical = edges
            elif axis == AXIS_MEDIOLATERAL:
                self._edges_mediolateral = edges
            else:
                self._edges_anteroposterior = edges
            save_poset_to_json(
                str(self._autosave_path),
                structures,
                self._edges_vertical,
                self._edges_mediolateral,
                self._edges_anteroposterior,
            )
        except Exception:
            pass

    # -------- Structure table helpers -------- #
    def load_structures_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load structures from JSON",
            str(INPUT_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            structures = load_structures_from_json(path)
            self.table.setRowCount(0)
            self._edges_vertical = set()
            self._edges_mediolateral = set()
            self._edges_anteroposterior = set()
            for s in structures:
                self.add_structure_row(
                    s.name,
                    str(s.com_vertical),
                    str(s.com_lateral),
                    str(s.com_anteroposterior),
                )
            self._autosave_path = self._builtposet_output_path(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to load",
                f"Could not load structures from:\n{path}\n\n{exc}",
            )

    def add_structure_row(
        self,
        name: str = "",
        com_vertical: str = "",
        com_lateral: str = "",
        com_anteroposterior: str = "",
    ) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        self.table.setItem(row, 0, QTableWidgetItem(name))
        self.table.setItem(row, 1, QTableWidgetItem(com_vertical))
        self.table.setItem(row, 2, QTableWidgetItem(com_lateral))
        self.table.setItem(row, 3, QTableWidgetItem(com_anteroposterior))

    def remove_selected_row(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def _collect_structures(self) -> List[Structure] | None:
        structures: List[Structure] = []

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            com_v_item = self.table.item(row, 1)
            com_l_item = self.table.item(row, 2)
            com_ap_item = self.table.item(row, 3)
            if not name_item:
                continue
            name = name_item.text().strip()
            com_v_text = (com_v_item.text() if com_v_item else "").strip()
            com_l_text = (com_l_item.text() if com_l_item else "").strip()
            com_ap_text = (com_ap_item.text() if com_ap_item else "").strip()
            if not name:
                continue
            if not com_v_text:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Row {row + 1}: CoM vertical is required.",
                )
                return None
            try:
                com_vertical = float(com_v_text)
                com_lateral = float(com_l_text) if com_l_text else 0.0
                com_ap = float(com_ap_text) if com_ap_text else 0.0
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Row {row + 1}: CoM values must be numbers.",
                )
                return None
            structures.append(
                Structure(
                    name=name,
                    com_vertical=com_vertical,
                    com_lateral=com_lateral,
                    com_anteroposterior=com_ap,
                )
            )

        if not structures:
            QMessageBox.warning(
                self,
                "No Structures",
                "Please define at least one structure with a valid CoM value.",
            )
            return None

        return structures

    # -------- Poset construction flow -------- #
    def start_poset_construction(self) -> None:
        structures = self._collect_structures()
        if structures is None:
            return

        if self.axis_vertical_rb.isChecked():
            axis = AXIS_VERTICAL
        elif self.axis_frontal_rb.isChecked():
            axis = AXIS_MEDIOLATERAL
        else:
            axis = AXIS_ANTERIOR_POSTERIOR
        self.poset_builder = PosetBuilder(structures, axis=axis)
        self.start_btn.setEnabled(False)

        # 1) Generic welcome/instructions window (always shown)
        welcome_dialog = InstructionsDialog(axis=axis)
        if welcome_dialog.exec() != QDialog.DialogCode.Accepted:
            self.start_btn.setEnabled(True)
            return

        # 2) Axis-specific definition window
        if axis == AXIS_VERTICAL:
            axis_dialog: QDialog = VerticalDefinitionDialog()
        elif axis == AXIS_MEDIOLATERAL:
            axis_dialog = MediolateralDefinitionDialog()
        else:
            axis_dialog = AnteroposteriorDefinitionDialog()
        if axis_dialog.exec() != QDialog.DialogCode.Accepted:
            self.start_btn.setEnabled(True)
            return

        # 3) Start the query dialog
        query_dialog = QueryDialog(
            self.poset_builder,
            self._autosave_path,
            axis=axis,
            save_callback=self._on_poset_autosave,
        )
        query_dialog.finished.connect(lambda: self.start_btn.setEnabled(True))
        # Open the query window maximized to make best use of the display
        query_dialog.showMaximized()

    def _open_viewer(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose poset file to view",
            str(OUTPUT_DIR),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            win = PosetViewerWindow(path)
            win.setWindowFlags(Qt.Window)
            self._viewer_windows.append(win)
            win.show()
            win.raise_()
            win.activateWindow()
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to open viewer",
                f"Could not open viewer:\n{exc}",
            )