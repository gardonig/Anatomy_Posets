from pathlib import Path
from typing import List, Optional, Set, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QFileDialog,
    QDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core.builder import PosetBuilder, MatrixBuilder
from ..core.config import INPUT_DIR, OUTPUT_DIR
from ..core.io import load_structures_from_json, load_poset_from_json, save_poset_to_json
from ..core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)
from ..core.structure_regions import (
    REGION_IDS,
    REGION_LABELS,
    query_allowed_indices_for_regions,
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
        self.resize(920, 540)

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

        # Axis + optional body-region subset (same row)
        axis_region_row = QHBoxLayout()
        axis_group = QGroupBox("Axis for This Run:")
        axis_layout = QVBoxLayout(axis_group)
        self.axis_vertical_rb = QRadioButton(
            'Vertical Axis (Top-Bottom)'
        )
        self.axis_vertical_rb.setChecked(True)
        self.axis_frontal_rb = QRadioButton(
            'Lateral Axis (Right-Left, Patient\'s View)'
        )
        self.axis_ap_rb = QRadioButton(
            'Anteroposterior Axis (Back-Front)'
        )
        axis_layout.addWidget(self.axis_vertical_rb)
        axis_layout.addWidget(self.axis_frontal_rb)
        axis_layout.addWidget(self.axis_ap_rb)
        axis_region_row.addWidget(axis_group, stretch=1)

        region_group = QGroupBox("Structures for This Run:")
        region_layout = QVBoxLayout(region_group)
        self._region_all_rb = QRadioButton("All structures in the table")
        self._region_all_rb.setChecked(True)
        self._region_subset_rb = QRadioButton("Only selected region(s)")
        region_mode = QButtonGroup(self)
        region_mode.addButton(self._region_all_rb)
        region_mode.addButton(self._region_subset_rb)
        region_layout.addWidget(self._region_all_rb)
        region_layout.addWidget(self._region_subset_rb)
        self._region_checks: dict[str, QCheckBox] = {}
        for rid in REGION_IDS:
            cb = QCheckBox(REGION_LABELS[rid])
            cb.setEnabled(False)
            self._region_checks[rid] = cb
            region_layout.addWidget(cb)

        self._region_all_rb.toggled.connect(self._on_region_mode_toggled)
        self._region_subset_rb.toggled.connect(self._on_region_mode_toggled)
        region_layout.addWidget(
            QLabel(
                "Subset: only pairs within the selected region(s) are asked; "
                "saved JSON still contains every structure (merge-safe)."
            )
        )
        axis_region_row.addWidget(region_group, stretch=1)
        left_layout.addLayout(axis_region_row)

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

        # Load structures from file: CLI arg, or default to the latest CoM/poset set
        load_path = self._input_path
        if load_path is None:
            # Prefer the normalized cleaned CoM set if present, otherwise fall back.
            normalized = INPUT_DIR / "CoM_cleaned_normalized_global_avg_xyz.json"
            cleaned = INPUT_DIR / "CoM_cleaned_global_avg_xyz.json"
            if normalized.exists():
                load_path = str(normalized)
            elif cleaned.exists():
                load_path = str(cleaned)
        if load_path is not None:
            self._autosave_path = self._builtposet_output_path(Path(load_path))
            try:
                # Try to load a full poset file (structures + edges for all axes)
                structures, ev, em, ea = load_poset_from_json(load_path)
                self._edges_vertical = ev
                self._edges_mediolateral = em
                self._edges_anteroposterior = ea
            except Exception:
                # Fall back to simple structures-only JSON
                try:
                    structures = load_structures_from_json(load_path)
                    self._edges_vertical = set()
                    self._edges_anteroposterior = set()
                    self._edges_mediolateral = set()
                except Exception as exc:  # noqa: BLE001
                    QMessageBox.warning(
                        self,
                        "Failed to load input",
                        f"Could not load structures from:\n{load_path}\n\n{exc}",
                    )
                    structures = []

            for s in structures:
                self.add_structure_row(
                    s.name,
                    str(s.com_vertical),
                    str(s.com_lateral),
                    str(s.com_anteroposterior),
                )

    def _builtposet_output_path(self, input_path: Path) -> Path:
        """Autosave goes to Output_constructed_posets folder, not the input folder."""
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        return OUTPUT_DIR / f"{input_path.stem}.poset_autosave.json"

    def _on_poset_autosave(
        self, axis: str, structures: List[Structure], matrix: List[List[int]]
    ) -> None:
        """Called by QueryDialog on autosave (after each answer/undo, on completion, or on close); updates that axis matrix and saves all axes."""
        if not self._autosave_path:
            return
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            # Lazily allocate per-axis matrices from structures length
            n = len(structures)
            def ensure_matrix(mat: List[List[int]]) -> List[List[int]]:
                if len(mat) == n and all(len(row) == n for row in mat):
                    # Ensure diagonal convention for all loaded/saved matrices.
                    for i in range(n):
                        mat[i][i] = -1
                    return mat
                out = [[-2 for _ in range(n)] for _ in range(n)]
                for i in range(n):
                    out[i][i] = -1
                return out

            if not hasattr(self, "_matrix_vertical"):
                self._matrix_vertical: List[List[int]] = ensure_matrix([])
                self._matrix_mediolateral: List[List[int]] = ensure_matrix([])
                self._matrix_anteroposterior: List[List[int]] = ensure_matrix([])
            else:
                # Keep non-selected axes intact if dimensions match; otherwise
                # reinitialize to an empty matrix for current structures.
                self._matrix_vertical = ensure_matrix(self._matrix_vertical)
                self._matrix_mediolateral = ensure_matrix(self._matrix_mediolateral)
                self._matrix_anteroposterior = ensure_matrix(self._matrix_anteroposterior)

            if axis == AXIS_VERTICAL:
                self._matrix_vertical = matrix
            elif axis == AXIS_MEDIOLATERAL:
                self._matrix_mediolateral = matrix
            else:
                self._matrix_anteroposterior = matrix
            save_poset_to_json(
                str(self._autosave_path),
                structures,
                getattr(self, "_matrix_vertical", matrix),
                getattr(self, "_matrix_mediolateral", []),
                getattr(self, "_matrix_anteroposterior", []),
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
            # Prefer a full poset file if it exists (structures + matrices per axis)
            structures, Mv, Mml, Map = load_poset_from_json(path)
            # Store matrices for downstream querying; edge sets are now derived on demand.
            self._matrix_vertical = Mv
            self._matrix_mediolateral = Mml
            self._matrix_anteroposterior = Map
        except Exception:
            try:
                structures = load_structures_from_json(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Failed to load",
                    f"Could not load structures from:\n{path}\n\n{exc}",
                )
                return

        self.table.setRowCount(0)
        for s in structures:
            self.add_structure_row(
                s.name,
                str(s.com_vertical),
                str(s.com_lateral),
                str(s.com_anteroposterior),
            )
        self._autosave_path = self._builtposet_output_path(Path(path))

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

    def _on_region_mode_toggled(self) -> None:
        use_subset = self._region_subset_rb.isChecked()
        for cb in self._region_checks.values():
            cb.setEnabled(use_subset)

    # -------- Poset construction flow -------- #
    def start_poset_construction(self) -> None:
        structures = self._collect_structures()
        if structures is None:
            return

        use_all_regions = self._region_all_rb.isChecked()
        selected_region_ids = {
            rid for rid, cb in self._region_checks.items() if cb.isChecked()
        }
        if not use_all_regions:
            if not selected_region_ids:
                QMessageBox.warning(
                    self,
                    "No region selected",
                    "Choose at least one of regions 1–3, or switch back to “All structures”.",
                )
                return

        if self.axis_vertical_rb.isChecked():
            axis = AXIS_VERTICAL
        elif self.axis_frontal_rb.isChecked():
            axis = AXIS_MEDIOLATERAL
        else:
            axis = AXIS_ANTERIOR_POSTERIOR

        self.start_btn.setEnabled(False)

        # 1) Ask where to save this query's autosave file before starting.
        # If the user points to an existing poset file, we load it and continue
        # where it left off (only the current axis matrix is overwritten on save;
        # other axes stay intact). We show an info dialog after a successful load.
        # If they choose a new path, we create a blank poset file containing all
        # structures and empty matrices for all three axes.
        # (The system file dialog may still ask to "replace" an existing path—that
        # only confirms the filename; it does not wipe the file before we load it.)
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        suggested = self._autosave_path or self._builtposet_output_path(Path("poset_autosave.json"))
        save_path_str, _ = QFileDialog.getSaveFileName(
            self,
            "Choose where to save this query's poset",
            str(suggested.parent),
            "JSON Files (*.json);;All Files (*)",
        )
        if not save_path_str:
            # User cancelled save-location selection; abort starting the query.
            self.start_btn.setEnabled(True)
            return

        self._autosave_path = Path(save_path_str)

        # If this is a brand-new output file, initialize it with all structures
        # and empty matrices for all axes so subsequent queries can fill it.
        if not self._autosave_path.exists():
            n = len(structures)
            empty = [[-2 for _ in range(n)] for _ in range(n)]
            for i in range(n):
                empty[i][i] = -1
            self._matrix_vertical = [row[:] for row in empty]
            self._matrix_mediolateral = [row[:] for row in empty]
            self._matrix_anteroposterior = [row[:] for row in empty]
            save_poset_to_json(
                str(self._autosave_path),
                structures,
                self._matrix_vertical,
                self._matrix_mediolateral,
                self._matrix_anteroposterior,
            )
        else:
            # Existing output selected: load all three axes so we only overwrite
            # the axis currently being queried and preserve the other two.
            try:
                loaded_structures, Mv, Mml, Map = load_poset_from_json(str(self._autosave_path))
                # Keep index consistency with the selected output file by using
                # its structure ordering for continuation.
                structures = loaded_structures

                self._matrix_vertical = Mv
                self._matrix_mediolateral = Mml
                self._matrix_anteroposterior = Map
                QMessageBox.information(
                    self,
                    "Continuing from saved file",
                    "You chose an existing poset file. Nothing in it is discarded: "
                    "structures and the matrices for the other axes are kept as saved. "
                    "Only the axis you are about to query will be updated when you save.\n\n"
                    "The query will pick up where this file left off for that session.",
                )
            except Exception:
                QMessageBox.warning(
                    self,
                    "Failed to load existing output",
                    "Could not read the selected output file.\n\n"
                    "To avoid deleting existing data, the query was not started.",
                )
                self.start_btn.setEnabled(True)
                return

        query_allowed = query_allowed_indices_for_regions(
            structures,
            use_all=use_all_regions,
            selected_region_ids=selected_region_ids,
        )
        if query_allowed is not None and len(query_allowed) < 2:
            QMessageBox.warning(
                self,
                "Region subset too small",
                "The selected region(s) must include at least two structures in your table "
                "so pairs can be asked. Choose more regions or use “All structures”.",
            )
            self.start_btn.setEnabled(True)
            return

        self.poset_builder = MatrixBuilder(
            structures,
            axis=axis,
            query_allowed_indices=query_allowed,
        )

        # If we have an active MatrixBuilder and preloaded matrices, continue
        # from unfinished state by injecting the selected-axis matrix.
        if isinstance(self.poset_builder, MatrixBuilder):
            if axis == AXIS_VERTICAL:
                self.poset_builder.M = [row[:] for row in self._matrix_vertical]
            elif axis == AXIS_MEDIOLATERAL:
                self.poset_builder.M = [row[:] for row in self._matrix_mediolateral]
            else:
                self.poset_builder.M = [row[:] for row in self._matrix_anteroposterior]
            self.poset_builder.finished = False
            self.poset_builder.current_gap = 1
            self.poset_builder.current_i = 0
            self.poset_builder._propagate()

        # 2) Generic welcome/instructions window (always shown)
        welcome_dialog = InstructionsDialog(axis=axis)
        if welcome_dialog.exec() != QDialog.DialogCode.Accepted:
            self.start_btn.setEnabled(True)
            return

        # 3) Axis-specific definition window
        if axis == AXIS_VERTICAL:
            axis_dialog: QDialog = VerticalDefinitionDialog()
        elif axis == AXIS_MEDIOLATERAL:
            axis_dialog = MediolateralDefinitionDialog()
        else:
            axis_dialog = AnteroposteriorDefinitionDialog()
        if axis_dialog.exec() != QDialog.DialogCode.Accepted:
            self.start_btn.setEnabled(True)
            return

        # 4) Start the query dialog
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
        try:
            win = PosetViewerWindow()
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