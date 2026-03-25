"""
Poset viewer GUI: inspect saved poset JSON files. Main class: ``PosetViewer`` (opened from the main
window). Shows per-axis structure lists, Hasse diagrams, matrix merge, and feedback logs.

Standalone slice/atlas UIs were removed as unused; the expert query flow uses ``FullBodyVolumePanel``
in ``query_dialog.py`` for NumPy volume browsing.
"""
from typing import Dict, List, Optional, Set, Tuple
import json
import math

import numpy as np
from pathlib import Path

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QGuiApplication, QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
    QPushButton,
    QLabel,
    QFileDialog,
    QDialog,
    QTableWidget,
    QTableWidgetItem,
    QPlainTextEdit,
    QHeaderView,
)

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import ListedColormap, BoundaryNorm

from ..core.config import OUTPUT_DIR
from ..core.io import load_poset_from_json, save_poset_to_json
from ..core.aggregation import (
    CellAggregate,
    aggregate_matrices_with_counts,
    aggregate_to_consensus_matrix,
    align_matrix_lists_to_reference,
    apply_canonical_per_axis_orders,
    cell_aggregate_to_display_matrix,
    enforce_axis_lower_triangle_inplace,
    reindex_matrix_to_structure_order,
)
from ..core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)
# Removed (unused in app): CoronalSliceViewer — PNG coronal stacks; QueryDialog uses
# FullBodyVolumePanel (query_dialog.py) for NumPy volume slices. StructureViewsWindow and
# FullBodyVolumeViewer at EOF were also removed (nothing opened them).

class HasseDiagramView(QGraphicsView):
    """
    Interactive drawing surface for the Hasse diagram:
    - circular nodes containing structure names
    - straight edges for cover relations
    - mouse wheel zoom, drag to pan
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self.setRenderHints(
            self.renderHints()
            | QPainter.Antialiasing
            | QPainter.TextAntialiasing
        )
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)

    def clear(self) -> None:
        self._scene.clear()

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        """
        Zoom in/out with the mouse wheel, centered under the cursor.
        """
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor

        if event.angleDelta().y() > 0:
            factor = zoom_in_factor
        else:
            factor = zoom_out_factor

        self.scale(factor, factor)

    def draw_diagram(
        self,
        structures: List[Structure],
        edges: Set[Tuple[int, int]],
        axis: str = "vertical",
        unsure_edges: Optional[Set[Tuple[int, int]]] = None,
    ) -> None:
        """
        Lay out nodes and draw edges. Vertical axis: levels (top to bottom), x spread by index.
        Frontal axis: y by level, x by com_lateral so left is left of spine is left of right.

        ``edges`` are solid cover relations from strict ``+1`` matrix entries (Hasse reduction).
        ``unsure_edges`` is ignored (kept for API compatibility); the diagram shows only ``+1``.
        """
        self.clear()

        n = len(structures)
        if n == 0:
            return

        # Build adjacency and indegree for level computation
        adj: Dict[int, List[int]] = {i: [] for i in range(n)}
        indeg: Dict[int, int] = {i: 0 for i in range(n)}
        for u, v in edges:
            adj[u].append(v)
            indeg[v] += 1

        # Longest-path style levels from sources (indeg == 0)
        levels: Dict[int, int] = {i: 0 for i in range(n)}

        from collections import deque

        q: deque[int] = deque(i for i in range(n) if indeg[i] == 0)
        seen: Set[int] = set(q)
        while q:
            u = q.popleft()
            for v in adj[u]:
                if levels[v] < levels[u] + 1:
                    levels[v] = levels[u] + 1
                if v not in seen:
                    seen.add(v)
                    q.append(v)

        level_nodes: Dict[int, List[int]] = {}
        max_level = 0
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)
            if lvl > max_level:
                max_level = lvl

        node_radius = 35.0
        h_spacing = 140.0
        v_spacing = 140.0

        positions: Dict[int, QPointF] = {}

        # Layout like a standard layered Hasse diagram for both axes:
        # - y coordinate encodes level
        # - x coordinate spreads nodes within each level
        # The axis only affects which relation is encoded in the edges; the geometry is identical.
        for lvl in range(0, max_level + 1):
            nodes_at_level = level_nodes.get(lvl, [])
            if not nodes_at_level:
                continue
            nodes_sorted = sorted(nodes_at_level)
            count = len(nodes_sorted)
            total_width = (count - 1) * h_spacing
            start_x = -total_width / 2.0
            y = lvl * v_spacing
            for idx, node in enumerate(nodes_sorted):
                x = start_x + idx * h_spacing
                positions[node] = QPointF(x, y)

        # Draw directed edges (with arrowheads) first so they appear behind nodes
        edge_pen = QPen(QColor(80, 80, 80))
        edge_pen.setWidth(2)
        arrow_pen = QPen(QColor(80, 80, 80))
        arrow_pen.setWidth(2)
        arrow_size = 12.0

        for u, v in edges:
            p1 = positions.get(u)
            p2 = positions.get(v)
            if p1 is None or p2 is None:
                continue

            # Direction from u -> v
            dx = p2.x() - p1.x()
            dy = p2.y() - p1.y()
            length = math.hypot(dx, dy)
            if length == 0:
                continue

            ux = dx / length
            uy = dy / length

            # Shorten the line so it touches the node borders instead of their centres
            start_x = p1.x() + ux * node_radius
            start_y = p1.y() + uy * node_radius
            end_x = p2.x() - ux * node_radius
            end_y = p2.y() - uy * node_radius

            # Main edge line
            self._scene.addLine(start_x, start_y, end_x, end_y, edge_pen)

            # Arrowhead at the target node (pointing into v)
            angle = math.atan2(dy, dx)
            left_angle = angle - math.radians(25)
            right_angle = angle + math.radians(25)

            left_x = end_x - arrow_size * math.cos(left_angle)
            left_y = end_y - arrow_size * math.sin(left_angle)
            right_x = end_x - arrow_size * math.cos(right_angle)
            right_y = end_y - arrow_size * math.sin(right_angle)

            self._scene.addLine(end_x, end_y, left_x, left_y, arrow_pen)
            self._scene.addLine(end_x, end_y, right_x, right_y, arrow_pen)

        # Draw nodes with labels
        node_brush = QBrush(QColor(84, 160, 255))
        node_pen = QPen(QColor(20, 60, 120))
        node_pen.setWidth(2)

        for idx, structure in enumerate(structures):
            pos = positions.get(idx, QPointF(0.0, 0.0))

            # Circle centered at (pos.x, pos.y)
            ellipse = QGraphicsEllipseItem(
                -node_radius,
                -node_radius,
                2 * node_radius,
                2 * node_radius,
            )
            ellipse.setBrush(node_brush)
            ellipse.setPen(node_pen)
            ellipse.setPos(pos)
            self._scene.addItem(ellipse)

            # Label centered within the node
            label_item = QGraphicsTextItem(structure.name)
            label_item.setDefaultTextColor(QColor(255, 255, 255))
            br = label_item.boundingRect()
            label_item.setPos(
                pos.x() - br.width() / 2.0,
                pos.y() - br.height() / 2.0,
            )
            self._scene.addItem(label_item)

        # Fit everything in view initially
        self.fitInView(self._scene.itemsBoundingRect(), Qt.KeepAspectRatio)


class PosetViewer(QWidget):
    """Saved poset inspector: lists + Hasse diagram per axis (vertical / mediolateral / anteroposterior)."""

    def __init__(self, poset_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Poset viewer")
        self.resize(900, 600)

        self._path: str = ""
        self._structures: List[Structure] = []
        # When merged, each axis matrix may use a different CoM sort; labels per tab:
        self._structures_ml: Optional[List[Structure]] = None
        self._structures_ap: Optional[List[Structure]] = None
        self._M_vertical: List[List[int]] = []
        self._M_mediolateral: List[List[int]] = []
        self._M_anteroposterior: List[List[int]] = []
        self._merged_mode: bool = False
        self._merge_k: int = 0
        self._agg_vertical: Optional[List[List[CellAggregate]]] = None
        self._agg_mediolateral: Optional[List[List[CellAggregate]]] = None
        self._agg_anteroposterior: Optional[List[List[CellAggregate]]] = None
        self._matrix_windows: List[QDialog] = []

        self._tabs = QTabWidget()
        root = QVBoxLayout(self)

        toolbar = QHBoxLayout()
        self._open_btn = QPushButton("Open JSON…")
        self._open_btn.setToolTip("Load a single saved poset JSON file.")
        self._open_btn.clicked.connect(self._open_json_file)
        toolbar.addWidget(self._open_btn)
        self._merge_btn = QPushButton("Merge JSON files…")
        self._merge_btn.setToolTip(
            "Select multiple poset JSONs with identical structure lists; aggregate matrices with per-cell counts."
        )
        self._merge_btn.clicked.connect(self._merge_json_files)
        toolbar.addWidget(self._merge_btn)
        self._feedback_btn = QPushButton("View Feedback…")
        self._feedback_btn.setToolTip(
            "Open and inspect a feedback .jsonl log created during expert queries."
        )
        self._feedback_btn.clicked.connect(self._open_feedback_file)
        toolbar.addWidget(self._feedback_btn)
        self._save_merged_btn = QPushButton("Save merged consensus…")
        self._save_merged_btn.setToolTip(
            "Export consensus tri-valued matrices (majority vote per cell) to one JSON file. "
            "Structures are in vertical CoM order; ML/AP matrices are reindexed to match."
        )
        self._save_merged_btn.setEnabled(False)
        self._save_merged_btn.clicked.connect(self._save_merged_consensus_json)
        toolbar.addWidget(self._save_merged_btn)
        toolbar.addStretch(1)
        self._status_label = QLabel("")
        self._status_label.setStyleSheet("color: #555;")
        toolbar.addWidget(self._status_label)
        root.addLayout(toolbar)
        root.addWidget(self._tabs)

        if poset_path:
            self._load_from_path(poset_path)
        else:
            self._rebuild_tabs()

        screen = QGuiApplication.primaryScreen()
        if screen is not None:
            geom = screen.availableGeometry()
            w = min(self.width(), geom.width())
            h = min(self.height(), geom.height())
            self.resize(w, h)
            self.setMaximumSize(geom.width(), geom.height())

    def _clear_tabs(self) -> None:
        while self._tabs.count():
            w = self._tabs.widget(0)
            self._tabs.removeTab(0)
            if w is not None:
                w.deleteLater()

    def _update_title_and_status(self) -> None:
        if self._merged_mode:
            self.setWindowTitle("Poset viewer — merged")
            self._status_label.setText(f"Merged: K = {self._merge_k} file(s)")
        elif self._path:
            self.setWindowTitle(f"Poset viewer — {Path(self._path).name}")
            self._status_label.setText(str(Path(self._path).resolve()))
        else:
            self.setWindowTitle("Poset viewer")
            self._status_label.setText("No file loaded")

    def _open_json_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open poset JSON",
            str(OUTPUT_DIR),
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        self._load_from_path(path)

    def _open_feedback_file(self) -> None:
        start_dir = str(Path(self._path).resolve().parent) if self._path else str(OUTPUT_DIR)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open feedback log",
            start_dir,
            "Feedback Logs (*.jsonl);;All Files (*)",
        )
        if not path:
            return
        self._show_feedback_log(path)

    def _show_feedback_log(self, path: str) -> None:
        entries: List[Dict[str, str]] = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line_no, raw in enumerate(f, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        entries.append(
                            {
                                "axis": "",
                                "answer": "",
                                "question": f"[Invalid JSON at line {line_no}]",
                                "feedback": line,
                            }
                        )
                        continue
                    entries.append(
                        {
                            "axis": str(obj.get("axis", "")),
                            "answer": str(obj.get("answer", "")),
                            "question": str(obj.get("question", "")),
                            "feedback": str(obj.get("feedback", "")),
                        }
                    )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "Open feedback log", f"Could not read:\n{path}\n\n{exc}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Feedback Log — {Path(path).name}")
        dlg.resize(1050, 680)
        dlg.setModal(False)
        root = QVBoxLayout(dlg)

        info = QLabel(f"{len(entries)} entr{'y' if len(entries) == 1 else 'ies'} from {path}")
        info.setStyleSheet("color: #555;")
        root.addWidget(info)

        split = QHBoxLayout()
        root.addLayout(split, stretch=1)

        table = QTableWidget(len(entries), 4, dlg)
        table.setHorizontalHeaderLabels(["Axis", "Question", "Answer", "Feedback"])
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setAlternatingRowColors(True)
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)

        def _short(s: str, n: int = 90) -> str:
            return s if len(s) <= n else s[: n - 1] + "…"

        for i, e in enumerate(entries):
            table.setItem(i, 0, QTableWidgetItem(e["axis"]))
            table.setItem(i, 1, QTableWidgetItem(_short(e["question"])))
            table.setItem(i, 2, QTableWidgetItem(e["answer"]))
            table.setItem(i, 3, QTableWidgetItem(_short(e["feedback"])))

        detail = QPlainTextEdit(dlg)
        detail.setReadOnly(True)
        detail.setPlaceholderText("Select a row to view full question + feedback.")

        def _render_row(row: int) -> None:
            if row < 0 or row >= len(entries):
                detail.clear()
                return
            e = entries[row]
            detail.setPlainText(
                "\n".join(
                    [
                        f"Axis: {e['axis']}",
                        "Question:",
                        e["question"],
                        "",
                        f"Answer: {e['answer']}",
                        "",
                        "Feedback:",
                        e["feedback"],
                    ]
                )
            )

        table.currentCellChanged.connect(lambda cur, *_: _render_row(cur))
        if entries:
            table.selectRow(0)
            _render_row(0)

        split.addWidget(table, stretch=3)
        split.addWidget(detail, stretch=2)
        dlg.show()

    def _load_from_path(self, path: str) -> None:
        try:
            (
                self._structures,
                self._M_vertical,
                self._M_mediolateral,
                self._M_anteroposterior,
            ) = load_poset_from_json(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Failed to Load",
                f"Could not load poset from:\n{path}\n\n{exc}",
            )
            return
        self._path = path
        self._merged_mode = False
        self._merge_k = 0
        self._agg_vertical = None
        self._agg_mediolateral = None
        self._agg_anteroposterior = None
        self._structures_ml = None
        self._structures_ap = None
        self._rebuild_tabs()

    def _save_merged_consensus_json(self) -> None:
        """Write consensus {-2,-1,0,+1} matrices to JSON; loadable like a normal poset file."""
        if not self._merged_mode or not self._structures:
            QMessageBox.information(
                self,
                "Save merged",
                "Merge at least two JSON files first; then you can save the consensus.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save merged consensus poset",
            str(OUTPUT_DIR / "merged_consensus.json"),
            "JSON (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            s_ml = self._structures_ml if self._structures_ml is not None else self._structures
            s_ap = self._structures_ap if self._structures_ap is not None else self._structures
            ml_save = reindex_matrix_to_structure_order(
                self._structures, s_ml, self._M_mediolateral
            )
            ap_save = reindex_matrix_to_structure_order(
                self._structures, s_ap, self._M_anteroposterior
            )
            save_poset_to_json(
                path,
                self._structures,
                self._M_vertical,
                ml_save,
                ap_save,
                extra={
                    "merged_consensus": True,
                    "merged_from_raters": self._merge_k,
                    "merged_source_files": self._path,
                },
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Save failed",
                f"Could not save:\n{exc}",
            )
            return
        QMessageBox.information(self, "Saved", f"Merged consensus saved to:\n{path}")

    def _structures_for_tab(self, axis: str) -> List[Structure]:
        """Row/column label order for this axis (merged: per-axis CoM sort; else shared list)."""
        if axis == "Vertical":
            return self._structures
        if axis == "Lateral":
            return self._structures_ml if self._structures_ml is not None else self._structures
        if axis == "Anteroposterior":
            return self._structures_ap if self._structures_ap is not None else self._structures
        return self._structures

    def _merge_json_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select JSON poset files to merge",
            str(OUTPUT_DIR),
            "JSON (*.json);;All Files (*)",
        )
        if len(paths) < 2:
            QMessageBox.information(
                self,
                "Merge",
                "Select at least two JSON files with identical structure lists.",
            )
            return
        structures_list: List[List[Structure]] = []
        mv_list: List[List[List[int]]] = []
        ml_list: List[List[List[int]]] = []
        ap_list: List[List[List[int]]] = []
        for path in paths:
            try:
                st, mv, ml, ap = load_poset_from_json(path)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Merge",
                    f"Could not load:\n{path}\n\n{exc}",
                )
                return
            structures_list.append(st)
            mv_list.append(mv)
            ml_list.append(ml)
            ap_list.append(ap)
        ok, msg, mv_list, ml_list, ap_list = align_matrix_lists_to_reference(
            structures_list, mv_list, ml_list, ap_list
        )
        if not ok:
            QMessageBox.warning(self, "Incompatible posets", msg)
            return
        # Each axis matrix uses indices sorted by THAT axis's CoM descending (MatrixBuilder).
        (
            self._structures,
            self._structures_ml,
            self._structures_ap,
            mv_list,
            ml_list,
            ap_list,
        ) = apply_canonical_per_axis_orders(structures_list[0], mv_list, ml_list, ap_list)
        self._agg_vertical, k1 = aggregate_matrices_with_counts(mv_list)
        self._agg_mediolateral, k2 = aggregate_matrices_with_counts(ml_list)
        self._agg_anteroposterior, k3 = aggregate_matrices_with_counts(ap_list)
        if not (k1 == k2 == k3):
            QMessageBox.warning(
                self,
                "Merge",
                "Internal error: axis matrix lists had different lengths.",
            )
            return
        self._merge_k = k1
        self._M_vertical = aggregate_to_consensus_matrix(self._agg_vertical)
        enforce_axis_lower_triangle_inplace(self._M_vertical)
        self._M_mediolateral = aggregate_to_consensus_matrix(self._agg_mediolateral)
        enforce_axis_lower_triangle_inplace(self._M_mediolateral)
        self._M_anteroposterior = aggregate_to_consensus_matrix(self._agg_anteroposterior)
        enforce_axis_lower_triangle_inplace(self._M_anteroposterior)
        self._merged_mode = True
        self._path = " + ".join(Path(p).name for p in paths)
        self._rebuild_tabs()

    def _unsure_edges_from_matrix(self, M: List[List[int]]) -> Set[Tuple[int, int]]:
        """Directed pairs with value 0 (not sure)."""
        out: Set[Tuple[int, int]] = set()
        n = len(M)
        for i in range(n):
            row = M[i]
            for j in range(min(n, len(row))):
                if i != j and row[j] == 0:
                    out.add((i, j))
        return out

    def _matrix_to_edges(self, M: List[List[int]]) -> Set[Tuple[int, int]]:
        """Derive pDAG edges (all +1 entries) from a tri-valued matrix."""
        edges: Set[Tuple[int, int]] = set()
        n = len(M)
        for i in range(n):
            row = M[i]
            for j in range(min(n, len(row))):
                if i != j and row[j] == 1:
                    edges.add((i, j))
        return edges

    def _matrix_summary_counts(self, M: List[List[int]]) -> Tuple[int, int, int, int]:
        """Return counts of (+1, 0, -1, -2) entries (off-diagonal only)."""
        yes = no = unsure = not_asked = 0
        n = len(M)
        for i in range(n):
            row = M[i]
            for j in range(min(n, len(row))):
                if i == j:
                    continue
                v = row[j]
                if v == 1:
                    yes += 1
                elif v == 0:
                    unsure += 1
                elif v == -1:
                    no += 1
                else:
                    not_asked += 1
        return yes, unsure, no, not_asked

    def _fill_tab(
        self,
        list_widget: QListWidget,
        hasse_view: HasseDiagramView,
        structures: List[Structure],
        M: List[List[int]],
        axis_label: str,
        relation_label: str,
    ) -> None:
        list_widget.clear()
        if self._merged_mode:
            list_widget.addItem(f"Merged from {self._merge_k} file(s):")
            list_widget.addItem(f"  {self._path}")
            list_widget.addItem(
                "Consensus matrix: majority vote per cell; ties broken by rounding the mean."
            )
        else:
            list_widget.addItem(f"Loaded from: {self._path or '(none)'}")
        list_widget.addItem("")
        list_widget.addItem(f"Structures ({axis_label}):")
        for idx, s in enumerate(structures):
            if axis_label == "Vertical":
                list_widget.addItem(f"  {idx}: {s.name} (CoM vertical = {s.com_vertical})")
            elif axis_label == "Lateral":
                list_widget.addItem(f"  {idx}: {s.name} (CoM lateral = {s.com_lateral})")
            else:  # Anteroposterior
                list_widget.addItem(
                    f"  {idx}: {s.name} (CoM anteroposterior = {s.com_anteroposterior})"
                )
        list_widget.addItem("")
        if not M:
            list_widget.addItem("No matrix data available for this axis.")
            edges: Set[Tuple[int, int]] = set()
        else:
            # Summarize matrix state
            yes, unsure, no, not_asked = self._matrix_summary_counts(M)
            list_widget.addItem("Matrix summary (off-diagonal entries):")
            list_widget.addItem(f"  +1 (YES / above): {yes}")
            list_widget.addItem(f"   0 (not sure):   {unsure}")
            list_widget.addItem(f"  -1 (NO / not-above): {no}")
            list_widget.addItem(f"  -2 (not asked yet): {not_asked}")
            list_widget.addItem("")

            # Derive pDAG (= all +1 edges) and then Hasse edges via transitive reduction.
            from anatomy_poset.core.builder import MatrixBuilder

            if axis_label == "Vertical":
                axis_key = AXIS_VERTICAL
            elif axis_label == "Lateral":
                axis_key = AXIS_MEDIOLATERAL
            else:
                axis_key = AXIS_ANTERIOR_POSTERIOR

            mb = MatrixBuilder(structures, axis=axis_key)
            mb.M = [row[:] for row in M]  # shallow copy
            mb._propagate()  # ensure closure for +1 before reduction
            try:
                hasse_edges = mb.get_hasse()
            except Exception:
                # If cyclic or invalid, fall back to raw pDAG edges.
                hasse_edges = mb.get_pdag()

            edges = hasse_edges

            if not edges:
                list_widget.addItem(f"No strict '{relation_label}' relations derived from matrix.")
            else:
                list_widget.addItem("Cover relations (Hasse / reduced +1 edges):")
                for u, v in sorted(edges):
                    su, sv = structures[u], structures[v]
                    list_widget.addItem(f"{su.name}  ≻  {sv.name}")
            n_unsure = len(self._unsure_edges_from_matrix(M))
            if n_unsure:
                list_widget.addItem("")
                list_widget.addItem(
                    f"Unsure (0) directed pairs in matrix: {n_unsure} (Hasse diagram shows +1 edges only)."
                )

        if axis_label == "Vertical":
            axis = AXIS_VERTICAL
        elif axis_label == "Lateral":
            axis = AXIS_MEDIOLATERAL
        else:
            axis = AXIS_ANTERIOR_POSTERIOR
        hasse_view.draw_diagram(structures, edges, axis=axis)

    def _show_discrete_matrix(
        self, M: List[List[int]], title: str, label_structures: List[Structure]
    ) -> None:
        """Tri-valued matrix: discrete levels {-2,-1,0,+1}."""
        if not M:
            QMessageBox.information(self, "No data", f"No matrix data available for {title}.")
            return
        n = len(M)
        dlg = QDialog(None)
        dlg.setWindowTitle(f"{title} — relation matrix")
        dlg.resize(800, 600)
        dlg.setModal(False)
        layout = QVBoxLayout(dlg)

        arr = np.full((n, n), -2, dtype=int)
        for i in range(n):
            row = M[i]
            for j in range(min(n, len(row))):
                arr[i, j] = int(row[j])

        levels = [-2, -1, 0, 1]
        colors = [
            "#f0f0f0",
            "#d73027",
            "#fc8d59",
            "#1a9850",
        ]
        cmap = ListedColormap(colors)
        norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5], cmap.N)

        fig = Figure(figsize=(6, 6), tight_layout=True)
        canvas = FigureCanvas(fig)
        try:
            from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

            toolbar = NavigationToolbar2QT(canvas, dlg)
            layout.addWidget(toolbar)
        except Exception:
            pass

        ax = fig.add_subplot(111)
        im = ax.imshow(arr, cmap=cmap, norm=norm, origin="upper")

        if len(label_structures) == n:
            labels = [s.name for s in label_structures]
        else:
            labels = [str(i) for i in range(n)]

        ax.set_title(title)
        ax.set_xlabel("j (target structure)")
        ax.set_ylabel("i (source structure)")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_yticklabels(labels, fontsize=6)

        cbar = fig.colorbar(im, ax=ax, ticks=levels)
        cbar.ax.set_yticklabels(["-2 not asked", "-1 no", "0 unsure", "+1 yes"])

        layout.addWidget(canvas, stretch=1)
        self._matrix_windows.append(dlg)
        dlg.destroyed.connect(
            lambda *_: self._matrix_windows.remove(dlg) if dlg in self._matrix_windows else None
        )
        dlg.show()

    def _show_agg_matrix(
        self, agg: List[List[CellAggregate]], title: str, label_structures: List[Structure]
    ) -> None:
        """
        Merged matrices: color = P(yes) = (μ+1)/2 per cell over answered raters (ignores -2).

        After CoM sort + seal, lower triangle (j < i) is -1 for every rater → μ=-1 → P=0 (red),
        so 0.5 (orange) can only appear in the strict upper triangle (j > i), not in full rows/columns.
        """
        # Mean-based display: P(yes)=(μ+1)/2 over answered codes only (−2 excluded per rater).
        mk = self._merge_k if self._merge_k > 0 else None
        Z, ann, _ = cell_aggregate_to_display_matrix(agg, color_mode="mean", merge_k=mk)
        n = len(Z)
        Zarr = np.asarray(Z, dtype=float)
        Z_masked = np.ma.masked_invalid(Zarr)

        dlg = QDialog(None)
        dlg.setWindowTitle(f"{title} — merged P(yes) heatmap")
        dlg.resize(900, 700)
        dlg.setModal(False)
        layout = QVBoxLayout(dlg)

        fig = Figure(figsize=(7, 6), tight_layout=True)
        canvas = FigureCanvas(fig)
        try:
            from matplotlib.backends.backend_qt5 import NavigationToolbar2QT

            layout.addWidget(NavigationToolbar2QT(canvas, dlg))
        except Exception:
            pass

        ax = fig.add_subplot(111)
        try:
            from matplotlib import colormaps

            cmap = colormaps["RdYlGn"].copy()
        except Exception:
            from matplotlib import pyplot as plt

            cmap = plt.cm.get_cmap("RdYlGn").copy()
        cmap.set_bad("#dddddd")
        im = ax.imshow(Z_masked, cmap=cmap, vmin=0.0, vmax=1.0, origin="upper")

        if len(label_structures) == n:
            labels = [s.name for s in label_structures]
        else:
            labels = [str(i) for i in range(n)]

        ax.set_xlabel("j (target structure)")
        ax.set_ylabel("i (source structure)")
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, fontsize=6, rotation=90)
        ax.set_yticklabels(labels, fontsize=6)

        if n <= 10:
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    ax.text(
                        j,
                        i,
                        ann[i][j],
                        ha="center",
                        va="center",
                        fontsize=4,
                        color="black",
                        zorder=4,
                    )

        cbar = fig.colorbar(im, ax=ax)
        cbar.set_label("P(yes) = (mean + 1) / 2 over answered raters (−2 excluded)")

        layout.addWidget(canvas, stretch=1)
        self._matrix_windows.append(dlg)
        dlg.destroyed.connect(
            lambda *_: self._matrix_windows.remove(dlg) if dlg in self._matrix_windows else None
        )
        dlg.show()

    def _rebuild_tabs(self) -> None:
        self._clear_tabs()
        self._update_title_and_status()
        self._save_merged_btn.setEnabled(bool(self._merged_mode and self._structures))

        if not self._structures:
            welcome = QWidget()
            wlay = QVBoxLayout(welcome)
            hint = QLabel(
                "Use “Open JSON…” to load a saved poset, or “Merge JSON files…” to combine\n"
                "multiple raters. Merge requires identical structure order, names, and CoM values."
            )
            hint.setWordWrap(True)
            wlay.addWidget(hint)
            wlay.addStretch(1)
            self._tabs.addTab(welcome, "Welcome")
            return

        # Tab: Vertical (top–bottom)
        vert_widget = QWidget()
        vert_layout = QHBoxLayout(vert_widget)
        vert_list_group = QGroupBox("Poset Data")
        vert_list_layout = QVBoxLayout(vert_list_group)
        vert_list = QListWidget()
        vert_list_layout.addWidget(vert_list)
        vert_diagram_group = QGroupBox("Hasse diagram (+1 strict-above relations only)")
        vert_diagram_layout = QVBoxLayout(vert_diagram_group)
        vert_hasse = HasseDiagramView()
        vert_hasse.setMinimumSize(400, 300)
        vert_diagram_layout.addWidget(vert_hasse)
        # Matrix view button
        vert_matrix_btn = QPushButton("Show matrix…")
        vert_matrix_btn.setToolTip(
            "Tri-valued matrix (single file) or probability heatmap (merged)."
        )
        vert_diagram_layout.addWidget(vert_matrix_btn)
        vert_layout.addWidget(vert_list_group, stretch=1)
        vert_layout.addWidget(vert_diagram_group, stretch=2)
        self._fill_tab(
            vert_list,
            vert_hasse,
            self._structures_for_tab("Vertical"),
            self._M_vertical,
            "Vertical",
            "above",
        )
        self._tabs.addTab(vert_widget, "Vertical (top–bottom)")

        # Tab: Lateral (right–left)
        ml_widget = QWidget()
        ml_layout = QHBoxLayout(ml_widget)
        ml_list_group = QGroupBox("Poset Data")
        ml_list_layout = QVBoxLayout(ml_list_group)
        ml_list = QListWidget()
        ml_list_layout.addWidget(ml_list)
        ml_diagram_group = QGroupBox("Hasse diagram (+1 strict-above relations only)")
        ml_diagram_layout = QVBoxLayout(ml_diagram_group)
        ml_hasse = HasseDiagramView()
        ml_hasse.setMinimumSize(400, 300)
        ml_diagram_layout.addWidget(ml_hasse)
        ml_matrix_btn = QPushButton("Show matrix…")
        ml_matrix_btn.setToolTip(
            "Tri-valued matrix (single file) or probability heatmap (merged)."
        )
        ml_diagram_layout.addWidget(ml_matrix_btn)
        ml_layout.addWidget(ml_list_group, stretch=1)
        ml_layout.addWidget(ml_diagram_group, stretch=2)
        self._fill_tab(
            ml_list,
            ml_hasse,
            self._structures_for_tab("Lateral"),
            self._M_mediolateral,
            "Lateral",
            "to the left of",
        )
        self._tabs.addTab(ml_widget, "Lateral (right–left, patient's view)")

        # Tab: Anteroposterior (front–back)
        ap_widget = QWidget()
        ap_layout = QHBoxLayout(ap_widget)
        ap_list_group = QGroupBox("Poset Data")
        ap_list_layout = QVBoxLayout(ap_list_group)
        ap_list = QListWidget()
        ap_list_layout.addWidget(ap_list)
        ap_diagram_group = QGroupBox("Hasse diagram (+1 strict-above relations only)")
        ap_diagram_layout = QVBoxLayout(ap_diagram_group)
        ap_hasse = HasseDiagramView()
        ap_hasse.setMinimumSize(400, 300)
        ap_diagram_layout.addWidget(ap_hasse)
        ap_matrix_btn = QPushButton("Show matrix…")
        ap_matrix_btn.setToolTip(
            "Tri-valued matrix (single file) or probability heatmap (merged)."
        )
        ap_diagram_layout.addWidget(ap_matrix_btn)
        ap_layout.addWidget(ap_list_group, stretch=1)
        ap_layout.addWidget(ap_diagram_group, stretch=2)
        self._fill_tab(
            ap_list,
            ap_hasse,
            self._structures_for_tab("Anteroposterior"),
            self._M_anteroposterior,
            "Anteroposterior",
            "in front of",
        )
        self._tabs.addTab(ap_widget, "Anteroposterior (front–back)")

        # Wire matrix buttons (merged → probability heatmap; single file → discrete tri-values)
        def _vert_matrix() -> None:
            labs = self._structures_for_tab("Vertical")
            if self._merged_mode and self._agg_vertical is not None:
                self._show_agg_matrix(self._agg_vertical, "Vertical axis", labs)
            else:
                self._show_discrete_matrix(self._M_vertical, "Vertical axis", labs)

        def _ml_matrix() -> None:
            labs = self._structures_for_tab("Lateral")
            if self._merged_mode and self._agg_mediolateral is not None:
                self._show_agg_matrix(self._agg_mediolateral, "Lateral axis", labs)
            else:
                self._show_discrete_matrix(self._M_mediolateral, "Lateral axis", labs)

        def _ap_matrix() -> None:
            labs = self._structures_for_tab("Anteroposterior")
            if self._merged_mode and self._agg_anteroposterior is not None:
                self._show_agg_matrix(self._agg_anteroposterior, "Anteroposterior axis", labs)
            else:
                self._show_discrete_matrix(self._M_anteroposterior, "Anteroposterior axis", labs)

        vert_matrix_btn.clicked.connect(_vert_matrix)
        ml_matrix_btn.clicked.connect(_ml_matrix)
        ap_matrix_btn.clicked.connect(_ap_matrix)

