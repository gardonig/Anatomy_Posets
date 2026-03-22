from typing import Dict, List, Optional, Set, Tuple
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
    QSlider,
    QComboBox,
    QDialog,
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
from ..core.config import ASSETS_DIR
from .dialogs import ClickableImageLabel


class CoronalSliceViewer(QWidget):
    """
    Viewer for coronal image stacks (e.g. Visible Human PNG slices).
    Lets the user pick a base folder and then browse stacks by region with a slider and arrow buttons.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Coronal Slice Viewer")
        self.resize(900, 720)

        self._base_dir: str | None = None
        self._stacks: Dict[str, List[str]] = {}
        self._current_region: str | None = None
        self._current_index: int = 0
        # Cache loaded pixmaps to keep scrubbing responsive.
        self._pixmap_cache: Dict[str, QPixmap] = {}

        root = QVBoxLayout(self)

        # Top controls: data folder chooser + region selector
        top_row = QHBoxLayout()
        root.addLayout(top_row)

        self._choose_btn = QPushButton("Choose image folder…")
        self._choose_btn.setToolTip(
            "Select the folder that contains the subfolders "
            "e.g. head, thorax, abdomen, pelvis, thighs, legs"
        )
        self._choose_btn.clicked.connect(self._select_base_folder)
        top_row.addWidget(self._choose_btn)

        top_row.addStretch(1)

        self._region_combo = QComboBox()
        self._region_combo.setEnabled(False)
        self._region_combo.currentTextChanged.connect(self._on_region_changed)
        top_row.addWidget(QLabel("Region:"))
        top_row.addWidget(self._region_combo)

        # Image area
        self._image_label = ClickableImageLabel("Coronal slice — full view", parent=self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 8px; "
            "background: #000000; padding: 4px; margin: 8px;"
        )
        self._image_label.setMinimumHeight(540)
        root.addWidget(self._image_label, stretch=1)

        # Navigation controls: left / slider / right
        nav_row = QHBoxLayout()
        root.addLayout(nav_row)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(40)
        self._prev_btn.clicked.connect(self._step_prev)
        self._prev_btn.setEnabled(False)
        nav_row.addWidget(self._prev_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        nav_row.addWidget(self._slider, stretch=1)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(40)
        self._next_btn.clicked.connect(self._step_next)
        self._next_btn.setEnabled(False)
        nav_row.addWidget(self._next_btn)

        self._index_label = QLabel("Slice: – / –")
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._index_label)

        # If the assets live inside the repository, offer them as a default base folder.
        # Prefer assets/visible_human_male, fall back to assets/visible_human.
        default_base = None
        vh_male = ASSETS_DIR / "visible_human_male"
        vh_generic = ASSETS_DIR / "visible_human"
        if vh_male.exists():
            default_base = vh_male
        elif vh_generic.exists():
            default_base = vh_generic

        if default_base is not None:
            self._load_base_folder(str(default_base))
        else:
            # Otherwise prompt the user the first time
            self._select_base_folder(initial_prompt=True)

    def _select_base_folder(self, initial_prompt: bool = False) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Select base folder with coronal image stacks",
            self._base_dir or str(ASSETS_DIR),
        )
        if not folder:
            if initial_prompt:
                # Show a gentle hint if the viewer is opened without a valid folder.
                self._image_label.setText(
                    "No image folder selected.\n\n"
                    "Click 'Choose image folder…' to pick the directory that contains\n"
                    "subfolders such as head, thorax, abdomen, pelvis, thighs, legs."
                )
            return
        self._load_base_folder(folder)

    def _load_base_folder(self, folder: str) -> None:
        from pathlib import Path

        base = Path(folder)
        if not base.exists() or not base.is_dir():
            QMessageBox.warning(
                self,
                "Folder not found",
                f"The selected folder does not exist or is not a directory:\n{folder}",
            )
            return

        self._base_dir = folder
        self._stacks.clear()

        # Discover subfolders with PNG slices (case-insensitive), ignore other files like .txt
        for sub in sorted(base.iterdir()):
            if not sub.is_dir():
                continue
            pngs = sorted(
                str(p)
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
                "The selected folder should contain subfolders (e.g. head, thorax, abdomen, pelvis,\n"
                "thighs, legs) with .png slices inside each of them."
            )
            self._index_label.setText("Slice: – / –")
            return

        # Select the first region by default
        first_region = self._region_combo.currentText()
        if first_region:
            self._on_region_changed(first_region)

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
        # Use cache to avoid re-reading from disk on every slider tick.
        pix = self._pixmap_cache.get(path)
        if pix is None:
            pix = QPixmap(path)
            if not pix.isNull():
                if len(self._pixmap_cache) > 1000:
                    self._pixmap_cache.clear()
                self._pixmap_cache[path] = pix
        if pix.isNull():
            from pathlib import Path as _Path

            self._image_label.setText(f"[Could not load slice: {_Path(path).name}]")
            self._image_label.setPixmap(QPixmap())
        else:
            self._image_label.set_full_pixmap(pix)
            # Let ClickableImageLabel handle drawing/scaling; just trigger repaint.
            self._image_label.update()
        self._index_label.setText(f"Slice: {idx + 1} / {len(stack)}")


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


class PosetViewerWindow(QWidget):
    """View saved poset(s): tabular data + Hasse diagram per axis (Vertical / Mediolateral / Anteroposterior)."""

    def __init__(self, poset_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Viewer")
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
            self.setWindowTitle("Anatomical Poset Viewer — merged")
            self._status_label.setText(f"Merged: K = {self._merge_k} file(s)")
        elif self._path:
            self.setWindowTitle(f"Anatomical Poset Viewer — {Path(self._path).name}")
            self._status_label.setText(str(Path(self._path).resolve()))
        else:
            self.setWindowTitle("Anatomical Poset Viewer")
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

        ax.set_title(
            f"{title}\n"
            "Merge does not propagate across cells — each (i,j) is independent. "
            "P(yes)=(μ+1)/2 uses only raters who answered that cell (−2 ignored). "
            "0.25/0.75 needs both raters to answer: e.g. 0 & +1 → μ=0.5 → P=0.75. "
            "If one rater has unsure (0) and the other −2 here, μ=0 → P=0.5 (not 0.75). "
            "Many such cells in one row/column can look like a stripe — check "
            "“answered 1/2” in cell text. Builder gap order still asks 1>4 after 1>3 "
            "unless inference fills the cell."
        )
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


class StructureViewsWindow(QWidget):
    """
    Image-based atlas of anatomical structures and standard views.
    Users can navigate through structures (tabs) and rotations (front / side / rear).
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Anatomy Views")
        self.resize(1000, 720)

        self._tabs = QTabWidget()
        root = QVBoxLayout(self)
        root.addWidget(self._tabs)

        # Map structures and rotations to asset filenames
        self._structure_views: Dict[str, Dict[str, str]] = {
            "Skeleton": {
                "Front": "skeleton_front.png",
                "Side": "skeleton_side.png",
                "Rear": "skeleton_rear.png",
            },
            "Superficial muscles": {
                "Front": "mm_super_front.png",
                "Side": "mm_super_side.png",
                "Rear": "mm_super_rear.png",
            },
            "Subcutaneous muscles": {
                "Front": "mm_sub_front.png",
                "Side": "mm_sub_side.png",
                "Rear": "mm_sub_rear.png",
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

        for structure_name, views in self._structure_views.items():
            tab = QWidget()
            layout = QVBoxLayout(tab)

            heading = QLabel(structure_name)
            heading.setAlignment(Qt.AlignmentFlag.AlignCenter)
            heading.setStyleSheet(
                "font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #1a1a1a;"
            )
            layout.addWidget(heading)

            # View selector buttons
            btn_row = QHBoxLayout()
            layout.addLayout(btn_row)

            image_label = ClickableImageLabel(
                f"{structure_name} — full view", parent=self
            )
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            image_label.setFixedHeight(540)
            image_label.setStyleSheet(
                "border: 1px solid #e0e0e0; border-radius: 8px; "
                "background: #ffffff; padding: 4px; margin: 8px;"
            )
            layout.addWidget(image_label)

            def make_handler(view_key: str, label: ClickableImageLabel) -> None:
                def _handler() -> None:
                    filename = views.get(view_key)
                    if not filename:
                        label.setText(f"[No {view_key.lower()} view image available]")
                        label.setPixmap(QPixmap())
                        return
                    path = ASSETS_DIR / "images" / filename
                    if not path.exists():
                        label.setText(f"[Missing image: {filename}]")
                        label.setPixmap(QPixmap())
                        return
                    pix = QPixmap(str(path))
                    if pix.isNull():
                        label.setText(f"[Could not load image: {filename}]")
                        label.setPixmap(QPixmap())
                        return
                    label.set_full_pixmap(pix)
                    label.setPixmap(
                        pix.scaledToHeight(540, Qt.SmoothTransformation)
                    )

                return _handler

            # Create buttons in an intuitive left-to-right order
            buttons: Dict[str, QPushButton] = {}
            for view_key in ("Front", "Side", "Rear"):
                btn = QPushButton(view_key)
                btn.setCheckable(True)
                btn.setStyleSheet(
                    """
                    QPushButton {
                        padding: 6px 16px;
                        border-radius: 6px;
                        border: 1px solid #d0d0d0;
                        background-color: #f5f5f5;
                    }
                    QPushButton:hover {
                        background-color: #eaeaea;
                    }
                    QPushButton:checked {
                        background-color: #007aff;
                        color: white;
                        border-color: #0051d5;
                    }
                    """
                )

                def on_clicked(checked: bool, key: str = view_key) -> None:  # type: ignore[override]
                    if not checked:
                        # Keep one view active; ignore unchecking
                        btn.setChecked(True)
                        return
                    for other_key, other_btn in buttons.items():
                        if other_key != key:
                            other_btn.setChecked(False)
                    make_handler(key, image_label)()

                btn.clicked.connect(on_clicked)
                buttons[view_key] = btn
                btn_row.addWidget(btn)

            btn_row.addStretch(1)

            # Load default view (Front) on startup
            default_view = "Front" if "Front" in views else next(iter(views))
            if default_view in buttons:
                buttons[default_view].setChecked(True)
                make_handler(default_view, image_label)()

            self._tabs.addTab(tab, structure_name)

        # Slice/volume viewer entry points
        bottom_row = QHBoxLayout()
        root.addLayout(bottom_row)
        bottom_row.addStretch(1)

        open_coronal_btn = QPushButton("Open coronal slice viewer…")
        open_coronal_btn.setToolTip(
            "Browse coronal image stacks (e.g. Visible Human) with a slider and arrow keys."
        )

        def _open_coronal_viewer() -> None:
            viewer = CoronalSliceViewer(self)
            viewer.show()

        open_coronal_btn.clicked.connect(_open_coronal_viewer)
        bottom_row.addWidget(open_coronal_btn)

        open_volume_btn = QPushButton("Open full-body volume viewer…")
        open_volume_btn.setToolTip(
            "Browse the downsampled full-body volume from a NumPy tensor with axial/coronal/sagittal views."
        )

        def _open_volume_viewer() -> None:
            viewer = FullBodyVolumeViewer(self)
            viewer.show()

        open_volume_btn.clicked.connect(_open_volume_viewer)
        bottom_row.addWidget(open_volume_btn)

        bottom_row.addStretch(1)


class FullBodyVolumeViewer(QWidget):
    """
    Viewer for the downsampled full-body volume stored as a NumPy tensor.
    Lets the user pick a .npy file and browse axial / coronal / sagittal slices with a slider.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Full-body Slice Viewer")
        self.resize(900, 720)

        self._volume: np.ndarray | None = None  # shape (Z, Y, X), float32 [0, 1]
        self._plane: str = "axial"  # axial / coronal / sagittal
        self._index: int = 0

        root = QVBoxLayout(self)

        # Top row: choose tensor file + plane selector
        top_row = QHBoxLayout()
        root.addLayout(top_row)

        self._load_btn = QPushButton("Choose volume (.npy)…")
        self._load_btn.setToolTip(
            "Select a NumPy tensor file containing the full-body volume "
            "(e.g. full_body_tensor.npy)."
        )
        self._load_btn.clicked.connect(self._select_tensor_file)
        top_row.addWidget(self._load_btn)

        top_row.addStretch(1)

        self._plane_buttons: Dict[str, QPushButton] = {}
        plane_row = QHBoxLayout()
        top_row.addLayout(plane_row)

        def _make_plane_button(label: str, key: str) -> QPushButton:
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

            def on_clicked(checked: bool, k: str = key) -> None:  # type: ignore[override]
                if not checked:
                    # keep one plane active; ignore unchecking
                    btn.setChecked(True)
                    return
                for other_key, other_btn in self._plane_buttons.items():
                    if other_key != k:
                        other_btn.setChecked(False)
                self._plane = k
                self._reset_slider_for_plane()
                self._update_image()

            btn.clicked.connect(on_clicked)
            return btn

        for label, key in (("Axial", "axial"), ("Coronal", "coronal"), ("Sagittal", "sagittal")):
            b = _make_plane_button(label, key)
            self._plane_buttons[key] = b
            plane_row.addWidget(b)

        # Default plane
        self._plane_buttons["axial"].setChecked(True)

        # Image area
        self._image_label = ClickableImageLabel("Full-body slice — full view", parent=self)
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setStyleSheet(
            "border: 1px solid #e0e0e0; border-radius: 8px; "
            "background: #000000; padding: 4px; margin: 8px;"
        )
        self._image_label.setMinimumHeight(540)
        root.addWidget(self._image_label, stretch=1)

        # Navigation controls
        nav_row = QHBoxLayout()
        root.addLayout(nav_row)

        self._prev_btn = QPushButton("◀")
        self._prev_btn.setFixedWidth(40)
        self._prev_btn.clicked.connect(self._step_prev)
        self._prev_btn.setEnabled(False)
        nav_row.addWidget(self._prev_btn)

        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(0)
        self._slider.setSingleStep(1)
        self._slider.setPageStep(10)
        self._slider.setEnabled(False)
        self._slider.valueChanged.connect(self._on_slider_changed)
        nav_row.addWidget(self._slider, stretch=1)

        self._next_btn = QPushButton("▶")
        self._next_btn.setFixedWidth(40)
        self._next_btn.clicked.connect(self._step_next)
        self._next_btn.setEnabled(False)
        nav_row.addWidget(self._next_btn)

        self._index_label = QLabel("Slice: – / –")
        self._index_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        nav_row.addWidget(self._index_label)

        # Try to auto-load a default tensor near the repository root or assets dir.
        self._try_auto_load_tensor()

    # ---- Tensor loading ----
    def _select_tensor_file(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select NumPy volume file",
            str(Path.cwd()),
            "NumPy files (*.npy *.npz);;All files (*)",
        )
        if not path:
            return
        self._load_tensor(Path(path))

    def _try_auto_load_tensor(self) -> None:
        candidates = []
        # 1) assets/visible_human_tensors (prefer RGB tensor, fall back to grayscale)
        vh_dir = ASSETS_DIR / "visible_human_tensors"
        candidates.append(vh_dir / "full_body_tensor_rgb.npy")
        candidates.append(vh_dir / "full_body_tensor.npy")
        # 2) repository root (three levels above this file)
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
            "Click 'Choose volume (.npy)…' to load a NumPy tensor created from the "
            "downsampled full-body slices."
        )

    def _load_tensor(self, path: Path) -> None:
        try:
            arr = np.load(str(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to load volume",
                f"Could not load NumPy array from:\n{path}\n\n{exc}",
            )
            return

        if arr.ndim not in (3, 4):
            QMessageBox.warning(
                self,
                "Invalid volume",
                f"Expected a 3D or 4D RGB array, got shape {arr.shape!r}.",
            )
            return

        self._volume = arr.astype(np.float32, copy=False)
        z_dim = self._volume.shape[0]
        self._index = z_dim // 2

        self._slider.setEnabled(True)
        self._prev_btn.setEnabled(True)
        self._next_btn.setEnabled(True)
        self._reset_slider_for_plane()
        self._update_image()

    # ---- Navigation ----
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

        self._index = max(min_idx, min(self._index, max_idx))

        self._slider.blockSignals(True)
        self._slider.setMinimum(min_idx)
        self._slider.setMaximum(max_idx)
        self._slider.setValue(self._index)
        self._slider.blockSignals(False)

    def _on_slider_changed(self, value: int) -> None:
        self._index = int(value)
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
        target_h = self._image_label.height() or 540
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