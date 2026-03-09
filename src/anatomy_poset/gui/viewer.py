from typing import Dict, List, Set, Tuple
import math
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap
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
)

from ..core.io import load_poset_from_json
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
    ) -> None:
        """
        Lay out nodes and draw edges. Vertical axis: levels (top to bottom), x spread by index.
        Frontal axis: y by level, x by com_lateral so left is left of spine is left of right.
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

    def __init__(self, poset_path: str) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Viewer")
        self.resize(900, 600)

        self._path = poset_path
        self._structures: List[Structure] = []
        self._edges_vertical: Set[Tuple[int, int]] = set()
        self._edges_mediolateral: Set[Tuple[int, int]] = set()
        self._edges_anteroposterior: Set[Tuple[int, int]] = set()

        self._tabs = QTabWidget()
        root = QVBoxLayout(self)
        root.addWidget(self._tabs)

        self._load(poset_path)

    def _fill_tab(
        self,
        list_widget: QListWidget,
        hasse_view: HasseDiagramView,
        structures: List[Structure],
        edges: Set[Tuple[int, int]],
        axis_label: str,
        relation_label: str,
    ) -> None:
        list_widget.clear()
        list_widget.addItem(f"Loaded from: {self._path}")
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
        if not edges:
            list_widget.addItem(f"No strict '{relation_label}' relations recorded.")
        else:
            list_widget.addItem("Cover relations (Hasse diagram edges):")
            for u, v in sorted(edges):
                su, sv = structures[u], structures[v]
                list_widget.addItem(f"{su.name}  ≻  {sv.name}")

        if axis_label == "Vertical":
            axis = AXIS_VERTICAL
        elif axis_label == "Lateral":
            axis = AXIS_MEDIOLATERAL
        else:
            axis = AXIS_ANTERIOR_POSTERIOR
        hasse_view.draw_diagram(structures, edges, axis=axis)

    def _load(self, path: str) -> None:
        try:
            (
                self._structures,
                self._edges_vertical,
                self._edges_mediolateral,
                self._edges_anteroposterior,
            ) = load_poset_from_json(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Failed to Load",
                f"Could not load poset from:\n{path}\n\n{exc}",
            )
            return

        # Tab: Vertical (top–bottom)
        vert_widget = QWidget()
        vert_layout = QHBoxLayout(vert_widget)
        vert_list_group = QGroupBox("Poset Data")
        vert_list_layout = QVBoxLayout(vert_list_group)
        vert_list = QListWidget()
        vert_list_layout.addWidget(vert_list)
        vert_diagram_group = QGroupBox("Hasse Diagram")
        vert_diagram_layout = QVBoxLayout(vert_diagram_group)
        vert_hasse = HasseDiagramView()
        vert_hasse.setMinimumSize(400, 300)
        vert_diagram_layout.addWidget(vert_hasse)
        vert_layout.addWidget(vert_list_group, stretch=1)
        vert_layout.addWidget(vert_diagram_group, stretch=2)
        self._fill_tab(
            vert_list,
            vert_hasse,
            self._structures,
            self._edges_vertical,
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
        ml_diagram_group = QGroupBox("Hasse Diagram")
        ml_diagram_layout = QVBoxLayout(ml_diagram_group)
        ml_hasse = HasseDiagramView()
        ml_hasse.setMinimumSize(400, 300)
        ml_diagram_layout.addWidget(ml_hasse)
        ml_layout.addWidget(ml_list_group, stretch=1)
        ml_layout.addWidget(ml_diagram_group, stretch=2)
        self._fill_tab(
            ml_list,
            ml_hasse,
            self._structures,
            self._edges_mediolateral,
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
        ap_diagram_group = QGroupBox("Hasse Diagram")
        ap_diagram_layout = QVBoxLayout(ap_diagram_group)
        ap_hasse = HasseDiagramView()
        ap_hasse.setMinimumSize(400, 300)
        ap_diagram_layout.addWidget(ap_hasse)
        ap_layout.addWidget(ap_list_group, stretch=1)
        ap_layout.addWidget(ap_diagram_group, stretch=2)
        self._fill_tab(
            ap_list,
            ap_hasse,
            self._structures,
            self._edges_anteroposterior,
            "Anteroposterior",
            "in front of",
        )
        self._tabs.addTab(ap_widget, "Anteroposterior (front–back)")


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

        # Coronal slice viewer entry point
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
        bottom_row.addStretch(1)