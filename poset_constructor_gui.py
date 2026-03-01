import json
import os
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from PySide6.QtCore import Qt, QPointF
from PySide6.QtGui import QColor, QPainter, QPen, QBrush, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QListWidget,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

def _ensure_qt_platform_plugin_path() -> None:
    """
    macOS fix for:
      qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""

    PySide6 bundles the plugin at:
      <site-packages>/PySide6/Qt/plugins/platforms/libqcocoa.dylib
    """
    if sys.platform != "darwin":
        return

    try:
        import PySide6
    except Exception:
        return

    base = Path(PySide6.__file__).resolve().parent
    platforms_dir = base / "Qt" / "plugins" / "platforms"
    if platforms_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))


@dataclass
class Structure: # this is a node in the poset which is an anatomical structure (organ, bone, muscle, etc.)
    name: str 
    com_vertical: float  # Center of Mass along vertical (superior–inferior) axis
# e.g. Structure(name="Skull", com_vertical=90.0)


def load_structures_from_json(path: str) -> List[Structure]:
    """
    Load a list of structures from a JSON file.

    Expected format:
    {
      "structures": [
        {"name": "Skull", "com_vertical": 90.0},
        ...
      ]
    }
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("structures", [])
    structures: List[Structure] = []
    for item in items:
        try:
            name = str(item["name"])
            com_vertical = float(item["com_vertical"])
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(Structure(name=name, com_vertical=com_vertical))
    return structures


def save_poset_to_json(
    path: str,
    structures: List[Structure],
    edges: Set[Tuple[int, int]],
) -> None:
    """
    Save a fully constructed poset (structures + Hasse edges) to JSON.

    Format:
    {
      "structures": [{"name": ..., "com_vertical": ...}, ...],
      "edges": [[u, v], ...]
    }
    """
    payload = {
        "structures": [asdict(s) for s in structures],
        "edges": [[int(u), int(v)] for (u, v) in sorted(edges)],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_poset_from_json(path: str) -> Tuple[List[Structure], Set[Tuple[int, int]]]:
    """Load a poset (structures + edges) from JSON."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    structures_data = data.get("structures", [])
    edges_data = data.get("edges", [])

    structures: List[Structure] = []
    for item in structures_data:
        try:
            name = str(item["name"])
            com_vertical = float(item["com_vertical"])
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(Structure(name=name, com_vertical=com_vertical))

    edges: Set[Tuple[int, int]] = set()
    for item in edges_data:
        try:
            u, v = int(item[0]), int(item[1])
            edges.add((u, v))
        except (TypeError, ValueError, IndexError):
            continue

    return structures, edges


class PosetViewerWindow(QWidget):
    """View a saved poset: tabular data + Hasse diagram."""

    def __init__(self, poset_path: str) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Viewer")
        self.resize(900, 600)

        root = QHBoxLayout(self)
        list_group = QGroupBox("Poset Data")
        list_layout = QVBoxLayout(list_group)
        self.list_widget = QListWidget()
        list_layout.addWidget(self.list_widget)

        diagram_group = QGroupBox("Hasse Diagram")
        diagram_layout = QVBoxLayout(diagram_group)
        self.hasse_view = HasseDiagramView()
        self.hasse_view.setMinimumSize(400, 300)
        diagram_layout.addWidget(self.hasse_view)

        root.addWidget(list_group, stretch=1)
        root.addWidget(diagram_group, stretch=2)

        self._load(poset_path)

    def _load(self, path: str) -> None:
        try:
            structures, edges = load_poset_from_json(path)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(
                self,
                "Failed to Load",
                f"Could not load poset from:\n{path}\n\n{exc}",
            )
            return

        self.list_widget.clear()
        self.list_widget.addItem(f"Loaded from: {path}")
        self.list_widget.addItem("")
        self.list_widget.addItem("Structures (sorted superior → inferior, as saved):")
        for idx, s in enumerate(structures):
            self.list_widget.addItem(f"  {idx}: {s.name} (CoM vertical = {s.com_vertical})")
        self.list_widget.addItem("")

        if not edges:
            self.list_widget.addItem("No strict 'above' relations recorded.")
        else:
            self.list_widget.addItem("Cover relations (Hasse diagram edges):")
            for u, v in sorted(edges):
                su, sv = structures[u], structures[v]
                self.list_widget.addItem(f"{su.name}  ≻  {sv.name}")

        self.hasse_view.draw_diagram(structures, edges)


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
    ) -> None:
        """
        Lay out nodes in levels (superior at the top, inferior at the bottom),
        then draw nodes with labels and connecting edges.
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

        # Simple topological order to propagate levels
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

        # Group nodes by level (0 = most superior)
        level_nodes: Dict[int, List[int]] = {}
        max_level = 0
        for node, lvl in levels.items():
            level_nodes.setdefault(lvl, []).append(node)
            if lvl > max_level:
                max_level = lvl

        # Compute positions
        node_radius = 35.0
        h_spacing = 140.0
        v_spacing = 140.0

        positions: Dict[int, QPointF] = {}

        for lvl in range(0, max_level + 1):
            nodes_at_level = level_nodes.get(lvl, [])
            if not nodes_at_level:
                continue
            count = len(nodes_at_level)
            total_width = (count - 1) * h_spacing
            start_x = -total_width / 2.0
            y = lvl * v_spacing

            for idx, node in enumerate(sorted(nodes_at_level)):
                x = start_x + idx * h_spacing
                positions[node] = QPointF(x, y)

        # Draw edges first so they appear behind nodes
        edge_pen = QPen(QColor(80, 80, 80))
        edge_pen.setWidth(2)
        for u, v in edges:
            p1 = positions.get(u)
            p2 = positions.get(v)
            if p1 is None or p2 is None:
                continue
            self._scene.addLine(
                p1.x(),
                p1.y(),
                p2.x(),
                p2.y(),
                edge_pen,
            )

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

class PosetBuilder:
    """
    Implements Algorithm 1 using the gap-based CoM strategy.
    Structures are first sorted by CoM (descending).
    The user is then queried step‑by‑step via Q(x, y) provided by the GUI.
    """

    def __init__(self, structures: List[Structure]) -> None:
        # Sort descending by CoM
        self.structures: List[Structure] = sorted(
            structures, key=lambda s: s.com_vertical, reverse=True
        )
        self.n = len(self.structures)

        # Graph represented as adjacency list using indices into self.structures
        self.edges: Set[Tuple[int, int]] = set()

        # Iteration state for gap-based strategy
        self.current_gap = 1
        self.current_i = 0
        self.finished = self.n <= 1

    # -------- Core graph helpers -------- #
    def path_exists(self, start: int, end: int, edges: Set[Tuple[int, int]] | None = None) -> bool:
        if start == end:
            return True
        if edges is None:
            edges = self.edges

        adjacency: Dict[int, List[int]] = {}
        for u, v in edges:
            adjacency.setdefault(u, []).append(v)

        stack = [start]
        visited = set()
        while stack: #seems slow, any alternatives? (i guess not bad for small graphs)
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in adjacency.get(u, []):
                if v == end:
                    return True
                if v not in visited:
                    stack.append(v)
        return False

    def transitive_reduction(self) -> Set[Tuple[int, int]]:
        """
        Naive O(V * E * (V + E)) transitive reduction.
        Fine for small N (e.g., <= 50).
        """
        reduced: Set[Tuple[int, int]] = set(self.edges)
        for u, v in list(self.edges):
            # Temporarily remove edge and test if an alternative path still exists
            temp_edges = set(reduced)
            temp_edges.discard((u, v))
            if self.path_exists(u, v, temp_edges):
                # Edge is redundant
                reduced.discard((u, v))
        return reduced

    # -------- Gap‑based query iteration -------- #
    def next_pair(self) -> Tuple[int, int] | None:
        """
        Advance the (gap, i) loops until the next pair requiring a human query is found.
        Returns (i, j) or None when finished.
        """
        if self.finished:
            return None

        while self.current_gap <= self.n - 1:
            while self.current_i <= self.n - 1 - self.current_gap:
                i = self.current_i
                j = i + self.current_gap
                self.current_i += 1

                # Skip if relation already implied by transitivity
                if self.path_exists(i, j):
                    continue

                # We have a new pair to query
                return i, j

            # Move to next gap
            self.current_gap += 1
            self.current_i = 0

        # No more pairs
        self.finished = True
        return None

    def get_iteration_progress(self) -> float:
        """
        Progress through the fixed (gap, i) iteration, 0.0 to 1.0.
        One answer can advance this by a lot when many pairs are skipped by transitivity.
        """
        if self.finished or self.n <= 1:
            return 1.0
        total = self.n * (self.n - 1) // 2
        if total == 0:
            return 1.0
        # Pairs already passed: gaps 1..current_gap-1 fully, plus current_i in current_gap
        steps_done = (self.current_gap - 1) * self.n - (self.current_gap - 1) * self.current_gap // 2 + self.current_i
        return min(1.0, steps_done / total)

    def record_response(self, i: int, j: int, is_above: bool) -> None:
        """
        Called by the GUI after the clinician/user answers Q(si, sj).
        """
        if is_above:
            self.edges.add((i, j))

    def get_final_relations(self) -> Tuple[List[Structure], Set[Tuple[int, int]]]:
        """
        Run transitive reduction and return the sorted structures
        and the minimal set of cover relations (Hasse diagram edges).
        """
        reduced_edges = self.transitive_reduction()
        return self.structures, reduced_edges


class DefinitionDialog(QDialog):
    """
    Shown before the query window. Displays the definition and example images.
    User presses "Understood" to proceed to questions.
    """

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Definition — Poset Construction")
        self.resize(560, 600)
        self.setModal(True)

        layout = QVBoxLayout(self)

        def_label = QLabel(
            "'Strictly above': the lowest point of the superior structure is higher "
            "than the highest point of the inferior one (head–to–toes axis)."
        )
        def_label.setWordWrap(True)
        def_label.setStyleSheet(
            "color: #222; font-size: 16px; padding: 12px 16px; "
            "background: #f5f5f5; border-radius: 8px;"
        )
        layout.addWidget(def_label)

        # Example images with references (stored in assets/definition_images)
        _script_dir = Path(__file__).resolve().parent
        _img_dir = _script_dir / "assets" / "definition_images"
        _img_height = 140

        # Image 1: Femur strictly above tarsal bones (full leg X-ray)
        ex1_label = QLabel("Example: Femur strictly above tarsal bones")
        ex1_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 14px;")
        layout.addWidget(ex1_label)
        femur_tarsal_path = _img_dir / "femur_tarsal_leg.jpg"
        self.example_img_femur_tarsal = QLabel()
        self.example_img_femur_tarsal.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.example_img_femur_tarsal.setFixedHeight(_img_height)
        self.example_img_femur_tarsal.setStyleSheet(
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 10px; "
            "background: #fafafa;"
        )
        if femur_tarsal_path.exists():
            pix = QPixmap(str(femur_tarsal_path))
            if not pix.isNull():
                self.example_img_femur_tarsal.setPixmap(
                    pix.scaledToHeight(_img_height, Qt.SmoothTransformation)
                )
        if self.example_img_femur_tarsal.pixmap() is None or self.example_img_femur_tarsal.pixmap().isNull():
            self.example_img_femur_tarsal.setText("Example:\nFemur strictly above tarsal bones\n[image not found]")
            self.example_img_femur_tarsal.setStyleSheet(
                "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 10px; "
                "color: #555; font-size: 13px;"
            )
        layout.addWidget(self.example_img_femur_tarsal)
        ref1 = QLabel(
            "Source: X-ray of leg (femur, tibia, ankle). Nizil Shah, CC BY-SA 4.0. "
            "commons.wikimedia.org/wiki/File:X_ray_internal_fixation_leg_fracture.jpg"
        )
        ref1.setWordWrap(True)
        ref1.setStyleSheet("color: #666; font-size: 10px; margin-top: 2px;")
        layout.addWidget(ref1)

        # Image 2: Pelvis and femur overlapping (3D CT)
        ex2_label = QLabel("Example: Pelvis and femur overlapping")
        ex2_label.setStyleSheet("font-weight: bold; font-size: 14px; margin-top: 14px;")
        layout.addWidget(ex2_label)
        pelvis_femur_path = _img_dir / "pelvis_femur_ct.jpg"
        self.example_img_pelvis_femur = QLabel()
        self.example_img_pelvis_femur.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.example_img_pelvis_femur.setFixedHeight(_img_height)
        self.example_img_pelvis_femur.setStyleSheet(
            "border: 1px solid #cccccc; border-radius: 8px; margin-top: 6px; "
            "background: #fafafa;"
        )
        if pelvis_femur_path.exists():
            pix = QPixmap(str(pelvis_femur_path))
            if not pix.isNull():
                self.example_img_pelvis_femur.setPixmap(
                    pix.scaledToHeight(_img_height, Qt.SmoothTransformation)
                )
        if self.example_img_pelvis_femur.pixmap() is None or self.example_img_pelvis_femur.pixmap().isNull():
            self.example_img_pelvis_femur.setText("Example:\nPelvis and femur overlapping\n[image not found]")
            self.example_img_pelvis_femur.setStyleSheet(
                "border: 1px dashed #bbbbbb; border-radius: 8px; margin-top: 6px; "
                "color: #555; font-size: 13px;"
            )
        layout.addWidget(self.example_img_pelvis_femur)
        ref2 = QLabel(
            "Source: 3D rendered CT of bony pelvis. Mikael Häggström, CC0 (Public Domain). "
            "commons.wikimedia.org/wiki/File:3D_rendered_CT_of_bony_pelvis_2.jpg"
        )
        ref2.setWordWrap(True)
        ref2.setStyleSheet("color: #666; font-size: 10px; margin-top: 2px;")
        layout.addWidget(ref2)

        understood_btn = QPushButton("Understood")
        understood_btn.setStyleSheet(
            """
            QPushButton {
                background-color: #007aff; color: white; border: none; border-radius: 8px;
                padding: 12px 24px; font-size: 15px;
            }
            QPushButton:hover { background-color: #5ac8fa; }
            QPushButton:pressed { background-color: #0051d5; }
            """
        )
        understood_btn.clicked.connect(self.accept)
        layout.addWidget(understood_btn)


class QueryDialog(QDialog):
    """
    Standalone dialog for expert queries only.
    Clinicians focus on answering questions; no structure input.
    """

    def __init__(
        self,
        poset_builder: PosetBuilder,
        autosave_path: Path,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Expert Query — Anatomical Poset")
        self.resize(520, 420)
        self.setModal(False)

        self.poset_builder = poset_builder
        self._autosave_path = autosave_path
        self.pending_pair: Tuple[int, int] | None = None
        self._answer_history: List[Tuple[int, int, bool]] = []

        layout = QVBoxLayout(self)

        # Question card
        self.question_card = QFrame()
        self.question_card.setMinimumHeight(160)
        self.question_card.setStyleSheet(
            """
            QFrame {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 12px;
                padding: 24px;
                margin: 12px 0;
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
        layout.addWidget(self.question_card)

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
        btn_row.addStretch()
        btn_row.addWidget(self.no_btn)
        btn_row.addWidget(self.yes_btn)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        # Progress bar (no numbers): reflects iteration progress; can jump when answers imply many skips
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
        layout.addWidget(self.progress_bar)

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
        layout.addWidget(self.finish_btn)

        self._advance_to_next_query()

    def _autosave_poset(self) -> None:
        if not self._autosave_path:
            return
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            structures, edges = self.poset_builder.get_final_relations()

            # Use fixed path (replaces same file each time)
            save_path = self._autosave_path

            # TODO Uncomment below to save timestamped files instead (no overwrite): #################################################
            # base = self._autosave_path.stem.replace(".poset_autosave", "")
            # ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # save_path = self._autosave_path.parent / f"{base}_{ts}.poset_autosave.json"

            save_poset_to_json(str(save_path), structures, edges)
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
            self.back_btn.hide()
            self.finish_btn.show()
            self.progress_bar.setValue(100)
            return
        i, j = pair
        si, sj = self.poset_builder.structures[i], self.poset_builder.structures[j]
        self.query_label.setText(f"Is the {si.name} strictly above the {sj.name}?")
        self.progress_bar.setValue(int(self.poset_builder.get_iteration_progress() * 100))

    def answer_query(self, is_above: bool) -> None:
        if self.pending_pair is None:
            return
        i, j = self.pending_pair
        self._answer_history.append((i, j, is_above))
        self.back_btn.setEnabled(True)
        self.poset_builder.record_response(i, j, is_above)
        self._autosave_poset()
        self._advance_to_next_query()

    def go_back_one_question(self) -> None:
        if not self._answer_history:
            return
        last_i, last_j, last_answer = self._answer_history.pop()
        if last_answer:
            self.poset_builder.edges.discard((last_i, last_j))
        self.poset_builder.finished = False
        self.poset_builder.current_gap = last_j - last_i
        self.poset_builder.current_i = last_i + 1
        self.pending_pair = (last_i, last_j)
        si, sj = self.poset_builder.structures[last_i], self.poset_builder.structures[last_j]
        self.query_label.setText(f"(Correcting) Is the {si.name} strictly above the {sj.name}?")
        self.yes_btn.setEnabled(True)
        self.no_btn.setEnabled(True)
        if not self._answer_history:
            self.back_btn.setEnabled(False)
        self.progress_bar.setValue(int(self.poset_builder.get_iteration_progress() * 100))
        self._autosave_poset()

class MainWindow(QMainWindow):
    def __init__(self, input_path: Optional[str] = None) -> None:
        super().__init__()
        self.setWindowTitle("Anatomical Poset Builder (Head → Toes)")
        self.resize(520, 500)

        # Remember optional input path for use during UI setup
        self._input_path: Optional[str] = input_path
        # Where to auto-save; set after we know the actual load path in _init_ui
        output_dir = Path(__file__).resolve().parent / "Output_constructed_posets"
        self._autosave_path: Optional[Path] = output_dir / "poset_autosave.json"

        self.poset_builder: PosetBuilder | None = None
        self._viewer_windows: List[QWidget] = []

        self._init_ui()

    def _init_ui(self) -> None:
        central = QWidget()
        root_layout = QVBoxLayout(central)

        # Structure definition only (no query UI until Start is clicked)
        left_group = QGroupBox("Anatomical Structures (Input)")
        left_layout = QVBoxLayout(left_group)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "CoM (vertical axis, arbitrary units)"])
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.verticalHeader().setVisible(False)
        left_layout.addWidget(self.table)

        load_btn = QPushButton("Load Structures")
        load_btn.clicked.connect(self.load_structures_dialog)
        left_layout.addWidget(load_btn)

        add_remove_row = QHBoxLayout()
        add_row_btn = QPushButton("+ Add Structure")
        add_row_btn.clicked.connect(self.add_structure_row)
        remove_row_btn = QPushButton("− Remove Selected")
        remove_row_btn.clicked.connect(self.remove_selected_row)
        add_remove_row.addWidget(add_row_btn)
        add_remove_row.addWidget(remove_row_btn)
        left_layout.addLayout(add_remove_row)

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

        view_btn = QPushButton("View Poset")
        view_btn.clicked.connect(self._open_viewer)
        left_layout.addWidget(view_btn)

        root_layout.addWidget(left_group)

        self.setCentralWidget(central)

        # Load structures from file: CLI arg, or default test_structures.json
        load_path = self._input_path
        if load_path is None:
            default_file = Path(__file__).resolve().parent / "Input_CoM_structures" / "test_structures.json"
            if default_file.exists():
                load_path = str(default_file)
        if load_path is not None:
            self._autosave_path = self._builtposet_output_path(Path(load_path))
            try:
                structures = load_structures_from_json(load_path)
                for s in structures:
                    self.add_structure_row(s.name, str(s.com_vertical))
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Failed to load input",
                    f"Could not load structures from:\n{load_path}\n\n{exc}",
                )

    def _builtposet_output_path(self, input_path: Path) -> Path:
        """Autosave goes to Output_constructed_posets folder, not the input folder."""
        output_dir = Path(__file__).resolve().parent / "Output_constructed_posets"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / f"{input_path.stem}.poset_autosave.json"

    def _autosave_poset(self) -> None:
        """
        Persist the current best-known poset (transitively reduced)
        to a JSON file after each question / correction.
        """
        if not self.poset_builder or not self._autosave_path:
            return
        try:
            self._autosave_path.parent.mkdir(parents=True, exist_ok=True)
            structures, edges = self.poset_builder.get_final_relations()

            # Use fixed path (replaces same file each time)
            save_path = self._autosave_path

            # TODO Uncomment below to save timestamped files instead (no overwrite): #################################################
            # base = self._autosave_path.stem.replace(".poset_autosave", "")
            # ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            # save_path = self._autosave_path.parent / f"{base}_{ts}.poset_autosave.json"

            save_poset_to_json(str(save_path), structures, edges)
        except Exception:
            # Autosave failures should not break the interactive session
            pass

    # -------- Structure table helpers -------- #
    def load_structures_dialog(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Load structures from JSON",
            str(Path(__file__).resolve().parent),
            "JSON Files (*.json);;All Files (*)",
        )
        if not path:
            return
        try:
            structures = load_structures_from_json(path)
            self.table.setRowCount(0)
            for s in structures:
                self.add_structure_row(s.name, str(s.com_vertical))
            self._autosave_path = self._builtposet_output_path(Path(path))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(
                self,
                "Failed to load",
                f"Could not load structures from:\n{path}\n\n{exc}",
            )

    def add_structure_row(self, name: str = "", com: str = "") -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        name_item = QTableWidgetItem(name)
        com_item = QTableWidgetItem(com)
        self.table.setItem(row, 0, name_item)
        self.table.setItem(row, 1, com_item)

    def remove_selected_row(self) -> None:
        rows = {idx.row() for idx in self.table.selectedIndexes()}
        for row in sorted(rows, reverse=True):
            self.table.removeRow(row)

    def _collect_structures(self) -> List[Structure] | None:
        structures: List[Structure] = []

        for row in range(self.table.rowCount()):
            name_item = self.table.item(row, 0)
            com_item = self.table.item(row, 1)
            if not name_item or not com_item:
                continue

            name = name_item.text().strip()
            com_text = com_item.text().strip()

            if not name or not com_text:
                continue

            try:
                com_vertical = float(com_text)
            except ValueError:
                QMessageBox.warning(
                    self,
                    "Invalid Input",
                    f"Row {row + 1}: CoM must be a number.",
                )
                return None

            structures.append(Structure(name=name, com_vertical=com_vertical))

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

        self.poset_builder = PosetBuilder(structures)
        self.start_btn.setEnabled(False)

        # Show definition first; questions start after user clicks Understood
        def_dialog = DefinitionDialog()
        if def_dialog.exec() == QDialog.DialogCode.Accepted:
            query_dialog = QueryDialog(self.poset_builder, self._autosave_path)
            query_dialog.finished.connect(lambda: self.start_btn.setEnabled(True))
            query_dialog.show()
        else:
            self.start_btn.setEnabled(True)

    def _open_viewer(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose poset file to view",
            str(Path(__file__).resolve().parent),
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


def main() -> None:
    """
    Optional usage:
      python anatomy_poset_gui.py path/to/structures.json

    Where structures.json has:
    {
      "structures": [
        {"name": "Skull", "com_vertical": 90.0},
        ...
      ]
    }
    """
    _ensure_qt_platform_plugin_path()

    # Optional first positional argument = input JSON with anatomical structures
    input_path: Optional[str] = sys.argv[1] if len(sys.argv) > 1 else None

    app = QApplication(sys.argv)
    window = MainWindow(input_path=input_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

