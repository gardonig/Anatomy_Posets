from typing import Dict, List, Optional, Set, Tuple

from .models import AXIS_ANTERIOR_POSTERIOR, AXIS_MEDIOLATERAL, AXIS_VERTICAL, Structure


def _parse_bilateral_core(name: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Detect side (Left/Right) and core name for bilateral structures.
    Handles \"Left X\", \"X left\", \"x_left\", etc. Returns (side, core) or (None, None).
    """
    raw = (name or "").strip()
    if not raw:
        return None, None
    norm = raw.lower().replace("_", " ").replace("-", " ")
    tokens = [t for t in norm.split() if t]
    side: Optional[str] = None
    if "left" in tokens:
        side = "Left"
    elif "right" in tokens:
        side = "Right"
    if side is None:
        return None, None
    core_tokens = [t for t in tokens if t not in ("left", "right")]
    if not core_tokens:
        return side, None
    core = " ".join(w.capitalize() for w in core_tokens)
    return side, core


class PosetBuilder:
    """
    Implements Algorithm 1 using the gap-based CoM strategy.
    Structures are sorted by the chosen axis CoM (descending).
    The user is then queried step‑by‑step via Q(x, y) provided by the GUI.
    """

    def __init__(self, structures: List[Structure], axis: str = AXIS_VERTICAL) -> None:
        if axis == AXIS_MEDIOLATERAL:
            key = lambda s: s.com_lateral
        elif axis == AXIS_ANTERIOR_POSTERIOR:
            key = lambda s: s.com_anteroposterior
        else:
            key = lambda s: s.com_vertical
        self.structures: List[Structure] = sorted(structures, key=key, reverse=True)
        self.n = len(self.structures)
        self.axis = axis

        # Graph represented as adjacency list using indices into self.structures
        self.edges: Set[Tuple[int, int]] = set()
        # Pairs the user explicitly skipped ("Not sure"). Stored with i < j.
        self.skipped_pairs: Set[Tuple[int, int]] = set()

        # Symmetry info for vertical axis: detect left/right pairs with same core name
        self._core_names: List[str] = []
        self._symmetric_partner: Dict[int, int] = {}
        if self.axis == AXIS_VERTICAL:
            side_and_core: List[Tuple[Optional[str], str]] = []
            for s in self.structures:
                side, core = _parse_bilateral_core(s.name)
                side_and_core.append((side, core if core else s.name.strip()))
            core_to_sides: Dict[str, Dict[str, int]] = {}
            for idx, (side, core) in enumerate(side_and_core):
                self._core_names.append(core)
                if side is None:
                    continue
                core_to_sides.setdefault(core, {})[side] = idx
            for core, sides in core_to_sides.items():
                if "Left" in sides and "Right" in sides:
                    li = sides["Left"]
                    ri = sides["Right"]
                    self._symmetric_partner[li] = ri
                    self._symmetric_partner[ri] = li

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

    def edge_redundancy_reduction(self) -> Set[Tuple[int, int]]:
        """
        Remove redundant edges implied by transitivity (naive O(V * E * (V + E)) algorithm).
        This is the transitive reduction of the current directed acyclic graph and yields
        exactly the cover relations used in the (directed) Hasse diagram.
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

                if self.axis == AXIS_VERTICAL:
                    # For vertical axis, enforce a canonical representative for each
                    # left/right pair so that we never ask both
                    #   (Left X, Y) and (Right X, Y),
                    # nor both
                    #   (Y, Left X) and (Y, Right X).
                    pi = self._symmetric_partner.get(i)
                    pj = self._symmetric_partner.get(j)

                    # If i has a symmetric partner with a smaller index,
                    # skip this pair and let that partner represent the core.
                    if pi is not None and pi < i:
                        continue
                    # Similarly for j.
                    if pj is not None and pj < j:
                        continue

                    # Also skip direct Left/Right comparison for the same core vertically
                    if pi is not None and pi == j:
                        continue

                # Skip if the user explicitly skipped this comparison
                if (i, j) in self.skipped_pairs:
                    continue

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
        Progress based on how many unordered pairs {i, j}, i < j, are already
        determined (comparable) in the current graph, 0.0 to 1.0.
        """
        if self.n <= 1:
            return 1.0
        total = self.n * (self.n - 1) // 2
        if total == 0:
            return 1.0

        # Count incomparable pairs that might still need questions in worst case
        remaining = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if (i, j) in self.skipped_pairs:
                    continue
                if not self.path_exists(i, j) and not self.path_exists(j, i):
                    remaining += 1

        known = total - remaining
        return min(1.0, known / total)

    def estimate_remaining_questions(self) -> int:
        """
        Estimate, in the *worst case*, how many questions Algorithm 1 would still
        ask starting from the current (gap, i) state and current edges.
        """
        if self.finished or self.n <= 1:
            return 0

        remaining = 0
        g = self.current_gap
        i = self.current_i

        while g <= self.n - 1:
            while i <= self.n - 1 - g:
                s = i
                t = i + g
                i += 1

                if (s, t) in self.skipped_pairs:
                    continue
                if self.path_exists(s, t):
                    continue

                remaining += 1

            g += 1
            i = 0

        return remaining

    def record_response(self, i: int, j: int, is_above: bool) -> None:
        """
        Called by the GUI after the clinician/user answers Q(si, sj).
        """
        if is_above:
            # Always add the directly answered relation
            self.edges.add((i, j))

            # For vertical axis, also apply symmetry to left/right counterparts:
            # if "Left X" is above Y, then "Right X" is also above Y, and similarly
            # for a symmetric counterpart of Y.
            if self.axis == AXIS_VERTICAL:
                # Mirror on source
                mi = self._symmetric_partner.get(i)
                if mi is not None:
                    self.edges.add((mi, j))

                # Mirror on target
                mj = self._symmetric_partner.get(j)
                if mj is not None:
                    self.edges.add((i, mj))

    def record_skip(self, i: int, j: int) -> None:
        """
        User explicitly chose "Not sure" for this pair.
        We treat this as an unknown/incomparable decision and never re-ask it.
        """
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        self.skipped_pairs.add((a, b))

    def unskip_pair(self, i: int, j: int) -> None:
        """Undo a previous skip for this pair."""
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        self.skipped_pairs.discard((a, b))

    def get_final_relations(self) -> Tuple[List[Structure], Set[Tuple[int, int]]]:
        """
        Run edge redundancy reduction (transitive reduction) and return the sorted structures
        and the minimal set of cover relations (Hasse diagram edges).
        """
        reduced_edges = self.edge_redundancy_reduction()
        return self.structures, reduced_edges