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


class MatrixBuilder(PosetBuilder):
    """
    Extended builder that maintains a tri-valued relation matrix M:

        M[i][j] = +1  -> "i is (strictly) above j"  (YES)
        M[i][j] =  0  -> "not sure / unknown but asked"
        M[i][j] = -1  -> "i is not strictly above j" (NO / overlap / opposite)
        M[i][j] = -2  -> not asked yet

    At construction, structures are sorted by axis CoM (descending). The lower
    triangle (i > j) and diagonal are set to -1; only the strict upper triangle
    starts as -2. Equal-CoM pairs are completed by _apply_com_not_above_prior().

    The underlying DAG used for Hasse diagrams is still derived solely from
    the +1 entries; the matrix simply preserves richer annotation state.
    """

    def __init__(self, structures: List[Structure], axis: str = AXIS_VERTICAL) -> None:
        super().__init__(structures, axis=axis)
        # Structures are sorted by this axis CoM descending (see PosetBuilder.__init__).
        # M[i][j] = "i strictly above j". For i > j, structure i has lower (or equal) CoM than j,
        # so i cannot be strictly above j when CoMs differ; lower triangle is prefilled -1.
        # Upper triangle (i < j) stays -2 (still to ask / infer). Diagonal: -1.
        # Equal-CoM pairs get both directions closed via _apply_com_not_above_prior().
        n = self.n
        self.M = [[-2 for _ in range(n)] for _ in range(n)]
        for i in range(n):
            self.M[i][i] = -1
            for j in range(i):
                self.M[i][j] = -1

        # Cache CoM values for fast constraint checks.
        if self.axis == AXIS_MEDIOLATERAL:
            self._com_values: List[float] = [s.com_lateral for s in self.structures]
        elif self.axis == AXIS_ANTERIOR_POSTERIOR:
            self._com_values = [s.com_anteroposterior for s in self.structures]
        else:
            self._com_values = [s.com_vertical for s in self.structures]

        # Vertical axis: left/right of the *same* anatomical core cannot be
        # strictly above each other. Since the gap-based iterator skips
        # those pairs, we must seed them as explicit NO (-1) so they do not
        # remain "not asked" (-2).
        if self.axis == AXIS_VERTICAL and self._symmetric_partner:
            for i, j in self._symmetric_partner.items():
                if i != j:
                    self.M[i][j] = -1

        # CoM prior (ties): if CoM(a) == CoM(b), neither can be strictly above
        # the other; close any remaining -2 in both directions. Strict CoM order
        # is already reflected in the lower-triangle init above.
        self._apply_com_not_above_prior()

    # ---- Matrix-based helpers ----
    def _apply_com_not_above_prior(self) -> None:
        n = self.n
        for a in range(n):
            com_a = self._com_values[a]
            for b in range(n):
                if a == b:
                    continue
                com_b = self._com_values[b]
                # Strict relation uses ">".
                # If CoM is exactly equal, neither direction can be strictly above.
                if com_a == com_b:
                    if self.M[a][b] == -2:
                        self.M[a][b] = -1
                    if self.M[b][a] == -2:
                        self.M[b][a] = -1
                    continue
                if com_a > com_b:
                    # b cannot be above a
                    if self.M[b][a] == -2:
                        self.M[b][a] = -1

    def _is_left_right_symmetric_pair(self, i: int, j: int) -> bool:
        """True if i and j are symmetric Left/Right partners for the vertical axis."""
        if self.axis != AXIS_VERTICAL:
            return False
        return self._symmetric_partner.get(i) == j

    def _enforce_symmetric_no_constraints(self) -> None:
        """Re-apply strict NO (-1) for left/right symmetric pairs."""
        if self.axis != AXIS_VERTICAL or not self._symmetric_partner:
            return
        for i, j in self._symmetric_partner.items():
            if i != j:
                self.M[i][j] = -1

    def _merge_cell(self, a: int, b: int, new_val: int) -> None:
        """
        Merge logic:
        - never let -2 overwrite a more informative existing value
        - symmetric Left/Right same-core pairs are always forced to -1
        """
        if self._is_left_right_symmetric_pair(a, b):
            self.M[a][b] = -1
            return

        # If new_val is "not asked", keep any prior knowledge.
        if new_val == -2 and self.M[a][b] != -2:
            return

        self.M[a][b] = new_val

    def _enforce_vertical_symmetry_consistency(self) -> None:
        """
        Guarantee that whenever one side of a bilateral comparison becomes
        known (+1/0/-1), the corresponding Left/Right mirrored cells become
        known with the same value.

        This prevents situations where (Left core -> X) is answered but
        (Right core -> X) remains -2.
        """
        if self.axis != AXIS_VERTICAL or not self._symmetric_partner:
            return

        n = self.n
        for i in range(n):
            mi = self._symmetric_partner.get(i)
            for j in range(n):
                mj = self._symmetric_partner.get(j)

                val = self.M[i][j]
                # Source mirror (only if source has a partner)
                if mi is not None:
                    self._merge_cell(mi, j, val)
                # Target mirror (only if target has a partner)
                if mj is not None:
                    self._merge_cell(i, mj, val)
                # Double mirror (only if both sides have partners)
                if mi is not None and mj is not None:
                    self._merge_cell(mi, mj, val)

        # Maintain asymmetry for explicitly set values.
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if self.M[a][b] == 1 and self.M[b][a] == -2:
                    self.M[b][a] = -1
                elif self.M[a][b] == 0 and self.M[b][a] == -2:
                    self.M[b][a] = 0

        # Finally, re-force symmetric pair strict NO
        self._enforce_symmetric_no_constraints()

    def record_response_matrix(self, i: int, j: int, value: int) -> None:
        """
        value ∈ {+1, -1, 0}.

        +1 -> i above j
         0 -> not sure
        -1 -> i not strictly above j
        """
        if value not in (-1, 0, 1):
            raise ValueError(f"Invalid relation value {value}; expected -1, 0, or +1.")

        self.M[i][j] = value

        # Symmetry propagation (mirror Left/Right counterparts for vertical axis)
        assigned_pairs: List[Tuple[int, int]] = [(i, j)]
        if self.axis == AXIS_VERTICAL:
            mi = self._symmetric_partner.get(i)
            mj = self._symmetric_partner.get(j)

            if mi is not None:
                self.M[mi][j] = value
                assigned_pairs.append((mi, j))
            if mj is not None:
                self.M[i][mj] = value
                assigned_pairs.append((i, mj))
            if mi is not None and mj is not None:
                # Double-mirror: if (i, j) is answered, propagate to the
                # symmetric pair (s(i), s(j)) as well.
                # If this happens to be the Left/Right *same core* pair,
                # hard enforce NO (-1).
                if self._is_left_right_symmetric_pair(mi, mj):
                    self.M[mi][mj] = -1
                else:
                    self.M[mi][mj] = value
                assigned_pairs.append((mi, mj))

        # Strict "above" is asymmetric:
        # if a relation is explicitly +1, the inverse must be NO (-1).
        if value == 1:
            for a, b in assigned_pairs:
                if self.M[a][b] == 1 and self.M[b][a] == -2:
                    self.M[b][a] = -1
        # If the user answered "not sure" (0), treat the inverse query as
        # also not sure unless it was already decided.
        if value == 0:
            for a, b in assigned_pairs:
                if self.M[a][b] == 0 and self.M[b][a] == -2:
                    self.M[b][a] = 0

        # Run inference after every update
        self._propagate()

    def record_unknown(self, i: int, j: int) -> None:
        """Explicitly mark a queried pair as unknown/not sure."""
        self.record_response_matrix(i, j, 0)

    def path_exists_matrix(self, start: int, end: int) -> bool:
        """
        Reachability using only +1 relations in M.
        """
        stack = [start]
        visited: Set[int] = set()

        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u)
            for v in range(self.n):
                if self.M[u][v] == 1:
                    if v == end:
                        return True
                    stack.append(v)
        return False

    def _propagate(self) -> None:
        """
        Transitive closure / propagation on M:
        - Transitivity for +1: i→j and j→k ⇒ i→k
        - Never infer +1 into symmetric Left/Right pairs (vertical axis)
        - Never infer +1 that contradicts the CoM "not above" prior
        - Maintain asymmetry: if i→k becomes +1, then k→i becomes -1
        """
        changed = True
        while changed:
            changed = False
            for i in range(self.n):
                for j in range(self.n):
                    if self.M[i][j] != 1:
                        continue
                    for k in range(self.n):
                        # Need a +1 chain: i→j and j→k
                        if self.M[j][k] != 1:
                            continue

                        # Don't overwrite explicit NO (-1)
                        if self.M[i][k] == -1:
                            continue

                        # Symmetry hard constraint: same core Left/Right cannot be strictly ordered.
                        if self._is_left_right_symmetric_pair(i, k):
                            continue

                        # CoM prior: to be strictly above, com(i) must be strictly greater than com(k).
                        if self._com_values[i] <= self._com_values[k]:
                            continue

                        if self.M[i][k] != 1:
                            self.M[i][k] = 1
                            # Strict above is asymmetric.
                            if self.M[k][i] == -2:
                                self.M[k][i] = -1
                            changed = True

        # After propagation, keep self.edges in sync with +1 entries
        # and re-apply the strict NO constraint for symmetric pairs.
        self._enforce_symmetric_no_constraints()
        self._enforce_vertical_symmetry_consistency()
        self.edges = self.get_pdag()

    # ---- Query iteration using M ----
    def next_pair(self) -> Tuple[int, int] | None:  # type: ignore[override]
        """
        Same gap-based iteration as PosetBuilder, but:
        - skips pairs where M[i][j] != -2 (already answered)
        - also skips pairs whose relation is implied by transitivity
        """
        if self.finished:
            return None

        while self.current_gap <= self.n - 1:
            while self.current_i <= self.n - 1 - self.current_gap:
                i = self.current_i
                j = i + self.current_gap
                self.current_i += 1

                if self.axis == AXIS_VERTICAL:
                    pi = self._symmetric_partner.get(i)
                    pj = self._symmetric_partner.get(j)
                    if pi is not None and pi < i:
                        continue
                    if pj is not None and pj < j:
                        continue
                    if pi is not None and pi == j:
                        continue

                # Skip if already answered in any way
                if self.M[i][j] != -2:
                    continue

                # Skip if relation is already implied by transitivity in either direction
                if self.path_exists_matrix(i, j) or self.path_exists_matrix(j, i):
                    continue

                return i, j

            self.current_gap += 1
            self.current_i = 0

        self.finished = True
        return None

    def get_iteration_progress(self) -> float:  # type: ignore[override]
        """
        Progress based on how many unordered pairs {i, j} have some annotation
        (M[i][j] != -2), 0.0 to 1.0.
        """
        if self.n <= 1:
            return 1.0
        total = self.n * (self.n - 1) // 2
        if total == 0:
            return 1.0

        answered = 0
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.M[i][j] != -2:
                    answered += 1

        return min(1.0, answered / total)

    # ---- Graph views derived from M ----
    def get_pdag(self) -> Set[Tuple[int, int]]:
        """
        Partial DAG: all +1 relations as directed edges.
        May contain cycles if annotations are inconsistent.
        """
        edges: Set[Tuple[int, int]] = set()
        for i in range(self.n):
            for j in range(self.n):
                if self.M[i][j] == 1 and i != j:
                    edges.add((i, j))
        return edges

    def _has_cycle(self) -> bool:
        """
        Very simple cycle detection: check reachability from i back to i.
        """
        for i in range(self.n):
            if self.path_exists_matrix(i, i):
                return True
        return False

    def get_hasse(self) -> Set[Tuple[int, int]]:
        """
        Hasse diagram edges derived from the current +1 relations in M.
        Raises if the underlying graph is cyclic.
        """
        if self._has_cycle():
            raise ValueError("Cannot build Hasse diagram from cyclic graph.")
        self.edges = self.get_pdag()
        return self.edge_redundancy_reduction()


def aggregate_matrices(matrices: List[List[List[int]]]) -> List[List[float]]:
    """
    Aggregate multiple tri-valued matrices M into an averaged weight matrix W.

    For each (i, j):
      - ignore entries where M[i][j] == -2 (never asked)
      - otherwise, average the values across raters/sessions
    """
    if not matrices:
        return []

    n = len(matrices[0])
    W: List[List[float]] = [[0.0 for _ in range(n)] for _ in range(n)]
    C: List[List[int]] = [[0 for _ in range(n)] for _ in range(n)]

    for M in matrices:
        if len(M) != n:
            raise ValueError("All matrices must have the same size.")
        for i in range(n):
            if len(M[i]) != n:
                raise ValueError("All matrices must be square n x n.")
            for j in range(n):
                if M[i][j] != -2:
                    W[i][j] += float(M[i][j])
                    C[i][j] += 1

    for i in range(n):
        for j in range(n):
            if C[i][j] > 0:
                W[i][j] /= C[i][j]

    return W


