from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .axis_models import AXIS_ANTERIOR_POSTERIOR, AXIS_MEDIOLATERAL, AXIS_VERTICAL, Structure


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


class MatrixBuilder:
    """
    Algorithm 1 (gap-based CoM strategy) with a tri-valued relation matrix M:

        M[i][j] = +1  -> "i is (strictly) above j"  (YES)
        M[i][j] =  0  -> "not sure / unknown but asked"
        M[i][j] = -1  -> "i is not strictly above j" (NO / overlap / opposite)
        M[i][j] = -2  -> not asked yet

    At construction, structures are sorted by axis CoM (descending). The lower
    triangle (i > j) and diagonal are set to -1; only the strict upper triangle
    starts as -2. Equal-CoM pairs are completed by _apply_com_not_above_prior().

    The underlying DAG used for Hasse diagrams is still derived solely from
    the +1 entries; the matrix simply preserves richer annotation state.

    Optional ``query_allowed_indices`` (constructor keyword): if provided, :meth:`next_pair`
    only returns pairs ``(i, j)`` with both endpoints in that index set. The matrix
    remains ``n×n`` for the full structure list so saved JSON stays merge-compatible
    across raters; cells outside the query subset may stay ``-2`` or be filled by
    propagation only.
    """

    def __init__(
        self,
        structures: List[Structure],
        axis: str = AXIS_VERTICAL,
        *,
        query_allowed_indices: Optional[Set[int]] = None,
    ) -> None:
        if axis == AXIS_MEDIOLATERAL:
            key = lambda s: s.com_lateral
        elif axis == AXIS_ANTERIOR_POSTERIOR:
            key = lambda s: s.com_anteroposterior
        else:
            key = lambda s: s.com_vertical
        self.structures = sorted(structures, key=key, reverse=True)
        self.n = len(self.structures)
        self.axis = axis

        self.edges: Set[Tuple[int, int]] = set()
        self.skipped_pairs: Set[Tuple[int, int]] = set()

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

        self.current_gap = 1
        self.current_i = 0
        self.finished = self.n <= 1

        self._query_allowed_indices: Optional[Set[int]] = query_allowed_indices
        # Structures are sorted by this axis CoM descending.
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

    def record_skip(self, i: int, j: int) -> None:
        """
        Mark a pair as skipped outside the main matrix flow (e.g. bilateral core dedup).
        """
        if i == j:
            return
        a, b = (i, j) if i < j else (j, i)
        self.skipped_pairs.add((a, b))

    def estimate_remaining_questions(self) -> int:
        """
        Worst-case count of further gap-iteration questions from the current state,
        using transitive reachability on ``self.edges`` (kept in sync with M).
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
                if self.path_exists_matrix(s, t):
                    continue

                remaining += 1

            g += 1
            i = 0

        return remaining

    def seal_lower_triangle_com_prior(self) -> None:
        """
        Re-apply CoM-based lower triangle (-1), bilateral symmetric NOs, and tie
        closure. Call before persisting so JSON does not keep spurious `-2`
        below the diagonal from older partial saves or loaded files.
        """
        n = self.n
        for i in range(n):
            for j in range(i):
                self.M[i][j] = -1
        if self.axis == AXIS_VERTICAL and self._symmetric_partner:
            for i, j in self._symmetric_partner.items():
                if i != j:
                    self.M[i][j] = -1
        self._apply_com_not_above_prior()
        if self.axis == AXIS_VERTICAL and self._symmetric_partner:
            self._sync_vertical_bilateral_mirrors()

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

    def _sync_vertical_bilateral_mirrors(self) -> None:
        """
        Left/Right same-core pairs must end with identical tri-values for mirrored
        relations: (Left, X) matches (Right, X) and (Y, Left) matches (Y, Right).

        Uses the smaller index within each bilateral pair as tie-breaker when both
        sides were already non--2 but disagree (should be rare).
        """
        if self.axis != AXIS_VERTICAL or not self._symmetric_partner:
            return

        n = self.n
        changed = True
        while changed:
            changed = False
            # Row partners: for each column k, M[a][k] == M[b][k]
            for i in range(n):
                mi = self._symmetric_partner.get(i)
                if mi is None or i > mi:
                    continue
                a, b = i, mi
                for k in range(n):
                    va, vb = self.M[a][k], self.M[b][k]
                    if va == -2 and vb == -2:
                        continue
                    if va != -2 and vb == -2:
                        self.M[b][k] = va
                        changed = True
                    elif vb != -2 and va == -2:
                        self.M[a][k] = vb
                        changed = True
                    elif va != -2 and vb != -2 and va != vb:
                        self.M[b][k] = va
                        changed = True
            # Column partners: for each row k, M[k][a] == M[k][b]
            for j in range(n):
                mj = self._symmetric_partner.get(j)
                if mj is None or j > mj:
                    continue
                a, b = j, mj
                for k in range(n):
                    va, vb = self.M[k][a], self.M[k][b]
                    if va == -2 and vb == -2:
                        continue
                    if va != -2 and vb == -2:
                        self.M[k][b] = va
                        changed = True
                    elif vb != -2 and va == -2:
                        self.M[k][a] = vb
                        changed = True
                    elif va != -2 and vb != -2 and va != vb:
                        self.M[k][b] = va
                        changed = True

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

        # Copy non--2 answers across bilateral mirrors (handles propagation / close
        # filling one side but not the other; _merge_cell alone cannot copy -2 over known).
        self._sync_vertical_bilateral_mirrors()

        n = self.n
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

        # Transitive +1 inference above is blocked when com(i) <= com(k) even if a +1 path
        # exists in M (user answers can disagree with raw CoM). next_pair() would still
        # skip such pairs because path_exists_matrix is True, leaving M[i][j] == -2 forever.
        self._close_transitive_unknowns()

        # After propagation, keep self.edges in sync with +1 entries
        # and re-apply the strict NO constraint for symmetric pairs.
        self._enforce_symmetric_no_constraints()
        self._enforce_vertical_symmetry_consistency()
        self.edges = self.get_pdag()

    def _close_transitive_unknowns(self) -> None:
        """
        For any cell still -2, if reachability on +1 edges shows i→…→j or j→…→i,
        set the directed cells to match (so next_pair and saved matrices stay consistent).
        """
        n = self.n
        changed = True
        while changed:
            changed = False
            for i in range(n):
                for j in range(n):
                    if i == j or self.M[i][j] != -2:
                        continue
                    ij = self.path_exists_matrix(i, j)
                    ji = self.path_exists_matrix(j, i)
                    if ij and ji:
                        # Cyclic reachability on +1 — leave ambiguous; do not guess.
                        continue
                    if ij:
                        self.M[i][j] = 1
                        if self.M[j][i] == -2:
                            self.M[j][i] = -1
                        changed = True
                    elif ji:
                        self.M[i][j] = -1
                        if self.M[j][i] == -2:
                            self.M[j][i] = 1
                        changed = True

    # ---- Query iteration using M ----
    def next_pair(self) -> Tuple[int, int] | None:
        """
        Gap-based iteration:
        - skips pairs where M[i][j] != -2 (already answered)
        - skips pairs whose relation is implied by transitivity on +1
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

                # Subset runs: only ask pairs whose endpoints are both in the allowed index set
                if self._query_allowed_indices is not None:
                    if i not in self._query_allowed_indices or j not in self._query_allowed_indices:
                        continue

                # Skip if already answered in any way
                if self.M[i][j] != -2:
                    continue

                # Transitive reachability on +1 (after _close_transitive_unknowns in _propagate,
                # ij xor ji should have closed M[i][j]; if both directions exist, +1 graph has a cycle).
                ij = self.path_exists_matrix(i, j)
                ji = self.path_exists_matrix(j, i)
                if ij or ji:
                    if ij and ji:
                        # Cycle: we cannot infer; still ask the expert
                        return i, j
                    continue

                return i, j

            self.current_gap += 1
            self.current_i = 0

        self.finished = True
        return None

    def restore_matrix(self, M: List[List[int]]) -> None:
        """Replace M and re-run propagation so ``edges`` and invariants stay consistent."""
        if len(M) != self.n or any(len(row) != self.n for row in M):
            raise ValueError("Matrix shape must match current structure count.")
        self.M = [row[:] for row in M]
        self._propagate()

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

