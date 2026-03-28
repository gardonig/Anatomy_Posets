"""
Aggregate multiple relation matrices from different raters / sessions (or merged summaries).

Inputs may be **tri-valued** ``{-2,-1,0,+1}`` and/or **probability** cells in ``[0, 1]``
(see :func:`aggregate_matrices_with_counts`). :class:`CellAggregate` keeps per-cell
``counts``, ``mean``, ``n_answered``, and ``n_notasked`` so raw vote detail is not dropped
until you **project** to a single number (e.g. :func:`aggregate_to_p_yes_matrix`), which
is a lossy summary by design.

**Merge semantics:** Off-diagonal cells where **no** file contributes an answer stay
``null`` / NaN in the saved / display P output — nothing is invented there. However,
**canonical sort + lower-triangle sealing** overwrites geometrically forbidden cells with
``-1`` or ``0.0`` before aggregation (that is imposed structure, not measured data).
Merging **probability** files treats each file as one summary observation per cell, which
does not recreate pooling all underlying raters unless you merge raw tri-valued sessions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .axis_models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)

# n×n matrices per file: tri-valued ints in {-2,-1,0,1} and/or probability summaries in [0, 1]
TriMatricesPerFile = List[List[List[Union[int, float]]]]

# JSON round-trip and different serializers can change float bits slightly; CoMs are in [0, 100].
_COM_RTOL = 1e-9
_COM_ATOL = 1e-5


def structure_list_signature(structures: Sequence[Structure]) -> Tuple[Tuple[Any, ...], ...]:
    """
    Canonical tuple for compatibility: (name, v, ml, ap) per index.

    Note: merge uses :func:`align_matrix_lists_to_reference`, which allows small CoM drift
    and optional reordering; this tuple still uses raw floats (exact equality).
    """
    return tuple(
        (s.name.strip(), float(s.com_vertical), float(s.com_lateral), float(s.com_anteroposterior))
        for s in structures
    )


def _norm_name(name: str) -> str:
    return name.strip()


def _coords_close(a: Structure, b: Structure) -> bool:
    return (
        math.isclose(a.com_vertical, b.com_vertical, rel_tol=_COM_RTOL, abs_tol=_COM_ATOL)
        and math.isclose(a.com_lateral, b.com_lateral, rel_tol=_COM_RTOL, abs_tol=_COM_ATOL)
        and math.isclose(
            a.com_anteroposterior,
            b.com_anteroposterior,
            rel_tol=_COM_RTOL,
            abs_tol=_COM_ATOL,
        )
    )


def _pair_matches_reference(ref_s: Structure, other_s: Structure) -> bool:
    """Same structure row: same name and matching CoMs (within tolerance)."""
    if _norm_name(ref_s.name) != _norm_name(other_s.name):
        return False
    return _coords_close(ref_s, other_s)


def structures_match_same_order(ref: List[Structure], other: List[Structure]) -> Tuple[bool, str]:
    """Same index i: same name and CoMs within tolerance."""
    if len(ref) != len(other):
        return False, f"different lengths ({len(ref)} vs {len(other)})"
    for i, (a, b) in enumerate(zip(ref, other)):
        if _norm_name(a.name) != _norm_name(b.name):
            return False, f"index {i}: name {a.name!r} vs {b.name!r}"
        if not _coords_close(a, b):
            return (
                False,
                f"index {i} ({a.name}): CoM ref=({a.com_vertical}, {a.com_lateral}, {a.com_anteroposterior}) "
                f"vs ({b.com_vertical}, {b.com_lateral}, {b.com_anteroposterior})",
            )
    return True, ""


def find_alignment_permutation(
    ref: List[Structure], other: List[Structure]
) -> Tuple[Optional[List[int]], str]:
    """
    Find ``perm`` such that ``other[perm[i]]`` matches ``ref[i]`` (name + CoM tolerance).

    Returns permutation of indices into ``other`` (one unique bijection), or None.
    """
    if len(ref) != len(other):
        return None, f"different lengths ({len(ref)} vs {len(other)})"
    n = len(ref)
    # cand[i] = list of j in other that match ref[i]
    cand: List[List[int]] = []
    for i in range(n):
        matches = [j for j in range(n) if _pair_matches_reference(ref[i], other[j])]
        if not matches:
            return (
                None,
                f"no matching row for ref index {i} ({ref[i].name!r}, "
                f"CoM≈({ref[i].com_vertical:.6g}, {ref[i].com_lateral:.6g}, {ref[i].com_anteroposterior:.6g}))",
            )
        cand.append(matches)

    used: set[int] = set()
    perm: List[int] = [-1] * n

    def dfs(i: int) -> bool:
        if i == n:
            return True
        for j in cand[i]:
            if j in used:
                continue
            used.add(j)
            perm[i] = j
            if dfs(i + 1):
                return True
            used.remove(j)
            perm[i] = -1
        return False

    if dfs(0):
        return perm, ""
    return (
        None,
        "cannot uniquely match structures (duplicate names with overlapping CoMs, or conflicting rows).",
    )


def permute_relation_matrix(
    M: List[List[Union[int, float]]], perm: List[int]
) -> List[List[Union[int, float]]]:
    """
    Reindex rows/columns so structure at new index i is the former structure perm[i].

    Semantics preserved: ``out[i][j]`` is the relation from (new) structure i to (new) j,
    equal to ``M[perm[i]][perm[j]]`` — the same directed cell in the old indexing.

    This matches the permutation of basis vectors P M P in matrix terms (same convention
    as NumPy ``A[np.ix_(perm, perm)]`` when perm lists old row/col index for each new row/col).
    """
    n = len(M)
    out: List[List[Union[int, float]]] = [[-2] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = M[perm[i]][perm[j]]
    return out


def permutation_matrix_order_to_target(
    structures_target: List[Structure],
    structures_source: List[Structure],
) -> List[int]:
    """
    ``perm[i]`` = index in ``structures_source`` of the same structure as
    ``structures_target[i]`` (matched by name + CoMs via :func:`structure_list_signature`).
    """
    n = len(structures_target)
    if len(structures_source) != n:
        raise ValueError(
            f"Structure list length mismatch: target {n} vs source {len(structures_source)}"
        )
    by_sig: Dict[Tuple[Any, ...], int] = {}
    for j, t in enumerate(structures_source):
        sig = structure_list_signature([t])[0]
        if sig in by_sig:
            raise ValueError("Duplicate structure in source ordering (same name + CoMs).")
        by_sig[sig] = j
    perm: List[int] = []
    for s in structures_target:
        sig = structure_list_signature([s])[0]
        if sig not in by_sig:
            raise ValueError(
                f"No matching structure in source order for {s.name!r} in target list."
            )
        perm.append(by_sig[sig])
    return perm


def reindex_matrix_to_structure_order(
    structures_target: List[Structure],
    structures_matrix_order: List[Structure],
    M: List[List[Union[int, float]]],
) -> List[List[Union[int, float]]]:
    """
    ``M`` is indexed by ``structures_matrix_order`` (rows/cols); return the same
    relation matrix indexed by ``structures_target`` (same anatomical set, new order).

    Used when saving merged posets: one ``structures`` list (e.g. vertical CoM order)
    with all three matrices expressed in that indexing.
    """
    perm = permutation_matrix_order_to_target(structures_target, structures_matrix_order)
    return permute_relation_matrix(M, perm)


def reindex_count_matrix_to_structure_order(
    structures_target: List[Structure],
    structures_matrix_order: List[Structure],
    M: List[List[Optional[int]]],
) -> List[List[Optional[int]]]:
    """Same reindexing as :func:`reindex_matrix_to_structure_order` for optional-int count grids."""
    n = len(structures_target)
    perm = permutation_matrix_order_to_target(structures_target, structures_matrix_order)
    out: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            out[i][j] = M[perm[i]][perm[j]]
    return out


def canonical_sort_permutation_for_axis(structures: List[Structure], axis: str) -> List[int]:
    """
    Indices sorted by the chosen axis CoM descending, matching :class:`MatrixBuilder`.

    Uses stable sort on ``enumerate`` so tie-breaking matches the builder (preserve
    relative order among equal CoMs on that axis).
    """
    indexed = list(enumerate(structures))
    if axis == AXIS_MEDIOLATERAL:
        indexed.sort(key=lambda p: p[1].com_lateral, reverse=True)
    elif axis == AXIS_ANTERIOR_POSTERIOR:
        indexed.sort(key=lambda p: p[1].com_anteroposterior, reverse=True)
    else:
        indexed.sort(key=lambda p: p[1].com_vertical, reverse=True)
    return [p[0] for p in indexed]


def matrix_has_float_probability_entries(M: List[List[Any]]) -> bool:
    """True if any off-diagonal cell is a ``float`` (excludes ``bool``). Used to choose seal fill."""
    n = len(M)
    for i in range(n):
        row = M[i]
        for j in range(min(n, len(row))):
            if i == j:
                continue
            v = row[j]
            if isinstance(v, float) and not isinstance(v, bool):
                return True
    return False


def enforce_axis_lower_triangle_inplace(M: List[List[Any]]) -> None:
    """
    After indices follow axis CoM descending (see :class:`MatrixBuilder`), ``i > j``
    cannot be a strict ``+1`` relation along that axis.

    - **Tri-valued** matrices: lower triangle is set to ``-1``.
    - **Probability** matrices (any off-diagonal ``float``): lower triangle is set to ``0.0``
      (``P = 0``), so later aggregation does not treat sealed cells as discrete ``-1`` votes.
    """
    n = len(M)
    fill: Union[int, float] = 0.0 if matrix_has_float_probability_entries(M) else -1
    for i in range(n):
        for j in range(i):
            M[i][j] = fill


def apply_canonical_per_axis_orders(
    structures: List[Structure],
    mv_list: TriMatricesPerFile,
    ml_list: TriMatricesPerFile,
    ap_list: TriMatricesPerFile,
) -> Tuple[
    List[Structure],
    List[Structure],
    List[Structure],
    TriMatricesPerFile,
    TriMatricesPerFile,
    TriMatricesPerFile,
]:
    """
    Reorder each axis matrix by **that** axis's CoM descending (as in :class:`MatrixBuilder`).

    Call after :func:`align_matrix_lists_to_reference`. Vertical, mediolateral, and
    anteroposterior matrices each get their own permutation; structure row labels for
    each tab should use the corresponding returned list ``(sv, sml, sap)``.

    Seals the lower triangle on each per-rater matrix after reordering.
    """
    if not structures:
        return [], [], [], mv_list, ml_list, ap_list
    n = len(structures)
    perm_v = canonical_sort_permutation_for_axis(structures, AXIS_VERTICAL)
    perm_ml = canonical_sort_permutation_for_axis(structures, AXIS_MEDIOLATERAL)
    perm_ap = canonical_sort_permutation_for_axis(structures, AXIS_ANTERIOR_POSTERIOR)

    new_mv: TriMatricesPerFile = []
    for M in mv_list:
        m = permute_relation_matrix(M, perm_v)
        enforce_axis_lower_triangle_inplace(m)
        new_mv.append(m)
    new_ml: TriMatricesPerFile = []
    for M in ml_list:
        m = permute_relation_matrix(M, perm_ml)
        enforce_axis_lower_triangle_inplace(m)
        new_ml.append(m)
    new_ap: TriMatricesPerFile = []
    for M in ap_list:
        m = permute_relation_matrix(M, perm_ap)
        enforce_axis_lower_triangle_inplace(m)
        new_ap.append(m)

    sv = [structures[perm_v[i]] for i in range(n)]
    sml = [structures[perm_ml[i]] for i in range(n)]
    sap = [structures[perm_ap[i]] for i in range(n)]

    return sv, sml, sap, new_mv, new_ml, new_ap


def align_matrix_lists_to_reference(
    structures_list: List[List[Structure]],
    mv_list: TriMatricesPerFile,
    ml_list: TriMatricesPerFile,
    ap_list: TriMatricesPerFile,
) -> Tuple[bool, str, TriMatricesPerFile, TriMatricesPerFile, TriMatricesPerFile]:
    """
    Align all matrices to the structure order of ``structures_list[0]``.

    Uses same-order matching with CoM tolerance; if that fails, tries a unique permutation
    of each file's indices so that structure identity (name + CoM) matches the reference.
    """
    if not structures_list:
        return False, "No structure lists provided.", [], [], []
    if not (len(mv_list) == len(ml_list) == len(ap_list) == len(structures_list)):
        return False, "Mismatched number of files (structures vs matrices).", [], [], []
    ref = structures_list[0]
    out_v: TriMatricesPerFile = [mv_list[0]]
    out_ml: TriMatricesPerFile = [ml_list[0]]
    out_ap: TriMatricesPerFile = [ap_list[0]]

    for idx in range(1, len(structures_list)):
        lst = structures_list[idx]
        ok, _ = structures_match_same_order(ref, lst)
        if ok:
            out_v.append(mv_list[idx])
            out_ml.append(ml_list[idx])
            out_ap.append(ap_list[idx])
            continue
        perm, err = find_alignment_permutation(ref, lst)
        if perm is None:
            return (
                False,
                f"File #{idx + 1}: {err}",
                [],
                [],
                [],
            )
        out_v.append(permute_relation_matrix(mv_list[idx], perm))
        out_ml.append(permute_relation_matrix(ml_list[idx], perm))
        out_ap.append(permute_relation_matrix(ap_list[idx], perm))

    return True, "", out_v, out_ml, out_ap


@dataclass
class CellAggregate:
    """Per-directed-cell summary across K matrices (same (i,j) everywhere)."""

    mean: float
    n_answered: int
    n_notasked: int
    counts: Dict[int, int] = field(default_factory=dict)
    """Counts for values in {-1, 0, +1} among answered cells."""

    @property
    def probability_yes_green(self) -> float:
        """
        Map mean answer in [-1, 1] to [0, 1] for red (-1) → green (+1) coloring.
        """
        return max(0.0, min(1.0, (self.mean + 1.0) / 2.0))


def aggregate_matrices_with_counts(
    matrices: List[List[List[Union[int, float]]]],
) -> Tuple[List[List[CellAggregate]], int]:
    """
    Build a grid of CellAggregate for each (i, j).

    - Entries with -2 in a matrix count as "not asked" for that rater.
    - **Tri-valued** answered entries (``int`` in ``{-1, 0, +1}``) contribute to vote
      counts and to the arithmetic mean **μ** of codes.
    - **Probability summaries** (``float`` in ``[0, 1]``), e.g. from an already merged JSON,
      contribute one pseudo-observation with code ``μ_k = 2 P_k - 1`` (so averaging **P**
      across files with the same weight equals averaging these **μ** and mapping back with
      ``(μ+1)/2``). **This is not** the same as merging all underlying raw raters unless each
      file stands for one equal-weight cohort and you only care about this summary rule.

    **Per-cell independence:** There is no propagation across cells during merge. Each
    ``(i, j)`` is aggregated from that cell in each file only.

    **Partial overlap:** If rater A has ``0`` (unsure) at ``(i, j)`` and rater B still has
    ``-2`` (never asked) there, the mean uses **only** A's answer → ``μ = 0`` →
    ``P(yes) = 0.5``. That is **not** the same as "unsure + yes" from two raters (which
    would need both answered, giving e.g. ``μ = 0.5`` from ``0`` and ``+1``, hence
    ``P = 0.75``). Many such cells in the same row/column can look like a **stripe** of
    orange in the merged heatmap even though each cell is computed correctly.
    """
    if not matrices:
        return [], 0
    n = len(matrices[0])
    for M in matrices:
        if len(M) != n:
            raise ValueError("All matrices must have the same dimension n.")
        for row in M:
            if len(row) != n:
                raise ValueError("All matrices must be square n x n.")

    K = len(matrices)
    out: List[List[CellAggregate]] = []

    for i in range(n):
        row_out: List[CellAggregate] = []
        for j in range(n):
            if i == j:
                row_out.append(
                    CellAggregate(mean=-1.0, n_answered=K, n_notasked=0, counts={-1: K})
                )
                continue
            counts: Dict[int, int] = {-1: 0, 0: 0, 1: 0}
            n_notasked = 0
            total = 0.0
            n_ans = 0
            for M in matrices:
                v = M[i][j]
                if isinstance(v, bool):
                    n_notasked += 1
                    continue
                if v is None:
                    n_notasked += 1
                    continue
                # Probability summary: one pseudo-sample with code 2P - 1 (no tri-valued counts).
                if isinstance(v, float) and 0.0 <= v <= 1.0:
                    total += 2.0 * v - 1.0
                    n_ans += 1
                    continue
                if isinstance(v, int):
                    if v == -2:
                        n_notasked += 1
                        continue
                    if v in (-1, 0, 1):
                        counts[v] = counts.get(v, 0) + 1
                        total += float(v)
                        n_ans += 1
                        continue
                    n_notasked += 1
                    continue
                try:
                    fv = float(v)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    n_notasked += 1
                    continue
                if 0.0 <= fv <= 1.0:
                    total += 2.0 * fv - 1.0
                    n_ans += 1
                elif abs(fv - round(fv)) < 1e-9:
                    iv = int(round(fv))
                    if iv == -2:
                        n_notasked += 1
                    elif iv in (-1, 0, 1):
                        counts[iv] = counts.get(iv, 0) + 1
                        total += float(iv)
                        n_ans += 1
                    else:
                        n_notasked += 1
                else:
                    n_notasked += 1
            mean = total / n_ans if n_ans > 0 else 0.0
            row_out.append(
                CellAggregate(
                    mean=mean,
                    n_answered=n_ans,
                    n_notasked=n_notasked,
                    counts=counts,
                )
            )
        out.append(row_out)
    return out, K


def cell_aggregate_to_display_matrix(
    agg: List[List[CellAggregate]],
    merge_k: Optional[int] = None,
) -> Tuple[List[List[float]], List[List[str]], List[List[bool]]]:
    """
    Float matrix for imshow and per-cell annotation strings (merged heatmap).

    Uses ``P = (mean + 1) / 2`` from :attr:`CellAggregate.probability_yes_green` over
    answered contributions only (same rule as :func:`aggregate_to_p_yes_matrix`).

    Off-diagonal cells with no answers use NaN (grey). Diagonal maps to ``0.0``.
    ``tie_mask`` is always false (kept for call-site compatibility).

    ``merge_k``: when set, annotations show ``answered/total`` for partial overlap visibility.
    """
    n = len(agg)
    nan = float("nan")
    Z = [[nan for _ in range(n)] for _ in range(n)]
    ann = [["" for _ in range(n)] for _ in range(n)]
    tie_mask = [[False for _ in range(n)] for _ in range(n)]

    for i in range(n):
        for j in range(n):
            c = agg[i][j]
            if i == j:
                ann[i][j] = "diag: −1 (n/a)"
                Z[i][j] = 0.0  # P_yes for −1
                continue
            if c.n_answered == 0:
                Z[i][j] = nan
                ann[i][j] = "no data"
                continue

            parts: List[str] = []
            parts.append(f"μ={c.mean:.2f}")
            if merge_k is not None and merge_k > 0:
                parts.append(f"answered {c.n_answered}/{merge_k}")
            else:
                parts.append(f"n={c.n_answered}")
            ch = ", ".join(f"{k:+d}:{v}" for k, v in sorted(c.counts.items()) if v > 0)
            if ch:
                parts.append(ch)
            if c.n_notasked > 0:
                parts.append(f"na={c.n_notasked}")
            if merge_k is not None and merge_k > 1 and c.n_answered < merge_k:
                parts.append("μ uses answered only (−2 excluded)")

            Z[i][j] = c.probability_yes_green
            ann[i][j] = "\n".join(parts)
    return Z, ann, tie_mask


def aggregate_to_p_yes_matrix(agg: List[List[CellAggregate]]) -> List[List[Optional[float]]]:
    """
    **Probability consensus** over raters: for each directed cell, ``P(yes) = (μ + 1) / 2`` where
    **μ** is the mean of answered codes only (``-2`` excluded per rater). Same convention as the
    merged ``P(yes)`` heatmap and :attr:`CellAggregate.probability_yes_green`.

    - Diagonal: ``0.0`` (relation −1 for self maps to ``P = 0``).
    - Off-diagonal, no answered raters: ``None`` (JSON ``null`` when saved).
    """
    n = len(agg)
    out: List[List[Optional[float]]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                out[i][j] = 0.0
                continue
            c = agg[i][j]
            if c.n_answered == 0:
                out[i][j] = None
            else:
                out[i][j] = c.probability_yes_green
    return out


def aggregate_to_n_answered_matrix(agg: List[List[CellAggregate]]) -> List[List[Optional[int]]]:
    """
    Per-cell number of files/raters that contributed an answer at ``(i, j)`` (same rule as
    :attr:`CellAggregate.n_answered`). Saved next to ``P`` so merged JSON is not fully lossy.

    - Diagonal: merge depth ``K`` (same as synthetic diagonal aggregate).
    - Off-diagonal, none answered: ``None`` (JSON ``null``).
    """
    n = len(agg)
    out: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c = agg[i][j]
            if i == j:
                out[i][j] = c.n_answered
                continue
            if c.n_answered == 0:
                out[i][j] = None
            else:
                out[i][j] = c.n_answered
    return out


def aggregate_to_n_notasked_matrix(agg: List[List[CellAggregate]]) -> List[List[Optional[int]]]:
    """
    Per-cell count of files/raters with ``-2`` / missing at ``(i, j)``.

    - Diagonal: ``0``.
    - Off-diagonal: always an int in ``[0, K]`` (use with ``n_answered`` to recover ``K``).
    """
    n = len(agg)
    out: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            c = agg[i][j]
            if i == j:
                out[i][j] = 0
            else:
                out[i][j] = c.n_notasked
    return out
