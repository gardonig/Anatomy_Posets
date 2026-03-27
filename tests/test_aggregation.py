"""Tests for multi-rater matrix aggregation and structure compatibility."""

from anatomy_poset.core.matrix_aggregation import (
    CellAggregate,
    aggregate_matrices_mean_only,
    aggregate_matrices_with_counts,
    aggregate_to_consensus_matrix,
    aggregate_to_p_yes_matrix,
    align_matrix_lists_to_reference,
    apply_canonical_per_axis_orders,
    cell_aggregate_to_display_matrix,
    enforce_vertical_lower_triangle_inplace,
    find_alignment_permutation,
    reindex_matrix_to_structure_order,
    structure_list_signature,
    validate_structures_compatible,
)
from anatomy_poset.core.axis_models import Structure


def _struct(name: str, v: float = 0.0, ml: float = 0.0, ap: float = 0.0) -> Structure:
    return Structure(name=name, com_vertical=v, com_lateral=ml, com_anteroposterior=ap)


def test_structure_signature_match() -> None:
    a = [_struct("A", 1, 2, 3), _struct("B", 4, 5, 6)]
    b = [_struct("A", 1, 2, 3), _struct("B", 4, 5, 6)]
    assert structure_list_signature(a) == structure_list_signature(b)
    ok, msg = validate_structures_compatible([a, b])
    assert ok and msg == ""


def test_structure_signature_mismatch_name() -> None:
    a = [_struct("A")]
    b = [_struct("X")]
    ok, msg = validate_structures_compatible([a, b])
    assert not ok
    assert "name" in msg.lower() or "match" in msg.lower()


def test_validate_tolerant_float_json_drift() -> None:
    """Same structures; CoMs differ only by tiny float noise (JSON)."""
    a = [_struct("A", 50.123456789, 10.0, 20.0)]
    b = [_struct("A", 50.1234567890001, 10.0, 20.0)]
    ok, msg = validate_structures_compatible([a, b])
    assert ok, msg


def test_align_reordered_structures() -> None:
    """Same structures different row order — matrices permute to reference."""
    ref = [_struct("A", 1, 0, 0), _struct("B", 2, 0, 0)]
    other = [_struct("B", 2, 0, 0), _struct("A", 1, 0, 0)]
    # M[i][j]: identity-ish 2x2 off-diagonal
    m_ref = [[-1, 0], [1, -1]]
    m_other = [[-1, 1], [0, -1]]  # rows/cols in B,A order
    perm, err = find_alignment_permutation(ref, other)
    assert perm == [1, 0], err
    ok, msg, v, ml, ap = align_matrix_lists_to_reference(
        [ref, other], [m_ref, m_other], [m_ref, m_other], [m_ref, m_other]
    )
    assert ok, msg
    assert v[1] == m_ref  # second file aligned to first order A,B


def test_canonical_vertical_no_plus_one_below_diagonal() -> None:
    """Mis-ordered structures can put +1 below diagonal; per-axis canonical + seal fixes."""
    ref = [_struct("A", 1, 0, 0), _struct("B", 2, 0, 0)]  # not vertically sorted
    m = [[-1, 0], [1, -1]]  # +1 at [1][0] (below diagonal) — invalid until reordered
    sv, _, _, mv, _, _ = apply_canonical_per_axis_orders(ref, [m], [m], [m])
    assert sv[0].name == "B" and sv[1].name == "A"
    assert mv[0][1][0] == -1  # sealed lower triangle on vertical matrix


def test_enforce_vertical_seals_even_when_already_sorted() -> None:
    """Stale JSON can have +1 below diagonal even when structure list is CoM-sorted."""
    st = [_struct("B", 2, 0, 0), _struct("A", 1, 0, 0)]
    m = [[-1, 1], [1, -1]]  # [1][0] invalid +1 below diagonal
    _, _, _, mv, _, _ = apply_canonical_per_axis_orders(st, [m], [m], [m])
    assert mv[0][1][0] == -1


def test_reindex_matrix_to_common_structure_order() -> None:
    """Export path: ML/AP matrices in axis order → vertical order indexing."""
    a = _struct("A", 1, 0, 0)
    b = _struct("B", 2, 0, 0)
    vertical_order = [b, a]  # B higher vertical first
    lateral_order = [a, b]  # different permutation
    M = [[-1, 1], [0, -1]]  # indexed by lateral_order rows/cols
    out = reindex_matrix_to_structure_order(vertical_order, lateral_order, M)
    assert out[0][1] == M[1][0]  # vertical row0=B, col1=a ↔ lateral row1, col0


def test_per_axis_canonical_orders_differ() -> None:
    """Vertical vs lateral CoM sort can reorder structures differently."""
    s = [
        _struct("A", 100.0, 0.0, 0.0),
        _struct("B", 50.0, 50.0, 0.0),
    ]
    identity = [[-1, -2], [-2, -1]]
    sv, sml, _, _, _, _ = apply_canonical_per_axis_orders(s, [identity], [identity], [identity])
    assert sv[0].name == "A"  # higher vertical CoM
    assert sml[0].name == "B"  # higher lateral CoM


def test_enforce_vertical_inplace() -> None:
    m = [[-1, 0], [1, -1]]
    enforce_vertical_lower_triangle_inplace(m)
    assert m[1][0] == -1


def test_cell_aggregate_display_diagonal_p_zero() -> None:
    m1 = [[-1, 1], [-1, -1]]
    m2 = [[-1, 1], [-1, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2])
    Z, _, _ = cell_aggregate_to_display_matrix(agg)
    import math

    assert not math.isnan(Z[0][0])
    assert Z[0][0] == 0.0


def test_aggregate_counts_and_mean() -> None:
    # 2x2 off-diagonal: rater1 says +1, rater2 says -1
    m1 = [[-1, 1], [-1, -1]]
    m2 = [[-1, -1], [1, -1]]
    agg, k = aggregate_matrices_with_counts([m1, m2])
    assert k == 2
    c = agg[0][1]
    assert c.n_answered == 2
    assert c.mean == 0.0
    assert c.counts[1] == 1 and c.counts[-1] == 1


def test_aggregate_notasked_minus2() -> None:
    m1 = [[-1, -2], [-2, -1]]
    m2 = [[-1, 1], [-1, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2])
    c = agg[0][1]
    assert c.n_notasked == 1
    assert c.n_answered == 1
    assert c.mean == 1.0


def test_aggregate_mean_only_matches_float_mean() -> None:
    m1 = [[-1, 1], [-1, -1]]
    m2 = [[-1, 0], [0, -1]]
    w = aggregate_matrices_mean_only([m1, m2])
    assert w[0][1] == 0.5  # (1+0)/2


def test_probability_green() -> None:
    c = CellAggregate(mean=1.0, n_answered=1, n_notasked=0, counts={1: 1})
    assert c.probability_yes_green == 1.0
    c2 = CellAggregate(mean=-1.0, n_answered=1, n_notasked=0, counts={-1: 1})
    assert c2.probability_yes_green == 0.0


def test_cell_aggregate_display_nan_no_data() -> None:
    m1 = [[-1, -2], [-2, -1]]
    m2 = [[-1, -2], [-2, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2])
    Z, _, _ = cell_aggregate_to_display_matrix(agg)
    import math

    assert math.isnan(Z[0][1])


def test_cell_aggregate_display_majority_tie_not_mean_half() -> None:
    """+1 vs −1 with K=2 is a vote tie — not orange "half yes" from mean-based P."""
    import math

    m1 = [[-1, 1], [-1, -1]]
    m2 = [[-1, -1], [1, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2])
    Z, ann, tm = cell_aggregate_to_display_matrix(agg)
    assert math.isnan(Z[0][1])
    assert tm[0][1]
    assert "tie" in ann[0][1]

    Zm, _, _ = cell_aggregate_to_display_matrix(agg, color_mode="mean")
    assert Zm[0][1] == 0.5  # legacy mean map


def test_consensus_majority() -> None:
    m1 = [[-1, 1], [-1, -1]]
    m2 = [[-1, 1], [1, -1]]
    m3 = [[-1, -1], [-1, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2, m3])
    cons = aggregate_to_consensus_matrix(agg)
    assert cons[0][1] == 1  # two +1, one -1


def test_p_yes_off_diagonal_mean() -> None:
    """P(yes) from mean of answered codes; −2 excluded (one −2, one 0 → μ=0 → P=0.5)."""
    m1 = [[-1, -2], [-2, -1]]
    m2 = [[-1, 0], [0, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2])
    p = aggregate_to_p_yes_matrix(agg)
    assert p[0][1] == 0.5
    assert p[1][0] == 0.5


def test_p_yes_none_when_no_answers() -> None:
    m1 = [[-1, -2], [-2, -1]]
    m2 = [[-1, -2], [-2, -1]]
    agg, _ = aggregate_matrices_with_counts([m1, m2])
    p = aggregate_to_p_yes_matrix(agg)
    assert p[0][1] is None
    assert p[0][0] == 0.0  # diagonal
