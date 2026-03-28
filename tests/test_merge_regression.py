"""
Regression: merged probability matrices from test5 + test6 match saved merged_consensus_5_6.json
(if present — same pipeline as the poset viewer merge, using aggregate_to_p_yes_matrix).

Golden file must use the same schema (float / null cells). Legacy int-only merged files will fail.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from anatomy_poset.core.matrix_aggregation import (
    aggregate_matrices_with_counts,
    aggregate_to_p_yes_matrix,
    align_matrix_lists_to_reference,
    apply_canonical_per_axis_orders,
    reindex_matrix_to_structure_order,
)
from anatomy_poset.core.io import load_poset_from_json

_REPO = Path(__file__).resolve().parents[1]
_DATA = _REPO / "data" / "Output_constructed_posets"
_P5 = _DATA / "test5.json"
_P6 = _DATA / "test6.json"
_MERGED = _DATA / "merged_consensus_5_6.json"


@pytest.mark.skipif(not _MERGED.is_file(), reason="merged_consensus_5_6.json not present")
def test_merge_test5_test6_matches_saved_merged_consensus() -> None:
    p5 = load_poset_from_json(str(_P5))
    p6 = load_poset_from_json(str(_P6))
    st5, mv5, ml5, ap5 = p5.structures, p5.matrix_vertical, p5.matrix_mediolateral, p5.matrix_anteroposterior
    st6, mv6, ml6, ap6 = p6.structures, p6.matrix_vertical, p6.matrix_mediolateral, p6.matrix_anteroposterior

    ok, msg, mv_a, ml_a, ap_a, _, _, _ = align_matrix_lists_to_reference(
        [st5, st6], [mv5, mv6], [ml5, ml6], [ap5, ap6]
    )
    assert ok, msg

    sv, sml, sap, mv_ord, ml_ord, ap_ord, _, _, _ = apply_canonical_per_axis_orders(
        st5, mv_a, ml_a, ap_a
    )

    def merged_p_yes(mats: list) -> list:
        agg, _k = aggregate_matrices_with_counts(mats)
        return aggregate_to_p_yes_matrix(agg)

    mv = merged_p_yes(mv_ord)
    mml = merged_p_yes(ml_ord)
    map_ = merged_p_yes(ap_ord)
    mml_save = reindex_matrix_to_structure_order(sv, sml, mml)
    map_save = reindex_matrix_to_structure_order(sv, sap, map_)

    with open(_MERGED, encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["matrix_vertical"] == mv
    assert saved["matrix_mediolateral"] == mml_save
    assert saved["matrix_anteroposterior"] == map_save


def test_manual_vertical_cell_counts_match_aggregate() -> None:
    """Spot-check: per-cell mean and P(yes) for a few (i,j) pairs (K=2)."""
    if not _P5.is_file() or not _P6.is_file():
        pytest.skip("test5/test6 JSON not present")

    p5 = load_poset_from_json(str(_P5))
    p6 = load_poset_from_json(str(_P6))
    st5, mv5, ml5, ap5 = p5.structures, p5.matrix_vertical, p5.matrix_mediolateral, p5.matrix_anteroposterior
    st6, mv6, ml6, ap6 = p6.structures, p6.matrix_vertical, p6.matrix_mediolateral, p6.matrix_anteroposterior
    ok, msg, mv_a, ml_a, ap_a, _, _, _ = align_matrix_lists_to_reference(
        [st5, st6], [mv5, mv6], [ml5, ml6], [ap5, ap6]
    )
    assert ok, msg
    _, _, _, mv_ord, _, _, _, _, _ = apply_canonical_per_axis_orders(st5, mv_a, ml_a, ap_a)
    agg, k = aggregate_matrices_with_counts(mv_ord)
    assert k == 2
    p = aggregate_to_p_yes_matrix(agg)

    # Lower triangle sealed: j < i => -1 for both raters
    assert mv_ord[0][3][1] == -1 and mv_ord[1][3][1] == -1
    assert agg[3][1].mean == -1.0
    assert p[3][1] == 0.0  # (μ+1)/2 with μ=-1

    # [0][2]: -1 vs +1 => mean 0 => P(yes)=0.5
    assert mv_ord[0][0][2] == -1 and mv_ord[1][0][2] == 1
    assert abs(agg[0][2].mean) < 1e-9
    assert abs(p[0][2] - 0.5) < 1e-9
