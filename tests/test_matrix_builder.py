"""
Tests for `MatrixBuilder`: tri-valued matrix M, CoM-based initialization,
propagation, and gap iteration using `path_exists_matrix` + unknown cells (None).
"""

import pytest

from helpers import create_mock_structures

from src.anatomy_poset.core.matrix_builder import MatrixBuilder, initial_tri_valued_relation_matrix
from src.anatomy_poset.core.axis_models import AXIS_VERTICAL, Structure


def test_matrix_builder_sorting_inherited() -> None:
    """Same CoM sort order as MatrixBuilder base ordering."""
    structures = [
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    assert mb.structures[0].name == "Skull"
    assert mb.structures[1].name == "Pelvis"


def test_matrix_initialization_lower_triangle_and_diagonal() -> None:
    """Diagonal -1; strict lower triangle -1; strict upper triangle None."""
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    n = mb.n
    for i in range(n):
        assert mb.M[i][i] == -1
        for j in range(i):
            assert mb.M[i][j] == -1
        for j in range(i + 1, n):
            assert mb.M[i][j] is None


def test_initial_tri_valued_relation_matrix_shape() -> None:
    for n in (0, 1, 4):
        M = initial_tri_valued_relation_matrix(n)
        assert len(M) == n
        for i in range(n):
            assert len(M[i]) == n
            assert M[i][i] == -1
            for j in range(i):
                assert M[i][j] == -1
            for j in range(i + 1, n):
                assert M[i][j] is None


def test_next_pair_only_upper_triangle() -> None:
    """Gap iterator always has i < j (strict upper triangle in CoM index order)."""
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    for _ in range(50):
        p = mb.next_pair()
        if p is None:
            break
        i, j = p
        assert i < j


def test_matrix_equal_com_prefills_both_directions() -> None:
    """Equal axis CoM: neither direction can be strictly above (both -1 off-diagonal)."""
    structures = [
        Structure("A", com_vertical=50.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("B", com_vertical=50.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    assert mb.M[0][1] == -1
    assert mb.M[1][0] == -1


def test_record_response_matrix_invalid_value() -> None:
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    with pytest.raises(ValueError, match="Invalid relation value"):
        mb.record_response_matrix(0, 1, 2)


def test_record_response_matrix_yes_sets_inverse_no() -> None:
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    assert mb.M[0][1] == 1
    assert mb.M[1][0] == -1


def test_propagation_transitive_plus_one() -> None:
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    mb.record_response_matrix(1, 2, 1)
    assert mb.M[0][2] == 1
    assert mb.M[2][0] == -1


def test_edges_sync_with_matrix_pdag() -> None:
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    assert (0, 1) in mb.edges
    assert mb.get_pdag() == mb.edges


def test_vertical_bilateral_mirrors_always_match_after_propagation() -> None:
    """
    Left/Right rows (and columns) must share the same tri-value for each target;
    _sync_vertical_bilateral_mirrors runs after propagation.
    """
    structures = [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=0.0),
        Structure("Left Arm", com_vertical=40.0, com_lateral=80.0, com_anteroposterior=0.0),
        Structure("Right Arm", com_vertical=40.0, com_lateral=20.0, com_anteroposterior=0.0),
        Structure("Foot", com_vertical=10.0, com_lateral=50.0, com_anteroposterior=0.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    # Indices after sort: 0 Skull, 1 Left Arm, 2 Right Arm, 3 Foot
    mb.record_response_matrix(0, 3, 1)  # Skull above Foot (mirrors to both arms vs Foot)
    for j in range(mb.n):
        assert mb.M[1][j] == mb.M[2][j]
    for i in range(mb.n):
        assert mb.M[i][1] == mb.M[i][2]


def test_matrix_record_response_symmetry_vertical() -> None:
    """Mirrors YES from Left core to Right core (vertical bilateral symmetry)."""
    structures = [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=0.0),
        Structure("Left Femur", com_vertical=20.0, com_lateral=80.0, com_anteroposterior=0.0),
        Structure("Right Femur", com_vertical=20.0, com_lateral=20.0, com_anteroposterior=0.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    assert mb.M[0][1] == 1
    assert mb.M[0][2] == 1
    assert (0, 1) in mb.edges and (0, 2) in mb.edges


def test_matrix_next_pair_skips_transitively_implied() -> None:
    """Skips (0,2) when M already has a +1 chain 0→1→2."""
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    mb.record_response_matrix(1, 2, 1)

    mb.current_gap = 2
    mb.current_i = 0
    mb.finished = False

    pair = mb.next_pair()
    assert pair == (1, 3)


def test_path_exists_matrix_matches_edges() -> None:
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    mb.record_response_matrix(1, 2, 1)
    assert mb.path_exists_matrix(0, 2) is True
    assert mb.path_exists_matrix(2, 0) is False


def test_close_transitive_unknowns_fills_cells_when_path_exists() -> None:
    """
    Propagation can refuse a transitive +1 when com(i) <= com(k) even though a +1 path
    exists in M (user answers vs raw CoM). _close_transitive_unknowns must still set
    M[i][j] so next_pair does not skip with M[i][j] stuck at None.
    """
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    mb.record_response_matrix(1, 2, 1)
    # Simulate stale None on the transitive pair (e.g. before _close existed)
    mb.M[0][2] = None
    mb.M[2][0] = None
    mb._close_transitive_unknowns()
    assert mb.M[0][2] == 1
    assert mb.M[2][0] == -1


def test_seal_lower_triangle_restores_com_prior() -> None:
    """Simulate a loaded partial file: lower triangle None; seal fixes before save."""
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.M[2][0] = None  # would be invalid after a naive load
    mb.seal_lower_triangle_com_prior()
    for i in range(mb.n):
        for j in range(i):
            assert mb.M[i][j] == -1


def test_next_pair_respects_query_allowed_indices() -> None:
    """Only pairs with both endpoints in the allowed index set are returned."""
    structures = [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Thorax", com_vertical=70.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Femur", com_vertical=20.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL, query_allowed_indices={0, 3})
    pair = mb.next_pair()
    assert pair is not None
    i, j = pair
    assert i in {0, 3} and j in {0, 3}


# ---------------------------------------------------------------------------
# No-duplicate and bilateral mirroring tests
# ---------------------------------------------------------------------------

def _make_bilateral_builder() -> tuple[MatrixBuilder, dict]:
    """
    Six structures: two bilateral pairs (Left/Right Lung, Left/Right Kidney)
    plus two singletons (Brain, Bladder).

    CoM order (descending): Brain(95) > Left Lung(70) ≈ Right Lung(70)
                            > Left Kidney(55) ≈ Right Kidney(55) > Bladder(20)

    After MatrixBuilder sort the index order is:
      0: Brain, 1: Left Lung, 2: Right Lung, 3: Left Kidney, 4: Right Kidney, 5: Bladder
    (bilateral partners have equal CoM so they sort by insertion order).
    """
    structures = [
        Structure("Brain",        com_vertical=95.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Left Lung",    com_vertical=70.0, com_lateral=70.0, com_anteroposterior=50.0),
        Structure("Right Lung",   com_vertical=70.0, com_lateral=30.0, com_anteroposterior=50.0),
        Structure("Left Kidney",  com_vertical=55.0, com_lateral=65.0, com_anteroposterior=50.0),
        Structure("Right Kidney", com_vertical=55.0, com_lateral=35.0, com_anteroposterior=50.0),
        Structure("Bladder",      com_vertical=20.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    # Build a name→index map for readable assertions
    idx = {s.name: i for i, s in enumerate(mb.structures)}
    return mb, idx


def test_no_pair_asked_twice_exhaustive_yes() -> None:
    """Drive the builder to completion answering every question YES; verify no pair twice."""
    mb, _ = _make_bilateral_builder()
    seen: set[tuple[int, int]] = set()
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        assert pair not in seen, f"Pair {pair} was returned by next_pair() twice!"
        seen.add(pair)
        mb.record_response_matrix(pair[0], pair[1], 1)


def test_no_pair_asked_twice_exhaustive_no() -> None:
    """Drive the builder to completion answering every question NO; verify no pair twice."""
    mb, _ = _make_bilateral_builder()
    seen: set[tuple[int, int]] = set()
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        assert pair not in seen, f"Pair {pair} was returned by next_pair() twice!"
        seen.add(pair)
        mb.record_response_matrix(pair[0], pair[1], -1)


def test_no_pair_asked_twice_exhaustive_not_sure() -> None:
    """Drive the builder to completion answering every question UNSURE; verify no pair twice."""
    mb, _ = _make_bilateral_builder()
    seen: set[tuple[int, int]] = set()
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        assert pair not in seen, f"Pair {pair} was returned by next_pair() twice!"
        seen.add(pair)
        mb.record_response_matrix(pair[0], pair[1], 0)


def test_no_pair_asked_twice_mixed_answers() -> None:
    """Cycle through YES/NO/NOT-SURE; verify no pair twice."""
    mb, _ = _make_bilateral_builder()
    seen: set[tuple[int, int]] = set()
    answers = [1, -1, 0]
    tick = 0
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        assert pair not in seen, f"Pair {pair} was returned by next_pair() twice!"
        seen.add(pair)
        mb.record_response_matrix(pair[0], pair[1], answers[tick % 3])
        tick += 1


def test_bilateral_same_core_never_asked() -> None:
    """Left Lung vs Right Lung (and Left Kidney vs Right Kidney) must never be returned."""
    mb, idx = _make_bilateral_builder()
    forbidden = {
        (min(idx["Left Lung"],    idx["Right Lung"]),   max(idx["Left Lung"],    idx["Right Lung"])),
        (min(idx["Left Kidney"],  idx["Right Kidney"]), max(idx["Left Kidney"],  idx["Right Kidney"])),
    }
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        assert pair not in forbidden, (
            f"Same-core bilateral pair {pair} should never be asked!"
        )
        mb.record_response_matrix(pair[0], pair[1], 1)


def test_double_mirror_fill_on_single_answer() -> None:
    """
    Answering (Brain, Left Lung) fills all four bilateral combinations:
      Brain→Left Lung, Brain→Right Lung (single mirror),
      and — if Brain had a partner — the double mirror too.
    Here Brain has no partner so only two cells are filled directly.
    Left Lung→Right Lung must stay -1 (same-core constraint).
    """
    mb, idx = _make_bilateral_builder()
    brain    = idx["Brain"]
    l_lung   = idx["Left Lung"]
    r_lung   = idx["Right Lung"]

    mb.record_response_matrix(brain, l_lung, 1)

    assert mb.M[brain][l_lung] == 1,  "Brain→Left Lung should be 1"
    assert mb.M[brain][r_lung] == 1,  "Brain→Right Lung should be mirrored to 1"
    assert mb.M[l_lung][brain] == -1, "Left Lung→Brain should be -1 (asymmetry)"
    assert mb.M[r_lung][brain] == -1, "Right Lung→Brain should be -1 (asymmetry)"
    # Same-core constraint must hold
    assert mb.M[l_lung][r_lung] == -1, "Left Lung→Right Lung must stay -1 (same-core)"
    assert mb.M[r_lung][l_lung] == -1, "Right Lung→Left Lung must stay -1 (same-core)"


def test_double_mirror_fill_both_sides_bilateral() -> None:
    """
    Answering (Left Lung, Left Kidney) fills all four bilateral combinations:
      Left Lung→Left Kidney, Left Lung→Right Kidney,
      Right Lung→Left Kidney, Right Lung→Right Kidney.
    """
    mb, idx = _make_bilateral_builder()
    l_lung   = idx["Left Lung"]
    r_lung   = idx["Right Lung"]
    l_kidney = idx["Left Kidney"]
    r_kidney = idx["Right Kidney"]

    # The gap iterator gives us the lowest-indexed representative first;
    # answer it directly to trigger the double mirror.
    i = min(l_lung, l_kidney)
    j = max(l_lung, l_kidney)
    mb.record_response_matrix(i, j, 1)

    assert mb.M[l_lung][l_kidney]  == 1, "Left Lung→Left Kidney should be 1"
    assert mb.M[l_lung][r_kidney]  == 1, "Left Lung→Right Kidney should be mirrored"
    assert mb.M[r_lung][l_kidney]  == 1, "Right Lung→Left Kidney should be mirrored"
    assert mb.M[r_lung][r_kidney]  == 1, "Right Lung→Right Kidney should be mirrored (double)"


def test_double_mirror_no_answer_propagates_correctly() -> None:
    """
    Answering NO for (Left Lung, Left Kidney) should mirror NO to all four
    bilateral combinations (neither left nor right lung above either kidney).
    """
    mb, idx = _make_bilateral_builder()
    l_lung   = idx["Left Lung"]
    r_lung   = idx["Right Lung"]
    l_kidney = idx["Left Kidney"]
    r_kidney = idx["Right Kidney"]

    i = min(l_lung, l_kidney)
    j = max(l_lung, l_kidney)
    mb.record_response_matrix(i, j, -1)

    assert mb.M[l_lung][l_kidney]  == -1
    assert mb.M[l_lung][r_kidney]  == -1
    assert mb.M[r_lung][l_kidney]  == -1
    assert mb.M[r_lung][r_kidney]  == -1


def test_double_mirror_not_sure_propagates_correctly() -> None:
    """
    Answering UNSURE for (Left Lung, Left Kidney) should mirror 0 to all four cells.
    """
    mb, idx = _make_bilateral_builder()
    l_lung   = idx["Left Lung"]
    r_lung   = idx["Right Lung"]
    l_kidney = idx["Left Kidney"]
    r_kidney = idx["Right Kidney"]

    i = min(l_lung, l_kidney)
    j = max(l_lung, l_kidney)
    mb.record_response_matrix(i, j, 0)

    assert mb.M[l_lung][l_kidney]  == 0
    assert mb.M[l_lung][r_kidney]  == 0
    assert mb.M[r_lung][l_kidney]  == 0
    assert mb.M[r_lung][r_kidney]  == 0


def test_mirrored_pairs_never_asked_after_direct_answer() -> None:
    """
    After answering (Left Lung, Left Kidney), the three mirrored combinations
    must never be returned by next_pair().
    """
    mb, idx = _make_bilateral_builder()
    l_lung   = idx["Left Lung"]
    r_lung   = idx["Right Lung"]
    l_kidney = idx["Left Kidney"]
    r_kidney = idx["Right Kidney"]

    # Answer the first representative pair
    i = min(l_lung, l_kidney)
    j = max(l_lung, l_kidney)
    mb.record_response_matrix(i, j, 1)

    # All four combinations that were filled must not appear again
    filled = {
        (min(l_lung,  l_kidney), max(l_lung,  l_kidney)),
        (min(l_lung,  r_kidney), max(l_lung,  r_kidney)),
        (min(r_lung,  l_kidney), max(r_lung,  l_kidney)),
        (min(r_lung,  r_kidney), max(r_lung,  r_kidney)),
    }
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        assert pair not in filled, (
            f"Pair {pair} was already filled by bilateral mirror but asked again!"
        )
        mb.record_response_matrix(pair[0], pair[1], 1)


def test_upper_triangle_fully_filled_after_exhaustion_yes() -> None:
    """After driving to completion with YES, no upper-triangle cell should be None."""
    mb, _ = _make_bilateral_builder()
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        mb.record_response_matrix(pair[0], pair[1], 1)

    n = mb.n
    for i in range(n):
        for j in range(i + 1, n):
            assert mb.M[i][j] is not None, (
                f"Upper-triangle cell ({i},{j}) is still None after exhaustion!"
            )


def test_upper_triangle_fully_filled_after_exhaustion_not_sure() -> None:
    """After driving to completion with UNSURE, no upper-triangle cell should be None."""
    mb, _ = _make_bilateral_builder()
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        mb.record_response_matrix(pair[0], pair[1], 0)

    n = mb.n
    for i in range(n):
        for j in range(i + 1, n):
            assert mb.M[i][j] is not None, (
                f"Upper-triangle cell ({i},{j}) is still None after exhaustion!"
            )
