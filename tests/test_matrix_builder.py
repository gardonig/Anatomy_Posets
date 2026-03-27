"""
Tests for `MatrixBuilder`: tri-valued matrix M, CoM-based initialization,
propagation, and gap iteration using `path_exists_matrix` + unknown cells (-2).
"""

import pytest

from helpers import create_mock_structures

from src.anatomy_poset.core.matrix_builder import MatrixBuilder
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
    """Diagonal -1; strict lower triangle -1; strict upper triangle -2."""
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    n = mb.n
    for i in range(n):
        assert mb.M[i][i] == -1
        for j in range(i):
            assert mb.M[i][j] == -1
        for j in range(i + 1, n):
            assert mb.M[i][j] == -2


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
    M[i][j] so next_pair does not skip with M[i][j] stuck at -2.
    """
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.record_response_matrix(0, 1, 1)
    mb.record_response_matrix(1, 2, 1)
    # Simulate stale -2 on the transitive pair (e.g. before _close existed)
    mb.M[0][2] = -2
    mb.M[2][0] = -2
    mb._close_transitive_unknowns()
    assert mb.M[0][2] == 1
    assert mb.M[2][0] == -1


def test_seal_lower_triangle_restores_com_prior() -> None:
    """Simulate a loaded partial file: lower triangle -2; seal fixes before save."""
    mb = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    mb.M[2][0] = -2  # would be invalid after a naive load
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
