from src.anatomy_poset.core.axis_models import (
    AXIS_VERTICAL,
    Structure,
)
from src.anatomy_poset.core.matrix_builder import MatrixBuilder


def create_mock_structures():
    return [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Thorax", com_vertical=70.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Femur", com_vertical=20.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]


def test_matrix_builder_sorting():
    """Structures are sorted descending by the chosen axis CoM."""
    structures = [
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=0.0),
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=0.0),
    ]
    builder = MatrixBuilder(structures, axis=AXIS_VERTICAL)

    assert builder.structures[0].name == "Skull"
    assert builder.structures[1].name == "Pelvis"


def test_path_exists_transitivity():
    """Graph reachability on ``edges`` (kept in sync with +1 in M)."""
    builder = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    builder.record_response_matrix(0, 1, 1)
    builder.record_response_matrix(1, 2, 1)

    assert builder.path_exists(0, 2) is True
    assert builder.path_exists(2, 0) is False


def test_edge_redundancy_reduction():
    """Transitive reduction on the current edge set."""
    builder = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)
    builder.edges.update([(0, 1), (1, 2), (0, 2)])

    reduced_edges = builder.edge_redundancy_reduction()

    assert (0, 1) in reduced_edges
    assert (1, 2) in reduced_edges
    assert (0, 2) not in reduced_edges


def test_symmetry_vertical_axis():
    """YES for Left core mirrors to Right core (+1 in M -> edges)."""
    structures = [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=0.0),
        Structure("Left Femur", com_vertical=20.0, com_lateral=80.0, com_anteroposterior=0.0),
        Structure("Right Femur", com_vertical=20.0, com_lateral=20.0, com_anteroposterior=0.0),
    ]
    builder = MatrixBuilder(structures, axis=AXIS_VERTICAL)

    builder.record_response_matrix(0, 1, 1)

    assert (0, 2) in builder.edges


def test_gap_iteration_skips_implied_relations():
    """next_pair skips pairs already implied by +1 propagation."""
    builder = MatrixBuilder(create_mock_structures(), axis=AXIS_VERTICAL)

    builder.record_response_matrix(0, 1, 1)
    builder.record_response_matrix(1, 2, 1)

    builder.current_gap = 2
    builder.current_i = 0

    pair = builder.next_pair()
    assert pair == (1, 3)
