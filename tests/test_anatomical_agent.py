"""
Integration-style tests: a simulated "expert" with fixed anatomical knowledge
answers MatrixBuilder queries the same way the GUI would (via
`record_response_matrix`), without Qt.

The expert assumes a *strict chain* on each axis: after sorting by that axis
CoM (descending), structure 0 is strictly above 1 above … above n−1. That
matches the usual interpretation of CoM as a coarse proxy for position along
the axis (head above foot on vertical, etc.).
"""

import pytest

from src.anatomy_poset.core.builder import MatrixBuilder
from src.anatomy_poset.core.models import (
    AXIS_ANTERIOR_POSTERIOR,
    AXIS_MEDIOLATERAL,
    AXIS_VERTICAL,
    Structure,
)


def _axis_chain_structures() -> list[Structure]:
    """
    Four structures with distinct CoM on each axis so sort order differs
    per axis (tests that the agent does not depend on the vertical order).
    """
    return [
        Structure("A", com_vertical=90.0, com_lateral=30.0, com_anteroposterior=10.0),
        Structure("B", com_vertical=70.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("C", com_vertical=50.0, com_lateral=70.0, com_anteroposterior=90.0),
        Structure("D", com_vertical=20.0, com_lateral=90.0, com_anteroposterior=30.0),
    ]


def run_chain_expert(mb: MatrixBuilder) -> None:
    """
    Answer every query as YES: for each pair (i, j) with i < j returned by
    `next_pair`, record that i is strictly above j (+1).

    This is the "anatomical" prior: indices follow descending CoM on the
    active axis, so the expert asserts a total order consistent with that CoM
    ordering.
    """
    max_steps = mb.n * mb.n * 4 + 10
    steps = 0
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        i, j = pair
        assert i < j, "gap iterator should only yield i < j"
        mb.record_response_matrix(i, j, 1)
        steps += 1
        assert steps <= max_steps, "possible infinite loop in expert simulation"


def assert_strict_upper_triangle_no_unknown(mb: MatrixBuilder) -> None:
    """Strict upper triangle (i < j) should not remain -2 after a full expert run."""
    for i in range(mb.n):
        for j in range(i + 1, mb.n):
            assert mb.M[i][j] != -2, f"Unresolved ({i},{j}) for axis={mb.axis!r}"


@pytest.mark.parametrize(
    "axis",
    [AXIS_VERTICAL, AXIS_MEDIOLATERAL, AXIS_ANTERIOR_POSTERIOR],
)
def test_anatomical_chain_expert_completes_matrix(axis: str) -> None:
    structures = _axis_chain_structures()
    mb = MatrixBuilder(structures, axis=axis)

    run_chain_expert(mb)

    assert mb.next_pair() is None
    assert mb.finished
    assert_strict_upper_triangle_no_unknown(mb)

    # Chain should yield a directed path along sorted indices: 0→1→…→n−1
    for k in range(mb.n - 1):
        assert mb.M[k][k + 1] == 1 and mb.M[k + 1][k] == -1


def test_anatomical_expert_vertical_symmetry_bilateral() -> None:
    """
    Expert answers a vertical chain; Left/Right pairs are forced to -1 by the
    builder and should not break completion.
    """
    structures = [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Left Femur", com_vertical=20.0, com_lateral=80.0, com_anteroposterior=50.0),
        Structure("Right Femur", com_vertical=20.0, com_lateral=20.0, com_anteroposterior=50.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    run_chain_expert(mb)
    assert mb.finished
    assert_strict_upper_triangle_no_unknown(mb)
