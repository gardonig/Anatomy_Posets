"""
Integration-style tests: simulated experts answer MatrixBuilder queries the same
way the GUI would (via `record_response_matrix` / `record_unknown`), without Qt.

1. **Chain expert** — always YES along the CoM sort order (deterministic checks).
2. **Stochastic expert** — YES / NO / unsure with configurable probabilities
   (default 70% / 20% / 10%) and optional simulated **feedback** on a fraction
   of questions (default 5%), mirroring the QueryDialog feedback box.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Tuple

import pytest

from src.anatomy_poset.core.matrix_builder import MatrixBuilder
from src.anatomy_poset.core.axis_models import (
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


@dataclass
class StochasticExpertResult:
    """Outcome of `run_stochastic_expert` (mirrors yes/no/unsure + optional feedback)."""

    yes_count: int = 0
    no_count: int = 0
    unsure_count: int = 0
    question_count: int = 0
    # Same idea as QueryDialog feedback_box: occasional free-text notes per (i, j).
    feedback_log: List[Tuple[int, int, str]] = field(default_factory=list)


def run_stochastic_expert(
    mb: MatrixBuilder,
    rng: random.Random,
    *,
    p_yes: float = 0.7,
    p_no: float = 0.2,
    p_unsure: float = 0.1,
    feedback_prob: float = 0.05,
) -> StochasticExpertResult:
    """
    Answer each `next_pair` randomly:
      - `p_yes`     → YES (+1)
      - `p_no`      → NO (-1)
      - `p_unsure`  → not sure (0), via `record_unknown`

    Independently with probability `feedback_prob`, append a simulated feedback
    string (as if the expert typed in the feedback box).

    Default probabilities: 70% / 20% / 10%.
    """
    total = p_yes + p_no + p_unsure
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Probabilities must sum to 1.0, got {total}")

    cut_yes = p_yes
    cut_no = p_yes + p_no

    result = StochasticExpertResult()
    _feedback_snippets = (
        "Borderline case — double-check on next specimen.",
        "Naming convention unclear for this pair.",
        "Possible motion artifact in CoM.",
        "Consider revisiting after segmentation fix.",
    )

    max_steps = mb.n * mb.n * 4 + 10
    steps = 0
    while True:
        pair = mb.next_pair()
        if pair is None:
            break
        i, j = pair
        assert i < j
        r = rng.random()
        if r < cut_yes:
            mb.record_response_matrix(i, j, 1)
            result.yes_count += 1
        elif r < cut_no:
            mb.record_response_matrix(i, j, -1)
            result.no_count += 1
        else:
            mb.record_unknown(i, j)
            result.unsure_count += 1

        result.question_count += 1

        if rng.random() < feedback_prob:
            msg = rng.choice(_feedback_snippets)
            result.feedback_log.append((i, j, msg))

        steps += 1
        assert steps <= max_steps, "possible infinite loop in stochastic expert simulation"

    return result


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


def test_stochastic_expert_completes_session() -> None:
    """Random expert (70/20/10) finishes without crashing; counts stay consistent."""
    structures = _axis_chain_structures()
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    rng = random.Random(20250316)

    out = run_stochastic_expert(mb, rng, feedback_prob=0.05)

    assert mb.finished
    assert out.question_count > 0
    assert out.yes_count + out.no_count + out.unsure_count == out.question_count
    # Feedback is independent Bernoulli(p); with few questions, 0 entries is common.
    assert len(out.feedback_log) <= out.question_count
    for i, j, text in out.feedback_log:
        assert i < j
        assert text.strip()


def test_stochastic_expert_feedback_every_question_when_prob_one() -> None:
    """When feedback_prob=1.0, every asked question gets a feedback line (like the GUI box)."""
    structures = _axis_chain_structures()
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    rng = random.Random(1)

    out = run_stochastic_expert(mb, rng, feedback_prob=1.0)

    assert out.question_count > 0
    assert len(out.feedback_log) == out.question_count


def test_seventy_twenty_ten_split_on_raw_rng() -> None:
    """The answer sampler matches 70% / 20% / 10% over many draws (no MatrixBuilder)."""
    p_yes, p_no, p_unsure = 0.7, 0.2, 0.1
    cut_yes = p_yes
    cut_no = p_yes + p_no
    rng = random.Random(12345)
    n = 20_000
    yes = no = unsure = 0
    for _ in range(n):
        r = rng.random()
        if r < cut_yes:
            yes += 1
        elif r < cut_no:
            no += 1
        else:
            unsure += 1
    assert abs(yes / n - 0.7) < 0.02
    assert abs(no / n - 0.2) < 0.02
    assert abs(unsure / n - 0.1) < 0.02
