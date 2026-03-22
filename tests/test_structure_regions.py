"""Body-region presets for subset runs."""

from anatomy_poset.core.models import Structure
from anatomy_poset.core.structure_regions import (
    REGION_1_TRUNK_VISCERA_NO_ARMS,
    REGION_2_NEUROAXIS_SHOULDER_ARMS,
    REGION_3_LUMBPELVIS_LEGS,
    query_allowed_indices_for_regions,
    union_region_names,
)


def test_union_covers_same_56_as_merged_atlas() -> None:
    U = REGION_1_TRUNK_VISCERA_NO_ARMS | REGION_2_NEUROAXIS_SHOULDER_ARMS | REGION_3_LUMBPELVIS_LEGS
    assert len(U) == 56


def test_region2_includes_humeri_not_femora() -> None:
    assert "humerus_left" in REGION_2_NEUROAXIS_SHOULDER_ARMS
    assert "femur_left" not in REGION_2_NEUROAXIS_SHOULDER_ARMS
    assert "femur_left" in REGION_3_LUMBPELVIS_LEGS


def test_query_allowed_indices_subset() -> None:
    s = [
        Structure("brain", 90, 50, 50),
        Structure("femur_left", 20, 80, 70),
    ]
    idx = query_allowed_indices_for_regions(
        s,
        use_all=False,
        selected_region_ids={"2_neuroaxis_shoulder_arms"},
    )
    assert idx == {0}


def test_union_multi_select() -> None:
    u = union_region_names(["1_trunk_viscera_no_arms", "3_lumbopelvis_legs"])
    assert "heart" in u and "femur_left" in u and "humerus_left" not in u
