"""
Predefined body-region groupings for subset poset runs (overlap intentional).

Region 1 — Trunk / thoracoabdominal (roughly skull base → pelvic organs, major vessels,
viscera): heart, lungs, mediastinal GI, abdominal organs, kidneys, iliac vessels as they
course through abdomen/pelvis, bladder & prostate as pelvic viscera. **Excludes** arm bones
and the locomotor pelvis–leg chain (hips, glutes, femora, etc.).

Region 2 — Neuroaxis & upper limb girdle/arm: brain, spinal cord, vertebral column
(including discs), **humeri** (shoulder/arm; no separate scapula/clavicle labels in atlas).

Region 3 — Lumbopelvic junction & lower limb: sacrum, iliopsoas, hips, glutei, thigh
compartments, femora, tibia/fibula, plus shared axial labels (vertebrae, cord, discs) and
deep back (autochthon) where it supports lumbar/pelvic relations.

Structures can appear in multiple regions (e.g. vertebrae, spinal_cord, intervertebral_discs,
autochthon, bladder/prostate, iliac vessels).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Iterable, List, Optional, Set

# --- Region 1: trunk / viscera / thorax (no humeri; no dedicated lower-limb bones) ---
REGION_1_TRUNK_VISCERA_NO_ARMS: FrozenSet[str] = frozenset(
    {
        "brain",
        "lung_left",
        "lung_right",
        "esophagus",
        "spinal_cord",
        "heart",
        "aorta",
        "spleen",
        "vertebrae",
        "stomach",
        "liver",
        "adrenal_gland_right",
        "adrenal_gland_left",
        "intervertebral_discs",
        "pancreas",
        "portal_vein_and_splenic_vein",
        "gallbladder",
        "kidney_left",
        "kidney_right",
        "inferior_vena_cava",
        "duodenum",
        "colon",
        "small_bowel",
        "iliac_artery_right",
        "iliac_artery_left",
        "iliac_vena_right",
        "iliac_vena_left",
        "urinary_bladder",
        "prostate",
        # Deep back muscles — trunk column (overlap with 2 & 3)
        "autochthon_left",
        "autochthon_right",
    }
)

# --- Region 2: brain, spine, arms (humeri) ---
REGION_2_NEUROAXIS_SHOULDER_ARMS: FrozenSet[str] = frozenset(
    {
        "brain",
        "spinal_cord",
        "vertebrae",
        "intervertebral_discs",
        "humerus_left",
        "humerus_right",
        "autochthon_left",
        "autochthon_right",
    }
)

# --- Region 3: lumbar/pelvis + lower limb ---
REGION_3_LUMBPELVIS_LEGS: FrozenSet[str] = frozenset(
    {
        "vertebrae",
        "intervertebral_discs",
        "spinal_cord",
        "autochthon_left",
        "autochthon_right",
        "iliopsoas_left",
        "iliopsoas_right",
        "sacrum",
        "iliac_artery_right",
        "iliac_artery_left",
        "iliac_vena_right",
        "iliac_vena_left",
        "hip_right",
        "hip_left",
        "gluteus_medius_right",
        "gluteus_medius_left",
        "gluteus_minimus_right",
        "gluteus_minimus_left",
        "gluteus_maximus_right",
        "gluteus_maximus_left",
        "urinary_bladder",
        "prostate",
        "thigh_medial_compartment_left",
        "thigh_medial_compartment_right",
        "sartorius_right",
        "sartorius_left",
        "quadriceps_femoris_left",
        "quadriceps_femoris_right",
        "thigh_posterior_compartment_right",
        "thigh_posterior_compartment_left",
        "femur_right",
        "femur_left",
        "tibia",
        "fibula",
    }
)

REGION_IDS: tuple[str, str, str] = (
    "1_trunk_viscera_no_arms",
    "2_neuroaxis_shoulder_arms",
    "3_lumbopelvis_legs",
)

REGION_LABELS: Dict[str, str] = {
    "1_trunk_viscera_no_arms": "1 — Trunk and viscera (thorax/abdomen/pelvic organs, vessels; no arms)",
    "2_neuroaxis_shoulder_arms": "2 — Brain, spine and arms",
    "3_lumbopelvis_legs": "3 — Sacrum, pelvis, hips and legs",
}

REGION_MEMBERS: Dict[str, FrozenSet[str]] = {
    "1_trunk_viscera_no_arms": REGION_1_TRUNK_VISCERA_NO_ARMS,
    "2_neuroaxis_shoulder_arms": REGION_2_NEUROAXIS_SHOULDER_ARMS,
    "3_lumbopelvis_legs": REGION_3_LUMBPELVIS_LEGS,
}


def union_region_names(selected_ids: Iterable[str]) -> Set[str]:
    """Union of structure names for the given region id keys."""
    out: Set[str] = set()
    for rid in selected_ids:
        out |= REGION_MEMBERS[rid]
    return out


def query_allowed_indices_for_regions(
    structures: List,
    *,
    use_all: bool,
    selected_region_ids: Set[str],
) -> Optional[Set[int]]:
    """
    Map region selection to indices into ``structures`` for :class:`MatrixBuilder`.

    Returns ``None`` when all structures may be queried; otherwise a set of indices
    whose names appear in the union of selected regions (full matrix size unchanged).
    """
    if use_all:
        return None
    allowed_names = union_region_names(selected_region_ids)
    return {i for i, s in enumerate(structures) if s.name in allowed_names}
