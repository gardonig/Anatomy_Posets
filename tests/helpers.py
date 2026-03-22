"""Shared helpers for anatomy_poset tests."""

from __future__ import annotations

from src.anatomy_poset.core.models import Structure


def create_mock_structures() -> list[Structure]:
    """Four structures with distinct vertical CoMs (descending when sorted)."""
    return [
        Structure("Skull", com_vertical=90.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Thorax", com_vertical=70.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Pelvis", com_vertical=40.0, com_lateral=50.0, com_anteroposterior=50.0),
        Structure("Femur", com_vertical=20.0, com_lateral=50.0, com_anteroposterior=50.0),
    ]
