import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple

from .models import Structure

def load_structures_from_json(path: str) -> List[Structure]:
    """
    Load a list of structures from a JSON file.
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("structures", [])
    structures: List[Structure] = []
    for item in items:
        try:
            name = str(item["name"])
            com_vertical = float(item["com_vertical"])
            com_lateral = float(item.get("com_lateral", 0.0))
            com_ap = float(item.get("com_anteroposterior", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(
            Structure(
                name=name,
                com_vertical=com_vertical,
                com_lateral=com_lateral,
                com_anteroposterior=com_ap,
            )
        )
    return structures

def save_poset_to_json(
    path: str,
    structures: List[Structure],
    matrix_vertical: List[List[int]],
    matrix_mediolateral: List[List[int]] | None = None,
    matrix_anteroposterior: List[List[int]] | None = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save tri-valued relation matrices to JSON. All axes stored in one file.

    Each matrix entry is in {-2, -1, 0, +1} with the following meaning:
      -2: not asked
      -1: explicit "no" / not-above
       0: asked but "not sure"
      +1: "yes" / above

    ``extra`` is merged into the top-level JSON object (e.g. merge metadata).
    """
    if matrix_mediolateral is None:
        matrix_mediolateral = []
    if matrix_anteroposterior is None:
        matrix_anteroposterior = []

    payload: Dict[str, Any] = {
        "structures": [asdict(s) for s in structures],
        "matrix_vertical": matrix_vertical,
        "matrix_mediolateral": matrix_mediolateral,
        "matrix_anteroposterior": matrix_anteroposterior,
    }
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_poset_from_json(
    path: str,
) -> Tuple[
    List[Structure],
    List[List[int]],
    List[List[int]],
    List[List[int]],
]:
    """
    Load poset(s) from JSON.
    Returns:
      (structures, M_vertical, M_mediolateral, M_anteroposterior)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    structures_data = data.get("structures", [])
    # New schema: tri-valued matrices per axis
    M_v = data.get("matrix_vertical")
    M_ml = data.get("matrix_mediolateral")
    M_ap = data.get("matrix_anteroposterior")

    structures: List[Structure] = []
    for item in structures_data:
        try:
            name = str(item["name"])
            com_vertical = float(item["com_vertical"])
            com_lateral = float(item.get("com_lateral", 0.0))
            com_ap = float(item.get("com_anteroposterior", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        structures.append(
            Structure(
                name=name,
                com_vertical=com_vertical,
                com_lateral=com_lateral,
                com_anteroposterior=com_ap,
            )
        )

    n = len(structures_data)

    def _fallback_matrix_from_edges(key_edges: str, key_adj: str) -> List[List[int]]:
        """
        Backward compat: build a tri-valued matrix from older edge/adjaency-only files.
        """
        # Prefer explicit adjacency if present
        adj = data.get(key_adj)
        if isinstance(adj, list) and all(isinstance(row, list) for row in adj):
            # Normalize entries: >0 -> +1, else 0 (unknown)
            mat = [[0 for _ in range(n)] for _ in range(n)]
            for i in range(min(n, len(adj))):
                row = adj[i]
                for j in range(min(n, len(row))):
                    mat[i][j] = 1 if int(row[j]) != 0 else 0
            return mat

        edges_data = data.get(key_edges, [])
        mat = [[0 for _ in range(n)] for _ in range(n)]
        for item in edges_data:
            try:
                u, v = int(item[0]), int(item[1])
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= u < n and 0 <= v < n:
                mat[u][v] = 1
        return mat

    # If new matrices are present, use them; otherwise derive from old fields.
    if M_v is None:
        M_v = _fallback_matrix_from_edges("edges_vertical", "adjacency_vertical")
    if M_ml is None:
        # Backward compat: older files used edges_frontal for left-right axis
        M_ml = _fallback_matrix_from_edges("edges_mediolateral", "adjacency_mediolateral")
    if M_ap is None:
        M_ap = _fallback_matrix_from_edges("edges_anteroposterior", "adjacency_anteroposterior")

    # Ensure matrices are n x n with ints
    def _normalize_matrix(M: list) -> List[List[int]]:
        mat = [[-2 for _ in range(n)] for _ in range(n)]
        # By convention, diagonal is always explicit NO (-1): nothing is strictly above itself.
        for i in range(n):
            mat[i][i] = -1
        if not isinstance(M, list):
            return mat
        for i in range(min(n, len(M))):
            row = M[i]
            if not isinstance(row, list):
                continue
            for j in range(min(n, len(row))):
                try:
                    mat[i][j] = int(row[j])
                except (TypeError, ValueError):
                    mat[i][j] = -2
        # Re-enforce diagonal convention even if source data had other values/missing cells.
        for i in range(n):
            mat[i][i] = -1
        return mat

    M_v_norm = _normalize_matrix(M_v)
    M_ml_norm = _normalize_matrix(M_ml)
    M_ap_norm = _normalize_matrix(M_ap)

    return (
        structures,
        M_v_norm,
        M_ml_norm,
        M_ap_norm,
    )