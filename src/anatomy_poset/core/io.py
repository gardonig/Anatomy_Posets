import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from .axis_models import Structure

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
    matrix_vertical: List[List[Union[int, float, None]]],
    matrix_mediolateral: Optional[List[List[Union[int, float, None]]]] = None,
    matrix_anteroposterior: Optional[List[List[Union[int, float, None]]]] = None,
    *,
    matrix_vertical_p_yes: Optional[List[List[Optional[float]]]] = None,
    matrix_mediolateral_p_yes: Optional[List[List[Optional[float]]]] = None,
    matrix_anteroposterior_p_yes: Optional[List[List[Optional[float]]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save relation matrices to JSON. All axes stored in one file.

    Matrix values may be either:
    - tri-valued entries in {-2, -1, 0, +1}, or
    - probability entries in [0, 1] with ``null`` for unanswered cells.

    Optional ``matrix_*_p_yes``: merged **probability consensus** (``P(yes) ∈ [0, 1]`` or
    ``null`` where no rater answered that cell), same convention as ``(μ+1)/2`` over answered
    codes only. Omitted when not provided.

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
    if matrix_vertical_p_yes is not None:
        payload["matrix_vertical_p_yes"] = matrix_vertical_p_yes
    if matrix_mediolateral_p_yes is not None:
        payload["matrix_mediolateral_p_yes"] = matrix_mediolateral_p_yes
    if matrix_anteroposterior_p_yes is not None:
        payload["matrix_anteroposterior_p_yes"] = matrix_anteroposterior_p_yes
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

def load_poset_from_json(
    path: str,
) -> Tuple[
    List[Structure],
    List[List[Union[int, float]]],
    List[List[Union[int, float]]],
    List[List[Union[int, float]]],
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
    def _normalize_matrix(M: list) -> List[List[Union[int, float]]]:
        mat: List[List[Union[int, float]]] = [[-2 for _ in range(n)] for _ in range(n)]
        has_probability = False
        if not isinstance(M, list):
            for i in range(n):
                mat[i][i] = -1
            return mat
        for i in range(min(n, len(M))):
            row = M[i]
            if not isinstance(row, list):
                continue
            for j in range(min(n, len(row))):
                try:
                    raw = row[j]
                    if raw is None:
                        mat[i][j] = -2
                        continue
                    fv = float(raw)
                    if -2 <= fv <= 1 and abs(fv - round(fv)) < 1e-9:
                        mat[i][j] = int(round(fv))
                    elif 0.0 <= fv <= 1.0:
                        mat[i][j] = fv
                        has_probability = True
                    else:
                        mat[i][j] = -2
                except (TypeError, ValueError):
                    mat[i][j] = -2
        # Diagonal convention depends on matrix kind:
        # - discrete: explicit NO (-1)
        # - probability: P(self above self) = 0.0
        for i in range(n):
            mat[i][i] = 0.0 if has_probability else -1
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