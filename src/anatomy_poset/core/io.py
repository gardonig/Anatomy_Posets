import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

from .axis_models import Structure


@dataclass
class PosetFromJson:
    """Payload from :func:`load_poset_from_json` (structures + per-axis matrices)."""

    structures: List[Structure]
    matrix_vertical: List[List[Union[int, float, None]]]
    matrix_mediolateral: List[List[Union[int, float, None]]]
    matrix_anteroposterior: List[List[Union[int, float, None]]]
    n_answered_vertical: Optional[List[List[Optional[int]]]] = None
    n_answered_mediolateral: Optional[List[List[Optional[int]]]] = None
    n_answered_anteroposterior: Optional[List[List[Optional[int]]]] = None
    n_notasked_vertical: Optional[List[List[Optional[int]]]] = None
    n_notasked_mediolateral: Optional[List[List[Optional[int]]]] = None
    n_notasked_anteroposterior: Optional[List[List[Optional[int]]]] = None


def load_structures_from_json(path: str) -> List[Structure]:
    """
    Load a list of structures from JSON.
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
    matrix_vertical_n_answered: Optional[List[List[Optional[int]]]] = None,
    matrix_vertical_n_notasked: Optional[List[List[Optional[int]]]] = None,
    matrix_mediolateral_n_answered: Optional[List[List[Optional[int]]]] = None,
    matrix_mediolateral_n_notasked: Optional[List[List[Optional[int]]]] = None,
    matrix_anteroposterior_n_answered: Optional[List[List[Optional[int]]]] = None,
    matrix_anteroposterior_n_notasked: Optional[List[List[Optional[int]]]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save relation matrices to JSON. All axes stored in one file.

    Matrix values are tri-valued ``{-1, 0, +1}`` or probability cells in ``[0,1]``;
    unanswered cells are ``null`` (JSON) / ``None`` (Python).
    Legacy files with ``-2`` are normalised to ``null`` on load.

    Optional ``matrix_*_n_answered`` / ``matrix_*_n_notasked``: per-cell merge statistics
    (integers or JSON ``null``) aligned with the same indexing as ``matrix_*``. For merged
    probability saves, ``n_answered`` is the **effective weight sum** (Σw) used when merging,
    not only the file count. Omitted when not provided.

    ``extra`` is merged into the top-level JSON object (e.g. merge metadata).
    """
    if matrix_mediolateral is None:
        matrix_mediolateral = []
    if matrix_anteroposterior is None:
        matrix_anteroposterior = []

    payload: Dict[str, Any] = {
        "structures": [
            {
                "name": s.name,
                "com_vertical": s.com_vertical,
                "com_lateral": s.com_lateral,
                "com_anteroposterior": s.com_anteroposterior,
            }
            for s in structures
        ],
        "matrix_vertical": matrix_vertical,
        "matrix_mediolateral": matrix_mediolateral,
        "matrix_anteroposterior": matrix_anteroposterior,
    }
    if matrix_vertical_n_answered is not None:
        payload["matrix_vertical_n_answered"] = matrix_vertical_n_answered
    if matrix_vertical_n_notasked is not None:
        payload["matrix_vertical_n_notasked"] = matrix_vertical_n_notasked
    if matrix_mediolateral_n_answered is not None:
        payload["matrix_mediolateral_n_answered"] = matrix_mediolateral_n_answered
    if matrix_mediolateral_n_notasked is not None:
        payload["matrix_mediolateral_n_notasked"] = matrix_mediolateral_n_notasked
    if matrix_anteroposterior_n_answered is not None:
        payload["matrix_anteroposterior_n_answered"] = matrix_anteroposterior_n_answered
    if matrix_anteroposterior_n_notasked is not None:
        payload["matrix_anteroposterior_n_notasked"] = matrix_anteroposterior_n_notasked
    if extra:
        payload.update(extra)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_poset_from_json(path: str) -> PosetFromJson:
    """
    Load poset(s) from JSON.

    Returns a :class:`PosetFromJson` with optional ``n_answered_*`` / ``n_notasked_*`` matrices
    when present in the file (merged saves).
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    structures_data = data.get("structures", [])
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
        adj = data.get(key_adj)
        if isinstance(adj, list) and all(isinstance(row, list) for row in adj):
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

    if M_v is None:
        M_v = _fallback_matrix_from_edges("edges_vertical", "adjacency_vertical")
    if M_ml is None:
        M_ml = _fallback_matrix_from_edges("edges_mediolateral", "adjacency_mediolateral")
    if M_ap is None:
        M_ap = _fallback_matrix_from_edges("edges_anteroposterior", "adjacency_anteroposterior")

    def _normalize_matrix(M: list) -> List[List[Union[int, float, None]]]:
        # None = not asked yet (unanswered); -1/0/+1 = answered; float [0,1] = probability.
        mat: List[List[Union[int, float, None]]] = [[None for _ in range(n)] for _ in range(n)]
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
                        # Already None (unanswered) — leave as None.
                        continue
                    fv = float(raw)
                    # Legacy -2 and anything outside valid range → None (unanswered).
                    if abs(fv - (-2)) < 1e-9:
                        mat[i][j] = None
                    elif -1 <= fv <= 1 and abs(fv - round(fv)) < 1e-9:
                        mat[i][j] = int(round(fv))
                    elif 0.0 <= fv <= 1.0:
                        mat[i][j] = fv
                        has_probability = True
                    else:
                        mat[i][j] = None
                except (TypeError, ValueError):
                    mat[i][j] = None
        for i in range(n):
            mat[i][i] = 0.0 if has_probability else -1
        return mat

    def _normalize_count_matrix(M: Any) -> Optional[List[List[Optional[int]]]]:
        if not isinstance(M, list):
            return None
        mat: List[List[Optional[int]]] = [[None] * n for _ in range(n)]
        for i in range(min(n, len(M))):
            row = M[i]
            if not isinstance(row, list):
                continue
            for j in range(min(n, len(row))):
                raw = row[j]
                if raw is None:
                    mat[i][j] = None
                else:
                    try:
                        mat[i][j] = int(raw)
                    except (TypeError, ValueError):
                        mat[i][j] = None
        return mat

    M_v_norm = _normalize_matrix(M_v)
    M_ml_norm = _normalize_matrix(M_ml)
    M_ap_norm = _normalize_matrix(M_ap)

    return PosetFromJson(
        structures=structures,
        matrix_vertical=M_v_norm,
        matrix_mediolateral=M_ml_norm,
        matrix_anteroposterior=M_ap_norm,
        n_answered_vertical=_normalize_count_matrix(data.get("matrix_vertical_n_answered")),
        n_answered_mediolateral=_normalize_count_matrix(data.get("matrix_mediolateral_n_answered")),
        n_answered_anteroposterior=_normalize_count_matrix(
            data.get("matrix_anteroposterior_n_answered")
        ),
        n_notasked_vertical=_normalize_count_matrix(data.get("matrix_vertical_n_notasked")),
        n_notasked_mediolateral=_normalize_count_matrix(data.get("matrix_mediolateral_n_notasked")),
        n_notasked_anteroposterior=_normalize_count_matrix(
            data.get("matrix_anteroposterior_n_notasked")
        ),
    )
