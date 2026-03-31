# Tests

## Requirements

- **Python ≥ 3.9** (matches `pyproject.toml`; the package uses modern type syntax).
- **pytest** (install once, see below).

## Install pytest

From the **repository root** (the folder that contains `src/` and `tests/`):

```bash
python3 -m pip install pytest
```

Or install the project with dev extras (if you use the optional `dev` group from `pyproject.toml`):

```bash
python3 -m pip install -e ".[dev]"
```

## Run all tests

```bash
cd /path/to/Anatomy_Posets
python3 -m pytest tests/ -v
```

## Run a single file

```bash
python3 -m pytest tests/test_anatomical_agent.py -v
```

## Run one test by name

```bash
python3 -m pytest tests/test_anatomical_agent.py::test_anatomical_chain_expert_completes_matrix -v
```

(Use tab-completion or copy the exact name from `pytest --collect-only`.)

## Notes

- Tests import the package as `src.anatomy_poset` with `tests/conftest.py` adding the repo root to `sys.path`.
- `tests/conftest.py` also adds the `tests/` directory so modules can `from helpers import ...` (`tests/helpers.py`).
- **`test_builder.py`** — `MatrixBuilder` (sorting, `path_exists`, Hasse reduction, `next_pair`).
- **`test_matrix_builder.py`** — `MatrixBuilder` (tri-valued `M`, propagation, `next_pair` with `-2`).
- If `python3` points to an old interpreter (e.g. 3.8), use `python3.11` or another 3.9+ binary.

---

## Merging (for reading tests)

**Flow:** load → **`align_matrix_lists_to_reference`** (matrices + optional count grids) → **`apply_canonical_per_axis_orders`** (per-axis CoM sort + lower-triangle seal) → **`aggregate_matrices_with_counts(..., answer_weight_grids=...)`**.

**Per cell:** each file skips the mean if `-2` / missing; else tri-valued codes or \(\mu=2P-1\) from probability, each with weight **1** unless **`matrix_*_n_answered`** has an int ≥ 1. \(\mu = \sum w_k\mu_k / \sum w_k\). **`aggregate_to_p_yes_matrix`:** \(P=(\mu+1)/2\) or `null` if no weight. Saves **`matrix_*_n_answered`** (Σw, used for the next merge) and **`matrix_*_n_notasked`** (how many files were missing).

**Tests:** **`test_aggregation.py`** (weights, nulls, `io`); **`test_merge_regression.py`** (fixtures, no sidecars).

```bash
python3 -m pytest tests/test_aggregation.py tests/test_merge_regression.py -v
```

## JSON (`io` — save / load)

**`save_poset_to_json`** writes one object: **`structures`** (list of `{name, com_vertical, com_lateral, com_anteroposterior}`), **`matrix_vertical`**, **`matrix_mediolateral`**, **`matrix_anteroposterior`** (tri-valued ints or probability with `null` off-diagonal where unknown). Optional **`matrix_*_n_answered`** / **`matrix_*_n_notasked`** (ints or `null` per cell, same shape) are only added if passed. Top-level merge metadata goes through **`extra`** (e.g. `merged_probability_matrix`, `merged_from_raters`, `merged_source_files`). There is no separate `matrix_*_p_yes` key — **P lives in `matrix_*`**.

**`load_poset_from_json`** returns a **`PosetFromJson`** dataclass (not a tuple): matrices + optional count fields when those keys exist. Older files without count keys still load; backward compat for edge-only JSON unchanged.
