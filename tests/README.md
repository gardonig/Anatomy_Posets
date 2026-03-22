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
- If `python3` points to an old interpreter (e.g. 3.8), use `python3.11` or another 3.9+ binary.
