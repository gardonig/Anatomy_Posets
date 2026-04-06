# Quick Reference: Common Development Tasks

## Running the Application

### GUI via installed CLI
```bash
anatomy-poset [structures_file.json]
```

### GUI via run.py
```bash
python run.py
```

### With specific structures file
```bash
python run.py data/structures/test_structures_2.json
```

---

## Running Tests

### All tests
```bash
python -m pytest tests/
```

### Specific test file
```bash
python -m pytest tests/test_matrix_builder.py -v
```

### Specific test function
```bash
python -m pytest tests/test_matrix_builder.py::test_bilateral_same_core_never_asked -v
```

### With coverage report
```bash
python -m pytest tests/ --cov=src/anatomy_poset --cov-report=html
```

---

## Code Quality

### Format code (Black)
```bash
black src/ tests/
```

### Type checking (Pylance/mypy)
```bash
mypy src/
```

### Lint (Ruff/Flake8)
```bash
ruff check src/ tests/
```

---

## Key Imports and Classes

### Creating a Matrix Builder
```python
from src.anatomy_poset.core.matrix_builder import MatrixBuilder
from src.anatomy_poset.core.axis_models import AXIS_VERTICAL, Structure

structures = [
    Structure("Brain", 90.0, 50.0, 50.0),
    Structure("Thorax", 65.0, 50.0, 50.0),
    Structure("Pelvis", 35.0, 50.0, 50.0),
]

mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
while True:
    pair = mb.next_pair()
    if pair is None:
        break
    # Present pair to expert...
    mb.record_response_matrix(pair[0], pair[1], answer)
```

### Loading/Saving Posets
```python
from src.anatomy_poset.core.file_io import load_poset_from_json, save_poset_to_json

# Load
poset = load_poset_from_json("data/posets/tests/test1.json")

# Save
save_poset_to_json(
    "data/posets/tests/new_session.json",
    poset.structures,
    poset.matrix_vertical,
    poset.matrix_mediolateral,
    poset.matrix_anteroposterior,
)
```

### Merging Posets
```python
from src.anatomy_poset.core.matrix_aggregation import aggregate_matrices_with_counts

count_data = aggregate_matrices_with_counts([poset1.matrix_vertical, poset2.matrix_vertical])
# count_data['aggregated']: per-cell mean
# count_data['n_answered']: per-cell count of non-null responses
```

---

## File Paths and Configuration

### From config.py
```python
from src.anatomy_poset.core.config import INPUT_DIR, OUTPUT_DIR, ASSETS_DIR, PROJECT_ROOT

# INPUT_DIR: data/structures/
# OUTPUT_DIR: data/posets/
# ASSETS_DIR: assets/
# PROJECT_ROOT: /Users/rabbit/Desktop/ETH/semester_project/Anatomy_Posets
```

### Standard paths (use these, not hardcoded strings!)
```python
OUTPUT_DIR / "tests" / "my_session.json"          # data/posets/tests/my_session.json
OUTPUT_DIR / "clinician_sessions" / "session.json"  # data/posets/clinician_sessions/session.json
OUTPUT_DIR / "merged_sessions" / "merged.json"      # data/posets/merged_sessions/merged.json
INPUT_DIR / "structures.json"                       # data/structures/structures.json
```

---

## Debugging Matrix State

### Print raw matrix
```python
mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
for i, row in enumerate(mb.M):
    print(f"Row {i}: {row}")
```

### Check if pair will be asked
```python
i, j = 0, 2
if mb.M[i][j] is None:
    print(f"Pair ({i},{j}) will be asked")
else:
    print(f"Pair ({i},{j}) is already known: {mb.M[i][j]}")
```

### Inspect current progress
```python
remaining = mb.estimate_remaining_questions()
print(f"Questions remaining: {remaining}")
```

---

## Common Pitfalls

### Matrix index order
- Always: `matrix[row][col]` = `matrix[i][j]` where `i` is structure index, `j` is target
- Not: `matrix[x][y]` or mixed indexing

### Null vs. -2
- Old code used `-2` for "not asked"; migrated to `None` (Python `null` in JSON)
- Always check `is None` or `is not None`, never `== -2`

### Bilateral pairs
- Only apply to vertical axis by default
- Left/right pairs have same CoM on mediolateral axis
- Never ask "Left X vs. Right X" (same-core constraint)
- Always mirror responses across left-right pairs

### Structure ordering
- Structures are sorted by CoM (descending) per axis
- Order differs across axes (vertical, mediolateral, anteroposterior)
- Always use the sorted order from `MatrixBuilder.structures`, not input order

### Reloading sessions
- Load all three matrices from file (not just current axis)
- Modify only current axis on save
- Always call `seal_lower_triangle_com_prior()` before save if loading partial/legacy data

---

## Version Info

- **Python:** 3.9+
- **Main dependencies:** PySide6 (GUI), NumPy (arrays), pytest (tests)
- **QT version:** PySide6 (Qt6)

See `requirements.txt` for full list.

---

## Documentation Files

- `DEVELOPER_GUIDE.md`: Complete system architecture and design decisions
- `TESTING_FILE_FLOW.md`: Detailed scenarios for file selection flow verification
- `SESSION_2_CHANGES.md`: Summary of recent modifications and testing checklist
- `report/main.tex`: Academic paper with Introduction, Methods, Results, Discussion

---

## Getting Help

1. **Design questions:** See `DEVELOPER_GUIDE.md`
2. **Testing:** See `TESTING_FILE_FLOW.md`
3. **Algorithm details:** See `report/main.tex` and code docstrings
4. **Recent changes:** See `SESSION_2_CHANGES.md`
5. **Test examples:** See `tests/test_matrix_builder.py`
