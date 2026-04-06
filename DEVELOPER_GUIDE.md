# Developer Guide: Data Organization and Workflow

## Project Structure

```
Anatomy_Posets/
├── src/anatomy_poset/
│   ├── core/
│   │   ├── config.py              # Path definitions (INPUT_DIR, OUTPUT_DIR)
│   │   ├── axis_models.py         # Structure dataclass and axis constants
│   │   ├── matrix_builder.py      # Core algorithm: gap-based queries, propagation
│   │   ├── matrix_aggregation.py  # Multi-annotator merging
│   │   └── file_io.py             # JSON serialization (load/save)
│   ├── gui/
│   │   ├── main_window.py         # App entry point, file selection flow
│   │   ├── query_dialog.py        # Query interface, anatomy views, slicing
│   │   ├── poset_viewer.py        # Hasse diagram visualization
│   │   └── dialog_widgets.py      # Reusable UI components
│   └── scripts/                   # Utilities (merge, convert, etc.)
├── tests/
│   ├── test_matrix_builder.py    # 19 unit tests (all pass)
│   ├── helpers.py                 # Test fixtures
│   └── conftest.py                # pytest configuration
├── data/
│   ├── structures/                # INPUT_DIR: CoM definitions
│   │   └── *.json                 # input/Input_CoM_structures → renamed to structures/
│   ├── posets/                    # OUTPUT_DIR: annotation results
│   │   ├── tests/                 # test/debug sessions
│   │   ├── clinician_sessions/    # real clinical annotations (when added)
│   │   ├── merged_sessions/       # aggregated multi-rater results
│   │   └── feedback/              # annotator feedback (when added)
│   ├── segmentations/             # imaging segmentations (external)
│   ├── imaging_datasets/          # imaging data (external)
│   └── totalseg_output/           # TotalSegmentator results (external)
├── assets/
│   ├── images/                    # Anatomy diagrams (Complete Anatomy)
│   └── visible_human_tensors/     # Full-body volume slices (downsampled)
├── report/
│   └── main.tex                   # LaTeX paper: Introduction, Methods, Results, Discussion
└── requirements.txt               # Python dependencies
```

---

## Core Configuration: `src/anatomy_poset/core/config.py`

```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ASSETS_DIR = PROJECT_ROOT / "assets"
INPUT_DIR = PROJECT_ROOT / "data" / "structures"      # Structure CoM definitions
OUTPUT_DIR = PROJECT_ROOT / "data" / "posets"         # Annotation sessions
```

**Key Decision:** Folder names are semantic (`structures`, `posets`) rather than snake_case (`Input_CoM_structures`, `Output`). New code should use constants from `config.py` rather than hardcoded paths.

---

## Data Files: Format and Organization

### Structure Definition (INPUT_DIR)

**File:** `data/structures/*.json`
**Format:** Array of structures with CoM coordinates

```json
[
  {
    "name": "Skull",
    "com_vertical": 90.0,
    "com_lateral": 50.0,
    "com_anteroposterior": 50.0
  },
  ...
]
```

**Usage:**
- Loaded via GUI: File menu → Load structures
- Passed to `MatrixBuilder` to initialize canonical ordering and matrices
- Saved in every poset file (output) so merges preserve structure identity

---

### Poset File (OUTPUT_DIR)

**File:** `data/posets/tests/*.json` (or clinician_sessions/merged_sessions/)
**Format:** Structures + three tri-valued matrices

```json
{
  "structures": [...],                       // Full list, same order for all axes
  "matrix_vertical": [[+1, -1, null, ...], ...],
  "matrix_mediolateral": [[null, ...], ...],
  "matrix_anteroposterior": [[null, ...], ...],
  "matrix_vertical_n_answered": null,        // Optional: for probability-based re-merging
  "matrix_mediolateral_n_answered": null,
  "matrix_anteroposterior_n_answered": null,
  "extra": {}                                // Arbitrary metadata (for future use)
}
```

**Values** in matrices:
- `+1`: "Yes" (one structure is strictly above the other)
- `-1`: "No" (not strictly above)
- `0`: "Unsure" (expert stated uncertainty)
- `null`: Not asked / not evaluated

**Invariants:**
- Diagonal entries: always `-1`
- Lower triangle: always `-1` (sealed by CoM-based canonical order)
- Upper triangle: where queries happen (`+1`, `0`, `-1`, or `null`)

---

## File Selection Flow (Session 2 Change)

### New Implementation in `main_window.py`

When user clicks "Start":

```
┌─────────────────────────────┐
│ "Start New or Continue?"    │
├──────────┬──────────────────┤
│   Yes    │       No         │
└──────┬───┴──────────┬───────┘
       │              │
       ▼              ▼
getOpenFileName()  getSaveFileName()
    │               (DontConfirmOverwrite)
    │              │
    ▼              ▼
Load existing    Create new
file with all    file with
three matrices   empty matrices
    │              │
    └──────┬───────┘
           ▼
    Start Query Dialog
```

**Key Points:**
1. **Intent first:** User chooses "continue" or "new" before file selection
2. **No OS dialogs:**
   - `getOpenFileName()` doesn't trigger "replace?" warnings
   - `getSaveFileName(..., DontConfirmOverwrite)` suppresses OS confirmation
3. **Matrix preservation:** When loading, all three axes are loaded; when saving, only current axis updated

**Code Location:** `src/anatomy_poset/gui/main_window.py:414–481`

---

## Algorithm Overview: `src/anatomy_poset/core/matrix_builder.py`

### Initialization
```python
mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
# Sorts structures by CoM (descending) on the chosen axis
# Initializes M: diagonal -1, lower triangle -1, upper triangle null
```

### Query Loop
```python
while True:
    pair = mb.next_pair()       # Returns (i, j) with i < j
    if pair is None:            # All gaps exhausted
        break

    expert_answer = get_expert_response(pair)  # +1, -1, or 0
    mb.record_response_matrix(i, j, expert_answer)
    # Propagates: transitive, symmetric, bilateral, cycle detection
```

### Skip Conditions
Pairs are not asked if:
- Already answered (not null)
- Implied by transitivity (path of +1 edges exists)
- Same-core bilateral pair (left/right on vertical axis)
- CoM inequality precludes the relationship

### Propagation
After each answer:
- **Transitivity:** +1 chains → +1 cells + inverse -1
- **Bilateral:** Vertical axis left-right mirroring
- **Cycle detection:** Contradictions sealed as 0 (unsure)
- **Closure:** Reachable unknowns filled from +1 paths

---

## Testing: `tests/test_matrix_builder.py`

**Coverage:**
- Matrix initialization (diagonal, lower triangle, upper triangle)
- Response recording and inverse logic
- Transitivity propagation (single-step, multi-step)
- Bilateral symmetry (same-core never asked, double-mirror fill)
- Exhaustive validation (all answer types: yes, no, not-sure)
- No-duplicate guarantee (19 tests across 6 test structures)

**Run Tests:**
```bash
python -m pytest tests/test_matrix_builder.py -v
```

**All 19 tests pass** ✓

---

## Multi-Annotator Merging: `src/anatomy_poset/core/matrix_aggregation.py`

### Workflow
1. **Align structures:** Match names + CoM across raters (within tolerance)
2. **Reseal per axis:** Each rater's matrix resealed (lower triangle → -1)
3. **Aggregate per cell:** Collect answered codes (-1, 0, +1) from each rater, exclude null
4. **Probability:** `P(yes) = (μ + 1) / 2` where `μ = mean of tri-values`
5. **Hasse extraction:** Use edges with `P(yes) == 1.0` only

### Chained Merging
If merging previously merged results, use:
- `matrix_*_n_answered`: Weight factor per cell (count of original annotators)
- Per-cell aggregation becomes weighted to preserve calibration

---

## GUI Architecture: `src/anatomy_poset/gui/`

### Main Window (`main_window.py`)
- Structures file selection
- Axis selection (vertical, mediolateral, anteroposterior)
- Region subset filtering
- File selection flow (Intent → File dialog → Query dialog)

### Query Dialog (`query_dialog.py`)
- **Anatomy panel:** Structure tabs (Skeleton, Muscles, GI, etc.) with Front/Side/Rear views
- **Questions panel:** Current pair, three answer buttons (Yes/No/Unsure) + Undo
- **Axial slicing:** Full-body volume viewer with plane selector and slice navigation
- **Progress:** Query count and completion estimate
- **Saving:** Autosave on each answer, explicit save on exit

### Poset Viewer (`poset_viewer.py`)
- Loads and visualizes poset as directed acyclic graph (DAG)
- Hasse diagram extraction (cover edges only)
- Merge functionality (multiple JSON files)
- Export to GraphML/PDF

---

## Common Tasks

### Add a New Data Path
1. Update `src/anatomy_poset/core/config.py`
2. Create the directory if needed
3. Reference via constant in code (never hardcode paths)

### Test a Specific Feature
Example: Test bilateral mirroring on vertical axis
```bash
python -m pytest tests/test_matrix_builder.py::test_double_mirror_fill_both_sides_bilateral -v
```

### Load and Inspect a Poset File
```python
from src.anatomy_poset.core.file_io import load_poset_from_json
poset = load_poset_from_json("data/posets/tests/test1.json")
print(f"Structures: {[s.name for s in poset.structures]}")
print(f"Vertical matrix shape: {len(poset.matrix_vertical)}x{len(poset.matrix_vertical[0])}")
```

### Merge Multiple Sessions
1. GUI: Open Poset Viewer → "Merge JSON files…"
2. Select 2+ poset files
3. Choose aggregation type (consensus or probability-weighted)
4. Specify output file
5. Merged result saved with `P(yes)` values per cell

---

## Important Design Decisions

1. **Tri-valued matrices:** `+1 / 0 / -1 / null` preserves uncertainty where binary labels would lose information
2. **Gap-based iteration:** O(n²) questions replaced with O(n² / k) where k is sparsity factor (typically 3–5×)
3. **CoM-based canonical order:** Ensures consistency across axes and raters
4. **Bilateral symmetry:** Hard constraint (not asked) rather than soft (post-hoc matching)
5. **Cycle detection via sealing:** When contradictions arise, mark as 0 (unsure) to prevent infinite loops
6. **Multi-annotator aggregation via probability:** Easier to interpret and re-merge than raw consensus

---

## Future Considerations

- **Partial orders from imaging:** Extract implicit spatial relations from segmentation masks
- **Uncertainty propagation:** Model CoM coordinate distributions, propagate through closure
- **Cross-dataset validation:** Apply learned posets to different imaging modalities
- **Integration with segmentation refinement:** Feed poset constraints as loss terms to improve models

---

## References

- **Repository:** `/Users/rabbit/Desktop/ETH/semester_project/Anatomy_Posets`
- **Report:** `report/main.tex` (complete system description, complexity analysis, validation)
- **Testing Guide:** `TESTING_FILE_FLOW.md` (file selection flow verification)
- **Main API:** `src/anatomy_poset/core/matrix_builder.MatrixBuilder`
