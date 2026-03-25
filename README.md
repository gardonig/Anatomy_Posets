# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

## Overview

This project enforces **anatomical spatial consistency** in multi-organ segmentation using partially ordered sets (posets).  
Clinicians specify relative positions of structures (vertical, mediolateral, anteroposterior), and the resulting posets can be used to check and correct segmentation outputs.

---

## Relation matrices (one per axis)

For each anatomical axis (vertical, mediolateral, anteroposterior), the app maintains a square **tri-valued relation matrix** \(M\) with one row/column per structure (after sorting—see below).

| Entry | Meaning |
|-------|---------|
| **+1** | Structure \(i\) is **strictly above** structure \(j\) along that axis (directed “yes”). |
| **0** | The pair was **asked**; the expert is **not sure** (still directional). |
| **−1** | \(i\) is **not** strictly above \(j\) (includes “no”, overlap, or opposite direction implied by the CoM prior). |
| **−2** | **Not asked yet** (or never queried for that directed cell). |

**Diagonal** entries are fixed to **−1** (self-relations are not “strictly above”).  
After **canonical sorting** by CoM on the active axis (descending), **lower triangle** entries with **column index \(j < i\)** are **sealed** to **−1**: in that order, structure \(i\) cannot lie strictly above \(j\) on the \(i\!>\!j\) side of the diagonal. The **strict upper triangle** (\(j > i\) in row-major layout) is where expert queries and inference fill **+1**, **0**, or **−1**; **−2** is “still open”.

---

## How the matrix is built (algorithm)

1. **Sort structures** by the chosen axis CoM (descending), matching `MatrixBuilder`.
2. **Initialize** \(M\): diagonal **−1**, lower triangle **−1**, upper triangle **−2** (plus bilateral / equal-CoM rules as implemented).
3. **Gap-based queries** (`next_pair`): pairs \((i, j)\) with \(j = i + \text{gap}\) are considered in order of increasing gap (1, 2, …). Pairs already decided (not **−2**), implied by **transitive +1** reachability, or skipped by vertical bilateral symmetry rules are not asked again.
4. **Optional region subsets**: you can restrict **which pairs are asked** to those whose **both** endpoints lie in selected body-region presets; the **saved JSON still lists every structure** and the same **n×n** matrix size so merges stay compatible.
5. **After each answer**, **propagation** updates the matrix: transitive **+1** chains, inverse **−1** when a side is **+1**, mirroring for left/right cores on the vertical axis, and **closure of unknowns** where reachability on **+1** edges forces a direction.  
6. **Saved file** stores three matrices (`matrix_vertical`, `matrix_mediolateral`, `matrix_anteroposterior`) plus the full **structures** list.

**Hasse diagram (poset viewer)** shows only **cover edges** derived from **+1** entries (transitive reduction of the strict “above” relation). It does **not** draw “unsure” **0** pairs.

---

## Merging multiple raters / sessions

The poset viewer’s **Merge JSON files…** combines several saved posets that describe the **same** set of anatomical structures.

1. **Align** non-reference files to the **first** file’s structure order: index \(i\) must refer to the same organ across files. If the JSON order differs, matrices are **permuted** by matching **name + CoM** (within tolerance), or merge fails if sets are incompatible.
2. **Canonical sort per axis** (vertical / mediolateral / anteroposterior) reorders each rater’s matrix for that axis (CoM descending) and **reseals** the lower triangle to **−1**.
3. **Per-cell aggregation** (`aggregate_matrices_with_counts`): for each directed pair \((i,j)\), each rater contributes **−2** (not asked), or **−1 / 0 / +1** if answered. **−2** is excluded from the mean and vote counts.
4. **Consensus matrix** (for Hasse / saved merged JSON): **plurality vote** on \(\{-1,0,+1\}\) over answered raters. If **two** values tie for the count, the tie is broken by **rounding the arithmetic mean** to \(\{-1,0,+1\}\) (same convention as `aggregate_to_consensus_matrix` in code).
5. **Merged heatmap** (optional): **P(yes) = (μ + 1) / 2** where **μ** is the mean of answered codes **only** (so partial overlap with one rater **−2** and one **0** gives **μ = 0** → **P = 0.5**; **0** and **+1** from both raters gives **μ = 0.5** → **P = 0.75**).
6. **Save** writes one JSON with **structures** in vertical-CoM order and **reindexed** mediolateral / anteroposterior matrices to that same label order.

---

## Setup

Requirements:

- Python 3.9+

Install (recommended in a virtual environment):

```bash
pip install -e .
```

Alternatively:

```bash
pip install -r requirements.txt
```

---

## Running the GUI

### Via installed CLI

After installation you can launch the GUI with:

```bash
anatomy-poset
```

To start with a specific structures file (new CoMs):

```bash
anatomy-poset data/Input_CoM_structures/test_structures_2.json
```

### Via `run.py` (no installation)

From the project root:

```bash
python run.py
```

With an explicit input file:

```bash
python run.py data/Input_CoM_structures/test_structures_2.json
```

---

## Input JSON format

Example (`data/Input_CoM_structures/test_structures.json`):

```json
{
  "structures": [
    {
      "name": "Skull",
      "com_vertical": 90.0,
      "com_lateral": 50.0,
      "com_anteroposterior": 50.0
    }
  ]
}
```

- **`com_vertical`**: CoM along the superior–inferior (vertical) axis, scaled to \([0, 100]\):  
  `0` = toes / feet (most inferior), `100` = vertex / head (most superior).
- **`com_lateral`**: CoM along the right–left (lateral) axis (patient’s perspective), scaled to \([0, 100]\):  
  `0` = far right (e.g. right thumb), `100` = far left (e.g. left thumb).
- **`com_anteroposterior`**: CoM along the back–front (anteroposterior) axis, scaled to \([0, 100]\):  
  `0` = back / dorsal side, `100` = front / ventral side.

Output posets are saved under `data/Output_constructed_posets/` (autosave triggers during each query).

---

## Full-body volume tensors

The GUI’s **Full-Body Volume** panel and the helper script `scripts/view_full_body_male.py` expect precomputed NumPy tensors:

- `assets/visible_human_tensors/full_body_tensor_rgb.npy` — RGB tensor `(Z, Y, X, 3)`
- `assets/visible_human_tensors/full_body_tensor.npy` — grayscale tensor `(Z, Y, X)`

These files are **large and not tracked in git**.

- To obtain them, please contact **Gian** (e-mail); he will provide a download link (e.g. via Dropbox).  
- After downloading, place the `.npy` files into `assets/visible_human_tensors/` so the GUI and scripts can find them automatically.

---

## Project structure (short)

```text
anatomy_poset/
├── assets/                      # UI images, tensors, diagrams
├── data/
│   ├── Input_CoM_structures/    # Input CoM JSONs
│   └── Output_constructed_posets/  # Saved posets (autosave)
├── src/anatomy_poset/
│   ├── core/                    # Models, IO, MatrixBuilder
│   └── gui/                     # PySide6 GUI (main window, dialogs, poset_viewer)
├── scripts/                     # Helper scripts (e.g. view_full_body_male.py)
├── run.py                       # Quick-start GUI launcher
└── README.md
```
