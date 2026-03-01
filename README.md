# Enforcing Anatomical Spatial Consistency in Multi-Organ Segmentation via Posets

## Overview
Integrates explicit anatomical knowledge into deep learning-based medical image segmentation using Partially Ordered Sets (Posets). Addresses errors like anatomically impossible organ positions from models such as TotalSegmentator and VIBESegmentator.

## Goal
Bridge classical anatomical knowledge with modern deep learning to create accurate and anatomically coherent segmentation models.

## Workflow
1. **Clinical Knowledge Extraction** – interactive GUI for clinicians to encode spatial relations.
2. **Post-Processing Correction** – enforce spatial rules to clean model outputs.
3. **Weakly-Supervised Training** – use cleaned outputs as pseudo-labels to train 3D networks.

---

## Setup and run (Poset Constructor GUI)

### Requirements
- **Python 3** (3.8+)
- **PySide6** (Qt for Python)

### Setup
1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install PySide6
   ```
   Or with a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   pip install PySide6
   ```

### Run the application
From the project root:
```bash
python poset_constructor_gui.py
```
To load a specific structures file at startup:
```bash
python poset_constructor_gui.py path/to/structures.json
```

Input JSON format (e.g. `Input_CoM_structures/test_structures.json`):
```json
{
  "structures": [
    {"name": "Skull", "com_vertical": 90.0},
    {"name": "Spine", "com_vertical": 70.0}
  ]
}
```

Output posets are saved under `Output_constructed_posets/` (autosave during each query).

---

## Buttons (Poset Constructor GUI)

### Main window — *Anatomical Structures (Input)*

| Button | Action |
|--------|--------|
| **Load Structures** | Opens a file dialog to load a JSON file of structures (name + CoM on vertical axis). Fills the table and sets the autosave path from the loaded file. |
| **+ Add Structure** | Appends a new empty row to the table. Enter name and CoM value manually. |
| **− Remove Selected** | Deletes the currently selected table row(s). |
| **▶ Start Poset Construction** | Validates the table, builds the poset from your structures, then shows the **Definition** dialog. After you click *Understood*, opens the **Expert Query** window with Yes/No questions. Disabled until the query session is closed. Progress is autosaved after each answer. |
| **View Poset** | Opens a file dialog to pick a saved poset JSON file, then opens the **Poset Viewer** (list of structures + edges and interactive Hasse diagram). |

### Definition dialog (before questions)

| Button | Action |
|--------|--------|
| **Understood** | Closes the definition dialog and starts the query session (Expert Query window). |

### Expert Query window (questions)

| Button | Action |
|--------|--------|
| **← Undo** | Reverts the last answer: removes that relation from the poset and shows the same question again. Disabled when there is no previous answer. |
| **Yes** | Records that the *first* structure in the question is strictly above the *second* (adds the relation), then advances to the next question. |
| **No** | Records that the first is *not* strictly above the second, then advances to the next question. |
| **Done** | Shown when all queries are finished. Closes the Expert Query window and re-enables **Start Poset Construction** in the main window. |

## Related Models
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [VIBESegmentator](https://github.com/robert-graf/VIBESegmentator/tree/main)
- [Segment Anything Model 3 (SAM3)](https://ai.meta.com/research/sam3/)

## Key References
- KG-SAM (2025)
- Learning to Zoom with Anatomical Relations (NeurIPS 2025)
- 3D Spatial Priors (STIPPLE)
