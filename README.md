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
2. (Recommended) create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate   # on Windows: venv\Scripts\activate
   ```
3. Install dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
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
    {"name": "Skull", "com_vertical": 90.0, "com_lateral": 50.0},
    {"name": "Spine", "com_vertical": 70.0, "com_lateral": 50.0}
  ]
}
```

- **`com_vertical`**: CoM along the superior–inferior (**vertical**) axis, scaled to \([0, 100]\):  
  `0` = toes / feet (most inferior), `100` = vertex / head (most superior).
- **`com_lateral`**: CoM along the right–left (**lateral**) axis (patient’s perspective), scaled to \([0, 100]\):  
  `0` = far right (e.g. right thumb), `100` = far left (e.g. left thumb).  
  If `com_lateral` is omitted, it is treated as `0.0`.
- **`com_anteroposterior`** (optional): CoM along the back–front (**anteroposterior**) axis, scaled to \([0, 100]\):  
  `0` = back / dorsal side, `100` = front / ventral side.

Output posets are saved under `Output_constructed_posets/` (autosave during each query).

---

## Buttons (Poset Constructor GUI)

### Main window — *Anatomical Structures (Input)*

| Button | Action |
|--------|--------|
| **Load Structures** | Opens a file dialog to load a JSON file of structures (`name`, `com_vertical`, `com_lateral`, optional `com_anteroposterior`). Fills the table and sets the autosave path from the loaded file. |
| **+ Add Structure** | Appends a new empty row to the table. Enter name, vertical CoM, lateral CoM, and (optionally) anteroposterior CoM manually. |
| **− Remove Selected** | Deletes the currently selected table row(s). |
| **View Poset** | Opens a file dialog to pick a saved poset JSON file (typically from `Output_constructed_posets/`), then opens the **Poset Viewer** (list of structures + Hasse diagrams for vertical, lateral, and anteroposterior axes). |
| **▶ Start Poset Construction** | Validates the table, lets you choose the **axis for this run** (vertical “strictly above”, lateral “strictly to the left of”, or anteroposterior “strictly in front of”), shows the **Definition** dialog(s), and then opens the **Expert Query** window with Yes/No questions. Disabled until the query session is closed. Progress is autosaved after each answer. |

Below the table you also choose the **axis for this run** (Vertical / Lateral / Anteroposterior). Left/right are always interpreted from the **patient’s perspective**.

The three axes follow a simple \([0, 100]\) convention in standard anatomical position:

- **Vertical (Superior–Inferior)**: `0` = toes/feet, `100` = head/vertex.  
- **Lateral (Right–Left)**: `0` = far right (e.g. right thumb), `100` = far left (e.g. left thumb), patient’s view.  
- **Anteroposterior (Back–Front)**: `0` = back (dorsal), `100` = front (ventral).

### Definition dialog (before questions)

| Button | Action |
|--------|--------|
| **Proceed** | Closes the definition dialog and starts the query session (Expert Query window). |

The dialog:
- Welcomes the user and explains the goal of the questionnaire.
- States the assumption of **standard anatomical position** (e.g. supine, arms supinated/standard) and that left/right and above/below refer to this reference.
- Defines the chosen relation for this run:
  - Vertical: “**strictly above**”.
  - Frontal: “**strictly to the left of**”.
- Shows concrete examples (“Question: Is the … → Answer: …”) and, for the vertical axis, example images with attributions.

### Expert Query window (questions)

| Button | Action |
|--------|--------|
| **← Undo** | Reverts the last answer: removes that relation from the poset and shows the same question again. Disabled when there is no previous answer. |
| **Yes** | Records that the *first* structure in the question stands in the **strict relation for the chosen axis** to the *second* (strictly above for vertical runs, strictly to the left of for frontal runs), then advances to the next question. |
| **No** | Records that the first structure is **not** in that strict relation to the second, then advances to the next question. |
| **Done** | Shown when all queries are finished. Closes the Expert Query window and re-enables **Start Poset Construction** in the main window. |

The questions are shown one per “card” with a flip animation between questions and a small progress bar indicating how far through the fixed pair-iteration you are (it can jump when many pairs are implied by transitivity).

---

## Code structure (high level)

- **`Structure` (dataclass)**: Represents one anatomical structure with `name`, `com_vertical`, `com_lateral` (lateral right–left axis, patient-view), and `com_anteroposterior` (back–front axis).
- **`load_structures_from_json` / `save_poset_to_json` / `load_poset_from_json`**: I/O helpers for reading the structure list and saving/loading posets (vertical, lateral, and anteroposterior edges in a single JSON, with backward compatibility for older files that only stored vertical or “frontal” edges).
- **`PosetBuilder`**: Core algorithm; sorts structures by the chosen axis, iterates gap-wise over pairs, skips relations implied by transitivity, records answers, and returns the **transitive reduction** (Hasse edges).
- **`HasseDiagramView`**: `QGraphicsView` that renders the directed Hasse diagram; all axes use a layered layout (levels top–down) while the chosen axis only changes which strict relation the directed edges encode (above, left-of, in-front-of).
- **`PosetViewerWindow`**: Standalone viewer for saved posets with three tabs: **Vertical**, **Lateral**, and **Anteroposterior**, each with a structure list and an interactive Hasse diagram.
- **`DefinitionDialog`**: Intro + instructions + examples shown before querying; text and examples are adapted to the selected axis.
- **`QueryDialog`**: The questionnaire window: shows one pair at a time, collects Yes/No answers, supports Undo, drives autosave, shows progress, and ends with a thank-you message.
- **`MainWindow`**: Entry point UI for defining structures, choosing the axis for the current run, launching the questionnaire, and opening the viewer.

## Related Models
- [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- [VIBESegmentator](https://github.com/robert-graf/VIBESegmentator/tree/main)
- [Segment Anything Model 3 (SAM3)](https://ai.meta.com/research/sam3/)

## Key References
- KG-SAM (2025)
- Learning to Zoom with Anatomical Relations (NeurIPS 2025)
- 3D Spatial Priors (STIPPLE)
