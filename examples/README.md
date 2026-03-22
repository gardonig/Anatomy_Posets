# Examples

## Algorithm 1 matrix walkthrough (`algorithm1_matrix_walkthrough.py`)

Explains the **gap-based** question order for **n = 4** structures and saves a figure with the tri-valued matrix **M** after each simulated answer.

Requires **matplotlib** and **numpy** (same as the GUI).

```bash
cd /path/to/Anatomy_Posets
PYTHONPATH=src python examples/algorithm1_matrix_walkthrough.py
```

Output: `examples/algorithm1_walkthrough.png`

**Interpretation:**

- Rows/columns are structures sorted by **vertical CoM descending** (index 0 = most superior).
- **M[i][j] = +1** means “structure i is strictly above j”.
- **Lower triangle** (i > j) is fixed to **−1** (geometrically impossible for “strictly above” on this axis).
- **Algorithm 1** visits pairs with **gap** g = 1, 2, …, n−1: pairs **(i, i+g)** for **i = 0 … n−1−g**.
- After each answer, **propagation** (`_propagate`) may fill more cells (transitive +1, inverse −1, etc.), so **next_pair** may skip pairs that are already implied.

The script uses a **mixed** answer pattern (e.g. +1 then −1 on the chain) so more than three steps appear; answering **+1** on every adjacent pair along the chain only needs **three** questions before the rest is implied.
