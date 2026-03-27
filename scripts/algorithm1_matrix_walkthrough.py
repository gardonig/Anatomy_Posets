#!/usr/bin/env python3
"""
Walk through Algorithm 1 (gap-based CoM strategy) for n=4 structures.

Run from repo root:
  PYTHONPATH=src python examples/algorithm1_matrix_walkthrough.py

Produces ``examples/algorithm1_walkthrough.png`` with the matrix M after each
human answer (tri-valued: -2 not asked, -1 no, 0 unsure, +1 yes / strictly above).

Structure indices follow :class:`anatomy_poset.core.matrix_builder.MatrixBuilder`:
  index 0 = highest vertical CoM (most superior), …, index n-1 = lowest.
  M[i][j] = +1 means “structure i is strictly above structure j” on the vertical axis.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running without installing the package
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO / "src") not in sys.path:
    sys.path.insert(0, str(_REPO / "src"))

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap

from anatomy_poset.core.matrix_builder import MatrixBuilder
from anatomy_poset.core.axis_models import AXIS_VERTICAL, Structure


def _print_theory_n4() -> None:
    print(
        """
=== Algorithm 1 (vertical axis): gap-based order over upper triangle ===

Structures are sorted by vertical CoM descending → rows/cols 0…3 (4 nodes).
Matrix convention: M[i][j] = +1 means “i strictly above j”. Lower triangle
(i > j) is fixed to -1 (impossible direction given CoM order).

The iterator uses gap g = 1, 2, …, n-1 and, for each g, pairs (i, j) with
  j = i + g,  i = 0 … n-1-g.

For n = 4 the *theoretical* order of *directed* upper-triangle pairs is:
  g=1: (0,1), (1,2), (2,3)
  g=2: (0,2), (1,3)
  g=3: (0,3)

MatrixBuilder.next_pair() skips a pair if:
  • M[i][j] is already filled (not -2), or
  • the relation is already implied by existing +1 edges (transitive closure).

So you may see *fewer* questions than 5 — e.g. three “+1” answers along the chain
can imply the rest. The demo below uses a mixed answer pattern so more steps
appear before completion.
"""
    )


def _matrix_to_array(M: list[list[int]]) -> np.ndarray:
    n = len(M)
    arr = np.full((n, n), -2, dtype=int)
    for i in range(n):
        for j in range(n):
            arr[i, j] = int(M[i][j])
    return arr


def _plot_snapshots(
    snapshots: list[tuple[str, list[list[int]]]],
    names: list[str],
    out_path: Path,
) -> None:
    levels = [-2, -1, 0, 1]
    colors = ["#f0f0f0", "#d73027", "#fc8d59", "#1a9850"]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-2.5, -1.5, -0.5, 0.5, 1.5], cmap.N)

    n_snaps = len(snapshots)
    ncols = min(3, n_snaps)
    nrows = (n_snaps + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
    if n_snaps == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for ax, (title, M) in zip(axes, snapshots):
        arr = _matrix_to_array(M)
        im = ax.imshow(arr, cmap=cmap, norm=norm, origin="upper")
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(names, fontsize=8)
        ax.set_xlabel("j (target)")
        ax.set_ylabel("i (source)")
        ax.set_title(title, fontsize=9)

    for k in range(len(snapshots), len(axes)):
        axes[k].set_visible(False)

    fig.suptitle(
        "Algorithm 1 — matrix M after each response (4 structures, vertical axis)",
        fontsize=11,
    )
    fig.colorbar(
        im,
        ax=list(axes[: len(snapshots)]),
        orientation="vertical",
        fraction=0.035,
        pad=0.02,
        ticks=levels,
    )
    cbar_ax = fig.axes[-1]
    if hasattr(cbar_ax, "set_yticklabels"):
        cbar_ax.set_yticklabels(["-2 not asked", "-1 no", "0 unsure", "+1 yes"])
    fig.subplots_adjust(top=0.92, right=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")


def run_mixed_demo() -> None:
    """
    After (0,1)=+1 and (1,2)=-1, transitivity does not close all cells; more pairs are asked.
    Remaining answers chosen as +1 where needed to finish.
    """
    structures = [
        Structure("A_top", 100.0, 0.0, 0.0),
        Structure("B", 75.0, 0.0, 0.0),
        Structure("C", 50.0, 0.0, 0.0),
        Structure("D_bot", 25.0, 0.0, 0.0),
    ]
    mb = MatrixBuilder(structures, axis=AXIS_VERTICAL)
    names = [s.name for s in mb.structures]

    snapshots: list[tuple[str, list[list[int]]]] = []
    snapshots.append(("0 — Initial (lower triangle = -1, upper = -2)", [row[:] for row in mb.M]))

    def answer(i: int, j: int) -> int:
        # Same rule as the trace we used above: force an interesting multi-step path
        if (i, j) == (0, 1):
            return 1
        if (i, j) == (1, 2):
            return -1
        return 1

    step = 0
    while True:
        p = mb.next_pair()
        if p is None:
            break
        i, j = p
        v = answer(i, j)
        mb.record_response_matrix(i, j, v)
        step += 1
        sym = {-1: "−1", 0: "0", 1: "+1"}.get(v, str(v))
        snapshots.append(
            (
                f"{step} — Q({i},{j}): “{names[i]} above {names[j]}?” → {sym}",
                [row[:] for row in mb.M],
            )
        )

    out = _REPO / "examples" / "algorithm1_walkthrough.png"
    _plot_snapshots(snapshots, names, out)


def main() -> None:
    _print_theory_n4()
    run_mixed_demo()
    print("\nOpen examples/algorithm1_walkthrough.png to see the matrices.")


if __name__ == "__main__":
    main()
