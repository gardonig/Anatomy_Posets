"""
Compare segmentation outputs from multiple tools on the same CT dataset.

Supported tools
---------------
  totalseg   TotalSegmentator (already installed in .totalseg_venv)
  medsam     MedSAM           (set up with scripts/setup_medsam.sh)

Note: VIBESegmentator targets VIBE MRI, not CT.  If you have MRI data
and VIBESegmentator outputs, place them under:
  data/imaging_datasets/vibeseg_output_small_v201/<subject>/<structure>.nii.gz
and add "vibeseg" to --segmenters.

Usage examples
--------------
Compare all available tools for spleen on subject s0011:
  python scripts/compare_segmenters.py \\
      --subject s0011 --structure spleen

Compare on all subjects, save metrics CSV:
  python scripts/compare_segmenters.py \\
      --structure spleen --all_subjects --csv results/spleen_comparison.csv

Compare multiple structures:
  python scripts/compare_segmenters.py \\
      --subject s0011 --structure spleen liver aorta

Skip visualization (metrics only):
  python scripts/compare_segmenters.py \\
      --subject s0011 --structure spleen --no_plot

Dependencies (all in .totalseg_venv or .venv):
  nibabel, numpy, matplotlib, scipy
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR  = PROJECT_ROOT / "data" / "imaging_datasets" / "Totalsegmentator_dataset_small_v201"

OUTPUT_DIRS = {
    "totalseg": PROJECT_ROOT / "data" / "imaging_datasets" / "totalseg_output_small_v201",
    "medsam":   PROJECT_ROOT / "data" / "imaging_datasets" / "medsam_output_small_v201",
    "vibeseg":  PROJECT_ROOT / "data" / "imaging_datasets" / "vibeseg_output_small_v201",
}

COLORS = {
    "totalseg": (1.0, 0.30, 0.30),   # red
    "medsam":   (0.30, 0.70, 1.00),  # blue
    "vibeseg":  (0.30, 1.00, 0.50),  # green
    "gt":       (1.00, 1.00, 0.30),  # yellow
}

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path):
    img  = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data, img.affine


def find_seg_path(tool: str, subject: str, structure: str) -> Path | None:
    base = OUTPUT_DIRS[tool] / subject / f"{structure}.nii.gz"
    return base if base.exists() else None


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    denom = p.sum() + g.sum()
    return (2.0 * (p & g).sum() / denom) if denom > 0 else 1.0


def jaccard(pred, gt):
    p, g  = pred.astype(bool), gt.astype(bool)
    union = (p | g).sum()
    return ((p & g).sum() / union) if union > 0 else 1.0


def precision_recall(pred, gt):
    p, g = pred.astype(bool), gt.astype(bool)
    tp   = (p & g).sum()
    prec = (tp / p.sum()) if p.sum() > 0 else 0.0
    rec  = (tp / g.sum()) if g.sum() > 0 else 0.0
    return float(prec), float(rec)


def surface_dice(pred, gt, tol: int = 1):
    def surface(mask):
        return mask.astype(bool) & ~binary_erosion(mask.astype(bool))

    ps, gs = surface(pred), surface(gt)
    pd = binary_dilation(ps, iterations=tol)
    gd = binary_dilation(gs, iterations=tol)
    denom = ps.sum() + gs.sum()
    return float((ps & gd).sum() + (gs & pd).sum()) / denom if denom > 0 else 1.0


def volume_ml(mask, affine):
    vox_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return mask.astype(bool).sum() * vox_mm3 / 1000.0


def compute_metrics(pred, gt, affine) -> dict:
    prec, rec = precision_recall(pred, gt)
    return {
        "dice":      round(dice(pred, gt), 4),
        "jaccard":   round(jaccard(pred, gt), 4),
        "precision": round(prec, 4),
        "recall":    round(rec, 4),
        "surf_dice": round(surface_dice(pred, gt), 4),
        "vol_pred_ml": round(volume_ml(pred, affine), 2),
        "vol_gt_ml":   round(volume_ml(gt, affine), 2),
    }


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def window_ct(ct, wl=60, ww=400):
    lo, hi = wl - ww / 2, wl + ww / 2
    return (np.clip(ct, lo, hi) - lo) / (hi - lo)


def centre_slices(mask):
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return tuple(d // 2 for d in mask.shape[::-1])
    cx, cy, cz = coords.mean(axis=0).astype(int)
    return int(cz), int(cy), int(cx)


def overlay(ax, ct_slice, seg_slice, title, color, alpha=0.45):
    ax.imshow(ct_slice.T, cmap="gray", origin="lower", interpolation="nearest")
    if seg_slice.any():
        rgba = np.zeros((*seg_slice.T.shape, 4))
        rgba[..., :3] = color
        rgba[..., 3]  = seg_slice.T.astype(float) * alpha
        ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=8, color="white", pad=2)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Single-subject, single-structure comparison
# ---------------------------------------------------------------------------

def compare_one(subject: str, structure: str, tools: list[str],
                wl=60, ww=400, save_path: Path | None = None,
                no_plot=False) -> list[dict]:
    ct_path = DATASET_DIR / subject / "ct.nii.gz"
    gt_path = DATASET_DIR / subject / "segmentations" / f"{structure}.nii.gz"

    if not ct_path.exists():
        print(f"  [skip] CT missing: {ct_path}")
        return []
    if not gt_path.exists():
        print(f"  [skip] GT missing for '{structure}' in {subject}")
        return []

    ct, ct_affine = load_nifti(ct_path)
    gt_raw, _     = load_nifti(gt_path)
    gt            = (gt_raw > 0).astype(np.uint8)

    ct_w = window_ct(ct, wl, ww)
    ax_sl, cor_sl, sag_sl = centre_slices(gt)

    rows = []

    # ---- compute metrics for each tool ----
    available = {}
    for tool in tools:
        seg_path = find_seg_path(tool, subject, structure)
        if seg_path is None:
            print(f"  [skip] No output for {tool}/{subject}/{structure}")
            continue
        seg_raw, seg_affine = load_nifti(seg_path)
        seg = (seg_raw > 0).astype(np.uint8)
        if seg.shape != gt.shape:
            print(f"  [warn] Shape mismatch {tool}: {seg.shape} vs GT {gt.shape} — skipping metrics")
            available[tool] = (seg, seg_affine, None)
            continue
        metrics = compute_metrics(seg, gt, ct_affine)
        available[tool] = (seg, seg_affine, metrics)
        row = {"subject": subject, "structure": structure, "tool": tool, **metrics}
        rows.append(row)

        print(f"  {tool:12s}  Dice={metrics['dice']:.4f}  IoU={metrics['jaccard']:.4f}"
              f"  Prec={metrics['precision']:.4f}  Rec={metrics['recall']:.4f}"
              f"  SurfDice={metrics['surf_dice']:.4f}"
              f"  Vol={metrics['vol_pred_ml']:.1f}ml (GT {metrics['vol_gt_ml']:.1f}ml)")

    if no_plot or not available:
        return rows

    # ---- build figure ----
    # rows: GT first, then each tool
    n_rows = 1 + len(available)
    fig = plt.figure(figsize=(15, 4.5 * n_rows))
    fig.patch.set_facecolor("#1a1a1a")

    def _slices(mask):
        return [mask[:, :, ax_sl], mask[:, cor_sl, :], mask[sag_sl, :, :]]

    ct_slices = [ct_w[:, :, ax_sl], ct_w[:, cor_sl, :], ct_w[sag_sl, :, :]]

    for row_idx, (label, mask, color) in enumerate(
        [("Ground Truth", gt, COLORS["gt"])]
        + [(tool, seg, COLORS[tool]) for tool, (seg, _, _) in available.items()]
    ):
        top    = 1 - row_idx / n_rows
        bottom = 1 - (row_idx + 1) / n_rows
        gs = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.02, right=0.98,
                               top=top - 0.01, bottom=bottom + 0.02,
                               wspace=0.04)
        view_labels = [f"Axial z={ax_sl}", f"Coronal y={cor_sl}", f"Sagittal x={sag_sl}"]
        for col in range(3):
            ax = fig.add_subplot(gs[0, col])
            title = f"{label}  —  {view_labels[col]}"
            if col == 0:
                m = available.get(label.lower().replace(" ", "").replace("truth", ""), (None, None, None))[2]
                if m:
                    title += f"\nDice={m['dice']:.3f}  IoU={m['jaccard']:.3f}"
            overlay(ax, ct_slices[col], _slices(mask)[col], title, color)

    suptitle = f"Segmentation comparison — {subject} / {structure}"
    fig.suptitle(suptitle, color="white", fontsize=11, y=0.995)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=140, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  Saved figure: {save_path}")
        plt.close(fig)
    else:
        plt.tight_layout()
        plt.show()

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Compare medical image segmentation tools")
    parser.add_argument("--subject",     default=None,
                        help="Subject ID (e.g. s0011). Omit with --all_subjects.")
    parser.add_argument("--all_subjects", action="store_true",
                        help="Run comparison on all subjects in the dataset.")
    parser.add_argument("--structure",   nargs="+", required=True,
                        help="One or more structure names (e.g. spleen liver aorta)")
    parser.add_argument("--segmenters",  nargs="+", default=list(OUTPUT_DIRS.keys()),
                        choices=list(OUTPUT_DIRS.keys()),
                        help="Which tools to include (default: all)")
    parser.add_argument("--csv",         default=None, type=Path,
                        help="Save metrics table to this CSV file")
    parser.add_argument("--figures_dir", default=None, type=Path,
                        help="Save one PNG per (subject, structure) to this directory")
    parser.add_argument("--no_plot",     action="store_true",
                        help="Skip visualisation (metrics only)")
    parser.add_argument("--wl",          default=60,  type=int, help="CT window level")
    parser.add_argument("--ww",          default=400, type=int, help="CT window width")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.all_subjects:
        subjects = sorted(p.name for p in DATASET_DIR.iterdir()
                          if p.is_dir() and not p.name.startswith("."))
    elif args.subject:
        subjects = [args.subject]
    else:
        sys.exit("Specify --subject <id> or --all_subjects")

    all_rows = []

    for subject in subjects:
        for structure in args.structure:
            print(f"\n{'='*60}")
            print(f"  {subject}  /  {structure}")
            print(f"{'='*60}")

            save_fig = None
            if args.figures_dir:
                save_fig = args.figures_dir / f"{subject}_{structure}.png"

            rows = compare_one(
                subject=subject,
                structure=structure,
                tools=args.segmenters,
                wl=args.wl,
                ww=args.ww,
                save_path=save_fig,
                no_plot=args.no_plot or bool(args.figures_dir),
            )
            all_rows.extend(rows)

    if not all_rows:
        print("\nNo metrics computed (no matching segmentation outputs found).")
        return

    # ---- print summary table ----
    print(f"\n{'='*90}")
    print(f"  SUMMARY TABLE")
    print(f"{'='*90}")
    header = f"{'subject':<10} {'structure':<28} {'tool':<12} {'dice':>6} {'jaccard':>7} {'prec':>6} {'rec':>6} {'surf':>6} {'vol_p':>7} {'vol_g':>7}"
    print(header)
    print("-" * 90)
    for r in all_rows:
        print(
            f"{r['subject']:<10} {r['structure']:<28} {r['tool']:<12}"
            f" {r['dice']:>6.4f} {r['jaccard']:>7.4f}"
            f" {r['precision']:>6.4f} {r['recall']:>6.4f} {r['surf_dice']:>6.4f}"
            f" {r['vol_pred_ml']:>7.1f} {r['vol_gt_ml']:>7.1f}"
        )

    # ---- CSV ----
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(all_rows[0].keys())
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nMetrics saved to: {args.csv}")

    print()


if __name__ == "__main__":
    main()
