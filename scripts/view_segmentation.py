"""
Visualise a TotalSegmentator output and compute segmentation metrics.

Usage (activate your totalseg venv first):

    python scripts/view_segmentation.py \
        --ct       path/to/ct.nii.gz \
        --seg      path/to/totalseg_output/spleen.nii.gz \
        --gt       path/to/labelsTr/spleen_001.nii.gz   # optional, enables Dice etc.
        --label    1                                      # label index in gt mask (default 1)

If --seg is a *directory* (TotalSegmentator output folder with one file per structure),
pass the structure name with --structure and the script picks the right file:

    python scripts/view_segmentation.py \
        --ct        path/to/ct.nii.gz \
        --seg       path/to/totalseg_output/ \
        --structure spleen \
        --gt        path/to/labelsTr/spleen_001.nii.gz

Dependencies (all included in the totalseg venv):
    nibabel, numpy, matplotlib, scipy
"""

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import nibabel as nib
import numpy as np
from scipy.ndimage import binary_erosion


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path) -> tuple[np.ndarray, object]:
    img = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data, img.affine


def resolve_seg_path(seg_path: Path, structure: str | None) -> Path:
    if seg_path.is_dir():
        if structure is None:
            raise ValueError(
                "--seg is a directory but --structure was not given. "
                "Pass e.g. --structure spleen"
            )
        candidate = seg_path / f"{structure}.nii.gz"
        if not candidate.exists():
            # list what is there
            files = sorted(seg_path.glob("*.nii.gz"))
            names = [f.stem.replace(".nii", "") for f in files]
            raise FileNotFoundError(
                f"No file '{candidate.name}' in {seg_path}.\n"
                f"Available structures: {names}"
            )
        return candidate
    return seg_path


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    intersection = (pred_b & gt_b).sum()
    denom = pred_b.sum() + gt_b.sum()
    if denom == 0:
        return 1.0 if pred_b.sum() == 0 else 0.0
    return 2.0 * intersection / denom


def jaccard_score(pred: np.ndarray, gt: np.ndarray) -> float:
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    inter = (pred_b & gt_b).sum()
    union = (pred_b | gt_b).sum()
    return inter / union if union > 0 else 1.0


def precision_recall(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    pred_b = pred.astype(bool)
    gt_b   = gt.astype(bool)
    tp = (pred_b & gt_b).sum()
    prec = tp / pred_b.sum() if pred_b.sum() > 0 else 0.0
    rec  = tp / gt_b.sum()   if gt_b.sum()   > 0 else 0.0
    return prec, rec


def volume_ml(mask: np.ndarray, affine: np.ndarray) -> float:
    vox_vol_mm3 = abs(float(np.linalg.det(affine[:3, :3])))
    return mask.astype(bool).sum() * vox_vol_mm3 / 1000.0


def surface_dice(pred: np.ndarray, gt: np.ndarray, tolerance_vox: int = 1) -> float:
    """Approximate surface Dice (boundary Dice) at given voxel tolerance."""
    def surface(mask):
        eroded = binary_erosion(mask.astype(bool))
        return mask.astype(bool) & ~eroded

    pred_s = surface(pred)
    gt_s   = surface(gt)
    # dilate surfaces by tolerance using binary_erosion's inverse
    from scipy.ndimage import binary_dilation
    pred_dilated = binary_dilation(pred_s, iterations=tolerance_vox)
    gt_dilated   = binary_dilation(gt_s,   iterations=tolerance_vox)

    tp_pred = (pred_s & gt_dilated).sum()
    tp_gt   = (gt_s   & pred_dilated).sum()
    denom   = pred_s.sum() + gt_s.sum()
    return (tp_pred + tp_gt) / denom if denom > 0 else 1.0


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def window_ct(ct: np.ndarray, wl: int = 60, ww: int = 400) -> np.ndarray:
    """Apply CT window/level and return float in [0, 1]."""
    lo = wl - ww / 2
    hi = wl + ww / 2
    ct_w = np.clip(ct, lo, hi)
    return (ct_w - lo) / (hi - lo)


def get_centre_slices(mask: np.ndarray) -> tuple[int, int, int]:
    """Return axial/coronal/sagittal slice indices at the mask's centre of mass."""
    coords = np.argwhere(mask)
    if len(coords) == 0:
        s = [d // 2 for d in mask.shape]
        return s[2], s[1], s[0]
    cx, cy, cz = coords.mean(axis=0).astype(int)
    return cz, cy, cx   # axial=z, coronal=y, sagittal=x


def overlay_slice(ax, ct_slice, seg_slice, title, cmap_ct="gray", alpha=0.4, color=(1, 0.2, 0.2)):
    ax.imshow(ct_slice.T, cmap=cmap_ct, origin="lower", interpolation="nearest")
    if seg_slice.any():
        rgba = np.zeros((*seg_slice.T.shape, 4))
        rgba[..., 0] = color[0]
        rgba[..., 1] = color[1]
        rgba[..., 2] = color[2]
        rgba[..., 3] = seg_slice.T.astype(float) * alpha
        ax.imshow(rgba, origin="lower", interpolation="nearest")
    ax.set_title(title, fontsize=10)
    ax.axis("off")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Visualise TotalSegmentator output + metrics")
    parser.add_argument("--ct",        required=True,  type=Path, help="Input CT NIfTI file")
    parser.add_argument("--seg",       required=True,  type=Path, help="Segmentation NIfTI file or output directory")
    parser.add_argument("--gt",        default=None,   type=Path, help="Ground-truth NIfTI file (optional)")
    parser.add_argument("--structure", default=None,   type=str,  help="Structure name if --seg is a directory (e.g. spleen)")
    parser.add_argument("--label",     default=1,      type=int,  help="Label index in GT mask (default 1)")
    parser.add_argument("--wl",        default=60,     type=int,  help="CT window level (default 60 = soft tissue)")
    parser.add_argument("--ww",        default=400,    type=int,  help="CT window width (default 400)")
    parser.add_argument("--out",       default=None,   type=Path, help="Save figure to this path instead of showing it")
    args = parser.parse_args()

    # --- load data ---
    print("Loading CT...")
    ct, ct_affine = load_nifti(args.ct)

    seg_file = resolve_seg_path(args.seg, args.structure)
    print(f"Loading segmentation: {seg_file}")
    seg_raw, _ = load_nifti(seg_file)
    seg = (seg_raw > 0).astype(np.uint8)

    gt = None
    if args.gt is not None:
        print("Loading ground truth...")
        gt_raw, gt_affine = load_nifti(args.gt)
        gt = (gt_raw == args.label).astype(np.uint8)

    # --- metrics ---
    print("\n" + "="*50)
    print("SEGMENTATION METRICS")
    print("="*50)
    print(f"  Predicted volume : {volume_ml(seg, ct_affine):.1f} ml")

    if gt is not None:
        if gt.shape != seg.shape:
            print(f"\n  WARNING: seg shape {seg.shape} != gt shape {gt.shape}.")
            print("  Metrics skipped — shapes must match (resample if needed).")
        else:
            print(f"  GT volume        : {volume_ml(gt, gt_affine):.1f} ml")
            d = dice_score(seg, gt)
            j = jaccard_score(seg, gt)
            prec, rec = precision_recall(seg, gt)
            sd = surface_dice(seg, gt, tolerance_vox=1)
            print(f"\n  Dice score       : {d:.4f}  (1.0 = perfect overlap)")
            print(f"  Jaccard (IoU)    : {j:.4f}")
            print(f"  Precision        : {prec:.4f}  (what fraction of pred is correct)")
            print(f"  Recall           : {rec:.4f}  (what fraction of GT is covered)")
            print(f"  Surface Dice     : {sd:.4f}  (boundary agreement, tol=1 vox)")
    print("="*50 + "\n")

    # --- visualisation ---
    ct_w = window_ct(ct, wl=args.wl, ww=args.ww)
    ax_sl, cor_sl, sag_sl = get_centre_slices(seg)

    has_gt = gt is not None and gt.shape == seg.shape
    n_rows = 2 if has_gt else 1
    fig = plt.figure(figsize=(14, 5 * n_rows))
    fig.patch.set_facecolor("#1a1a1a")

    structure_name = args.structure or seg_file.stem.replace(".nii", "")

    def make_row(row_idx, mask, label_text, color):
        gs = gridspec.GridSpec(n_rows, 3, figure=fig,
                               left=0.03, right=0.97,
                               top=1 - row_idx / n_rows + 0.01,
                               bottom=1 - (row_idx + 1) / n_rows + 0.03,
                               wspace=0.05)
        titles = [
            f"{label_text} — Axial (z={ax_sl})",
            f"{label_text} — Coronal (y={cor_sl})",
            f"{label_text} — Sagittal (x={sag_sl})",
        ]
        slices_ct  = [ct_w[:, :, ax_sl], ct_w[:, cor_sl, :], ct_w[sag_sl, :, :]]
        slices_seg = [mask[:, :, ax_sl], mask[:, cor_sl, :], mask[sag_sl, :, :]]

        for col in range(3):
            ax = fig.add_subplot(gs[0, col])
            overlay_slice(ax, slices_ct[col], slices_seg[col], titles[col], color=color)

    make_row(0, seg, f"Prediction: {structure_name}", color=(1.0, 0.25, 0.25))
    if has_gt:
        make_row(1, gt, "Ground truth", color=(0.25, 1.0, 0.25))

    # metrics text box
    metrics_text = f"Vol pred: {volume_ml(seg, ct_affine):.1f} ml"
    if has_gt and seg.shape == gt.shape:
        metrics_text += (
            f"  |  Vol GT: {volume_ml(gt, gt_affine):.1f} ml"
            f"  |  Dice: {dice_score(seg, gt):.3f}"
            f"  |  IoU: {jaccard_score(seg, gt):.3f}"
            f"  |  Prec: {precision_recall(seg, gt)[0]:.3f}"
            f"  |  Rec: {precision_recall(seg, gt)[1]:.3f}"
            f"  |  SurfDice: {surface_dice(seg, gt):.3f}"
        )
    fig.text(0.5, 0.01, metrics_text, ha="center", va="bottom",
             color="white", fontsize=9,
             bbox=dict(facecolor="#333333", edgecolor="none", boxstyle="round,pad=0.3"))

    if args.out:
        plt.savefig(args.out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Saved figure to {args.out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
