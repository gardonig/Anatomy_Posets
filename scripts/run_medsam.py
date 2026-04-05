"""
Run MedSAM inference on one or all subjects of the TotalSegmentator dataset.

MedSAM is a 2D model — for 3D CT volumes it runs slice-by-slice along the
axial axis.  A bounding box prompt is required per slice; by default it is
derived automatically from the ground-truth segmentation mask (ideal for
benchmarking).  You can also supply an explicit 3D bounding box.

Prerequisites
-------------
  source .medsam_venv/bin/activate            # activate the MedSAM venv
  # weights must be downloaded first (see scripts/setup_medsam.sh)

Usage examples
--------------
Single subject, single structure:
  python scripts/run_medsam.py \\
      --subject s0011 --structure spleen

All subjects, one structure:
  python scripts/run_medsam.py --structure spleen

All subjects, all 117 structures (slow!):
  python scripts/run_medsam.py --all_structures

Outputs land in:
  data/imaging_datasets/medsam_output_small_v201/<subject>/<structure>.nii.gz
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from skimage import transform
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Locate the MedSAM repo (handles both installed and editable installs)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR    = PROJECT_ROOT / "tools" / "MedSAM"
DEFAULT_CKPT = TOOLS_DIR / "work_dir" / "MedSAM" / "medsam_vit_b.pth"
DATASET_DIR  = PROJECT_ROOT / "data" / "imaging_datasets" / "Totalsegmentator_dataset_small_v201"
OUTPUT_BASE  = PROJECT_ROOT / "data" / "imaging_datasets" / "medsam_output_small_v201"

if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_nifti(path: Path):
    img  = nib.load(str(path))
    data = np.asarray(img.dataobj)
    return data, img.header, img.affine


def normalise_slice(slice_2d: np.ndarray) -> np.ndarray:
    """Normalise a 2-D CT slice to uint8 [0, 255], replicated to 3 channels."""
    lo, hi = slice_2d.min(), slice_2d.max()
    norm = (slice_2d - lo) / np.clip(hi - lo, 1e-8, None)  # [0, 1]
    rgb  = np.repeat(norm[:, :, None], 3, axis=-1)           # H W 3
    return (rgb * 255).astype(np.uint8)


def bbox_from_mask(mask_3d: np.ndarray):
    """Return (z_min, z_max, y_min, y_max, x_min, x_max) of nonzero voxels."""
    coords = np.argwhere(mask_3d > 0)
    if len(coords) == 0:
        return None
    z0, y0, x0 = coords.min(axis=0)
    z1, y1, x1 = coords.max(axis=0)
    return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)


def scale_bbox_to_1024(y0, y1, x0, x1, orig_h, orig_w, target=1024):
    """Scale a 2-D bounding box from original resolution to 1024×1024."""
    sy = target / orig_h
    sx = target / orig_w
    return np.array([x0 * sx, y0 * sy, x1 * sx, y1 * sy])  # [x0, y0, x1, y1]


# ---------------------------------------------------------------------------
# MedSAM inference (one 2-D slice)
# ---------------------------------------------------------------------------

@torch.no_grad()
def medsam_infer_slice(model, img_embed, box_1024, orig_h, orig_w):
    """Run MedSAM decoder for a single slice embedding and return binary mask."""
    box_torch = torch.as_tensor(box_1024[None, None, :], dtype=torch.float,
                                device=img_embed.device)  # (1, 1, 4)

    sparse_emb, dense_emb = model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_emb,
        dense_prompt_embeddings=dense_emb,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)          # (1, 1, 256, 256)
    pred = F.interpolate(low_res_pred, size=(orig_h, orig_w),
                         mode="bilinear", align_corners=False)
    return (pred.squeeze().cpu().numpy() > 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# Per-subject, per-structure inference
# ---------------------------------------------------------------------------

def run_inference(subject: str, structure: str, model, device: str,
                  target_size: int = 1024):
    subject_dir = DATASET_DIR / subject
    ct_path     = subject_dir / "ct.nii.gz"
    gt_path     = subject_dir / "segmentations" / f"{structure}.nii.gz"
    out_path    = OUTPUT_BASE / subject / f"{structure}.nii.gz"

    if not ct_path.exists():
        print(f"  [skip] CT not found: {ct_path}")
        return
    if not gt_path.exists():
        print(f"  [skip] GT not found for '{structure}' in {subject}")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ct, ct_hdr, ct_affine = load_nifti(ct_path)
    gt, _, _              = load_nifti(gt_path)
    gt_bin = (gt > 0).astype(np.uint8)

    bbox = bbox_from_mask(gt_bin)
    if bbox is None:
        print(f"  [skip] GT mask empty for '{structure}' in {subject}")
        # Save empty mask so the comparison script can still load it
        nib.save(nib.Nifti1Image(np.zeros_like(ct, dtype=np.uint8), ct_affine, ct_hdr), str(out_path))
        return

    z0, z1, y0, y1, x0, x1 = bbox
    # Small margin around bbox
    margin = 5
    y0m = max(0, y0 - margin)
    y1m = min(ct.shape[1] - 1, y1 + margin)
    x0m = max(0, x0 - margin)
    x1m = min(ct.shape[0] - 1, x1 + margin)

    orig_h, orig_w = ct.shape[1], ct.shape[0]

    pred_vol = np.zeros_like(ct, dtype=np.uint8)

    for z in tqdm(range(z0, z1 + 1), desc=f"{subject}/{structure}", leave=False):
        slice_2d = ct[:, :, z]          # shape (X, Y) = (orig_w, orig_h)
        # MedSAM expects H x W — transpose so rows=Y, cols=X
        slice_hw = slice_2d.T           # (orig_h, orig_w)

        # Preprocess
        img_uint8 = normalise_slice(slice_hw)                      # (H, W, 3)
        img_1024  = transform.resize(img_uint8,
                                     (target_size, target_size, 3),
                                     order=3, preserve_range=True,
                                     anti_aliasing=True).astype(np.uint8)

        # Convert to tensor  (1, 3, 1024, 1024)
        img_tensor = torch.as_tensor(img_1024, dtype=torch.float32,
                                     device=device).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Get image embedding
        img_embed = model.image_encoder(img_tensor)  # (1, 256, 64, 64)

        # Scale bbox
        box = scale_bbox_to_1024(y0m, y1m, x0m, x1m, orig_h, orig_w, target_size)

        # Infer
        mask_hw = medsam_infer_slice(model, img_embed, box, orig_h, orig_w)

        # Transpose back to (orig_w, orig_h) = (X, Y)
        pred_vol[:, :, z] = mask_hw.T

    nib.save(nib.Nifti1Image(pred_vol, ct_affine, ct_hdr), str(out_path))
    print(f"  Saved: {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="MedSAM 3D CT inference (slice-by-slice)")
    parser.add_argument("--subject",       default=None, help="Subject ID (e.g. s0011). Omit for all subjects.")
    parser.add_argument("--structure",     default=None, help="Structure name (e.g. spleen). Omit with --all_structures.")
    parser.add_argument("--all_structures", action="store_true", help="Run inference for every structure with a GT mask.")
    parser.add_argument("--checkpoint",    default=str(DEFAULT_CKPT), type=Path, help="Path to medsam_vit_b.pth")
    parser.add_argument("--device",        default="auto", help="cpu | cuda | mps | auto")
    parser.add_argument("--target_size",   default=1024, type=int, help="Image size fed to MedSAM (default 1024)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ---- device ----
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"Using device: {device}")

    # ---- load model ----
    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(
            f"\nCheckpoint not found: {ckpt}\n"
            "Run  scripts/setup_medsam.sh  and download the weights first.\n"
            "See the printed instructions at the end of that script."
        )

    try:
        from segment_anything import sam_model_registry
    except ImportError:
        sys.exit(
            "\nCannot import segment_anything. Is the MedSAM venv active?\n"
            "  source .medsam_venv/bin/activate"
        )

    print("Loading MedSAM model...")
    model = sam_model_registry["vit_b"](checkpoint=str(ckpt))
    model = model.to(device)
    model.eval()

    # ---- subjects ----
    if args.subject:
        subjects = [args.subject]
    else:
        subjects = sorted(p.name for p in DATASET_DIR.iterdir() if p.is_dir() and p.name.startswith("s"))

    # ---- structures ----
    if args.all_structures:
        first_subject = subjects[0]
        seg_dir = DATASET_DIR / first_subject / "segmentations"
        structures = sorted(p.stem.replace(".nii", "") for p in seg_dir.glob("*.nii.gz"))
    elif args.structure:
        structures = [args.structure]
    else:
        sys.exit("Specify --structure <name> or --all_structures")

    print(f"Subjects   : {subjects}")
    print(f"Structures : {structures}")
    print()

    for subject in subjects:
        for structure in structures:
            run_inference(subject, structure, model, device, args.target_size)

    print("\nDone.")


if __name__ == "__main__":
    main()
