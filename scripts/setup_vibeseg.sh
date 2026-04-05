#!/usr/bin/env bash
# =============================================================================
# Set up VIBESegmentator in its own virtual environment.
#
# !! IMPORTANT !!
# VIBESegmentator is designed for VIBE MRI images, NOT for CT scans.
# The Totalsegmentator_dataset_small_v201 dataset contains CT data.
# Running VIBESegmentator on CT will NOT produce valid segmentations.
# This setup is provided so you can use VIBESegmentator on MRI data
# and include those results in the comparison pipeline.
#
# Run from the project root:
#   bash scripts/setup_vibeseg.sh
#
# After this script:
#   1. Activate the venv:  source .vibeseg_venv/bin/activate
#   2. Model weights are downloaded automatically on first run
#   3. Run inference:      python tools/VIBESegmentator/run_VIBESegmentator.py --help
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.vibeseg_venv"
TOOLS="$ROOT/tools"
VIBESEG_DIR="$TOOLS/VIBESegmentator"

echo ""
echo "=== VIBESegmentator Setup ==="
echo ""
echo "  !! WARNING: VIBESegmentator targets VIBE MRI — not CT !!"
echo "  !! Use on MRI data only for valid comparisons.          !!"
echo ""
echo "Project root : $ROOT"
echo "Venv         : $VENV"
echo "Repo         : $VIBESEG_DIR"
echo ""

# ---- 1. Virtual environment ------------------------------------------------
if [ -d "$VENV" ]; then
    echo "[skip] Venv already exists: $VENV"
else
    echo "[1/5] Creating venv with python3.11..."
    python3.11 -m venv "$VENV"
fi

source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# ---- 2. PyTorch ------------------------------------------------------------
echo "[2/5] Installing PyTorch (CPU build)..."
# For Apple Silicon MPS: pip install torch torchvision torchaudio
# For Nvidia GPU:        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision torchaudio --quiet

# ---- 3. VIBESegmentator dependencies ---------------------------------------
echo "[3/5] Installing VIBESegmentator dependencies..."
pip install \
    "TPTBox>=1.6" \
    "ruamel.yaml>=0.18.6" \
    "configargparse>=1.7" \
    "nnunetv2>=2.4.2" \
    nibabel numpy matplotlib scipy tqdm \
    --quiet

# ---- 4. Clone VIBESegmentator ----------------------------------------------
echo "[4/5] Cloning VIBESegmentator..."
mkdir -p "$TOOLS"
if [ -d "$VIBESEG_DIR" ]; then
    echo "[skip] VIBESegmentator already cloned."
else
    git clone https://github.com/robert-graf/VIBESegmentator.git "$VIBESEG_DIR"
fi

# ---- 5. Set nnUNet env vars ------------------------------------------------
echo "[5/5] Configuring nnUNet paths..."
NNUNET_RAW="$VIBESEG_DIR/nnUNet/nnUNet_raw"
NNUNET_PREPROCESSED="$VIBESEG_DIR/nnUNet/nnUNet_preprocessed"
NNUNET_RESULTS="$VIBESEG_DIR/nnUNet/nnUNet_results"
mkdir -p "$NNUNET_RAW" "$NNUNET_PREPROCESSED" "$NNUNET_RESULTS"

ACTIVATE_SCRIPT="$VENV/bin/activate"
if ! grep -q "nnUNet_results" "$ACTIVATE_SCRIPT"; then
    echo "" >> "$ACTIVATE_SCRIPT"
    echo "# nnUNet paths for VIBESegmentator" >> "$ACTIVATE_SCRIPT"
    echo "export nnUNet_raw=\"$NNUNET_RAW\"" >> "$ACTIVATE_SCRIPT"
    echo "export nnUNet_preprocessed=\"$NNUNET_PREPROCESSED\"" >> "$ACTIVATE_SCRIPT"
    echo "export nnUNet_results=\"$NNUNET_RESULTS\"" >> "$ACTIVATE_SCRIPT"
    echo "Added nnUNet env vars to $ACTIVATE_SCRIPT"
fi

echo ""
echo "============================================================"
echo "  VIBESegmentator setup complete!"
echo "============================================================"
echo ""
echo "Usage on VIBE MRI data:"
echo "  source .vibeseg_venv/bin/activate"
echo "  python tools/VIBESegmentator/run_VIBESegmentator.py \\"
echo "    --img  path/to/vibe.nii.gz \\"
echo "    --out_path path/to/output/ \\"
echo "    --device cpu"
echo ""
echo "  (Model weights download automatically on first run ~1-2 GB)"
echo ""
echo "  Supported model IDs: 100, 99, 278, 282, 511, 512, 520"
echo "  Add --model_id 100 (default) or another ID to select variant."
echo ""
