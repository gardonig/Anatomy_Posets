#!/usr/bin/env bash
# =============================================================================
# Set up MedSAM in its own virtual environment.
#
# Run from the project root:
#   bash scripts/setup_medsam.sh
#
# After this script:
#   1. Activate the venv:  source .medsam_venv/bin/activate
#   2. Download weights:   see message at the end of this script
#   3. Run inference:      python scripts/run_medsam.py --help
# =============================================================================
set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="$ROOT/.medsam_venv"
TOOLS="$ROOT/tools"
MEDSAM_DIR="$TOOLS/MedSAM"
WEIGHTS_DIR="$MEDSAM_DIR/work_dir/MedSAM"

echo ""
echo "=== MedSAM Setup ==="
echo "Project root : $ROOT"
echo "Venv         : $VENV"
echo "Repo         : $MEDSAM_DIR"
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
# For GPU replace with: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install torch torchvision torchaudio --quiet

# ---- 3. Other dependencies -------------------------------------------------
echo "[3/5] Installing dependencies..."
pip install nibabel numpy matplotlib scipy scikit-image tqdm opencv-python-headless --quiet

# ---- 4. Clone MedSAM -------------------------------------------------------
echo "[4/5] Cloning MedSAM..."
mkdir -p "$TOOLS"
if [ -d "$MEDSAM_DIR" ]; then
    echo "[skip] MedSAM already cloned."
else
    git clone https://github.com/bowang-lab/MedSAM.git "$MEDSAM_DIR"
fi

# ---- 5. Install MedSAM (provides segment_anything package) ----------------
echo "[5/5] Installing MedSAM package..."
pip install -e "$MEDSAM_DIR" --quiet

# ---- Weights placeholder ---------------------------------------------------
mkdir -p "$WEIGHTS_DIR"

echo ""
echo "============================================================"
echo "  MedSAM setup complete!"
echo "============================================================"
echo ""
echo "NEXT STEP — download model weights (medsam_vit_b.pth, ~375 MB):"
echo ""
echo "  Option A (wget):"
echo "    wget -O $WEIGHTS_DIR/medsam_vit_b.pth \\"
echo "      https://huggingface.co/bowang-lab/MedSAM/resolve/main/medsam_vit_b.pth"
echo ""
echo "  Option B (HuggingFace CLI — pip install huggingface_hub first):"
echo "    huggingface-cli download bowang-lab/MedSAM medsam_vit_b.pth \\"
echo "      --local-dir $WEIGHTS_DIR"
echo ""
echo "Then run inference:"
echo "  source .medsam_venv/bin/activate"
echo "  python scripts/run_medsam.py \\"
echo "    --subject s0011 \\"
echo "    --structure spleen \\"
echo "    --dataset data/imaging_datasets/Totalsegmentator_dataset_small_v201"
echo ""
