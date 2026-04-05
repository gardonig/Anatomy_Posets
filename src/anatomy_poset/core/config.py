from pathlib import Path

# This dynamically finds the project root directory (anatomy_poset)
# by going up 3 levels from src/anatomy_poset/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

# Define the standard paths used across the application
ASSETS_DIR = PROJECT_ROOT / "assets"
INPUT_DIR = PROJECT_ROOT / "data" / "structures"
OUTPUT_DIR = PROJECT_ROOT / "data" / "posets"