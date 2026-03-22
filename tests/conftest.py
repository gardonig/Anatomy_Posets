import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root (folder containing src/)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# So test modules can `from helpers import ...` (shared fixtures in tests/helpers.py).
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))