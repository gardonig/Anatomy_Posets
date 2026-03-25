import sys
import argparse
from pathlib import Path
from typing import Optional

# To ensure Python can find your new package structure even if you haven't 
# formally "installed" it via pip, we add the 'src' directory to the path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from anatomy_poset.gui.utils import _ensure_qt_platform_plugin_path

# Set Qt plugin path on macOS before any Qt import (Qt reads it when loaded)
_ensure_qt_platform_plugin_path()

from PySide6.QtWidgets import QApplication
from anatomy_poset.gui.main_window import MainWindow
from anatomy_poset.core.builder import MatrixBuilder
from anatomy_poset.core.models import AXIS_VERTICAL
from anatomy_poset.gui.query_dialog import QueryDialog


def main() -> None:
    """
    Optional usage:
      python main.py path/to/structures.json
    """

    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "input_path",
        nargs="?",
        default=None,
        help="Optional JSON file with anatomical structures (or an existing poset file).",
    )
    parser.add_argument(
        "-q",
        "--query",
        action="store_true",
        help="Skip straight to the vertical-axis query window.",
    )
    args = parser.parse_args()
    input_path: Optional[str] = args.input_path

    app = QApplication(sys.argv)
    window = MainWindow(input_path=input_path)
    window.show()
    if args.query:
        # Build a vertical-axis matrix-based builder from the structures currently in the table.
        # This skips the Start/Instructions/Definition flow and opens the query window directly.
        structures = window._collect_structures()
        if structures is not None:
            builder = MatrixBuilder(structures, axis=AXIS_VERTICAL)
            # Reuse the current autosave path selection behavior.
            # (User will be asked where to save when starting normally; here we just use current autosave path.)
            autosave = window._autosave_path
            if autosave is None:
                autosave = window._builtposet_output_path(Path(input_path) if input_path else Path("poset_autosave.json"))
                window._autosave_path = autosave
            qd = QueryDialog(
                builder,
                autosave,
                axis=AXIS_VERTICAL,
                save_callback=window._on_poset_autosave,
            )
            qd.showMaximized()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()