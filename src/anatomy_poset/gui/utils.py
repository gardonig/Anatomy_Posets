import os
import sys
from pathlib import Path

from ..core.models import AXIS_MEDIOLATERAL, AXIS_VERTICAL


def _ensure_qt_platform_plugin_path() -> None:
    """
    macOS fix for:
      qt.qpa.plugin: Could not find the Qt platform plugin "cocoa" in ""

    Must set QT_QPA_PLATFORM_PLUGIN_PATH before any PySide6 import (Qt reads it at load).
    We derive the path from the current interpreter's site-packages so we don't import Qt.
    """
    if sys.platform != "darwin":
        return

    prefix = Path(sys.executable).resolve().parent.parent
    site_packages = (
        prefix
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    plugin_root = site_packages / "PySide6" / "Qt" / "plugins"
    platforms_dir = plugin_root / "platforms"

    # QT_PLUGIN_PATH makes Qt discover styles/platformthemes/etc.
    # QT_QPA_PLATFORM_PLUGIN_PATH specifically points at platform plugins (cocoa).
    if plugin_root.exists():
        os.environ.setdefault("QT_PLUGIN_PATH", str(plugin_root))
    if platforms_dir.exists():
        os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", str(platforms_dir))


# Known plural anatomical terms (last word of structure name). Default is singular
# so e.g. pelvis, lens, pons are correctly singular.
_PLURAL_LAST_WORDS: frozenset[str] = frozenset(
    {
        "lungs", "kidneys", "glands", "nodes", "ribs", "bones", "adrenals",
        "ovaries", "testes", "eyes", "ears", "nostrils", "hands", "feet",
        "thumbs", "fingers", "toes", "tonsils", "vertebrae", "muscles",
    }
)


def _is_plural_structure(name: str) -> bool:
    """
    Return True only if the structure name is known to be plural (main noun in
    whitelist). Otherwise singular (e.g. pelvis, femur, liver, lens).
    """
    if not name or not name.strip():
        return False
    parts = name.strip().split()
    if not parts:
        return False
    last = parts[-1].lower()
    return last in _PLURAL_LAST_WORDS


def _relation_verb(axis: str) -> str:
    if axis == AXIS_VERTICAL:
        return "strictly above"
    if axis == AXIS_MEDIOLATERAL:
        return "strictly to the left of"
    return "strictly in front of"