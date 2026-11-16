"""Where bundled resources live (presets, fonts, models).

Defaults to the repository layout (engine/mog/... -> repo root). Set MOG_ROOT when the
package is installed somewhere else (e.g. in the Docker image).
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("MOG_ROOT") or Path(__file__).resolve().parents[2])
PRESETS = ROOT / "presets"
FONTS = ROOT / "assets" / "fonts"
MODELS = ROOT / "assets" / "models"
