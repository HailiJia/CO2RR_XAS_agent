from __future__ import annotations

import runpy
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from top_navigation import install_top_navigation

install_top_navigation(active_page="calculation")
runpy.run_path(str(APP_DIR / "main.py"), run_name="__main__")
