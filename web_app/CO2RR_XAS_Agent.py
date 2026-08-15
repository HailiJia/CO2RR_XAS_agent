from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for path in [APP_DIR, REPO_ROOT]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from top_navigation import install_top_navigation
from tools import fresh_generation_compat, nersc_account_compat, sfapi_upload_compat, stripe_width_compat
from tools.nersc_portability_patch import run_patched_main

# Apply compatibility fixes before main.py imports workflow helpers. This keeps
# old uploaded NERSC packages usable, restores realistic Cu/Au stripe widths,
# uses NERSC's direct file-upload endpoint for workflow packages, and makes
# explicit new chemistry-generation prompts start from fresh structure state.
nersc_account_compat.install()
sfapi_upload_compat.install()
stripe_width_compat.install()
fresh_generation_compat.install()

install_top_navigation(active_page="calculation")
run_patched_main(APP_DIR)
