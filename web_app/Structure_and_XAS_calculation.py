# web_app/Structure_and_XAS_calculation.py
"""Compatibility entrypoint with a clearer Streamlit navigation label.

Run with:
    streamlit run web_app/Structure_and_XAS_calculation.py

The legacy entrypoint web_app/app.py is kept for backward compatibility.
"""

from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).with_name("app.py")), run_name="__main__")
