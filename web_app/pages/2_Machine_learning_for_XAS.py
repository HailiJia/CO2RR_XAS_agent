from __future__ import annotations

from pathlib import Path


_IMPL_PATH = Path(__file__).resolve().parents[1] / "ml_pages" / "machine_learning_for_XAS_impl.py"
_source = _IMPL_PATH.read_text()
_source = _source.replace('"peak_window"', '"peak_descriptors"')
_source = _source.replace(
    'ML_PAGE_UPDATE_TAG = "v46_2026-07-03_xas_ml_target_cleanup"',
    'ML_PAGE_UPDATE_TAG = "v47_2026-07-16_peak_descriptors"',
)
exec(compile(_source, str(_IMPL_PATH), "exec"), globals())
