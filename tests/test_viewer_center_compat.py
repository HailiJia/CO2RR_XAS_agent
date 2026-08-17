from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.viewer_center_compat import center_interface_adsorbate_for_view


def _lateral_interface_structure():
    return {
        "atoms": ["Cu", "Au", "C", "O"],
        "positions": np.array(
            [
                [2.0, 2.0, 0.0],
                [8.0, 8.0, 0.0],
                [9.8, 5.0, 2.0],
                [0.2, 5.0, 2.0],
            ],
            dtype=float,
        ),
        "cell": np.diag([10.0, 10.0, 20.0]),
        "metadata": {
            "type": "interface",
            "interface_type": "lateral",
            "interface": {"type": "lateral"},
            "adsorbate": "CO",
        },
    }


def test_interface_adsorbate_crossing_periodic_edge_is_centered_for_view():
    structure = _lateral_interface_structure()
    original = structure["positions"].copy()

    displayed = center_interface_adsorbate_for_view(structure, substrate_n=2)

    # The two adsorbate atoms originally straddle x=0/10. After periodic
    # remapping they should sit together around the middle of the displayed cell.
    ads_x = np.sort(displayed[2:, 0])
    assert np.allclose(ads_x, [4.8, 5.2], atol=1e-8)
    assert np.allclose(displayed[2:, 1], [5.0, 5.0], atol=1e-8)

    # Viewer centering must never mutate the scientific structure coordinates.
    assert np.array_equal(structure["positions"], original)


def test_non_interface_structure_is_not_recentered():
    structure = _lateral_interface_structure()
    structure["metadata"] = {"type": "surface", "adsorbate": "CO"}

    displayed = center_interface_adsorbate_for_view(structure, substrate_n=2)

    assert np.array_equal(displayed, structure["positions"])


def test_app_installs_viewer_center_compatibility():
    source = (REPO_ROOT / "web_app" / "CO2RR_XAS_Agent.py").read_text()
    assert "viewer_center_compat.install()" in source
