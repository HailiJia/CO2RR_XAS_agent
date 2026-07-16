from pathlib import Path

import numpy as np

from tools.xas_ml_utils import feature_matrix


def _rows():
    x = np.linspace(0.0, 4.0, 5)
    return [
        {
            "x": x,
            "y": np.asarray([0.0, 1.0, 3.0, 1.0, 0.0]),
            "n_points": 5,
            "metadata": {"test.target": "A"},
            "record": {},
        },
        {
            "x": x,
            "y": np.asarray([0.0, 0.5, 2.0, 1.5, 0.0]),
            "n_points": 5,
            "metadata": {"test.target": "B"},
            "record": {},
        },
    ]


def test_peak_descriptors_name_and_legacy_alias_are_equivalent():
    current, y_current, names_current, _ = feature_matrix(
        _rows(), "test.target", ["peak_descriptors"], "minmax", 8
    )
    legacy, y_legacy, names_legacy, _ = feature_matrix(
        _rows(), "test.target", ["peak_window"], "minmax", 8
    )

    assert current.shape == (2, 9)
    assert np.allclose(current, legacy)
    assert np.array_equal(y_current, y_legacy)
    assert names_current == names_legacy == [
        "peak_x",
        "peak_y",
        "area",
        "derivative_max",
        "derivative_min",
        "window_1_area",
        "window_2_area",
        "window_3_area",
        "window_4_area",
    ]


def test_ml_page_displays_peak_descriptors():
    repo_root = Path(__file__).resolve().parents[1]
    wrapper = (repo_root / "web_app" / "pages" / "2_Machine_learning_for_XAS.py").read_text()
    implementation = repo_root / "web_app" / "ml_pages" / "machine_learning_for_XAS_impl.py"

    assert implementation.is_file()
    assert 'replace(\'"peak_window"\', \'"peak_descriptors"\')' in wrapper
