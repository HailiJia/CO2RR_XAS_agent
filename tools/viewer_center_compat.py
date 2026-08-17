from __future__ import annotations

from typing import Any, Callable, Dict

import numpy as np


def _periodic_fractional_mean(values: np.ndarray) -> float:
    """Return the circular mean of fractional coordinates in [0, 1)."""
    frac = np.mod(np.asarray(values, dtype=float), 1.0)
    if frac.size == 0:
        return 0.5
    if frac.size == 1:
        return float(frac[0])

    angles = 2.0 * np.pi * frac
    vector = np.exp(1j * angles).mean()
    if abs(vector) < 1e-12:
        # Degenerate/symmetric case: use the first coordinate as a stable
        # reference rather than allowing an arbitrary phase to move the model.
        return float(frac[0])
    angle = float(np.angle(vector))
    if angle < 0.0:
        angle += 2.0 * np.pi
    return angle / (2.0 * np.pi)


def center_interface_adsorbate_for_view(
    structure: Dict[str, Any],
    substrate_n: int | None,
) -> np.ndarray:
    """Periodically recenter a lateral-interface adsorbate for display only.

    The returned coordinates are a copy used only by the 3D viewer. The
    structure dictionary, POSCAR, and metadata are never modified. For a
    periodic Cu/Au interface the adsorption site can naturally lie at the cell
    edge; remapping the periodic image so that the adsorbate is near fractional
    (0.5, 0.5) makes the local binding environment much easier to inspect.
    """
    positions = np.asarray((structure or {}).get("positions", []), dtype=float).copy()
    if positions.ndim != 2 or positions.shape[1:] != (3,):
        return positions
    if substrate_n is None:
        return positions

    substrate_n = int(substrate_n)
    if substrate_n < 0 or substrate_n >= len(positions):
        return positions

    metadata = dict((structure or {}).get("metadata", {}) or {})
    interface = metadata.get("interface", {})
    interface = interface if isinstance(interface, dict) else {}
    is_lateral_interface = (
        interface.get("type") == "lateral"
        or metadata.get("interface_type") == "lateral"
    )
    if not is_lateral_interface:
        return positions

    adsorbate_positions = positions[substrate_n:]
    if len(adsorbate_positions) == 0:
        return positions

    cell = np.asarray((structure or {}).get("cell", []), dtype=float)
    if cell.shape != (3, 3):
        return positions
    try:
        inv_cell = np.linalg.inv(cell)
    except np.linalg.LinAlgError:
        return positions

    # Row-vector convention: Cartesian = fractional @ cell.
    fractional = positions @ inv_cell
    ads_frac = fractional[substrate_n:]

    center_x = _periodic_fractional_mean(ads_frac[:, 0])
    center_y = _periodic_fractional_mean(ads_frac[:, 1])
    shift = np.array([0.5 - center_x, 0.5 - center_y, 0.0], dtype=float)

    fractional[:, :2] = np.mod(fractional[:, :2] + shift[:2], 1.0)
    return fractional @ cell


def _patch_viewer_source(source: str) -> str:
    import_anchor = "from tools.contact_checks import minimum_pair as element_aware_minimum_pair\n"
    injected_import = (
        import_anchor
        + "from tools.viewer_center_compat import center_interface_adsorbate_for_view\n"
    )
    if "from tools.viewer_center_compat import center_interface_adsorbate_for_view" not in source:
        if import_anchor not in source:
            raise RuntimeError("Could not locate viewer import anchor in web_app/main.py.")
        source = source.replace(import_anchor, injected_import, 1)

    old_view = (
        '    view_structure_3dmol(atoms, positions, sorted(set(atoms)), substrate_n=substrate_n)\n'
    )
    new_view = (
        '    display_positions = center_interface_adsorbate_for_view(structure, substrate_n)\n'
        '    view_structure_3dmol(atoms, display_positions, sorted(set(atoms)), substrate_n=substrate_n)\n'
    )
    if old_view not in source and new_view not in source:
        raise RuntimeError("Could not locate the structure viewer call in web_app/main.py.")
    if old_view in source:
        source = source.replace(old_view, new_view, 1)
    return source


def install() -> None:
    """Install display-only periodic centering on top of runtime source patches."""
    from tools import nersc_portability_patch as portability

    if getattr(portability, "_viewer_center_compat_installed", False):
        return

    original_source_patch: Callable[[str], str] = portability.patched_main_source

    def patched_source(source: str) -> str:
        return _patch_viewer_source(original_source_patch(source))

    portability.patched_main_source = patched_source
    portability._viewer_center_compat_installed = True
