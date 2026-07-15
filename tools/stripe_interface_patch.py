from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

STRIPE_RATIO_CHOICES = ["1:1", "2:1", "3:1", "1:2", "1:3"]
# Display ratios use a 2-row base unit:
# 1:1 -> 2 Cu rows + 2 Au rows; 2:1 -> 4 Cu rows + 2 Au rows; etc.
STRIPE_ROW_BASE_UNIT = 2
ADSORPTION_REGION_ADSORBATES = {
    "Cu_side_interface": ["CO", "OH", "O", "H", "CHO", "CO2"],
    "Cu_Au_boundary_bridge": ["COCO", "OCCO"],
    "Cu_side_interface_dimer_along_x": ["COCO", "OCCO"],
    "Cu_side_interface_dimer_row0_row1": ["COCO", "OCCO"],
    "Cu_near_interface_row_1": ["CO", "OH", "O", "H", "CHO", "CO2"],
    "Cu_near_interface_row_2": ["CO", "OH", "O", "H", "CHO", "CO2"],
    "Au_side_interface": ["CO"],
}
ADSORPTION_REGION_CHOICES = list(ADSORPTION_REGION_ADSORBATES)
SINGLE_SITE_REGIONS = {
    "Cu_side_interface": ("Cu", 0),
    "Cu_near_interface_row_1": ("Cu", 1),
    "Cu_near_interface_row_2": ("Cu", 2),
    "Au_side_interface": ("Au", 0),
}
DIMER_CU_ONLY_REGIONS = {"Cu_side_interface_dimer_along_x", "Cu_side_interface_dimer_row0_row1"}
DIMER_REGIONS = {"Cu_Au_boundary_bridge", *DIMER_CU_ONLY_REGIONS}


def parse_stripe_ratio(value: Any, default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
    """Return the display ratio units, not actual row counts."""
    if isinstance(value, (tuple, list)) and len(value) == 2:
        try:
            return max(1, int(value[0])), max(1, int(value[1]))
        except Exception:
            return default
    text = str(value or "").strip()
    m = re.search(r"(\d+)\s*:\s*(\d+)", text)
    if not m:
        return default
    return max(1, int(m.group(1))), max(1, int(m.group(2)))


def stripe_ratio_label(cu_units: int, au_units: int) -> str:
    return f"{int(cu_units)}:{int(au_units)}"


def actual_rows_from_ratio(value: Any) -> Tuple[int, int, str]:
    cu_units, au_units = parse_stripe_ratio(value)
    return STRIPE_ROW_BASE_UNIT * cu_units, STRIPE_ROW_BASE_UNIT * au_units, stripe_ratio_label(cu_units, au_units)


def region_min_cu_rows(region: str) -> int:
    if region in {"Cu_near_interface_row_1", "Cu_side_interface_dimer_row0_row1"}:
        return 4
    if region == "Cu_near_interface_row_2":
        return 6
    return 2


def region_is_valid_for_ratio(region: str, stripe_ratio: Any) -> bool:
    cu_rows, _au_rows, _label = actual_rows_from_ratio(stripe_ratio)
    return cu_rows >= region_min_cu_rows(region)


def region_options_for_adsorbate(adsorbate: str, stripe_ratio: Any = "1:1") -> List[str]:
    if not adsorbate or adsorbate == "clean":
        return ["Cu_side_interface"]
    options = [region for region, ads_list in ADSORPTION_REGION_ADSORBATES.items() if adsorbate in ads_list]
    options = [region for region in options if region_is_valid_for_ratio(region, stripe_ratio)]
    return options or ["Cu_side_interface"]


def _safe_first_region(adsorbate: str, requested: Optional[str], stripe_ratio: Any = "1:1") -> str:
    options = region_options_for_adsorbate(adsorbate, stripe_ratio=stripe_ratio)
    if requested in options:
        return str(requested)
    if requested in ADSORPTION_REGION_CHOICES:
        raise ValueError(
            f"Adsorption region {requested} is not valid for adsorbate={adsorbate} and stripe_ratio={stripe_ratio}. "
            f"Available regions: {options}"
        )
    return options[0]


def is_all_region_request(text: str) -> bool:
    low = str(text or "").lower()
    wants_all = bool(re.search(r"\b(all|every|full|complete)\b", low))
    mentions_region = "adsorption region" in low or "adsorption regions" in low or "region" in low
    mentions_cu_au = "cu/au" in low or "cu-au" in low or ("cu" in low and "au" in low)
    mentions_generate = bool(re.search(r"\b(generate|make|build|create)\b", low))
    mentions_possible = "possible" in low or "combination" in low or "iterate" in low or "screen" in low
    return mentions_generate and mentions_cu_au and mentions_region and (wants_all or mentions_possible)


def binding_element_for_region(region: str) -> Optional[str]:
    if region == "Au_side_interface":
        return "Au"
    if region == "Cu_Au_boundary_bridge":
        return None
    return "Cu"


def install() -> None:
    """Install Cu/Au stripe-ratio and adsorption-region support by patching the generator module."""
    from tools import utils
    import tools.structure_generator as sg

    if "O" not in utils.ADSORBATES:
        utils.ADSORBATES["O"] = {"atoms": ["O"], "positions": np.array([[0.0, 0.0, 0.0]]), "binding_atom": 0}
    if "O" not in sg.ADSORBATES:
        sg.ADSORBATES["O"] = utils.ADSORBATES["O"]
    sg.DEFAULT_ADSORBATE_SITES["O"] = "fcc"
    sg.DEFAULT_INTERFACE_BINDING_ELEMENTS["O"] = "Cu"
    sg.INTERFACE_GENERATOR_UPDATE_TAG = "v6_2026-07-08_stripe_2row_base_dimer_regions"

    cls = sg.StructureGenerator
    if getattr(cls, "_stripe_region_patch_installed", False):
        return

    cls._stripe_original_add_adsorbate = cls.add_adsorbate
    cls._stripe_original_generate_interface = cls.generate_interface
    cls.generate_interface = _patched_generate_interface
    cls.add_adsorbate = _patched_add_adsorbate
    cls._stripe_region_patch_installed = True


def _rows_for_elements(
    element1: str,
    element2: str,
    requested_rows_per_side: int,
    stripe_ratio: Any = None,
    cu_rows: Optional[int] = None,
    au_rows: Optional[int] = None,
    element1_rows: Optional[int] = None,
    element2_rows: Optional[int] = None,
) -> Tuple[int, int, int, int, str]:
    requested_rows_per_side = max(1, int(requested_rows_per_side))
    if element1_rows is not None or element2_rows is not None:
        e1_rows = max(1, int(element1_rows or requested_rows_per_side))
        e2_rows = max(1, int(element2_rows or requested_rows_per_side))
        final_cu_rows = e1_rows if element1 == "Cu" else (e2_rows if element2 == "Cu" else 0)
        final_au_rows = e1_rows if element1 == "Au" else (e2_rows if element2 == "Au" else 0)
        ratio = stripe_ratio_label(max(1, final_cu_rows // STRIPE_ROW_BASE_UNIT), max(1, final_au_rows // STRIPE_ROW_BASE_UNIT)) if final_cu_rows and final_au_rows else f"{e1_rows}:{e2_rows}"
    else:
        if cu_rows is None or au_rows is None:
            cu_rows_parsed, au_rows_parsed, ratio = actual_rows_from_ratio(stripe_ratio)
            cu_rows = cu_rows if cu_rows is not None else cu_rows_parsed
            au_rows = au_rows if au_rows is not None else au_rows_parsed
        else:
            ratio = stripe_ratio_label(max(1, int(cu_rows) // STRIPE_ROW_BASE_UNIT), max(1, int(au_rows) // STRIPE_ROW_BASE_UNIT))
        cu_rows = max(1, int(cu_rows))
        au_rows = max(1, int(au_rows))
        e1_rows = cu_rows if element1 == "Cu" else (au_rows if element1 == "Au" else requested_rows_per_side)
        e2_rows = cu_rows if element2 == "Cu" else (au_rows if element2 == "Au" else requested_rows_per_side)
        final_cu_rows = e1_rows if element1 == "Cu" else (e2_rows if element2 == "Cu" else 0)
        final_au_rows = e1_rows if element1 == "Au" else (e2_rows if element2 == "Au" else 0)
    return e1_rows, e2_rows, final_cu_rows, final_au_rows, ratio


def _patched_generate_interface(
    self,
    element1: str,
    element2: str,
    facet1: str = "111",
    facet2: str = "111",
    supercell: Tuple[int, int] = (3, 3),
    layers1: int = 3,
    layers2: int = 3,
    vacuum: float = 15.0,
    interface_distance: float = 2.5,
    element1_repeats: Optional[int] = None,
    element2_repeats: Optional[int] = None,
    match_mode: str = "auto",
    stripe_ratio: Any = None,
    cu_rows: Optional[int] = None,
    au_rows: Optional[int] = None,
    element1_rows: Optional[int] = None,
    element2_rows: Optional[int] = None,
) -> Dict[str, Any]:
    if facet1 != facet2:
        raise ValueError("Lateral interfaces currently require matching facets")
    if facet1 != "111":
        raise ValueError("Lateral interface generation currently supports fcc(111) facets")
    if element1 not in self.metal_data or element2 not in self.metal_data:
        raise ValueError(f"Unknown interface elements: {element1}, {element2}")
    if self.metal_data[element1]["structure"] != "fcc" or self.metal_data[element2]["structure"] != "fcc":
        raise ValueError("Lateral interface generation currently supports fcc/fcc interfaces")

    requested_nx = int(supercell[0]) if supercell else 3
    requested_rows_per_side = int(supercell[1]) if supercell else 3
    e1_rows, e2_rows, cu_rows_final, au_rows_final, ratio = _rows_for_elements(
        element1,
        element2,
        requested_rows_per_side,
        stripe_ratio=stripe_ratio,
        cu_rows=cu_rows,
        au_rows=au_rows,
        element1_rows=element1_rows,
        element2_rows=element2_rows,
    )
    layers = max(int(layers1), int(layers2))

    period1 = self.metal_data[element1]["a"] / np.sqrt(2)
    period2 = self.metal_data[element2]["a"] / np.sqrt(2)

    if match_mode == "manual" or element1_repeats is not None or element2_repeats is not None:
        n1 = max(1, int(element1_repeats or requested_nx))
        n2 = max(1, int(element2_repeats or requested_nx))
        length1 = n1 * period1
        length2 = n2 * period2
        matched = 0.5 * (length1 + length2)
        strain1 = (matched - length1) / length1
        strain2 = (matched - length2) / length2
        match = {
            "match_mode": "manual",
            "match_direction": "x/interface",
            "element1_repeats": n1,
            "element2_repeats": n2,
            "element1_natural_length": length1,
            "element2_natural_length": length2,
            "matched_block_length": matched,
            "element1_strain": strain1,
            "element2_strain": strain2,
            "max_abs_strain": max(abs(strain1), abs(strain2)),
        }
    else:
        match = self._match_lateral_repeats(element1, element2)
        match["match_mode"] = "auto"
        match["match_direction"] = "x/interface"
        n1 = int(match["element1_repeats"])
        n2 = int(match["element2_repeats"])

    common_x = float(match["matched_block_length"])
    row_spacing1 = period1 * np.sqrt(3) / 2
    row_spacing2 = period2 * np.sqrt(3) / 2
    block1_y = e1_rows * row_spacing1
    block2_y = e2_rows * row_spacing2
    total_y = block1_y + block2_y
    split_y = block1_y
    layer_spacing = 0.5 * (self.metal_data[element1]["a"] / np.sqrt(3) + self.metal_data[element2]["a"] / np.sqrt(3))
    stacking = [np.array([0.0, 0.0]), np.array([1 / 3, 1 / 3]), np.array([2 / 3, 2 / 3])]

    atoms: List[str] = []
    positions: List[np.ndarray] = []
    for layer in range(layers):
        shift = stacking[layer % 3]
        z = layer * layer_spacing
        for j in range(e1_rows):
            row_shift = 0.5 * (j % 2)
            y = (j + shift[1]) * row_spacing1
            for i in range(n1):
                x_frac = (i + shift[0] + row_shift) / n1
                atoms.append(element1)
                positions.append(np.array([(x_frac % 1.0) * common_x, y, z]))
        for j in range(e2_rows):
            row_shift = 0.5 * (j % 2)
            y = split_y + (j + shift[1]) * row_spacing2
            for i in range(n2):
                x_frac = (i + shift[0] + row_shift) / n2
                atoms.append(element2)
                positions.append(np.array([(x_frac % 1.0) * common_x, y, z]))

    all_positions = np.array(positions, dtype=float)
    all_positions[:, 2] -= all_positions[:, 2].min()
    slab_height = all_positions[:, 2].max()
    cell = np.array([[common_x, 0.0, 0.0], [0.0, total_y, 0.0], [0.0, 0.0, slab_height + vacuum]])

    descriptors = {
        "cu_rows": cu_rows_final,
        "au_rows": au_rows_final,
        "stripe_ratio": ratio,
        "stripe_row_base_unit": STRIPE_ROW_BASE_UNIT,
        "interface_axis": "x",
        "split_axis": "y",
    }
    return {
        "atoms": atoms,
        "positions": all_positions,
        "cell": cell,
        "metadata": {
            "element1": element1,
            "element2": element2,
            "structure_type": "interface",
            "type": "interface",
            "interface_type": "lateral",
            "facet1": facet1,
            "facet2": facet2,
            "supercell": [max(n1, n2), e1_rows + e2_rows],
            "requested_supercell": list(supercell),
            "layers1": layers,
            "layers2": layers,
            "vacuum": vacuum,
            "interface_distance": interface_distance,
            "cu_rows": cu_rows_final,
            "au_rows": au_rows_final,
            "stripe_ratio": ratio,
            "stripe_row_base_unit": STRIPE_ROW_BASE_UNIT,
            "descriptors": dict(descriptors),
            "interface": {
                "type": "lateral",
                "interface_axis": "x",
                "split_axis": "y",
                "split_coordinate": split_y,
                "rows_per_side": max(e1_rows, e2_rows),
                "element1_rows": e1_rows,
                "element2_rows": e2_rows,
                "cu_rows": cu_rows_final,
                "au_rows": au_rows_final,
                "stripe_ratio": ratio,
                "stripe_row_base_unit": STRIPE_ROW_BASE_UNIT,
                "element1_block": {"y_min": 0.0, "y_max": split_y},
                "element2_block": {"y_min": split_y, "y_max": total_y},
                "actual_repeats": {element1: n1, element2: n2, "element1_rows": e1_rows, "element2_rows": e2_rows},
                "match": match,
            },
        },
    }


def _top_row_values(positions: np.ndarray, indices: Iterable[int], axis: int, tol: float = 1e-5) -> List[float]:
    values = sorted(float(positions[idx, axis]) for idx in indices)
    out: List[float] = []
    for value in values:
        if not out or abs(value - out[-1]) > tol:
            out.append(value)
    return out


def _rank_rows_by_nearest_interface(row_values: List[float], y_min: float, y_max: float) -> Dict[float, int]:
    ordered = sorted(row_values, key=lambda y: (min(abs(y - y_min), abs(y_max - y)), y))
    ranks: Dict[float, int] = {}
    # Pairs of rows at both stripe boundaries are the same distance class.
    class_distances: List[float] = []
    for y in ordered:
        dist = min(abs(y - y_min), abs(y_max - y))
        class_id = None
        for idx, prev in enumerate(class_distances):
            if abs(dist - prev) < 1e-5:
                class_id = idx
                break
        if class_id is None:
            class_distances.append(dist)
            class_id = len(class_distances) - 1
        ranks[y] = class_id
    return ranks


def _row_candidates_for_region(self, atoms: List[str], positions: np.ndarray, top_indices: np.ndarray, metadata: Dict[str, Any], element: str, row_rank: int) -> List[int]:
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    split_axis_name = interface_meta.get("split_axis", "y")
    split_axis = 1 if split_axis_name == "y" else 0
    element1 = metadata.get("element1")
    element2 = metadata.get("element2")
    if element == element1:
        y_min = float(interface_meta.get("element1_block", {}).get("y_min", 0.0))
        y_max = float(interface_meta.get("element1_block", {}).get("y_max", interface_meta.get("split_coordinate", 0.0)))
    elif element == element2:
        y_min = float(interface_meta.get("element2_block", {}).get("y_min", interface_meta.get("split_coordinate", 0.0)))
        y_max = float(interface_meta.get("element2_block", {}).get("y_max", positions[:, split_axis].max()))
    else:
        y_min = float(positions[:, split_axis].min())
        y_max = float(positions[:, split_axis].max())

    candidates = [idx for idx in top_indices if atoms[idx] == element]
    if not candidates:
        return []
    row_values = _top_row_values(positions, candidates, split_axis)
    rank_by_y = _rank_rows_by_nearest_interface(row_values, y_min, y_max)
    allowed_rows = [y for y, rank in rank_by_y.items() if rank == row_rank]
    out = [idx for idx in candidates if any(abs(float(positions[idx, split_axis]) - y) < 1e-5 for y in allowed_rows)]
    return out


def _choose_atom_near_x(self, positions: np.ndarray, candidates: List[int], cell: np.ndarray, line_axis: int, ref_line: Optional[float] = None) -> int:
    line_length = float(cell[line_axis, line_axis]) if cell[line_axis, line_axis] else None
    ref = 0.5 * line_length if ref_line is None and line_length else float(ref_line or 0.0)
    return min(candidates, key=lambda idx: self._periodic_delta(float(positions[idx, line_axis]), ref, line_length))


def _choose_neighbor_along_x(self, positions: np.ndarray, candidates: List[int], cell: np.ndarray, line_axis: int, primary_idx: int) -> int:
    line_length = float(cell[line_axis, line_axis]) if cell[line_axis, line_axis] else None
    x0 = float(positions[primary_idx, line_axis])
    others = [idx for idx in candidates if idx != primary_idx]
    if not others:
        raise ValueError("Need at least two Cu atoms along the interface for dimer_along_x placement.")
    # Prefer a nearby but not identical x-neighbor along the same Cu row.
    return min(others, key=lambda idx: self._periodic_delta(float(positions[idx, line_axis]), x0, line_length))


def _site_index_for_region(self, atoms: List[str], positions: np.ndarray, cell: np.ndarray, top_indices: np.ndarray, metadata: Dict[str, Any], region: str) -> Tuple[int, Optional[str], int]:
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    split_axis_name = interface_meta.get("split_axis", "y")
    split_axis = 1 if split_axis_name == "y" else 0
    line_axis = 0 if split_axis == 1 else 1
    element, row_rank = SINGLE_SITE_REGIONS.get(region, ("Cu", 0))
    candidates = _row_candidates_for_region(self, atoms, positions, top_indices, metadata, element, row_rank)
    if not candidates:
        raise ValueError(f"No valid {element} top-layer candidates for adsorption_region={region}.")
    chosen_global_idx = _choose_atom_near_x(self, positions, candidates, cell, line_axis)
    return list(top_indices).index(chosen_global_idx), element, row_rank


def _custom_add_cu_dimer_adsorbate(self, structure: Dict[str, Any], adsorbate_name: str, region: str, height: float) -> Dict[str, Any]:
    if adsorbate_name not in {"OCCO", "COCO"}:
        raise ValueError(f"{region} is only configured for OCCO/COCO, not {adsorbate_name}.")
    atoms = list(structure["atoms"])
    positions = np.asarray(structure["positions"], dtype=float)
    cell = np.asarray(structure["cell"], dtype=float)
    metadata = dict(structure.get("metadata", {}) or {})
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    split_axis_name = interface_meta.get("split_axis", "y")
    split_axis = 1 if split_axis_name == "y" else 0
    line_axis = 0 if split_axis == 1 else 1
    top_indices = self._top_layer_indices(positions)
    z_max = positions[:, 2].max()

    row0_candidates = _row_candidates_for_region(self, atoms, positions, top_indices, metadata, "Cu", 0)
    if not row0_candidates:
        raise ValueError("No Cu interface row candidates are available for Cu-only dimer placement.")
    first_idx = _choose_atom_near_x(self, positions, row0_candidates, cell, line_axis)
    if region == "Cu_side_interface_dimer_along_x":
        second_idx = _choose_neighbor_along_x(self, positions, row0_candidates, cell, line_axis, first_idx)
        distance_to_interface_row = 0
    else:
        row1_candidates = _row_candidates_for_region(self, atoms, positions, top_indices, metadata, "Cu", 1)
        if not row1_candidates:
            raise ValueError("Cu_side_interface_dimer_row0_row1 requires at least 4 Cu rows.")
        second_idx = _choose_atom_near_x(self, positions, row1_candidates, cell, line_axis, positions[first_idx, line_axis])
        distance_to_interface_row = 1

    ads_data = self.adsorbates[adsorbate_name]
    ads_atoms = list(ads_data["atoms"])
    ads_positions = np.array(ads_data["positions"], dtype=float).copy()
    c_indices = [idx for idx, atom in enumerate(ads_atoms) if atom == "C"]
    if len(c_indices) < 2:
        raise ValueError(f"{adsorbate_name} needs at least two C atoms for dimer placement.")
    c0, c1 = c_indices[:2]
    original_c0 = ads_positions[c0].copy()
    original_c1 = ads_positions[c1].copy()
    target0 = positions[first_idx].copy()
    target1 = positions[second_idx].copy()
    target0[2] = z_max + height
    target1[2] = z_max + height
    new_ads_positions = ads_positions.copy()
    for k in range(len(ads_atoms)):
        d0 = np.linalg.norm(ads_positions[k] - original_c0)
        d1 = np.linalg.norm(ads_positions[k] - original_c1)
        if d0 <= d1:
            new_ads_positions[k] = target0 + (ads_positions[k] - original_c0)
        else:
            new_ads_positions[k] = target1 + (ads_positions[k] - original_c1)
    new_ads_positions[c0] = target0
    new_ads_positions[c1] = target1

    new_metadata = metadata.copy()
    new_metadata["adsorbate"] = adsorbate_name
    new_metadata["adsorption_site"] = region
    new_metadata["adsorption_height"] = height
    new_metadata["adsorption_binding_element"] = "Cu"
    new_metadata = self._with_ml_metadata(new_metadata, adsorbate_name, region, None)
    result = {
        "atoms": atoms + ads_atoms,
        "positions": np.vstack([positions, new_ads_positions]),
        "cell": cell.copy(),
        "metadata": new_metadata,
    }
    _attach_region_descriptors(result, adsorbate_name, region, "Cu", distance_to_interface_row, region)
    return result


def _patched_add_adsorbate(self, structure: Dict[str, Any], adsorbate_name: str, *args: Any, adsorption_region: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    metadata = structure.get("metadata", {})
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    if interface_meta.get("type") != "lateral" or adsorbate_name == "clean":
        return self._stripe_original_add_adsorbate(structure, adsorbate_name, *args, **kwargs)

    stripe_ratio = metadata.get("stripe_ratio") or interface_meta.get("stripe_ratio") or "1:1"
    region = _safe_first_region(adsorbate_name, adsorption_region, stripe_ratio=stripe_ratio)
    height = float(kwargs.get("height", 2.0))
    kwargs.pop("binding_element", None)  # Region is authoritative; no independent binding-element selector.

    if region in DIMER_CU_ONLY_REGIONS:
        return _custom_add_cu_dimer_adsorbate(self, structure, adsorbate_name, region, height)

    positions = np.asarray(structure["positions"], dtype=float)
    atoms = list(structure["atoms"])
    cell = np.asarray(structure["cell"], dtype=float)
    top_indices = self._top_layer_indices(positions)
    binding_element = binding_element_for_region(region)
    site_index = kwargs.pop("site_index", None)
    row_distance = 0
    binding_mode = "single_site"

    if region == "Cu_Au_boundary_bridge":
        if adsorbate_name not in {"OCCO", "COCO"}:
            raise ValueError("Cu_Au_boundary_bridge is only configured for OCCO/COCO.")
        binding_mode = "dual_C_bridge"
        kwargs["site"] = "bridge_interface"
    else:
        site_index, binding_element, row_distance = _site_index_for_region(self, atoms, positions, cell, top_indices, metadata, region)

    result = self._stripe_original_add_adsorbate(
        structure,
        adsorbate_name,
        *args,
        site_index=site_index,
        binding_element=binding_element,
        **kwargs,
    )
    _attach_region_descriptors(result, adsorbate_name, region, binding_element, row_distance, binding_mode)
    return result


def _attach_region_descriptors(structure: Dict[str, Any], adsorbate_name: str, region: str, binding_element: Optional[str], row_distance: int, binding_mode: str) -> None:
    metadata = structure.setdefault("metadata", {})
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    cu_rows = metadata.get("cu_rows") or interface_meta.get("cu_rows")
    au_rows = metadata.get("au_rows") or interface_meta.get("au_rows")
    stripe_ratio = metadata.get("stripe_ratio") or interface_meta.get("stripe_ratio")
    descriptors = dict(metadata.get("descriptors") or {})
    descriptors.update({
        "cu_rows": cu_rows,
        "au_rows": au_rows,
        "stripe_ratio": stripe_ratio,
        "stripe_row_base_unit": metadata.get("stripe_row_base_unit") or interface_meta.get("stripe_row_base_unit", STRIPE_ROW_BASE_UNIT),
        "interface_axis": interface_meta.get("interface_axis", "x"),
        "split_axis": interface_meta.get("split_axis", "y"),
        "adsorption_region": region,
        "adsorbate_identity": adsorbate_name,
        "adsorption_binding_element": binding_element,
        "distance_to_interface_row": int(row_distance),
        "binding_mode": binding_mode,
    })
    metadata["descriptors"] = descriptors
    for key, value in descriptors.items():
        if key in {"adsorption_region", "adsorption_binding_element", "distance_to_interface_row", "binding_mode", "cu_rows", "au_rows", "stripe_ratio", "stripe_row_base_unit"}:
            metadata[key] = value
    ads_meta = dict(metadata.get("adsorbate_metadata") or {})
    ads_meta.update({
        "identity": adsorbate_name,
        "adsorption_region": region,
        "binding_element": binding_element,
        "binding_mode": binding_mode,
        "distance_to_interface_row": int(row_distance),
    })
    if region == "Cu_Au_boundary_bridge":
        ads_meta["binding_site_pair"] = "Cu_interface-Au_interface"
    elif region == "Cu_side_interface_dimer_along_x":
        ads_meta["binding_site_pair"] = "Cu_interface-Cu_interface_along_x"
    elif region == "Cu_side_interface_dimer_row0_row1":
        ads_meta["binding_site_pair"] = "Cu_interface-Cu_near_interface_row_1"
    metadata["adsorbate_metadata"] = ads_meta


STRIPE_BATCH_HELPER_SOURCE = r'''

def _stripe_generate_all_adsorption_region_batch(params):
    """Generate all valid Cu/Au(111) stripe-ratio × adsorption-region × allowed-adsorbate structures."""
    params["structure_source"] = "Build from settings"
    params["structure_mode"] = "Interface"
    params["element1"] = "Cu"
    params["element2"] = "Au"
    params["facet"] = "111"
    params["batch_source"] = "Generate pathway from current structure setup"

    generator = StructureGenerator()
    items = []
    combo_manifest = []

    for ratio in stripe_patch_ratio_choices():
        cu_rows, au_rows, _ = stripe_patch_actual_rows(ratio)
        for region, adsorbates in stripe_patch_region_adsorbates().items():
            if not stripe_patch_region_valid(region, ratio):
                continue
            for ads in adsorbates:
                base = generator.generate_interface(
                    element1="Cu",
                    element2="Au",
                    facet1="111",
                    facet2="111",
                    supercell=(int(params.get("nx", 3)), int(params.get("ny", 3))),
                    layers1=int(params.get("layers", 4)),
                    layers2=int(params.get("layers", 4)),
                    vacuum=float(params.get("vacuum", 15.0)),
                    match_mode=params.get("interface_match_mode", "auto"),
                    element1_repeats=int(params.get("element1_repeats", 4)) if params.get("interface_match_mode") == "manual" else None,
                    element2_repeats=int(params.get("element2_repeats", 4)) if params.get("interface_match_mode") == "manual" else None,
                    stripe_ratio=ratio,
                )
                structure = generator.add_adsorbate(
                    base,
                    ads,
                    site="auto",
                    height=float(params.get("height", 2.0)),
                    adsorption_region=region,
                )
                name = sanitize_name(f"Cu_Au_111_stripe_{ratio.replace(':', 'to')}_{region}_{ads}")
                items.append({
                    "name": name,
                    "source": "stripe_adsorption_region_grid",
                    "structure": structure,
                    "substrate_n": len(base.get("atoms", [])),
                    "source_poscar": None,
                })
                combo_manifest.append({
                    "name": name,
                    "stripe_ratio": ratio,
                    "cu_rows": cu_rows,
                    "au_rows": au_rows,
                    "adsorption_region": region,
                    "adsorbate": ads,
                })

    batch_result = generate_batch_input_package(
        batch_items=items,
        output_dir=params.get("batch_output_dir", "generated_outputs/web_xas_agent/batch"),
        params=params,
        include_relaxation=params.get("batch_include_relaxation", True),
        include_xas=params.get("batch_include_xas", True),
        absorber_mode=params.get("batch_absorber_mode", "auto"),
    )
    batch_result["stripe_adsorption_region_grid"] = {
        "stripe_ratios": stripe_patch_ratio_choices(),
        "stripe_row_base_unit": 2,
        "adsorption_region_adsorbates": stripe_patch_region_adsorbates(),
        "n_combinations": len(combo_manifest),
        "combinations": combo_manifest,
    }
    manifest_path = Path(batch_result["run_dir"]) / "stripe_adsorption_region_manifest.json"
    manifest_path.write_text(json.dumps(batch_result["stripe_adsorption_region_grid"], indent=2))
    st.session_state["last_batch_result"] = batch_result
    st.session_state["last_generated_package_kind"] = "batch"
    params["nersc_upload_source"] = "Last batch package"
    return batch_result
'''


def patched_main_source(source: str) -> str:
    """Patch main.py source at runtime so the web UI exposes stripe/region controls and agent batch commands."""
    source = source.replace(
        'ADSORBATE_CHOICES = ["clean", "CO", "OCCO", "CHO", "COCO", "CO2", "OH", "H", "CHOH", "CH", "CH2", "CH3", "CH4"]',
        'ADSORBATE_CHOICES = ["clean", "CO", "O", "OCCO", "CHO", "COCO", "CO2", "OH", "H", "CHOH", "CH", "CH2", "CH3", "CH4"]',
    )
    source = source.replace(
        '    interface_binding_element="Cu",\n):',
        '    interface_binding_element="Cu",\n    stripe_ratio="1:1",\n    adsorption_region=None,\n):',
    )
    source = source.replace(
        '            match_mode=interface_match_mode,\n            **kwargs,',
        '            match_mode=interface_match_mode,\n            stripe_ratio=stripe_ratio,\n            **kwargs,',
    )
    source = source.replace(
        '                binding_element=interface_binding_element,\n            )',
        '                binding_element=interface_binding_element,\n                adsorption_region=adsorption_region,\n            )',
        1,
    )
    for old in [
        'interface_binding_element=params.get("interface_binding_element", params.get("element1", "Cu")),',
        'interface_binding_element=params.get("interface_binding_element", params["element1"]),',
        'interface_binding_element=p.get("interface_binding_element", p["element1"]),',
    ]:
        replacement = old + ('\n                stripe_ratio=params.get("stripe_ratio", "1:1"),' if 'params' in old else '\n                stripe_ratio=p.get("stripe_ratio", "1:1"),')
        source = source.replace(old, replacement)
    source = source.replace(
        'stripe_ratio=params.get("stripe_ratio", "1:1"),',
        'stripe_ratio=params.get("stripe_ratio", "1:1"),\n                adsorption_region=params.get("adsorption_region"),',
    )
    source = source.replace(
        'stripe_ratio=p.get("stripe_ratio", "1:1"),',
        'stripe_ratio=p.get("stripe_ratio", "1:1"),\n                adsorption_region=p.get("adsorption_region"),',
    )
    source = source.replace(
        '        "interface_binding_element": "Cu",\n',
        '        "interface_binding_element": "Cu",\n        "stripe_ratio": "1:1",\n        "adsorption_region": "Cu_side_interface",\n',
    )
    source = source.replace(
        '            else:\n                st.write("Auto mode chooses low-strain repeats along the interface. `ny` controls rows away from the interface.")',
        '            else:\n                st.write("Auto mode chooses low-strain repeats along the interface. Stripe ratio controls Cu/Au row widths perpendicular to the interface.")',
    )
    source = source.replace(
        '            with c2:\n                p["ny"] = st.number_input("Rows away from interface", min_value=1, max_value=30, value=int(p["ny"]))\n\n        p["facet"] = st.selectbox("Facet", ["111"], index=0)',
        '            with c2:\n                p["ny"] = st.number_input("Rows fallback", min_value=1, max_value=30, value=int(p["ny"]), help="Used only if no Cu:Au stripe ratio is selected.")\n            ratio_options = stripe_patch_ratio_choices()\n            p["stripe_ratio"] = st.selectbox("Cu:Au stripe row ratio", ratio_options, index=safe_index(ratio_options, p.get("stripe_ratio", "1:1")))\n            cu_rows, au_rows, _ = stripe_patch_actual_rows(p["stripe_ratio"])\n            st.caption(f"Stripe width: Cu rows = {cu_rows}, Au rows = {au_rows}; interface axis x, split axis y. Ratio labels use a 2-row base unit.")\n\n        p["facet"] = st.selectbox("Facet", ["111"], index=0)',
    )
    source = source.replace(
        '            p["interface_binding_element"] = st.selectbox(\n                "Adsorbate binding site metal",\n                binding_options,\n                index=safe_index(binding_options, p["interface_binding_element"]),\n            )',
        '            region_options = stripe_patch_region_options(p["adsorbate"], p.get("stripe_ratio", "1:1"))\n            if p.get("adsorption_region") not in region_options:\n                p["adsorption_region"] = region_options[0]\n            p["adsorption_region"] = st.selectbox(\n                "Adsorption region",\n                region_options,\n                index=safe_index(region_options, p.get("adsorption_region", region_options[0])),\n                help="Region is authoritative and determines Cu/Au binding automatically. Stored in structure_info.metadata.descriptors.adsorption_region.",\n            )\n            p["interface_binding_element"] = stripe_patch_binding_element_for_region(p["adsorption_region"]) or "Cu"\n            st.caption(f"Binding element is inferred from region: {p[\'interface_binding_element\']}.")',
    )
    source = source.replace(
        '    return parsed\n\n\ndef generate_structure',
        '    stripe_match = re.search(r"(?:cu\\s*[:/]\\s*au\\s*)?(1|2|3)\\s*:\\s*(1|2|3)", text)\n    if stripe_match:\n        parsed["stripe_ratio"] = f"{stripe_match.group(1)}:{stripe_match.group(2)}"\n    for region in stripe_patch_region_choices():\n        if region.lower() in text.replace("-", "_"):\n            parsed["adsorption_region"] = region\n            parsed["interface_binding_element"] = stripe_patch_binding_element_for_region(region) or "Cu"\n            parsed.setdefault("structure_mode", "Interface")\n            parsed.setdefault("element1", "Cu")\n            parsed.setdefault("element2", "Au")\n            break\n    if "stripe_ratio" in parsed:\n        parsed.setdefault("structure_mode", "Interface")\n        parsed.setdefault("element1", "Cu")\n        parsed.setdefault("element2", "Au")\n\n    return parsed\n\n\ndef generate_structure',
    )
    source = source.replace(
        '["OCCO", "COCO", "CHOH", "CHO", "CO2", "CO", "OH", "CH3", "CH2", "CH", "H"]',
        '["OCCO", "COCO", "CHOH", "CHO", "CO2", "CO", "OH", "O", "CH3", "CH2", "CH", "H"]',
    )
    source = source.replace(
        'def build_uploaded_batch_items(uploaded_files, workspace):',
        STRIPE_BATCH_HELPER_SOURCE + '\n\ndef build_uploaded_batch_items(uploaded_files, workspace):',
    )
    source = source.replace(
        '    low = str(text or "").lower()\n    plan = plan or {}\n',
        '    low = str(text or "").lower()\n    plan = plan or {}\n\n    if stripe_patch_is_all_region_request(text):\n        result = _stripe_generate_all_adsorption_region_batch(params)\n        combo = result.get("stripe_adsorption_region_grid", {})\n        return (\n            f"Generated Cu/Au(111) stripe adsorption-region batch with `{combo.get(\'n_combinations\', result.get(\'n_structures\'))}` valid structures at `{result.get(\'run_dir\')}`.\\n"\n            f"Stripe ratios: `{\', \'.join(combo.get(\'stripe_ratios\', []))}`. Ratio labels use a 2-row base unit.\\n"\n            "Invalid row_1/row_2 combinations are skipped automatically. Next: upload to NERSC, or download the batch ZIP from the Batch generation panel."\n        )\n',
        1,
    )
    return source


def run_patched_main(app_dir: Path) -> None:
    install()
    source_path = Path(app_dir) / "main.py"
    source = patched_main_source(source_path.read_text())
    globals_dict = {
        "__name__": "__main__",
        "__file__": str(source_path),
        "stripe_patch_ratio_choices": lambda: STRIPE_RATIO_CHOICES,
        "stripe_patch_region_choices": lambda: ADSORPTION_REGION_CHOICES,
        "stripe_patch_region_options": region_options_for_adsorbate,
        "stripe_patch_region_adsorbates": lambda: ADSORPTION_REGION_ADSORBATES,
        "stripe_patch_parse_ratio": parse_stripe_ratio,
        "stripe_patch_actual_rows": actual_rows_from_ratio,
        "stripe_patch_region_valid": region_is_valid_for_ratio,
        "stripe_patch_binding_element_for_region": binding_element_for_region,
        "stripe_patch_is_all_region_request": is_all_region_request,
    }
    exec(compile(source, str(source_path), "exec"), globals_dict)
