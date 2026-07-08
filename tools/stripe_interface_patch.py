from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

STRIPE_RATIO_CHOICES = ["1:1", "2:1", "3:1", "1:2", "1:3"]
ADSORPTION_REGION_ADSORBATES = {
    "Cu_side_interface": ["CO", "OH", "O", "H", "CHO", "CO2"],
    "Cu_Au_boundary_bridge": ["COCO", "OCCO"],
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


def parse_stripe_ratio(value: Any, default: Tuple[int, int] = (1, 1)) -> Tuple[int, int]:
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


def stripe_ratio_label(cu_rows: int, au_rows: int) -> str:
    return f"{int(cu_rows)}:{int(au_rows)}"


def region_options_for_adsorbate(adsorbate: str) -> List[str]:
    if not adsorbate or adsorbate == "clean":
        return ["Cu_side_interface"]
    return [region for region, ads_list in ADSORPTION_REGION_ADSORBATES.items() if adsorbate in ads_list] or ADSORPTION_REGION_CHOICES


def _safe_first_region(adsorbate: str, requested: Optional[str]) -> str:
    options = region_options_for_adsorbate(adsorbate)
    return requested if requested in options else options[0]


def is_all_region_request(text: str) -> bool:
    low = str(text or "").lower()
    wants_all = bool(re.search(r"\b(all|every|full|complete)\b", low))
    mentions_region = "adsorption region" in low or "adsorption regions" in low or "region" in low
    mentions_cu_au = "cu/au" in low or "cu-au" in low or ("cu" in low and "au" in low)
    mentions_generate = bool(re.search(r"\b(generate|make|build|create)\b", low))
    mentions_possible = "possible" in low or "combination" in low or "iterate" in low or "screen" in low
    return mentions_generate and mentions_cu_au and mentions_region and (wants_all or mentions_possible)


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
    sg.INTERFACE_GENERATOR_UPDATE_TAG = "v5_2026-07-08_stripe_ratios_adsorption_regions"

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
) -> Tuple[int, int, int, int]:
    requested_rows_per_side = max(1, int(requested_rows_per_side))
    if element1_rows is not None or element2_rows is not None:
        e1_rows = max(1, int(element1_rows or requested_rows_per_side))
        e2_rows = max(1, int(element2_rows or requested_rows_per_side))
    else:
        if cu_rows is None or au_rows is None:
            cu_rows_parsed, au_rows_parsed = parse_stripe_ratio(stripe_ratio, (requested_rows_per_side, requested_rows_per_side))
            cu_rows = cu_rows if cu_rows is not None else cu_rows_parsed
            au_rows = au_rows if au_rows is not None else au_rows_parsed
        cu_rows = max(1, int(cu_rows))
        au_rows = max(1, int(au_rows))
        if element1 == "Cu":
            e1_rows = cu_rows
        elif element1 == "Au":
            e1_rows = au_rows
        else:
            e1_rows = requested_rows_per_side
        if element2 == "Cu":
            e2_rows = cu_rows
        elif element2 == "Au":
            e2_rows = au_rows
        else:
            e2_rows = requested_rows_per_side
    final_cu_rows = e1_rows if element1 == "Cu" else (e2_rows if element2 == "Cu" else 0)
    final_au_rows = e1_rows if element1 == "Au" else (e2_rows if element2 == "Au" else 0)
    return e1_rows, e2_rows, final_cu_rows, final_au_rows


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
    e1_rows, e2_rows, cu_rows_final, au_rows_final = _rows_for_elements(
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

    ratio = stripe_ratio_label(cu_rows_final, au_rows_final) if cu_rows_final and au_rows_final else f"{e1_rows}:{e2_rows}"
    descriptors = {
        "cu_rows": cu_rows_final,
        "au_rows": au_rows_final,
        "stripe_ratio": ratio,
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


def _site_index_for_region(self, atoms: List[str], positions: np.ndarray, cell: np.ndarray, top_indices: np.ndarray, metadata: Dict[str, Any], region: str) -> Tuple[int, Optional[str], int]:
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    split_axis_name = interface_meta.get("split_axis", "y")
    split_axis = 1 if split_axis_name == "y" else 0
    line_axis = 0 if split_axis == 1 else 1
    split_coord = float(interface_meta.get("split_coordinate", cell[split_axis, split_axis] / 2.0))
    line_length = float(cell[line_axis, line_axis]) if cell[line_axis, line_axis] else None
    ref_line = 0.5 * line_length if line_length else 0.0
    element, row_offset = SINGLE_SITE_REGIONS.get(region, ("Cu", 0))
    candidates = [idx for idx in top_indices if atoms[idx] == element]
    if not candidates:
        candidates = list(top_indices)
    rows = sorted(_top_row_values(positions, candidates, split_axis), key=lambda y: abs(y - split_coord))
    target_row = rows[min(row_offset, len(rows) - 1)] if rows else split_coord
    row_candidates = [idx for idx in candidates if abs(float(positions[idx, split_axis]) - target_row) < 1e-5] or candidates

    def score(idx: int) -> float:
        return self._periodic_delta(float(positions[idx, line_axis]), ref_line, line_length)

    chosen_global_idx = min(row_candidates, key=score)
    return list(top_indices).index(chosen_global_idx), element if element in atoms else None, row_offset


def _patched_add_adsorbate(self, structure: Dict[str, Any], adsorbate_name: str, *args: Any, adsorption_region: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    metadata = structure.get("metadata", {})
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    if interface_meta.get("type") != "lateral" or adsorbate_name == "clean":
        return self._stripe_original_add_adsorbate(structure, adsorbate_name, *args, **kwargs)

    requested_region = adsorption_region
    region = _safe_first_region(adsorbate_name, requested_region)
    if requested_region and requested_region != region:
        metadata.setdefault("adsorption_region_request_warning", f"Requested {requested_region} is not configured for {adsorbate_name}; using {region}.")

    positions = np.asarray(structure["positions"], dtype=float)
    atoms = list(structure["atoms"])
    cell = np.asarray(structure["cell"], dtype=float)
    top_indices = self._top_layer_indices(positions)
    binding_element = kwargs.pop("binding_element", None)
    site_index = kwargs.pop("site_index", None)
    row_distance = 0
    binding_mode = "single_site"

    if region == "Cu_Au_boundary_bridge":
        binding_element = binding_element or "Cu"
        binding_mode = "dual_C_bridge"
        kwargs["site"] = "bridge_interface"
    else:
        site_index, inferred_binding_element, row_distance = _site_index_for_region(self, atoms, positions, cell, top_indices, metadata, region)
        binding_element = binding_element or inferred_binding_element

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
        if key in {"adsorption_region", "adsorption_binding_element", "distance_to_interface_row", "binding_mode", "cu_rows", "au_rows", "stripe_ratio"}:
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
    metadata["adsorbate_metadata"] = ads_meta


STRIPE_BATCH_HELPER_SOURCE = r'''

def _stripe_generate_all_adsorption_region_batch(params):
    """Generate all Cu/Au(111) stripe-ratio × adsorption-region × allowed-adsorbate structures."""
    params["structure_source"] = "Build from settings"
    params["structure_mode"] = "Interface"
    params["element1"] = "Cu"
    params["element2"] = "Au"
    params["facet"] = "111"
    params["batch_source"] = "Generate pathway from current structure setup"

    workspace = Path(params.get("batch_output_dir", "generated_outputs/web_xas_agent/batch")) / "_stripe_adsorption_region_workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    generator = StructureGenerator()
    items = []
    combo_manifest = []

    for ratio in stripe_patch_ratio_choices():
        for region, adsorbates in stripe_patch_region_adsorbates().items():
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
                    binding_element="Cu" if region != "Au_side_interface" else "Au",
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
                combo_manifest.append({"name": name, "stripe_ratio": ratio, "adsorption_region": region, "adsorbate": ads})

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
        '            with c2:\n                p["ny"] = st.number_input("Rows fallback", min_value=1, max_value=30, value=int(p["ny"]), help="Used only if no Cu:Au stripe ratio is selected.")\n            ratio_options = stripe_patch_ratio_choices()\n            p["stripe_ratio"] = st.selectbox("Cu:Au stripe row ratio", ratio_options, index=safe_index(ratio_options, p.get("stripe_ratio", "1:1")))\n            cu_rows, au_rows = stripe_patch_parse_ratio(p["stripe_ratio"])\n            st.caption(f"Stripe width: Cu rows = {cu_rows}, Au rows = {au_rows}; interface axis x, split axis y.")\n\n        p["facet"] = st.selectbox("Facet", ["111"], index=0)',
    )
    source = source.replace(
        '            p["interface_binding_element"] = st.selectbox(\n                "Adsorbate binding site metal",\n                binding_options,\n                index=safe_index(binding_options, p["interface_binding_element"]),\n            )',
        '            p["interface_binding_element"] = st.selectbox(\n                "Adsorbate binding site metal",\n                binding_options,\n                index=safe_index(binding_options, p["interface_binding_element"]),\n            )\n            region_options = stripe_patch_region_options(p["adsorbate"])\n            if p.get("adsorption_region") not in region_options:\n                p["adsorption_region"] = region_options[0]\n            p["adsorption_region"] = st.selectbox(\n                "Adsorption region",\n                region_options,\n                index=safe_index(region_options, p.get("adsorption_region", region_options[0])),\n                help="Stored in structure_info.metadata.descriptors.adsorption_region for ISAAC/ML.",\n            )',
    )
    source = source.replace(
        '    return parsed\n\n\ndef generate_structure',
        '    stripe_match = re.search(r"(?:cu\\s*[:/]\\s*au\\s*)?(1|2|3)\\s*:\\s*(1|2|3)", text)\n    if stripe_match:\n        parsed["stripe_ratio"] = f"{stripe_match.group(1)}:{stripe_match.group(2)}"\n    for region in stripe_patch_region_choices():\n        if region.lower() in text.replace("-", "_"):\n            parsed["adsorption_region"] = region\n            parsed.setdefault("structure_mode", "Interface")\n            parsed.setdefault("element1", "Cu")\n            parsed.setdefault("element2", "Au")\n            break\n    if "stripe_ratio" in parsed:\n        parsed.setdefault("structure_mode", "Interface")\n        parsed.setdefault("element1", "Cu")\n        parsed.setdefault("element2", "Au")\n\n    return parsed\n\n\ndef generate_structure',
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
        '    low = str(text or "").lower()\n    plan = plan or {}\n\n    if stripe_patch_is_all_region_request(text):\n        result = _stripe_generate_all_adsorption_region_batch(params)\n        combo = result.get("stripe_adsorption_region_grid", {})\n        return (\n            f"Generated Cu/Au(111) stripe adsorption-region batch with `{combo.get(\'n_combinations\', result.get(\'n_structures\'))}` structures at `{result.get(\'run_dir\')}`.\\n"\n            f"Stripe ratios: `{\', \'.join(combo.get(\'stripe_ratios\', []))}`.\\n"\n            "This batch iterates all configured adsorption regions and allowed adsorbates. Next: upload to NERSC, or download the batch ZIP from the Batch generation panel."\n        )\n',
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
        "stripe_patch_is_all_region_request": is_all_region_request,
    }
    exec(compile(source, str(source_path), "exec"), globals_dict)
