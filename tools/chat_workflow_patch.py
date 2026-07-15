from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools import stripe_interface_patch as stripe

LOCAL_PLANNER_LABEL = "Local deterministic planner"
CRYSTALLOGRAPHIC_SITE_CHOICES = ("top", "bridge", "fcc", "hcp")
GRID_INTERMEDIATES = (
    "CO2",
    "CO",
    "CHO",
    "CHOH",
    "CH",
    "CH2",
    "CH3",
    "OH",
    "O",
    "H",
    "OCCO",
    "COCO",
)
EXTRA_SINGLE_SITE_INTERMEDIATES = ("CHOH", "CH", "CH2", "CH3")
CU_SINGLE_SITE_REGIONS = (
    "Cu_side_interface",
    "Cu_near_interface_row_1",
    "Cu_near_interface_row_2",
)
METAL_SYMBOLS = ("Cu", "Au", "Ni", "Ag", "Pt", "Pd", "Ir", "Rh", "Al")
ADSORBATE_TOKENS = (
    "OCCO",
    "COCO",
    "CHOH",
    "CHO",
    "CO2",
    "CO",
    "OH",
    "CH3",
    "CH2",
    "CH",
    "H",
    "O",
)


def install() -> None:
    """Install the existing stripe patch and extend its site compatibility table."""
    stripe.install()
    for region in CU_SINGLE_SITE_REGIONS:
        allowed = stripe.ADSORPTION_REGION_ADSORBATES.setdefault(region, [])
        for adsorbate in EXTRA_SINGLE_SITE_INTERMEDIATES:
            if adsorbate not in allowed:
                allowed.append(adsorbate)
    stripe.ADSORPTION_REGION_CHOICES[:] = list(stripe.ADSORPTION_REGION_ADSORBATES)


def _reduce_ratio(first: int, second: int) -> tuple[int, int]:
    first = max(1, int(first))
    second = max(1, int(second))
    divisor = math.gcd(first, second)
    return first // divisor, second // divisor


def parse_requested_ratios(text: str, current_ratio: str = "1:1") -> List[Dict[str, Any]]:
    """Parse requested composition ratios and normalize them to the generator's Cu:Au convention."""
    raw = str(text or "")
    low = raw.lower()
    order_match = re.search(r"\b(cu|au)\s*:\s*(cu|au)\b", low)
    requested_order = "Cu:Au"
    if order_match and order_match.group(1) != order_match.group(2):
        requested_order = f"{order_match.group(1).title()}:{order_match.group(2).title()}"

    pairs = [
        (int(first), int(second))
        for first, second in re.findall(r"(?<![A-Za-z0-9])(\d+)\s*:\s*(\d+)(?![A-Za-z0-9])", raw)
        if int(first) > 0 and int(second) > 0
    ]
    if not pairs and re.search(r"\b(all|every)\s+(?:possible\s+)?ratios?\b", low):
        pairs = [tuple(int(value) for value in ratio.split(":")) for ratio in stripe.STRIPE_RATIO_CHOICES]
        requested_order = "Cu:Au"
    if not pairs:
        match = re.search(r"(\d+)\s*:\s*(\d+)", str(current_ratio or "1:1"))
        pairs = [(int(match.group(1)), int(match.group(2)))] if match else [(1, 1)]
        requested_order = "Cu:Au"

    parsed: List[Dict[str, Any]] = []
    seen = set()
    for first, second in pairs:
        if requested_order == "Au:Cu":
            au_units, cu_units = _reduce_ratio(first, second)
        else:
            cu_units, au_units = _reduce_ratio(first, second)
        canonical = f"{cu_units}:{au_units}"
        if canonical in seen:
            continue
        seen.add(canonical)
        if requested_order == "Au:Cu":
            requested_label = f"Au:Cu={au_units}:{cu_units}"
        else:
            requested_label = f"Cu:Au={cu_units}:{au_units}"
        parsed.append(
            {
                "requested_order": requested_order,
                "requested_label": requested_label,
                "canonical_cu_au_ratio": canonical,
                "cu_units": cu_units,
                "au_units": au_units,
            }
        )
    return parsed


def is_site_ratio_grid_request(text: str) -> bool:
    low = str(text or "").lower()
    has_generate = bool(re.search(r"\b(generate|create|build|make|screen|enumerate)\b", low))
    has_bimetal = bool(re.search(r"\bcu\b", low) and re.search(r"\bau\b", low))
    has_intermediate_scope = bool(
        re.search(r"\b(intermediate|intermediates|pathway|adsorbate|adsorbates)\b", low)
    )
    has_site_scope = bool(re.search(r"\b(site|sites|region|regions|binding\s+sites?)\b", low))
    has_all_scope = bool(re.search(r"\b(all|every|possible|complete|full)\b", low))
    return has_generate and has_bimetal and has_intermediate_scope and has_site_scope and has_all_scope


def _contains_adsorbate(text: str) -> bool:
    low = str(text or "").lower()
    for adsorbate in ADSORBATE_TOKENS:
        pattern = r"(?<![A-Za-z0-9])\*?" + re.escape(adsorbate.lower()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, low):
            return True
    return False


def should_generate_structure(text: str) -> bool:
    low = str(text or "").lower()
    if is_site_ratio_grid_request(text):
        return False
    if not re.search(r"\b(generate|create|build|make)\b", low):
        return False
    if re.search(r"\b(batch|pathway|workflow|relaxation|relax|xas)\b", low):
        return False
    has_metal = any(re.search(rf"\b{symbol.lower()}\b", low) for symbol in METAL_SYMBOLS)
    return has_metal and _contains_adsorbate(text)


def parse_prompt_metals(text: str, parsed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Return metal settings while preserving the order used in the prompt."""
    result = dict(parsed or {})
    pattern = r"\b(" + "|".join(symbol.lower() for symbol in METAL_SYMBOLS) + r")\b"
    canonical = {symbol.lower(): symbol for symbol in METAL_SYMBOLS}
    metals: List[str] = []
    for match in re.finditer(pattern, str(text or "").lower()):
        symbol = canonical[match.group(1)]
        if symbol not in metals:
            metals.append(symbol)
    low = str(text or "").lower()
    interface_requested = len(metals) >= 2 and (
        "/" in low
        or "interface" in low
        or "bimetal" in low
        or "alloy" in low
        or "ratio" in low
        or re.search(r"\b(?:cu|au)\s*[-:]\s*(?:cu|au)\b", low)
    )
    if interface_requested:
        result["structure_mode"] = "Interface"
        result["element1"] = metals[0]
        result["element2"] = metals[1]
    elif metals:
        result["element"] = metals[0]
    return result


def grid_intermediates() -> List[str]:
    return list(GRID_INTERMEDIATES)


def sites_for_region(region: str) -> List[str]:
    if region in stripe.DIMER_REGIONS:
        return ["region_defined"]
    return list(CRYSTALLOGRAPHIC_SITE_CHOICES)


CHAT_BATCH_HELPER_SOURCE = r'''

def _chat_attach_site_ratio_metadata(structure, ratio_spec, region, site, adsorbate):
    metadata = structure.setdefault("metadata", {})
    interface_meta = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    descriptors = dict(metadata.get("descriptors") or {})
    descriptors.update({
        "requested_ratio_label": ratio_spec.get("requested_label"),
        "requested_ratio_order": ratio_spec.get("requested_order"),
        "canonical_cu_au_ratio": ratio_spec.get("canonical_cu_au_ratio"),
        "cu_rows": metadata.get("cu_rows") or interface_meta.get("cu_rows"),
        "au_rows": metadata.get("au_rows") or interface_meta.get("au_rows"),
        "adsorption_region": region,
        "crystallographic_site": site,
        "adsorbate_identity": adsorbate,
    })
    metadata["descriptors"] = descriptors
    metadata["requested_ratio_label"] = ratio_spec.get("requested_label")
    metadata["canonical_cu_au_ratio"] = ratio_spec.get("canonical_cu_au_ratio")
    metadata["crystallographic_site"] = site
    if region is not None:
        metadata["adsorption_region"] = region
    ads_meta = dict(metadata.get("adsorbate_metadata") or {})
    ads_meta.update({
        "identity": adsorbate,
        "adsorption_region": region,
        "crystallographic_site": site,
    })
    metadata["adsorbate_metadata"] = ads_meta


def _chat_generate_requested_site_ratio_batch(params, text, plan=None):
    """Expand requested ratios × intermediates × valid regions × crystallographic sites."""
    ratio_specs = chat_patch_parse_requested_ratios(text, params.get("stripe_ratio", "1:1"))
    params["structure_source"] = "Build from settings"
    params["structure_mode"] = "Interface"
    params["element1"] = "Cu"
    params["element2"] = "Au"
    params["facet"] = "111"
    params["batch_source"] = "Generate pathway from current structure setup"

    generator = StructureGenerator()
    items = []
    combinations = []

    def make_base(canonical_ratio):
        return generator.generate_interface(
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
            stripe_ratio=canonical_ratio,
        )

    for ratio_spec in ratio_specs:
        canonical_ratio = ratio_spec["canonical_cu_au_ratio"]
        clean = make_base(canonical_ratio)
        _chat_attach_site_ratio_metadata(clean, ratio_spec, None, "clean", "clean")
        clean_name = sanitize_name(
            f"Cu_Au_111_ratio_{canonical_ratio.replace(':', 'to')}_clean"
        )
        items.append({
            "name": clean_name,
            "source": "requested_ratio_site_grid",
            "structure": clean,
            "substrate_n": len(clean.get("atoms", [])),
            "source_poscar": None,
        })
        combinations.append({
            "name": clean_name,
            **ratio_spec,
            "adsorbate": "clean",
            "adsorption_region": None,
            "crystallographic_site": "clean",
        })

        for adsorbate in chat_patch_grid_intermediates():
            regions = stripe_patch_region_options(adsorbate, canonical_ratio)
            for region in regions:
                for site in chat_patch_sites_for_region(region):
                    base = make_base(canonical_ratio)
                    structure = generator.add_adsorbate(
                        base,
                        adsorbate,
                        site="auto" if site == "region_defined" else site,
                        height=float(params.get("height", 2.0)),
                        adsorption_region=region,
                    )
                    _chat_attach_site_ratio_metadata(structure, ratio_spec, region, site, adsorbate)
                    name = sanitize_name(
                        f"Cu_Au_111_ratio_{canonical_ratio.replace(':', 'to')}_{region}_{site}_{adsorbate}"
                    )
                    items.append({
                        "name": name,
                        "source": "requested_ratio_site_grid",
                        "structure": structure,
                        "substrate_n": len(base.get("atoms", [])),
                        "source_poscar": None,
                    })
                    combinations.append({
                        "name": name,
                        **ratio_spec,
                        "adsorbate": adsorbate,
                        "adsorption_region": region,
                        "crystallographic_site": site,
                    })

    batch_result = generate_batch_input_package(
        batch_items=items,
        output_dir=params.get("batch_output_dir", "generated_outputs/web_xas_agent/batch"),
        params=params,
        include_relaxation=params.get("batch_include_relaxation", True),
        include_xas=params.get("batch_include_xas", True),
        absorber_mode=params.get("batch_absorber_mode", "auto"),
    )
    grid = {
        "requested_ratios": ratio_specs,
        "canonical_cu_au_ratios": [item["canonical_cu_au_ratio"] for item in ratio_specs],
        "intermediates": chat_patch_grid_intermediates(),
        "crystallographic_sites": ["top", "bridge", "fcc", "hcp", "region_defined"],
        "n_combinations": len(combinations),
        "combinations": combinations,
    }
    batch_result["requested_site_ratio_grid"] = grid
    manifest_path = Path(batch_result["run_dir"]) / "requested_site_ratio_manifest.json"
    manifest_path.write_text(json.dumps(grid, indent=2))
    st.session_state["last_batch_result"] = batch_result
    st.session_state["last_generated_package_kind"] = "batch"
    params["nersc_upload_source"] = "Last batch package"
    return batch_result
'''


def patched_main_source(source: str) -> str:
    """Apply the existing stripe patch, then improve chat routing and grid expansion."""
    source = stripe.patched_main_source(source)

    replacements = {
        '"chat_backend": "Rule-based"': f'"chat_backend": "{LOCAL_PLANNER_LABEL}"',
        'BACKEND_CHOICES = ["Rule-based", "Argonne ALCF", "OpenAI", "Anthropic"]': (
            f'BACKEND_CHOICES = ["{LOCAL_PLANNER_LABEL}", "Argonne ALCF", "OpenAI", "Anthropic"]'
        ),
        'p.get("chat_backend", "Rule-based")': f'p.get("chat_backend", "{LOCAL_PLANNER_LABEL}")',
        'params.get("chat_backend", "Rule-based")': f'params.get("chat_backend", "{LOCAL_PLANNER_LABEL}")',
        'planner_used = "Rule-based"': f'planner_used = "{LOCAL_PLANNER_LABEL}"',
        '"Rule-based forced full workflow"': f'"{LOCAL_PLANNER_LABEL} forced full workflow"',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    source = source.replace(
        '"Return only one JSON object. Do not execute anything. Valid intents: "\n        "generate_structure,',
        '"Return only one JSON object. Do not execute anything. Valid intents: "\n        "generate_site_ratio_grid, generate_structure,',
    )
    source = source.replace(
        '"For \'full CO2RR reaction pathway\', use generate_batch_pathway and pathway_adsorbates "',
        '"For requests to generate all possible intermediates at all possible sites/regions on Cu/Au with one or more composition ratios, use generate_site_ratio_grid. "\n        "For \'full CO2RR reaction pathway\', use generate_batch_pathway and pathway_adsorbates "',
    )
    source = source.replace(
        '"Allowed JSON fields: intent, settings, pathway_adsorbates, remote_run_name, remote_dir, preview_file, explanation. "',
        '"Allowed JSON fields: intent, settings, pathway_adsorbates, stripe_ratios, adsorption_sites, remote_run_name, remote_dir, preview_file, explanation. "',
    )

    source = source.replace(
        '    return parsed\n\n\ndef generate_structure',
        '    parsed = chat_patch_parse_prompt_metals(text, parsed)\n\n    return parsed\n\n\ndef generate_structure',
        1,
    )

    classifier_anchor = (
        '    full_workflow_override = _full_workflow_intent_from_text(message)\n'
        '    if full_workflow_override:\n'
        '        return full_workflow_override\n'
    )
    source = source.replace(
        classifier_anchor,
        classifier_anchor
        + '    if chat_patch_is_site_ratio_grid_request(message):\n'
        + '        return "generate_site_ratio_grid"\n',
        1,
    )
    metal_fallback = (
        '    if re.search(r"\\b(cu|au|ni|ag|pt|pd|ir|rh|al)\\b", low) or re.search(r"\\d+\\s*[x×*]\\s*\\d+", low):\n'
        '        return "update_settings"\n'
    )
    source = source.replace(
        metal_fallback,
        '    if chat_patch_should_generate_structure(message):\n'
        '        return "generate_structure"\n'
        + metal_fallback,
        1,
    )

    source = source.replace(
        'def build_uploaded_batch_items(uploaded_files, workspace):',
        CHAT_BATCH_HELPER_SOURCE + '\n\ndef build_uploaded_batch_items(uploaded_files, workspace):',
        1,
    )

    executor_anchor = (
        '    low = str(text or "").lower()\n'
        '    plan = plan or {}\n\n'
        '    if stripe_patch_is_all_region_request(text):\n'
    )
    executor_replacement = (
        '    low = str(text or "").lower()\n'
        '    plan = plan or {}\n\n'
        '    if intent == "generate_site_ratio_grid" or chat_patch_is_site_ratio_grid_request(text):\n'
        '        result = _chat_generate_requested_site_ratio_batch(params, text, plan=plan)\n'
        '        grid = result.get("requested_site_ratio_grid", {})\n'
        '        requested = ", ".join(item.get("requested_label", "") for item in grid.get("requested_ratios", []))\n'
        '        canonical = ", ".join(grid.get("canonical_cu_au_ratios", []))\n'
        '        return (\n'
        '            f"Generated `{grid.get(\'n_combinations\', result.get(\'n_structures\'))}` Cu/Au(111) structures at `{result.get(\'run_dir\')}`.\\n"\n'
        '            f"Requested ratios: `{requested}`; internal Cu:Au ratios: `{canonical}`.\\n"\n'
        '            "Expanded intermediates across every valid adsorption region and top/bridge/fcc/hcp sites; dimer intermediates use region-defined dual-C sites. "\n'
        '            "A requested_site_ratio_manifest.json file records every combination."\n'
        '        )\n\n'
        '    if stripe_patch_is_all_region_request(text):\n'
    )
    source = source.replace(executor_anchor, executor_replacement, 1)

    trace_anchor = (
        '    else:\n'
        '        intent = _classify_chat_intent_rule_based(text)\n'
        f'        planner_used = "{LOCAL_PLANNER_LABEL}"\n\n'
        '    st.session_state["last_chat_planner_trace"] = {\n'
    )
    source = source.replace(
        trace_anchor,
        '    else:\n'
        '        intent = _classify_chat_intent_rule_based(text)\n'
        f'        planner_used = "{LOCAL_PLANNER_LABEL}"\n\n'
        '    if chat_patch_is_site_ratio_grid_request(text):\n'
        '        intent = "generate_site_ratio_grid"\n\n'
        '    st.session_state["last_chat_planner_trace"] = {\n',
        1,
    )

    source = source.replace(
        '            "- `Cu 3x3 with CO`\\n"\n',
        '            "- `Cu 3x3 with CO`\\n"\n'
        '            "- `generate CO on Cu`\\n"\n'
        '            "- `generate all possible intermediates with all possible sites for Au/Cu, use ratio Au:Cu=1:1 and 1:2`\\n"\n',
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
        "stripe_patch_ratio_choices": lambda: stripe.STRIPE_RATIO_CHOICES,
        "stripe_patch_region_choices": lambda: stripe.ADSORPTION_REGION_CHOICES,
        "stripe_patch_region_options": stripe.region_options_for_adsorbate,
        "stripe_patch_region_adsorbates": lambda: stripe.ADSORPTION_REGION_ADSORBATES,
        "stripe_patch_parse_ratio": stripe.parse_stripe_ratio,
        "stripe_patch_actual_rows": stripe.actual_rows_from_ratio,
        "stripe_patch_region_valid": stripe.region_is_valid_for_ratio,
        "stripe_patch_binding_element_for_region": stripe.binding_element_for_region,
        "stripe_patch_is_all_region_request": stripe.is_all_region_request,
        "chat_patch_is_site_ratio_grid_request": is_site_ratio_grid_request,
        "chat_patch_parse_requested_ratios": parse_requested_ratios,
        "chat_patch_should_generate_structure": should_generate_structure,
        "chat_patch_parse_prompt_metals": parse_prompt_metals,
        "chat_patch_grid_intermediates": grid_intermediates,
        "chat_patch_sites_for_region": sites_for_region,
    }
    exec(compile(source, str(source_path), "exec"), globals_dict)
