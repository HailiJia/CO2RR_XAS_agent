from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools import stripe_interface_patch as stripe

LOCAL_PLANNER_LABEL = "Local deterministic planner"
METAL_SYMBOLS = ("Cu", "Au", "Ni", "Ag", "Pt", "Pd", "Ir", "Rh", "Al")
ADSORBATE_TOKENS = (
    "OCCO", "COCO", "CHOH", "CHO", "CO2", "CO", "OH", "CH3", "CH2", "CH", "H", "O"
)


def install() -> None:
    """Install the existing Cu/Au stripe and adsorption-region implementation."""
    stripe.install()


def _reduce_ratio(first: int, second: int) -> tuple[int, int]:
    first = max(1, int(first))
    second = max(1, int(second))
    divisor = math.gcd(first, second)
    return first // divisor, second // divisor


def requested_cu_au_ratios(text: str, current_ratio: str = "1:1") -> List[str]:
    """Return requested ratios in the existing generator's Cu:Au convention.

    The prompt may use either Cu:Au or Au:Cu ordering. For example,
    ``Au:Cu=1:2`` is normalized to ``Cu:Au=2:1``.
    """
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
        return list(stripe.STRIPE_RATIO_CHOICES)
    if not pairs:
        match = re.search(r"(\d+)\s*:\s*(\d+)", str(current_ratio or "1:1"))
        pairs = [(int(match.group(1)), int(match.group(2)))] if match else [(1, 1)]
        requested_order = "Cu:Au"

    ratios: List[str] = []
    for first, second in pairs:
        if requested_order == "Au:Cu":
            au_units, cu_units = _reduce_ratio(first, second)
        else:
            cu_units, au_units = _reduce_ratio(first, second)
        ratio = f"{cu_units}:{au_units}"
        if ratio not in ratios:
            ratios.append(ratio)
    return ratios


def is_all_site_request(text: str) -> bool:
    """Recognize all-intermediate/all-site Cu/Au enumeration requests."""
    low = str(text or "").lower()
    has_generate = bool(re.search(r"\b(generate|make|build|create|screen|enumerate)\b", low))
    has_cu_au = bool(re.search(r"\bcu\b", low) and re.search(r"\bau\b", low))
    has_intermediates = bool(re.search(r"\b(intermediate|intermediates|pathway|adsorbate|adsorbates)\b", low))
    has_sites = bool(re.search(r"\b(site|sites|region|regions|binding\s+sites?)\b", low))
    has_all = bool(re.search(r"\b(all|every|possible|complete|full)\b", low))
    return has_generate and has_cu_au and has_intermediates and has_sites and has_all


def _contains_adsorbate(text: str) -> bool:
    low = str(text or "").lower()
    return any(
        re.search(r"(?<![A-Za-z0-9])\*?" + re.escape(token.lower()) + r"(?![A-Za-z0-9])", low)
        for token in ADSORBATE_TOKENS
    )


def should_generate_structure(text: str) -> bool:
    """Treat concise chemistry imperatives as structure-generation actions."""
    low = str(text or "").lower()
    if is_all_site_request(text):
        return False
    if not re.search(r"\b(generate|create|build|make)\b", low):
        return False
    if re.search(r"\b(batch|pathway|workflow|relaxation|relax|xas)\b", low):
        return False
    has_metal = any(re.search(rf"\b{symbol.lower()}\b", low) for symbol in METAL_SYMBOLS)
    return has_metal and _contains_adsorbate(text)


def parse_prompt_metals(text: str, parsed: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Parse all UI-supported metals while preserving prompt order."""
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


def patched_main_source(source: str) -> str:
    """Extend the existing stripe workflow without duplicating its batch generator."""
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

    # Tell optional LLM planners about the existing stripe/site workflow. The
    # deterministic executor guard below remains authoritative if the model
    # returns only the generic pathway intent.
    source = source.replace(
        '"For \'full CO2RR reaction pathway\', use generate_batch_pathway and pathway_adsorbates "',
        '"For requests to generate all possible intermediates at all possible sites or regions on Cu/Au with specified ratios, use generate_batch_pathway; the local executor expands the existing stripe workflow. "\n        "For \'full CO2RR reaction pathway\', use generate_batch_pathway and pathway_adsorbates "',
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
        + '    if chat_patch_is_all_site_request(message):\n'
        + '        return "generate_batch_pathway"\n',
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

    # Parameterize the existing stripe batch helper. No second batch generator
    # or adsorption-placement implementation is introduced here.
    source = source.replace(
        'def _stripe_generate_all_adsorption_region_batch(params):',
        'def _stripe_generate_all_adsorption_region_batch(params, text=""):',
        1,
    )
    source = source.replace(
        '    for ratio in stripe_patch_ratio_choices():',
        '    selected_ratios = chat_patch_requested_cu_au_ratios(text, params.get("stripe_ratio", "1:1"))\n'
        '    for ratio in selected_ratios:',
        1,
    )
    source = source.replace(
        '        "stripe_ratios": stripe_patch_ratio_choices(),',
        '        "stripe_ratios": selected_ratios,',
        1,
    )
    source = source.replace(
        'result = _stripe_generate_all_adsorption_region_batch(params)',
        'result = _stripe_generate_all_adsorption_region_batch(params, text)',
        1,
    )
    source = source.replace(
        '            "Invalid row_1/row_2 combinations are skipped automatically. Next: upload to NERSC, or download the batch ZIP from the Batch generation panel."',
        '            "Generated with the existing stripe-ratio and adsorption-region definitions, including Cu-side, Au-side, along-interface, across-interface, and valid dimer placements. Invalid ratio/region combinations are skipped automatically. Next: upload to NERSC, or download the batch ZIP from the Batch generation panel."',
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
        "stripe_patch_is_all_region_request": is_all_site_request,
        "chat_patch_is_all_site_request": is_all_site_request,
        "chat_patch_requested_cu_au_ratios": requested_cu_au_ratios,
        "chat_patch_should_generate_structure": should_generate_structure,
        "chat_patch_parse_prompt_metals": parse_prompt_metals,
    }
    exec(compile(source, str(source_path), "exec"), globals_dict)
