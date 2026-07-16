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


def explicit_adsorbate(text: str) -> Optional[str]:
    """Return only an explicitly written adsorbate token.

    Boundary-safe matching prevents O from being inferred from words such as
    ``to`` and prevents CO from being inferred from unrelated words.
    """
    low = str(text or "").lower()
    if re.search(r"\bclean(?:\s+(?:surface|slab))?\b", low):
        return "clean"
    for token in ADSORBATE_TOKENS:
        pattern = r"(?<![A-Za-z0-9])\*?" + re.escape(token.lower()) + r"(?![A-Za-z0-9])"
        if re.search(pattern, low):
            return token
    return None


def requested_cu_au_ratios(text: str, current_ratio: str = "1:1") -> List[str]:
    """Return requested ratios in the existing generator's Cu:Au convention.

    The prompt may use either Cu:Au or Au:Cu ordering. For example,
    ``Au:Cu=1:2`` is normalized to ``Cu:Au=2:1``. Returned values are passed
    directly to the pre-existing stripe batch generator.
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
    asks_all_ratios = (
        "ratio" in low
        and bool(re.search(r"\b(all|every|possible|complete|full)\b", low))
    )
    if not pairs and asks_all_ratios:
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


def explicit_cu_au_ratio(text: str) -> Optional[str]:
    """Return one explicitly requested ratio, normalized to Cu:Au."""
    raw = str(text or "")
    low = raw.lower()
    if not re.search(r"(?<![A-Za-z0-9])\d+\s*:\s*\d+(?![A-Za-z0-9])", raw):
        return None
    if "ratio" not in low and not re.search(r"\b(?:cu|au)\s*:\s*(?:cu|au)\b", low):
        return None
    return requested_cu_au_ratios(raw)[0]


def is_all_site_request(text: str) -> bool:
    """Recognize Cu/Au enumeration requests using sites, regions, or CO2RR wording."""
    low = str(text or "").lower()
    has_generate = bool(re.search(r"\b(generate|make|build|create|screen|enumerate)\b", low))
    has_cu_au = bool(re.search(r"\bcu\b", low) and re.search(r"\bau\b", low))
    has_intermediates = bool(re.search(r"\b(intermediate|intermediates|pathway|adsorbate|adsorbates)\b", low))
    has_co2rr = bool(re.search(r"\bco2rr\b|co\s*2\s*rr", low))
    has_ratio_scope = "ratio" in low
    has_sites = bool(re.search(r"\b(site|sites|region|regions|binding\s+sites?)\b", low))
    has_all = bool(re.search(r"\b(all|every|possible|complete|full)\b", low))
    return has_generate and has_cu_au and has_sites and has_all and (
        has_intermediates or has_co2rr or has_ratio_scope
    )


def _contains_adsorbate(text: str) -> bool:
    return explicit_adsorbate(text) not in (None, "clean")


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


def natural_language_adsorption_region(
    text: str,
    stripe_ratio: str = "1:1",
    adsorbate: str = "CO",
) -> Optional[str]:
    """Map user-facing site descriptions to the existing region vocabulary."""
    low = str(text or "").lower()
    normalized = re.sub(r"[\s\-/]+", "_", low)

    # Canonical names remain accepted directly.
    for region in stripe.ADSORPTION_REGION_CHOICES:
        if region.lower() in normalized:
            return region

    # Explicit row distances take precedence over generic Cu-side wording.
    row2 = re.search(
        r"\b(?:cu\s+side\s+)?(?:row\s*)?(?:2|two)\s+rows?\s+"
        r"(?:from|to|away\s+from)\s+(?:the\s+)?interface\b",
        low,
    )
    if row2 or re.search(r"\bcu\s+(?:near\s+)?interface\s+row\s*2\b", low):
        return "Cu_near_interface_row_2"

    row1 = re.search(
        r"\b(?:cu\s+side\s+)?(?:row\s*)?(?:1|one)\s+rows?\s+"
        r"(?:from|to|away\s+from)\s+(?:the\s+)?interface\b",
        low,
    )
    if row1 or re.search(r"\bcu\s+(?:near\s+)?interface\s+row\s*1\b", low):
        return "Cu_near_interface_row_1"

    if re.search(
        r"\b(cu\s+site\s+interface|cu\s+interface\s+site|cu\s+side\s+interface|"
        r"cu\s+at\s+(?:the\s+)?interface|on\s+cu\s+at\s+(?:the\s+)?interface)\b",
        low,
    ):
        return "Cu_side_interface"

    if re.search(
        r"\b(au\s+site\s+interface|au\s+interface\s+site|au\s+side\s+interface|"
        r"au\s+at\s+(?:the\s+)?interface)\b",
        low,
    ):
        return "Au_side_interface"

    if re.search(r"\b(across|bridge|bridging)\s+(?:the\s+)?(?:cu\s*[-/]\s*au\s+)?interface\b", low):
        return "Cu_Au_boundary_bridge"

    if re.search(r"\balong\s+(?:the\s+)?interface\b", low):
        if adsorbate in {"OCCO", "COCO"}:
            return "Cu_side_interface_dimer_along_x"
        return "Cu_side_interface"

    # "Cu side" means an interior/pure-Cu row, not the Cu boundary row. Select
    # the farthest implemented valid row. If the ratio has no interior Cu row,
    # return row_1 so the existing validator raises a clear incompatibility.
    if re.search(r"\b(?:pure\s+|bulk\s+)?cu\s+side\b", low):
        options = stripe.region_options_for_adsorbate(adsorbate, stripe_ratio)
        for region in ("Cu_near_interface_row_2", "Cu_near_interface_row_1"):
            if region in options:
                return region
        return "Cu_near_interface_row_1"

    if re.search(r"\bau\s+side\b", low):
        return "Au_side_interface"

    return None


def resolve_text_settings(
    text: str,
    params: Optional[Dict[str, Any]],
    parsed: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Apply deterministic chemistry parsing after either local or LLM planning."""
    current = params or {}
    result = parse_prompt_metals(text, dict(parsed or {}))

    adsorbate = explicit_adsorbate(text)
    if adsorbate is not None:
        result["adsorbate"] = adsorbate
    else:
        # Never let substring matching or an LLM change the adsorbate when the
        # user only changes a ratio/site or says "generate the structure".
        result.pop("adsorbate", None)

    ratio = explicit_cu_au_ratio(text)
    if ratio is not None:
        result["stripe_ratio"] = ratio

    effective_adsorbate = str(result.get("adsorbate") or current.get("adsorbate") or "CO")
    effective_ratio = str(result.get("stripe_ratio") or current.get("stripe_ratio") or "1:1")
    region = natural_language_adsorption_region(text, effective_ratio, effective_adsorbate)
    if region is not None:
        result["adsorption_region"] = region
        result["interface_binding_element"] = stripe.binding_element_for_region(region) or "Cu"
        result.setdefault("structure_mode", "Interface")
        result.setdefault("element1", current.get("element1", "Cu"))
        result.setdefault("element2", current.get("element2", "Au"))

    return result


def sanitize_plan_settings(
    text: str,
    params: Optional[Dict[str, Any]],
    settings: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Keep LLM settings only when supported by explicit user language."""
    result = resolve_text_settings(text, params, settings)
    if explicit_cu_au_ratio(text) is None:
        result.pop("stripe_ratio", None)
    effective_adsorbate = str(result.get("adsorbate") or (params or {}).get("adsorbate") or "CO")
    effective_ratio = str(result.get("stripe_ratio") or (params or {}).get("stripe_ratio") or "1:1")
    if natural_language_adsorption_region(text, effective_ratio, effective_adsorbate) is None:
        result.pop("adsorption_region", None)
        result.pop("interface_binding_element", None)
    return result


def patched_main_source(source: str) -> str:
    """Extend the existing stripe workflow without duplicating its generator."""
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
        '"For \'full CO2RR reaction pathway\', use generate_batch_pathway and pathway_adsorbates "',
        '"For requests to generate all possible intermediates, sites, regions, or ratios on Cu/Au in CO2RR, use generate_batch_pathway; the local executor expands the existing stripe workflow. "\n'
        '        "For single Cu/Au structures, preserve explicit site language in settings: Cu site interface means Cu_side_interface; Cu side means an interior Cu row; N rows from the interface means Cu_near_interface_row_N. "\n'
        '        "For \'full CO2RR reaction pathway\', use generate_batch_pathway and pathway_adsorbates "',
    )

    source = source.replace(
        '    return parsed\n\n\ndef generate_structure',
        '    parsed = chat_patch_parse_prompt_metals(text, parsed)\n\n    return parsed\n\n\ndef generate_structure',
        1,
    )

    # Replace substring-derived adsorbates with boundary-safe explicit tokens.
    source = source.replace(
        '    return parsed\n\n\ndef _extract_json_object',
        '    explicit_adsorbate = chat_patch_explicit_adsorbate(text)\n'
        '    if allow_adsorbate:\n'
        '        if explicit_adsorbate is None:\n'
        '            parsed.pop("adsorbate", None)\n'
        '        else:\n'
        '            parsed["adsorbate"] = explicit_adsorbate\n'
        '    else:\n'
        '        parsed.pop("adsorbate", None)\n'
        '    return parsed\n\n\ndef _extract_json_object',
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

    # Sanitize optional LLM settings before they touch persistent session state.
    plan_block = (
        '    if plan.get("settings") and isinstance(plan.get("settings"), dict):\n'
        '        for key, value in plan["settings"].items():\n'
        '            if key in params and value not in (None, ""):\n'
        '                params[key] = value\n'
    )
    source = source.replace(
        plan_block,
        '    if plan.get("settings") and isinstance(plan.get("settings"), dict):\n'
        '        safe_plan_settings = chat_patch_sanitize_plan_settings(text, params, plan["settings"])\n'
        '        for key, value in safe_plan_settings.items():\n'
        '            if key in params and value not in (None, ""):\n'
        '                params[key] = value\n',
        1,
    )

    # Apply ratio ordering and natural-language adsorption regions consistently
    # to structure generation, combined generation, workflow preparation, and
    # settings-only updates.
    source = source.replace(
        '        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)\n',
        '        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)\n'
        '        parsed = chat_patch_resolve_text_settings(text, params, parsed)\n',
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
        "chat_patch_explicit_adsorbate": explicit_adsorbate,
        "chat_patch_resolve_text_settings": resolve_text_settings,
        "chat_patch_sanitize_plan_settings": sanitize_plan_settings,
    }
    exec(compile(source, str(source_path), "exec"), globals_dict)
