from __future__ import annotations

import re
from pathlib import Path

from tools import chat_workflow_patch as chat

_CHAT_PATCHED_MAIN_SOURCE = chat.patched_main_source
_CHAT_IS_ALL_SITE_REQUEST = chat.is_all_site_request
_CHAT_REQUESTED_CU_AU_RATIOS = chat.requested_cu_au_ratios
_CHAT_NATURAL_LANGUAGE_ADSORPTION_REGION = chat.natural_language_adsorption_region
_CHAT_RESOLVE_TEXT_SETTINGS = chat.resolve_text_settings


def _compat_is_all_site_request(text: str) -> bool:
    """Restore broad Cu/Au enumeration phrases accepted by the earlier chat UI."""
    if _CHAT_IS_ALL_SITE_REQUEST(text):
        return True
    low = str(text or "").lower()
    has_generate = bool(re.search(r"\b(generate|make|build|create|screen|enumerate)\b", low))
    has_cu_au = bool(re.search(r"\bcu\b", low) and re.search(r"\bau\b", low))
    broad_scope = bool(
        re.search(
            r"\b(all\s+possibilit(?:y|ies)|every\s+possibility|all\s+combinations?|possible\s+combinations?)\b",
            low,
        )
    )
    return has_generate and has_cu_au and broad_scope


def _compat_requested_cu_au_ratios(text: str, current_ratio: str = "1:1"):
    """Use the full supported stripe grid for an unqualified 'all possibilities' request."""
    raw = str(text or "")
    has_explicit_ratio = bool(
        re.search(r"(?<![A-Za-z0-9])\d+\s*:\s*\d+(?![A-Za-z0-9])", raw)
    )
    if _compat_is_all_site_request(raw) and not has_explicit_ratio:
        return list(chat.stripe.STRIPE_RATIO_CHOICES)
    return _CHAT_REQUESTED_CU_AU_RATIOS(raw, current_ratio=current_ratio)


def _compat_natural_language_adsorption_region(
    text: str,
    stripe_ratio: str = "1:1",
    adsorbate: str = "CO",
):
    """Accept natural bridge wording such as 'OCCO bridge on Cu/Au'."""
    region = _CHAT_NATURAL_LANGUAGE_ADSORPTION_REGION(text, stripe_ratio, adsorbate)
    if region is not None:
        return region

    low = str(text or "").lower()
    is_dimer = str(adsorbate or "").upper() in {"OCCO", "COCO"}
    mentions_bridge = bool(re.search(r"\b(bridge|bridging|bridged)\b", low))
    mentions_cu_au = bool(
        re.search(r"\bcu\b", low) and re.search(r"\bau\b", low)
    )
    if is_dimer and mentions_bridge and mentions_cu_au:
        return "Cu_Au_boundary_bridge"
    return None


def _compat_resolve_text_settings(text, params, parsed):
    """Prevent a stale adsorption region from invalidating a newly requested adsorbate."""
    current = params or {}
    result = dict(_CHAT_RESOLVE_TEXT_SETTINGS(text, params, parsed) or {})

    explicit_adsorbate = chat.explicit_adsorbate(text)
    effective_adsorbate = str(result.get("adsorbate") or current.get("adsorbate") or "CO")
    effective_ratio = str(result.get("stripe_ratio") or current.get("stripe_ratio") or "1:1")
    region = _compat_natural_language_adsorption_region(
        text,
        effective_ratio,
        effective_adsorbate,
    )

    if region is not None:
        result["adsorption_region"] = region
        result["interface_binding_element"] = chat.stripe.binding_element_for_region(region) or "Cu"
        result.setdefault("structure_mode", "Interface")
        result.setdefault("element1", current.get("element1", "Cu"))
        result.setdefault("element2", current.get("element2", "Au"))
        return result

    # When the user explicitly changes the adsorbate, do not carry an invalid
    # region from the previous sidebar/chat state into the new structure.
    if explicit_adsorbate not in (None, "clean"):
        options = chat.stripe.region_options_for_adsorbate(explicit_adsorbate, effective_ratio)
        selected = result.get("adsorption_region") or current.get("adsorption_region")
        if options and selected not in options:
            default_region = options[0]
            result["adsorption_region"] = default_region
            result["interface_binding_element"] = chat.stripe.binding_element_for_region(default_region) or "Cu"

    return result


def patched_main_source(source: str) -> str:
    """Apply chat/stripe patches and remove user-specific NERSC defaults."""
    source = _CHAT_PATCHED_MAIN_SOURCE(source)

    old_defaults = (
        'DEFAULT_NERSC_ACCOUNT = os.environ.get("CO2RR_NERSC_ACCOUNT", os.environ.get("NERSC_ACCOUNT", "m5268"))\n'
        'DEFAULT_NERSC_REMOTE_BASE = os.environ.get("CO2RR_NERSC_REMOTE_BASE", "/pscratch/sd/h/hjia/CO2RR")\n'
    )
    new_defaults = (
        'DEFAULT_NERSC_USERNAME = (\n'
        '    os.environ.get("CO2RR_NERSC_USERNAME")\n'
        '    or os.environ.get("NERSC_USERNAME")\n'
        '    or os.environ.get("USER")\n'
        '    or os.environ.get("LOGNAME")\n'
        '    or Path.home().name\n'
        ').strip()\n'
        'if not DEFAULT_NERSC_USERNAME:\n'
        '    raise RuntimeError(\n'
        '        "Could not determine the NERSC username. Set CO2RR_NERSC_USERNAME explicitly."\n'
        '    )\n'
        'DEFAULT_NERSC_ACCOUNT = (\n'
        '    os.environ.get("CO2RR_NERSC_ACCOUNT")\n'
        '    or os.environ.get("NERSC_ACCOUNT")\n'
        '    or ""\n'
        ').strip()\n'
        'DEFAULT_NERSC_REMOTE_BASE = (\n'
        '    os.environ.get("CO2RR_NERSC_REMOTE_BASE")\n'
        '    or f"/pscratch/sd/{DEFAULT_NERSC_USERNAME[0].lower()}/{DEFAULT_NERSC_USERNAME}/CO2RR"\n'
        ')\n'
    )
    if old_defaults not in source:
        raise RuntimeError("Could not locate the NERSC default configuration block to patch.")
    source = source.replace(old_defaults, new_defaults, 1)

    old_delete_guard = (
        '    if remote_dir in {DEFAULT_NERSC_REMOTE_BASE, allowed_root, "/", "/pscratch", '
        '"/pscratch/sd", "/pscratch/sd/h", "/pscratch/sd/h/hjia"}:\n'
    )
    new_delete_guard = (
        '    nersc_user_root = str(PurePosixPath(DEFAULT_NERSC_REMOTE_BASE).parent)\n'
        '    nersc_shard_root = str(PurePosixPath(nersc_user_root).parent)\n'
        '    if remote_dir in {\n'
        '        DEFAULT_NERSC_REMOTE_BASE,\n'
        '        allowed_root,\n'
        '        "/",\n'
        '        "/pscratch",\n'
        '        "/pscratch/sd",\n'
        '        nersc_shard_root,\n'
        '        nersc_user_root,\n'
        '    }:\n'
    )
    if old_delete_guard not in source:
        raise RuntimeError("Could not locate the NERSC deletion safety guard to patch.")
    source = source.replace(old_delete_guard, new_delete_guard, 1)

    old_job_username = (
        '        username = str(params.get("nersc_username") or os.environ.get("USER") or "hjia").strip() or "hjia"\n'
    )
    new_job_username = (
        '        username = str(params.get("nersc_username") or DEFAULT_NERSC_USERNAME).strip()\n'
    )
    if old_job_username not in source:
        raise RuntimeError("Could not locate the NERSC job-list username fallback to patch.")
    source = source.replace(old_job_username, new_job_username, 1)

    return source


def run_patched_main(app_dir: Path) -> None:
    """Run the app with stripe, chat, and portable NERSC defaults installed."""
    original_source_patch = chat.patched_main_source
    original_all_site = chat.is_all_site_request
    original_ratios = chat.requested_cu_au_ratios
    original_region = chat.natural_language_adsorption_region
    original_resolve = chat.resolve_text_settings

    chat.patched_main_source = patched_main_source
    chat.is_all_site_request = _compat_is_all_site_request
    chat.requested_cu_au_ratios = _compat_requested_cu_au_ratios
    chat.natural_language_adsorption_region = _compat_natural_language_adsorption_region
    chat.resolve_text_settings = _compat_resolve_text_settings
    try:
        chat.run_patched_main(app_dir)
    finally:
        chat.patched_main_source = original_source_patch
        chat.is_all_site_request = original_all_site
        chat.requested_cu_au_ratios = original_ratios
        chat.natural_language_adsorption_region = original_region
        chat.resolve_text_settings = original_resolve
