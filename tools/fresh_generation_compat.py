from __future__ import annotations

import re
from typing import Any, Callable, Dict


STRUCTURE_GENERATION_DEFAULTS: Dict[str, Any] = {
    "structure_source": "Build from settings",
    "structure_mode": "Single metal surface",
    "element": "Cu",
    "element1": "Cu",
    "element2": "Au",
    "facet": "111",
    "nx": 3,
    "ny": 3,
    "coverage_preset": "1/9 ML (3×3, default)",
    "layers": 4,
    "vacuum": 15.0,
    "adsorbate": "CO",
    "height": 2.0,
    "interface_match_mode": "auto",
    "element1_repeats": 4,
    "element2_repeats": 4,
    "interface_binding_element": "Cu",
    "stripe_ratio": "1:1",
    "adsorption_region": "Cu_side_interface",
}

_OPTIONAL_STRUCTURE_KEYS = (
    "cu_rows",
    "au_rows",
    "element1_rows",
    "element2_rows",
    "site_index",
    "adsorption_site",
)


def should_start_fresh_generation(text: str) -> bool:
    """True when a prompt asks for a new chemistry model rather than an edit.

    Bare commands such as ``generate structure`` intentionally keep the sidebar
    settings.  Once the user supplies chemistry/system information, a new
    generate/build/create/make command is treated as a fresh request so stale
    Cu/Au interface settings cannot leak into the next model.
    """
    low = str(text or "").lower()
    if not re.search(r"\b(generate|build|create|make|enumerate|screen)\b", low):
        return False

    chemistry_terms = re.search(
        r"\b(cu|au|ni|ag|pt|pd|ir|rh|al|co2rr|pathway|reaction|surface|interface|"
        r"alloy|bimetal|adsorbate|clean|occo|coco|choh|cho|co2|co|oh|ch3|ch2|ch|h|o)\b",
        low,
    )
    explicit_ratio = re.search(r"(?<![A-Za-z0-9])\d+\s*:\s*\d+(?![A-Za-z0-9])", low)
    explicit_facet = re.search(r"\(?(?:111|100|110)\)?", low)
    return bool(chemistry_terms or explicit_ratio or explicit_facet)


def reset_structure_generation_params(params: Dict[str, Any]) -> Dict[str, Any]:
    """Reset only structure-generation state; preserve workflow/NERSC/XAS state."""
    params.update(STRUCTURE_GENERATION_DEFAULTS)
    for key in _OPTIONAL_STRUCTURE_KEYS:
        params.pop(key, None)
    return params


def _patch_fresh_generation_source(source: str) -> str:
    helper_marker = "def _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):\n"
    if helper_marker not in source:
        raise RuntimeError("Could not locate chat intent executor for fresh-generation handling.")

    defaults_literal = repr(STRUCTURE_GENERATION_DEFAULTS)
    optional_literal = repr(_OPTIONAL_STRUCTURE_KEYS)
    helper_source = f'''def _chat_start_fresh_generation_if_requested(intent, text, params):
    """Reset stale structure state for explicit new chemistry-generation requests."""
    low = str(text or "").lower()
    fresh_intents = {{
        "generate_structure",
        "generate_structure_and_relaxation",
        "generate_batch_pathway",
        "prepare_full_workflow",
        "prepare_upload_full_workflow",
        "prepare_upload_start_full_workflow",
    }}
    if intent not in fresh_intents:
        return False
    if not re.search(r"\\b(generate|build|create|make|enumerate|screen)\\b", low):
        return False
    chemistry_terms = re.search(
        r"\\b(cu|au|ni|ag|pt|pd|ir|rh|al|co2rr|pathway|reaction|surface|interface|"
        r"alloy|bimetal|adsorbate|clean|occo|coco|choh|cho|co2|co|oh|ch3|ch2|ch|h|o)\\b",
        low,
    )
    explicit_ratio = re.search(r"(?<![A-Za-z0-9])\\d+\\s*:\\s*\\d+(?![A-Za-z0-9])", low)
    explicit_facet = re.search(r"\\(?(?:111|100|110)\\)?", low)
    if not (chemistry_terms or explicit_ratio or explicit_facet):
        return False

    params.update({defaults_literal})
    for key in {optional_literal}:
        params.pop(key, None)

    # A fresh request must not accidentally reuse the prior generated object or
    # package if parsing/generation later fails.
    for key in [
        "last_structure",
        "last_paths",
        "last_substrate_n",
        "last_structure_error",
        "last_relax_result",
        "last_xas_result",
        "last_batch_result",
        "last_generated_package_kind",
    ]:
        st.session_state.pop(key, None)
    st.session_state["last_generation_started_fresh"] = True
    return True


'''
    source = source.replace(helper_marker, helper_source + helper_marker, 1)

    executor_head = (
        'def _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):\n'
        '    """Execute one explicit chat intent. Rule-based and optional LLM plans both use this."""\n'
        '    low = str(text or "").lower()\n'
        '    plan = plan or {}\n\n'
    )
    executor_new = (
        'def _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):\n'
        '    """Execute one explicit chat intent. Rule-based and optional LLM plans both use this."""\n'
        '    low = str(text or "").lower()\n'
        '    plan = plan or {}\n'
        '    _chat_start_fresh_generation_if_requested(intent, text, params)\n\n'
    )
    if executor_head not in source:
        raise RuntimeError("Could not insert fresh-generation reset into chat intent executor.")
    return source.replace(executor_head, executor_new, 1)


def install() -> None:
    """Install fresh-request semantics on top of the existing runtime patches."""
    from tools import nersc_portability_patch as portability

    if getattr(portability, "_fresh_generation_compat_installed", False):
        return

    original_source_patch: Callable[[str], str] = portability.patched_main_source

    def patched_source(source: str) -> str:
        return _patch_fresh_generation_source(original_source_patch(source))

    portability.patched_main_source = patched_source
    portability._fresh_generation_compat_installed = True
