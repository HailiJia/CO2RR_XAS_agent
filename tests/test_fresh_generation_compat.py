from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools import fresh_generation_compat as fresh


def test_full_cu_pathway_starts_from_fresh_single_metal_state():
    params = {
        "structure_source": "Build from settings",
        "structure_mode": "Interface",
        "element": "Au",
        "element1": "Cu",
        "element2": "Au",
        "facet": "111",
        "nx": 8,
        "ny": 12,
        "layers": 6,
        "adsorbate": "OCCO",
        "stripe_ratio": "3:1",
        "adsorption_region": "Cu_side_interface",
        "interface_binding_element": "Cu",
        "nersc_account": "m5268",
        "nersc_queue": "regular",
        "nersc_walltime": "08:00:00",
        "xas_absorber": "Cu",
    }

    assert fresh.should_start_fresh_generation("Generate full CO2RR pathway on Cu")
    fresh.reset_structure_generation_params(params)

    assert params["structure_mode"] == "Single metal surface"
    assert params["element"] == "Cu"
    assert params["nx"] == 3
    assert params["ny"] == 3
    assert params["layers"] == 4
    assert params["stripe_ratio"] == "1:1"
    assert params["adsorption_region"] == "Cu_side_interface"

    # Workflow/runtime settings must survive a fresh chemistry request.
    assert params["nersc_account"] == "m5268"
    assert params["nersc_queue"] == "regular"
    assert params["nersc_walltime"] == "08:00:00"
    assert params["xas_absorber"] == "Cu"


def test_bare_generate_structure_keeps_current_sidebar_semantics():
    assert not fresh.should_start_fresh_generation("generate structure")
    assert not fresh.should_start_fresh_generation("change stripe ratio to 2:1")
    assert fresh.should_start_fresh_generation("generate OCCO bridge on Cu/Au stripe 2:1")
    assert fresh.should_start_fresh_generation("build CO on Au(111)")


def test_runtime_patch_inserts_reset_before_planner_settings():
    source = (REPO_ROOT / "web_app" / "main.py").read_text()
    patched = fresh._patch_fresh_generation_source(source)
    compile(patched, str(REPO_ROOT / "web_app" / "main.py"), "exec")

    reset_call = "_chat_start_fresh_generation_if_requested(intent, text, params)"
    plan_settings = 'if plan.get("settings") and isinstance(plan.get("settings"), dict):'
    assert reset_call in patched
    assert patched.index(reset_call) < patched.index(plan_settings)


def test_app_installs_fresh_generation_compatibility():
    source = (REPO_ROOT / "web_app" / "CO2RR_XAS_Agent.py").read_text()
    assert "fresh_generation_compat.install()" in source
