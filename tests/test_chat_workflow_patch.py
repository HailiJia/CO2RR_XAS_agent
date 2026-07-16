from pathlib import Path

from tools.chat_workflow_patch import (
    LOCAL_PLANNER_LABEL,
    explicit_adsorbate,
    is_all_site_request,
    natural_language_adsorption_region,
    parse_prompt_metals,
    patched_main_source,
    requested_cu_au_ratios,
    resolve_text_settings,
    should_generate_structure,
)
from tools.stripe_interface_patch import ADSORPTION_REGION_ADSORBATES, STRIPE_RATIO_CHOICES


def test_generate_adsorbate_on_metal_is_an_action():
    assert should_generate_structure("generate CO on Cu")
    assert should_generate_structure("make CHO on Ni")
    assert not should_generate_structure("set CO on Cu")


def test_adsorbate_matching_is_boundary_safe():
    assert explicit_adsorbate("put CO on the Cu side") == "CO"
    assert explicit_adsorbate("now change the Au:Cu ratio to 3:1") is None
    assert explicit_adsorbate("move it two rows to the interface") is None


def test_all_site_request_recognizes_sites_regions_and_co2rr_wording():
    prompt = (
        "Generate all possible intermediates with all possible sites for Au/Cu, "
        "use ratio Au:Cu=1:1 and Au:Cu=1:2"
    )
    assert is_all_site_request(prompt)
    assert is_all_site_request(
        "Generate every intermediate in every adsorption region for Cu/Au ratio 1:3"
    )
    assert is_all_site_request(
        "generate all possible sites and ratio for Au/Cu in CO2RR"
    )


def test_au_cu_ratios_are_normalized_for_existing_cu_au_generator():
    ratios = requested_cu_au_ratios(
        "Generate all sites for Au/Cu, use ratio Au:Cu=1:1 and 1:2"
    )
    assert ratios == ["1:1", "2:1"]
    assert requested_cu_au_ratios("all sites Cu:Au=1:1 and 1:3") == ["1:1", "1:3"]
    assert requested_cu_au_ratios("change the Au:Cu ratio to 3:1") == ["1:3"]
    assert requested_cu_au_ratios(
        "generate all possible sites and ratio for Au/Cu in CO2RR"
    ) == STRIPE_RATIO_CHOICES


def test_ratio_only_followup_does_not_change_co_to_o():
    params = {
        "adsorbate": "CO",
        "stripe_ratio": "1:2",
        "structure_mode": "Interface",
        "element1": "Au",
        "element2": "Cu",
    }
    # Simulate a bad LLM/parser proposal containing adsorbate=O. The explicit
    # user text must remain authoritative.
    parsed = resolve_text_settings(
        "now change the Au:Cu ratio to 3:1",
        params,
        {"adsorbate": "O", "stripe_ratio": "3:1"},
    )
    assert "adsorbate" not in parsed
    assert parsed["stripe_ratio"] == "1:3"
    assert parsed["element1"] == "Au"
    assert parsed["element2"] == "Cu"


def test_natural_language_site_phrases_map_to_existing_regions():
    assert natural_language_adsorption_region(
        "put CO on Cu site interface", "3:1", "CO"
    ) == "Cu_side_interface"
    assert natural_language_adsorption_region(
        "put CO on Cu side", "3:1", "CO"
    ) == "Cu_near_interface_row_2"
    assert natural_language_adsorption_region(
        "Put CO on Cu side 2 rows to the interface", "3:1", "CO"
    ) == "Cu_near_interface_row_2"
    assert natural_language_adsorption_region(
        "put OCCO along the interface", "3:1", "OCCO"
    ) == "Cu_side_interface_dimer_along_x"
    assert natural_language_adsorption_region(
        "put OCCO across the interface", "3:1", "OCCO"
    ) == "Cu_Au_boundary_bridge"


def test_metal_parser_supports_all_ui_metals_and_preserves_interface_order():
    assert parse_prompt_metals("generate CO on Ni")["element"] == "Ni"
    parsed = parse_prompt_metals("generate all sites for Au/Cu ratio 1:2")
    assert parsed["structure_mode"] == "Interface"
    assert parsed["element1"] == "Au"
    assert parsed["element2"] == "Cu"


def test_runtime_patch_reuses_existing_stripe_batch_helper():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "main.py").read_text()
    patched = patched_main_source(source)
    compile(patched, str(repo_root / "web_app" / "main.py"), "exec")
    assert LOCAL_PLANNER_LABEL in patched
    assert "def _stripe_generate_all_adsorption_region_batch(params, text=\"\")" in patched
    assert "selected_ratios = chat_patch_requested_cu_au_ratios" in patched
    assert "def _chat_generate_requested_site_ratio_batch" not in patched
    assert "generate_site_ratio_grid" not in patched
    assert "stripe_patch_region_adsorbates" in patched
    assert "Cu_side_interface_dimer_along_x" in ADSORPTION_REGION_ADSORBATES
    assert "Cu_Au_boundary_bridge" in ADSORPTION_REGION_ADSORBATES
    assert "chat_patch_should_generate_structure" in patched
    assert "chat_patch_sanitize_plan_settings" in patched
    assert "chat_patch_resolve_text_settings" in patched
