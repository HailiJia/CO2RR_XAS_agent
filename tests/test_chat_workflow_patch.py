from pathlib import Path

from tools.chat_workflow_patch import (
    LOCAL_PLANNER_LABEL,
    is_all_site_request,
    parse_prompt_metals,
    patched_main_source,
    requested_cu_au_ratios,
    should_generate_structure,
)


def test_generate_adsorbate_on_metal_is_an_action():
    assert should_generate_structure("generate CO on Cu")
    assert should_generate_structure("make CHO on Ni")
    assert not should_generate_structure("set CO on Cu")


def test_all_site_request_recognizes_sites_and_existing_regions():
    prompt = (
        "Generate all possible intermediates with all possible sites for Au/Cu, "
        "use ratio Au:Cu=1:1 and Au:Cu=1:2"
    )
    assert is_all_site_request(prompt)
    assert is_all_site_request(
        "Generate every intermediate in every adsorption region for Cu/Au ratio 1:3"
    )


def test_au_cu_ratios_are_normalized_for_existing_cu_au_generator():
    ratios = requested_cu_au_ratios(
        "Generate all sites for Au/Cu, use ratio Au:Cu=1:1 and 1:2"
    )
    assert ratios == ["1:1", "2:1"]
    assert requested_cu_au_ratios("all sites Cu:Au=1:1 and 1:3") == ["1:1", "1:3"]


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
    assert "chat_patch_should_generate_structure" in patched
