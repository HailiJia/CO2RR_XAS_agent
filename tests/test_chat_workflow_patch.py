from pathlib import Path

from tools.chat_workflow_patch import (
    LOCAL_PLANNER_LABEL,
    is_site_ratio_grid_request,
    parse_prompt_metals,
    parse_requested_ratios,
    patched_main_source,
    should_generate_structure,
)


def test_generate_adsorbate_on_metal_is_an_action():
    assert should_generate_structure("generate CO on Cu")
    assert should_generate_structure("make CHO on Ni")
    assert not should_generate_structure("set CO on Cu")


def test_ratio_grid_request_recognizes_sites_wording():
    prompt = (
        "Generate all possible intermediates with all possible sites for Au/Cu, "
        "use ratio Au:Cu=1:1 and Au:Cu=1:2"
    )
    assert is_site_ratio_grid_request(prompt)


def test_au_cu_ratios_are_normalized_without_losing_user_notation():
    ratios = parse_requested_ratios(
        "Generate all sites for Au/Cu, use ratio Au:Cu=1:1 and 1:2"
    )
    assert [item["requested_label"] for item in ratios] == [
        "Au:Cu=1:1",
        "Au:Cu=1:2",
    ]
    assert [item["canonical_cu_au_ratio"] for item in ratios] == [
        "1:1",
        "2:1",
    ]


def test_metal_parser_supports_all_ui_metals_and_preserves_interface_order():
    assert parse_prompt_metals("generate CO on Ni")["element"] == "Ni"
    parsed = parse_prompt_metals("generate all sites for Au/Cu ratio 1:2")
    assert parsed["structure_mode"] == "Interface"
    assert parsed["element1"] == "Au"
    assert parsed["element2"] == "Cu"


def test_runtime_patched_main_compiles_and_contains_new_routes():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "main.py").read_text()
    patched = patched_main_source(source)
    compile(patched, str(repo_root / "web_app" / "main.py"), "exec")
    assert LOCAL_PLANNER_LABEL in patched
    assert 'return "generate_site_ratio_grid"' in patched
    assert "def _chat_generate_requested_site_ratio_batch" in patched
    assert "chat_patch_should_generate_structure" in patched
