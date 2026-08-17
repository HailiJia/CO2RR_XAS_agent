from pathlib import Path

from tools.nersc_portability_patch import (
    _compat_is_all_site_request,
    _compat_natural_language_adsorption_region,
    _compat_requested_cu_au_ratios,
    _compat_resolve_text_settings,
    patched_main_source,
)
from tools.stripe_interface_patch import STRIPE_RATIO_CHOICES


def test_runtime_source_removes_user_specific_nersc_fallbacks():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "main.py").read_text()
    patched = patched_main_source(source)

    compile(patched, str(repo_root / "web_app" / "main.py"), "exec")

    forbidden_fallbacks = [
        'os.environ.get("NERSC_ACCOUNT", "m5268")',
        '"/pscratch/sd/h/hjia/CO2RR"',
        'os.environ.get("USER") or "hjia"',
        '.strip() or "hjia"',
        '"/pscratch/sd/h/hjia"',
    ]
    for pattern in forbidden_fallbacks:
        assert pattern not in patched

    assert 'os.environ.get("CO2RR_NERSC_USERNAME")' in patched
    assert 'os.environ.get("NERSC_USERNAME")' in patched
    assert 'os.environ.get("CO2RR_NERSC_ACCOUNT")' in patched
    assert 'os.environ.get("NERSC_ACCOUNT")' in patched
    assert 'os.environ.get("CO2RR_NERSC_REMOTE_BASE")' in patched
    assert 'f"/pscratch/sd/{DEFAULT_NERSC_USERNAME[0].lower()}/{DEFAULT_NERSC_USERNAME}/CO2RR"' in patched


def test_remote_delete_guard_and_job_listing_use_resolved_username():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "main.py").read_text()
    patched = patched_main_source(source)

    assert "nersc_user_root = str(PurePosixPath(DEFAULT_NERSC_REMOTE_BASE).parent)" in patched
    assert "nersc_shard_root = str(PurePosixPath(nersc_user_root).parent)" in patched
    assert 'username = str(params.get("nersc_username") or DEFAULT_NERSC_USERNAME).strip()' in patched


def test_all_possibilities_without_sites_restores_full_cu_au_grid():
    prompt = "generate all possibilities on Cu/Au"
    assert _compat_is_all_site_request(prompt)
    assert _compat_requested_cu_au_ratios(prompt, current_ratio="3:1") == STRIPE_RATIO_CHOICES


def test_explicit_ratio_still_limits_all_possibilities_batch():
    prompt = "generate all possibilities on Cu/Au, stripe Cu:Au=3:1"
    assert _compat_is_all_site_request(prompt)
    assert _compat_requested_cu_au_ratios(prompt, current_ratio="1:1") == ["3:1"]


def test_occco_bridge_on_cu_au_maps_to_boundary_bridge():
    prompt = "generate OCCO bridge on Cu/Au, stripe Cu:Au=3:1"
    assert _compat_natural_language_adsorption_region(prompt, "3:1", "OCCO") == "Cu_Au_boundary_bridge"


def test_new_adsorbate_does_not_inherit_incompatible_old_region():
    params = {
        "adsorbate": "CO",
        "stripe_ratio": "3:1",
        "adsorption_region": "Cu_side_interface",
        "interface_binding_element": "Cu",
        "structure_mode": "Interface",
        "element1": "Cu",
        "element2": "Au",
    }
    result = _compat_resolve_text_settings(
        "generate OCCO on Cu/Au, stripe Cu:Au=3:1",
        params,
        {"adsorbate": "OCCO", "stripe_ratio": "3:1"},
    )
    assert result["adsorbate"] == "OCCO"
    assert result["adsorption_region"] in {
        "Cu_Au_boundary_bridge",
        "Cu_side_interface_dimer_along_x",
        "Cu_side_interface_dimer_row0_row1",
    }
