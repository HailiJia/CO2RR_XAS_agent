from pathlib import Path

from tools.nersc_account_compat import sanitize_local_workflow_package
from tools import stripe_interface_patch as stripe


def test_four_row_ratio_mapping_is_realistic_minimum(monkeypatch):
    monkeypatch.setattr(stripe, "STRIPE_ROW_BASE_UNIT", 4)
    assert stripe.actual_rows_from_ratio("1:1")[:2] == (4, 4)
    assert stripe.actual_rows_from_ratio("2:1")[:2] == (8, 4)
    assert stripe.actual_rows_from_ratio("3:1")[:2] == (12, 4)
    assert stripe.actual_rows_from_ratio("1:2")[:2] == (4, 8)
    assert stripe.actual_rows_from_ratio("1:3")[:2] == (4, 12)


def test_placeholder_account_is_removed_but_real_account_is_preserved(tmp_path):
    relax = tmp_path / "01_structure"
    relax.mkdir()
    submit = relax / "submit_relax.sh"
    submit.write_text(
        "#!/bin/bash\n#SBATCH -J relax\n#SBATCH -A mXXXX\n#SBATCH -q regular\nsrun vasp_std\n"
    )
    workflow_xas = tmp_path / "workflow_xas.sh"
    workflow_xas.write_text(
        "#!/bin/bash\n#SBATCH -A mXXXX\n"
        "python remote_xas_from_contcar.py --account \"mXXXX\"\n"
        "for name in VASP FDMNES FEFF; do\n  :\ndone\n"
    )

    result = sanitize_local_workflow_package(tmp_path, account="mXXXX")
    assert result["account_mode"] == "iris_default"
    assert "#SBATCH -A mXXXX" not in submit.read_text()
    assert "#SBATCH -A mXXXX" not in workflow_xas.read_text()
    assert '--account ""' in workflow_xas.read_text()
    assert "__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__" in workflow_xas.read_text()

    submit.write_text("#!/bin/bash\n#SBATCH -A m1234\n")
    sanitize_local_workflow_package(tmp_path, account="m1234")
    assert "#SBATCH -A m1234" in submit.read_text()


def test_app_entrypoint_installs_runtime_compatibility_fixes():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "CO2RR_XAS_Agent.py").read_text()
    assert "nersc_account_compat.install()" in source
    assert "stripe_width_compat.install()" in source
