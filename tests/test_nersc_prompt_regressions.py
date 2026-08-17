from pathlib import Path

from tools.nersc_account_compat import _remote_account_source, _set_account_in_text
from tools.nersc_portability_patch import patched_main_source


def test_remote_account_rewriter_source_compiles():
    compile(_remote_account_source("m5268"), "<remote-account-update>", "exec")


def test_explicit_account_replaces_placeholder():
    source = "#!/bin/bash\n#SBATCH -J relax\n#SBATCH -A mXXXX\n#SBATCH -q regular\n"
    updated = _set_account_in_text(source, "m5268")
    assert "#SBATCH -A m5268" in updated
    assert "mXXXX" not in updated


def test_account_argument_is_updated_for_post_relax_xas():
    source = (
        "#!/bin/bash\n"
        "#SBATCH -J workflow_xas\n"
        "#SBATCH -A mXXXX\n"
        "python remote_xas_from_contcar.py --account \"mXXXX\" \\\n"
        "  --queue regular\n"
    )
    updated = _set_account_in_text(source, "m5268")
    assert "#SBATCH -A m5268" in updated
    assert '--account "m5268"' in updated


def test_main_patch_masks_account_from_facet_parser_and_accepts_in_run_name():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "main.py").read_text()
    patched = patched_main_source(source)
    compile(patched, str(repo_root / "web_app" / "main.py"), "exec")
    assert "chemistry_text = re.sub" in patched
    assert "params.update(workflow_updates)" in patched
    assert "set_remote_workflow_account" in patched
    assert 'r"\\b(?:in|at|from)\\s+([A-Za-z0-9_.-]+)\\b"' in patched
