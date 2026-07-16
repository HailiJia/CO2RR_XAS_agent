from pathlib import Path

from tools.nersc_portability_patch import patched_main_source


def test_runtime_source_removes_user_specific_nersc_fallbacks():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "main.py").read_text()
    patched = patched_main_source(source)

    compile(patched, str(repo_root / "web_app" / "main.py"), "exec")

    assert "hjia" not in patched
    assert "m5268" not in patched
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
