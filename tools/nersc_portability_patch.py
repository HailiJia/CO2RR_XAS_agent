from __future__ import annotations

from pathlib import Path

from tools import chat_workflow_patch as chat

_CHAT_PATCHED_MAIN_SOURCE = chat.patched_main_source


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
    original = chat.patched_main_source
    chat.patched_main_source = patched_main_source
    try:
        chat.run_patched_main(app_dir)
    finally:
        chat.patched_main_source = original
