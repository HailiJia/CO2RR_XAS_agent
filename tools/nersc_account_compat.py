from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Callable, Dict

PLACEHOLDER_ACCOUNT = "mXXXX"
_CLEANUP_TAG = "__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__"


def _is_placeholder_or_blank(account: Any) -> bool:
    text = str(account or "").strip()
    return not text or text.lower() == PLACEHOLDER_ACCOUNT.lower()


def _sanitize_slurm_text(text: str) -> str:
    """Remove only blank/placeholder account directives; preserve real accounts."""
    text = re.sub(
        r"(?m)^#SBATCH\s+-A\s*(?:mXXXX)?\s*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"(?m)^#SBATCH\s+--account(?:=|\s+)\s*(?:mXXXX)?\s*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"--account\s+([\"'])mXXXX\1",
        '--account ""',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"--account\s+mXXXX\b",
        '--account ""',
        text,
        flags=re.IGNORECASE,
    )
    return text


def _ensure_downstream_account_cleanup(text: str) -> str:
    """After post-relax XAS generation, drop placeholder accounts before sbatch."""
    if _CLEANUP_TAG in text:
        return text
    marker = "for name in VASP FDMNES FEFF; do"
    if marker not in text:
        return text
    block = '''# __CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__
# If no project was explicitly selected, use the user's Iris default Slurm
# account. remote_xas_from_contcar may still be an older checkout whose
# generator writes the historical mXXXX placeholder, so remove only that
# placeholder (and blank directives) before submitting the generated XAS jobs.
for generated_submit in 02_XAS/*/submit.sh; do
  [ -f "${generated_submit}" ] || continue
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]+mXXXX[[:space:]]*$/Id' "${generated_submit}"
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]*$/d' "${generated_submit}"
done
'''
    return text.replace(marker, block + marker, 1)


def sanitize_local_workflow_package(package_root: str | Path, account: Any = "") -> Dict[str, Any]:
    """Make a generated package portable when no real NERSC account was given.

    A real explicitly supplied project is never modified.  For a blank account
    or the historical ``mXXXX`` placeholder, Slurm account directives are
    omitted so NERSC can use the user's default project configured in Iris.
    """
    root = Path(package_root)
    result: Dict[str, Any] = {"package_root": str(root), "changed": [], "account_mode": "explicit"}
    if not _is_placeholder_or_blank(account):
        return result

    result["account_mode"] = "iris_default"
    candidates = [
        root / "01_structure" / "submit_relax.sh",
        root / "workflow_xas.sh",
    ]
    candidates.extend(root.glob("02_XAS/*/submit.sh"))

    for path in candidates:
        if not path.exists() or not path.is_file():
            continue
        original = path.read_text()
        updated = _sanitize_slurm_text(original)
        if path.name == "workflow_xas.sh":
            updated = _ensure_downstream_account_cleanup(updated)
        if updated != original:
            path.write_text(updated)
            path.chmod(0o755)
            result["changed"].append(str(path.relative_to(root)))

    manifest_path = root / "workflow_manifest.json"
    if manifest_path.exists():
        try:
            manifest = json.loads(manifest_path.read_text())
            settings = manifest.setdefault("settings", {})
            if _is_placeholder_or_blank(settings.get("account")):
                settings["account"] = ""
                settings["account_mode"] = "iris_default"
                manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
                result["changed"].append("workflow_manifest.json")
        except Exception:
            pass

    return result


def _remote_sanitize_source() -> str:
    """Python source executed on NERSC before submit/restart of an old package."""
    return r'''
import json
import re
from pathlib import Path

root = Path('.')
changed = []

def sanitize(text):
    text = re.sub(r'(?m)^#SBATCH\s+-A\s*(?:mXXXX)?\s*\n?', '', text, flags=re.I)
    text = re.sub(r'(?m)^#SBATCH\s+--account(?:=|\s+)\s*(?:mXXXX)?\s*\n?', '', text, flags=re.I)
    text = re.sub(r'--account\s+([\"\'])mXXXX\1', '--account ""', text, flags=re.I)
    text = re.sub(r'--account\s+mXXXX\b', '--account ""', text, flags=re.I)
    return text

for path in [root / '01_structure' / 'submit_relax.sh', root / 'workflow_xas.sh', *root.glob('02_XAS/*/submit.sh')]:
    if not path.exists() or not path.is_file():
        continue
    original = path.read_text()
    updated = sanitize(original)
    if path.name == 'workflow_xas.sh' and '__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__' not in updated:
        marker = 'for name in VASP FDMNES FEFF; do'
        if marker in updated:
            block = '''# __CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__
# Use the Iris default account when an older generator emits mXXXX.
for generated_submit in 02_XAS/*/submit.sh; do
  [ -f "${generated_submit}" ] || continue
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]+mXXXX[[:space:]]*$/Id' "${generated_submit}"
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]*$/d' "${generated_submit}"
done
'''
            updated = updated.replace(marker, block + marker, 1)
    if updated != original:
        path.write_text(updated)
        path.chmod(0o755)
        changed.append(str(path))

manifest_path = root / 'workflow_manifest.json'
if manifest_path.exists():
    try:
        manifest = json.loads(manifest_path.read_text())
        settings = manifest.setdefault('settings', {})
        account = str(settings.get('account') or '').strip()
        if not account or account.lower() == 'mxxxx':
            settings['account'] = ''
            settings['account_mode'] = 'iris_default'
            manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
            changed.append(str(manifest_path))
    except Exception:
        pass

print(json.dumps({'status': 'ok', 'changed': changed}))
'''


def sanitize_remote_workflow_package(client: Any, remote_dir: str, timeout_s: int = 120) -> Dict[str, Any]:
    """Repair only historical blank/mXXXX account directives in an uploaded package."""
    quoted_dir = client.shell_quote(str(remote_dir).rstrip("/"))
    quoted_source = client.shell_quote(_remote_sanitize_source())
    cmd = (
        f"cd {quoted_dir} && "
        "PY=$(command -v python3.11 || command -v python3 || command -v python || true); "
        "if [ -z \"$PY\" ]; then echo 'ERROR: Python is required for workflow account compatibility patch' >&2; exit 91; fi; "
        f'"$PY" -c {quoted_source}'
    )
    task = client.run_command(cmd)
    task_id = str(task.get("task_id") or task.get("id") or "")
    if not task_id:
        raise RuntimeError(f"SF API did not return a task ID while checking workflow account directives: {task}")
    task_result = client.wait_task(task_id, timeout_s=timeout_s)
    result_json = client.task_result_json(task_result) if hasattr(client, "task_result_json") else {}
    if not isinstance(result_json, dict):
        result_json = {"raw_result": result_json}

    task_status = str(task_result.get("status", "")).strip().lower() if isinstance(task_result, dict) else ""
    command_status = str(result_json.get("status", "")).strip().lower()
    error = result_json.get("error")
    exit_code = result_json.get("exit_code")
    bad = {"failed", "error", "cancelled", "canceled"}
    if task_status in bad or command_status in bad or error not in (None, "", False) or exit_code not in (None, 0, "0"):
        detail = result_json.get("output") or result_json.get("raw_result") or result_json
        raise RuntimeError(
            "Could not sanitize historical NERSC account placeholders before workflow start. "
            f"SF API task ID: {task_id}. Result: {detail}"
        )
    return {"task_id": task_id, "result": result_json}


def install() -> None:
    """Install account portability around the existing workflow helpers."""
    from tools import web_nersc_workflow_integration as workflow

    if getattr(workflow, "_nersc_account_compat_installed", False):
        return

    original_prepare: Callable[..., Dict[str, Any]] = workflow.prepare_single_structure_workflow_package
    original_run: Callable[..., Dict[str, Any]] = workflow.run_remote_workflow_script

    def prepare_wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = original_prepare(*args, **kwargs)
        package_root = result.get("package_root") or kwargs.get("package_root")
        account = kwargs.get("account", "")
        if package_root and _is_placeholder_or_blank(account):
            result["account_compat"] = sanitize_local_workflow_package(package_root, account=account)
        return result

    def run_wrapper(
        client: Any,
        remote_dir: str,
        script_name: str,
        timeout_s: int = 120,
    ) -> Dict[str, Any]:
        if script_name in {"workflow_submit.sh", "workflow_restart.sh"}:
            sanitize_remote_workflow_package(client, remote_dir, timeout_s=min(timeout_s, 120))
        return original_run(client, remote_dir, script_name, timeout_s=timeout_s)

    workflow.prepare_single_structure_workflow_package = prepare_wrapper
    workflow.run_remote_workflow_script = run_wrapper
    workflow._nersc_account_compat_installed = True
