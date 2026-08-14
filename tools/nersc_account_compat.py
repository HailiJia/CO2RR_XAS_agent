from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict

PLACEHOLDER_ACCOUNT = "mXXXX"
_CLEANUP_TAG = "__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__"


def _is_placeholder_or_blank(account: Any) -> bool:
    value = str(account or "").strip()
    return not value or value.lower() == PLACEHOLDER_ACCOUNT.lower()


def _sanitize_slurm_text(text: str) -> str:
    """Remove only blank/mXXXX account settings; never touch a real project."""
    text = re.sub(
        r"(?mi)^#SBATCH[ \t]+-A[ \t]+mXXXX[ \t]*$\n?",
        "",
        text,
    )
    text = re.sub(
        r"(?mi)^#SBATCH[ \t]+-A[ \t]*$\n?",
        "",
        text,
    )
    text = re.sub(
        r"(?mi)^#SBATCH[ \t]+--account(?:=|[ \t]+)mXXXX[ \t]*$\n?",
        "",
        text,
    )
    text = re.sub(
        r"--account[ \t]+([\"'])mXXXX\1",
        '--account ""',
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"--account[ \t]+mXXXX\b",
        '--account ""',
        text,
        flags=re.IGNORECASE,
    )
    return text


def _ensure_downstream_cleanup(text: str) -> str:
    """Strip blank/mXXXX accounts from XAS scripts generated after relaxation."""
    if _CLEANUP_TAG in text:
        return text
    marker = "for name in VASP FDMNES FEFF; do"
    if marker not in text:
        return text
    block = '''# __CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__
# No explicit project: let Slurm use the user's default NERSC account.
for generated_submit in 02_XAS/*/submit.sh; do
  [ -f "${generated_submit}" ] || continue
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]+mXXXX[[:space:]]*$/Id' "${generated_submit}"
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]*$/d' "${generated_submit}"
done
'''
    return text.replace(marker, block + marker, 1)


def sanitize_local_workflow_package(package_root: str | Path, account: Any = "") -> Dict[str, Any]:
    """Make generated workflow scripts use the Iris default when account is unset."""
    root = Path(package_root)
    result: Dict[str, Any] = {
        "package_root": str(root),
        "changed": [],
        "account_mode": "explicit",
    }
    if not _is_placeholder_or_blank(account):
        return result

    result["account_mode"] = "iris_default"
    paths = [root / "01_structure" / "submit_relax.sh", root / "workflow_xas.sh"]
    paths.extend(root.glob("02_XAS/*/submit.sh"))
    for path in paths:
        if not path.is_file():
            continue
        original = path.read_text()
        updated = _sanitize_slurm_text(original)
        if path.name == "workflow_xas.sh":
            updated = _ensure_downstream_cleanup(updated)
        if updated != original:
            path.write_text(updated)
            path.chmod(0o755)
            result["changed"].append(str(path.relative_to(root)))
    return result


def _remote_sanitize_source() -> str:
    # Use double-triple quotes here because the source itself contains a
    # triple-single-quoted shell block.
    return r"""
import re
from pathlib import Path

root = Path('.')

def sanitize(text):
    text = re.sub(r'(?mi)^#SBATCH[ \t]+-A[ \t]+mXXXX[ \t]*$\n?', '', text)
    text = re.sub(r'(?mi)^#SBATCH[ \t]+-A[ \t]*$\n?', '', text)
    text = re.sub(r'(?mi)^#SBATCH[ \t]+--account(?:=|[ \t]+)mXXXX[ \t]*$\n?', '', text)
    text = re.sub(r'--account[ \t]+([\"\'])mXXXX\1', '--account ""', text, flags=re.I)
    text = re.sub(r'--account[ \t]+mXXXX\b', '--account ""', text, flags=re.I)
    return text

for path in [root / '01_structure' / 'submit_relax.sh', root / 'workflow_xas.sh', *root.glob('02_XAS/*/submit.sh')]:
    if not path.is_file():
        continue
    original = path.read_text()
    updated = sanitize(original)
    if path.name == 'workflow_xas.sh' and '__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__' not in updated:
        marker = 'for name in VASP FDMNES FEFF; do'
        if marker in updated:
            block = '''# __CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__
# No explicit project: let Slurm use the user's default NERSC account.
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
"""


def sanitize_remote_workflow_package(client: Any, remote_dir: str, timeout_s: int = 120) -> Dict[str, Any]:
    """Repair historical mXXXX directives in an already-uploaded workflow."""
    quoted_dir = client.shell_quote(str(remote_dir).rstrip("/"))
    quoted_source = client.shell_quote(_remote_sanitize_source())
    cmd = (
        f"cd {quoted_dir} && "
        "PY=$(command -v python3.11 || command -v python3 || command -v python || true); "
        "if [ -z \"$PY\" ]; then echo 'ERROR: Python is required for account cleanup' >&2; exit 91; fi; "
        f'"$PY" -c {quoted_source}'
    )
    task = client.run_command(cmd)
    task_id = str(task.get("task_id") or task.get("id") or "")
    if not task_id:
        raise RuntimeError(f"SF API did not return a task ID for account cleanup: {task}")
    task_result = client.wait_task(task_id, timeout_s=timeout_s)
    result = client.task_result_json(task_result) if hasattr(client, "task_result_json") else {}
    if not isinstance(result, dict):
        result = {"raw_result": result}
    task_status = str(task_result.get("status", "")).lower() if isinstance(task_result, dict) else ""
    result_status = str(result.get("status", "")).lower()
    error = result.get("error")
    exit_code = result.get("exit_code")
    bad = {"failed", "error", "cancelled", "canceled"}
    if task_status in bad or result_status in bad or error not in (None, "", False) or exit_code not in (None, 0, "0"):
        raise RuntimeError(
            "Could not remove the historical mXXXX account before workflow start. "
            f"SF API task ID: {task_id}. Result: {result}"
        )
    return {"task_id": task_id, "result": result}


def install() -> None:
    """Wrap workflow preparation/start with account portability handling."""
    from tools import web_nersc_workflow_integration as workflow

    if getattr(workflow, "_nersc_account_compat_installed", False):
        return

    original_prepare: Callable[..., Dict[str, Any]] = workflow.prepare_single_structure_workflow_package
    original_run: Callable[..., Dict[str, Any]] = workflow.run_remote_workflow_script

    def prepare_wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = original_prepare(*args, **kwargs)
        account = kwargs.get("account", "")
        package_root = result.get("package_root") or kwargs.get("package_root")
        if package_root and _is_placeholder_or_blank(account):
            result["account_compat"] = sanitize_local_workflow_package(package_root, account)
        return result

    def run_wrapper(client: Any, remote_dir: str, script_name: str, timeout_s: int = 120) -> Dict[str, Any]:
        if script_name in {"workflow_submit.sh", "workflow_restart.sh"}:
            sanitize_remote_workflow_package(client, remote_dir, timeout_s=min(timeout_s, 120))
        return original_run(client, remote_dir, script_name, timeout_s=timeout_s)

    workflow.prepare_single_structure_workflow_package = prepare_wrapper
    workflow.run_remote_workflow_script = run_wrapper
    workflow._nersc_account_compat_installed = True
