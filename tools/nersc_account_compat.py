from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict

PLACEHOLDER_ACCOUNT = "mXXXX"
_CLEANUP_TAG = "__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__"
_ACCOUNT_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")


def _normalize_account(account: Any) -> str:
    value = str(account or "").strip()
    if not value or value.lower() == PLACEHOLDER_ACCOUNT.lower():
        return ""
    if not _ACCOUNT_RE.fullmatch(value):
        raise ValueError(f"Invalid NERSC account/project name: {value!r}")
    return value


def _is_placeholder_or_blank(account: Any) -> bool:
    return not _normalize_account(account)


def _remove_account_directives(text: str) -> str:
    return re.sub(
        r"(?mi)^#SBATCH[ \t]+(?:-A|--account(?:=|[ \t]+))[^\n]*\n?",
        "",
        text,
    )


def _insert_account_directive(text: str, account: str) -> str:
    if not account:
        return text
    directive = f"#SBATCH -A {account}\n"
    job_match = re.search(r"(?m)^#SBATCH[ \t]+-J[^\n]*\n", text)
    if job_match:
        return text[: job_match.end()] + directive + text[job_match.end() :]
    shebang = re.match(r"^#![^\n]*\n", text)
    if shebang:
        return text[: shebang.end()] + directive + text[shebang.end() :]
    return directive + text


def _set_account_in_text(text: str, account: Any) -> str:
    """Set one explicit Slurm account, or omit it to use the Iris default."""
    normalized = _normalize_account(account)
    text = _remove_account_directives(text)
    text = _insert_account_directive(text, normalized)

    # workflow_xas.sh passes the account to remote_xas_from_contcar.py so that
    # post-relaxation VASP/FDMNES/FEFF scripts use the same project.
    replacement = f'--account "{normalized}"'
    text = re.sub(
        r"--account[ \t]+(?:\"[^\"]*\"|'[^']*'|[^\s\\]+)",
        replacement,
        text,
        flags=re.IGNORECASE,
    )
    return text


def _ensure_downstream_cleanup(text: str) -> str:
    """Strip legacy blank/mXXXX accounts from XAS scripts generated downstream."""
    if _CLEANUP_TAG in text:
        return text
    marker = "for name in VASP FDMNES FEFF; do"
    if marker not in text:
        return text
    block = '''# __CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__
# No explicit project in an older generated script: let Slurm use the user's
# default NERSC account rather than the historical mXXXX placeholder.
for generated_submit in 02_XAS/*/submit.sh; do
  [ -f "${generated_submit}" ] || continue
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]+mXXXX[[:space:]]*$/Id' "${generated_submit}"
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]*$/d' "${generated_submit}"
done
'''
    return text.replace(marker, block + marker, 1)


def set_local_workflow_account(package_root: str | Path, account: Any = "") -> Dict[str, Any]:
    """Apply a requested NERSC account to a generated workflow package."""
    root = Path(package_root)
    normalized = _normalize_account(account)
    result: Dict[str, Any] = {
        "package_root": str(root),
        "changed": [],
        "account": normalized,
        "account_mode": "explicit" if normalized else "iris_default",
    }
    paths = [root / "01_structure" / "submit_relax.sh", root / "workflow_xas.sh"]
    paths.extend(root.glob("02_XAS/*/submit.sh"))
    for path in paths:
        if not path.is_file():
            continue
        original = path.read_text()
        updated = _set_account_in_text(original, normalized)
        if path.name == "workflow_xas.sh" and not normalized:
            updated = _ensure_downstream_cleanup(updated)
        if updated != original:
            path.write_text(updated)
            path.chmod(0o755)
            result["changed"].append(str(path.relative_to(root)))
    return result


def sanitize_local_workflow_package(package_root: str | Path, account: Any = "") -> Dict[str, Any]:
    """Backward-compatible alias used by existing tests/callers."""
    return set_local_workflow_account(package_root, account)


def _remote_account_source(account: Any) -> str:
    normalized = _normalize_account(account)
    account_literal = repr(normalized)
    return f'''\
import re
from pathlib import Path

root = Path('.')
account = {account_literal}

def set_account(text):
    text = re.sub(r'(?mi)^#SBATCH[ \\t]+(?:-A|--account(?:=|[ \\t]+))[^\\n]*\\n?', '', text)
    if account:
        directive = '#SBATCH -A ' + account + '\\n'
        m = re.search(r'(?m)^#SBATCH[ \\t]+-J[^\\n]*\\n', text)
        if m:
            text = text[:m.end()] + directive + text[m.end():]
        else:
            m = re.match(r'^#![^\\n]*\\n', text)
            text = text[:m.end()] + directive + text[m.end():] if m else directive + text
    replacement = '--account "' + account + '"'
    text = re.sub(r'--account[ \\t]+(?:"[^"]*"|\\'[^\\']*\\'|[^\\s\\\\]+)', replacement, text, flags=re.I)
    return text

for path in [root / '01_structure' / 'submit_relax.sh', root / 'workflow_xas.sh', *root.glob('02_XAS/*/submit.sh')]:
    if not path.is_file():
        continue
    original = path.read_text()
    updated = set_account(original)
    if path.name == 'workflow_xas.sh' and not account and '__CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__' not in updated:
        marker = 'for name in VASP FDMNES FEFF; do'
        if marker in updated:
            block = '''# __CO2RR_PLACEHOLDER_ACCOUNT_CLEANUP__
for generated_submit in 02_XAS/*/submit.sh; do
  [ -f "${{generated_submit}}" ] || continue
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]+mXXXX[[:space:]]*$/Id' "${{generated_submit}}"
  sed -i -E '/^#SBATCH[[:space:]]+-A[[:space:]]*$/d' "${{generated_submit}}"
done
'''
            updated = updated.replace(marker, block + marker, 1)
    if updated != original:
        path.write_text(updated)
        path.chmod(0o755)
'''


def set_remote_workflow_account(
    client: Any,
    remote_dir: str,
    account: Any = "",
    timeout_s: int = 120,
) -> Dict[str, Any]:
    """Rewrite account directives in an already-uploaded NERSC workflow."""
    normalized = _normalize_account(account)
    quoted_dir = client.shell_quote(str(remote_dir).rstrip("/"))
    quoted_source = client.shell_quote(_remote_account_source(normalized))
    cmd = (
        f"cd {quoted_dir} || exit 92; "
        "PY=$(command -v python3.11 || command -v python3 || command -v python || true); "
        "if [ -z \"$PY\" ]; then echo 'ERROR: Python is required for workflow account update' >&2; exit 91; fi; "
        f'"$PY" -c {quoted_source}'
    )
    task = client.run_command(cmd)
    task_id = str(task.get("task_id") or task.get("id") or "")
    if not task_id:
        raise RuntimeError(f"SF API did not return a task ID while updating the NERSC account: {task}")
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
            f"Could not set NERSC account {normalized or '<Iris default>'} in {remote_dir}. "
            f"SF API task ID: {task_id}. Result: {result}"
        )
    return {"task_id": task_id, "account": normalized, "result": result}


def sanitize_remote_workflow_package(client: Any, remote_dir: str, timeout_s: int = 120) -> Dict[str, Any]:
    """Backward-compatible placeholder cleanup using the Iris default account."""
    return set_remote_workflow_account(client, remote_dir, "", timeout_s=timeout_s)


def install() -> None:
    """Wrap package preparation so generated files carry the selected account."""
    from tools import web_nersc_workflow_integration as workflow

    if getattr(workflow, "_nersc_account_compat_installed", False):
        return

    original_prepare: Callable[..., Dict[str, Any]] = workflow.prepare_single_structure_workflow_package

    def prepare_wrapper(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        result = original_prepare(*args, **kwargs)
        account = kwargs.get("account", "")
        package_root = result.get("package_root") or kwargs.get("package_root")
        if package_root:
            result["account_compat"] = set_local_workflow_account(package_root, account)
        return result

    workflow.prepare_single_structure_workflow_package = prepare_wrapper
    workflow._nersc_account_compat_installed = True
