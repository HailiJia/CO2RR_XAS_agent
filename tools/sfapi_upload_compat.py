from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict


def _is_missing_remote_path_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "no such file" in text or "no such file or directory" in text


def _direct_upload_directory(
    self,
    local_dir: str,
    remote_dir: str,
    overwrite: bool = True,
    chunk_chars: int = 8000,
    timeout_s: int = 240,
) -> Dict[str, Any]:
    """Upload a workflow package with the SF API file-upload endpoint.

    The historical implementation sent the archive as many 8k base64 append
    command tasks. One task stuck in SF API status ``new`` could make the whole
    upload time out. NERSC provides ``PUT /utilities/upload`` for small files,
    so this path uploads one tar.gz and uses one command task to verify/unpack.

    ``chunk_chars`` is accepted for API compatibility but intentionally unused.
    """
    remote_dir = str(remote_dir).rstrip("/")
    local_root = Path(local_dir).resolve()
    if not local_root.is_dir():
        raise RuntimeError(f"Local directory does not exist: {local_dir}")

    b64, archive_sha256, archive_nbytes = self._local_dir_to_tar_bundle(str(local_root))
    archive_bytes = base64.b64decode(b64.encode("ascii"))

    remote_path = PurePosixPath(remote_dir)
    remote_parent = str(remote_path.parent)
    marker = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_name = remote_path.name or "workflow"
    remote_tmp_tgz = f"{remote_parent}/.{safe_name}.web_xas_agent_upload_{marker}.tar.gz"

    # IMPORTANT: keep the leading slash of the absolute NERSC filesystem path.
    # NERSC's official sfapi_client constructs the same form:
    #   utilities/upload/perlmutter//pscratch/...
    # The apparent double slash is intentional: one slash separates the API
    # route and the second slash is the leading slash of the absolute path.
    upload_url = self._url(f"utilities/upload/{self.system}/{remote_tmp_tgz}")

    def _put_archive():
        files = {
            "file": (
                PurePosixPath(remote_tmp_tgz).name,
                io.BytesIO(archive_bytes),
                "application/gzip",
            )
        }
        response = self.session.put(upload_url, files=files)
        return self._check(response, "workflow archive upload")

    try:
        upload_result = _put_archive()
    except Exception as exc:
        if not _is_missing_remote_path_error(exc):
            raise

        # The upload endpoint creates the file but not missing parent
        # directories. Create the parent once and retry the direct upload.
        mkdir_task = self.run_command(f"mkdir -p {self.shell_quote(remote_parent)}")
        mkdir_id = str(mkdir_task.get("task_id") or mkdir_task.get("id") or "")
        if not mkdir_id:
            raise RuntimeError(
                f"SF API did not return a task ID while creating upload parent {remote_parent}: {mkdir_task}"
            ) from exc
        self._wait_successful_command(mkdir_id, timeout_s=max(int(timeout_s), 600))
        upload_result = _put_archive()

    expected_sha = self.shell_quote(archive_sha256)
    overwrite_cmd = f"rm -rf {self.shell_quote(remote_dir)}; " if overwrite else ""
    unpack_cmd = (
        "set -e; "
        f"test -f {self.shell_quote(remote_tmp_tgz)}; "
        f"remote_sha=$(sha256sum {self.shell_quote(remote_tmp_tgz)} | awk '{{print $1}}'); "
        f"test \"$remote_sha\" = {expected_sha}; "
        f"tar -tzf {self.shell_quote(remote_tmp_tgz)} >/dev/null; "
        f"{overwrite_cmd}"
        f"mkdir -p {self.shell_quote(remote_dir)}; "
        f"tar -xzf {self.shell_quote(remote_tmp_tgz)} -C {self.shell_quote(remote_dir)}; "
        f"rm -f {self.shell_quote(remote_tmp_tgz)}; "
        f"find {self.shell_quote(remote_dir)} -maxdepth 4 -type f | sort | head -200"
    )
    unpack_task = self.run_command(unpack_cmd)
    unpack_id = str(unpack_task.get("task_id") or unpack_task.get("id") or "")
    if not unpack_id:
        raise RuntimeError(f"SF API did not return a task ID while unpacking workflow archive: {unpack_task}")

    # A single unpack task may occasionally sit in SF API status 'new'. Give it
    # a longer window than the old per-chunk timeout.
    unpack_timeout = max(int(timeout_s), 600)
    unpack_result = self._wait_successful_command(unpack_id, timeout_s=unpack_timeout)

    return {
        "status": "uploaded",
        "transport": "sfapi_utilities_upload_tar",
        "local_dir": str(local_root),
        "remote_dir": remote_dir,
        "archive_bytes": archive_nbytes,
        "archive_sha256": archive_sha256,
        "bytes_b64": len(b64),
        "chunk_chars": None,
        "chunk_count": 0,
        "chunk_task_ids": [],
        "upload_result": upload_result,
        "unpack_task_id": unpack_id,
        "unpack_result": unpack_result,
    }


def _direct_upload_was_forbidden(exc: Exception) -> bool:
    """True for auth/scope rejection of the direct file-upload endpoint."""
    text = str(exc).lower()
    return (
        "http 401" in text
        or "http 403" in text
        or "forbidden" in text
        or "unauthorized" in text
    )


def _upload_directory_with_fallback(
    self,
    local_dir: str,
    remote_dir: str,
    overwrite: bool = True,
    chunk_chars: int = 8000,
    timeout_s: int = 240,
) -> Dict[str, Any]:
    """Prefer direct upload, but preserve the previously working command path.

    SF API endpoints are independently permission-scoped. Some credentials that
    can run the command-based workflow transport may receive HTTP 401/403 from
    ``utilities/upload``. In that case, automatically fall back to the original
    command-based uploader instead of failing the user request.
    """
    try:
        return _direct_upload_directory(
            self,
            local_dir,
            remote_dir,
            overwrite=overwrite,
            chunk_chars=chunk_chars,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        if not _direct_upload_was_forbidden(exc):
            raise

        legacy = getattr(self, "_legacy_workflow_upload_directory", None)
        if legacy is None:
            raise

        # The legacy path previously worked with this app's SF API credentials.
        # Keep its conservative 8k chunks, but extend the per-task wait so an SF
        # task that remains in status='new' for several minutes does not trigger
        # the old 240-second false failure as readily.
        result = legacy(
            local_dir,
            remote_dir,
            overwrite=overwrite,
            chunk_chars=int(chunk_chars or 8000),
            timeout_s=max(int(timeout_s), 900),
        )
        if isinstance(result, dict):
            result = dict(result)
            result["transport"] = "sfapi_command_chunk_fallback"
            result["direct_upload_error"] = str(exc)
        return result


def install() -> None:
    """Install robust workflow upload transport on the existing SF API client."""
    from tools.nersc_job_manager import NERSCSFAPIClient

    if getattr(NERSCSFAPIClient, "_direct_workflow_upload_installed", False):
        return

    # Preserve the pre-patch implementation so credentials that cannot access
    # utilities/upload can still use the command-based transfer that worked in
    # earlier versions of the app.
    NERSCSFAPIClient._legacy_workflow_upload_directory = NERSCSFAPIClient.upload_directory
    NERSCSFAPIClient.upload_directory = _upload_directory_with_fallback
    NERSCSFAPIClient._direct_workflow_upload_installed = True
