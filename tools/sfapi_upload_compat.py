from __future__ import annotations

import base64
import io
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Dict


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
    command tasks.  One task stuck in SF API status ``new`` caused the whole
    upload to time out.  NERSC provides ``PUT /utilities/upload`` specifically
    for small files, so upload one tar.gz and use one command task to verify and
    unpack it.

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

    # Match NERSC's official sfapi_client implementation: multipart PUT to
    # utilities/upload/{machine}/{path}.  This request is synchronous and does
    # not create one SF task per archive chunk.
    upload_url = self._url(f"utilities/upload/{self.system}/{remote_tmp_tgz}")
    files = {
        "file": (
            PurePosixPath(remote_tmp_tgz).name,
            io.BytesIO(archive_bytes),
            "application/gzip",
        )
    }
    response = self.session.put(upload_url, files=files)
    upload_result = self._check(response, "workflow archive upload")

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
    # a longer window than the old per-chunk timeout while avoiding hundreds of
    # sequential waits.
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


def install() -> None:
    """Install the direct-file workflow uploader on the existing SF API client."""
    from tools.nersc_job_manager import NERSCSFAPIClient

    if getattr(NERSCSFAPIClient, "_direct_workflow_upload_installed", False):
        return
    NERSCSFAPIClient.upload_directory = _direct_upload_directory
    NERSCSFAPIClient._direct_workflow_upload_installed = True
