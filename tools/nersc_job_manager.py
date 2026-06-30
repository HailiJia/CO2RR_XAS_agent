"""
NERSC Superfacility API helpers for the CO2RR XAS Agent web app.

This module is intentionally optional: the web app can still generate and
package inputs without authlib/requests or NERSC credentials. When credentials
are available, it can upload small input folders, submit Slurm scripts, poll SF
API tasks/Slurm jobs, download selected outputs, and build a first-pass ISAAC
record from local/downloaded output folders.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import os
import re
import tarfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SFAPI_BASE_URL = "https://api.nersc.gov/api/v1.2"
SFAPI_TOKEN_URL = "https://oidc.nersc.gov/c2id/token"
NERSC_JOB_MANAGER_UPDATE_TAG = "v27_2026-06-30_safe_remote_ls_and_delete"


class NERSCSFAPIError(RuntimeError):
    """Raised when a NERSC Superfacility API request fails."""


def _load_private_key(private_key: Optional[str] = None, private_key_path: Optional[str] = None) -> str:
    """Load a private key from direct text, a path, or environment variables."""
    if private_key and "BEGIN" in private_key:
        return private_key

    path = private_key_path or os.environ.get("NERSC_SFAPI_PRIVATE_KEY_PATH", "")
    if path:
        expanded = os.path.expandvars(os.path.expanduser(path))
        if os.path.isfile(expanded):
            return Path(expanded).read_text()

    env_key = os.environ.get("NERSC_SFAPI_PRIVATE_KEY", "")
    if env_key:
        # Allows either a literal PEM or a PEM encoded with escaped newlines.
        return env_key.replace("\\n", "\n")

    raise NERSCSFAPIError(
        "No SF API private key found. Set NERSC_SFAPI_PRIVATE_KEY_PATH or "
        "NERSC_SFAPI_PRIVATE_KEY, or provide a private_key_path in the app."
    )


def make_sfapi_session(
    client_id: Optional[str] = None,
    private_key_path: Optional[str] = None,
    private_key: Optional[str] = None,
):
    """Create an authenticated OAuth2 session for the NERSC SF API."""
    try:
        from authlib.integrations.requests_client import OAuth2Session
        from authlib.oauth2.rfc7523 import PrivateKeyJWT
    except Exception as exc:  # pragma: no cover - runtime dependency check
        raise NERSCSFAPIError(
            "Missing dependency authlib. Install it with: pip install authlib requests"
        ) from exc

    client_id = client_id or os.environ.get("NERSC_SFAPI_CLIENT_ID", "")
    if not client_id:
        raise NERSCSFAPIError("Missing SF API client ID. Set NERSC_SFAPI_CLIENT_ID or enter it in the app.")

    key_text = _load_private_key(private_key=private_key, private_key_path=private_key_path)
    session = OAuth2Session(
        client_id,
        key_text,
        PrivateKeyJWT(SFAPI_TOKEN_URL),
        grant_type="client_credentials",
        token_endpoint=SFAPI_TOKEN_URL,
    )
    session.fetch_token()
    return session


class NERSCSFAPIClient:
    """Small wrapper around NERSC Superfacility API endpoints used by the app."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        private_key_path: Optional[str] = None,
        private_key: Optional[str] = None,
        system: str = "perlmutter",
        base_url: str = SFAPI_BASE_URL,
    ):
        self.system = system or "perlmutter"
        self.base_url = base_url.rstrip("/")
        self.session = make_sfapi_session(
            client_id=client_id,
            private_key_path=private_key_path,
            private_key=private_key,
        )

    def _url(self, path: str) -> str:
        return f"{self.base_url}/{path.lstrip('/')}"

    def _check(self, response, context: str) -> Dict[str, Any]:
        try:
            payload = response.json()
        except Exception:
            payload = {"text": response.text}
        if response.status_code >= 400:
            raise NERSCSFAPIError(f"{context} failed with HTTP {response.status_code}: {payload}")
        if isinstance(payload, dict) and payload.get("error"):
            raise NERSCSFAPIError(f"{context} returned error: {payload.get('error')}")
        return payload

    def status(self) -> Dict[str, Any]:
        r = self.session.get(self._url(f"status/{self.system}"))
        return self._check(r, "status")

    def run_command(self, executable: str) -> Dict[str, Any]:
        r = self.session.post(
            self._url(f"utilities/command/{self.system}"),
            data={"executable": executable},
        )
        return self._check(r, "command")

    def submit_job(self, remote_submit_script: str) -> Dict[str, Any]:
        r = self.session.post(
            self._url(f"compute/jobs/{self.system}"),
            data={"job": remote_submit_script, "isPath": True},
        )
        return self._check(r, "submit job")

    def task(self, task_id: str) -> Dict[str, Any]:
        r = self.session.get(self._url(f"tasks/{task_id}"))
        return self._check(r, "task")

    def wait_task(self, task_id: str, timeout_s: int = 120, poll_s: float = 2.0) -> Dict[str, Any]:
        start = time.time()
        last = None
        while time.time() - start < timeout_s:
            last = self.task(task_id)
            status = str(last.get("status", "")).lower()
            if status in {"completed", "complete", "failed", "error", "cancelled", "canceled"}:
                return last
            time.sleep(poll_s)
        raise NERSCSFAPIError(f"Timed out waiting for SF API task {task_id}. Last response: {last}")

    def task_result_json(self, task_response: Dict[str, Any]) -> Dict[str, Any]:
        result = task_response.get("result")
        if isinstance(result, dict):
            return result
        if isinstance(result, str) and result.strip():
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                return {"raw_result": result}
        return {}

    def list_remote(self, remote_path: str) -> Dict[str, Any]:
        r = self.session.get(self._url(f"utilities/ls/{self.system}{remote_path}"))
        return self._check(r, "ls")

    def remote_ls(self, remote_dir: str, timeout_s: int = 60, max_depth: int = 1) -> Dict[str, Any]:
        """List files in a remote directory using a shell command.

        This is more useful than the raw utilities/ls endpoint for the web UI
        because it returns plain text that can be parsed into a file dropdown.
        """
        remote_dir = str(remote_dir).rstrip("/") or "."
        cmd = (
            "set -e; "
            f"cd {self.shell_quote(remote_dir)}; "
            "echo '== pwd =='; pwd; "
            "echo; echo '== ls -lah =='; ls -lah; "
            f"echo; echo '== files, maxdepth {int(max_depth)} =='; "
            f"find . -maxdepth {int(max_depth)} -type f -printf '%P\\n' | sort"
        )
        task = self.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            return task
        result = self.wait_task(task_id, timeout_s=timeout_s)
        return self.task_result_json(result)

    def remote_rm_dir(self, remote_dir: str, timeout_s: int = 60) -> Dict[str, Any]:
        """Remove one remote run directory after app-side validation."""
        remote_dir = str(remote_dir).rstrip("/")
        if not remote_dir:
            raise NERSCSFAPIError("Missing remote directory to delete.")
        cmd = (
            "set -e; "
            f"if [ ! -e {self.shell_quote(remote_dir)} ]; then echo 'ERROR: target does not exist: {remote_dir}'; exit 44; fi; "
            f"if [ ! -d {self.shell_quote(remote_dir)} ]; then echo 'ERROR: target is not a directory: {remote_dir}'; exit 45; fi; "
            f"echo 'Deleting remote directory:' {self.shell_quote(remote_dir)}; "
            f"rm -rf -- {self.shell_quote(remote_dir)}; "
            f"if [ -e {self.shell_quote(remote_dir)} ]; then echo 'ERROR: delete failed'; exit 46; fi; "
            "echo 'Delete completed.'"
        )
        task = self.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            return task
        result = self.wait_task(task_id, timeout_s=timeout_s)
        return self.task_result_json(result)

    def shell_quote(self, text: str) -> str:
        return "'" + str(text).replace("'", "'\\''") + "'"

    def remote_tail(self, remote_file: str, n_lines: int = 80, timeout_s: int = 60) -> Dict[str, Any]:
        cmd = f"tail -n {int(n_lines)} {self.shell_quote(remote_file)} 2>&1"
        task = self.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            return task
        result = self.wait_task(task_id, timeout_s=timeout_s)
        return self.task_result_json(result)

    def slurm_status(self, job_id: str, timeout_s: int = 60) -> Dict[str, Any]:
        job_id = str(job_id).strip()
        if not job_id:
            raise NERSCSFAPIError("Missing Slurm job ID.")
        cmd = (
            f"squeue -j {self.shell_quote(job_id)} -h -o '%i|%j|%T|%M|%D|%R' 2>/dev/null; "
            f"sacct -j {self.shell_quote(job_id)} --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P -n 2>/dev/null | head -20"
        )
        task = self.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            return task
        result = self.wait_task(task_id, timeout_s=timeout_s)
        return self.task_result_json(result)

    def list_user_jobs(self, username: Optional[str] = None, timeout_s: int = 60) -> Dict[str, Any]:
        """List current and recent Slurm jobs for the authenticated user or username."""
        user_part = f"-u {self.shell_quote(username)}" if username else "--me"
        sacct_user_part = f"-u {self.shell_quote(username)}" if username else ""
        cmd = (
            "echo '== Current queued/running jobs =='; "
            f"squeue {user_part} -o '%i|%j|%u|%T|%M|%D|%R' 2>/dev/null; "
            "echo; echo '== Recent accounting records =='; "
            f"sacct {sacct_user_part} --starttime now-7days --format=JobID,JobName,State,ExitCode,Elapsed,Start,End -P -n 2>/dev/null | head -80"
        )
        task = self.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        if not task_id:
            return task
        result = self.wait_task(task_id, timeout_s=timeout_s)
        return self.task_result_json(result)

    def _local_dir_to_tar_bundle(self, local_dir: str) -> Tuple[str, str, int]:
        """Return a base64-encoded tar.gz archive plus binary sha256 and byte count."""
        local_root = Path(local_dir).resolve()
        if not local_root.is_dir():
            raise NERSCSFAPIError(f"Local directory does not exist: {local_dir}")
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            for path in sorted(local_root.rglob("*")):
                if path.is_file():
                    tar.add(path, arcname=str(path.relative_to(local_root)))
        archive_bytes = buffer.getvalue()
        return (
            base64.b64encode(archive_bytes).decode("ascii"),
            hashlib.sha256(archive_bytes).hexdigest(),
            len(archive_bytes),
        )

    def _wait_successful_command(self, task_id: str, timeout_s: int = 180) -> Dict[str, Any]:
        """Wait for a command task and raise if the remote command exited nonzero."""
        result = self.wait_task(task_id, timeout_s=timeout_s)
        parsed = self.task_result_json(result) if isinstance(result, dict) else {}
        status = str(result.get("status", "")).lower() if isinstance(result, dict) else ""
        exit_code = parsed.get("exit_code")
        if status in {"failed", "error", "cancelled", "canceled"} or (exit_code not in (None, 0, "0")):
            raise NERSCSFAPIError(f"Remote command task {task_id} failed: {parsed or result}")
        return parsed or result

    def upload_directory(
        self,
        local_dir: str,
        remote_dir: str,
        overwrite: bool = True,
        chunk_chars: int = 8000,
        timeout_s: int = 240,
    ) -> Dict[str, Any]:
        """Upload a small/medium input directory by chunking a base64 tarball.

        The previous larger-chunk upload could create a corrupt remote archive if
        a command was truncated by the API. This version uses smaller chunks,
        checks every append task, verifies the decoded tar.gz sha256 on NERSC,
        tests the archive with tar -tzf, and only then unpacks it.
        """
        remote_dir = remote_dir.rstrip("/")
        b64, archive_sha256, archive_nbytes = self._local_dir_to_tar_bundle(local_dir)
        marker = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        remote_tmp_b64 = f"{remote_dir}/.web_xas_agent_upload_{marker}.tar.gz.b64"
        remote_tmp_tgz = f"{remote_dir}/.web_xas_agent_upload_{marker}.tar.gz"

        init_cmd = f"mkdir -p {self.shell_quote(remote_dir)} && rm -f {self.shell_quote(remote_tmp_b64)} {self.shell_quote(remote_tmp_tgz)}"
        if overwrite:
            init_cmd = f"rm -rf {self.shell_quote(remote_dir)} && mkdir -p {self.shell_quote(remote_dir)} && rm -f {self.shell_quote(remote_tmp_b64)} {self.shell_quote(remote_tmp_tgz)}"
        init_task = self.run_command(init_cmd)
        init_id = str(init_task.get("task_id") or init_task.get("id") or "")
        if init_id:
            self._wait_successful_command(init_id, timeout_s=timeout_s)

        chunk_task_ids = []
        for i in range(0, len(b64), int(chunk_chars)):
            chunk = b64[i:i + int(chunk_chars)]
            # Use a quoted here-doc so base64 characters are appended exactly.
            cmd = f"cat >> {self.shell_quote(remote_tmp_b64)} <<'EOF_WEB_XAS_CHUNK'\n{chunk}\nEOF_WEB_XAS_CHUNK"
            task = self.run_command(cmd)
            task_id = str(task.get("task_id") or task.get("id") or "")
            if task_id:
                chunk_task_ids.append(task_id)
                self._wait_successful_command(task_id, timeout_s=timeout_s)

        expected_sha = self.shell_quote(archive_sha256)
        unpack_cmd = (
            f"set -e; "
            f"base64 -d {self.shell_quote(remote_tmp_b64)} > {self.shell_quote(remote_tmp_tgz)}; "
            f"remote_sha=$(sha256sum {self.shell_quote(remote_tmp_tgz)} | awk '{{print $1}}'); "
            f"test \"$remote_sha\" = {expected_sha}; "
            f"tar -tzf {self.shell_quote(remote_tmp_tgz)} >/dev/null; "
            f"tar -xzf {self.shell_quote(remote_tmp_tgz)} -C {self.shell_quote(remote_dir)}; "
            f"rm -f {self.shell_quote(remote_tmp_b64)} {self.shell_quote(remote_tmp_tgz)}; "
            f"find {self.shell_quote(remote_dir)} -maxdepth 4 -type f | sort | head -200"
        )
        unpack_task = self.run_command(unpack_cmd)
        unpack_id = str(unpack_task.get("task_id") or unpack_task.get("id") or "")
        unpack_result = self._wait_successful_command(unpack_id, timeout_s=timeout_s) if unpack_id else unpack_task
        return {
            "status": "uploaded",
            "local_dir": str(Path(local_dir).resolve()),
            "remote_dir": remote_dir,
            "archive_bytes": archive_nbytes,
            "archive_sha256": archive_sha256,
            "bytes_b64": len(b64),
            "chunk_chars": int(chunk_chars),
            "chunk_count": len(chunk_task_ids),
            "chunk_task_ids": chunk_task_ids,
            "unpack_task_id": unpack_id,
            "unpack_result": unpack_result,
        }

    def submit_and_wait_for_jobid(self, remote_submit_script: str, timeout_s: int = 120) -> Dict[str, Any]:
        submit = self.submit_job(remote_submit_script)
        task_id = str(submit.get("task_id") or submit.get("id") or "")
        result_task = self.wait_task(task_id, timeout_s=timeout_s) if task_id else submit
        result_json = self.task_result_json(result_task) if isinstance(result_task, dict) else {}
        job_id = result_json.get("jobid") or result_json.get("job_id") or result_json.get("jobId")
        if not job_id:
            raw_text = json.dumps(result_json)
            match = re.search(r'(?:Submitted batch job|jobid[\'"=: ]+)\s*([0-9]+)', raw_text, flags=re.IGNORECASE)
            if match:
                job_id = match.group(1)
        return {
            "status": "submitted",
            "remote_submit_script": remote_submit_script,
            "task_id": task_id,
            "task": result_task,
            "result": result_json,
            "job_id": job_id,
        }

    def download_remote_archive(
        self,
        remote_dir: str,
        include_glob: str = "*",
        timeout_s: int = 180,
    ) -> Tuple[bytes, Dict[str, Any]]:
        """Download a remote directory archive through command output.

        This is intended for moderate text outputs. For large WAVECAR/CHGCAR or
        full output folders, use Globus/DTN transfer instead.
        """
        remote_dir = remote_dir.rstrip("/")
        # Include matching files recursively. The include_glob string is a space-separated
        # list of filename globs, e.g. "*.out OUTCAR CONTCAR *_conv.txt".
        patterns = [p for p in str(include_glob or "*").split() if p.strip()]
        find_parts = []
        for pattern in patterns:
            find_parts.append(f"-name {self.shell_quote(pattern)}")
        find_expr = " -o ".join(find_parts) if find_parts else "-type f"
        cmd = (
            f"cd {self.shell_quote(remote_dir)} && "
            f"find . -type f \\( {find_expr} \\) -print0 | tar --null -T - -czf - 2>/dev/null | base64 -w 0"
        )
        task = self.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        task_resp = self.wait_task(task_id, timeout_s=timeout_s) if task_id else task
        result = self.task_result_json(task_resp)
        raw = result.get("output") or result.get("raw_result") or ""
        if not raw.strip():
            raise NERSCSFAPIError(f"No archive output returned. Task result: {result}")
        data = base64.b64decode(raw.strip())
        return data, {"task_id": task_id, "task": task_resp, "result": result, "remote_dir": remote_dir, "include_glob": include_glob}


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _media_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".xml"):
        return "application/xml"
    if name.endswith((".tar.gz", ".tgz", ".zip")):
        return "application/gzip" if not name.endswith(".zip") else "application/zip"
    return "text/plain"


def _content_role(path: Path) -> str:
    name = path.name
    lower = name.lower()
    if name in {"POSCAR", "CONTCAR"} or lower.endswith((".cif", ".xyz")):
        return "input_or_relaxed_structure"
    if name in {"INCAR", "KPOINTS", "make_potcar.sh", "submit.sh", "submit_relax.sh", "feff.inp", "HEADER", "PARAMETERS", "POTENTIALS", "ATOMS", "fdmfile.txt"} or name.endswith("_in.txt"):
        return "workflow_recipe"
    if name in {"xmu.dat", "chi.dat", "paths.dat"} or name.endswith("_conv.txt") or name == "CORE_DIELECTRIC_IMAG.dat":
        return "reduction_product"
    if name in {"OUTCAR", "vasprun.xml", "OSZICAR", "vasp.out", "fdmnes.out", "feff.out"}:
        return "raw_simulation_output"
    return "auxiliary_file"


def _parse_numeric_spectrum(path: Path, max_points: int = 20) -> Optional[Dict[str, Any]]:
    rows = []
    try:
        for line in path.read_text(errors="ignore").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith(("#", "*")):
                continue
            parts = stripped.replace(",", " ").split()
            nums = []
            for part in parts[:4]:
                try:
                    nums.append(float(part))
                except ValueError:
                    pass
            if len(nums) >= 2:
                rows.append(nums[:2])
            if len(rows) >= max_points:
                break
    except Exception:
        return None
    if not rows:
        return None
    return {
        "file": str(path.name),
        "columns": ["energy_or_relative_energy", "intensity_or_mu"],
        "preview_rows": rows,
        "n_preview_rows": len(rows),
    }


def build_isaac_record_from_outputs(
    output_dir: str,
    software: str = "auto",
    sample_name: str = "not_specified",
    notes: str = "Generated by CO2RR XAS Agent web_xas_agent.",
) -> Dict[str, Any]:
    """Build a first-pass ISAAC-style JSON record from a local output folder."""
    root = Path(output_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    metadata_files = sorted(root.rglob("isaac_run_metadata.json"))
    existing_metadata = None
    if metadata_files:
        try:
            existing_metadata = json.loads(metadata_files[0].read_text())
        except Exception:
            existing_metadata = None

    include_names = {
        "POSCAR", "CONTCAR", "INCAR", "KPOINTS", "make_potcar.sh", "submit.sh", "submit_relax.sh",
        "feff.inp", "HEADER", "PARAMETERS", "POTENTIALS", "ATOMS", "fdmfile.txt", "xmu.dat", "chi.dat",
        "paths.dat", "files.dat", "log1.dat", "OUTCAR", "vasprun.xml", "OSZICAR", "vasp.out", "fdmnes.out", "feff.out",
        "CORE_DIELECTRIC_IMAG.dat", "isaac_run_metadata.json", "isaac_record_draft.json",
    }
    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in include_names or path.name.endswith(("_in.txt", "_conv.txt", "_bav.txt")):
            files.append(path)

    assets = []
    spectra = []
    for idx, path in enumerate(files):
        rel = path.relative_to(root).as_posix()
        assets.append({
            "asset_id": f"asset_{idx:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', rel)[:80]}",
            "content_role": _content_role(path),
            "uri": rel,
            "media_type": _media_type(path),
            "sha256": _sha256_file(path),
            "notes": f"File collected from {rel}.",
        })
        if path.name in {"xmu.dat", "CORE_DIELECTRIC_IMAG.dat"} or path.name.endswith("_conv.txt"):
            preview = _parse_numeric_spectrum(path)
            if preview:
                preview["relative_uri"] = rel
                spectra.append(preview)

    if software == "auto":
        names = {p.name for p in files}
        if "xmu.dat" in names or "feff.inp" in names:
            software = "FEFF"
        elif any(p.name.endswith("_conv.txt") or p.name == "fdmfile.txt" for p in files):
            software = "FDMNES"
        elif "OUTCAR" in names or "vasp.out" in names or "vasprun.xml" in names:
            software = "VASP"
        else:
            software = "not_specified"

    base_record = existing_metadata if isinstance(existing_metadata, dict) else {}
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    record = {
        "record_type": "ISAAC_simulation_record_draft",
        "generator": {
            "name": "CO2RR XAS Agent web_xas_agent",
            "version": NERSC_JOB_MANAGER_UPDATE_TAG,
            "generated_utc": now,
        },
        "source": {
            "local_output_dir": str(root),
            "metadata_source": str(metadata_files[0].relative_to(root)) if metadata_files else "generated_from_output_files",
        },
        "sample": base_record.get("sample", {
            "material": {"name": sample_name, "formula": "not_specified", "provenance": "theoretical", "notes": notes},
            "sample_form": "slab_model",
        }),
        "context": base_record.get("context", {"environment": "in_silico", "temperature_K": 0, "notes": notes}),
        "system": base_record.get("system", {
            "domain": "computational",
            "technique": "xas_simulation_or_dft_relaxation",
            "facility": {"facility_name": "NERSC", "cluster": "Perlmutter"},
            "instrument": {"instrument_type": "simulation_engine", "instrument_name": software},
        }),
        "run": base_record.get("run", {"software": software, "scheduler": "SLURM", "exit_code": "not_parsed"}),
        "measurement": base_record.get("measurement", {"processing": {"type": "simulated_spectrum"}, "qc": {"status": "not_reviewed"}}),
        "assets": assets,
        "descriptors": {
            "outputs": spectra,
            "file_count": len(files),
        },
        "links": base_record.get("links", []),
    }
    return record


def write_isaac_record(output_dir: str, record: Dict[str, Any], filename: str = "isaac_record_draft.json") -> str:
    path = Path(output_dir).resolve() / filename
    path.write_text(json.dumps(record, indent=2))
    return str(path)


def extract_archive_bytes(archive_bytes: bytes, output_dir: str, clean: bool = True) -> str:
    """Extract a downloaded tar.gz archive to a local folder."""
    out = Path(output_dir).resolve()
    if clean and out.exists():
        for path in sorted(out.rglob("*"), reverse=True):
            if path.is_file() or path.is_symlink():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
    out.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(archive_bytes), mode="r:gz") as tar:
        def safe_members():
            for member in tar.getmembers():
                target = (out / member.name).resolve()
                if not str(target).startswith(str(out)):
                    continue
                yield member
        tar.extractall(out, members=safe_members())
    return str(out)
