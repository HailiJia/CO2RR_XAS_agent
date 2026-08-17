from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sfapi_upload_compat import (
    _direct_upload_directory,
    _upload_directory_with_fallback,
)


class _Response:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self.text = ""
        self._payload = payload if payload is not None else {"status": "ok", "error": None}

    def json(self):
        return self._payload


class _Session:
    def __init__(self, response=None):
        self.put_calls = []
        self.response = response or _Response()

    def put(self, url, files=None):
        self.put_calls.append((url, files))
        return self.response


class _FakeClient:
    system = "perlmutter"

    def __init__(self, response=None):
        self.session = _Session(response=response)
        self.commands = []
        self.waited = []
        self.legacy_calls = []
        self._legacy_workflow_upload_directory = self._legacy_upload

    def _url(self, path):
        return "https://api.nersc.gov/api/v1.2/" + path

    def _check(self, response, context):
        if response.status_code >= 400:
            raise RuntimeError(f"{context} failed with HTTP {response.status_code}: {response.json()}")
        return response.json()

    def _local_dir_to_tar_bundle(self, local_dir):
        payload = b"fake-tar-gz"
        return (
            base64.b64encode(payload).decode("ascii"),
            hashlib.sha256(payload).hexdigest(),
            len(payload),
        )

    def shell_quote(self, value):
        return "'" + str(value).replace("'", "'\\''") + "'"

    def run_command(self, command):
        self.commands.append(command)
        return {"task_id": "123"}

    def _wait_successful_command(self, task_id, timeout_s=180):
        self.waited.append((task_id, timeout_s))
        return {"status": "ok", "exit_code": 0}

    def _legacy_upload(self, local_dir, remote_dir, overwrite=True, chunk_chars=8000, timeout_s=240):
        self.legacy_calls.append(
            {
                "local_dir": local_dir,
                "remote_dir": remote_dir,
                "overwrite": overwrite,
                "chunk_chars": chunk_chars,
                "timeout_s": timeout_s,
            }
        )
        return {
            "status": "uploaded",
            "remote_dir": remote_dir,
            "chunk_count": 2,
            "chunk_task_ids": ["legacy1", "legacy2"],
        }


def test_workflow_upload_uses_one_file_put_and_one_unpack_task(tmp_path):
    (tmp_path / "workflow_submit.sh").write_text("#!/bin/bash\n")
    client = _FakeClient()

    result = _direct_upload_directory(
        client,
        str(tmp_path),
        "/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test12",
    )

    assert result["status"] == "uploaded"
    assert result["transport"] == "sfapi_utilities_upload_tar"
    assert result["chunk_count"] == 0
    assert result["chunk_task_ids"] == []
    assert len(client.session.put_calls) == 1
    url, files = client.session.put_calls[0]
    assert "/utilities/upload/perlmutter/pscratch/" in url
    assert "/utilities/upload/perlmutter//pscratch/" not in url
    assert "file" in files
    assert len(client.commands) == 1
    assert "tar -xzf" in client.commands[0]
    assert "rm -rf '/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test12'" in client.commands[0]
    assert client.waited == [("123", 600)]


def test_forbidden_direct_upload_falls_back_to_legacy_command_transport(tmp_path):
    (tmp_path / "workflow_submit.sh").write_text("#!/bin/bash\n")
    forbidden = _Response(status_code=403, payload={"detail": "Forbidden"})
    client = _FakeClient(response=forbidden)

    result = _upload_directory_with_fallback(
        client,
        str(tmp_path),
        "/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test12",
        timeout_s=240,
    )

    assert len(client.session.put_calls) == 1
    assert len(client.legacy_calls) == 1
    assert client.legacy_calls[0]["chunk_chars"] == 8000
    assert client.legacy_calls[0]["timeout_s"] == 900
    assert result["status"] == "uploaded"
    assert result["transport"] == "sfapi_command_chunk_fallback"
    assert "HTTP 403" in result["direct_upload_error"]
    assert "Forbidden" in result["direct_upload_error"]


def test_app_installs_direct_upload_compatibility():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "CO2RR_XAS_Agent.py").read_text()
    assert "sfapi_upload_compat.install()" in source
