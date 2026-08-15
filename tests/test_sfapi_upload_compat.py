from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.sfapi_upload_compat import _direct_upload_directory


class _Response:
    status_code = 200
    text = ""

    def json(self):
        return {"status": "ok", "error": None}


class _Session:
    def __init__(self):
        self.put_calls = []

    def put(self, url, files=None):
        self.put_calls.append((url, files))
        return _Response()


class _FakeClient:
    system = "perlmutter"

    def __init__(self):
        self.session = _Session()
        self.commands = []
        self.waited = []

    def _url(self, path):
        return "https://api.nersc.gov/api/v1.2/" + path

    def _check(self, response, context):
        assert response.status_code == 200
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


def test_workflow_upload_uses_one_file_put_and_one_unpack_task(tmp_path):
    (tmp_path / "workflow_submit.sh").write_text("#!/bin/bash\n")
    client = _FakeClient()

    result = _direct_upload_directory(
        client,
        str(tmp_path),
        "/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test_12",
    )

    assert result["status"] == "uploaded"
    assert result["transport"] == "sfapi_utilities_upload_tar"
    assert result["chunk_count"] == 0
    assert result["chunk_task_ids"] == []
    assert len(client.session.put_calls) == 1
    url, files = client.session.put_calls[0]
    assert "/utilities/upload/perlmutter//pscratch/" in url
    assert "file" in files
    assert len(client.commands) == 1
    assert "tar -xzf" in client.commands[0]
    assert "rm -rf '/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test_12'" in client.commands[0]
    assert client.waited == [("123", 600)]


def test_app_installs_direct_upload_compatibility():
    repo_root = Path(__file__).resolve().parents[1]
    source = (repo_root / "web_app" / "CO2RR_XAS_Agent.py").read_text()
    assert "sfapi_upload_compat.install()" in source
