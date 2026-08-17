from pathlib import Path

import pytest

from tools.structure_generator import StructureGenerator
from tools.web_nersc_workflow_integration import (
    prepare_single_structure_workflow_package,
    run_remote_workflow_script,
)


def test_prepare_single_structure_workflow_package_generates_relax_and_workflow_scripts(tmp_path):
    structure = StructureGenerator().generate_surface(
        element="Cu",
        facet="111",
        supercell=(1, 1),
        layers=2,
        vacuum=10.0,
    )
    structure.setdefault("metadata", {}).update({"element": "Cu", "facet": "111", "adsorbate": "clean"})

    result = prepare_single_structure_workflow_package(
        structure=structure,
        package_root=tmp_path / "web_xas_agent",
        repo_root=Path.cwd(),
        absorber="Cu",
        edge="K",
        relaxation_settings={"encut": 520, "nsw": 5},
        account="m5268",
        queue="regular",
        nodes=1,
        walltime="00:30:00",
    )

    root = Path(result["package_root"])
    assert (root / "01_structure" / "submit_relax.sh").is_file()
    assert (root / "01_structure" / "make_potcar.sh").is_file()
    assert (root / "workflow_submit.sh").is_file()
    assert (root / "workflow_xas.sh").is_file()
    assert (root / "workflow_status.sh").is_file()
    assert (root / "workflow_restart.sh").is_file()
    assert (root / "workflow_cancel.sh").is_file()
    assert (root / "workflow_state.py").is_file()
    assert (root / "workflow_manifest.json").is_file()
    assert (root / "workflow_state.json").is_file()

    submit_text = (root / "workflow_submit.sh").read_text()
    xas_text = (root / "workflow_xas.sh").read_text()
    relax_text = (root / "01_structure" / "submit_relax.sh").read_text()
    assert "sbatch --parsable 01_structure/submit_relax.sh" in submit_text
    assert "--dependency=afterok" in submit_text
    assert "remote_xas_from_contcar.py" in xas_text
    assert "01_structure/CONTCAR" in xas_text
    assert "./make_potcar.sh" in relax_text


class _FakeWorkflowClient:
    def __init__(self, *, task_status="completed", result_json=None, task_id="8573678"):
        self.task_status = task_status
        self.result_json = result_json or {}
        self.task_id = task_id
        self.command = ""

    def shell_quote(self, text):
        return "'" + str(text).replace("'", "'\\''") + "'"

    def run_command(self, command):
        self.command = command
        return {"task_id": self.task_id, "status": "ok", "error": None}

    def wait_task(self, task_id, timeout_s=120):
        assert task_id == self.task_id
        return {"id": task_id, "status": self.task_status, "result": "unused"}

    def task_result_json(self, _task_result):
        return self.result_json


def _successful_submit_output(job_id="60012345"):
    return (
        "Using Python: python3 (Python 3.11)\n"
        "__CO2RR_WORKFLOW_EXIT_CODE__=0\n"
        "__CO2RR_WORKFLOW_STATE_BEGIN__\n"
        '{"stage":"relax_submitted","jobs":{"relax":{"job_id":"' + job_id + '"}},"events":[]}\n'
        "__CO2RR_WORKFLOW_STATE_END__\n"
    )


def test_run_remote_workflow_script_requires_registered_relax_job_for_start():
    client = _FakeWorkflowClient(
        result_json={"status": "ok", "output": _successful_submit_output(), "error": None}
    )

    result = run_remote_workflow_script(
        client,
        "/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/test10",
        "workflow_submit.sh",
    )

    assert result["status"] == "completed"
    assert result["workflow_state"]["jobs"]["relax"]["job_id"] == "60012345"
    assert "__CO2RR_WORKFLOW_EXIT_CODE__" in client.command
    assert "workflow_state.json" in client.command


def test_run_remote_workflow_script_raises_when_shell_command_failed():
    client = _FakeWorkflowClient(
        result_json={
            "status": "ok",
            "output": "sbatch: error: Batch job submission failed\n__CO2RR_WORKFLOW_EXIT_CODE__=1\n",
            "error": None,
        }
    )

    with pytest.raises(RuntimeError, match=r"shell exit code=1") as exc_info:
        run_remote_workflow_script(client, "/pscratch/test10", "workflow_submit.sh")

    assert "8573678" in str(exc_info.value)
    assert "sbatch: error" in str(exc_info.value)


def test_run_remote_workflow_script_rejects_false_success_without_relax_job():
    output = (
        "__CO2RR_WORKFLOW_EXIT_CODE__=0\n"
        "__CO2RR_WORKFLOW_STATE_BEGIN__\n"
        '{"stage":"prepared","jobs":{},"events":[]}\n'
        "__CO2RR_WORKFLOW_STATE_END__\n"
    )
    client = _FakeWorkflowClient(
        result_json={"status": "ok", "output": output, "error": None}
    )

    with pytest.raises(RuntimeError, match=r"no relaxation Slurm job was registered"):
        run_remote_workflow_script(client, "/pscratch/test10", "workflow_submit.sh")


def test_run_remote_workflow_script_rejects_sfapi_task_failure():
    client = _FakeWorkflowClient(
        task_status="failed",
        result_json={"status": "error", "output": "permission denied", "error": "command failed"},
    )

    with pytest.raises(RuntimeError, match=r"SF API task status=failed"):
        run_remote_workflow_script(client, "/pscratch/test10", "workflow_status.sh")
