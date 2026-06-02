"""Tests for NERSC submit/monitor workflow helpers."""

import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.nersc_workflow import NERSCJobManager, discover_submit_scripts, execute_nersc_full_workflow


class FakeSLURMRunner:
    def __init__(self):
        self.commands = []
        self.next_job_id = 1000

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        exe = Path(cmd[0]).name
        if exe == "sbatch":
            self.next_job_id += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{self.next_job_id}\n", stderr="")
        if exe == "squeue":
            # Empty squeue means the job has left the active queue; manager then checks sacct.
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        if exe == "sacct":
            return subprocess.CompletedProcess(cmd, 0, stdout="COMPLETED\n", stderr="")
        return subprocess.CompletedProcess(cmd, 1, stdout="", stderr="unexpected command")


def test_discover_submit_scripts_from_generated_inputs_shape():
    inputs = {
        "relax": {"submit_script": "/tmp/run/relax/submit_relax.sh"},
        "xas": {
            "Cu_K_edge": {
                "FEFF": {"submit": "/tmp/run/xas/Cu_K_edge/FEFF/submit.sh"},
                "FDMNES": {"submit": "/tmp/run/xas/Cu_K_edge/FDMNES/submit.sh"},
                "VASP": {"submit": "/tmp/run/xas/Cu_K_edge/VASP/submit.sh"},
            }
        },
    }
    feff_scripts = discover_submit_scripts(inputs, software_filter="FEFF")
    assert len(feff_scripts) == 1
    assert feff_scripts[0]["software"] == "FEFF"
    assert feff_scripts[0]["absorber"] == "Cu"
    assert feff_scripts[0]["edge"] == "K"

    all_scripts = discover_submit_scripts(inputs, software_filter="all")
    assert len(all_scripts) == 4


def test_nersc_workflow_can_submit_and_monitor_with_fake_slurm():
    runner = FakeSLURMRunner()
    manager = NERSCJobManager(runner=runner)
    structure_file = REPO_ROOT / "example_ouput" / "VASP" / "XAS" / "POSCAR"

    with tempfile.TemporaryDirectory() as tmp:
        result = execute_nersc_full_workflow(
            structure_file=str(structure_file),
            output_dir=str(Path(tmp) / "nersc"),
            software="FEFF",
            submit=True,
            monitor=True,
            parse_when_complete=False,
            poll_interval=1,
            job_manager=manager,
            nersc_account="m1234",
            nersc_queue="regular",
            nersc_nodes=1,
            nersc_walltime="00:30:00",
            edge_override="K",
        )

    assert result["status"] == "success"
    assert len(result["submitted_jobs"]) == 1
    assert result["submitted_jobs"][0]["software"] == "FEFF"
    assert result["submitted_jobs"][0]["state"] == "COMPLETED"
    assert any(Path(cmd[0]).name == "sbatch" for cmd in runner.commands)
    assert any(Path(cmd[0]).name == "squeue" for cmd in runner.commands)
    assert any(Path(cmd[0]).name == "sacct" for cmd in runner.commands)


def test_agent_exposes_nersc_dry_run_workflow():
    from agent.co2rr_xas_agent import create_agent

    structure_file = REPO_ROOT / "example_ouput" / "VASP" / "XAS" / "POSCAR"
    with tempfile.TemporaryDirectory() as tmp:
        agent = create_agent(use_llm=False)
        result = agent.run_nersc_workflow(
            structure_file=str(structure_file),
            output_dir=str(Path(tmp) / "agent_dry_run"),
            software="FEFF",
            submit=False,
            nersc_account="m1234",
            nersc_nodes=1,
            edge_override="K",
        )

    assert result["status"] == "success"
    assert result["submitted_jobs"]
    assert result["submitted_jobs"][0]["state"] == "DRY_RUN"


if __name__ == "__main__":
    test_discover_submit_scripts_from_generated_inputs_shape()
    test_nersc_workflow_can_submit_and_monitor_with_fake_slurm()
    test_agent_exposes_nersc_dry_run_workflow()
    print("NERSC workflow tests passed")
