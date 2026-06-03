"""NERSC submit/monitor/finalize helpers for CO2RR XAS workflows.

The functions in this module are intentionally small wrappers around standard
NERSC/SLURM commands (``sbatch``, ``squeue``, and ``sacct``).  They are safe to
import off-NERSC; command execution only happens when submit/monitor methods are
called.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from custodian.custodian import Custodian, ErrorHandler, Job

from tools.result_parser import execute_result_parsing
from workflow.xas_custodian import run_xas_output_validation
from tools.xas_input_generator import execute_xas_input_generation

CommandRunner = Callable[..., subprocess.CompletedProcess]

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "BOOT_FAIL",
    "DEADLINE",
    "REVOKED",
}
SUCCESS_STATES = {"COMPLETED"}


class NERSCWorkflowStep(Job):
    """Custodian-compatible in-process workflow step."""

    def __init__(self, name: str, action: Callable[[Dict[str, Any]], None], context: Dict[str, Any]):
        self.step_name = name
        self.action = action
        self.context = context

    @property
    def name(self) -> str:
        return self.step_name

    def setup(self, directory: str = "./") -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)

    def run(self, directory: str = "./") -> None:
        self.action(self.context)
        return None

    def postprocess(self, directory: str = "./") -> None:
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "name": self.step_name,
        }


class NERSCWorkflowRecoveryHandler(ErrorHandler):
    """Custodian error handler for recoverable orchestration issues."""

    raises_runtime_error = False

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def check(self, directory: str = "./") -> bool:
        return bool(self.context.get("custodian_recovery_required"))

    def correct(self, directory: str = "./") -> Dict[str, Any]:
        recovery = self.context.pop("custodian_recovery_required", None)
        if recovery == "missing_submit_command":
            self.context["submit"] = False
            self.context["monitor"] = False
            self.context["parse_when_complete"] = False
            message = "custodian recovered missing sbatch/NERSC environment by switching to submit=False dry run."
            self.context.setdefault("recoveries", []).append(message)
            self.context.setdefault("warnings", []).append("submit=True requested, but sbatch/NERSC_HOST is unavailable.")
            return {"errors": [recovery], "actions": [message]}
        return {"errors": [str(recovery)], "actions": []}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
        }


def run_custodian_steps(steps: List[NERSCWorkflowStep], output_dir: str, context: Dict[str, Any]) -> Dict[str, Any]:
    """Run workflow steps with custodian job management and recovery."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    custodian = Custodian(
        handlers=[NERSCWorkflowRecoveryHandler(context)],
        jobs=steps,
        max_errors=int(os.environ.get("CO2RR_CUSTODIAN_MAX_ERRORS", context.get("custodian_max_errors", 5))),
        directory=output_dir,
        terminate_on_nonzero_returncode=False,
    )
    run_log = custodian.run()
    return {
        "backend": "custodian",
        "run_log": run_log,
        "warnings": context.get("warnings", []),
        "recovery_actions": context.get("recoveries", []),
    }


@dataclass
class NERSCJob:
    """Bookkeeping for a submitted NERSC/SLURM job."""

    job_id: str
    script: str
    output_dir: str
    software: Optional[str] = None
    absorber: Optional[str] = None
    edge: Optional[str] = None
    state: str = "SUBMITTED"
    record_path: Optional[str] = None
    parse_status: Optional[str] = None
    message: Optional[str] = None


@dataclass
class NERSCWorkflowResult:
    """Structured result returned by the end-to-end NERSC workflow."""

    status: str
    inputs: Dict[str, Any] = field(default_factory=dict)
    submitted_jobs: List[NERSCJob] = field(default_factory=list)
    records: List[Dict[str, Any]] = field(default_factory=list)
    message: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "status": self.status,
            "inputs": self.inputs,
            "submitted_jobs": [job.__dict__ for job in self.submitted_jobs],
            "records": self.records,
            "message": self.message,
        }


class NERSCJobManager:
    """Submit and monitor jobs with NERSC-compatible SLURM commands."""

    def __init__(
        self,
        runner: CommandRunner = subprocess.run,
        submit_command: str = "sbatch",
        queue_command: str = "squeue",
        accounting_command: str = "sacct",
    ):
        self.runner = runner
        self.submit_command = submit_command
        self.queue_command = queue_command
        self.accounting_command = accounting_command

    @staticmethod
    def is_nersc_environment() -> bool:
        """Return True when running on NERSC or where SLURM submit tools exist."""
        return bool(os.environ.get("NERSC_HOST")) or shutil.which("sbatch") is not None

    def submit(self, script: str, dependency_job_ids: Optional[Sequence[str]] = None) -> str:
        """Submit a script with ``sbatch --parsable`` and return the job id."""
        script_path = Path(script).resolve()
        cmd = [self.submit_command, "--parsable"]
        if dependency_job_ids:
            deps = ":".join(str(job_id) for job_id in dependency_job_ids if job_id)
            if deps:
                cmd.append(f"--dependency=afterok:{deps}")
        cmd.append(str(script_path))

        result = self.runner(cmd, cwd=str(script_path.parent), text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"sbatch failed for {script}: {result.stderr.strip() or result.stdout.strip()}")
        output = (result.stdout or "").strip().splitlines()[0].strip()
        # --parsable usually returns just the id, but can include extra fields.
        match = re.search(r"(\d+)", output)
        if not match:
            raise RuntimeError(f"Could not parse sbatch job id from output: {output!r}")
        return match.group(1)

    def state(self, job_id: str) -> str:
        """Return the best available SLURM state for a job id."""
        queued = self.runner(
            [self.queue_command, "-h", "-j", str(job_id), "-o", "%T"],
            text=True,
            capture_output=True,
            check=False,
        )
        if queued.returncode == 0 and queued.stdout.strip():
            return queued.stdout.strip().splitlines()[0].strip().upper()

        accounted = self.runner(
            [self.accounting_command, "-n", "-X", "-j", str(job_id), "-o", "State", "-P"],
            text=True,
            capture_output=True,
            check=False,
        )
        if accounted.returncode == 0 and accounted.stdout.strip():
            # State can be e.g. COMPLETED, FAILED, CANCELLED by 1234.
            return accounted.stdout.strip().splitlines()[0].split("|")[0].split()[0].strip().upper()
        return "UNKNOWN"

    def wait(
        self,
        jobs: Sequence[NERSCJob],
        poll_interval: int = 60,
        timeout: Optional[int] = None,
    ) -> List[NERSCJob]:
        """Poll jobs until all are terminal or timeout is reached."""
        start = time.time()
        pending = {job.job_id: job for job in jobs}
        while pending:
            for job_id, job in list(pending.items()):
                job.state = self.state(job_id)
                if job.state in TERMINAL_STATES:
                    pending.pop(job_id, None)
            if not pending:
                break
            if timeout is not None and time.time() - start >= timeout:
                for job in pending.values():
                    job.message = "monitor_timeout"
                break
            time.sleep(max(int(poll_interval), 1))
        return list(jobs)


def _software_from_path(path: Path) -> Optional[str]:
    for part in reversed(path.parts):
        upper = part.upper()
        if upper in {"FEFF", "FDMNES", "VASP"}:
            return upper
    return None


def _absorber_edge_from_path(path: Path) -> Tuple[Optional[str], Optional[str]]:
    for part in path.parts:
        match = re.match(r"^([A-Z][a-z]?)_(K|L2|L3|L23)_edge$", part, re.IGNORECASE)
        if match:
            return match.group(1).capitalize(), match.group(2).upper()
    return None, None


def discover_submit_scripts(inputs: Dict[str, Any], software_filter: str = "all") -> List[Dict[str, Any]]:
    """Discover generated submit scripts from an XAS input-generation result."""
    wanted = software_filter.upper()
    scripts: List[Dict[str, Any]] = []

    relax_script = inputs.get("relax", {}).get("submit_script")
    if relax_script and wanted in {"ALL", "RELAX", "VASP"}:
        scripts.append({"script": relax_script, "software": "VASP", "stage": "relax"})

    for edge_key, methods in (inputs.get("xas") or {}).items():
        edge_parts = edge_key.split("_")
        absorber = edge_parts[0] if edge_parts else None
        edge = edge_parts[1] if len(edge_parts) > 1 else None
        for software, data in methods.items():
            if wanted != "ALL" and software.upper() != wanted:
                continue
            script = data.get("submit") if isinstance(data, dict) else None
            if script:
                scripts.append({
                    "script": script,
                    "software": software.upper(),
                    "stage": "xas",
                    "absorber": absorber,
                    "edge": edge,
                    "output_dir": str(Path(script).resolve().parent),
                })
    return scripts


def parse_completed_jobs(
    jobs: Iterable[NERSCJob],
    structure_file: str,
    vasp_mode: str = "trace",
    parameters: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Parse outputs for completed XAS jobs and write ISAAC records."""
    records: List[Dict[str, Any]] = []
    for job in jobs:
        if job.state not in SUCCESS_STATES or not job.software:
            continue
        if job.software == "VASP" and Path(job.output_dir).name.lower() == "relax":
            continue
        record_parameters = dict(parameters or {})
        validation_result = run_xas_output_validation(
            software=job.software,
            output_dir=job.output_dir,
            structure_file=structure_file,
            absorber=job.absorber,
            edge=job.edge,
            vasp_mode=vasp_mode,
            parameters=record_parameters,
        )
        if validation_result.get("output_validation"):
            record_parameters.setdefault("output_validation", validation_result["output_validation"])

        if validation_result.get("skip_parse"):
            result = {
                "status": "error",
                "message": "Custodian output validation failed; ISAAC parsing skipped.",
                "output_validation": validation_result.get("output_validation", {}),
            }
            job.parse_status = result["status"]
            job.message = result["message"]
            records.append({
                "job_id": job.job_id,
                "software": job.software,
                "output_dir": job.output_dir,
                "validation_result": validation_result,
                "parse_result": result,
            })
            continue

        result = execute_result_parsing(
            output_dir=job.output_dir,
            software=job.software,
            structure_file=structure_file,
            absorber=job.absorber,
            edge=job.edge,
            parameters=record_parameters,
            vasp_mode=vasp_mode,
        )
        job.parse_status = result.get("status")
        job.record_path = result.get("record_path")
        job.message = result.get("message")
        records.append({
            "job_id": job.job_id,
            "software": job.software,
            "output_dir": job.output_dir,
            "validation_result": validation_result,
            "parse_result": result,
        })
    return records


def execute_nersc_full_workflow(
    structure_file: str,
    output_dir: str,
    software: str = "all",
    submit: bool = True,
    monitor: bool = True,
    parse_when_complete: bool = True,
    poll_interval: int = 60,
    timeout: Optional[int] = None,
    job_manager: Optional[NERSCJobManager] = None,
    **xas_kwargs: Any,
) -> Dict[str, Any]:
    """Generate NERSC inputs, optionally submit/monitor, and write ISAAC records.

    The orchestration is run through custodian. Custodian provides a standard
    job-management layer, applies recoveries through error handlers, and writes
    ``custodian.json`` in output_dir.
    """
    input_keys = {
        "nersc_account",
        "nersc_queue",
        "nersc_nodes",
        "nersc_walltime",
        "email",
        "cluster_radius",
        "fdmnes_options",
        "fdmnes_energy_range",
        "edge_override",
        "vasp_method",
        "potcar_dir",
        "structure_metadata",
    }
    input_kwargs = {k: v for k, v in xas_kwargs.items() if k in input_keys and v is not None}
    record_parameters = dict(xas_kwargs.get("record_parameters", {}) or {})
    if input_kwargs.get("structure_metadata") and "structure_metadata" not in record_parameters:
        record_parameters["structure_metadata"] = input_kwargs["structure_metadata"]

    manager = job_manager or NERSCJobManager()
    context: Dict[str, Any] = {
        "inputs": {},
        "scripts": [],
        "submitted": [],
        "records": [],
        "manager": manager,
        "submit": submit,
        "monitor": monitor,
        "parse_when_complete": parse_when_complete,
        "custodian_max_errors": xas_kwargs.get("custodian_max_errors", 5),
        "warnings": [],
        "recoveries": [],
        "skip_submit_environment_check": job_manager is not None,
    }

    def generate_inputs(ctx: Dict[str, Any]) -> None:
        ctx["inputs"] = execute_xas_input_generation(
            structure_file=structure_file,
            output_dir=output_dir,
            **input_kwargs,
        )
        ctx["scripts"] = discover_submit_scripts(ctx["inputs"], software_filter=software)

    def submit_jobs(ctx: Dict[str, Any]) -> None:
        scripts = ctx.get("scripts", [])
        if (
            ctx.get("submit")
            and not ctx.get("skip_submit_environment_check")
            and not ctx["manager"].is_nersc_environment()
        ):
            ctx["custodian_recovery_required"] = "missing_submit_command"
            return
        if not ctx.get("submit"):
            ctx["submitted"] = [
                NERSCJob(
                    job_id="not_submitted",
                    script=item["script"],
                    output_dir=item.get("output_dir", str(Path(item["script"]).resolve().parent)),
                    software=item.get("software"),
                    absorber=item.get("absorber"),
                    edge=item.get("edge"),
                    state="DRY_RUN",
                )
                for item in scripts
            ]
            return

        submitted: List[NERSCJob] = []
        relax_ids: List[str] = []
        for item in scripts:
            dependency = relax_ids if item.get("stage") == "xas" else None
            job_id = ctx["manager"].submit(item["script"], dependency_job_ids=dependency)
            job = NERSCJob(
                job_id=job_id,
                script=item["script"],
                output_dir=item.get("output_dir", str(Path(item["script"]).resolve().parent)),
                software=item.get("software") or _software_from_path(Path(item["script"])),
                absorber=item.get("absorber") or _absorber_edge_from_path(Path(item["script"]))[0],
                edge=item.get("edge") or _absorber_edge_from_path(Path(item["script"]))[1],
            )
            submitted.append(job)
            if item.get("stage") == "relax":
                relax_ids.append(job_id)
        ctx["submitted"] = submitted

    def monitor_jobs(ctx: Dict[str, Any]) -> None:
        if ctx.get("submit") and ctx.get("monitor"):
            ctx["manager"].wait(ctx.get("submitted", []), poll_interval=poll_interval, timeout=timeout)

    def parse_records(ctx: Dict[str, Any]) -> None:
        if ctx.get("submit") and ctx.get("monitor") and ctx.get("parse_when_complete"):
            ctx["records"] = parse_completed_jobs(
                ctx.get("submitted", []),
                structure_file=structure_file,
                vasp_mode=xas_kwargs.get("vasp_mode", "trace"),
                parameters=record_parameters,
            )

    steps = [
        NERSCWorkflowStep("generate_xas_inputs", generate_inputs, context),
        NERSCWorkflowStep("submit_jobs", submit_jobs, context),
        NERSCWorkflowStep("monitor_jobs", monitor_jobs, context),
        NERSCWorkflowStep("parse_completed_jobs", parse_records, context),
    ]
    custodian_result = run_custodian_steps(steps, output_dir=output_dir, context=context)

    submitted = context.get("submitted", [])
    status = "success"
    if submitted and any(
        job.state not in SUCCESS_STATES and job.state not in {"SUBMITTED", "DRY_RUN"}
        for job in submitted
    ):
        status = "warning"

    if not context.get("submit"):
        message = "Inputs generated; submit=False so no jobs were submitted."
    else:
        message = (
            "NERSC workflow completed."
            if context.get("monitor")
            else "Jobs submitted; monitor=False so records were not parsed."
        )

    result = NERSCWorkflowResult(
        status=status,
        inputs=context.get("inputs", {}),
        submitted_jobs=submitted,
        records=context.get("records", []),
        message=message,
    ).to_dict()
    result["job_manager"] = custodian_result
    return result
