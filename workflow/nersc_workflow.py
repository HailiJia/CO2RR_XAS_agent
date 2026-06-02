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

from tools.result_parser import execute_result_parsing
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
        result = execute_result_parsing(
            output_dir=job.output_dir,
            software=job.software,
            structure_file=structure_file,
            absorber=job.absorber,
            edge=job.edge,
            parameters=parameters or {},
            vasp_mode=vasp_mode,
        )
        job.parse_status = result.get("status")
        job.record_path = result.get("record_path")
        job.message = result.get("message")
        records.append({
            "job_id": job.job_id,
            "software": job.software,
            "output_dir": job.output_dir,
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

    This is intended for NERSC login nodes.  Set ``submit=False`` to only create
    inputs and submit scripts, which is useful on non-NERSC machines or tests.
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
    inputs = execute_xas_input_generation(
        structure_file=structure_file,
        output_dir=output_dir,
        **input_kwargs,
    )
    scripts = discover_submit_scripts(inputs, software_filter=software)
    manager = job_manager or NERSCJobManager()

    if not submit:
        jobs = [
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
        return NERSCWorkflowResult(
            status="success",
            inputs=inputs,
            submitted_jobs=jobs,
            message="Inputs generated; submit=False so no jobs were submitted.",
        ).to_dict()

    submitted: List[NERSCJob] = []
    relax_ids: List[str] = []
    for item in scripts:
        dependency = relax_ids if item.get("stage") == "xas" else None
        job_id = manager.submit(item["script"], dependency_job_ids=dependency)
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

    if monitor:
        manager.wait(submitted, poll_interval=poll_interval, timeout=timeout)

    records: List[Dict[str, Any]] = []
    if monitor and parse_when_complete:
        records = parse_completed_jobs(
            submitted,
            structure_file=structure_file,
            vasp_mode=xas_kwargs.get("vasp_mode", "trace"),
            parameters=xas_kwargs.get("record_parameters", {}),
        )

    status = "success"
    if submitted and any(job.state not in SUCCESS_STATES and job.state != "SUBMITTED" for job in submitted):
        status = "warning"
    return NERSCWorkflowResult(
        status=status,
        inputs=inputs,
        submitted_jobs=submitted,
        records=records,
        message="NERSC workflow completed." if monitor else "Jobs submitted; monitor=False so records were not parsed.",
    ).to_dict()
