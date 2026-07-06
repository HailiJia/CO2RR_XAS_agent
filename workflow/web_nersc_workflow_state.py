"""Nonblocking NERSC workflow state for the Streamlit Structure + XAS page.

The full blocking workflow in :mod:`workflow.nersc_workflow` is useful from a
terminal or tmux session.  A web app needs a refreshable state machine instead:
each page refresh checks SLURM, advances exactly the steps that are now ready,
and writes a JSON state file under the calculation output directory.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tools.xas_input_generator import execute_xas_input_generation
from workflow.nersc_workflow import (
    NERSCJob,
    NERSCJobManager,
    SUCCESS_STATES,
    TERMINAL_STATES,
    discover_submit_scripts,
    parse_completed_jobs,
)
from workflow.relaxed_xas_rewrite import (
    patch_submit_scripts,
    regenerate_xas_from_relaxed,
    require_relaxed_contcar,
)

STATE_FILENAME = ".nersc_xas_workflow_state.json"
STATE_VERSION = "v1_post_relax_xas_web_workflow"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def state_path_for(output_dir: str) -> Path:
    return Path(output_dir).resolve() / STATE_FILENAME


def job_to_dict(job: NERSCJob) -> Dict[str, Any]:
    return dict(job.__dict__)


def dict_to_job(data: Dict[str, Any]) -> NERSCJob:
    return NERSCJob(
        job_id=str(data.get("job_id", "not_submitted")),
        script=str(data.get("script", "")),
        output_dir=str(data.get("output_dir", "")),
        software=data.get("software"),
        absorber=data.get("absorber"),
        edge=data.get("edge"),
        stage=data.get("stage", "xas"),
        state=data.get("state", "UNKNOWN"),
        record_path=data.get("record_path"),
        parse_status=data.get("parse_status"),
        message=data.get("message"),
    )


def save_state(state: Dict[str, Any]) -> Path:
    path = state_path_for(state["output_dir"])
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_utc"] = utc_now()
    path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    return path


def load_state(output_dir: str) -> Dict[str, Any]:
    path = state_path_for(output_dir)
    if not path.is_file():
        raise FileNotFoundError("No workflow state found at {}".format(path))
    return json.loads(path.read_text(encoding="utf-8"))


def _submit_jobs(manager: NERSCJobManager, scripts: List[Dict[str, Any]], stage: str) -> List[NERSCJob]:
    jobs: List[NERSCJob] = []
    for item in scripts:
        script = item["script"]
        job_id = manager.submit(script)
        jobs.append(NERSCJob(
            job_id=job_id,
            script=script,
            output_dir=item.get("output_dir", str(Path(script).resolve().parent)),
            software=item.get("software"),
            absorber=item.get("absorber"),
            edge=item.get("edge"),
            stage=stage,
            state="SUBMITTED",
        ))
    return jobs


def _dry_jobs(scripts: List[Dict[str, Any]], stage: str) -> List[NERSCJob]:
    jobs: List[NERSCJob] = []
    for item in scripts:
        script = item["script"]
        jobs.append(NERSCJob(
            job_id="not_submitted",
            script=script,
            output_dir=item.get("output_dir", str(Path(script).resolve().parent)),
            software=item.get("software"),
            absorber=item.get("absorber"),
            edge=item.get("edge"),
            stage=stage,
            state="DRY_RUN",
        ))
    return jobs


def _update_job_states(manager: NERSCJobManager, job_dicts: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    updated: List[Dict[str, Any]] = []
    for item in job_dicts:
        job = dict_to_job(item)
        if job.job_id not in {"not_submitted", "already_completed"} and job.state not in TERMINAL_STATES:
            job.state = manager.state(job.job_id)
        updated.append(job_to_dict(job))
    return updated


def _all_success(job_dicts: List[Dict[str, Any]]) -> bool:
    return bool(job_dicts) and all(item.get("state") in SUCCESS_STATES for item in job_dicts)


def _any_failed_terminal(job_dicts: List[Dict[str, Any]]) -> bool:
    return any(item.get("state") in TERMINAL_STATES and item.get("state") not in SUCCESS_STATES for item in job_dicts)


def start_web_nersc_workflow(
    *,
    structure_file: str,
    output_dir: str,
    software: str = "all",
    submit: bool = True,
    parse_when_complete: bool = True,
    nersc_account: Optional[str] = None,
    nersc_queue: Optional[str] = None,
    nersc_nodes: Optional[int] = None,
    nersc_walltime: Optional[str] = None,
    email: Optional[str] = None,
    cluster_radius: float = 6.0,
    edge_override: Optional[str] = None,
    vasp_method: str = "PBE",
    potcar_dir: Optional[str] = None,
    manager: Optional[NERSCJobManager] = None,
) -> Dict[str, Any]:
    """Start the web-app workflow without blocking on job completion."""
    out = Path(output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    manager = manager or NERSCJobManager()
    input_kwargs = {
        "nersc_account": nersc_account,
        "nersc_queue": nersc_queue,
        "nersc_nodes": nersc_nodes,
        "nersc_walltime": nersc_walltime,
        "email": email,
        "cluster_radius": cluster_radius,
        "edge_override": edge_override,
        "vasp_method": vasp_method,
        "potcar_dir": potcar_dir,
    }
    input_kwargs = {k: v for k, v in input_kwargs.items() if v is not None}

    inputs = execute_xas_input_generation(structure_file=structure_file, output_dir=str(out), **input_kwargs)
    patched = patch_submit_scripts(str(out))
    scripts = discover_submit_scripts(inputs, software_filter="all")
    relax_scripts = [item for item in scripts if item.get("stage") == "relax"]

    warnings: List[str] = []
    relaxed_structure_file = None
    try:
        relaxed_structure_file = str(require_relaxed_contcar(str(out)))
        relax_jobs = [job_to_dict(NERSCJob(
            job_id="already_completed",
            script=item["script"],
            output_dir=item.get("output_dir", str(Path(item["script"]).resolve().parent)),
            software=item.get("software"),
            stage="relax",
            state="COMPLETED",
        )) for item in relax_scripts]
        status = "relax_completed"
        current_step = "relaxation already completed; ready to regenerate XAS inputs"
    except FileNotFoundError:
        if submit:
            if not manager.is_nersc_environment():
                raise RuntimeError("sbatch/SLURM not available. Run this page on NERSC login node or use submit=False.")
            relax_jobs = [job_to_dict(job) for job in _submit_jobs(manager, relax_scripts, "relax")]
            status = "relax_submitted"
            current_step = "relaxation submitted"
        else:
            relax_jobs = [job_to_dict(job) for job in _dry_jobs(relax_scripts, "relax")]
            status = "dry_run_relax_inputs"
            current_step = "relaxation inputs generated; not submitted"
            warnings.append("submit=False: XAS inputs will not be regenerated/submitted until relax/CONTCAR exists.")

    state = {
        "version": STATE_VERSION,
        "started_utc": utc_now(),
        "updated_utc": utc_now(),
        "status": status,
        "current_step": current_step,
        "structure_file": str(Path(structure_file).resolve()),
        "output_dir": str(out),
        "software": software,
        "submit": submit,
        "parse_when_complete": parse_when_complete,
        "workflow_parameters": input_kwargs,
        "patched_submit_scripts": patched,
        "relaxed_structure_file": relaxed_structure_file,
        "xas_regenerated": False,
        "relax_jobs": relax_jobs,
        "xas_jobs": [],
        "records": [],
        "warnings": warnings,
    }
    save_state(state)
    return refresh_web_nersc_workflow(str(out), manager=manager)


def refresh_web_nersc_workflow(output_dir: str, manager: Optional[NERSCJobManager] = None) -> Dict[str, Any]:
    """Refresh SLURM status and advance the workflow if the next step is ready."""
    manager = manager or NERSCJobManager()
    state = load_state(output_dir)
    state["relax_jobs"] = _update_job_states(manager, state.get("relax_jobs", []))
    state["xas_jobs"] = _update_job_states(manager, state.get("xas_jobs", []))

    if _any_failed_terminal(state.get("relax_jobs", [])):
        state["status"] = "relax_failed"
        state["current_step"] = "relaxation failed; XAS inputs were not regenerated"
        save_state(state)
        return state

    if not state.get("xas_regenerated") and _all_success(state.get("relax_jobs", [])):
        params = dict(state.get("workflow_parameters", {}))
        result = regenerate_xas_from_relaxed(output_dir=state["output_dir"], **params)
        state["relaxed_structure_file"] = result.get("xas_structure_source")
        state["xas_regenerated"] = True
        state["patched_submit_scripts"] = result.get("patched_submit_scripts", state.get("patched_submit_scripts", 0))
        xas_scripts = [
            item for item in discover_submit_scripts(result, software_filter=state.get("software", "all"))
            if item.get("stage") == "xas"
        ]
        if state.get("submit"):
            state["xas_jobs"] = [job_to_dict(job) for job in _submit_jobs(manager, xas_scripts, "xas")]
            state["status"] = "xas_submitted"
            state["current_step"] = "XAS jobs submitted from relaxed CONTCAR"
        else:
            state["xas_jobs"] = [job_to_dict(job) for job in _dry_jobs(xas_scripts, "xas")]
            state["status"] = "dry_run_xas_inputs"
            state["current_step"] = "XAS inputs regenerated from relaxed CONTCAR; not submitted"
        save_state(state)
        return state

    if _any_failed_terminal(state.get("xas_jobs", [])):
        state["status"] = "xas_failed_or_partial"
        state["current_step"] = "one or more XAS jobs failed"
        save_state(state)
        return state

    if _all_success(state.get("xas_jobs", [])):
        if state.get("parse_when_complete") and not state.get("records"):
            jobs = [dict_to_job(item) for item in state.get("xas_jobs", [])]
            state["records"] = parse_completed_jobs(
                jobs,
                structure_file=state.get("relaxed_structure_file") or state.get("structure_file"),
            )
        state["status"] = "completed"
        state["current_step"] = "XAS jobs completed; records parsed" if state.get("records") else "XAS jobs completed"
    elif state.get("xas_jobs"):
        state["status"] = "xas_running"
        state["current_step"] = "waiting for XAS jobs"
    elif state.get("relax_jobs"):
        state["status"] = "relax_running"
        state["current_step"] = "waiting for relaxation to finish"

    save_state(state)
    return state


def workflow_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for job in state.get("relax_jobs", []):
        rows.append({
            "stage": "relax",
            "job_id": job.get("job_id"),
            "state": job.get("state"),
            "software": job.get("software"),
            "absorber": job.get("absorber"),
            "edge": job.get("edge"),
            "script": job.get("script"),
            "output_dir": job.get("output_dir"),
        })
    for job in state.get("xas_jobs", []):
        rows.append({
            "stage": "xas",
            "job_id": job.get("job_id"),
            "state": job.get("state"),
            "software": job.get("software"),
            "absorber": job.get("absorber"),
            "edge": job.get("edge"),
            "script": job.get("script"),
            "output_dir": job.get("output_dir"),
        })
    return rows
