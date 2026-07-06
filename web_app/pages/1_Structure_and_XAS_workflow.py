# web_app/pages/1_Structure_and_XAS_workflow.py
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional in lightweight deployments
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from workflow.web_nersc_workflow_state import (  # noqa: E402
    load_state,
    refresh_web_nersc_workflow,
    start_web_nersc_workflow,
    state_path_for,
    workflow_rows,
)

PAGE_TAG = "v1_2026-07-06_post_relax_nersc_web_workflow"

st.set_page_config(page_title="CO2RR XAS Agent | Structure + XAS workflow", layout="wide")
st.title("Structure + XAS workflow")
st.caption(f"Update tag: `{PAGE_TAG}`")

st.markdown(
    """
Use this page for the connected production workflow:

1. Generate and submit VASP relaxation inputs.
2. Refresh the status until relaxation finishes.
3. Regenerate `xas/*` inputs from `relax/CONTCAR`.
4. Submit FEFF/FDMNES/VASP XAS jobs.
5. Refresh live SLURM status and parse ISAAC records when complete.

The relaxation `*.sh` itself does **not** submit XAS. The web workflow advances the next step only after `relax/CONTCAR` exists.
""".strip()
)


def display_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        st.info("No jobs found yet.")
        return
    if pd is not None:
        st.dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        st.table(rows)


def step_index(status: str) -> int:
    order = {
        "dry_run_relax_inputs": 1,
        "relax_submitted": 1,
        "relax_running": 1,
        "relax_completed": 2,
        "dry_run_xas_inputs": 3,
        "xas_submitted": 4,
        "xas_running": 4,
        "completed": 5,
        "relax_failed": 0,
        "xas_failed_or_partial": 4,
    }
    return order.get(status, 0)


with st.sidebar:
    st.header("Workflow inputs")
    structure_file = st.text_input("Initial structure file", value="POSCAR")
    output_dir = st.text_input("Output directory", value="generated_outputs/Cu_111_CO")
    software = st.selectbox("XAS software to submit after relaxation", ["all", "FEFF", "FDMNES", "VASP"], index=0)

    st.subheader("NERSC / SLURM")
    nersc_account = st.text_input("Account", value="m5268")
    nersc_queue = st.text_input("Queue", value="regular")
    nersc_nodes = st.number_input("Nodes", min_value=1, max_value=64, value=1, step=1)
    nersc_walltime = st.text_input("Wall time", value="04:00:00")
    submit_jobs = st.checkbox("Submit jobs with sbatch", value=True)
    parse_when_complete = st.checkbox("Parse ISAAC records when XAS completes", value=True)

    st.subheader("XAS generation")
    vasp_method = st.selectbox("VASP XAS method", ["PBE", "GW"], index=0)
    cluster_radius = st.number_input("FEFF/FDMNES cluster radius (Å)", min_value=2.0, max_value=12.0, value=6.0, step=0.5)
    edge_override = st.text_input("Optional edge override", value="")
    potcar_dir = st.text_input("Optional POTCAR root", value="")

    st.subheader("Refresh")
    auto_refresh = st.checkbox("Auto-refresh status", value=False)
    refresh_seconds = st.number_input("Refresh interval (s)", min_value=10, max_value=600, value=30, step=10)

state_file = state_path_for(output_dir)

col_start, col_refresh, col_state = st.columns([1, 1, 2])

with col_start:
    start_clicked = st.button("Start / submit workflow", type="primary")
with col_refresh:
    refresh_clicked = st.button("Refresh status")
with col_state:
    st.write("State file")
    st.code(str(state_file), language="text")

state: Dict[str, Any] | None = None

if start_clicked:
    try:
        if not Path(structure_file).expanduser().exists():
            st.error(f"Structure file not found: {structure_file}")
        else:
            state = start_web_nersc_workflow(
                structure_file=str(Path(structure_file).expanduser()),
                output_dir=str(Path(output_dir).expanduser()),
                software=software,
                submit=submit_jobs,
                parse_when_complete=parse_when_complete,
                nersc_account=nersc_account or None,
                nersc_queue=nersc_queue or None,
                nersc_nodes=int(nersc_nodes),
                nersc_walltime=nersc_walltime or None,
                cluster_radius=float(cluster_radius),
                edge_override=edge_override.strip() or None,
                vasp_method=vasp_method,
                potcar_dir=potcar_dir.strip() or None,
            )
            st.success("Workflow state started and saved.")
    except Exception as exc:
        st.exception(exc)

if refresh_clicked and not start_clicked:
    try:
        state = refresh_web_nersc_workflow(str(Path(output_dir).expanduser()))
        st.success("Status refreshed.")
    except Exception as exc:
        st.exception(exc)

if state is None and state_file.is_file():
    try:
        state = load_state(str(Path(output_dir).expanduser()))
    except Exception as exc:
        st.warning(f"Could not load existing state: {exc}")

if state is not None:
    st.divider()
    status = state.get("status", "unknown")
    current_step = state.get("current_step", "not started")
    st.subheader("Live workflow status")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", status)
    c2.metric("Current step", current_step)
    c3.metric("Relax jobs", len(state.get("relax_jobs", [])))
    c4.metric("XAS jobs", len(state.get("xas_jobs", [])))

    st.progress(step_index(status) / 5.0, text=current_step)

    if state.get("warnings"):
        with st.expander("Warnings", expanded=True):
            for warning in state.get("warnings", []):
                st.warning(warning)

    st.markdown("#### Job table")
    display_table(workflow_rows(state))

    st.markdown("#### Key paths")
    key_paths = {
        "initial_structure": state.get("structure_file"),
        "output_dir": state.get("output_dir"),
        "relaxed_structure_file": state.get("relaxed_structure_file"),
        "state_file": str(state_file),
    }
    st.json(key_paths)

    if state.get("records"):
        st.markdown("#### Parsed ISAAC records")
        display_table([
            {
                "job_id": item.get("job_id"),
                "software": item.get("software"),
                "output_dir": item.get("output_dir"),
                "parse_status": (item.get("parse_result") or {}).get("status"),
                "record_path": (item.get("parse_result") or {}).get("record_path"),
                "message": (item.get("parse_result") or {}).get("message"),
            }
            for item in state.get("records", [])
        ])

    with st.expander("Raw workflow state JSON"):
        st.code(json.dumps(state, indent=2), language="json")
else:
    st.info("No workflow state loaded yet. Start a workflow or set the output directory to an existing workflow folder.")

if auto_refresh and state is not None and state.get("status") not in {"completed", "relax_failed", "xas_failed_or_partial"}:
    time.sleep(int(refresh_seconds))
    st.rerun()
