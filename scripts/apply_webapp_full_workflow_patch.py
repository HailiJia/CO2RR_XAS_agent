#!/usr/bin/env python3
"""Patch web_app/main.py to add the full relax -> post-relax XAS workflow.

Run from the repository root:

    python scripts/apply_webapp_full_workflow_patch.py
    python -m py_compile web_app/main.py tools/xas_input_generator.py
    streamlit run web_app/main.py

Then test and commit the modified files.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "web_app" / "main.py"
XAS_GEN = ROOT / "tools" / "xas_input_generator.py"

HELPERS = r'''

# =============================================================================
# Full NERSC relaxation -> post-relaxation XAS workflow
# =============================================================================

def _full_workflow_slurm_text(result):
    if not isinstance(result, dict):
        return str(result)
    return str(result.get("output") or result.get("raw_result") or json.dumps(result, indent=2))


def _full_workflow_is_completed(result):
    text = _full_workflow_slurm_text(result).upper()
    return "COMPLETED" in text and not any(word in text for word in ["FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "OUT_OF_MEMORY"])


def _full_workflow_has_failed(result):
    text = _full_workflow_slurm_text(result).upper()
    return any(word in text for word in ["FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "OUT_OF_MEMORY"])


def _full_workflow_rows(state):
    rows = []
    if state.get("relax_job_id"):
        rows.append({
            "stage": "relax",
            "job_id": state.get("relax_job_id"),
            "state": state.get("relax_state", state.get("stage")),
            "remote_submit": state.get("relax_remote_submit"),
        })
    for item in state.get("xas_jobs", []) or []:
        rows.append({
            "stage": "xas",
            "software": item.get("software"),
            "job_id": item.get("job_id"),
            "state": item.get("state", "SUBMITTED"),
            "remote_submit": item.get("remote_submit"),
        })
    return rows


def _display_full_workflow_state(state):
    if not state:
        st.info("No full relax -> XAS workflow is active in this session.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow stage", state.get("stage", "not_started"))
    c2.metric("Relax job", state.get("relax_job_id") or "-")
    c3.metric("XAS jobs", len(state.get("xas_jobs", []) or []))
    st.write("Remote run directory:", "`" + str(state.get("remote_dir", "")) + "`")
    st.write("Local package root:", "`" + str(state.get("local_root", "")) + "`")
    rows = _full_workflow_rows(state)
    if rows:
        st.table(rows)
    with st.expander("Full workflow state JSON", expanded=False):
        st.json(state)


def _start_full_relax_xas_workflow(params, text="", uploaded_file=None):
    """Generate relaxation inputs, upload package, and submit relaxation first."""
    if not NERSC_JOB_CONTROL_AVAILABLE:
        raise RuntimeError(f"NERSC job-control module is not available: {NERSC_JOB_CONTROL_IMPORT_ERROR}")
    if st.session_state.get("last_structure") is None:
        _generate_structure_from_params(params, uploaded_file=uploaded_file)
    relax_result = _generate_relaxation_from_current_structure(params)
    params["nersc_upload_source"] = "Full package root"
    local_root = str(Path(params.get("xas_output_dir", "generated_outputs/web_xas_agent")).resolve())
    remote_dir = _remote_dir_from_natural_language(text or params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR), params)
    params["nersc_remote_dir"] = remote_dir
    client = _chat_client_from_params(params)
    upload_result = client.upload_directory(local_dir=local_root, remote_dir=remote_dir, overwrite=True, timeout_s=240)
    st.session_state["nersc_last_upload"] = upload_result
    remote_submit = str(PurePosixPath(remote_dir) / "01_structure" / "submit_relax.sh")
    submit_result = client.submit_and_wait_for_jobid(remote_submit, timeout_s=180)
    job_id = str(submit_result.get("job_id") or "")
    params["nersc_task_id"] = str(submit_result.get("task_id") or "")
    if job_id:
        params["nersc_job_id"] = job_id
    params["nersc_last_submit_dir"] = str(PurePosixPath(remote_submit).parent)
    params["nersc_download_remote_dir"] = params["nersc_last_submit_dir"]
    state = {
        "stage": "relax_submitted",
        "local_root": local_root,
        "remote_dir": remote_dir,
        "relax_folder": relax_result.get("relax_folder"),
        "relax_remote_submit": remote_submit,
        "relax_submit_result": submit_result,
        "relax_job_id": job_id,
        "relax_state": "SUBMITTED",
        "upload_result": upload_result,
        "xas_jobs": [],
        "notes": [
            "Refresh monitors relaxation. When relaxation is complete, the app downloads 01_structure/CONTCAR, regenerates 02_XAS from the relaxed structure, uploads 02_XAS, and submits XAS jobs."
        ],
    }
    st.session_state["nersc_full_workflow"] = state
    st.session_state["nersc_last_submit"] = submit_result
    return state


def _regenerate_and_submit_xas_after_relax(params, state, client):
    """Download relaxed CONTCAR, regenerate XAS inputs, upload, and submit XAS jobs."""
    local_root = Path(state["local_root"]).resolve()
    remote_dir = str(state["remote_dir"]).rstrip("/")
    archive_bytes, archive_meta = client.download_remote_archive(
        remote_dir=str(PurePosixPath(remote_dir) / "01_structure"),
        include_glob="CONTCAR OUTCAR OSZICAR vasprun.xml isaac_run_metadata.json isaac_record_draft.json",
        timeout_s=240,
    )
    download_dir = local_root / "_remote_relax_download"
    extract_archive_bytes(archive_bytes, str(download_dir), clean=True)
    contcars = sorted(download_dir.rglob("CONTCAR"))
    if not contcars:
        raise RuntimeError("Relaxation completed, but no CONTCAR was found in the downloaded 01_structure archive.")
    contcar = contcars[0]
    relaxed_structure = read_structure_file(str(contcar))
    relaxed_structure["metadata"] = dict(relaxed_structure.get("metadata", {}))
    relaxed_structure["metadata"].update({
        "source_file": str(contcar),
        "source_stage": "post_relaxation_CONTCAR",
        "workflow": "full_relax_then_xas",
    })
    st.session_state["last_structure"] = relaxed_structure
    st.session_state["last_paths"] = {"poscar": str(contcar)}
    st.session_state["last_substrate_n"] = len(relaxed_structure.get("atoms", []))

    # Archive any pre-relax XAS folder before regenerating from relaxed structure.
    xas_dir = Path(params.get("xas_output_dir", "generated_outputs/web_xas_agent")) / "02_XAS"
    if xas_dir.exists():
        archive_dir = xas_dir.with_name("02_XAS_pre_relax_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(xas_dir), str(archive_dir))

    xas_result, xas_dir, edge = _generate_xas_from_current_structure(params)
    xas_remote_dir = str(PurePosixPath(remote_dir) / "02_XAS")
    upload_result = client.upload_directory(local_dir=str(xas_dir), remote_dir=xas_remote_dir, overwrite=True, timeout_s=240)

    jobs = []
    for software, rel_submit in [("VASP", "VASP/submit.sh"), ("FDMNES", "FDMNES/submit.sh"), ("FEFF", "FEFF/submit.sh")]:
        local_submit = Path(xas_dir) / rel_submit
        if not local_submit.exists():
            continue
        remote_submit = str(PurePosixPath(remote_dir) / "02_XAS" / rel_submit)
        submit_result = client.submit_and_wait_for_jobid(remote_submit, timeout_s=180)
        jobs.append({
            "software": software,
            "remote_submit": remote_submit,
            "task_id": submit_result.get("task_id"),
            "job_id": submit_result.get("job_id"),
            "state": "SUBMITTED",
            "submit_result": submit_result,
        })

    state.update({
        "stage": "xas_submitted",
        "relaxed_contcar_local": str(contcar),
        "relax_download_meta": archive_meta,
        "xas_local_dir": str(xas_dir),
        "xas_remote_dir": xas_remote_dir,
        "xas_edge": edge,
        "xas_result": xas_result,
        "xas_upload_result": upload_result,
        "xas_jobs": jobs,
    })
    st.session_state["nersc_full_workflow"] = state
    return state


def _refresh_full_relax_xas_workflow(params):
    """Refresh relaxation/XAS status and advance the full workflow when ready."""
    state = st.session_state.get("nersc_full_workflow")
    if not state:
        raise ValueError("No full workflow is active. Start the full relax -> XAS workflow first.")
    client = _chat_client_from_params(params)
    stage = state.get("stage", "")

    if stage in {"relax_submitted", "relax_running", "relax_completed"}:
        job_id = str(state.get("relax_job_id") or "").strip()
        if not job_id:
            raise ValueError("No relaxation Slurm job ID is stored in the workflow state.")
        status = client.slurm_status(job_id, timeout_s=90)
        state["relax_status"] = status
        state["relax_status_text"] = _full_workflow_slurm_text(status)
        if _full_workflow_has_failed(status):
            state["stage"] = "relax_failed"
            state["relax_state"] = "FAILED"
        elif _full_workflow_is_completed(status):
            state["stage"] = "relax_completed"
            state["relax_state"] = "COMPLETED"
            state = _regenerate_and_submit_xas_after_relax(params, state, client)
        else:
            state["stage"] = "relax_running"
            state["relax_state"] = "RUNNING_OR_PENDING"
        st.session_state["nersc_full_workflow"] = state
        return state

    if stage in {"xas_submitted", "xas_running"}:
        all_done = True
        any_failed = False
        for job in state.get("xas_jobs", []) or []:
            job_id = str(job.get("job_id") or "").strip()
            if not job_id:
                all_done = False
                continue
            status = client.slurm_status(job_id, timeout_s=90)
            job["status"] = status
            job["status_text"] = _full_workflow_slurm_text(status)
            if _full_workflow_has_failed(status):
                job["state"] = "FAILED"
                any_failed = True
            elif _full_workflow_is_completed(status):
                job["state"] = "COMPLETED"
            else:
                job["state"] = "RUNNING_OR_PENDING"
                all_done = False
        if any_failed:
            state["stage"] = "xas_failed_or_partial"
        elif all_done and state.get("xas_jobs"):
            state["stage"] = "completed"
        else:
            state["stage"] = "xas_running"
        st.session_state["nersc_full_workflow"] = state
        return state

    st.session_state["nersc_full_workflow"] = state
    return state
'''

UI_BLOCK = r'''
        st.subheader("Full relax -> XAS workflow")
        st.caption("Submit relaxation first. When refresh detects completion, the app downloads `01_structure/CONTCAR`, regenerates local `02_XAS` from the relaxed structure, uploads the regenerated XAS folder, and submits VASP/FDMNES/FEFF XAS jobs.")
        fw1, fw2 = st.columns(2)
        with fw1:
            if st.button("Start full relax -> XAS workflow", type="primary", key="start_full_relax_xas_workflow"):
                try:
                    st.session_state["nersc_full_workflow"] = _start_full_relax_xas_workflow(p, text=p.get("nersc_remote_dir", ""), uploaded_file=uploaded_structure_file)
                    st.success("Full workflow started. Use Refresh full workflow status after the relaxation job finishes.")
                except Exception as exc:
                    st.exception(exc)
        with fw2:
            if st.button("Refresh full workflow status", key="refresh_full_relax_xas_workflow"):
                try:
                    st.session_state["nersc_full_workflow"] = _refresh_full_relax_xas_workflow(p)
                    st.success("Full workflow status refreshed.")
                except Exception as exc:
                    st.exception(exc)
        _display_full_workflow_state(st.session_state.get("nersc_full_workflow"))

'''


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f"Could not find marker for {label}")
    return text.replace(old, new, 1)


def patch_main() -> None:
    text = MAIN.read_text()

    # Resolve stale comments after rename.
    text = text.replace("# web_app/app.py", "# web_app/main.py", 1)
    text = text.replace("streamlit run web_app/app.py", "streamlit run web_app/main.py")

    # Add helper functions after the tested chat/NERSC runtime patch helpers.
    helper_marker = "\ndef _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):\n"
    if "Full NERSC relaxation -> post-relaxation XAS workflow" not in text:
        text = replace_once(text, helper_marker, HELPERS + helper_marker, "full workflow helpers")

    # Update LLM planner valid intents / examples.
    text = text.replace(
        "show_paths, help, cancel_job, answer_question, update_settings, unknown. ",
        "show_paths, help, cancel_job, start_full_workflow, refresh_full_workflow, answer_question, update_settings, unknown. ",
        1,
    )
    text = text.replace(
        "For 'generate relaxation inputs and upload to NERSC under X', use generate_relaxation_and_upload and remote_run_name X. ",
        "For 'generate relaxation inputs and upload to NERSC under X', use generate_relaxation_and_upload and remote_run_name X. For 'start full relax XAS workflow under X', use start_full_workflow and remote_run_name X. For 'refresh/check full workflow', use refresh_full_workflow. ",
        1,
    )

    # Add chat help examples.
    text = text.replace(
        "- `submit selected job`\n",
        "- `submit selected job`\n"
        "            \"- `start full relax XAS workflow under test04`\\n\"\n"
        "            \"- `refresh full workflow`\\n\"\n",
        1,
    )

    # Add execution intents before ISAAC conversion.
    executor_marker = "    if intent == \"convert_isaac\":\n"
    executor_block = """    if intent == \"start_full_workflow\":
        remote_text = plan.get(\"remote_run_name\") or plan.get(\"remote_dir\") or text
        state = _start_full_relax_xas_workflow(params, text=remote_text, uploaded_file=uploaded_file)
        return (
            f\"Full relax -> XAS workflow started. Relaxation job `{state.get('relax_job_id') or 'not available yet'}` \"
            f\"was submitted from `{state.get('relax_remote_submit')}`. Remote folder: `{state.get('remote_dir')}`. \"
            \"Use `refresh full workflow` to monitor relaxation and submit regenerated XAS jobs after CONTCAR is available.\"
        )

    if intent == \"refresh_full_workflow\":
        state = _refresh_full_relax_xas_workflow(params)
        return \"Full workflow refreshed. Current stage: `{}`. Jobs: `{}`.\".format(state.get(\"stage\"), len(_full_workflow_rows(state)))

"""
    if "if intent == \"start_full_workflow\":" not in text:
        text = replace_once(text, executor_marker, executor_block + executor_marker, "full workflow executor")

    # Add rule-based classifier hook before CO2RR pathway rules.
    classifier_marker = "    if any(p in low for p in [\"full co2rr\", \"full reaction pathway\", \"reaction pathway\", \"full pathway\", \"pathway on\"]):\n"
    classifier_block = """    if any(p in low for p in [\"full workflow\", \"relax xas workflow\", \"relax then xas\", \"relaxation then xas\", \"post-relax xas\", \"post relax xas\", \"full relax xas\"]):
        return \"refresh_full_workflow\" if any(p in low for p in [\"refresh\", \"check\", \"status\", \"monitor\"]) else \"start_full_workflow\"
"""
    if "return \"refresh_full_workflow\" if any(p in low" not in text:
        text = replace_once(text, classifier_marker, classifier_block + classifier_marker, "full workflow classifier")

    # LLM safety overrides for explicit full-workflow commands.
    override_marker = "        if _chat_requested_job_listing(low_text):\n"
    override_block = """        if any(p in low_text for p in [\"full workflow\", \"relax xas workflow\", \"relax then xas\", \"relaxation then xas\", \"post-relax xas\", \"post relax xas\", \"full relax xas\"]):
            intent = \"refresh_full_workflow\" if any(p in low_text for p in [\"refresh\", \"check\", \"status\", \"monitor\"]) else \"start_full_workflow\"
        elif _chat_requested_job_listing(low_text):
"""
    if "intent = \"refresh_full_workflow\" if any(p in low_text" not in text:
        text = replace_once(text, override_marker, override_block, "full workflow planner override")

    # Add UI under NERSC agent before upload package.
    ui_marker = "        st.subheader(\"Upload input package\")\n"
    if "Start full relax -> XAS workflow" not in text:
        text = replace_once(text, ui_marker, UI_BLOCK + ui_marker, "full workflow UI")

    # Make generated shell scripts use python3 for heredoc blocks.
    text = text.replace("python - <<'PY'", "python3 - <<'PY'")

    MAIN.write_text(text)


def patch_xas_generator() -> None:
    if not XAS_GEN.exists():
        return
    text = XAS_GEN.read_text()
    text = text.replace('"python - <<\'PY\'"', '"python3 - <<\'PY\'"')
    text = text.replace("python - <<'PY'", "python3 - <<'PY'")
    XAS_GEN.write_text(text)


if __name__ == "__main__":
    patch_main()
    patch_xas_generator()
    print("Patched web_app/main.py and tools/xas_input_generator.py")
