# web_app/nersc_full_workflow_patch.py
"""Runtime source patch for the Structure and XAS Streamlit app.

This keeps the existing app.py implementation intact while adding a small full
relaxation -> post-relaxation XAS workflow under the existing NERSC agent panel.
"""

from __future__ import annotations


HELPERS = r'''
# =============================================================================
# Web NERSC full relaxation -> post-relaxation XAS workflow helpers
# =============================================================================

def _fw_text(result):
    if not isinstance(result, dict):
        return str(result)
    return str(result.get("output") or result.get("raw_result") or json.dumps(result, indent=2))


def _fw_completed(result):
    text = _fw_text(result).upper()
    return "COMPLETED" in text and not any(x in text for x in ["FAILED", "CANCELLED", "CANCELED", "TIMEOUT"])


def _fw_failed(result):
    return any(x in _fw_text(result).upper() for x in ["FAILED", "CANCELLED", "CANCELED", "TIMEOUT", "OUT_OF_MEMORY"])


def _fw_rows(state):
    rows = []
    if state.get("relax_job_id"):
        rows.append({"stage": "relax", "job_id": state.get("relax_job_id"), "state": state.get("relax_state", state.get("stage")), "remote_submit": state.get("relax_remote_submit")})
    for item in state.get("xas_jobs", []) or []:
        rows.append({"stage": "xas", "software": item.get("software"), "job_id": item.get("job_id"), "state": item.get("state", "SUBMITTED"), "remote_submit": item.get("remote_submit")})
    return rows


def _fw_display(state):
    if not state:
        st.info("No full workflow is active in this session.")
        return
    c1, c2, c3 = st.columns(3)
    c1.metric("Workflow stage", state.get("stage", "not_started"))
    c2.metric("Relax job", state.get("relax_job_id") or "-")
    c3.metric("XAS jobs", len(state.get("xas_jobs", []) or []))
    st.write("Remote directory:", "`" + str(state.get("remote_dir", "")) + "`")
    st.write("Local package:", "`" + str(state.get("local_root", "")) + "`")
    rows = _fw_rows(state)
    if rows:
        st.table(rows)
    with st.expander("Full workflow state JSON", expanded=False):
        st.json(state)


def _fw_start(params, text="", uploaded_file=None):
    if not NERSC_JOB_CONTROL_AVAILABLE:
        raise RuntimeError(f"NERSC job-control module is not available: {NERSC_JOB_CONTROL_IMPORT_ERROR}")
    if st.session_state.get("last_structure") is None:
        _generate_structure_from_params(params, uploaded_file=uploaded_file)
    relax_result = _generate_relaxation_from_current_structure(params)
    local_root = str(Path(params.get("xas_output_dir", "generated_outputs/web_xas_agent")).resolve())
    remote_dir = _remote_dir_from_natural_language(text or params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR), params)
    params["nersc_remote_dir"] = remote_dir
    params["nersc_upload_source"] = "Full package root"
    client = _chat_client_from_params(params)
    upload_result = client.upload_directory(local_dir=local_root, remote_dir=remote_dir, overwrite=True, timeout_s=240)
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
        "notes": ["After relaxation completes, refresh downloads 01_structure/CONTCAR, regenerates local 02_XAS, uploads it, and submits XAS jobs."],
    }
    st.session_state["nersc_full_workflow"] = state
    st.session_state["nersc_last_upload"] = upload_result
    st.session_state["nersc_last_submit"] = submit_result
    return state


def _fw_regenerate_and_submit_xas(params, state, client):
    local_root = Path(state["local_root"]).resolve()
    remote_dir = str(state["remote_dir"]).rstrip("/")
    archive, download_meta = client.download_remote_archive(remote_dir=str(PurePosixPath(remote_dir) / "01_structure"), include_glob="CONTCAR", timeout_s=240)
    download_dir = local_root / "_remote_relax_CONTCAR"
    extract_archive_bytes(archive, str(download_dir), clean=True)
    contcars = sorted(download_dir.rglob("CONTCAR"))
    if not contcars:
        raise RuntimeError("Relaxation completed, but CONTCAR was not found in the downloaded 01_structure archive.")
    contcar = contcars[0]
    structure = read_structure_file(str(contcar))
    structure["metadata"] = dict(structure.get("metadata", {}))
    structure["metadata"].update({"source_file": str(contcar), "source_stage": "post_relaxation_CONTCAR"})
    atoms = list(dict.fromkeys(structure.get("atoms", [])))
    absorber = params.get("xas_absorber")
    if absorber not in atoms:
        absorber = choose_batch_absorber(structure, preferred=None)
        params["xas_absorber"] = absorber
    edge = normalize_edge_for_absorber(absorber, params.get("xas_edge", "K"))
    xas_dir = local_root / "02_XAS"
    if xas_dir.exists():
        archived = local_root / ("02_XAS_pre_relax_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        shutil.move(str(xas_dir), str(archived))
    xas_result = generate_xas_calculation_folders(
        structure=structure,
        output_dir=xas_dir,
        absorber=absorber,
        edge=edge,
        vasp_method=params.get("vasp_method", "PBE"),
        cluster_radius=float(params.get("cluster_radius", 6.0)),
        fdmnes_method=params.get("fdmnes_method", "Green"),
        fdmnes_scf=bool(params.get("fdmnes_scf", False)),
        fdmnes_quadrupole=bool(params.get("fdmnes_quadrupole", False)),
        fdmnes_spinorbit=bool(params.get("fdmnes_spinorbit", False)),
        fdmnes_energy_min=float(params.get("fdmnes_energy_min", -5.0)),
        fdmnes_energy_step=float(params.get("fdmnes_energy_step", 0.2)),
        fdmnes_energy_max=float(params.get("fdmnes_energy_max", 50.0)),
        potcar_dir=params.get("potcar_dir", DEFAULT_NERSC_POTCAR_DIR),
        account=params.get("nersc_account", DEFAULT_NERSC_ACCOUNT),
        queue=params.get("nersc_queue", "regular"),
        nodes=int(params.get("nersc_nodes", 1)),
        walltime=params.get("nersc_walltime", "02:00:00"),
        email=params.get("nersc_email", ""),
    )
    upload_result = client.upload_directory(local_dir=str(xas_dir), remote_dir=str(PurePosixPath(remote_dir) / "02_XAS"), overwrite=True, timeout_s=240)
    jobs = []
    for software, rel_submit in [("VASP", "VASP/submit.sh"), ("FDMNES", "FDMNES/submit.sh"), ("FEFF", "FEFF/submit.sh")]:
        if not (xas_dir / rel_submit).exists():
            continue
        remote_submit = str(PurePosixPath(remote_dir) / "02_XAS" / rel_submit)
        submit_result = client.submit_and_wait_for_jobid(remote_submit, timeout_s=180)
        jobs.append({"software": software, "remote_submit": remote_submit, "task_id": submit_result.get("task_id"), "job_id": submit_result.get("job_id"), "state": "SUBMITTED", "submit_result": submit_result})
    state.update({"stage": "xas_submitted", "relaxed_contcar_local": str(contcar), "xas_local_dir": str(xas_dir), "xas_result": xas_result, "xas_upload_result": upload_result, "xas_jobs": jobs})
    st.session_state["last_xas_result"] = xas_result
    st.session_state["last_xas_dir"] = str(xas_dir)
    return state


def _fw_refresh(params):
    state = st.session_state.get("nersc_full_workflow")
    if not state:
        raise ValueError("No full workflow is active. Click Start full relax -> XAS workflow first.")
    client = _chat_client_from_params(params)
    stage = state.get("stage", "")
    if stage in {"relax_submitted", "relax_running", "relax_completed"}:
        job_id = str(state.get("relax_job_id") or "").strip()
        if not job_id:
            raise ValueError("No relaxation Slurm job ID is stored in the workflow state.")
        status = client.slurm_status(job_id, timeout_s=90)
        state["relax_status"] = status
        state["relax_status_text"] = _fw_text(status)
        if _fw_failed(status):
            state["stage"] = "relax_failed"
            state["relax_state"] = "FAILED"
        elif _fw_completed(status):
            state["stage"] = "relax_completed"
            state["relax_state"] = "COMPLETED"
            state = _fw_regenerate_and_submit_xas(params, state, client)
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
            job["status_text"] = _fw_text(status)
            if _fw_failed(status):
                job["state"] = "FAILED"
                any_failed = True
            elif _fw_completed(status):
                job["state"] = "COMPLETED"
            else:
                job["state"] = "RUNNING_OR_PENDING"
                all_done = False
        state["stage"] = "xas_failed_or_partial" if any_failed else ("completed" if all_done and state.get("xas_jobs") else "xas_running")
        st.session_state["nersc_full_workflow"] = state
        return state
    st.session_state["nersc_full_workflow"] = state
    return state
'''


def patch_full_workflow_source(source: str) -> str:
    """Add full relax -> XAS workflow controls and chat intents to app.py source."""
    if "Web NERSC full relaxation -> post-relaxation XAS workflow helpers" not in source:
        marker = '\ndef _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):'
        if marker in source:
            source = source.replace(marker, HELPERS + marker, 1)

    source = source.replace(
        'answer_question, update_settings, unknown. ',
        'start_full_workflow, refresh_full_workflow, answer_question, update_settings, unknown. ',
        1,
    )

    classifier_marker = '    if any(p in low for p in ["full co2rr", "full reaction pathway", "reaction pathway", "full pathway", "pathway on"]):\n'
    classifier_insert = '    if any(p in low for p in ["full workflow", "relax xas workflow", "relax then xas", "relaxation then xas", "post-relax xas", "post relax xas"]):\n        return "refresh_full_workflow" if any(p in low for p in ["refresh", "check", "status", "monitor"]) else "start_full_workflow"\n'
    if classifier_marker in source and 'return "refresh_full_workflow" if any(p in low' not in source:
        source = source.replace(classifier_marker, classifier_insert + classifier_marker, 1)

    executor_marker = '    if intent == "convert_isaac":\n'
    executor_insert = '''    if intent == "start_full_workflow":
        state = _fw_start(params, text=text, uploaded_file=uploaded_file)
        return (
            f"Full relax -> XAS workflow started. Relaxation job `{state.get('relax_job_id') or 'not available yet'}` "
            f"was submitted from `{state.get('relax_remote_submit')}`. Remote folder: `{state.get('remote_dir')}`. "
            "Use `refresh full workflow` to monitor relaxation and submit regenerated XAS jobs after CONTCAR is available."
        )

    if intent == "refresh_full_workflow":
        state = _fw_refresh(params)
        return "Full workflow refreshed. Current stage: `{}`. Jobs: `{}`.".format(state.get("stage"), len(_fw_rows(state)))

'''
    if executor_marker in source and 'if intent == "start_full_workflow":' not in source:
        source = source.replace(executor_marker, executor_insert + executor_marker, 1)

    ui_marker = '        st.subheader("Upload input package")\n'
    ui_block = '''        st.subheader("Full relax -> XAS workflow")
        st.caption("Start relaxation first. When refresh detects completed relaxation, the app downloads `01_structure/CONTCAR`, regenerates local `02_XAS` from that relaxed structure, uploads the new XAS folder, and submits VASP/FDMNES/FEFF XAS jobs.")
        fw1, fw2 = st.columns(2)
        with fw1:
            if st.button("Start full relax -> XAS workflow", type="primary", key="start_full_relax_xas_workflow"):
                try:
                    st.session_state["nersc_full_workflow"] = _fw_start(p, text=p.get("nersc_remote_dir", ""), uploaded_file=uploaded_structure_file)
                    st.success("Full workflow started. Use Refresh full workflow status to advance after relaxation finishes.")
                except Exception as exc:
                    st.exception(exc)
        with fw2:
            if st.button("Refresh full workflow status", key="refresh_full_relax_xas_workflow"):
                try:
                    st.session_state["nersc_full_workflow"] = _fw_refresh(p)
                    st.success("Full workflow status refreshed.")
                except Exception as exc:
                    st.exception(exc)
        _fw_display(st.session_state.get("nersc_full_workflow"))

'''
    if ui_marker in source and 'Start full relax -> XAS workflow' not in source:
        source = source.replace(ui_marker, ui_block + ui_marker, 1)
    return source
