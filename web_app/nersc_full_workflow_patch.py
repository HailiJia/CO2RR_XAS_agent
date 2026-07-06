# web_app/nersc_full_workflow_patch.py
# Runtime source patch for the Structure and XAS Streamlit app.

from __future__ import annotations

HELPERS = r'''
# =============================================================================
# Web NERSC chat/runtime patches
# =============================================================================

def _fw_clean_run_name(name):
    raw_name = str(name or "").strip().strip(".,;:)>]")
    raw_name = re.sub(r"\s+", "_", raw_name)
    raw_name = re.sub(r"^(test)_+(\d+)$", r"\1\2", raw_name, flags=re.I)
    raw_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", raw_name).strip("_")
    return raw_name


def _remote_dir_from_natural_language(text, params):
    raw = str(text or "").strip()
    stop_names = {"", "none", "latest", "nersc", "perlmutter", "file", "files", "folder", "folders", "dir", "directory", "there", "that", "this", "current", "remote", "job", "jobs", "surface", "clean", "relaxation"}
    abs_match = re.search(r"(/pscratch/[^\s`'\"]+|/global/[^\s`'\"]+)", raw)
    if abs_match:
        return abs_match.group(1).rstrip(".,;:)>]")
    for pattern in [
        r"(?:under|run(?:\s+name)?|named)\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)",
        r"(?:remote\s+)?(?:folder|dir|directory)\s+(?:named\s+)?([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)",
    ]:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            name = _fw_clean_run_name(match.group(1))
            if name.lower() not in stop_names and "/" not in name:
                return f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/{name}"
    if re.search(r"\b(upload|copy|send|move)\b", raw, flags=re.I):
        for to_match in re.finditer(r"\bto\s+([A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?)", raw, flags=re.I):
            name = _fw_clean_run_name(to_match.group(1))
            if name.lower() not in stop_names and "/" not in name:
                return f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/{name}"
    if re.fullmatch(r"[A-Za-z0-9_.-]+(?:\s+[A-Za-z0-9_.-]+)?", raw):
        name = _fw_clean_run_name(raw)
        if name.lower() not in stop_names:
            return f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/{name}"
    return params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)


def _fw_text(result):
    if not isinstance(result, dict):
        return str(result)
    return str(result.get("output") or result.get("raw_result") or json.dumps(result, indent=2))


def _fw_cancel_job(params, text):
    client = _chat_client_from_params(params)
    job_id = _extract_slurm_job_id_from_text(text) or str(params.get("nersc_job_id", "")).strip()
    if not job_id:
        raise ValueError("No Slurm job ID was found. Say: cancel slurm job 55602116")
    params["nersc_job_id"] = str(job_id)
    cmd = "s" + "cancel " + client.shell_quote(job_id)
    task = client.run_command(cmd)
    task_id = str(task.get("task_id") or task.get("id") or "")
    result = client.wait_task(task_id, timeout_s=90) if task_id else task
    st.session_state["nersc_last_cancel"] = {"job_id": job_id, "task_id": task_id, "task": result}
    try:
        status = client.slurm_status(job_id, timeout_s=90)
        st.session_state["nersc_last_slurm"] = status
        status_text = _fw_text(status)
    except Exception as exc:
        status_text = f"Cancel request sent, but follow-up status check failed: {exc}"
    return f"Cancel request sent for Slurm job `{job_id}`. SF API task ID: `{task_id}`.\n\n{status_text[:3000]}"


def _fw_submit_with_optional_settings(params, text):
    client = _chat_client_from_params(params)
    workflow_updates = _parse_workflow_setting_changes(text)
    upload_note = ""
    if workflow_updates:
        params.update(workflow_updates)
        if st.session_state.get("last_structure") is not None:
            _generate_relaxation_from_current_structure(params)
        selected_local_dir = _selected_local_folder_from_params(params)
        remote_dir = (st.session_state.get("nersc_last_upload") or {}).get("remote_dir") or params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)
        params["nersc_remote_dir"] = remote_dir
        upload_result = client.upload_directory(local_dir=selected_local_dir, remote_dir=remote_dir, overwrite=True, timeout_s=240)
        st.session_state["nersc_last_upload"] = upload_result
        upload_note = " Updated settings: " + ", ".join(f"{k}={v}" for k, v in workflow_updates.items()) + f". Re-uploaded `{selected_local_dir}` to `{remote_dir}` before submission."
    natural_submit = _remote_submit_from_natural_language(text, params)
    remote_submit = natural_submit or _resolve_remote_submit_from_params(params)
    if natural_submit:
        params["nersc_remote_submit_script"] = remote_submit
    result = client.submit_and_wait_for_jobid(remote_submit, timeout_s=180)
    st.session_state["nersc_last_submit"] = result
    if result.get("task_id"):
        params["nersc_task_id"] = str(result["task_id"])
    if result.get("job_id"):
        params["nersc_job_id"] = str(result["job_id"])
    params["nersc_last_submit_dir"] = str(PurePosixPath(remote_submit).parent)
    params["nersc_download_remote_dir"] = params["nersc_last_submit_dir"]
    job_id_display = params.get("nersc_job_id") or "not available yet; run check task in a few seconds"
    return f"Submit request completed for `{remote_submit}`. SF API task ID: `{params.get('nersc_task_id', '')}`. Slurm job ID: `{job_id_display}`.{upload_note}"
'''


def patch_full_workflow_source(source: str) -> str:
    if "Web NERSC chat/runtime patches" not in source:
        marker = "\n" + "def _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):"
        if marker in source:
            source = source.replace(marker, HELPERS + marker, 1)

    source = source.replace("answer_question, update_settings, unknown. ", "cancel_job, answer_question, update_settings, unknown. ", 1)
    source = source.replace("For 'remove/delete remote folder /pscratch/...' use delete_remote_folder. ", "For 'remove/delete remote folder /pscratch/...' use delete_remote_folder. For 'cancel/scancel Slurm job 12345', use cancel_job. ", 1)

    classifier_marker = "    if _chat_requested_job_listing(low):\n"
    cancel_rule = "    if re.search(r'\\b(cancel|scancel)\\b', low) and re.search(r'\\b(job|slurm|squeue|sacct)\\b|[0-9]{5,12}', low):\n        return 'cancel_job'\n"
    if classifier_marker in source and "return 'cancel_job'" not in source:
        source = source.replace(classifier_marker, cancel_rule + classifier_marker, 1)

    submit_marker = "    if intent == \"submit\":\n"
    submit_insert = "    if intent == 'submit':\n        return _fw_submit_with_optional_settings(params, text)\n\n"
    if submit_marker in source and "_fw_submit_with_optional_settings(params, text)" not in source:
        source = source.replace(submit_marker, submit_insert + submit_marker.replace("if intent", "if False and intent"), 1)

    executor_marker = "    if intent == \"convert_isaac\":\n"
    executor_insert = "    if intent == 'cancel_job':\n        return _fw_cancel_job(params, text)\n\n"
    if executor_marker in source and "if intent == 'cancel_job':" not in source:
        source = source.replace(executor_marker, executor_insert + executor_marker, 1)
    return source
