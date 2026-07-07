#!/usr/bin/env python3
"""Patch web_app/main.py to use the durable NERSC workflow orchestrator.

This patch wires the existing Streamlit app to the NERSC-side workflow layer in
this branch.  It intentionally makes the corrected full-workflow action do all
local preparation in one step:

    generate current structure -> generate 01_structure/submit_relax.sh ->
    generate workflow_*.sh -> upload package -> remote bash workflow_submit.sh

Run from the repository root:

    python scripts/apply_webapp_nersc_workflow_orchestrator_patch.py
    python -m py_compile web_app/main.py tools/web_nersc_workflow_integration.py
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "web_app" / "main.py"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"Could not find patch anchor: {label}")
    return text.replace(old, new, 1)


def insert_once(text: str, anchor: str, addition: str, label: str, *, after: bool = True) -> str:
    if addition.strip() in text:
        return text
    if anchor not in text:
        raise RuntimeError(f"Could not find patch anchor: {label}")
    return text.replace(anchor, anchor + addition if after else addition + anchor, 1)


def main() -> None:
    text = MAIN.read_text()

    import_block = """from tools.xas_input_generator import (
    RelaxationInputGenerator,
    FEFFInputGenerator,
    FDMNESInputGenerator,
    VASPXASInputGenerator,
    NERSCScriptGenerator,
)
"""
    import_add = """from tools.web_nersc_workflow_integration import (
    prepare_single_structure_workflow_package,
    run_remote_workflow_script,
)
"""
    text = insert_once(text, import_block, import_add, "web workflow integration imports")

    old_default = '        "nersc_remote_submit_script": "01_structure/submit_relax.sh",\n'
    new_default = (
        '        "nersc_remote_submit_script": "workflow_submit.sh",\n'
        '        "nersc_repo_root": os.environ.get("CO2RR_AGENT_REPO", str(REPO_ROOT)),\n'
    )
    if old_default in text and '"nersc_repo_root"' not in text:
        text = text.replace(old_default, new_default, 1)

    helper_anchor = """def _generate_xas_from_current_structure(params):
"""
    helper_code = r'''
def _resolve_current_workflow_absorber_edge(params):
    structure = st.session_state.get("last_structure")
    atoms = list(dict.fromkeys((structure or {}).get("atoms", [])))
    absorber_options = [el for el in atoms if el not in {"H"}] or atoms
    absorber = params.get("xas_absorber")
    if absorber not in absorber_options:
        absorber = "Cu" if "Cu" in absorber_options else (absorber_options[0] if absorber_options else "Cu")
        params["xas_absorber"] = absorber
    edge = normalize_edge_for_absorber(absorber, params.get("xas_edge", "K"))
    return absorber, edge


def _prepare_full_workflow_from_current_structure(params):
    """Prepare package root with 01_structure + workflow_*.sh in one user action."""
    if st.session_state.get("last_structure") is None:
        _generate_structure_from_params(params, uploaded_file=None)
    structure = st.session_state["last_structure"]
    absorber, edge = _resolve_current_workflow_absorber_edge(params)
    result = prepare_single_structure_workflow_package(
        structure=structure,
        package_root=params["xas_output_dir"],
        repo_root=params.get("nersc_repo_root") or os.environ.get("CO2RR_AGENT_REPO", str(REPO_ROOT)),
        absorber=absorber,
        edge=edge,
        relaxation_settings=relaxation_settings_from_params(params),
        account=params.get("nersc_account", DEFAULT_NERSC_ACCOUNT),
        queue=params.get("nersc_queue", "regular"),
        nodes=int(params.get("nersc_nodes", 1)),
        walltime=params.get("nersc_walltime", "02:00:00"),
        email=params.get("nersc_email", ""),
        vasp_method=params.get("vasp_method", "PBE"),
        cluster_radius=float(params.get("cluster_radius", 6.0)),
        fdmnes_method=params.get("fdmnes_method", "Green"),
        fdmnes_scf=bool(params.get("fdmnes_scf", False)),
        potcar_dir=params.get("potcar_dir", DEFAULT_NERSC_POTCAR_DIR),
    )
    st.session_state["last_full_workflow_result"] = result
    st.session_state["last_relax_result"] = result.get("relaxation")
    st.session_state["last_generated_package_kind"] = "single_full_workflow"
    params["nersc_upload_source"] = "Full package root"
    params["nersc_remote_submit_script"] = "workflow_submit.sh"
    params["nersc_download_glob"] = "workflow_state.json workflow_manifest.json *.out OUTCAR OSZICAR CONTCAR vasprun.xml xmu.dat *_conv.txt CORE_DIELECTRIC_IMAG.dat isaac_run_metadata.json isaac_record_draft.json"
    return result, absorber, edge


def _upload_full_workflow_package(params, text, plan=None):
    remote_dir = _remote_dir_from_natural_language((plan or {}).get("remote_run_name") or text, params)
    params["nersc_remote_dir"] = remote_dir
    result, absorber, edge = _prepare_full_workflow_from_current_structure(params)
    client = _chat_client_from_params(params)
    upload_result = client.upload_directory(
        local_dir=result["package_root"],
        remote_dir=remote_dir,
        overwrite=bool(params.get("nersc_overwrite_remote", False)),
        timeout_s=240,
    )
    st.session_state["nersc_last_upload"] = upload_result
    params["nersc_download_remote_dir"] = remote_dir
    params["nersc_preview_dir"] = remote_dir
    ok, label = classify_upload_result(upload_result)
    return result, upload_result, label, remote_dir, absorber, edge


def _run_remote_full_workflow_action(params, text, script_name, state_key, timeout_s=180, plan=None):
    client = _chat_client_from_params(params)
    remote_dir = _remote_dir_explicitly_mentioned(text, params)
    if not remote_dir:
        remote_dir = (st.session_state.get("nersc_last_upload") or {}).get("remote_dir") or params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)
    remote_dir = str(remote_dir).rstrip("/")
    result = run_remote_workflow_script(client, remote_dir, script_name, timeout_s=timeout_s)
    st.session_state[state_key] = result
    params["nersc_download_remote_dir"] = remote_dir
    params["nersc_preview_dir"] = remote_dir
    output = ""
    if isinstance(result.get("result"), dict):
        output = str(result["result"].get("output") or result["result"].get("raw_result") or "")
    return remote_dir, result, output

'''
    text = insert_once(text, helper_anchor, helper_code, "full workflow helper block", after=False)

    execute_anchor = """    if intent == "generate_structure_and_relaxation":
"""
    execute_code = r'''    if intent == "prepare_full_workflow":
        if st.session_state.get("last_structure") is None:
            parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
            params.update(parsed)
            _generate_structure_from_params(params, uploaded_file=uploaded_file)
        result, absorber, edge = _prepare_full_workflow_from_current_structure(params)
        return (
            f"Full NERSC workflow package prepared at `{result.get('package_root')}`. "
            f"Relaxation inputs were generated internally at `{result.get('relaxation', {}).get('relax_folder')}`. "
            f"Workflow entry point: `workflow_submit.sh`; absorber `{absorber}`, edge `{edge}`. "
            "Next: `upload full workflow to NERSC under testXX`, or `start full workflow` after upload."
        )

    if intent == "prepare_upload_full_workflow":
        if st.session_state.get("last_structure") is None:
            parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
            params.update(parsed)
            _generate_structure_from_params(params, uploaded_file=uploaded_file)
        result, upload_result, label, remote_dir, absorber, edge = _upload_full_workflow_package(params, text, plan=plan)
        return (
            f"Full NERSC workflow package prepared and uploaded. {label}.\n"
            f"Local package: `{result.get('package_root')}`\n"
            f"Remote package: `{remote_dir}`\n"
            f"Absorber/edge: `{absorber}` `{edge}`.\n"
            "Next: `start full workflow`, which remotely runs `bash workflow_submit.sh`."
        )

    if intent == "prepare_upload_start_full_workflow":
        if st.session_state.get("last_structure") is None:
            parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
            params.update(parsed)
            _generate_structure_from_params(params, uploaded_file=uploaded_file)
        result, upload_result, label, remote_dir, absorber, edge = _upload_full_workflow_package(params, text, plan=plan)
        start_result = run_remote_workflow_script(_chat_client_from_params(params), remote_dir, "workflow_submit.sh", timeout_s=180)
        st.session_state["nersc_last_workflow_start"] = start_result
        output = ""
        if isinstance(start_result.get("result"), dict):
            output = str(start_result["result"].get("output") or start_result["result"].get("raw_result") or "")
        return (
            f"Full NERSC workflow package prepared, uploaded, and started. {label}.\n"
            f"Remote package: `{remote_dir}`\n"
            f"Entry point: `bash workflow_submit.sh`\n"
            f"SF API task ID: `{start_result.get('task_id', '')}`\n\n"
            f"{output[:4000]}"
        )

    if intent == "start_full_workflow":
        remote_dir, result, output = _run_remote_full_workflow_action(params, text, "workflow_submit.sh", "nersc_last_workflow_start", timeout_s=180, plan=plan)
        return f"Started full workflow in `{remote_dir}` by running `bash workflow_submit.sh`. SF API task ID: `{result.get('task_id', '')}`.\n\n{output[:4000]}"

    if intent == "refresh_full_workflow_status":
        remote_dir, result, output = _run_remote_full_workflow_action(params, text, "workflow_status.sh", "nersc_last_workflow_status", timeout_s=120, plan=plan)
        return f"Workflow status refreshed in `{remote_dir}`.\n\n{output[:5000]}"

    if intent == "restart_full_workflow":
        remote_dir, result, output = _run_remote_full_workflow_action(params, text, "workflow_restart.sh", "nersc_last_workflow_restart", timeout_s=180, plan=plan)
        return f"Workflow restart/continue requested in `{remote_dir}`. SF API task ID: `{result.get('task_id', '')}`.\n\n{output[:4000]}"

    if intent == "cancel_full_workflow":
        remote_dir, result, output = _run_remote_full_workflow_action(params, text, "workflow_cancel.sh", "nersc_last_workflow_cancel", timeout_s=120, plan=plan)
        return f"Workflow cancel requested in `{remote_dir}`. SF API task ID: `{result.get('task_id', '')}`.\n\n{output[:4000]}"

'''
    text = insert_once(text, execute_anchor, execute_code, "full workflow execute intents", after=False)

    classifier_anchor = """    if any(p in low for p in ["what is the full path", "uploaded folder", "upload path", "submitted folder", "remote path", "where did", "where is"]):
        return "show_paths"
"""
    classifier_code = r'''    full_workflow_requested = any(p in low for p in ["full workflow", "workflow package", "relax-to-xas", "relax to xas", "relaxation to xas"])
    if full_workflow_requested:
        if re.search(r"\b(cancel|stop)\b", low):
            return "cancel_full_workflow"
        if re.search(r"\b(restart|continue|resume)\b", low):
            return "restart_full_workflow"
        if re.search(r"\b(status|refresh|check)\b", low):
            return "refresh_full_workflow_status"
        if re.search(r"\b(start|run|submit)\b", low) and re.search(r"\b(upload|prepare|generate|make|build)\b", low):
            return "prepare_upload_start_full_workflow"
        if re.search(r"\b(upload|send|copy)\b", low):
            return "prepare_upload_full_workflow"
        if re.search(r"\b(start|run|submit)\b", low):
            return "start_full_workflow"
        if re.search(r"\b(prepare|generate|make|build)\b", low):
            return "prepare_full_workflow"

'''
    text = insert_once(text, classifier_anchor, classifier_code, "full workflow classifier", after=True)

    help_old = """- `generate relaxation inputs and upload to NERSC under test03`
- `submit selected job`
"""
    help_new = """- `prepare full workflow`
- `upload full workflow to NERSC under test03`
- `prepare, upload, and start full workflow under test03`
- `refresh full workflow status`, `restart full workflow`, `cancel full workflow`
- `generate relaxation inputs and upload to NERSC under test03`
- `submit selected job`
"""
    if help_old in text:
        text = text.replace(help_old, help_new, 1)

    sidebar_old = """"message": "I can answer questions, update settings, and run actions. Try `Generate full CO2RR pathway on Cu`, `generate structure and relaxation files`, `upload to NERSC under test03`, `what files are in that folder`, `change to 8 hrs`, or `submit clean surface relaxation job`."""
    sidebar_new = """"message": "I can answer questions, update settings, and run actions. Try `prepare, upload, and start full workflow under test03`, `refresh full workflow status`, `Generate full CO2RR pathway on Cu`, `what files are in that folder`, or `change to 8 hrs`."""
    if sidebar_old in text:
        text = text.replace(sidebar_old, sidebar_new, 1)

    MAIN.write_text(text)
    print(f"Patched {MAIN}")


if __name__ == "__main__":
    main()
