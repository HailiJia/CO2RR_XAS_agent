# web_app/Structure_and_XAS_calculation.py
'''Polished Streamlit entrypoint for the Structure and XAS workflow.

Run with:
    streamlit run web_app/Structure_and_XAS_calculation.py

This keeps ``web_app/app.py`` as the legacy implementation, but renders it with a
cleaner information architecture.
'''

from __future__ import annotations

import runpy
from pathlib import Path


_THIS_FILE = Path(__file__).resolve()
_APP_PATH = _THIS_FILE.with_name("app.py")
_RUNTIME_PATH = _THIS_FILE.with_name("_runtime_structure_xas_app.py")


def _indent_block(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "".join(prefix + line if line.strip() else line for line in text.splitlines(keepends=True))


def _dedent_block(text: str, spaces: int = 4) -> str:
    prefix = " " * spaces
    return "".join(line[spaces:] if line.startswith(prefix) else line for line in text.splitlines(keepends=True))


def _polish_labels(source: str) -> str:
    replacements = {
        'st.header("System settings")': 'st.header("Structure setup")',
        '"Structure source"': '"Input structure"',
        '"System type"': '"Surface model"',
        '"Metal"': '"Surface metal"',
        '"Nominal coverage"': '"Adsorbate coverage"',
        'st.number_input(\n                    "nx",': 'st.number_input(\n                    "Supercell nx",',
        'st.number_input(\n                    "ny",': 'st.number_input(\n                    "Supercell ny",',
        '"requested nx"': '"Interface repeat target"',
        '"rows per side"': '"Rows away from interface"',
        '"Vacuum, Å"': '"Vacuum thickness, Å"',
        '"Adsorbate height, Å"': '"Initial adsorbate height, Å"',
        '"Default site: **{DEFAULT_ADSORBATE_SITES.get(p[\'adsorbate\'], \'top\')}**"': '"Initial adsorption site: **{DEFAULT_ADSORBATE_SITES.get(p[\'adsorbate\'], \'top\')}**"',
        '"Adsorbate binding metal"': '"Adsorbate binding site metal"',
        '"Structure output directory"': '"Local output folder"',
        '"Chat backend"': '"Agent planner"',
        '"ALCF cluster"': '"ALCF inference cluster"',
        '"Show planner trace"': '"Show agent debug trace"',
        '"Agent responses"': '"Conversation"',
        'st.header("Structure generation")': 'st.subheader("Structure generation")',
        'st.header("Batch processing")': 'st.subheader("Batch generation")',
        '"Generate inputs for multiple structures or a reaction pathway"': '"Generate input packages for multiple structures or reaction pathways"',
        '"Generate pathway from sidebar system"': '"Generate pathway from current structure setup"',
        '"Generate VASP relaxation folders"': '"Include VASP relaxation inputs"',
        '"Generate XAS folders"': '"Include XAS inputs"',
        '"Batch absorber"': '"Absorber selection"',
        '"use selected absorber when present"': '"Use selected absorber if available"',
        'with st.expander("Sidebar request", expanded=False):': 'with st.expander("Current workflow parameters", expanded=False):',
        'st.header("NERSC job control")': 'st.subheader("NERSC agent")',
        'I can update sidebar settings and run workflow actions. Examples:': 'I can update workspace settings and run workflow actions. Examples:',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    title_line = 'st.title("CO2RR XAS Agent")'
    subtitle = (
        'st.title("CO2RR XAS Agent")\n'
        'st.caption("Use the sidebar as the agent console. Use the main workspace for catalyst structure setup, '
        'structure/XAS input generation, batch generation, NERSC jobs, ISAAC records, and ISAAC Portal upload.")'
    )
    source = source.replace(title_line, subtitle, 1)
    return source


def _patch_job_listing_logic(source: str) -> str:
    old = '''    if intent == "list_jobs":
        client = _chat_client_from_params(params)
        username = str(params.get("nersc_username") or os.environ.get("USER") or "hjia").strip() or "hjia"
        result = client.list_user_jobs(username=username, timeout_s=90)
        st.session_state["nersc_user_jobs"] = result
        output = str(result.get("output", json.dumps(result, indent=2)))
        return f"Current/recent NERSC jobs for `{username}`:\\n\\n" + output[:5000]
'''
    new = '''    if intent == "list_jobs":
        client = _chat_client_from_params(params)
        username = str(params.get("nersc_username") or os.environ.get("USER") or "hjia").strip() or "hjia"
        result = client.list_user_jobs(username=username, timeout_s=90)
        st.session_state["nersc_user_jobs"] = result
        output = str(result.get("output", json.dumps(result, indent=2)))
        current_only = (
            any(
                phrase in low
                for phrase in [
                    "only current",
                    "current job",
                    "current jobs",
                    "running job",
                    "running jobs",
                    "jobs running",
                    "queued job",
                    "queued jobs",
                    "on queue",
                    "in queue",
                    "squeue",
                ]
            )
            or ("queue" in low and "job" in low)
        )
        if current_only:
            marker = "== Current queued/running jobs =="
            recent_marker = "== Recent accounting records =="
            current_text = output
            if marker in current_text:
                current_text = current_text.split(marker, 1)[1]
                if recent_marker in current_text:
                    current_text = current_text.split(recent_marker, 1)[0]
                current_text = marker + "\\n" + current_text.strip()
            lines = [line.strip() for line in current_text.splitlines() if line.strip()]
            data_lines = [
                line for line in lines
                if not line.startswith("==") and not line.startswith("JOBID|") and not line.startswith("JOBID ")
            ]
            if data_lines:
                current_display = "\\n".join(lines)[:4000]
                return f"Current queued/running NERSC jobs for `{username}`:\\n\\n```text\\n{current_display}\\n```"
            hint = ""
            if params.get("nersc_task_id") and not params.get("nersc_job_id"):
                hint = f"\\n\\nLast submit SF API task ID is `{params.get('nersc_task_id')}`. If this was just submitted, run `check task` to see whether Slurm returned a job ID."
            return f"No current queued or running NERSC jobs were found for `{username}`." + hint
        return f"Current/recent NERSC jobs for `{username}`:\\n\\n" + output[:5000]
'''
    if old not in source:
        return source
    return source.replace(old, new, 1)


def _patch_isaac_portal_upload(source: str) -> str:
    block = r'''
# =============================================================================
# ISAAC Portal validation/upload
# =============================================================================

st.divider()
st.subheader("ISAAC Portal")
st.caption("Uses `ISAAC_URL` and `ISAAC_KEY` from the terminal environment. The API key is not stored in Streamlit state.")

try:
    from tools.isaac_portal_client import ISAACPortalClient
    _isaac_portal_import_error = None
except Exception as _exc:
    ISAACPortalClient = None
    _isaac_portal_import_error = _exc

_isaac_env_url = os.environ.get("ISAAC_URL", "")
_isaac_env_key = os.environ.get("ISAAC_KEY", "")
_col_url, _col_key = st.columns(2)
_col_url.caption(f"ISAAC_URL: `{_isaac_env_url or 'not set; using default portal API URL'}`")
_col_key.caption("ISAAC_KEY: " + ("found" if _isaac_env_key else "missing"))

_portal_mode = st.radio(
    "ISAAC Portal upload source",
    ["Generated/current ISAAC record", "Drop ISAAC JSON file(s)"],
    horizontal=True,
    key="isaac_portal_upload_mode",
)

_records_to_upload = []
if _portal_mode == "Generated/current ISAAC record":
    _candidate_record = st.session_state.get("last_isaac_record") or st.session_state.get("isaac_last_record")
    _candidate_path = st.session_state.get("last_isaac_record_path") or st.session_state.get("isaac_last_record_path")
    if not _candidate_record and _candidate_path and Path(str(_candidate_path)).exists():
        try:
            _candidate_record = json.loads(Path(str(_candidate_path)).read_text())
        except Exception as _exc:
            st.warning(f"Could not read generated ISAAC record path `{_candidate_path}`: {_exc}")
    if not _candidate_record:
        _fallback_path = Path(str(st.session_state.params.get("isaac_local_output_dir", "generated_outputs/web_xas_agent/downloaded_outputs"))) / "isaac_record_draft.json"
        if _fallback_path.exists():
            try:
                _candidate_record = json.loads(_fallback_path.read_text())
                _candidate_path = str(_fallback_path)
            except Exception as _exc:
                st.warning(f"Could not read `{_fallback_path}`: {_exc}")
    if _candidate_record:
        _records_to_upload.append((str(_candidate_path or _candidate_record.get("record_id") or "generated_record"), _candidate_record))
        st.success(f"Selected generated ISAAC record `{_candidate_record.get('record_id', 'no record_id')}`.")
        with st.expander("Generated ISAAC record preview", expanded=False):
            st.json(_candidate_record)
    else:
        st.info("No generated ISAAC record is available yet. Convert outputs to an ISAAC record first, or use the drop-file option.")
else:
    _uploaded_records = st.file_uploader(
        "Drop ISAAC record JSON file(s)",
        type=["json"],
        accept_multiple_files=True,
        key="isaac_portal_json_drop",
    )
    for _uploaded in _uploaded_records or []:
        try:
            _record = json.loads(_uploaded.getvalue().decode("utf-8"))
            if isinstance(_record, dict):
                _records_to_upload.append((_uploaded.name, _record))
        except Exception as _exc:
            st.error(f"Could not parse `{getattr(_uploaded, 'name', 'uploaded')}`: {_exc}")

if _isaac_portal_import_error is not None:
    st.error(f"ISAAC Portal client could not be imported: {_isaac_portal_import_error}")
elif not _isaac_env_key:
    st.warning("Set `ISAAC_KEY` before launching Streamlit to validate or upload records.")
elif _records_to_upload:
    _upload_cols = st.columns(2)
    if _upload_cols[0].button("Validate selected ISAAC record(s)", key="isaac_portal_validate"):
        try:
            _client = ISAACPortalClient()
            _results = []
            for _name, _record in _records_to_upload:
                _results.append({"source": _name, "record_id": _record.get("record_id"), "validation": _client.validate_record(_record)})
            st.session_state["isaac_portal_validation_results"] = _results
        except Exception as _exc:
            st.error(f"ISAAC Portal validation failed: {_exc}")
    if _upload_cols[1].button("Validate + upload selected ISAAC record(s)", type="primary", key="isaac_portal_upload"):
        try:
            _client = ISAACPortalClient()
            _results = []
            for _name, _record in _records_to_upload:
                _results.append({"source": _name, "record_id": _record.get("record_id"), "result": _client.validate_then_create(_record)})
            st.session_state["isaac_portal_upload_results"] = _results
        except Exception as _exc:
            st.error(f"ISAAC Portal upload failed: {_exc}")

if st.session_state.get("isaac_portal_validation_results"):
    with st.expander("ISAAC Portal validation results", expanded=True):
        st.json(st.session_state["isaac_portal_validation_results"])
if st.session_state.get("isaac_portal_upload_results"):
    with st.expander("ISAAC Portal upload results", expanded=True):
        st.json(st.session_state["isaac_portal_upload_results"])
'''
    if "ISAAC Portal validation/upload" in source:
        return source
    return source.rstrip() + "\n\n" + block + "\n"


def _extract_structure_setup(source: str) -> tuple[str, str]:
    assignment_start = source.find("uploaded_structure_file = None")
    sidebar_start = source.find("with st.sidebar:", assignment_start)
    chat_start = source.find("# Chat box: lightweight workflow console", sidebar_start)
    if assignment_start == -1 or sidebar_start == -1 or chat_start == -1:
        return source, ""

    assignment_block = source[assignment_start:sidebar_start]
    sidebar_block = source[sidebar_start:chat_start]
    sidebar_body = sidebar_block[sidebar_block.find("\n") + 1 :]
    structure_setup = _dedent_block(sidebar_body, 4).strip("\n") + "\n"
    structure_setup = structure_setup.replace('st.header("System settings")\n', "")

    source_without_sidebar = source[:assignment_start] + assignment_block + source[chat_start:]
    return source_without_sidebar, structure_setup


def _inject_structure_setup(structure_block: str, structure_setup: str) -> str:
    if not structure_setup:
        return structure_block
    header = 'st.header("Structure generation")\n'
    if header not in structure_block:
        return structure_block
    insertion = (
        header
        + '\nwith st.expander("Catalyst structure setup", expanded=True):\n'
        + _indent_block(structure_setup, 4)
        + '\n'
    )
    return structure_block.replace(header, insertion, 1)


def _move_chat_prompt_after_conversation(chat_block: str) -> str:
    prompt_start = chat_block.find('chat_prompt = st.chat_input(')
    conversation_start = chat_block.find('with st.expander("Agent responses"', prompt_start)
    debug_start = chat_block.find('if st.session_state.params.get("chat_debug_planner"', conversation_start)
    if prompt_start == -1 or conversation_start == -1:
        return chat_block
    if debug_start == -1:
        debug_start = len(chat_block)

    before_prompt = chat_block[:prompt_start].rstrip() + "\n\n"
    prompt_block = chat_block[prompt_start:conversation_start].strip("\n")
    prompt_block = prompt_block.replace(
        '    _chat_add("assistant", response)',
        '    _chat_add("assistant", response)\n'
        '    if hasattr(st, "rerun"):\n'
        '        st.rerun()\n'
        '    else:\n'
        '        st.experimental_rerun()',
        1,
    ) + "\n\n"
    conversation_block = chat_block[conversation_start:debug_start].strip("\n") + "\n\n"
    debug_block = chat_block[debug_start:].lstrip("\n")
    return before_prompt + conversation_block + prompt_block + debug_block


def _split_workflow_and_agent(source: str) -> str:
    source, structure_setup = _extract_structure_setup(source)

    chat_start = source.find("# Chat box: lightweight workflow console")
    batch_start = source.find('st.header("Batch processing")', chat_start)
    structure_start = source.find('st.header("Structure generation")', batch_start)
    if chat_start == -1 or batch_start == -1 or structure_start == -1:
        return source

    before = source[:chat_start]
    chat_block = source[chat_start:batch_start]
    batch_block = source[batch_start:structure_start]
    structure_and_nersc = source[structure_start:]

    nersc_marker = "# NERSC job control and ISAAC record conversion"
    nersc_start = structure_and_nersc.find(nersc_marker)
    if nersc_start == -1:
        structure_block = structure_and_nersc
        nersc_block = ""
    else:
        structure_block = structure_and_nersc[:nersc_start]
        nersc_block = structure_and_nersc[nersc_start:]

    structure_block = _inject_structure_setup(structure_block, structure_setup)
    chat_block = chat_block.replace(
        "chat_cols = st.columns([1, 1, 1, 2])",
        "chat_cols = [st.container(), st.container(), st.container(), st.container()]",
    )
    chat_block = _move_chat_prompt_after_conversation(chat_block)

    workflow_block = structure_block + "\n\n" + batch_block + "\n\n" + nersc_block
    layout_block = (
        '# Polished page layout: sidebar agent console + main workflow workspace.\n'
        'p = st.session_state.params\n'
        'with st.sidebar:\n'
        '    st.header("Agent console")\n'
        '    st.caption("Ask the agent to update settings, generate inputs, upload to NERSC, submit jobs, check Slurm, or inspect remote files.")\n'
        f'{_indent_block(chat_block, 4)}\n\n'
        'st.header("Workflow workspace")\n'
        f'{workflow_block}'
    )
    return before + layout_block


def _build_runtime_app() -> str:
    source = _APP_PATH.read_text(encoding="utf-8")
    source = _patch_job_listing_logic(source)
    source = _split_workflow_and_agent(source)
    source = _patch_isaac_portal_upload(source)
    source = _polish_labels(source)
    source = source.replace(
        'WEB_APP_UPDATE_TAG = "v33_2026-07-01_slurm_any_job_status"',
        'WEB_APP_UPDATE_TAG = "v44_2026-07-02_isaac_portal_upload"',
        1,
    )
    return source


_runtime_source = _build_runtime_app()
if not _RUNTIME_PATH.exists() or _RUNTIME_PATH.read_text(encoding="utf-8") != _runtime_source:
    _RUNTIME_PATH.write_text(_runtime_source, encoding="utf-8")
runpy.run_path(str(_RUNTIME_PATH), run_name="__main__")
