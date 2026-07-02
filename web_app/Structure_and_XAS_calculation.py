# web_app/Structure_and_XAS_calculation.py
"""Polished Streamlit entrypoint for the Structure and XAS workflow.

Run with:
    streamlit run web_app/Structure_and_XAS_calculation.py

This keeps ``web_app/app.py`` as the legacy implementation, but renders it with a
cleaner information architecture:

- sidebar = agent console only
- main workspace = structure generation, batch generation, and NERSC agent
- catalyst structure setup = inside the Structure generation section
"""

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
        'structure/XAS input generation, batch generation, NERSC jobs, and ISAAC records.")'
    )
    source = source.replace(title_line, subtitle, 1)
    return source


def _extract_structure_setup(source: str) -> tuple[str, str]:
    """Remove the legacy sidebar structure controls and return them as a main-page block."""
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
    """Place the prompt box after the visible conversation, ChatGPT-style."""
    prompt_start = chat_block.find('chat_prompt = st.chat_input(')
    conversation_start = chat_block.find('with st.expander("Agent responses"', prompt_start)
    debug_start = chat_block.find('if st.session_state.params.get("chat_debug_planner"', conversation_start)
    if prompt_start == -1 or conversation_start == -1:
        return chat_block
    if debug_start == -1:
        debug_start = len(chat_block)

    before_prompt = chat_block[:prompt_start].rstrip() + "\n\n"
    prompt_block = chat_block[prompt_start:conversation_start].strip("\n") + "\n\n"
    conversation_block = chat_block[conversation_start:debug_start].strip("\n") + "\n\n"
    debug_block = chat_block[debug_start:].lstrip("\n")
    return before_prompt + conversation_block + prompt_block + debug_block


def _split_workflow_and_agent(source: str) -> str:
    source, structure_setup = _extract_structure_setup(source)

    chat_start = source.find("# Chat box: lightweight workflow console")
    batch_start = source.find('st.header("Batch processing")', chat_start)
    structure_start = source.find('st.header("Structure generation")', batch_start)
    if chat_start == -1 or batch_start == -1 or structure_start == -1:
        # Keep the legacy rendering if anchors change in a future app.py.
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

    # The original chat UI used four horizontal columns. In the sidebar those are
    # replaced by stacked containers without changing the backend selection logic.
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
    source = _split_workflow_and_agent(source)
    source = _polish_labels(source)
    source = source.replace(
        'WEB_APP_UPDATE_TAG = "v33_2026-07-01_slurm_any_job_status"',
        'WEB_APP_UPDATE_TAG = "v41_2026-07-02_agent_prompt_after_conversation"',
        1,
    )
    return source


_runtime_source = _build_runtime_app()
if not _RUNTIME_PATH.exists() or _RUNTIME_PATH.read_text(encoding="utf-8") != _runtime_source:
    _RUNTIME_PATH.write_text(_runtime_source, encoding="utf-8")
runpy.run_path(str(_RUNTIME_PATH), run_name="__main__")
