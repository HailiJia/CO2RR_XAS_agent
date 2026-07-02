# web_app/Structure_and_XAS_calculation.py
"""Polished Streamlit entrypoint for the Structure and XAS workflow.

Run with:
    streamlit run web_app/Structure_and_XAS_calculation.py

This keeps ``web_app/app.py`` as the legacy implementation, but renders it with a
cleaner information architecture:

- sidebar = structure/XAS/HPC configuration
- main column = workflow workspace
- right column = agent console
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
        'st.header("Batch processing")': 'st.header("Batch / pathway generation")',
        '"Generate inputs for multiple structures or a reaction pathway"': '"Generate input packages for multiple structures or reaction pathways"',
        '"Generate VASP relaxation folders"': '"Include VASP relaxation inputs"',
        '"Generate XAS folders"': '"Include XAS inputs"',
        '"Batch absorber"': '"Absorber selection"',
        '"use selected absorber when present"': '"Use selected absorber if available"',
    }
    for old, new in replacements.items():
        source = source.replace(old, new)

    title_line = 'st.title("CO2RR XAS Agent")'
    subtitle = (
        'st.title("CO2RR XAS Agent")\n'
        'st.caption("Configure the catalyst structure and XAS/HPC settings in the sidebar. '
        'Use the workflow workspace for generated files, jobs, and ISAAC records; use the agent console for natural-language commands.")'
    )
    source = source.replace(title_line, subtitle, 1)
    return source


def _split_workflow_and_agent(source: str) -> str:
    chat_start = source.find("# Chat box: lightweight workflow console")
    batch_start = source.find('st.header("Batch processing")', chat_start)
    if chat_start == -1 or batch_start == -1:
        # Keep the legacy rendering if anchors change in a future app.py.
        return source

    before = source[:chat_start]
    chat_block = source[chat_start:batch_start]
    workflow_block = source[batch_start:]

    layout_block = (
        '# Polished page layout: workflow center + agent console side panel.\n'
        'workflow_col, agent_col = st.columns([2.35, 1.0], gap="large")\n\n'
        'with agent_col:\n'
        '    st.subheader("Agent console")\n'
        '    st.caption("Ask the agent to update settings, generate inputs, upload to NERSC, submit jobs, check Slurm, or inspect remote files.")\n'
        f'{_indent_block(chat_block, 4)}\n\n'
        'with workflow_col:\n'
        '    st.subheader("Workflow workspace")\n'
        f'{_indent_block(workflow_block, 4)}'
    )
    return before + layout_block


def _build_runtime_app() -> str:
    source = _APP_PATH.read_text(encoding="utf-8")
    source = _split_workflow_and_agent(source)
    source = _polish_labels(source)
    source = source.replace(
        'WEB_APP_UPDATE_TAG = "v33_2026-07-01_slurm_any_job_status"',
        'WEB_APP_UPDATE_TAG = "v34_2026-07-02_polished_layout"',
        1,
    )
    return source


_RUNTIME_PATH.write_text(_build_runtime_app(), encoding="utf-8")
runpy.run_path(str(_RUNTIME_PATH), run_name="__main__")
