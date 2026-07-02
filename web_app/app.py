# web_app/app.py
from __future__ import annotations

import io
import json
import os
import re
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Sequence

import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.structure_generator import StructureGenerator, DEFAULT_ADSORBATE_SITES
from tools.utils import read_structure_file
from tools.xas_input_generator import RelaxationInputGenerator, FEFFInputGenerator, FDMNESInputGenerator, VASPXASInputGenerator, NERSCScriptGenerator

try:
    from tools.nersc_job_manager import NERSCSFAPIClient, build_isaac_record_from_outputs, write_isaac_record
    NERSC_AVAILABLE = True
    NERSC_ERROR = None
except Exception as exc:
    NERSC_AVAILABLE = False
    NERSC_ERROR = exc

APP_UPDATE_TAG = "v38_2026-07-02_sidebar_agent_console_workspace"
DEFAULT_NERSC_ACCOUNT = os.environ.get("CO2RR_NERSC_ACCOUNT", os.environ.get("NERSC_ACCOUNT", "m5268"))
DEFAULT_REMOTE = os.environ.get("CO2RR_NERSC_REMOTE_DIR", "/pscratch/sd/h/hjia/CO2RR/web_xas_agent_runs/latest")
DEFAULT_POTCAR_DIR = "/global/common/software/nersc9/vasp/dependencies/pseudopotentials/PBE/potpaw_PBE"
METALS = ["Cu", "Au", "Ni", "Ag", "Pt", "Pd", "Ir", "Rh", "Al"]
ADSORBATES = ["clean", "CO", "OCCO", "CHO", "COCO", "CO2", "OH", "H", "CHOH", "CH", "CH2", "CH3", "CH4"]
COVERAGE = {"1/9 ML (3×3, default)": (3, 3), "1/16 ML (4×4)": (4, 4), "1/4 ML (2×2)": (2, 2), "1 ML (1×1)": (1, 1), "custom nx × ny": None}


def safe_index(options: Sequence[Any], value: Any, default: int = 0) -> int:
    return list(options).index(value) if value in options else default


def clean_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(text or "run")).strip("._-")[:80] or "run"


def zip_dir(path: Any) -> bytes:
    root = Path(path)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(root.rglob("*")):
            if p.is_file():
                zf.write(p, str(p.relative_to(root)))
    buf.seek(0)
    return buf.getvalue()


def tree(path: Any, limit: int = 220) -> str:
    root = Path(path)
    if not root.exists():
        return ""
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file():
            rows.append(str(p.relative_to(root)))
            if len(rows) >= limit:
                rows.append("...")
                break
    return "\n".join(rows)


def element_order(structure: Dict[str, Any]) -> list[str]:
    out = []
    for atom in structure.get("atoms", []):
        if atom not in out:
            out.append(atom)
    return out


def sample_name(structure: Dict[str, Any]) -> str:
    m = structure.get("metadata", {}) or {}
    ads = m.get("adsorbate") or "clean"
    facet = m.get("facet") or m.get("facet1") or "111"
    if m.get("element"):
        return clean_name(f"{m['element']}_{facet}_{ads}")
    return clean_name(f"{m.get('element1', 'M1')}_{m.get('element2', 'M2')}_{facet}_{ads}")


def write_make_potcar(structure: Dict[str, Any], out_dir: Any, potcar_dir: str) -> str:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    mapping = getattr(VASPXASInputGenerator, "DEFAULT_POTCAR_MAP", {})
    lines = ["#!/bin/bash", "set -euo pipefail", f"POTCAR_DIR=\"${{CO2RR_POTCAR_DIR:-{potcar_dir}}}\"", "rm -f POTCAR"]
    for el in element_order(structure):
        symbol = mapping.get(el, el)
        lines += [f"test -f \"${{POTCAR_DIR}}/{symbol}/POTCAR\"", f"cat \"${{POTCAR_DIR}}/{symbol}/POTCAR\" >> POTCAR"]
    path = out / "make_potcar.sh"
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    for spec in out.rglob("POTCAR.spec"):
        spec.unlink(missing_ok=True)
    return str(path)


def normalize_edge(absorber: str, edge: str) -> str:
    return "K" if str(edge).lower() == "auto" and absorber in {"C", "O", "Cu", "Ni", "Ag"} else ("L3" if str(edge).lower() == "auto" else str(edge).upper())


def generate_structure(p: Dict[str, Any], uploaded_file=None):
    sg = StructureGenerator()
    if p["structure_source"] == "Upload structure":
        if uploaded_file is None:
            raise ValueError("Upload POSCAR/CONTCAR/CIF/XYZ first.")
        out = Path(p["structure_output_dir"])
        out.mkdir(parents=True, exist_ok=True)
        local = out / clean_name(uploaded_file.name)
        local.write_bytes(uploaded_file.getbuffer())
        structure = read_structure_file(str(local))
        structure["metadata"] = {**structure.get("metadata", {}), "source_file": str(local), "source": "uploaded_structure"}
        paths = sg.save_structure(structure, str(out), "uploaded_structure")
        return structure, paths
    if p["structure_mode"] == "Single metal surface":
        base = sg.generate_surface(p["element"], p["facet"], (int(p["nx"]), int(p["ny"])), int(p["layers"]), float(p["vacuum"]))
        structure = sg.add_adsorbate(base, p["adsorbate"], site="auto", height=float(p["height"]), binding_element=p["element"]) if p["adsorbate"] != "clean" else base
        name = f"{p['element']}_{p['facet']}_{p['adsorbate']}"
    else:
        kw = {"element1_repeats": int(p["element1_repeats"]), "element2_repeats": int(p["element2_repeats"])} if p["interface_match_mode"] == "manual" else {}
        base = sg.generate_interface(p["element1"], p["element2"], p["facet"], p["facet"], (int(p["nx"]), int(p["ny"])), int(p["layers"]), int(p["layers"]), float(p["vacuum"]), match_mode=p["interface_match_mode"], **kw)
        structure = sg.add_adsorbate(base, p["adsorbate"], site="auto", height=float(p["height"]), binding_element=p["interface_binding_element"]) if p["adsorbate"] != "clean" else base
        name = f"{p['element1']}_{p['element2']}_{p['facet']}_{p['adsorbate']}"
    return structure, sg.save_structure(structure, p["structure_output_dir"], name)


def generate_relaxation(structure: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    out = Path(p["workspace_root"]) / "01_structure"
    out.mkdir(parents=True, exist_ok=True)
    StructureGenerator().save_structure(structure, str(out), "structure")
    files = RelaxationInputGenerator().write_inputs(structure, str(out))
    write_make_potcar(structure, out, p["potcar_dir"])
    submit = NERSCScriptGenerator().write_script(str(out), job_name=f"{sample_name(structure)}_structure_relax", software="VASP", filename="submit_relax.sh", account=p["nersc_account"], queue=p["nersc_queue"], nodes=int(p["nersc_nodes"]), walltime=p["nersc_walltime"], email=p["nersc_email"] or None)
    return {"directory": str(out), "files": files, "submit": submit}


def generate_xas(structure: Dict[str, Any], p: Dict[str, Any]) -> Dict[str, Any]:
    out = Path(p["workspace_root"]) / "02_XAS"
    out.mkdir(parents=True, exist_ok=True)
    absorber = p["xas_absorber"] if p["xas_absorber"] in structure.get("atoms", []) else element_order(structure)[0]
    edge = normalize_edge(absorber, p["xas_edge"])
    vasp = VASPXASInputGenerator().write_inputs(structure, str(out / "VASP"), absorber=absorber, method=p["vasp_method"], potcar_dir=p["potcar_dir"], account=p["nersc_account"], queue=p["nersc_queue"], nodes=int(p["nersc_nodes"]), walltime=p["nersc_walltime"], email=p["nersc_email"] or None, absorber_index=0, job_name=f"{sample_name(structure)}_XAS_VASP")
    write_make_potcar(structure, out / "VASP", p["potcar_dir"])
    opts = dict(FDMNESInputGenerator.DEFAULT_OPTIONS)
    opts.update({"Green": p["fdmnes_method"] == "Green", "SCF": p["fdmnes_scf"], "Quadrupole": p["fdmnes_quadrupole"], "Spinorbit": p["fdmnes_spinorbit"]})
    fdmnes = FDMNESInputGenerator().write_input(structure, str(out / "FDMNES"), absorber=absorber, edge=edge, radius=float(p["cluster_radius"]), energy_range=(float(p["e_min"]), float(p["e_step"]), float(p["e_max"])), options=opts)
    feff = FEFFInputGenerator().write_input(structure, str(out / "FEFF"), absorber=absorber, edge=edge, radius=float(p["cluster_radius"]))
    script = NERSCScriptGenerator()
    script.write_script(str(out / "FDMNES"), job_name=f"{sample_name(structure)}_XAS_FDMNES", software="FDMNES", account=p["nersc_account"], queue=p["nersc_queue"], nodes=int(p["nersc_nodes"]), walltime=p["nersc_walltime"], email=p["nersc_email"] or None)
    script.write_script(str(out / "FEFF"), job_name=f"{sample_name(structure)}_XAS_FEFF", software="FEFF", account=p["nersc_account"], queue=p["nersc_queue"], nodes=int(p["nersc_nodes"]), walltime=p["nersc_walltime"], email=p["nersc_email"] or None)
    return {"directory": str(out), "absorber": absorber, "edge": edge, "VASP": vasp, "FDMNES": fdmnes, "FEFF": feff}


def nersc_client(p: Dict[str, Any]) -> NERSCSFAPIClient:
    if not NERSC_AVAILABLE:
        raise RuntimeError(f"NERSC tools unavailable: {NERSC_ERROR}")
    return NERSCSFAPIClient(client_id=p["sfapi_client_id"] or None, private_key_path=p["sfapi_private_key_path"] or None, system=p["nersc_system"])


def run_agent_command(text: str, p: Dict[str, Any]) -> str:
    low = text.lower()
    try:
        if "generate" in low and "structure" in low:
            st.session_state.last_structure, st.session_state.last_paths = generate_structure(p, st.session_state.get("uploaded_structure"))
            return f"Generated structure: `{st.session_state.last_paths.get('poscar')}`"
        if "relax" in low:
            if "last_structure" not in st.session_state:
                st.session_state.last_structure, st.session_state.last_paths = generate_structure(p, st.session_state.get("uploaded_structure"))
            st.session_state.last_relax = generate_relaxation(st.session_state.last_structure, p)
            return f"Generated relaxation inputs: `{st.session_state.last_relax['directory']}`"
        if "xas" in low:
            if "last_structure" not in st.session_state:
                st.session_state.last_structure, st.session_state.last_paths = generate_structure(p, st.session_state.get("uploaded_structure"))
            st.session_state.last_xas = generate_xas(st.session_state.last_structure, p)
            return f"Generated XAS inputs: `{st.session_state.last_xas['directory']}`"
        if "batch" in low or "pathway" in low:
            return "Use the Batch generation workspace tab to select pathway adsorbates and create the package."
        if "upload" in low and "nersc" in low:
            result = nersc_client(p).upload_directory(p["workspace_root"], p["remote_dir"], overwrite=True)
            return f"Uploaded `{p['workspace_root']}` to `{p['remote_dir']}`."
        if "submit" in low:
            remote_script = p["remote_submit_script"] if p["remote_submit_script"].startswith("/") else str(PurePosixPath(p["remote_dir"]) / p["remote_submit_script"])
            result = nersc_client(p).submit_job(remote_script)
            return f"Submitted `{remote_script}`: `{json.dumps(result)[:1000]}`"
        if "list jobs" in low or "squeue" in low:
            result = nersc_client(p).list_user_jobs(username=p["nersc_username"] or None)
            return str(result.get("output", result))[:3000]
        if "isaac" in low or "convert" in low:
            record = build_isaac_record_from_outputs(p["isaac_output_dir"], software=p["isaac_software"], sample_name=p["isaac_sample_name"])
            path = write_isaac_record(p["isaac_output_dir"], record)
            return f"Wrote ISAAC draft: `{path}`"
        return "Try: generate structure, generate relaxation inputs, generate XAS inputs, upload to NERSC, submit selected job, list jobs, or convert ISAAC."
    except Exception as exc:
        return f"Failed: {exc}"


if "params" not in st.session_state:
    st.session_state.params = {"structure_source":"Build from settings","structure_mode":"Single metal surface","element":"Cu","element1":"Cu","element2":"Au","facet":"111","coverage":"1/9 ML (3×3, default)","nx":3,"ny":3,"layers":4,"vacuum":15.0,"adsorbate":"CO","height":2.0,"interface_match_mode":"auto","element1_repeats":4,"element2_repeats":4,"interface_binding_element":"Cu","structure_output_dir":"generated_outputs/web_xas_agent/structure","workspace_root":"generated_outputs/web_xas_agent/package","xas_absorber":"Cu","xas_edge":"K","cluster_radius":6.0,"e_min":-5.0,"e_step":0.2,"e_max":50.0,"vasp_method":"PBE","potcar_dir":DEFAULT_POTCAR_DIR,"fdmnes_method":"Green","fdmnes_scf":False,"fdmnes_quadrupole":False,"fdmnes_spinorbit":False,"nersc_system":"perlmutter","nersc_username":os.environ.get("USER","hjia"),"nersc_account":DEFAULT_NERSC_ACCOUNT,"nersc_queue":"regular","nersc_nodes":1,"nersc_walltime":"02:00:00","nersc_email":"","sfapi_client_id":os.environ.get("NERSC_SFAPI_CLIENT_ID",""),"sfapi_private_key_path":os.environ.get("NERSC_SFAPI_PRIVATE_KEY_PATH",""),"remote_dir":DEFAULT_REMOTE,"remote_submit_script":"01_structure/submit_relax.sh","batch_adsorbates":["clean","CO2","CO","CHO","OCCO","COCO","OH","H"],"batch_output_dir":"generated_outputs/web_xas_agent/batch","isaac_output_dir":"generated_outputs/web_xas_agent/downloaded_outputs","isaac_software":"auto","isaac_sample_name":"not_specified"}
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role":"assistant","message":"Configure settings below, or ask me to generate structures, XAS inputs, NERSC jobs, and ISAAC records."}]

p = st.session_state.params
st.set_page_config(page_title="CO2RR XAS Agent", layout="wide")
st.markdown("<style>section[data-testid='stSidebar']{width:430px!important;min-width:430px!important}.block-container{max-width:1600px;padding-top:1.4rem}</style>", unsafe_allow_html=True)

with st.sidebar:
    st.header("Agent console")
    for item in st.session_state.chat_history[-5:]:
        st.markdown(f"**{item['role'].capitalize()}**: {item['message']}")
    with st.form("agent_console", clear_on_submit=True):
        cmd = st.text_area("Natural-language command", placeholder="generate structure and XAS inputs", height=85)
        submit_cmd = st.form_submit_button("Run command", type="primary")
    if submit_cmd and cmd.strip():
        st.session_state.chat_history.append({"role":"user","message":cmd.strip()})
        st.session_state.chat_history.append({"role":"assistant","message":run_agent_command(cmd.strip(), p)})
        st.rerun()
    st.divider()
    st.header("Catalyst structure")
    p["structure_source"] = st.radio("Structure source", ["Build from settings", "Upload structure"], index=safe_index(["Build from settings","Upload structure"], p["structure_source"]))
    if p["structure_source"] == "Upload structure":
        st.session_state.uploaded_structure = st.file_uploader("Upload POSCAR/CONTCAR/CIF/XYZ")
    else:
        p["structure_mode"] = st.radio("System type", ["Single metal surface", "Interface"], index=safe_index(["Single metal surface","Interface"], p["structure_mode"]))
        if p["structure_mode"] == "Single metal surface":
            p["element"] = st.selectbox("Metal", METALS, index=safe_index(METALS, p["element"]))
            p["coverage"] = st.selectbox("Nominal coverage", list(COVERAGE), index=safe_index(list(COVERAGE), p["coverage"]))
            if COVERAGE[p["coverage"]]:
                p["nx"], p["ny"] = COVERAGE[p["coverage"]]
        else:
            p["element1"] = st.selectbox("Interface metal 1", METALS, index=safe_index(METALS, p["element1"]))
            p["element2"] = st.selectbox("Interface metal 2", METALS, index=safe_index(METALS, p["element2"], 1))
            p["interface_match_mode"] = st.radio("Interface repeat mode", ["auto", "manual"], index=safe_index(["auto","manual"], p["interface_match_mode"]))
            p["interface_binding_element"] = st.selectbox("Adsorbate binding metal", [p["element1"], p["element2"]], index=0)
        c1, c2 = st.columns(2)
        p["nx"] = c1.number_input("nx",1,30,int(p["nx"]))
        p["ny"] = c2.number_input("ny",1,30,int(p["ny"]))
        p["facet"] = st.selectbox("Facet", ["111"], index=0)
        p["layers"] = st.number_input("Layers",2,12,int(p["layers"]))
        p["vacuum"] = st.number_input("Vacuum, Å",5.0,40.0,float(p["vacuum"]))
        p["adsorbate"] = st.selectbox("Adsorbate", ADSORBATES, index=safe_index(ADSORBATES, p["adsorbate"]))
        p["height"] = st.number_input("Adsorbate height, Å",0.5,5.0,float(p["height"]),step=0.1)
        if p["adsorbate"] != "clean":
            st.caption(f"Default site: {DEFAULT_ADSORBATE_SITES.get(p['adsorbate'], 'top')}")
    p["structure_output_dir"] = st.text_input("Structure output directory", p["structure_output_dir"])
    st.divider()
    st.header("XAS/HPC settings")
    p["workspace_root"] = st.text_input("Workflow package root", p["workspace_root"])
    c1, c2 = st.columns(2)
    p["xas_absorber"] = c1.text_input("Absorber", p["xas_absorber"])
    p["xas_edge"] = c2.selectbox("Edge", ["K","L3","auto"], index=safe_index(["K","L3","auto"], p["xas_edge"]))
    p["cluster_radius"] = st.number_input("Cluster radius, Å",2.0,12.0,float(p["cluster_radius"]),step=0.5)
    with st.expander("Software details", expanded=False):
        p["vasp_method"] = st.selectbox("VASP method", ["PBE","GW"], index=safe_index(["PBE","GW"], p["vasp_method"]))
        p["potcar_dir"] = st.text_input("POTCAR root", p["potcar_dir"])
        p["fdmnes_method"] = st.selectbox("FDMNES method", ["Green","FDM"], index=safe_index(["Green","FDM"], p["fdmnes_method"]))
        p["fdmnes_scf"] = st.checkbox("SCF", p["fdmnes_scf"])
        p["fdmnes_quadrupole"] = st.checkbox("Quadrupole", p["fdmnes_quadrupole"])
        p["fdmnes_spinorbit"] = st.checkbox("Spinorbit", p["fdmnes_spinorbit"])
    with st.expander("NERSC", expanded=True):
        p["nersc_account"] = st.text_input("Project/account", p["nersc_account"])
        p["nersc_queue"] = st.selectbox("Queue/QOS", ["regular","debug","premium","shared"], index=safe_index(["regular","debug","premium","shared"], p["nersc_queue"]))
        p["nersc_nodes"] = st.number_input("Nodes",1,64,int(p["nersc_nodes"]))
        p["nersc_walltime"] = st.text_input("Walltime", p["nersc_walltime"])
        p["nersc_email"] = st.text_input("Email", p["nersc_email"])
        p["sfapi_client_id"] = st.text_input("SF API client ID", p["sfapi_client_id"], type="password")
        p["sfapi_private_key_path"] = st.text_input("Private key path", p["sfapi_private_key_path"])

st.title("CO2RR XAS Agent")
st.caption(f"Update tag: `{APP_UPDATE_TAG}`")
st.info("Configure the catalyst structure and XAS/HPC settings in the sidebar. Use the workflow workspace for generated files, jobs, and ISAAC records; use the agent console for natural-language commands.")
st.markdown("# Workflow workspace")
structure_tab, batch_tab, nersc_tab, isaac_tab = st.tabs(["Structure generation", "Batch generation", "NERSC jobs", "ISAAC records"])

with structure_tab:
    st.markdown("## Structure generation")
    c1, c2, c3 = st.columns(3)
    if c1.button("Generate structure", type="primary"):
        try:
            st.session_state.last_structure, st.session_state.last_paths = generate_structure(p, st.session_state.get("uploaded_structure"))
            st.success(f"POSCAR: `{st.session_state.last_paths.get('poscar')}`")
        except Exception as exc:
            st.exception(exc)
    if c2.button("Generate VASP relaxation inputs"):
        try:
            if "last_structure" not in st.session_state:
                st.session_state.last_structure, st.session_state.last_paths = generate_structure(p, st.session_state.get("uploaded_structure"))
            st.session_state.last_relax = generate_relaxation(st.session_state.last_structure, p)
            st.success(f"Relaxation folder: `{st.session_state.last_relax['directory']}`")
        except Exception as exc:
            st.exception(exc)
    if c3.button("Generate XAS inputs"):
        try:
            if "last_structure" not in st.session_state:
                st.session_state.last_structure, st.session_state.last_paths = generate_structure(p, st.session_state.get("uploaded_structure"))
            st.session_state.last_xas = generate_xas(st.session_state.last_structure, p)
            st.success(f"XAS folder: `{st.session_state.last_xas['directory']}`")
        except Exception as exc:
            st.exception(exc)
    if "last_structure" in st.session_state:
        s = st.session_state.last_structure
        st.metric("Atoms", len(s.get("atoms", [])))
        st.json(s.get("metadata", {}))
    for key, label in [("last_relax", "Relaxation"), ("last_xas", "XAS")]:
        if key in st.session_state:
            st.subheader(f"{label} files")
            st.code(tree(st.session_state[key]["directory"]), language="text")
            st.download_button(f"Download {label} ZIP", zip_dir(st.session_state[key]["directory"]), file_name=f"{label.lower()}_inputs.zip", mime="application/zip")

with batch_tab:
    st.markdown("## Batch generation")
    p["batch_output_dir"] = st.text_input("Batch output directory", p["batch_output_dir"])
    p["batch_adsorbates"] = st.multiselect("Adsorbates / pathway structures", ADSORBATES, default=[x for x in p["batch_adsorbates"] if x in ADSORBATES])
    if st.button("Generate batch package", type="primary"):
        try:
            root = Path(p["batch_output_dir"]) / datetime.now().strftime("batch_%Y%m%d_%H%M%S")
            root.mkdir(parents=True, exist_ok=True)
            old_root, old_ads = p["workspace_root"], p["adsorbate"]
            summary=[]
            for ads in p["batch_adsorbates"]:
                p["adsorbate"] = ads
                structure, _ = generate_structure(p)
                p["workspace_root"] = str(root / sample_name(structure))
                generate_relaxation(structure, p)
                generate_xas(structure, p)
                summary.append({"adsorbate": ads, "directory": p["workspace_root"]})
            p["workspace_root"], p["adsorbate"] = old_root, old_ads
            st.session_state.last_batch = {"run_dir": str(root), "summary": summary}
            st.success(f"Batch package: `{root}`")
        except Exception as exc:
            st.exception(exc)
    if "last_batch" in st.session_state:
        st.json(st.session_state.last_batch["summary"])
        st.code(tree(st.session_state.last_batch["run_dir"]), language="text")
        st.download_button("Download batch ZIP", zip_dir(st.session_state.last_batch["run_dir"]), file_name=f"{Path(st.session_state.last_batch['run_dir']).name}.zip", mime="application/zip")

with nersc_tab:
    st.markdown("## NERSC jobs")
    if not NERSC_AVAILABLE:
        st.warning(f"NERSC job control unavailable: {NERSC_ERROR}")
    p["remote_dir"] = st.text_input("Remote run directory", p["remote_dir"])
    p["remote_submit_script"] = st.text_input("Submit script path", p["remote_submit_script"])
    c1, c2, c3 = st.columns(3)
    if c1.button("Upload workspace", disabled=not NERSC_AVAILABLE):
        try:
            st.json(nersc_client(p).upload_directory(p["workspace_root"], p["remote_dir"], overwrite=True))
        except Exception as exc:
            st.exception(exc)
    if c2.button("Submit selected job", disabled=not NERSC_AVAILABLE):
        try:
            st.json(nersc_client(p).submit_job(str(PurePosixPath(p["remote_dir"]) / p["remote_submit_script"])))
        except Exception as exc:
            st.exception(exc)
    if c3.button("List my jobs", disabled=not NERSC_AVAILABLE):
        try:
            st.code(str(nersc_client(p).list_user_jobs(username=p["nersc_username"] or None).get("output", "")), language="text")
        except Exception as exc:
            st.exception(exc)

with isaac_tab:
    st.markdown("## ISAAC records")
    p["isaac_output_dir"] = st.text_input("Local output directory", p["isaac_output_dir"])
    p["isaac_software"] = st.selectbox("Software", ["auto","VASP","FDMNES","FEFF"], index=safe_index(["auto","VASP","FDMNES","FEFF"], p["isaac_software"]))
    p["isaac_sample_name"] = st.text_input("Sample name", p["isaac_sample_name"])
    if st.button("Convert outputs to ISAAC draft", type="primary"):
        try:
            record = build_isaac_record_from_outputs(p["isaac_output_dir"], software=p["isaac_software"], sample_name=p["isaac_sample_name"])
            path = write_isaac_record(p["isaac_output_dir"], record)
            st.success(f"Wrote `{path}`")
            st.json(record)
        except Exception as exc:
            st.exception(exc)
