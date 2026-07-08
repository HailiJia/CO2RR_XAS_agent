# web_app/main.py

import io
import json
import os
import re
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Make repo root importable when running: streamlit run web_app/main.py
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.structure_generator import (
    StructureGenerator,
    DEFAULT_ADSORBATE_SITES,
    INTERFACE_GENERATOR_UPDATE_TAG,
)
from tools.utils import read_structure_file
from tools.contact_checks import minimum_pair as element_aware_minimum_pair
from tools.xas_input_generator import (
    RelaxationInputGenerator,
    FEFFInputGenerator,
    FDMNESInputGenerator,
    VASPXASInputGenerator,
    NERSCScriptGenerator,
)
from tools.web_nersc_workflow_integration import (
    prepare_single_structure_workflow_package,
    run_remote_workflow_script,
)

WEB_APP_UPDATE_TAG = "v45_2026-07-06_nersc_full_workflow_ui"

try:
    from tools.nersc_job_manager import (
        NERSC_JOB_MANAGER_UPDATE_TAG,
        NERSCSFAPIClient,
        NERSCSFAPIError,
        build_isaac_record_from_outputs,
        extract_archive_bytes,
        write_isaac_record,
    )
    NERSC_JOB_CONTROL_AVAILABLE = True
    NERSC_JOB_CONTROL_IMPORT_ERROR = None
except Exception as exc:
    NERSC_JOB_CONTROL_AVAILABLE = False
    NERSC_JOB_CONTROL_IMPORT_ERROR = exc


METAL_CHOICES = ["Cu", "Au", "Ni", "Ag", "Pt", "Pd", "Ir", "Rh", "Al"]
ADSORBATE_CHOICES = ["clean", "CO", "OCCO", "CHO", "COCO", "CO2", "OH", "H", "CHOH", "CH", "CH2", "CH3", "CH4"]

# Same colors are used in the 3D viewer and the legend.
ELEMENT_COLORS = {
    "Cu": "#b87333",  # copper
    "Au": "#ffd700",  # gold
    "Ag": "#c0c0c0",
    "Ni": "#4daf4a",
    "Pt": "#6f6f6f",
    "Pd": "#9e9e9e",
    "Ir": "#4b5563",
    "Rh": "#7dd3fc",
    "Al": "#d1d5db",
    "C": "#404040",
    "O": "#e31a1c",
    "H": "#f8f8ff",
    "N": "#1f78b4",
}
DEFAULT_ELEMENT_COLOR = "#8a8a8a"

DEFAULT_NERSC_POTCAR_DIR = "/global/common/software/nersc9/vasp/dependencies/pseudopotentials/PBE/potpaw_PBE"
DEFAULT_NERSC_ACCOUNT = os.environ.get("CO2RR_NERSC_ACCOUNT", os.environ.get("NERSC_ACCOUNT", "m5268"))
DEFAULT_NERSC_REMOTE_BASE = os.environ.get("CO2RR_NERSC_REMOTE_BASE", "/pscratch/sd/h/hjia/CO2RR")
DEFAULT_NERSC_REMOTE_RUN_DIR = f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/latest"
DEFAULT_VASP6_MPI_RANKS_PER_NODE = 64
DEFAULT_VASP6_OMP_NUM_THREADS = 2
DEFAULT_VASP6_CPUS_PER_TASK = 4
DEFAULT_VASP_SRUN_COMMAND = 'srun -n "${SLURM_NTASKS:-64}" -c 4 --cpu-bind=cores vasp_std'
DEFAULT_VASP6_LAYOUT_NAME = "nersc_vasp6_cpu_64mpi_per_node_2omp_c4"
DEFAULT_VASP_MODULE = "vasp/6.4.3-cpu"
DEFAULT_ISAAC_CODE_VERSION = "VASP 6.4.3"
DEFAULT_ISAAC_ORGANIZATION = "ANL"


# Optional Argonne/ALCF LLM planner. The app works without this in rule-based
# mode. When Chat backend is switched to Argonne ALCF, the planner calls the
# ALCF Inference Endpoints through their OpenAI-compatible chat-completions API.
# The `openai` Python package is only the client library here; requests go to
# the ALCF base_url below, not to OpenAI-hosted models.
DEFAULT_ALCF_CLUSTER = os.environ.get("CO2RR_ALCF_INFERENCE_CLUSTER", "Metis")
DEFAULT_ALCF_BASE_URLS = {
    "Sophia": "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1",
    "Metis": "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1",
}
DEFAULT_ALCF_MODEL_BY_CLUSTER = {
    "Sophia": os.environ.get("CO2RR_ALCF_SOPHIA_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct"),
    "Metis": os.environ.get("CO2RR_ALCF_METIS_MODEL", "gpt-oss-120b"),
}
ALCF_INFERENCE_ACCESS_TOKEN = os.environ.get("ALCF_INFERENCE_ACCESS_TOKEN", "")
DEFAULT_OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o")
DEFAULT_ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")



def atoms_to_xyz(atoms, positions, comment="Generated structure"):
    """Convert atom symbols and Cartesian positions to XYZ text for 3Dmol.js."""
    lines = [str(len(atoms)), comment]
    for atom, pos in zip(atoms, positions):
        lines.append(f"{atom} {pos[0]:.8f} {pos[1]:.8f} {pos[2]:.8f}")
    return "\n".join(lines)


def view_structure_3dmol(atoms, positions, present_elements, substrate_n=None, height=560):
    """Render the generated structure using 3Dmol.js.

    The viewer uses the exact returned atom coordinates. Substrate atoms are
    rendered as spheres only. Adsorbate atoms are rendered as spheres, with
    internal adsorbate bonds drawn as explicit cylinders from the generated
    coordinates. This avoids 3Dmol/browser-guessed metal-metal or metal-adsorbate
    bonds while keeping adsorbates readable.
    """
    atoms = list(atoms)
    positions = np.asarray(positions, dtype=float)
    xyz_text = atoms_to_xyz(atoms, positions, comment="CO2RR generated structure")
    xyz_json = json.dumps(xyz_text)
    atom_json = json.dumps([
        {"elem": atom, "x": float(pos[0]), "y": float(pos[1]), "z": float(pos[2])}
        for atom, pos in zip(atoms, positions)
    ])
    color_json = json.dumps({el: ELEMENT_COLORS.get(el, DEFAULT_ELEMENT_COLOR) for el in present_elements})
    substrate_n_js = "null" if substrate_n is None else str(int(substrate_n))

    html = f"""
    <div id="viewer" style="width: 100%; height: {height}px; position: relative;"></div>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <script>
      const xyz = {xyz_json};
      const atoms = {atom_json};
      const elementColors = {color_json};
      const substrateN = {substrate_n_js};
      const metalElements = new Set(["Cu", "Au", "Ag", "Ni", "Pt", "Pd", "Ir", "Rh", "Al"]);

      const viewer = $3Dmol.createViewer("viewer", {{ backgroundColor: "white" }});
      viewer.addModel(xyz, "xyz");

      // Exact coordinates only: substrate atoms are spheres. Do not allow the
      // viewer to infer metal-metal or metal-adsorbate bonds.
      viewer.setStyle({{}}, {{ sphere: {{ scale: 0.30, color: "{DEFAULT_ELEMENT_COLOR}" }} }});
      for (const [elem, color] of Object.entries(elementColors)) {{
        const sphereScale = metalElements.has(elem) ? 0.42 : 0.32;
        viewer.setStyle({{ elem: elem }}, {{ sphere: {{ scale: sphereScale, color: color }} }});
      }}

      function distance(a, b) {{
        const dx = a.x - b.x;
        const dy = a.y - b.y;
        const dz = a.z - b.z;
        return Math.sqrt(dx * dx + dy * dy + dz * dz);
      }}

      function adsorbateBondCutoff(e1, e2) {{
        if (e1 === "H" && e2 === "H") return 0.95;
        if (e1 === "H" || e2 === "H") return 1.35;
        // C-O, C-C, C-N, O-H-like heavy-atom bonds in simple adsorbates.
        return 1.85;
      }}

      function shouldDrawAdsorbateBond(a, b) {{
        if (metalElements.has(a.elem) || metalElements.has(b.elem)) return false;
        const d = distance(a, b);
        return d > 0.45 && d <= adsorbateBondCutoff(a.elem, b.elem);
      }}

      // Draw only internal adsorbate bonds as explicit cylinders.
      if (substrateN !== null && substrateN < atoms.length) {{
        for (let i = substrateN; i < atoms.length; i++) {{
          for (let j = i + 1; j < atoms.length; j++) {{
            const a = atoms[i];
            const b = atoms[j];
            if (shouldDrawAdsorbateBond(a, b)) {{
              viewer.addCylinder({{
                start: {{ x: a.x, y: a.y, z: a.z }},
                end: {{ x: b.x, y: b.y, z: b.z }},
                radius: 0.08,
                fromCap: 1,
                toCap: 1,
                color: "#555555"
              }});
            }}
          }}
        }}
      }}

      viewer.zoomTo();
      viewer.render();
    </script>
    """
    components.html(html, height=height)


def show_color_legend(atoms):
    """Show color mapping for all elements present in the current structure."""
    present = sorted(set(atoms), key=lambda x: (x not in ["Cu", "Au", "C", "O", "H"], x))
    st.write("**Element color legend**")
    cols = st.columns(min(len(present), 5) or 1)
    for i, el in enumerate(present):
        color = ELEMENT_COLORS.get(el, DEFAULT_ELEMENT_COLOR)
        with cols[i % len(cols)]:
            st.markdown(
                f"<div style='display:flex; align-items:center; gap:0.45rem; margin-bottom:0.25rem;'>"
                f"<span style='display:inline-block; width:16px; height:16px; border-radius:50%; "
                f"background:{color}; border:1px solid #555;'></span>"
                f"<span><b>{el}</b></span></div>",
                unsafe_allow_html=True,
            )


def min_pair_distance(positions):
    """Simple non-PBC overlap check."""
    positions = np.asarray(positions, dtype=float)
    dmin = float("inf")
    pair = None
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            d = float(np.linalg.norm(positions[i] - positions[j]))
            if d < dmin:
                dmin = d
                pair = (i, j)
    return dmin, pair


def top_layer_indices(positions, tol=0.5):
    positions = np.asarray(positions, dtype=float)
    z_max = positions[:, 2].max()
    return np.where(np.abs(positions[:, 2] - z_max) < tol)[0]


def periodic_delta(value, ref, length):
    delta = abs(float(value) - float(ref))
    if length and length > 1e-12:
        delta = min(delta, length - delta)
    return delta


def get_interface_site_index(structure, preferred_element="Cu"):
    """Return site_index for the top-layer atom closest to the interface line.

    Works for both the older split_axis='x' geometry and the updated split_axis='y'
    geometry. For the updated geometry, the interface line is along x and the split
    coordinate is y, so the selected atom is near y = split and near the cell
    center along x.
    """
    atoms = structure["atoms"]
    positions = np.asarray(structure["positions"], dtype=float)
    cell = np.asarray(structure["cell"], dtype=float)
    metadata = structure.get("metadata", {})
    interface = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}

    top_indices = top_layer_indices(positions)
    if len(top_indices) == 0:
        return 0

    split_axis_name = interface.get("split_axis", "y")
    split_axis = 1 if split_axis_name == "y" else 0
    line_axis = 0 if split_axis == 1 else 1
    split_coord = interface.get("split_coordinate")
    if split_coord is None:
        return 0
    split_coord = float(split_coord)

    line_length = float(cell[line_axis, line_axis]) if cell[line_axis, line_axis] else None
    line_center = 0.5 * line_length if line_length else float(np.mean(positions[top_indices, line_axis]))

    candidates = [idx for idx in top_indices if atoms[idx] == preferred_element]
    if not candidates:
        candidates = list(top_indices)

    def score(idx):
        return (
            abs(float(positions[idx, split_axis]) - split_coord)
            + 0.25 * periodic_delta(positions[idx, line_axis], line_center, line_length)
        )

    chosen_global_idx = min(candidates, key=score)
    return list(top_indices).index(chosen_global_idx)


def get_center_site_index(structure, preferred_element=None):
    """Return site_index for top-layer atom closest to the cell center."""
    atoms = structure["atoms"]
    positions = np.asarray(structure["positions"], dtype=float)
    cell = np.asarray(structure["cell"], dtype=float)

    top_indices = top_layer_indices(positions)
    if len(top_indices) == 0:
        return 0

    center_xy = 0.5 * (cell[0, :2] + cell[1, :2])
    candidates = list(top_indices)
    if preferred_element is not None:
        element_candidates = [idx for idx in top_indices if atoms[idx] == preferred_element]
        if element_candidates:
            candidates = element_candidates

    chosen_global_idx = min(candidates, key=lambda idx: np.linalg.norm(positions[idx, :2] - center_xy))
    return list(top_indices).index(chosen_global_idx)


def nearest_substrate_atom(pos, atoms, positions, substrate_n):
    d = np.linalg.norm(np.asarray(positions[:substrate_n]) - np.asarray(pos), axis=1)
    idx = int(np.argmin(d))
    return idx, atoms[idx], float(d[idx])


def adsorbate_geometry_report(structure, substrate_n):
    """Return user-facing geometry checks after adsorbate placement."""
    atoms = structure["atoms"]
    positions = np.asarray(structure["positions"], dtype=float)
    metadata = structure.get("metadata", {})
    ads = metadata.get("adsorbate")
    reports = []

    if not ads or ads == "clean" or len(atoms) <= substrate_n:
        return reports

    ads_atoms = atoms[substrate_n:]
    ads_pos = positions[substrate_n:]

    # Report all bond-like distances within the adsorbate that are short enough to matter.
    for i in range(len(ads_atoms)):
        for j in range(i + 1, len(ads_atoms)):
            d = float(np.linalg.norm(ads_pos[i] - ads_pos[j]))
            if d < 2.2:
                reports.append((f"{ads} internal {ads_atoms[i]}{i + 1}-{ads_atoms[j]}{j + 1}", d, "Å", d > 0.5))

    binding_symbol = StructureGenerator()._binding_atom_symbol(ads)
    binding_indices = [i for i, el in enumerate(ads_atoms) if el == binding_symbol]
    if not binding_indices:
        binding_indices = [0]

    for bind_i in binding_indices[:2]:
        _, elem, bind = nearest_substrate_atom(ads_pos[bind_i], atoms, positions, substrate_n)
        reports.append((f"{ads} binding atom {ads_atoms[bind_i]}{bind_i + 1}-nearest substrate ({elem})", bind, "Å", 1.3 <= bind <= 3.0))

    return reports


def parse_prompt(prompt):
    """Simple first-pass parser. Sidebar values remain the source of truth."""
    text = prompt.lower()
    parsed = {}

    if ("single" in text or "surface" in text or "pure" in text or "pure metal" in text) and "interface" not in text:
        parsed["structure_mode"] = "Single metal surface"
    if "interface" in text or "/" in text:
        parsed["structure_mode"] = "Interface"

    cell_match = re.search(r"(\d+)\s*[x×\*]\s*(\d+)", text)
    if cell_match:
        parsed["nx"] = int(cell_match.group(1))
        parsed["ny"] = int(cell_match.group(2))
        parsed["coverage_preset"] = "custom nx × ny"

    coverage_text = text.replace(" ", "")
    coverage_aliases = {
        "1/9ml": "1/9 ML (3×3, default)",
        "1/16ml": "1/16 ML (4×4)",
        "1/4ml": "1/4 ML (2×2)",
        "1ml": "1 ML (1×1)",
    }
    for key, label in coverage_aliases.items():
        if key in coverage_text:
            parsed["coverage_preset"] = label
            nx, ny = COVERAGE_PRESETS[label]
            parsed["nx"] = nx
            parsed["ny"] = ny
            break

    ads_order = ["OCCO", "COCO", "CHOH", "CHO", "CO2", "CO", "OH", "CH3", "CH2", "CH", "H"]
    for ads in ads_order:
        if ads.lower() in text or f"*{ads.lower()}" in text:
            parsed["adsorbate"] = ads
            break

    if "au/cu" in text:
        parsed["element1"] = "Au"
        parsed["element2"] = "Cu"
    elif "cu/au" in text:
        parsed["element1"] = "Cu"
        parsed["element2"] = "Au"
    elif "cu" in text and "au" in text:
        parsed["element1"] = "Cu"
        parsed["element2"] = "Au"
    elif "cu" in text:
        parsed["element"] = "Cu"
    elif "au" in text:
        parsed["element"] = "Au"

    facet_match = re.search(r"\(?(\d{3})\)?", text)
    if facet_match:
        parsed["facet"] = facet_match.group(1)

    layer_match = re.search(r"(\d+)\s*layers?", text)
    if layer_match:
        parsed["layers"] = int(layer_match.group(1))

    vac_match = re.search(r"(\d+(\.\d+)?)\s*(a|å|angstrom)", text)
    if vac_match and "vacuum" in text:
        parsed["vacuum"] = float(vac_match.group(1))

    return parsed


def generate_structure(
    structure_mode,
    element,
    element1,
    element2,
    facet,
    nx,
    ny,
    layers,
    vacuum,
    adsorbate,
    height,
    output_dir,
    interface_match_mode="auto",
    element1_repeats=None,
    element2_repeats=None,
    interface_binding_element="Cu",
):
    """Generate structure through StructureGenerator only.

    The viewer uses exactly the returned atoms/positions. Adsorbate site defaults
    are resolved in StructureGenerator.add_adsorbate, so command-line generation
    and web visualization stay consistent.
    """
    sg = StructureGenerator()

    if structure_mode == "Single metal surface":
        base = sg.generate_surface(
            element=element,
            facet=facet,
            supercell=(int(nx), int(ny)),
            layers=int(layers),
            vacuum=float(vacuum),
        )
        substrate_n = len(base["atoms"])

        if adsorbate and adsorbate != "clean":
            structure = sg.add_adsorbate(
                base,
                adsorbate,
                site="auto",
                height=float(height),
                binding_element=element,
            )
        else:
            structure = base

        name = f"{element}_{facet}_{adsorbate}"

    else:
        kwargs = {}
        if interface_match_mode == "manual":
            kwargs["element1_repeats"] = int(element1_repeats)
            kwargs["element2_repeats"] = int(element2_repeats)

        base = sg.generate_interface(
            element1=element1,
            element2=element2,
            facet1=facet,
            facet2=facet,
            supercell=(int(nx), int(ny)),
            layers1=int(layers),
            layers2=int(layers),
            vacuum=float(vacuum),
            match_mode=interface_match_mode,
            **kwargs,
        )
        substrate_n = len(base["atoms"])

        if adsorbate and adsorbate != "clean":
            structure = sg.add_adsorbate(
                base,
                adsorbate,
                site="auto",
                height=float(height),
                binding_element=interface_binding_element,
            )
        else:
            structure = base

        name = f"{element1}_{element2}_{facet}_{adsorbate}"

    paths = sg.save_structure(structure, output_dir=output_dir, name=name)
    return structure, paths, substrate_n

def show_interface_strain(metadata):
    interface = metadata.get("interface", {}) if isinstance(metadata.get("interface"), dict) else {}
    match = interface.get("match", {}) if isinstance(interface.get("match"), dict) else {}
    if not match:
        return

    e1 = metadata.get("element1", "element1")
    e2 = metadata.get("element2", "element2")
    rows = interface.get("rows_per_side", "unknown")

    st.write("**Interface strain**")
    st.info(
        f"Interface direction is **x**. Auto matching minimizes strain along the interface: "
        f"{e1} repeats = {match.get('element1_repeats')}, {e2} repeats = {match.get('element2_repeats')}. "
        f"Rows away from the interface on each side = {rows}."
    )
    st.table({
        "quantity": [
            "match mode",
            "match direction",
            f"{e1} repeats along interface",
            f"{e2} repeats along interface",
            "rows per side, perpendicular to interface",
            f"{e1} natural interface length, Å",
            f"{e2} natural interface length, Å",
            "matched interface length, Å",
            f"{e1} strain along interface, %",
            f"{e2} strain along interface, %",
            "max absolute strain, %",
        ],
        "value": [
            match.get("match_mode", "unknown"),
            match.get("match_direction", "x/interface"),
            match.get("element1_repeats"),
            match.get("element2_repeats"),
            rows,
            f"{match.get('element1_natural_length', 0):.4f}",
            f"{match.get('element2_natural_length', 0):.4f}",
            f"{match.get('matched_block_length', 0):.4f}",
            f"{100 * match.get('element1_strain', 0):.3f}",
            f"{100 * match.get('element2_strain', 0):.3f}",
            f"{100 * match.get('max_abs_strain', 0):.3f}",
        ],
    })

    if match.get("max_abs_strain", 0) > 0.03:
        st.warning("Large interface strain > 3%. Good for debugging, but probably not good for production XAS.")
    elif match.get("max_abs_strain", 0) > 0.01:
        st.info("Moderate interface strain between 1% and 3%.")
    else:
        st.success("Low-strain interface match.")


def safe_index(options, value, default=0):
    return options.index(value) if value in options else default



def file_tree(root_dir):
    """Return a compact text tree for generated input files."""
    root = Path(root_dir)
    if not root.exists():
        return ""
    lines = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            lines.append(str(path.relative_to(root)))
    return "\n".join(lines)


def compact_directory_tree(root_dir, max_depth=4, max_entries=220):
    """Return a readable directory tree with submit scripts visible.

    This is shown in the NERSC section so the user can choose the correct
    relative submit path after uploading a batch folder, e.g.
    Cu_111_CO/01_structure/submit_relax.sh instead of 01_structure/submit_relax.sh.
    """
    root = Path(root_dir)
    if not root.exists() or not root.is_dir():
        return ""

    lines = [f"{root.name}/"]
    count = 0

    def walk(directory, prefix="", depth=0):
        nonlocal count
        if depth >= max_depth or count >= max_entries:
            return
        try:
            entries = sorted(
                [x for x in directory.iterdir() if not x.name.startswith(".")],
                key=lambda x: (not x.is_dir(), x.name.lower()),
            )
        except Exception:
            return
        for i, entry in enumerate(entries):
            if count >= max_entries:
                lines.append(f"{prefix}...")
                return
            connector = "└── " if i == len(entries) - 1 else "├── "
            lines.append(f"{prefix}{connector}{entry.name}{'/' if entry.is_dir() else ''}")
            count += 1
            if entry.is_dir():
                extension = "    " if i == len(entries) - 1 else "│   "
                walk(entry, prefix + extension, depth + 1)

    walk(root)
    return "\n".join(lines)


def discover_submit_scripts(local_root, max_scripts=300):
    """Return relative submit scripts under a local folder.

    For a batch package, this returns paths like:
      Cu_111_CO/01_structure/submit_relax.sh
      Cu_111_CO/02_XAS/VASP/submit.sh
    """
    root = Path(local_root)
    if not root.exists() or not root.is_dir():
        return []
    scripts = []
    for path in sorted(root.rglob("*.sh")):
        name = path.name
        if name in {"submit.sh", "submit_relax.sh"}:
            scripts.append(str(path.relative_to(root)))
    return scripts[:max_scripts]


def parse_remote_ls_files(ls_output):
    """Parse file names from remote ls/find output for preview dropdown."""
    names = []
    for line in str(ls_output or "").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("=="):
            continue
        # Handles lines from: find . -maxdepth ... -type f -printf '%P\n'
        candidate = stripped.split()[-1] if stripped.startswith("-") else stripped
        candidate = candidate.lstrip("./")
        if candidate and not candidate.endswith("/") and candidate not in names:
            names.append(candidate)
    preferred = [
        "vasp.out", "OUTCAR", "OSZICAR", "CONTCAR", "isaac_run_metadata.json",
        "isaac_record_draft.json", "xmu.dat", "fdmnes.out", "feff.out"
    ]
    return sorted(names, key=lambda x: (preferred.index(Path(x).name) if Path(x).name in preferred else 999, x))[:200]


def classify_upload_result(result):
    """Return (ok, label) for an upload result dictionary."""
    if not isinstance(result, dict):
        return False, "Upload result: Failed or unavailable"
    if result.get("status") not in {"uploaded", "success"}:
        return False, "Upload result: Failed"
    unpack = result.get("unpack_result")
    if isinstance(unpack, dict):
        exit_code = unpack.get("exit_code")
        unpack_status = str(unpack.get("status", "")).lower()
        if exit_code not in (None, 0, "0") or unpack_status in {"error", "failed", "failure"}:
            return False, "Upload result: Failed during remote unpack"
    return True, "Upload result: Success"


def zip_directory_bytes(root_dir):
    """Zip a generated input directory and return bytes for Streamlit download."""
    root = Path(root_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                zf.write(path, arcname=str(path.relative_to(root)))
    buffer.seek(0)
    return buffer.getvalue()


def sanitize_name(name):
    """Return a filesystem-safe short name for batch folders."""
    text = str(name or "structure").strip()
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", text)
    text = text.strip("._-")
    return text[:80] or "structure"


def slurm_safe_job_name(name, max_len=64):
    """Return a compact Slurm-safe job name with chemical/sample context."""
    text = sanitize_name(name).replace(".", "_")
    return text[:max_len] or "web_xas_job"


def sample_label_from_output_dir(output_dir, stage_folder):
    """Infer sample label from 01_structure/02_XAS folder path."""
    root = Path(output_dir)
    if root.name == stage_folder and root.parent.name:
        return slurm_safe_job_name(root.parent.name)
    return slurm_safe_job_name(root.name)


def sample_label_from_structure(structure, fallback="web_xas_agent"):
    """Build a chemically meaningful sample label for Slurm job names.

    This avoids generic names such as ``web_xas_agent_structure`` for single
    structures and keeps batch names such as ``Cu_111_clean`` unchanged when the
    folder already carries the sample label.
    """
    meta = dict((structure or {}).get("metadata", {}) or {})
    ads = meta.get("adsorbate") or "clean"
    if ads in {None, "", "None"}:
        ads = "clean"
    facet = meta.get("facet") or meta.get("facet1") or "111"
    if meta.get("element"):
        base = f"{meta.get('element')}_{facet}_{ads}"
    elif meta.get("element1") or meta.get("element2"):
        e1 = meta.get("element1", "M1")
        e2 = meta.get("element2", "M2")
        base = f"{e1}_{e2}_{facet}_{ads}"
    else:
        base = fallback
    return slurm_safe_job_name(base)


def remove_potcar_spec_files(root_dir):
    """Delete POTCAR.spec files because make_potcar.sh is the authoritative source."""
    root = Path(root_dir)
    for path in root.rglob("POTCAR.spec") if root.exists() else []:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


def relative_dirname(path_text):
    """Return POSIX dirname for a relative or absolute path string."""
    clean = str(path_text or "").strip().rstrip("/")
    if not clean:
        return ""
    parent = str(PurePosixPath(clean).parent)
    return "" if parent == "." else parent


def is_supported_structure_path(path):
    """Return True for common structure-file names used in this project."""
    p = Path(path)
    name = p.name.lower()
    suffix = p.suffix.lower()
    return (
        name in {"poscar", "contcar", "structure"}
        or name.startswith("poscar")
        or name.startswith("contcar")
        or suffix in {".vasp", ".poscar", ".contcar", ".cif", ".xyz"}
    )


def collect_uploaded_structure_files(uploaded_files, workspace):
    """Write uploaded files/ZIPs to a workspace and return readable structure paths."""
    workspace = Path(workspace)
    raw_dir = workspace / "uploaded_raw"
    extracted_dir = workspace / "uploaded_extracted"
    raw_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    structure_paths = []
    for uploaded in uploaded_files or []:
        original_name = Path(uploaded.name).name
        local_path = raw_dir / sanitize_name(original_name)
        local_path.write_bytes(uploaded.getbuffer())

        if original_name.lower().endswith(".zip"):
            with zipfile.ZipFile(local_path, "r") as zf:
                for member in zf.infolist():
                    if member.is_dir():
                        continue
                    member_path = Path(member.filename)
                    if not is_supported_structure_path(member_path):
                        continue
                    safe_parts = [sanitize_name(part) for part in member_path.parts if part not in {"", ".", ".."}]
                    if not safe_parts:
                        continue
                    out_path = extracted_dir.joinpath(*safe_parts)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member, "r") as src, out_path.open("wb") as dst:
                        dst.write(src.read())
                    structure_paths.append(out_path)
        elif is_supported_structure_path(original_name):
            structure_paths.append(local_path)

    # Preserve order but remove duplicate paths.
    deduped = []
    seen = set()
    for path in structure_paths:
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def relaxation_settings_from_params(params):
    """Collect VASP relaxation settings from session parameters."""
    return {
        "encut": params.get("relax_encut", 520),
        "ediff": params.get("relax_ediff", "1E-6"),
        "ibrion": params.get("relax_ibrion", 2),
        "nsw": params.get("relax_nsw", 200),
        "ediffg": params.get("relax_ediffg", "-0.02"),
        "isif": params.get("relax_isif", 2),
        "ispin": params.get("relax_ispin", 2),
        "ncore": params.get("relax_ncore", 8),
        "lreal": params.get("relax_lreal", "Auto"),
        "ismear": params.get("relax_ismear", 0),
        "sigma": params.get("relax_sigma", "0.05"),
    }


def choose_batch_absorber(structure, preferred=None):
    """Choose a reasonable absorber for batch generation when a file contains many elements."""
    present = list(dict.fromkeys(structure.get("atoms", [])))
    if preferred in present:
        return preferred
    for candidate in ["Cu", "Au", "Ni", "Pt", "Pd", "Ag", "Ir", "Rh", "C", "O"]:
        if candidate in present:
            return candidate
    non_h = [el for el in present if el != "H"]
    return non_h[0] if non_h else (present[0] if present else "Cu")


def build_pathway_batch_items(params, adsorbates, workspace):
    """Generate one structure per selected adsorbate from the current sidebar system."""
    items = []
    source_dir = Path(workspace) / "generated_pathway_structures"
    source_dir.mkdir(parents=True, exist_ok=True)
    for ads in adsorbates:
        structure, paths, substrate_n = generate_structure(
            structure_mode=params["structure_mode"],
            element=params["element"],
            element1=params["element1"],
            element2=params["element2"],
            facet=params["facet"],
            nx=int(params["nx"]),
            ny=int(params["ny"]),
            layers=int(params["layers"]),
            vacuum=float(params["vacuum"]),
            adsorbate=ads,
            height=float(params["height"]),
            output_dir=str(source_dir / sanitize_name(ads)),
            interface_match_mode=params["interface_match_mode"],
            element1_repeats=int(params["element1_repeats"]),
            element2_repeats=int(params["element2_repeats"]),
            interface_binding_element=params.get("interface_binding_element", params.get("element1", "Cu")),
        )
        meta = structure.get("metadata", {})
        system = meta.get("element") or f"{meta.get('element1', 'interface')}_{meta.get('element2', '')}".strip("_")
        name = sanitize_name(f"{system}_{meta.get('facet', params['facet'])}_{ads}")
        items.append({
            "name": name,
            "source": "generated_pathway",
            "structure": structure,
            "substrate_n": substrate_n,
            "source_poscar": paths.get("poscar"),
        })
    return items


def build_uploaded_batch_items(uploaded_files, workspace):
    """Read uploaded structure files and return batch items."""
    items = []
    for path in collect_uploaded_structure_files(uploaded_files, workspace):
        structure = read_structure_file(str(path))
        structure["metadata"] = dict(structure.get("metadata", {}))
        structure["metadata"].update({
            "type": "uploaded_batch_structure",
            "source_file": str(path),
            "source": "web_xas_agent_batch_upload",
        })
        items.append({
            "name": sanitize_name(path.stem if path.stem else path.name),
            "source": "uploaded_structure",
            "structure": structure,
            "substrate_n": len(structure.get("atoms", [])),
            "source_poscar": str(path),
        })
    return items


def generate_batch_input_package(
    batch_items,
    output_dir,
    params,
    include_relaxation=True,
    include_xas=True,
    absorber_mode="auto",
):
    """Generate batch folders with the production layout:

        sample/
          01_structure/
            POSCAR
            structure_info.json
            INCAR / KPOINTS / make_potcar.sh / submit_relax.sh
          02_XAS/
            VASP/            # contains only 01_scf, 02_xas, submit.sh, etc.
            FDMNES/
            FEFF/
    """
    if not batch_items:
        raise ValueError("No batch structures were found.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(output_dir) / f"batch_{timestamp}"
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    relaxation_settings = relaxation_settings_from_params(params)
    summary = []
    for item in batch_items:
        structure = item["structure"]
        name = sanitize_name(item["name"])
        sample_dir = run_dir / name
        structure_dir = sample_dir / "01_structure"
        xas_dir = sample_dir / "02_XAS"
        structure_dir.mkdir(parents=True, exist_ok=True)
        xas_dir.mkdir(parents=True, exist_ok=True)

        StructureGenerator().save_structure(structure, output_dir=str(structure_dir), name=name)

        absorber = choose_batch_absorber(
            structure,
            preferred=params.get("xas_absorber") if absorber_mode == "Use selected absorber if available" else None,
        )
        edge = normalize_edge_for_absorber(absorber, params.get("xas_edge", "K"))

        relax_result = None
        xas_result = None
        if include_relaxation:
            relax_result = generate_relaxation_folder(
                structure=structure,
                output_dir=str(structure_dir),
                relaxation_settings=relaxation_settings,
                potcar_dir=params.get("potcar_dir", DEFAULT_NERSC_POTCAR_DIR),
                account=params.get("nersc_account", DEFAULT_NERSC_ACCOUNT),
                queue=params.get("nersc_queue", "regular"),
                nodes=int(params.get("nersc_nodes", 1)),
                walltime=params.get("nersc_walltime", "05:00:00"),
                email=params.get("nersc_email", ""),
            )
        if include_xas:
            xas_result = generate_xas_calculation_folders(
                structure=structure,
                output_dir=str(xas_dir),
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
                walltime=params.get("nersc_walltime", "05:00:00"),
                email=params.get("nersc_email", ""),
            )

        summary.append({
            "name": name,
            "source": item.get("source"),
            "n_atoms": len(structure.get("atoms", [])),
            "absorber": absorber if include_xas else "not_generated",
            "edge": edge if include_xas else "not_generated",
            "structure_dir": str(structure_dir),
            "xas_dir": str(xas_dir),
            "relaxation": "generated" if relax_result else "skipped",
            "xas": "generated" if xas_result else "skipped",
        })

    manifest = {
        "status": "success",
        "n_structures": len(batch_items),
        "run_dir": str(run_dir),
        "layout": "sample/01_structure + sample/02_XAS/{VASP,FDMNES,FEFF}; relaxation files are directly in 01_structure",
        "include_relaxation": include_relaxation,
        "include_xas": include_xas,
        "nersc_account": params.get("nersc_account", DEFAULT_NERSC_ACCOUNT),
        "nersc_queue": params.get("nersc_queue", "regular"),
        "nersc_walltime": params.get("nersc_walltime", "05:00:00"),
        "summary": summary,
    }
    (run_dir / "batch_manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest

def normalize_edge_for_absorber(absorber, edge_choice):
    """Resolve web edge choice to the edge string used in FEFF/FDMNES/VASP input files."""
    edge_choice = edge_choice or "K"
    if edge_choice != "auto":
        return edge_choice
    atomic_number = {
        "H": 1, "C": 6, "N": 7, "O": 8, "Al": 13,
        "Ni": 28, "Cu": 29, "Zn": 30, "Rh": 45, "Pd": 46,
        "Ag": 47, "Ir": 77, "Pt": 78, "Au": 79,
    }.get(absorber, 0)
    return "K" if atomic_number <= 30 else "L3"



def patch_incar_value(lines, key, value):
    """Patch or append one VASP INCAR tag."""
    key_upper = key.upper()
    out = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith(f"{key_upper} ="):
            out.append(f"{key_upper} = {value}")
            replaced = True
        else:
            out.append(line)
    if not replaced:
        out.append(f"{key_upper} = {value}")
    return out


def patch_vasp_relaxation_incar(incar_path, settings):
    """Apply web-selected relaxation defaults to the generated VASP relaxation INCAR."""
    path = Path(incar_path)
    lines = path.read_text().splitlines()
    replacements = {
        "ENCUT": int(settings["encut"]),
        "EDIFF": settings["ediff"],
        "IBRION": int(settings["ibrion"]),
        "NSW": int(settings["nsw"]),
        "EDIFFG": settings["ediffg"],
        "ISIF": int(settings["isif"]),
        "ISPIN": int(settings["ispin"]),
        "NCORE": int(settings["ncore"]),
        "LREAL": settings["lreal"],
        "ISMEAR": int(settings["ismear"]),
        "SIGMA": settings["sigma"],
    }
    remove_tags = {"LWAVE", "LCHARG"}
    lines = [
        line for line in lines
        if not any(line.strip().upper().startswith(f"{tag} =") for tag in remove_tags)
    ]
    for key, value in replacements.items():
        lines = patch_incar_value(lines, key, value)
    path.write_text("\n".join(lines) + "\n")



def unique_elements_in_order(structure):
    """Return unique element order consistent with POSCAR/POTCAR order."""
    elems = []
    for atom in structure.get("atoms", []):
        if atom not in elems:
            elems.append(atom)
    return elems


def potcar_symbol_for_element(element):
    """Use the same default potential names as the VASP XAS generator."""
    return VASPXASInputGenerator.DEFAULT_POTCAR_MAP.get(element, element)


def write_make_potcar_script(structure, output_dir, potcar_dir=DEFAULT_NERSC_POTCAR_DIR):
    """Write make_potcar.sh that cats NERSC PBE POTCARs in POSCAR order."""
    output_dir = Path(output_dir)
    potcar_dir = (potcar_dir or DEFAULT_NERSC_POTCAR_DIR).strip() or DEFAULT_NERSC_POTCAR_DIR
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        "",
        "# Generated by CO2RR XAS Agent.",
        "# Concatenate POTCAR files in the same element order as POSCAR.",
        f"POTCAR_DIR=\"${{CO2RR_POTCAR_DIR:-{potcar_dir}}}\"",
        "",
        "if [ ! -d \"${POTCAR_DIR}\" ]; then",
        "  echo \"ERROR: POTCAR_DIR does not exist: ${POTCAR_DIR}\" >&2",
        "  exit 1",
        "fi",
        "rm -f POTCAR",
    ]
    for elem in unique_elements_in_order(structure):
        symbol = potcar_symbol_for_element(elem)
        lines.extend([
            f"if [ ! -f \"${{POTCAR_DIR}}/{symbol}/POTCAR\" ]; then",
            f"  echo \"ERROR: Missing POTCAR: ${{POTCAR_DIR}}/{symbol}/POTCAR\" >&2",
            "  exit 1",
            "fi",
            f"cat \"${{POTCAR_DIR}}/{symbol}/POTCAR\" >> POTCAR",
        ])
    lines.extend(["", "echo \"Wrote POTCAR using ${POTCAR_DIR}\""])
    path = output_dir / "make_potcar.sh"
    path.write_text("\n".join(lines) + "\n")
    os.chmod(path, 0o755)
    return str(path)



def patch_vasp_submit_for_nersc_vasp6_cpu_text(text):
    """Patch a generated VASP 6 Slurm script to follow NERSC's Perlmutter CPU layout.

    NERSC's VASP 6 CPU example uses a hybrid MPI/OpenMP layout: 64 MPI ranks
    per CPU node and 2 OpenMP threads per rank. The srun -c 4 setting is a
    Slurm logical-CPU spacing value, not four physical OpenMP cores. On one
    Perlmutter CPU node this maps to 64 MPI ranks x 2 OpenMP threads and uses
    the 128 physical CPU cores. FEFF/FDMNES scripts are not passed through this
    function because they may be serial/non-MPI builds.
    """
    # Remove old/conflicting task directives generated by earlier app versions.
    lines = text.splitlines()
    cleaned = []
    inserted = False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^#SBATCH\s+(--ntasks-per-node(?:=|\s)|--ntasks(?:=|\s)|-n\b|--cpus-per-task(?:=|\s))", stripped):
            continue
        cleaned.append(line)
        if not inserted and re.match(r"^#SBATCH\s+(-N\b|--nodes(?:=|\s))", stripped):
            cleaned.append(f"#SBATCH --ntasks-per-node={DEFAULT_VASP6_MPI_RANKS_PER_NODE}")
            cleaned.append(f"#SBATCH --cpus-per-task={DEFAULT_VASP6_CPUS_PER_TASK}")
            inserted = True
    if not inserted:
        # Insert after the last #SBATCH line if a nodes directive was absent.
        insert_at = 0
        for i, line in enumerate(cleaned):
            if line.strip().startswith("#SBATCH"):
                insert_at = i + 1
        cleaned[insert_at:insert_at] = [
            f"#SBATCH --ntasks-per-node={DEFAULT_VASP6_MPI_RANKS_PER_NODE}",
            f"#SBATCH --cpus-per-task={DEFAULT_VASP6_CPUS_PER_TASK}",
        ]
    text = "\n".join(cleaned).rstrip() + "\n"

    # Keep CPU-node constraint and standard Slurm output/error files explicit.
    if not re.search(r"^#SBATCH\s+(-C\s+cpu|--constraint(?:=|\s)cpu)", text, flags=re.M):
        text = re.sub(r"^(#SBATCH\s+-A\s+.*)$", r"\1\n#SBATCH -C cpu", text, count=1, flags=re.M)
    if not re.search(r"^#SBATCH\s+(-o|--output)(?:=|\s)", text, flags=re.M):
        text = re.sub(r"^(#SBATCH\s+-J\s+.*)$", r"\1\n#SBATCH -o %x-%j.out", text, count=1, flags=re.M)
    if not re.search(r"^#SBATCH\s+(-e|--error)(?:=|\s)", text, flags=re.M):
        text = re.sub(r"^(#SBATCH\s+-o\s+.*|#SBATCH\s+--output(?:=|\s).*)$", r"\1\n#SBATCH -e %x-%j.err", text, count=1, flags=re.M)

    # Force the NERSC-recommended VASP 6 hybrid MPI/OpenMP defaults for CPU jobs.
    text = re.sub(r"^export\s+OMP_NUM_THREADS=.*$", f"export OMP_NUM_THREADS={DEFAULT_VASP6_OMP_NUM_THREADS}", text, flags=re.M)
    text = re.sub(r"^export\s+OMP_PLACES=.*$", "export OMP_PLACES=threads", text, flags=re.M)
    text = re.sub(r"^export\s+OMP_PROC_BIND=.*$", "export OMP_PROC_BIND=spread", text, flags=re.M)
    omp_block = f"export OMP_NUM_THREADS={DEFAULT_VASP6_OMP_NUM_THREADS}\nexport OMP_PLACES=threads\nexport OMP_PROC_BIND=spread"
    if "export OMP_NUM_THREADS=" not in text:
        if re.search(r"^module\s+load\s+vasp.*$", text, flags=re.M):
            text = re.sub(r"^(module\s+load\s+vasp.*)$", r"\1\n" + omp_block, text, count=1, flags=re.M)
        else:
            text = text.replace("\n", "\n" + omp_block + "\n", 1)

    # Export version/layout metadata so isaac_run_metadata.json and
    # isaac_record_draft.json match the actual submitted script.
    metadata_exports = (
        f"export ISAAC_GENERATOR_VERSION='{WEB_APP_UPDATE_TAG}'\n"
        f"export ISAAC_SOFTWARE='VASP'\n"
        f"export ISAAC_CODE_VERSION='{DEFAULT_ISAAC_CODE_VERSION}'\n"
        f"export ISAAC_SOFTWARE_MODULE='{DEFAULT_VASP_MODULE}'\n"
        f"export ISAAC_ORGANIZATION='{DEFAULT_ISAAC_ORGANIZATION}'\n"
        f"export ISAAC_VASP_LAYOUT='{DEFAULT_VASP6_LAYOUT_NAME}'\n"
        "export ISAAC_RUN_COMMAND='srun -n ${SLURM_NTASKS:-64} -c 4 --cpu-bind=cores vasp_std'"
    )
    if "export ISAAC_GENERATOR_VERSION=" not in text:
        if re.search(r"^export\s+OMP_PROC_BIND=.*$", text, flags=re.M):
            text = re.sub(r"^(export\s+OMP_PROC_BIND=.*)$", r"\1\n" + metadata_exports, text, count=1, flags=re.M)
        elif re.search(r"^module\s+load\s+vasp.*$", text, flags=re.M):
            text = re.sub(r"^(module\s+load\s+vasp.*)$", r"\1\n" + metadata_exports, text, count=1, flags=re.M)
        else:
            text = metadata_exports + "\n" + text
    else:
        text = re.sub(r"^export\s+ISAAC_GENERATOR_VERSION=.*$", f"export ISAAC_GENERATOR_VERSION='{WEB_APP_UPDATE_TAG}'", text, flags=re.M)
        text = re.sub(r"^export\s+ISAAC_VASP_LAYOUT=.*$", f"export ISAAC_VASP_LAYOUT='{DEFAULT_VASP6_LAYOUT_NAME}'", text, flags=re.M)

    # Synchronize ISAAC defaults with the actual VASP module/layout in this script.
    isaac_sync = {
        "ISAAC_SOFTWARE": "VASP",
        "ISAAC_CODE_VERSION": DEFAULT_ISAAC_CODE_VERSION,
        "ISAAC_SOFTWARE_MODULE": DEFAULT_VASP_MODULE,
        "ISAAC_ORGANIZATION": DEFAULT_ISAAC_ORGANIZATION,
        "ISAAC_VENDOR_OR_PROJECT": "VASP",
        "ISAAC_INSTRUMENT_NAME": "VASP_Standard",
        "ISAAC_FACILITY_NAME": "NERSC",
        "ISAAC_RUN_COMMAND": "srun -n ${SLURM_NTASKS:-64} -c 4 --cpu-bind=cores vasp_std",
    }
    for key, value in isaac_sync.items():
        line = f"export {key}='{value}'"
        if re.search(rf"^export\s+{key}=.*$", text, flags=re.M):
            text = re.sub(rf"^export\s+{key}=.*$", line, text, flags=re.M)
        else:
            text = re.sub(r"^(export\s+ISAAC_GENERATOR_VERSION=.*)$", r"\1\n" + line, text, count=1, flags=re.M)

    # Replace any previous VASP srun command with the NERSC VASP 6 CPU command.
    # This handles both old one-rank scripts and v27/v28 idempotent rewrites.
    text = re.sub(
        r"srun\b(?:\s+(?:-[nc]\s+\S+|--cpu-bind=\S+|--cpu_bind=\S+|--ntasks(?:=|\s)\S+|--cpus-per-task(?:=|\s)\S+))*\s+vasp_std",
        DEFAULT_VASP_SRUN_COMMAND,
        text,
    )

    # Update ISAAC run-command metadata after the generic srun replacement so
    # nested shell quoting remains valid.
    text = re.sub(
        r"^export\s+ISAAC_RUN_COMMAND=.*vasp_std.*$",
        "export ISAAC_RUN_COMMAND='srun -n ${SLURM_NTASKS:-64} -c 4 --cpu-bind=cores vasp_std'",
        text,
        flags=re.M,
    )
    return text


def patch_submit_to_make_potcar(submit_path):
    """Ensure VASP submit scripts build POTCAR and use the NERSC VASP 6 CPU layout.

    The generated submit script is intentionally self-contained because only one
    shell script is submitted to Slurm/SF API. It calls make_potcar.sh from
    inside the job directory before launching VASP, then runs VASP 6 with the
    NERSC-recommended hybrid MPI/OpenMP Perlmutter CPU layout: 64 MPI ranks per
    node, 2 OpenMP threads per rank, and srun -c 4 for logical-CPU rank spacing.
    """
    path = Path(submit_path)
    if not path.exists():
        return
    text = path.read_text()

    tag = "# Build POTCAR before launching VASP"
    if tag not in text:
        potcar_block = """# Build POTCAR before launching VASP.
# The submit script is self-contained for SF API submission.
if [ ! -s POTCAR ]; then
  echo 'POTCAR not found or empty; running ./make_potcar.sh'
  ./make_potcar.sh || EXIT_CODE=$?
fi

if [ "${EXIT_CODE}" -eq 0 ]; then
  __VASP_SRUN_COMMAND__ > vasp.out 2>&1 || EXIT_CODE=$?
else
  echo 'Skipping VASP because POTCAR generation failed with exit code '${EXIT_CODE}
fi""".replace("__VASP_SRUN_COMMAND__", DEFAULT_VASP_SRUN_COMMAND)

        # Replace the single-line VASP run from NERSCScriptGenerator while preserving
        # the metadata setup/finalize sections around it.
        replaced = False
        old_patterns = [
            "srun vasp_std > vasp.out 2>&1 || EXIT_CODE=$?",
            "srun vasp_std > vasp.out 2>&1",
            "srun vasp_std > vasp.out",
        ]
        for old in old_patterns:
            if old in text:
                text = text.replace(old, potcar_block, 1)
                replaced = True
                break
        if not replaced:
            marker = "export ISAAC_RUN_COMMAND=\"srun vasp_std\"\n"
            if marker in text:
                text = text.replace(marker, marker + potcar_block + "\n", 1)
            else:
                text = text.rstrip() + "\n" + potcar_block + "\n"

    # Always apply the CPU layout patch, even to scripts generated by older
    # versions that already contain the POTCAR block.
    text = patch_vasp_submit_for_nersc_vasp6_cpu_text(text)
    path.write_text(text)

def isaac_record_inline_python_block():
    """Return a self-contained shell/Python block that writes isaac_record_draft.json.

    This is embedded directly in submit_relax.sh / submit.sh so each NERSC job
    writes a lightweight science-level ISAAC draft record in addition to the
    lower-level isaac_run_metadata.json produced by the existing metadata block.
    The block is intentionally dependency-free and recursively scans the job
    folder for FEFF, FDMNES, VASP, and relaxation outputs.
    """
    return """# =============================================================================
# ISAAC draft record: parse available outputs into isaac_record_draft.json
# =============================================================================
python - <<'PYEOF'
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path('.').resolve()

SPECTRUM_FILES = {
    'xmu.dat': 'FEFF',
    'CORE_DIELECTRIC_IMAG.dat': 'VASP',
}
INCLUDE_NAMES = {
    'POSCAR', 'CONTCAR', 'INCAR', 'KPOINTS', 'make_potcar.sh',
    'submit.sh', 'submit_relax.sh', 'feff.inp', 'HEADER', 'PARAMETERS',
    'POTENTIALS', 'ATOMS', 'fdmfile.txt', 'xmu.dat', 'chi.dat', 'paths.dat',
    'files.dat', 'log1.dat', 'OUTCAR', 'vasprun.xml', 'OSZICAR', 'vasp.out',
    'fdmnes.out', 'feff.out', 'CORE_DIELECTRIC_IMAG.dat',
    'isaac_run_metadata.json', 'isaac_record_draft.json'
}


def utc_now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def sha256_file(path):
    try:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return 'not_available'


def media_type(path):
    name = path.name.lower()
    if name.endswith('.json'):
        return 'application/json'
    if name.endswith('.xml'):
        return 'application/xml'
    return 'text/plain'


def content_role(path):
    name = path.name
    lower = name.lower()
    if name in {'POSCAR', 'CONTCAR'} or lower.endswith(('.cif', '.xyz')):
        return 'input_or_relaxed_structure'
    if name in {'INCAR', 'KPOINTS', 'make_potcar.sh', 'submit.sh', 'submit_relax.sh', 'feff.inp', 'HEADER', 'PARAMETERS', 'POTENTIALS', 'ATOMS', 'fdmfile.txt'} or name.endswith('_in.txt'):
        return 'workflow_recipe'
    if name in {'xmu.dat', 'chi.dat', 'paths.dat'} or name.endswith('_conv.txt') or name == 'CORE_DIELECTRIC_IMAG.dat':
        return 'reduction_product'
    if name in {'OUTCAR', 'vasprun.xml', 'OSZICAR', 'vasp.out', 'fdmnes.out', 'feff.out'}:
        return 'raw_simulation_output'
    return 'auxiliary_file'


def collect_files(root):
    files = []
    for path in sorted(root.rglob('*')):
        if not path.is_file():
            continue
        if path.name in INCLUDE_NAMES or path.name.endswith(('_in.txt', '_conv.txt', '_bav.txt')):
            files.append(path)
    return files


def parse_numeric_table(path, max_rows=None):
    rows = []
    try:
        for line in path.read_text(errors='ignore').splitlines():
            s = line.strip()
            if not s or s.startswith(('#', '*', '!', '%')):
                continue
            parts = s.replace(',', ' ').split()
            nums = []
            for part in parts:
                try:
                    value = float(part)
                    if math.isfinite(value):
                        nums.append(value)
                except Exception:
                    pass
            if len(nums) >= 2:
                rows.append(nums)
                if max_rows is not None and len(rows) >= max_rows:
                    break
    except Exception:
        return []
    return rows


def minmax(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-15:
        return [0.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def spectrum_from_file(path):
    rows = parse_numeric_table(path)
    if not rows:
        return None
    name = path.name
    software = 'unknown'
    columns = ['x', 'y']
    y_index = 1
    max_cols = max(len(r) for r in rows)
    if name == 'xmu.dat':
        software = 'FEFF'
        columns = ['energy_eV', 'k', 'mu', 'mu0', 'chi'][:max_cols]
        y_index = 2 if max_cols >= 3 else 1
    elif name.endswith('_conv.txt'):
        software = 'FDMNES'
        columns = ['energy_eV_or_relative_energy', 'intensity']
        y_index = 1
    elif name == 'CORE_DIELECTRIC_IMAG.dat':
        software = 'VASP'
        columns = ['energy_eV', 'epsilon2_or_xas_intensity']
        y_index = 1
    else:
        columns = [f'col_{i}' for i in range(max_cols)]
    x = [r[0] for r in rows if len(r) > y_index]
    y = [r[y_index] for r in rows if len(r) > y_index]
    if not x or not y:
        return None
    return {
        'source_file': str(path.relative_to(ROOT)),
        'software': software,
        'columns': columns,
        'x_label': columns[0] if columns else 'x',
        'y_label': columns[y_index] if y_index < len(columns) else f'col_{y_index}',
        'n_points': len(y),
        'x_min': min(x),
        'x_max': max(x),
        'y_min': min(y),
        'y_max': max(y),
        'normalization': 'minmax_y_for_preview_and_ml_ingestion',
        'preview': {'x': x[:25], 'y': y[:25], 'y_minmax': minmax(y)[:25]},
    }


def detect_software(files):
    names = {p.name for p in files}
    if 'xmu.dat' in names or 'feff.inp' in names:
        return 'FEFF'
    if any(p.name.endswith('_conv.txt') or p.name == 'fdmfile.txt' for p in files):
        return 'FDMNES'
    if 'CORE_DIELECTRIC_IMAG.dat' in names or 'OUTCAR' in names or 'vasprun.xml' in names or 'vasp.out' in names:
        return 'VASP'
    return os.environ.get('ISAAC_SOFTWARE', 'not_specified')


def load_run_metadata(root):
    path = root / 'isaac_run_metadata.json'
    if path.exists():
        try:
            return json.loads(path.read_text()), path
        except Exception:
            pass
    candidates = sorted(root.rglob('isaac_run_metadata.json'))
    for cand in candidates:
        try:
            return json.loads(cand.read_text()), cand
        except Exception:
            continue
    return {}, None

files = collect_files(ROOT)
base, meta_path = load_run_metadata(ROOT)
assets = []
for idx, path in enumerate(files):
    if path.name == 'isaac_record_draft.json':
        continue
    rel = path.relative_to(ROOT).as_posix()
    assets.append({
        'asset_id': f"asset_{idx:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', rel)[:80]}",
        'content_role': content_role(path),
        'uri': rel,
        'media_type': media_type(path),
        'sha256': sha256_file(path),
        'notes': f'Collected from {rel}.',
    })

spectra = []
for path in files:
    if path.name in SPECTRUM_FILES or path.name.endswith('_conv.txt'):
        spec = spectrum_from_file(path)
        if spec:
            spectra.append(spec)

software = detect_software(files)
record = {
    'record_type': 'ISAAC_simulation_record_draft',
    'schema_note': 'Draft CO2RR XAS Agent export; validate against the current ISAAC portal schema before bulk upload.',
    'generator': {
        'name': 'CO2RR XAS Agent web_xas_agent',
        'version': os.environ.get('ISAAC_GENERATOR_VERSION', 'v33_2026-07-01_slurm_any_job_status'),
        'generated_utc': utc_now(),
    },
    'source': {
        'working_directory': str(ROOT),
        'metadata_source': str(meta_path.relative_to(ROOT)) if meta_path else 'generated_from_output_files',
    },
    'sample': base.get('sample', {
        'material': {
            'name': os.environ.get('ISAAC_SAMPLE_NAME', 'not_specified'),
            'formula': os.environ.get('ISAAC_SAMPLE_FORMULA', 'not_specified'),
            'provenance': os.environ.get('ISAAC_MATERIAL_PROVENANCE', 'theoretical'),
            'notes': os.environ.get('ISAAC_SAMPLE_NOTES', 'not_specified'),
        },
        'sample_form': os.environ.get('ISAAC_SAMPLE_FORM', 'slab_model'),
    }),
    'context': base.get('context', {
        'environment': os.environ.get('ISAAC_CONTEXT_ENVIRONMENT', 'in_silico'),
        'temperature_K': float(os.environ.get('ISAAC_TEMPERATURE_K', '0') or 0),
        'notes': os.environ.get('ISAAC_CONTEXT_NOTES', 'Static 0 K electronic structure unless otherwise specified.'),
    }),
    'system': base.get('system', {
        'domain': 'computational',
        'technique': 'xas_simulation_or_dft_relaxation',
        'facility': {'facility_name': 'NERSC', 'cluster': 'Perlmutter'},
        'instrument': {'instrument_type': 'simulation_engine', 'instrument_name': software},
    }),
    'run': base.get('run', {
        'software': software,
        'scheduler': 'SLURM',
        'exit_code': os.environ.get('ISAAC_EXIT_CODE', 'not_parsed'),
        'working_directory': str(ROOT),
    }),
    'measurement': base.get('measurement', {
        'processing': {'type': 'simulated_spectrum'},
        'qc': {'status': 'valid' if os.environ.get('ISAAC_EXIT_CODE', '0') == '0' else 'failed'},
    }),
    'assets': assets,
    'descriptors': {
        'outputs': spectra,
        'file_count': len(files),
        'spectrum_count': len(spectra),
    },
    'links': base.get('links', []),
}
# Ensure the draft record carries the same command/layout/version that was
# embedded in the submitted *.sh script. These overwrite stale defaults from
# older isaac_run_metadata.json blocks when the job script has newer exports.
if isinstance(record.get('run'), dict):
    record['run']['run_command'] = os.environ.get('ISAAC_RUN_COMMAND', 'not_specified')
    record['run']['vasp_layout'] = os.environ.get('ISAAC_VASP_LAYOUT', 'not_specified')
    record['run']['generator_version'] = os.environ.get('ISAAC_GENERATOR_VERSION', record['generator']['version'])
    record['run']['code_version'] = os.environ.get('ISAAC_CODE_VERSION', 'VASP 6.4.3' if software == 'VASP' else 'not_specified')
    record['run']['software_module'] = os.environ.get('ISAAC_SOFTWARE_MODULE', 'vasp/6.4.3-cpu' if software == 'VASP' else 'not_specified')
if isinstance(record.get('system'), dict):
    record['system']['software_module'] = os.environ.get('ISAAC_SOFTWARE_MODULE', 'vasp/6.4.3-cpu' if software == 'VASP' else 'not_specified')
    record['system']['organization'] = os.environ.get('ISAAC_ORGANIZATION', 'ANL')
    if isinstance(record['system'].get('facility'), dict):
        record['system']['facility']['organization'] = os.environ.get('ISAAC_ORGANIZATION', 'ANL')
Path('isaac_record_draft.json').write_text(json.dumps(record, indent=2))
print('Wrote isaac_record_draft.json with', len(assets), 'assets and', len(spectra), 'spectra')
PYEOF
"""


def patch_submit_to_write_isaac_record(submit_path):
    """Insert the ISAAC draft-record converter into a generated submit script."""
    path = Path(submit_path)
    if not path.exists():
        return
    text = path.read_text()
    tag = "ISAAC draft record: parse available outputs into isaac_record_draft.json"
    if tag in text:
        return
    block = isaac_record_inline_python_block()
    marker = 'exit "${EXIT_CODE}"'
    if marker in text:
        text = text.replace(marker, block + "\n" + marker)
    else:
        text = text.rstrip() + "\n" + block + "\n"
    path.write_text(text)

def generate_relaxation_folder(
    structure,
    output_dir,
    relaxation_settings,
    potcar_dir=DEFAULT_NERSC_POTCAR_DIR,
    account=DEFAULT_NERSC_ACCOUNT,
    queue="regular",
    nodes=1,
    walltime="05:00:00",
    email="",
):
    """Generate VASP structure-relaxation inputs directly under 01_structure."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    # Keep the structure and relaxation inputs together directly under 01_structure.
    StructureGenerator().save_structure(structure, output_dir=str(root), name="structure")
    relax_dir = root
    email = email.strip() or None

    relax_gen = RelaxationInputGenerator()
    script_gen = NERSCScriptGenerator()

    relax_files = relax_gen.write_inputs(structure, str(relax_dir))
    incar_path = relax_dir / "INCAR"
    if incar_path.exists():
        patch_vasp_relaxation_incar(incar_path, relaxation_settings)

    make_potcar_path = write_make_potcar_script(structure, relax_dir, potcar_dir=potcar_dir)
    # POTCAR.spec is redundant: make_potcar.sh fully defines POTCAR construction.
    remove_potcar_spec_files(relax_dir)

    folder_label = sample_label_from_output_dir(relax_dir, "01_structure")
    sample_label = folder_label if folder_label not in {"web_xas_agent", "generated_outputs"} else sample_label_from_structure(structure, folder_label)
    relax_submit = script_gen.write_script(
        str(relax_dir),
        job_name=slurm_safe_job_name(f"{sample_label}_structure_relax"),
        software="VASP",
        filename="submit_relax.sh",
        account=account,
        queue=queue,
        nodes=int(nodes),
        walltime=walltime,
        email=email,
    )
    patch_submit_to_make_potcar(relax_submit)
    patch_submit_to_write_isaac_record(relax_submit)

    return {
        "status": "success",
        "directory": str(relax_dir),
        "relax_folder": str(relax_dir),
        "files": sorted(str(p) for p in Path(relax_dir).rglob("*") if p.is_file()),
        "submit": relax_submit,
        "make_potcar": make_potcar_path,
        "potcar_dir": potcar_dir or DEFAULT_NERSC_POTCAR_DIR,
        "settings": relaxation_settings,
    }

def generate_xas_calculation_folders(
    structure,
    output_dir,
    absorber,
    edge="K",
    vasp_method="PBE",
    cluster_radius=6.0,
    fdmnes_method="Green",
    fdmnes_scf=False,
    fdmnes_quadrupole=False,
    fdmnes_spinorbit=False,
    fdmnes_energy_min=-5.0,
    fdmnes_energy_step=0.2,
    fdmnes_energy_max=50.0,
    potcar_dir=DEFAULT_NERSC_POTCAR_DIR,
    account=DEFAULT_NERSC_ACCOUNT,
    queue="regular",
    nodes=1,
    walltime="05:00:00",
    email="",
):
    """Generate XAS inputs into 02_XAS/{VASP,FDMNES,FEFF}.

    The VASP folder contains only the two-step XAS workflow generated by
    VASPXASInputGenerator: 01_scf/, 02_xas/, and submit.sh. Structure relaxation
    is generated separately under 01_structure.
    """
    root = Path(output_dir)
    vasp_root = root / "VASP"
    fdmnes_root = root / "FDMNES"
    feff_root = root / "FEFF"
    edge_key = f"{absorber}_{edge}_edge"

    potcar_dir = (potcar_dir or DEFAULT_NERSC_POTCAR_DIR).strip() or DEFAULT_NERSC_POTCAR_DIR
    email = email.strip() or None
    folder_label = sample_label_from_output_dir(root, "02_XAS")
    sample_label = folder_label if folder_label not in {"web_xas_agent", "generated_outputs"} else sample_label_from_structure(structure, folder_label)

    vasp_gen = VASPXASInputGenerator()
    fdmnes_gen = FDMNESInputGenerator()
    feff_gen = FEFFInputGenerator()
    script_gen = NERSCScriptGenerator()

    vasp_warning = None
    if edge != "K":
        vasp_warning = (
            f"Current VASP XAS generator is K-edge/core-hole oriented; "
            f"requested edge={edge}. VASP files are still generated, but check INCAR tags manually."
        )
    vasp_result = vasp_gen.write_inputs(
        structure=structure,
        output_dir=str(vasp_root),
        absorber=absorber,
        method=vasp_method,
        potcar_dir=potcar_dir,
        account=account,
        queue=queue,
        nodes=int(nodes),
        walltime=walltime,
        email=email,
        absorber_index=0,
        job_name=slurm_safe_job_name(f"{sample_label}_XAS_VASP"),
    )
    remove_potcar_spec_files(vasp_root)
    if isinstance(vasp_result, dict) and vasp_result.get("submit"):
        patch_submit_to_make_potcar(vasp_result["submit"])
        patch_submit_to_write_isaac_record(vasp_result["submit"])

    fdmnes_options = dict(FDMNESInputGenerator.DEFAULT_OPTIONS)
    fdmnes_options.update({
        "Green": fdmnes_method.lower().startswith("green"),
        "SCF": bool(fdmnes_scf),
        "Quadrupole": bool(fdmnes_quadrupole),
        "Spinorbit": bool(fdmnes_spinorbit),
    })
    fdmnes_files = fdmnes_gen.write_input(
        structure=structure,
        output_dir=str(fdmnes_root),
        absorber=absorber,
        edge=edge,
        radius=float(cluster_radius),
        energy_range=(float(fdmnes_energy_min), float(fdmnes_energy_step), float(fdmnes_energy_max)),
        options=fdmnes_options,
    )
    fdmnes_submit = script_gen.write_script(
        str(fdmnes_root),
        job_name=slurm_safe_job_name(f"{sample_label}_XAS_FDMNES"),
        software="FDMNES",
        account=account,
        queue=queue,
        nodes=int(nodes),
        walltime=walltime,
        email=email,
    )
    patch_submit_to_write_isaac_record(fdmnes_submit)

    feff_input = feff_gen.write_input(
        structure=structure,
        output_dir=str(feff_root),
        absorber=absorber,
        edge=edge,
        radius=float(cluster_radius),
    )
    feff_submit = script_gen.write_script(
        str(feff_root),
        job_name=slurm_safe_job_name(f"{sample_label}_XAS_FEFF"),
        software="FEFF",
        account=account,
        queue=queue,
        nodes=int(nodes),
        walltime=walltime,
        email=email,
    )
    patch_submit_to_write_isaac_record(feff_submit)

    result = {
        "status": "success",
        "root_dir": str(root),
        "layout": "02_XAS/{VASP,FDMNES,FEFF}",
        "edge_key": edge_key,
        "xas": {
            "VASP": {
                "directory": str(vasp_root),
                "result": vasp_result,
                "warning": vasp_warning,
            },
            "FDMNES": {
                "directory": str(fdmnes_root),
                "files": fdmnes_files,
                "submit": fdmnes_submit,
                "options": fdmnes_options,
            },
            "FEFF": {
                "directory": str(feff_root),
                "input": feff_input,
                "submit": feff_submit,
            },
        },
        "warnings": [
            "XAS inputs are generated from the current structure. Relax the structure first if it hasn't been relaxed."
        ],
    }
    if vasp_warning:
        result["warnings"].append(vasp_warning)
    return result


# =============================================================================
# Chat command console helpers
# =============================================================================

def _chat_add(role, message):
    """Append a persistent chat message to the Streamlit chat history."""
    st.session_state.setdefault("chat_history", [])
    st.session_state["chat_history"].append({
        "role": role,
        "message": str(message),
        "time": datetime.now().isoformat(timespec="seconds"),
    })


def _chat_client_from_params(params):
    if not NERSC_JOB_CONTROL_AVAILABLE:
        raise RuntimeError(f"NERSC job-control module is not available: {NERSC_JOB_CONTROL_IMPORT_ERROR}")
    return NERSCSFAPIClient(
        client_id=params.get("sfapi_client_id") or None,
        private_key_path=params.get("sfapi_private_key_path") or None,
        system=params.get("nersc_system", "perlmutter"),
    )



def _workflow_state_file(params=None):
    """Local JSON file used to persist full NERSC workflow state across Streamlit restarts."""
    params = params or st.session_state.get("params", {})
    root = Path(params.get("xas_output_dir", "generated_outputs/web_xas_agent"))
    return root / ".nersc_full_workflow_state.json"


def _save_full_workflow_state(state, params=None):
    """Persist full workflow state to disk so the workflow can continue after app restart."""
    if not isinstance(state, dict) or not state:
        return
    state_path = _workflow_state_file(params)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["saved_utc"] = datetime.utcnow().isoformat(timespec="seconds") + "Z"
    state_path.write_text(json.dumps(payload, indent=2))


def _load_full_workflow_state(params=None):
    """Load persisted full workflow state, if present."""
    state_path = _workflow_state_file(params)
    if not state_path.exists():
        return None
    try:
        state = json.loads(state_path.read_text())
        if isinstance(state, dict) and state.get("stage"):
            return state
    except Exception:
        return None
    return None

def _selected_local_folder_from_params(params):
    """Mirror the NERSC upload folder selection used by the UI."""
    source = params.get("nersc_upload_source", "Full package root")
    if source == "Last batch package" and st.session_state.get("last_batch_result"):
        return st.session_state["last_batch_result"].get("run_dir")
    if source == "VASP relaxation folder" and st.session_state.get("last_relax_result"):
        return st.session_state["last_relax_result"].get("relax_folder")
    if source == "Last XAS input root" and st.session_state.get("last_xas_dir"):
        return st.session_state.get("last_xas_dir")
    if source == "Custom local folder":
        return params.get("nersc_custom_local_dir", "")
    # Full package root is the default.
    return params.get("xas_output_dir") or params.get("output_dir") or "generated_outputs/web_xas_agent"


def _prefer_last_batch_for_chat_upload(params, text=""):
    """Return True when chat upload should use the last batch package.

    The decision must follow the *last generated package type*, not merely the
    existence of an older batch result. If the user generated a single structure
    after generating a batch, a plain "upload to NERSC under test03" should upload
    the single-structure package. If the last generated package was a batch, the
    same plain upload command should upload the batch folder.
    """
    if not st.session_state.get("last_batch_result"):
        return False

    low = str(text or "").lower()
    source = params.get("nersc_upload_source")
    package_kind = st.session_state.get("last_generated_package_kind")

    if any(term in low for term in ["single", "current structure", "full package root", "web_xas_agent root"]):
        return False
    if "workflow" in low and not any(term in low for term in ["batch", "pathway", "full co2rr", "reaction pathway"]):
        return False
    if any(term in low for term in ["batch", "pathway", "full co2rr", "reaction pathway"]):
        return True
    if source == "Last batch package" and package_kind != "single":
        return True

    return package_kind == "batch"


def _resolve_remote_submit_from_params(params):
    remote_submit = str(params.get("nersc_remote_submit_script", "submit.sh") or "submit.sh")
    if not remote_submit.startswith("/") and not remote_submit.startswith("$"):
        remote_submit = str(params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)).rstrip("/") + "/" + remote_submit
    return remote_submit


def _generate_structure_from_params(params, uploaded_file=None):
    """Generate a structure from current sidebar settings and store it in session state."""
    if params.get("structure_source") == "Upload structure":
        if uploaded_file is None:
            raise ValueError("Structure source is 'Upload structure', but no uploaded structure file is currently available.")
        output_dir = Path(params["output_dir"])
        upload_dir = output_dir / "uploaded_structure"
        upload_dir.mkdir(parents=True, exist_ok=True)
        upload_path = upload_dir / uploaded_file.name
        upload_path.write_bytes(uploaded_file.getbuffer())
        structure = read_structure_file(str(upload_path))
        structure["metadata"] = dict(structure.get("metadata", {}))
        structure["metadata"].update({
            "type": "uploaded_structure",
            "source_file": str(upload_path),
            "source": "web_xas_agent_upload",
        })
        substrate_n = len(structure["atoms"])
        paths = StructureGenerator().save_structure(structure, output_dir=params["output_dir"], name="uploaded_structure")
    else:
        structure, paths, substrate_n = generate_structure(
            structure_mode=params["structure_mode"],
            element=params["element"],
            element1=params["element1"],
            element2=params["element2"],
            facet=params["facet"],
            nx=int(params["nx"]),
            ny=int(params["ny"]),
            layers=int(params["layers"]),
            vacuum=float(params["vacuum"]),
            adsorbate=params["adsorbate"],
            height=float(params["height"]),
            output_dir=params["output_dir"],
            interface_match_mode=params["interface_match_mode"],
            element1_repeats=int(params["element1_repeats"]),
            element2_repeats=int(params["element2_repeats"]),
            interface_binding_element=params.get("interface_binding_element", params["element1"]),
        )
    st.session_state["last_structure"] = structure
    st.session_state["last_paths"] = paths
    st.session_state["last_substrate_n"] = substrate_n
    st.session_state["last_structure_error"] = None
    st.session_state["last_generated_package_kind"] = "single"
    st.session_state.pop("last_relax_result", None)
    st.session_state.pop("last_xas_result", None)
    return structure, paths, substrate_n


def _generate_relaxation_from_current_structure(params):
    if st.session_state.get("last_structure") is None:
        raise ValueError("No current structure. Generate a structure first.")
    structure = st.session_state["last_structure"]
    relaxation_settings = relaxation_settings_from_params(params)
    relax_result = generate_relaxation_folder(
        structure=structure,
        output_dir=str(Path(params["xas_output_dir"]) / "01_structure"),
        relaxation_settings=relaxation_settings,
        potcar_dir=params.get("potcar_dir", DEFAULT_NERSC_POTCAR_DIR),
        account=params["nersc_account"],
        queue=params["nersc_queue"],
        nodes=params["nersc_nodes"],
        walltime=params["nersc_walltime"],
        email=params["nersc_email"],
    )
    st.session_state["last_relax_result"] = relax_result
    st.session_state["last_xas_dir"] = params["xas_output_dir"]
    return relax_result



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
        walltime=params.get("nersc_walltime", "05:00:00"),
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
    run_name = PurePosixPath(str(remote_dir).rstrip("/")).name or "latest"
    local_base = Path(params.get("output_dir", "generated_outputs/web_xas_agent")).parent / "web_xas_agent_runs"
    params["xas_output_dir"] = str(local_base / run_name)
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

def _generate_xas_from_current_structure(params):
    if st.session_state.get("last_structure") is None:
        raise ValueError("No current structure. Generate a structure first.")
    structure = st.session_state["last_structure"]
    atoms = list(dict.fromkeys(structure.get("atoms", [])))
    absorber_options = [el for el in atoms if el not in {"H"}] or atoms
    if not absorber_options:
        raise ValueError("No absorber elements found in the current structure.")
    if params.get("xas_absorber") not in absorber_options:
        params["xas_absorber"] = "Cu" if "Cu" in absorber_options else absorber_options[0]
    resolved_edge = normalize_edge_for_absorber(params["xas_absorber"], params.get("xas_edge", "K"))
    xas_dir = Path(params["xas_output_dir"]) / "02_XAS"
    xas_result = generate_xas_calculation_folders(
        structure=structure,
        output_dir=xas_dir,
        absorber=params["xas_absorber"],
        edge=resolved_edge,
        vasp_method=params["vasp_method"],
        cluster_radius=params["cluster_radius"],
        fdmnes_method=params["fdmnes_method"],
        fdmnes_scf=params["fdmnes_scf"],
        fdmnes_quadrupole=params["fdmnes_quadrupole"],
        fdmnes_spinorbit=params["fdmnes_spinorbit"],
        fdmnes_energy_min=params["fdmnes_energy_min"],
        fdmnes_energy_step=params["fdmnes_energy_step"],
        fdmnes_energy_max=params["fdmnes_energy_max"],
        potcar_dir=params["potcar_dir"],
        account=params["nersc_account"],
        queue=params["nersc_queue"],
        nodes=params["nersc_nodes"],
        walltime=params["nersc_walltime"],
        email=params["nersc_email"],
    )
    st.session_state["last_xas_result"] = xas_result
    st.session_state["last_xas_dir"] = str(xas_dir)
    return xas_result, xas_dir, resolved_edge


def _generate_batch_from_current_settings(params):
    if params.get("batch_source") != "Generate pathway from current structure setup":
        raise ValueError("Chat batch generation currently supports 'Generate pathway from sidebar system'. Use the Batch panel for uploaded ZIP/files.")
    workspace = Path(params.get("batch_output_dir", "generated_outputs/web_xas_agent/batch")) / "_chat_workspace"
    adsorbates = [ads for ads in params.get("batch_adsorbates", []) if ads in ADSORBATE_CHOICES]
    if not adsorbates:
        raise ValueError("No batch adsorbates selected.")
    batch_items = build_pathway_batch_items(params, adsorbates, workspace)
    batch_result = generate_batch_input_package(
        batch_items=batch_items,
        output_dir=params["batch_output_dir"],
        params=params,
        include_relaxation=params.get("batch_include_relaxation", True),
        include_xas=params.get("batch_include_xas", True),
        absorber_mode=params.get("batch_absorber_mode", "auto"),
    )
    st.session_state["last_batch_result"] = batch_result
    st.session_state["last_generated_package_kind"] = "batch"
    return batch_result



def _pathway_adsorbates_from_text(text):
    """Return the default CO2RR pathway set, with optional text filtering."""
    default_pathway = [ads for ads in ["clean", "CO2", "CO", "CHO", "OCCO", "COCO", "OH", "H"] if ads in ADSORBATE_CHOICES]
    low = str(text or "").lower()
    if "full" in low or "pathway" in low or "reaction" in low:
        return default_pathway
    # Fall back to explicit adsorbates mentioned in text, without accidentally
    # matching H inside words like "what" or "path".
    found = []
    for ads in ["OCCO", "COCO", "CHOH", "CHO", "CO2", "CO", "OH", "CH3", "CH2", "CH", "H"]:
        pat = r"(?<![A-Za-z0-9])\*?" + re.escape(ads.lower()) + r"(?![A-Za-z0-9])"
        if re.search(pat, low) and ads in ADSORBATE_CHOICES and ads not in found:
            found.append(ads)
    return found or default_pathway


def _remote_dir_from_natural_language(text, params):
    """Extract a friendly run name such as test03 and map it to the NERSC run path.

    This should only treat a token as a run name when the user explicitly gives
    one, such as "under test03". It must not parse phrases like "folder files"
    as a run named "files".
    """
    raw = str(text or "").strip()
    stop_names = {
        "", "none", "latest", "nersc", "perlmutter", "file", "files",
        "folder", "folders", "dir", "directory", "there", "that", "this",
        "current", "remote", "job", "jobs", "surface", "clean", "relaxation",
    }

    abs_match = re.search(r"(/pscratch/[^\s`\'\"]+|/global/[^\s`\'\"]+)", raw)
    if abs_match:
        return abs_match.group(1).rstrip(".,;:)>]")

    # Prefer explicit run-name phrases after "under" because commands like
    # "upload to NERSC under test01" contain both "to NERSC" and "under test01".
    for pattern in [
        r"(?:under|run(?: name)?|named)\s+([A-Za-z0-9_.-]+)",
        r"(?:remote\s+)?(?:folder|dir|directory)\s+(?:named\s+)?([A-Za-z0-9_.-]+)",
    ]:
        match = re.search(pattern, raw, flags=re.I)
        if match:
            name = match.group(1).strip().strip(".,;:)")
            if name.lower() not in stop_names and "/" not in name:
                return f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/{name}"

    # Only parse "to NAME" as a run name for upload/copy commands.
    if re.search(r"\b(upload|copy|send|move)\b", raw, flags=re.I):
        to_match = re.search(r"\bto\s+([A-Za-z0-9_.-]+)", raw, flags=re.I)
        if to_match:
            name = to_match.group(1).strip().strip(".,;:)")
            if name.lower() not in stop_names and "/" not in name:
                return f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/{name}"

    # Planner may return only {"remote_run_name": "test03"}.
    if re.fullmatch(r"[A-Za-z0-9_.-]+", raw) and raw.lower() not in stop_names:
        return f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs/{raw}"

    return params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)


def _remote_dir_explicitly_mentioned(text, params):
    """Return an absolute NERSC path or named run folder explicitly present in a message."""
    raw = str(text or "")
    abs_match = re.search(r"(/pscratch/[^\s`'\"]+|/global/[^\s`'\"]+)", raw)
    if abs_match:
        return abs_match.group(1).rstrip(".,;:)>]")
    # Only map named run folders when the user explicitly provides a run name.
    # Do NOT treat phrases like "folder files" as a run folder called "files".
    if re.search(r"\b(under|run name|named)\b", raw, flags=re.I):
        return _remote_dir_from_natural_language(raw, params)
    if re.search(r"\b(?:folder|dir|directory)\s+named\s+[A-Za-z0-9_.-]+", raw, flags=re.I):
        candidate = _remote_dir_from_natural_language(raw, params)
        if candidate != params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR):
            return candidate
    return None


def _best_known_remote_root(params):
    """Best current remote run root for list/preview commands."""
    last_upload = st.session_state.get("nersc_last_upload") or {}
    for key in [
        None,
        "nersc_preview_dir",
        "nersc_last_submit_dir",
        "nersc_download_remote_dir",
        "nersc_remote_dir",
    ]:
        value = last_upload.get("remote_dir") if key is None else params.get(key)
        if value:
            return str(value).rstrip("/")
    return DEFAULT_NERSC_REMOTE_RUN_DIR


def _adsorbate_from_text_for_remote(text):
    low = str(text or "").lower()
    if re.search(r"\bclean(?:\s+surface)?\b", low):
        return "clean"
    for ads in ["OCCO", "COCO", "CHO", "CO2", "CO", "OH", "H"]:
        if re.search(rf"\b{re.escape(ads.lower())}\b", low):
            return ads
    return None


def _extract_slurm_job_id_from_text(text):
    """Return an explicit Slurm job id mentioned in a chat message.

    This lets the chat handle jobs that were submitted outside this app, e.g.
    "check slurm job 55362912", instead of always checking the last app-submitted job.
    """
    raw = str(text or "")
    # Prefer numbers near job/slurm/squeue wording.
    targeted = re.search(r"(?:slurm|squeue|sacct|job(?:\s+id)?|jobid)\D{0,30}([0-9]{5,12})(?:\.[0-9]+)?\b", raw, flags=re.I)
    if targeted:
        return targeted.group(1)
    # If the whole request is a status/check request and contains one standalone
    # long number, use it as the job id. Avoid treating dates like 20260701 as
    # job ids unless job/status wording is present.
    if re.search(r"\b(check|status|slurm|squeue|sacct|job)\b", raw, flags=re.I):
        ids = re.findall(r"(?<![A-Za-z0-9])([0-9]{5,12})(?:\.[0-9]+)?(?![A-Za-z0-9])", raw)
        if ids:
            return ids[0]
    return ""


def _chat_requested_job_listing(text):
    """True for natural-language requests asking about all current jobs."""
    low = str(text or "").lower().strip()
    return bool(
        re.search(r"\b(any|my|current|running|queued|recent)\b.*\bjobs?\b", low)
        or re.search(r"\bjobs?\b.*\b(running|queued|current|recent)\b", low)
        or re.search(r"\b(do i|have i|am i)\b.*\bjobs?\b", low)
        or "squeue" in low
        or "list jobs" in low
        or "list my jobs" in low
    )


def _remote_dir_for_listing_request(text, params, plan=None):
    """Resolve the remote folder to list from natural language.

    Handles phrases like "show clean surface folder files" by mapping to the
    clean sample inside the last uploaded batch when present, instead of
    accidentally making a run folder named "files".
    """
    if isinstance(plan, dict) and plan.get("remote_dir"):
        return str(plan.get("remote_dir")).rstrip("/")
    explicit = _remote_dir_explicitly_mentioned(text, params)
    if explicit:
        return explicit.rstrip("/")

    root = _best_known_remote_root(params)
    ads = _adsorbate_from_text_for_remote(text)
    if ads:
        low = str(text or "").lower()
        package_kind = st.session_state.get("last_generated_package_kind")

        # Single-structure runs do not have Cu_111_clean/Cu_111_CO...
        # subfolders. For phrases like "show clean surface folder files", list
        # the current single-structure remote root instead of inventing
        # root/Cu_111_clean.
        if package_kind != "batch" and ads == "clean":
            if any(k in low for k in ["relax", "relaxation", "structure"]):
                return root.rstrip("/") + "/01_structure"
            return root.rstrip("/")

        element = params.get("element") or params.get("element1") or "Cu"
        sample_dir = f"{element}_111_{ads}"
        # If the current remote root already points into a sample, do not append
        # another sample directory.
        if re.search(r"/[A-Za-z]{1,2}_111_(clean|CO2|CO|CHO|OCCO|COCO|OH|H)(/|$)", root):
            sample_root = root
        else:
            sample_root = root.rstrip("/") + "/" + sample_dir
        if any(k in low for k in ["relax", "relaxation", "structure"]):
            return sample_root.rstrip("/") + "/01_structure"
        if "vasp" in low:
            return sample_root.rstrip("/") + "/02_XAS/VASP"
        if "fdmnes" in low:
            return sample_root.rstrip("/") + "/02_XAS/FDMNES"
        if "feff" in low:
            return sample_root.rstrip("/") + "/02_XAS/FEFF"
        return sample_root.rstrip("/")

    return root.rstrip("/")


def _safe_remote_delete_target(remote_dir):
    """Validate remote delete target before calling rm -rf on NERSC."""
    remote_dir = str(remote_dir or "").rstrip("/")
    allowed_root = f"{DEFAULT_NERSC_REMOTE_BASE}/web_xas_agent_runs"
    if not remote_dir:
        raise ValueError("No remote folder was provided for deletion.")
    if remote_dir in {DEFAULT_NERSC_REMOTE_BASE, allowed_root, "/", "/pscratch", "/pscratch/sd", "/pscratch/sd/h", "/pscratch/sd/h/hjia"}:
        raise ValueError(f"Refusing to delete broad/root directory: {remote_dir}")
    if not remote_dir.startswith(allowed_root + "/"):
        raise ValueError(
            "For safety, chat deletion is only allowed under "
            f"{allowed_root}/... . Requested: {remote_dir}"
        )
    if any(ch in remote_dir for ch in ["*", "?", "[", "]", "{", "}", ";", "&", "|", "$", "`", "\n"]):
        raise ValueError(f"Unsafe characters in remote path: {remote_dir}")
    return remote_dir


def _describe_current_uploaded_paths(params):
    """Return the best-known local/remote upload paths from session state."""
    lines = []
    last_upload = st.session_state.get("nersc_last_upload")
    if last_upload:
        lines.append(f"Last upload status: `{last_upload.get('status', 'unknown')}`")
        if last_upload.get("local_dir"):
            lines.append(f"Local uploaded folder: `{last_upload.get('local_dir')}`")
        if last_upload.get("remote_dir"):
            lines.append(f"Remote uploaded folder: `{last_upload.get('remote_dir')}`")
    lines.append(f"Last generated package type: `{st.session_state.get('last_generated_package_kind', 'not set')}`")
    if st.session_state.get("last_batch_result"):
        lines.append(f"Last batch package: `{st.session_state['last_batch_result'].get('run_dir')}`")
    if params.get("nersc_last_submit_dir"):
        lines.append(f"Last submitted remote folder: `{params.get('nersc_last_submit_dir')}`")
    if params.get("nersc_remote_submit_script"):
        lines.append(f"Selected submit script: `{params.get('nersc_remote_submit_script')}`")
    if params.get("nersc_remote_dir"):
        lines.append(f"Configured remote run directory: `{params.get('nersc_remote_dir')}`")
    return "\n".join(lines) if lines else "No upload/submission path is recorded yet."


def _safe_parse_settings_from_text(text, allow_adsorbate=True):
    """Parse only chemistry/settings text, with less accidental adsorbate matching."""
    parsed = parse_prompt(text)
    if not allow_adsorbate and "adsorbate" in parsed:
        parsed.pop("adsorbate", None)
    # Avoid bad H matching from ordinary English. parse_prompt currently matches
    # H as a substring; require standalone H/*H for hydrogen.
    if parsed.get("adsorbate") == "H":
        if not re.search(r"(?<![A-Za-z0-9])\*?h(?![A-Za-z0-9])", str(text or "").lower()):
            parsed.pop("adsorbate", None)
    return parsed


def _extract_json_object(text):
    """Extract a JSON object from a model response that should be JSON-only."""
    raw = str(text or "").strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I).strip()
        raw = re.sub(r"\s*```$", "", raw).strip()
    try:
        return json.loads(raw)
    except Exception:
        match = re.search(r"\{.*\}", raw, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))



def _get_alcf_access_token():
    """Return an ALCF inference access token from env or inference_auth_token.py."""
    token = os.environ.get("ALCF_INFERENCE_ACCESS_TOKEN", "") or ALCF_INFERENCE_ACCESS_TOKEN
    if token:
        return token
    try:
        from inference_auth_token import get_access_token
    except Exception as exc:
        raise RuntimeError(
            "Could not import inference_auth_token. Download it with: "
            "curl -L -o inference_auth_token.py "
            "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py "
            "and run: python inference_auth_token.py authenticate"
        ) from exc
    token = get_access_token()
    if not token:
        raise RuntimeError("ALCF inference token is empty. Run: python inference_auth_token.py authenticate")
    return token


def _planner_system_prompt():
    """Shared planner prompt for all optional LLM providers."""
    return (
        "You are a command planner for a CO2RR XAS Streamlit app. "
        "Return only one JSON object. Do not execute anything. Valid intents: "
        "generate_structure, generate_structure_and_relaxation, generate_relaxation, generate_xas, generate_batch_pathway, "
        "upload_nersc, generate_relaxation_and_upload, submit, check_task, check_slurm, "
        "list_jobs, list_remote_folder, delete_remote_folder, preview_file, download_outputs, convert_isaac, "
        "show_paths, help, cancel_job, start_full_workflow, refresh_full_workflow, answer_question, update_settings, unknown. "
        "For 'full CO2RR reaction pathway', use generate_batch_pathway and pathway_adsorbates "
        "['clean','CO2','CO','CHO','OCCO','COCO','OH','H']. "
        "For questions asking paths, use show_paths. For questions asking current settings such as time limit, walltime, queue, account, or what something is, use answer_question. For requests like change time limit to 8 hrs or change queue to debug, use update_settings. "
        "For 'generate structure and relaxation files', use generate_structure_and_relaxation. "
        "For misspelled 'relation files' when context is VASP, infer relaxation files and use generate_structure_and_relaxation. "
        "For 'upload to NERSC under X' or 'uploade to NERSC under X', use upload_nersc and remote_run_name X. Do not set remote_run_name to NERSC. "
        "For 'what files are in that folder' or 'show clean surface folder files', use list_remote_folder. Do not invent a remote_run_name called files. "
        "For 'remove/delete remote folder /pscratch/...' use delete_remote_folder. For 'cancel/scancel Slurm job 12345', use cancel_job. "
        "For 'change to 8 hrs', use update_settings with settings {'nersc_walltime':'08:00:00'}. "
        "For 'generate relaxation inputs and upload to NERSC under X', use generate_relaxation_and_upload and remote_run_name X. For 'start full relax XAS workflow under X', use start_full_workflow and remote_run_name X. For 'refresh/check full workflow', use refresh_full_workflow. "
        "Allowed JSON fields: intent, settings, pathway_adsorbates, remote_run_name, remote_dir, preview_file, explanation. "
        "Do not include prose outside JSON."
    )


def _planner_user_payload(message, params):
    return {
        "message": message,
        "current_settings": {
            "element": params.get("element"),
            "structure_mode": params.get("structure_mode"),
            "adsorbate": params.get("adsorbate"),
            "batch_adsorbates": params.get("batch_adsorbates"),
            "remote_dir": params.get("nersc_remote_dir"),
            "preview_dir": params.get("nersc_preview_dir"),
            "last_submit_dir": params.get("nersc_last_submit_dir"),
            "submit_script": params.get("nersc_remote_submit_script"),
            "last_upload_remote_dir": (st.session_state.get("nersc_last_upload") or {}).get("remote_dir"),
            "last_batch_dir": (st.session_state.get("last_batch_result") or {}).get("run_dir"),
            "last_generated_package_kind": st.session_state.get("last_generated_package_kind"),
        },
    }


def _openai_chat_completion_plan(message, params, *, provider_label, api_key, base_url, model):
    """Call any OpenAI-compatible chat-completions provider and return a JSON plan."""
    try:
        from openai import OpenAI
    except Exception:
        return {"intent": "unknown", "explanation": f"Install the OpenAI-compatible client for {provider_label}: pip install openai"}
    if not api_key:
        return {"intent": "unknown", "explanation": f"Missing API key/token for {provider_label}."}
    try:
        kwargs = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = OpenAI(**kwargs)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _planner_system_prompt()},
                {"role": "user", "content": json.dumps(_planner_user_payload(message, params))},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        text = response.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as exc:
        return {"intent": "unknown", "explanation": f"{provider_label} planner failed: {exc}"}


def _alcf_plan_chat_intent(message, params):
    """Argonne ALCF inference planner using the OpenAI-compatible API surface."""
    cluster = params.get("alcf_cluster") or DEFAULT_ALCF_CLUSTER
    base_url = params.get("alcf_base_url") or DEFAULT_ALCF_BASE_URLS.get(cluster, DEFAULT_ALCF_BASE_URLS["Metis"])
    model = params.get("alcf_model") or DEFAULT_ALCF_MODEL_BY_CLUSTER.get(cluster, DEFAULT_ALCF_MODEL_BY_CLUSTER["Metis"])
    try:
        token = _get_alcf_access_token()
    except Exception as exc:
        return {"intent": "unknown", "explanation": f"Argonne ALCF planner failed: {exc}"}
    return _openai_chat_completion_plan(
        message,
        params,
        provider_label="Argonne ALCF",
        api_key=token,
        base_url=base_url,
        model=model,
    )


def _openai_plan_chat_intent(message, params):
    """OpenAI-hosted planner using the OpenAI-compatible chat-completions client."""
    api_key = os.environ.get("OPENAI_API_KEY", "")
    model = params.get("openai_model") or DEFAULT_OPENAI_MODEL
    return _openai_chat_completion_plan(
        message,
        params,
        provider_label="OpenAI",
        api_key=api_key,
        base_url=None,
        model=model,
    )


def _anthropic_plan_chat_intent(message, params):
    """Native Anthropic Claude planner."""
    try:
        from anthropic import Anthropic
    except Exception:
        return {"intent": "unknown", "explanation": "Install Anthropic client with: pip install anthropic"}
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return {"intent": "unknown", "explanation": "Missing ANTHROPIC_API_KEY."}
    model = params.get("anthropic_model") or DEFAULT_ANTHROPIC_MODEL
    try:
        client = Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=700,
            temperature=0,
            system=_planner_system_prompt(),
            messages=[{"role": "user", "content": json.dumps(_planner_user_payload(message, params))}],
        )
        parts = []
        for block in getattr(response, "content", []) or []:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return _extract_json_object("\n".join(parts))
    except Exception as exc:
        return {"intent": "unknown", "explanation": f"Anthropic planner failed: {exc}"}


def _format_walltime_from_text(text):
    """Parse natural language walltime such as '8 hrs' into HH:MM:SS."""
    raw = str(text or "").lower()
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?|h)\b", raw)
    if m:
        hours = float(m.group(1))
        whole = int(hours)
        minutes = int(round((hours - whole) * 60))
        if minutes >= 60:
            whole += 1
            minutes -= 60
        return f"{whole:02d}:{minutes:02d}:00"
    m = re.search(r"(\d+)\s*:\s*(\d{1,2})(?::\s*(\d{1,2}))?", raw)
    if m:
        hh = int(m.group(1)); mm = int(m.group(2)); ss = int(m.group(3) or 0)
        return f"{hh:02d}:{mm:02d}:{ss:02d}"
    m = re.search(r"(\d+)\s*(?:minutes?|mins?|m)\b", raw)
    if m:
        minutes = int(m.group(1))
        return f"{minutes // 60:02d}:{minutes % 60:02d}:00"
    return None


def _parse_workflow_setting_changes(text):
    """Parse settings-change requests that are not chemistry structure settings."""
    raw = str(text or "")
    low = raw.lower()
    updates = {}

    walltime = _format_walltime_from_text(raw)
    if walltime and any(k in low for k in ["time", "walltime", "wall time", "limit", "hours", "hrs", "hour"]):
        updates["nersc_walltime"] = walltime

    q = re.search(r"(?:queue|qos|partition)\s*(?:to|=|as)?\s*([A-Za-z0-9_.-]+)", raw, flags=re.I)
    if q:
        updates["nersc_queue"] = q.group(1)
    elif "debug" in low and any(k in low for k in ["queue", "qos", "partition"]):
        updates["nersc_queue"] = "debug"
    elif "regular" in low and any(k in low for k in ["queue", "qos", "partition"]):
        updates["nersc_queue"] = "regular"

    acct = re.search(r"(?:account|allocation|project(?: id)?|nersc project)\s*(?:to|=|as)?\s*([A-Za-z]\d{3,6}|[A-Za-z0-9_.-]+)", raw, flags=re.I)
    if acct:
        updates["nersc_account"] = acct.group(1)

    nodes = re.search(r"(?:nodes?)\s*(?:to|=|as)?\s*(\d+)", raw, flags=re.I)
    if nodes:
        updates["nersc_nodes"] = int(nodes.group(1))

    return updates


def _answer_chat_question(text, params):
    """Answer operational questions about the current app/job settings."""
    low = str(text or "").lower()
    if any(k in low for k in ["time limit", "walltime", "wall time", "calculation time"]):
        target = "structure relaxation"
        if "xas" in low or "feff" in low or "fdmnes" in low or "vasp" in low:
            target = "XAS calculation"
        queue = params.get("nersc_queue", "regular")
        account = params.get("nersc_account", DEFAULT_NERSC_ACCOUNT)
        walltime = params.get("nersc_walltime", "05:00:00")
        return (
            f"Current {target} NERSC walltime is `{walltime}` with queue `{queue}` and account `{account}`.\n"
            "You can change it by saying, for example: `change time limit to 8 hrs` or `change queue to debug`."
        )
    if "queue" in low or "account" in low or "allocation" in low or "project" in low:
        return (
            f"Current NERSC settings: account/project `{params.get('nersc_account', DEFAULT_NERSC_ACCOUNT)}`, "
            f"queue `{params.get('nersc_queue', 'regular')}`, nodes `{params.get('nersc_nodes', 1)}`, "
            f"walltime `{params.get('nersc_walltime', '05:00:00')}`. "
            f"Generated VASP 6 CPU scripts use `{DEFAULT_VASP6_MPI_RANKS_PER_NODE}` MPI ranks per node, `{DEFAULT_VASP6_OMP_NUM_THREADS}` OpenMP threads per rank, and `-c {DEFAULT_VASP6_CPUS_PER_TASK}`."
        )
    return (
        "I can answer current workflow settings and execute workflow commands. "
        "For example: `what is the relaxation time limit?`, `change time limit to 8 hrs`, "
        "`change queue to debug`, `prepare full workflow and upload to NERSC under test05`, or `refresh full workflow status`."
    )


def _extract_adsorbate_target_from_text(text):
    """Extract a batch adsorbate target from a submit/preview command."""
    low = str(text or "").lower()
    if "clean" in low or "bare" in low:
        return "clean"
    for ads in ["OCCO", "COCO", "CHOH", "CHO", "CO2", "CO", "OH", "CH3", "CH2", "CH", "H"]:
        if re.search(r"(?<![A-Za-z0-9])\*?" + re.escape(ads.lower()) + r"(?![A-Za-z0-9])", low):
            return ads
    return None


def _find_batch_sample_name_for_adsorbate(params, adsorbate):
    """Find generated batch sample folder name for an adsorbate such as CO or clean."""
    adsorbate = adsorbate or ""
    target = adsorbate.lower()
    batch = st.session_state.get("last_batch_result") or {}
    for item in batch.get("summary", []) or []:
        name = str(item.get("name", ""))
        low_name = name.lower()
        if target and (low_name.endswith("_" + target) or low_name == target or (target == "clean" and low_name.endswith("_clean"))):
            return name
    # Fall back to the naming convention used by build_pathway_batch_items.
    element = params.get("element", "Cu")
    facet = params.get("facet", "111")
    return sanitize_name(f"{element}_{facet}_{adsorbate}") if adsorbate else None


def _remote_submit_from_natural_language(text, params):
    """Resolve phrases like 'submit clean surface relaxation job' to a batch submit path."""
    low = str(text or "").lower()
    ads = _extract_adsorbate_target_from_text(text)
    if not ads:
        return None
    sample = _find_batch_sample_name_for_adsorbate(params, ads)
    if not sample:
        return None

    if "relax" in low or "structure" in low or "surface" in low:
        rel = f"{sample}/01_structure/submit_relax.sh"
    elif "fdmnes" in low:
        rel = f"{sample}/02_XAS/FDMNES/submit.sh"
    elif "feff" in low:
        rel = f"{sample}/02_XAS/FEFF/submit.sh"
    elif "vasp" in low or "xas" in low:
        rel = f"{sample}/02_XAS/VASP/submit.sh"
    else:
        rel = f"{sample}/01_structure/submit_relax.sh"

    base = None
    last_upload = st.session_state.get("nersc_last_upload") or {}
    if last_upload.get("remote_dir"):
        base = last_upload.get("remote_dir")
    else:
        base = params.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)
    return str(PurePosixPath(str(base).rstrip("/")) / rel)

# =============================================================================
# Web NERSC chat/runtime patches
# =============================================================================

def _fw_clean_run_name(name):
    raw_name = str(name or "").strip().strip(".,;:)>]")
    raw_name = re.sub(r"\s+", "_", raw_name)
    raw_name = re.sub(r"^(?:folder|dir|directory|run)_+", "", raw_name, flags=re.I)
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




def _fw_active_remote_dir(params, text=""):
    """Resolve the active full-workflow remote directory, even after Streamlit reruns."""
    raw = str(text or "")
    m = re.search(r"(/pscratch/[^\s`'\"]+|/global/[^\s`'\"]+)", raw)
    if m:
        remote_dir = m.group(1).rstrip(".,;:)>]")
        params["nersc_remote_dir"] = remote_dir
        params["nersc_last_submit_dir"] = remote_dir
        params["nersc_download_remote_dir"] = remote_dir
        params["nersc_preview_dir"] = remote_dir
        st.session_state["last_full_workflow_remote_dir"] = remote_dir
        return remote_dir

    for key in [
        "nersc_last_submit_dir",
        "nersc_download_remote_dir",
        "nersc_preview_dir",
        "nersc_remote_dir",
    ]:
        value = str(params.get(key, "") or "").strip()
        if value:
            return value.rstrip("/")

    value = str(st.session_state.get("last_full_workflow_remote_dir", "") or "").strip()
    return value.rstrip("/")




def _fw_extract_command_output(result):
    """Extract stdout from an SF API task result, including nested JSON string payloads."""
    if not isinstance(result, dict):
        return str(result)

    payload = result.get("result")
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            pass

    if isinstance(payload, dict):
        return str(
            payload.get("output")
            or payload.get("stdout")
            or payload.get("raw_result")
            or json.dumps(payload, indent=2)
        )

    return str(
        result.get("output")
        or result.get("stdout")
        or result.get("raw_result")
        or json.dumps(result, indent=2)
    )


def _fw_parse_workflow_state_from_output(output):
    """Find and parse the workflow_state JSON object from command output."""
    text = str(output or "")
    decoder = json.JSONDecoder()
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(text[i:])
        except Exception:
            continue
        if isinstance(obj, dict) and "stage" in obj and "jobs" in obj:
            return obj
    return None


def _fw_format_workflow_status(remote_dir, result):
    """Compact human-readable status report for the full relax -> XAS workflow."""
    output = _fw_extract_command_output(result)
    state = _fw_parse_workflow_state_from_output(output)

    if not state:
        return (
            f"Full workflow status for `{remote_dir}`:\n\n"
            "Could not parse `workflow_state.json` from the command output. Raw output:\n\n"
            f"```text\n{output[:5000]}\n```"
        )

    jobs = state.get("jobs", {}) or {}
    run_name = str(remote_dir).rstrip("/").split("/")[-1]
    stage = state.get("stage", "unknown")
    updated = state.get("updated_utc", "")

    lines = [
        f"Full workflow status: `{run_name}`",
        f"Remote: `{remote_dir}`",
        f"Stage: `{stage}`",
    ]

    if updated:
        lines.append(f"Updated UTC: `{updated}`")
    lines.append("")

    if not jobs:
        lines.append("No Slurm jobs are registered yet.")
    else:
        label_map = {
            "relax": "Relaxation",
            "workflow_xas": "XAS driver",
            "xas_VASP": "VASP XAS",
            "xas_FDMNES": "FDMNES XAS",
            "xas_FEFF": "FEFF XAS",
        }
        order = ["relax", "workflow_xas", "xas_VASP", "xas_FDMNES", "xas_FEFF"]
        ordered_names = [name for name in order if name in jobs]
        ordered_names += [name for name in jobs if name not in order]

        for name in ordered_names:
            job = jobs.get(name, {}) or {}
            label = label_map.get(name, name)
            job_id = job.get("job_id", "")
            job_state = job.get("state", "")
            dependency = job.get("dependency") or ""
            script = job.get("script", "")

            lines.append(f"**{label}**")
            lines.append(f"- Job ID: `{job_id}`")
            lines.append(f"- State: `{job_state}`")
            if dependency:
                lines.append(f"- Dependency: `{dependency}`")
            lines.append(f"- Script: `{script}`")
            lines.append("")

    events = state.get("events", []) or []
    if events:
        lines.append("Recent events:")
        for event in events[-3:]:
            time_utc = event.get("time_utc", "")
            message = event.get("message", "")
            lines.append(f"- `{time_utc}` — {message}")

    return "\n".join(lines)

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
    _save_full_workflow_state(state, params)
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
    _save_full_workflow_state(state, params)
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
        _save_full_workflow_state(state, params)
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
        _save_full_workflow_state(state, params)
        return state

    st.session_state["nersc_full_workflow"] = state
    _save_full_workflow_state(state, params)
    return state

def _execute_chat_intent(intent, text, params, uploaded_file=None, plan=None):
    """Execute one explicit chat intent. Rule-based and optional LLM plans both use this."""
    low = str(text or "").lower()
    plan = plan or {}

    if plan.get("settings") and isinstance(plan.get("settings"), dict):
        for key, value in plan["settings"].items():
            if key in params and value not in (None, ""):
                params[key] = value


    if intent in {"refresh_full_workflow_status", "check_full_workflow_status"} or "workflow status" in low:
        remote_dir = _fw_active_remote_dir(params, text)
        if not remote_dir:
            raise ValueError("No full workflow directory is known. Use: check workflow status in /pscratch/.../web_xas_agent_runs/test05")
        client = _chat_client_from_params(params)
        cmd = "cd " + client.shell_quote(remote_dir) + " && bash workflow_status.sh"
        task = client.run_command(cmd)
        task_id = str(task.get("task_id") or task.get("id") or "")
        result = client.wait_task(task_id, timeout_s=120) if task_id else task
        st.session_state["last_full_workflow_remote_dir"] = remote_dir
        params["nersc_remote_dir"] = remote_dir
        params["nersc_last_submit_dir"] = remote_dir
        params["nersc_download_remote_dir"] = remote_dir
        params["nersc_preview_dir"] = remote_dir
        return _fw_format_workflow_status(remote_dir, result)

    if intent == "help":
        return (
            "I can update workspace settings and run workflow actions. Examples:\n"
            "- `Cu 3x3 with CO`\n"
            "- `Generate full CO2RR reaction pathway on Cu`\n"
            "- `generate structure`\n"
            "- `generate relaxation inputs and upload to NERSC under test03`\n"
            "- `submit selected job`\n"
            "- `check task`, `check Slurm`, `list jobs`\n"
            "- `what is the full path of uploaded folder`\n"
            "- `list remote folder`, `show me what is under /pscratch/...`, `preview vasp.out`, `download outputs`, `convert ISAAC`"
        )

    if intent == "show_paths":
        return _describe_current_uploaded_paths(params)

    if intent == "answer_question":
        return _answer_chat_question(text, params)

    if intent == "generate_batch_pathway":
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=False)
        params.update(parsed)
        adsorbates = plan.get("pathway_adsorbates") or _pathway_adsorbates_from_text(text)
        params["batch_source"] = "Generate pathway from current structure setup"
        params["batch_adsorbates"] = [a for a in adsorbates if a in ADSORBATE_CHOICES]
        result = _generate_batch_from_current_settings(params)
        params["nersc_upload_source"] = "Last batch package"
        return (
            f"Full CO2RR pathway batch generated for {result.get('n_structures')} structures at `{result.get('run_dir')}`.\n"
            f"Pathway adsorbates: `{', '.join(params['batch_adsorbates'])}`.\n"
            "Next: upload to NERSC, or say `upload to NERSC under test03`."
        )

    if intent == "generate_structure":
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
        params.update(parsed)
        structure, paths, _substrate_n = _generate_structure_from_params(params, uploaded_file=uploaded_file)
        formula = " ".join(f"{el}{structure['atoms'].count(el)}" for el in dict.fromkeys(structure['atoms']))
        return f"Structure generated: `{formula}`. POSCAR: `{paths.get('poscar')}`. Next: generate relaxation inputs."

    if intent == "prepare_full_workflow":
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
        if parsed or st.session_state.get("last_structure") is None:
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
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
        if parsed or st.session_state.get("last_structure") is None:
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
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
        if parsed or st.session_state.get("last_structure") is None:
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

    if intent == "generate_structure_and_relaxation":
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
        params.update(parsed)
        structure, paths, _substrate_n = _generate_structure_from_params(params, uploaded_file=uploaded_file)
        relax_result = _generate_relaxation_from_current_structure(params)
        params["nersc_upload_source"] = "Full package root"
        formula = " ".join(f"{el}{structure['atoms'].count(el)}" for el in dict.fromkeys(structure['atoms']))
        return (
            f"Structure generated: `{formula}`. POSCAR: `{paths.get('poscar')}`.\n"
            f"VASP relaxation inputs generated at `{relax_result.get('relax_folder')}`. "
            "Next: upload to NERSC or submit `01_structure/submit_relax.sh`."
        )

    if intent == "generate_relaxation":
        result = _generate_relaxation_from_current_structure(params)
        params["nersc_upload_source"] = "Full package root"
        return f"VASP relaxation inputs generated at `{result.get('relax_folder')}`. Next: upload to NERSC or submit `01_structure/submit_relax.sh`."

    if intent == "generate_xas":
        result, xas_dir, edge = _generate_xas_from_current_structure(params)
        params["nersc_upload_source"] = "Full package root"
        return f"XAS input folders generated at `{xas_dir}` for absorber `{params.get('xas_absorber')}` `{edge}` edge. Next: upload/submit VASP, FDMNES, or FEFF."

    if intent == "generate_relaxation_and_upload":
        remote_dir = _remote_dir_from_natural_language(plan.get("remote_run_name") or text, params)
        params["nersc_remote_dir"] = remote_dir

        # If a pathway/batch package was just generated, do NOT generate a new
        # single-structure relaxation folder. The batch package already contains
        # item-specific 01_structure folders. Upload that package instead.
        using_batch = _prefer_last_batch_for_chat_upload(params, text)
        if using_batch:
            params["nersc_upload_source"] = "Last batch package"
            selected_local_dir = _selected_local_folder_from_params(params)
            prefix = "Using the last generated batch package; no single-structure relaxation folder was regenerated."
        else:
            if st.session_state.get("last_structure") is None:
                _generate_structure_from_params(params, uploaded_file=uploaded_file)
            relax_result = _generate_relaxation_from_current_structure(params)
            params["nersc_upload_source"] = "Full package root"
            selected_local_dir = _selected_local_folder_from_params(params)
            prefix = f"VASP relaxation inputs generated at `{relax_result.get('relax_folder')}`."

        client = _chat_client_from_params(params)
        upload_result = client.upload_directory(
            local_dir=selected_local_dir,
            remote_dir=remote_dir,
            overwrite=bool(params.get("nersc_overwrite_remote", False)),
            timeout_s=240,
        )
        st.session_state["nersc_last_upload"] = upload_result
        params["nersc_download_remote_dir"] = remote_dir
        ok, label = classify_upload_result(upload_result)
        next_hint = (
            "Next: choose an item-specific submit script such as `Cu_111_CO/01_structure/submit_relax.sh`."
            if using_batch else
            "Next: choose/submit `01_structure/submit_relax.sh`."
        )
        return (
            f"{prefix}\n"
            f"{label}. Uploaded `{selected_local_dir}` to `{remote_dir}`.\n"
            f"{next_hint}"
        )

    if intent == "upload_nersc":
        remote_dir = _remote_dir_from_natural_language(plan.get("remote_run_name") or text, params)
        params["nersc_remote_dir"] = remote_dir
        if _prefer_last_batch_for_chat_upload(params, text):
            params["nersc_upload_source"] = "Last batch package"
        client = _chat_client_from_params(params)
        selected_local_dir = _selected_local_folder_from_params(params)
        if not selected_local_dir or not Path(selected_local_dir).is_dir():
            raise ValueError(f"Local folder does not exist: {selected_local_dir}")
        result = client.upload_directory(
            local_dir=selected_local_dir,
            remote_dir=remote_dir,
            overwrite=bool(params.get("nersc_overwrite_remote", False)),
            timeout_s=240,
        )
        st.session_state["nersc_last_upload"] = result
        params["nersc_download_remote_dir"] = remote_dir
        ok, label = classify_upload_result(result)
        return f"{label}. Uploaded `{selected_local_dir}` to `{remote_dir}`."

    if intent == "submit":
        client = _chat_client_from_params(params)
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
        job_id_display = params.get("nersc_job_id") or "not available yet; run `check task` in a few seconds"
        return f"Submit request completed for `{remote_submit}`. SF API task ID: `{params.get('nersc_task_id', '')}`. Slurm job ID: `{job_id_display}`."

    if intent == "check_task":
        client = _chat_client_from_params(params)
        task_id = str(params.get("nersc_task_id", "")).strip()
        if not task_id:
            raise ValueError("No SF API task ID is set.")
        task_result = client.task(task_id)
        st.session_state["nersc_last_task"] = task_result
        parsed_task = client.task_result_json(task_result)
        if parsed_task.get("jobid"):
            params["nersc_job_id"] = str(parsed_task["jobid"])
        return "Task status checked. Open the Task status expander for full JSON."

    if intent == "check_slurm":
        client = _chat_client_from_params(params)
        explicit_job_id = _extract_slurm_job_id_from_text(text)
        planned_job_id = str(plan.get("job_id") or plan.get("slurm_job_id") or "").strip() if isinstance(plan, dict) else ""
        job_id = explicit_job_id or planned_job_id or str(params.get("nersc_job_id", "")).strip()
        if not job_id:
            raise ValueError("No Slurm job ID is set. Say, for example: `check slurm job 55362912`, or paste the ID in the Slurm job ID field.")
        params["nersc_job_id"] = str(job_id)
        result = client.slurm_status(job_id, timeout_s=90)
        st.session_state["nersc_last_slurm"] = result
        output = str(result.get("output", json.dumps(result, indent=2)))
        if not output.strip():
            output = f"No squeue/sacct output was returned for job {job_id}. It may be completed, purged from accounting, or not visible to this SF API client."
        return f"Slurm status checked for job `{job_id}`:\n\n" + output[:4000]

    if intent == "list_jobs":
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
                current_text = marker + "\n" + current_text.strip()
            lines = [line.strip() for line in current_text.splitlines() if line.strip()]
            data_lines = [
                line for line in lines
                if not line.startswith("==") and not line.startswith("JOBID|") and not line.startswith("JOBID ")
            ]
            if data_lines:
                current_display = "\n".join(lines)[:4000]
                return f"Current queued/running NERSC jobs for `{username}`:\n\n```text\n{current_display}\n```"
            hint = ""
            if params.get("nersc_task_id") and not params.get("nersc_job_id"):
                hint = f"\n\nLast submit SF API task ID is `{params.get('nersc_task_id')}`. If this was just submitted, run `check task` to see whether Slurm returned a job ID."
            return f"No current queued or running NERSC jobs were found for `{username}`." + hint
        return f"Current/recent NERSC jobs for `{username}`:\n\n" + output[:5000]

    if intent == "list_remote_folder":
        client = _chat_client_from_params(params)
        remote_dir = _remote_dir_for_listing_request(text, params, plan=plan)
        params["nersc_preview_dir"] = remote_dir
        result = client.remote_ls(remote_dir, timeout_s=90, max_depth=3)
        st.session_state["nersc_last_ls"] = result
        output = str(result.get("output", json.dumps(result, indent=2)))
        return f"Remote folder listed: `{remote_dir}`\n\n" + output[:6000]

    if intent == "delete_remote_folder":
        client = _chat_client_from_params(params)
        target = _remote_dir_explicitly_mentioned(text, params)
        if not target and any(p in str(text).lower() for p in ["that folder", "this folder"]):
            target = params.get("nersc_preview_dir") or params.get("nersc_last_submit_dir") or params.get("nersc_download_remote_dir") or params.get("nersc_remote_dir")
        target = _safe_remote_delete_target(target)
        result = client.remote_rm_dir(target, timeout_s=90)
        st.session_state["nersc_last_delete"] = result
        for key in ["nersc_preview_dir", "nersc_last_submit_dir", "nersc_download_remote_dir"]:
            if str(params.get(key, "")).rstrip("/") == target:
                params[key] = ""
        output = str(result.get("output", json.dumps(result, indent=2)))
        return f"Deleted remote folder: `{target}`\n\n" + output[:2000]

    if intent == "preview_file":
        client = _chat_client_from_params(params)
        filename = plan.get("preview_file")
        if not filename:
            match = re.search(r"(?:preview|tail)\s+([^\s]+)", str(text or ""), flags=re.I)
            filename = match.group(1) if match else params.get("nersc_tail_file", "vasp.out")
        params["nersc_tail_file"] = filename
        remote_dir = params.get("nersc_preview_dir") or params.get("nersc_last_submit_dir") or params.get("nersc_download_remote_dir") or params.get("nersc_remote_dir")
        remote_file = filename if str(filename).startswith("/") else str(remote_dir).rstrip("/") + "/" + str(filename)
        result = client.remote_tail(remote_file, n_lines=120, timeout_s=90)
        st.session_state["nersc_last_tail"] = result
        return f"Previewed `{remote_file}`:\n\n" + str(result.get("output", json.dumps(result, indent=2)))[:4000]

    if intent == "download_outputs":
        client = _chat_client_from_params(params)
        remote_dir = params.get("nersc_download_remote_dir") or params.get("nersc_last_submit_dir") or params.get("nersc_remote_dir")
        archive_bytes, archive_meta = client.download_remote_archive(
            remote_dir=remote_dir,
            include_glob=params.get("nersc_download_glob", "*.out OUTCAR OSZICAR CONTCAR vasprun.xml xmu.dat *_conv.txt CORE_DIELECTRIC_IMAG.dat isaac_run_metadata.json isaac_record_draft.json"),
            timeout_s=240,
        )
        st.session_state["nersc_download_archive_bytes"] = archive_bytes
        st.session_state["nersc_download_archive_meta"] = archive_meta
        local_extract_dir = params.get("isaac_local_output_dir", "generated_outputs/web_xas_agent/downloaded_outputs")
        extracted = extract_archive_bytes(archive_bytes, local_extract_dir, clean=True)
        st.session_state["nersc_last_download_extract_dir"] = extracted
        params["isaac_local_output_dir"] = extracted
        return f"Downloaded and extracted outputs from `{remote_dir}` to `{extracted}`. Next: convert ISAAC."

    if intent == "cancel_job":
        return _fw_cancel_job(params, text)

    if intent == "start_full_workflow":
        remote_text = plan.get("remote_run_name") or plan.get("remote_dir") or text
        state = _start_full_relax_xas_workflow(params, text=remote_text, uploaded_file=uploaded_file)
        return (
            f"Full relax -> XAS workflow started. Relaxation job `{state.get('relax_job_id') or 'not available yet'}` "
            f"was submitted from `{state.get('relax_remote_submit')}`. Remote folder: `{state.get('remote_dir')}`. "
            "Use `refresh full workflow` to monitor relaxation and submit regenerated XAS jobs after CONTCAR is available."
        )

    if intent == "refresh_full_workflow":
        state = _refresh_full_relax_xas_workflow(params)
        return "Full workflow refreshed. Current stage: `{}`. Jobs: `{}`.".format(state.get("stage"), len(_full_workflow_rows(state)))

    if intent == "convert_isaac":
        output_dir = params.get("isaac_local_output_dir", "generated_outputs/web_xas_agent/downloaded_outputs")
        record = build_isaac_record_from_outputs(
            output_dir=output_dir,
            software=params.get("isaac_software", "auto"),
            sample_name=params.get("isaac_sample_name", "not_specified"),
        )
        record_path = write_isaac_record(output_dir, record)
        st.session_state["isaac_last_record"] = record
        st.session_state["isaac_last_record_path"] = record_path
        return f"ISAAC draft record written to `{record_path}` with {len(record.get('assets', []))} assets and {len(record.get('descriptors', {}).get('outputs', []))} parsed outputs."

    if intent == "update_settings":
        workflow_updates = _parse_workflow_setting_changes(text)
        parsed = _safe_parse_settings_from_text(text, allow_adsorbate=True)
        updates = {}
        updates.update(parsed)
        updates.update(workflow_updates)
        if updates:
            params.update(updates)
            friendly = ", ".join(f"{k}={v}" for k, v in updates.items())
            if any(k.startswith("nersc_") for k in updates):
                return "Updated NERSC/workflow settings: " + friendly + ". These settings will be used for newly generated submit scripts."
            return "Updated sidebar settings: " + friendly + ". You can now say `generate structure`."
        return "I did not find settings to update."

    return "I did not understand this request. Say `help` to see supported commands."



def _full_workflow_intent_from_text(text):
    low = str(text or "").lower().strip()
    is_full = any(x in low for x in ["full workflow", "workflow package", "relax-to-xas", "relax to xas", "relaxation to xas"])
    if is_full:
        if any(x in low for x in ["cancel", "stop"]):
            return "cancel_full_workflow"
        if any(x in low for x in ["restart", "continue", "resume"]):
            return "restart_full_workflow"
        if any(x in low for x in ["status", "refresh", "check"]):
            return "refresh_full_workflow_status"
        has_upload = any(x in low for x in ["upload", "uploade", "uploaded", "send", "copy"])
        has_start = any(x in low for x in ["start", "run", "submit"])
        has_prepare = any(x in low for x in ["prepare", "generate", "make", "build"])
        if has_upload and has_start:
            return "prepare_upload_start_full_workflow"
        if has_upload:
            return "prepare_upload_full_workflow"
        if has_start:
            return "start_full_workflow"
        if has_prepare:
            return "prepare_full_workflow"
    if "upload" in low and any(x in low for x in ["did", "have", "was", "is", "where", "path", "status"]):
        return "show_paths"
    return None

def _classify_chat_intent_rule_based(message):
    """Intent-first rule classifier. Avoids parsing chemistry inside questions/actions."""
    low = str(message or "").lower().strip()
    full_workflow_override = _full_workflow_intent_from_text(message)
    if full_workflow_override:
        return full_workflow_override
    if low in {"help", "commands", "what can you do"}:
        return "help"
    if re.search(r"\b(remove|delete|rm)\b", low) and ("/pscratch/" in low or "/global/" in low or "that folder" in low or "remote" in low):
        return "delete_remote_folder"
    if ("/pscratch/" in low or "/global/" in low) and any(p in low for p in ["show", "list", "what", "what's", "under", "ls"]):
        return "list_remote_folder"
    if any(p in low for p in ["what is the full path", "uploaded folder", "upload path", "submitted folder", "remote path", "where did", "where is"]):
        return "show_paths"
    full_workflow_requested = any(p in low for p in ["full workflow", "workflow package", "relax-to-xas", "relax to xas", "relaxation to xas"])
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

    if re.search(r"\b(cancel|scancel)\b", low) and re.search(r"\b(job|slurm|squeue|sacct)\b|[0-9]{5,12}", low):
        return "cancel_job"
    if _chat_requested_job_listing(low):
        return "list_jobs"
    if _extract_slurm_job_id_from_text(low) and ("slurm" in low or "job" in low or "squeue" in low or "sacct" in low):
        return "check_slurm"
    if (
        "what files" in low or "which files" in low or "files are in" in low
        or ("files" in low and "folder" in low and any(v in low for v in ["show", "list", "print", "display"]))
        or "what's under" in low or "what is under" in low or "show me what's under" in low
        or "ls " in low or low.startswith("ls")
    ) and ("folder" in low or "remote" in low or "there" in low or "that" in low or "nersc" in low or "/pscratch/" in low or "/global/" in low):
        return "list_remote_folder"
    if re.search(r"\b(change|set|modify|update)\b", low) and (
        any(k in low for k in ["time", "walltime", "wall time", "limit", "queue", "account", "project", "allocation", "node"])
        or re.search(r"\d+(?:\.\d+)?\s*(?:hours?|hrs?|h)\b", low)
    ):
        return "update_settings"
    if low.endswith("?") or low.startswith(("what ", "what's ", "what is ", "how ", "can i ", "could i ", "do i ", "does ", "is ")):
        return "answer_question"
    if any(p in low for p in ["full workflow", "relax xas workflow", "relax then xas", "relaxation then xas", "post-relax xas", "post relax xas", "full relax xas"]):
        return "refresh_full_workflow" if any(p in low for p in ["refresh", "check", "status", "monitor"]) else "start_full_workflow"
    if any(p in low for p in ["full co2rr", "full reaction pathway", "reaction pathway", "full pathway", "pathway on"]):
        return "generate_batch_pathway" if "generate" in low or "make" in low or "build" in low else "update_settings"
    if "generate" in low and ("structure" in low or "surface" in low) and ("relax" in low or "relaxation" in low or "relation" in low):
        return "generate_structure_and_relaxation"
    if "generate" in low and ("relax" in low or "relaxation" in low or "relation" in low) and "upload" in low:
        return "generate_relaxation_and_upload"
    if ("upload" in low or "uploade" in low or "uploaded" in low) and "nersc" in low:
        return "upload_nersc"
    if "submit" in low:
        return "submit"
    if "check" in low and "task" in low:
        return "check_task"
    if ("check" in low and "slurm" in low) or "job status" in low or low == "status":
        return "check_slurm"
    if "list" in low and "job" in low:
        return "list_jobs"
    if "list" in low and "remote" in low:
        return "list_remote_folder"
    if "preview" in low or "tail" in low:
        return "preview_file"
    if "download" in low and ("output" in low or "nersc" in low):
        return "download_outputs"
    if "isaac" in low or "convert" in low:
        return "convert_isaac"
    if "generate" in low and "batch" in low:
        return "generate_batch_pathway"
    if "generate" in low and ("relax" in low or "relaxation" in low or "relation" in low):
        return "generate_relaxation"
    if "generate" in low and "xas" in low:
        return "generate_xas"
    if "generate" in low and "structure" in low:
        return "generate_structure"
    if re.search(r"\b(cu|au|ni|ag|pt|pd|ir|rh|al)\b", low) or re.search(r"\d+\s*[x×*]\s*\d+", low):
        return "update_settings"
    return "unknown"


def _force_full_workflow_intent(text):
    low = str(text or "").lower().strip()
    is_full = (
        any(x in low for x in [
            "full workflow", "workflow package", "relax-to-xas", "relax to xas", "relaxation to xas"
        ])
        or (
            "workflow" in low
            and any(x in low for x in ["generate", "prepare", "make", "build", "upload", "start", "submit"])
            and not any(x in low for x in ["batch workflow", "pathway workflow"])
        )
    )
    if not is_full:
        if "upload" in low and any(x in low for x in ["did", "have", "was", "is", "where", "path", "status"]):
            return "show_paths"
        return None
    if any(x in low for x in ["cancel", "stop"]):
        return "cancel_full_workflow"
    if any(x in low for x in ["restart", "continue", "resume"]):
        return "restart_full_workflow"
    if any(x in low for x in ["status", "refresh", "check"]):
        return "refresh_full_workflow_status"
    has_upload = any(x in low for x in ["upload", "uploade", "uploaded", "send", "copy"])
    has_prepare = any(x in low for x in ["prepare", "generate", "make", "build"])
    has_start = any(x in low for x in ["start", "submit"])
    if has_upload and has_start:
        return "prepare_upload_start_full_workflow"
    if has_upload:
        return "prepare_upload_full_workflow"
    if has_start:
        return "start_full_workflow"
    if has_prepare:
        return "prepare_full_workflow"
    return None

def _run_chat_command(message, params, uploaded_file=None):
    """Route a chat message through intent-first command handling.

    v19 used keyword/structure parsing first, which caused bugs such as:
    - "full CO2RR reaction pathway" becoming adsorbate=CO2 only;
    - "what is the uploaded path" becoming adsorbate=H;
    - compound commands such as "generate relaxation inputs and upload" stopping
      after the first action.

    v23 classifies workflow intent/questions first, supports settings edits, and parses chemistry only when
    that intent actually needs chemistry settings.
    """
    text = str(message or "").strip()
    if not text:
        return "Please type a command or structure request."
    forced_intent = _force_full_workflow_intent(text)
    if forced_intent:
        st.session_state["last_chat_planner_trace"] = {
            "backend_requested": params.get("chat_backend"),
            "planner_used": "Rule-based forced full workflow",
            "fallback_reason": "forced before LLM/default planner",
            "resolved_intent": forced_intent,
            "plan": None,
            "message": text,
        }
        try:
            return _execute_chat_intent(forced_intent, text, params, uploaded_file=uploaded_file, plan=None)
        except Exception as exc:
            return f"Failed: {exc}"

    plan = None
    backend = params.get("chat_backend", "Rule-based")
    planner_used = "Rule-based"
    if backend == "Argonne ALCF":
        plan = _alcf_plan_chat_intent(text, params)
        planner_used = "Argonne ALCF"
    elif backend == "OpenAI":
        plan = _openai_plan_chat_intent(text, params)
        planner_used = "OpenAI"
    elif backend == "Anthropic":
        plan = _anthropic_plan_chat_intent(text, params)
        planner_used = "Anthropic"

    intent = None
    fallback_reason = ""
    if isinstance(plan, dict):
        intent = plan.get("intent")
        # If the LLM planner failed or returned unknown, fall back to deterministic rules.
        if intent in {None, "", "unknown"}:
            fallback_reason = plan.get("explanation", "ALCF planner returned unknown/empty intent")
            intent = _classify_chat_intent_rule_based(text)

        # Deterministic safety/path overrides. The LLM planner is allowed to
        # interpret workflow intent, but remote path operations must be literal
        # and safe: do not let the model answer/help when the user explicitly
        # asked to list or delete a concrete NERSC path.
        low_text = text.lower()
        full_workflow_override = _full_workflow_intent_from_text(text)
        if full_workflow_override:
            intent = full_workflow_override

        if re.search(r"\b(cancel|scancel)\b", low_text) and re.search(r"\b(job|slurm|squeue|sacct)\b|[0-9]{5,12}", low_text):
            intent = "cancel_job"
        if any(p in low_text for p in ["full workflow", "relax xas workflow", "relax then xas", "relaxation then xas", "post-relax xas", "post relax xas", "full relax xas"]):
            intent = "refresh_full_workflow" if any(p in low_text for p in ["refresh", "check", "status", "monitor"]) else "start_full_workflow"
        elif intent != "cancel_job" and _chat_requested_job_listing(low_text):
            intent = "list_jobs"
        elif intent != "cancel_job" and _extract_slurm_job_id_from_text(low_text) and ("slurm" in low_text or "job" in low_text or "squeue" in low_text or "sacct" in low_text):
            intent = "check_slurm"
        elif re.search(r"\b(remove|delete|rm)\b", low_text) and ("/pscratch/" in low_text or "/global/" in low_text or "that folder" in low_text or "remote" in low_text):
            intent = "delete_remote_folder"
        elif ("/pscratch/" in low_text or "/global/" in low_text) and any(pat in low_text for pat in ["show", "list", "what", "what's", "under", "ls"]):
            intent = "list_remote_folder"
        # Guardrail: if the user only says "upload to NERSC under test03", do
        # not let the model reinterpret that as "generate a fresh relaxation and
        # upload". This preserves the last generated package.
        if intent == "generate_relaxation_and_upload" and ("upload" in low_text or "uploade" in low_text) and not re.search(r"generate|make|build|relax|relation", low_text):
            intent = "upload_nersc"
    else:
        intent = _classify_chat_intent_rule_based(text)
        planner_used = "Rule-based"

    st.session_state["last_chat_planner_trace"] = {
        "backend_requested": params.get("chat_backend"),
        "planner_used": planner_used,
        "fallback_reason": fallback_reason,
        "resolved_intent": intent,
        "plan": plan,
        "message": text,
    }

    try:
        result = _execute_chat_intent(intent, text, params, uploaded_file=uploaded_file, plan=plan)
        if params.get("chat_debug_planner", False):
            result += "\n\nPlanner trace: `" + json.dumps(st.session_state["last_chat_planner_trace"], default=str)[:1200] + "`"
        return result
    except Exception as exc:
        return f"Failed: {exc}"

st.set_page_config(page_title="CO2RR XAS Agent", layout="wide")
st.title("Agentic simulation workflow for XAS")
st.caption("Use the sidebar as the agent console. Use the main workspace for catalyst structure setup, structure/XAS input generation, batch generation, NERSC jobs, ISAAC records, and ISAAC Portal upload.")

COVERAGE_PRESETS = {
    "1/9 ML (3×3, default)": (3, 3),
    "1/16 ML (4×4)": (4, 4),
    "1/4 ML (2×2)": (2, 2),
    "1 ML (1×1)": (1, 1),
    "custom nx × ny": None,
}

if "params" not in st.session_state:
    st.session_state.params = {
        "structure_source": "Build from settings",
        "structure_mode": "Single metal surface",
        "element": "Cu",
        "element1": "Cu",
        "element2": "Au",
        "facet": "111",
        "nx": 3,
        "ny": 3,
        "coverage_preset": "1/9 ML (3×3, default)",
        "layers": 4,
        "vacuum": 15.0,
        "adsorbate": "CO",
        "height": 2.0,
        "interface_match_mode": "auto",
        "element1_repeats": 4,
        "element2_repeats": 4,
        "interface_binding_element": "Cu",
        "output_dir": "generated_outputs/web_xas_agent",
        "xas_output_dir": "generated_outputs/web_xas_agent",
        "xas_absorber": "Cu",
        "xas_edge": "K",
        "cluster_radius": 6.0,
        "fdmnes_energy_min": -5.0,
        "fdmnes_energy_step": 0.2,
        "fdmnes_energy_max": 50.0,
        "relax_encut": 520,
        "relax_ediff": "1E-6",
        "relax_ibrion": 2,
        "relax_nsw": 200,
        "relax_ediffg": "-0.02",
        "relax_isif": 2,
        "relax_ispin": 2,
        "relax_ncore": 8,
        "relax_lreal": "Auto",
        "relax_ismear": 0,
        "relax_sigma": "0.05",
        "vasp_method": "PBE",
        "potcar_dir": DEFAULT_NERSC_POTCAR_DIR,
        "fdmnes_method": "Green",
        "fdmnes_scf": False,
        "fdmnes_quadrupole": False,
        "fdmnes_spinorbit": False,
        "nersc_account": DEFAULT_NERSC_ACCOUNT,
        "nersc_username": os.environ.get("USER", "hjia"),
        "nersc_queue": "regular",
        "nersc_nodes": 1,
        "nersc_walltime": "05:00:00",
        "nersc_email": "",
        "batch_source": "Generate pathway from current structure setup",
        "batch_output_dir": "generated_outputs/web_xas_agent/batch",
        "batch_adsorbates": ["clean", "CO2", "CO", "CHO", "OCCO", "COCO", "OH", "H"],
        "batch_include_relaxation": True,
        "batch_include_xas": True,
        "batch_absorber_mode": "auto",
        "nersc_enable": False,
        "nersc_system": "perlmutter",
        "sfapi_client_id": os.environ.get("NERSC_SFAPI_CLIENT_ID", ""),
        "sfapi_private_key_path": os.environ.get("NERSC_SFAPI_PRIVATE_KEY_PATH", ""),
        "nersc_remote_dir": DEFAULT_NERSC_REMOTE_RUN_DIR,
        "nersc_upload_source": "Full package root",
        "nersc_custom_local_dir": "",
        "nersc_remote_submit_script": "workflow_submit.sh",
        "nersc_repo_root": os.environ.get("CO2RR_AGENT_REPO", str(REPO_ROOT)),
        "nersc_task_id": "",
        "nersc_job_id": "",
        "nersc_tail_file": "vasp.out",
        "nersc_last_submit_dir": "",
        "nersc_download_remote_dir": DEFAULT_NERSC_REMOTE_RUN_DIR,
        "nersc_download_glob": "*.out OUTCAR OSZICAR CONTCAR vasprun.xml xmu.dat *_conv.txt CORE_DIELECTRIC_IMAG.dat isaac_run_metadata.json isaac_record_draft.json",
        "isaac_local_output_dir": "generated_outputs/web_xas_agent/downloaded_outputs",
        "isaac_software": "auto",
        "isaac_sample_name": "not_specified",
        "chat_backend": "Rule-based",
        "alcf_model": DEFAULT_ALCF_MODEL_BY_CLUSTER["Metis"],
        "openai_model": DEFAULT_OPENAI_MODEL,
        "anthropic_model": DEFAULT_ANTHROPIC_MODEL,
    }

uploaded_structure_file = None

if "nersc_full_workflow" not in st.session_state:
    _persisted_workflow_state = _load_full_workflow_state(st.session_state.params)
    if _persisted_workflow_state:
        st.session_state["nersc_full_workflow"] = _persisted_workflow_state


# Polished page layout: sidebar agent console + main workflow workspace.
p = st.session_state.params
with st.sidebar:
    st.header("Agent console")
    st.caption("Ask the agent to update settings, generate inputs, upload to NERSC, submit jobs, check Slurm, or inspect remote files.")
    # Chat box: lightweight workflow console with persistent responses.
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = [
            {
                "role": "assistant",
                "message": "I can answer questions, update settings, and run actions. Try `prepare, upload, and start full workflow under test03`, `refresh full workflow status`, `Generate full CO2RR pathway on Cu`, `what files are in that folder`, or `change to 8 hrs`.",
                "time": datetime.now().isoformat(timespec="seconds"),
            }
        ]

    chat_cols = [st.container(), st.container(), st.container(), st.container()]
    BACKEND_CHOICES = ["Rule-based", "Argonne ALCF", "OpenAI", "Anthropic"]
    with chat_cols[0]:
        p["chat_backend"] = st.selectbox(
            "Agent planner",
            BACKEND_CHOICES,
            index=safe_index(BACKEND_CHOICES, p.get("chat_backend", "Rule-based")),
            help="The LLM backend only plans commands. Local Python executes structures, NERSC actions, and ISAAC conversion.",
        )
    with chat_cols[1]:
        alcf_clusters = ["Metis", "Sophia"]
        p["alcf_cluster"] = st.selectbox(
            "ALCF inference cluster",
            alcf_clusters,
            index=safe_index(alcf_clusters, p.get("alcf_cluster", DEFAULT_ALCF_CLUSTER)),
            disabled=p.get("chat_backend") != "Argonne ALCF",
        )
    with chat_cols[2]:
        selected_cluster = p.get("alcf_cluster", DEFAULT_ALCF_CLUSTER)
        p["alcf_base_url"] = DEFAULT_ALCF_BASE_URLS.get(selected_cluster, DEFAULT_ALCF_BASE_URLS["Metis"])
        if p.get("chat_backend") == "Anthropic":
            p["anthropic_model"] = st.text_input("Anthropic model", value=p.get("anthropic_model") or DEFAULT_ANTHROPIC_MODEL)
        elif p.get("chat_backend") == "OpenAI":
            p["openai_model"] = st.text_input("OpenAI model", value=p.get("openai_model") or DEFAULT_OPENAI_MODEL)
        else:
            p["alcf_model"] = st.text_input(
                "ALCF model",
                value=p.get("alcf_model") or DEFAULT_ALCF_MODEL_BY_CLUSTER.get(selected_cluster, DEFAULT_ALCF_MODEL_BY_CLUSTER["Metis"]),
                disabled=p.get("chat_backend") != "Argonne ALCF",
            )
    with chat_cols[3]:
        if p.get("chat_backend") == "Argonne ALCF":
            token_ready = bool(os.environ.get("ALCF_INFERENCE_ACCESS_TOKEN", ""))
            token_helper_ready = (REPO_ROOT / "inference_auth_token.py").exists() or Path("inference_auth_token.py").exists()
            if token_ready or token_helper_ready:
                st.success("Argonne ALCF planner enabled. Local Python executes workflow actions.")
            else:
                st.warning("Authenticate first: `python inference_auth_token.py authenticate`, or set ALCF_INFERENCE_ACCESS_TOKEN.")
        elif p.get("chat_backend") == "OpenAI":
            if os.environ.get("OPENAI_API_KEY", ""):
                st.success("OpenAI planner enabled. Local Python executes workflow actions.")
            else:
                st.warning("Set OPENAI_API_KEY before using the OpenAI backend.")
        elif p.get("chat_backend") == "Anthropic":
            if os.environ.get("ANTHROPIC_API_KEY", ""):
                st.success("Anthropic planner enabled. Local Python executes workflow actions.")
            else:
                st.warning("Set ANTHROPIC_API_KEY and install `anthropic` before using Claude.")

    with chat_cols[3]:
        p["chat_debug_planner"] = st.checkbox(
            "Show agent debug trace",
            value=bool(p.get("chat_debug_planner", False)),
            help="Show which planner handled the last chat command, plus the resolved intent and JSON plan.",
        )

    with st.expander("Conversation", expanded=True):
        for item in st.session_state.get("chat_history", [])[-12:]:
            with st.chat_message(item.get("role", "assistant")):
                st.markdown(item.get("message", ""))

    chat_prompt = st.chat_input("Ask the agent, e.g. generate full CO2RR pathway on Cu, upload under test03, check Slurm")
    if chat_prompt:
        _chat_add("user", chat_prompt)
        response = _run_chat_command(chat_prompt, st.session_state.params, uploaded_file=uploaded_structure_file)
        _chat_add("assistant", response)
        if hasattr(st, "rerun"):
            st.rerun()
        else:
            st.experimental_rerun()

    if st.session_state.params.get("chat_debug_planner", False) and st.session_state.get("last_chat_planner_trace"):
        with st.expander("Last chat planner trace", expanded=False):
            st.json(st.session_state["last_chat_planner_trace"])



st.header("Workflow workspace")
st.subheader("Structure generation")

with st.expander("Catalyst structure setup", expanded=True):
    p = st.session_state.params

    p["structure_source"] = st.radio(
        "Input structure",
        ["Build from settings", "Upload structure"],
        index=safe_index(["Build from settings", "Upload structure"], p.get("structure_source", "Build from settings")),
    )

    if p["structure_source"] == "Upload structure":
        uploaded_structure_file = st.file_uploader(
            "Upload POSCAR/CONTCAR/CIF/XYZ"
        )
        st.info("Uploaded structures are used directly for relaxation and XAS input generation.")
    else:
        p["structure_mode"] = st.radio(
            "Surface model",
            ["Single metal surface", "Interface"],
            index=safe_index(["Single metal surface", "Interface"], p.get("structure_mode", "Single metal surface")),
        )

        if p["structure_mode"] == "Single metal surface":
            p["element"] = st.selectbox("Surface metal", METAL_CHOICES, index=safe_index(METAL_CHOICES, p["element"]))
            p["coverage_preset"] = st.selectbox(
                "Adsorbate coverage",
                list(COVERAGE_PRESETS),
                index=safe_index(list(COVERAGE_PRESETS), p.get("coverage_preset", "1/9 ML (3×3, default)")),
                help="Coverage is reported as monolayer (ML) for one adsorbate per surface supercell.",
            )
            preset_cell = COVERAGE_PRESETS[p["coverage_preset"]]
            if preset_cell is not None:
                p["nx"], p["ny"] = preset_cell
            c1, c2 = st.columns(2)
            with c1:
                p["nx"] = st.number_input(
                    "Supercell nx",
                    min_value=1,
                    max_value=30,
                    value=int(p["nx"]),
                    disabled=p["coverage_preset"] != "custom nx × ny",
                )
            with c2:
                p["ny"] = st.number_input(
                    "Supercell ny",
                    min_value=1,
                    max_value=30,
                    value=int(p["ny"]),
                    disabled=p["coverage_preset"] != "custom nx × ny",
                )
            if p["adsorbate"] != "clean":
                coverage = 1.0 / max(1, int(p["nx"]) * int(p["ny"]))
                st.write(f"Nominal coverage: **{coverage:.3f} ML** for one adsorbate per {int(p['nx'])}×{int(p['ny'])} cell.")
        else:
            p["element1"] = st.selectbox("Interface metal 1", METAL_CHOICES, index=safe_index(METAL_CHOICES, p["element1"]))
            p["element2"] = st.selectbox("Interface metal 2", METAL_CHOICES, index=safe_index(METAL_CHOICES, p["element2"], 1))
            p["interface_match_mode"] = st.radio(
                "Interface repeat mode",
                ["auto", "manual"],
                index=safe_index(["auto", "manual"], p["interface_match_mode"]),
                help="Auto minimizes strain along the interface direction.",
            )
            if p["interface_match_mode"] == "manual":
                c1, c2 = st.columns(2)
                with c1:
                    p["element1_repeats"] = st.number_input(f"{p['element1']} repeats along interface", min_value=1, max_value=30, value=int(p["element1_repeats"]))
                with c2:
                    p["element2_repeats"] = st.number_input(f"{p['element2']} repeats along interface", min_value=1, max_value=30, value=int(p["element2_repeats"]))
            else:
                st.write("Auto mode chooses low-strain repeats along the interface. `ny` controls rows away from the interface.")
            c1, c2 = st.columns(2)
            with c1:
                p["nx"] = st.number_input("Interface repeat target", min_value=1, max_value=30, value=int(p["nx"]))
            with c2:
                p["ny"] = st.number_input("Rows away from interface", min_value=1, max_value=30, value=int(p["ny"]))

        p["facet"] = st.selectbox("Facet", ["111"], index=0)
        p["layers"] = st.number_input("Layers", min_value=2, max_value=12, value=int(p["layers"]))
        p["vacuum"] = st.number_input("Vacuum thickness, Å", min_value=5.0, max_value=40.0, value=float(p["vacuum"]), step=1.0)

        p["adsorbate"] = st.selectbox("Adsorbate", ADSORBATE_CHOICES, index=safe_index(ADSORBATE_CHOICES, p["adsorbate"]))
        p["height"] = st.number_input("Initial adsorbate height, Å", min_value=0.5, max_value=5.0, value=float(p["height"]), step=0.1)
        if p["adsorbate"] != "clean":
            st.write(f"Initial adsorption site: **{DEFAULT_ADSORBATE_SITES.get(p['adsorbate'], 'top')}**")

        if p["structure_mode"] == "Interface" and p["adsorbate"] != "clean":
            binding_options = [p["element1"], p["element2"]]
            if p.get("interface_binding_element") not in binding_options:
                p["interface_binding_element"] = "Cu" if "Cu" in binding_options else binding_options[0]
            p["interface_binding_element"] = st.selectbox(
                "Adsorbate binding site metal",
                binding_options,
                index=safe_index(binding_options, p["interface_binding_element"]),
            )

    p["output_dir"] = st.text_input("Local output folder", value=p["output_dir"])


with st.expander("Current workflow parameters", expanded=False):
    st.json(st.session_state.params)

if st.button("Generate structure", type="primary"):
    p = st.session_state.params
    try:
        if p.get("structure_source") == "Upload structure":
            if uploaded_structure_file is None:
                raise ValueError("Please upload a POSCAR, CONTCAR, CIF, or XYZ file first.")
            output_dir = Path(p["output_dir"])
            upload_dir = output_dir / "uploaded_structure"
            upload_dir.mkdir(parents=True, exist_ok=True)
            upload_path = upload_dir / uploaded_structure_file.name
            upload_path.write_bytes(uploaded_structure_file.getbuffer())
            structure = read_structure_file(str(upload_path))
            structure["metadata"] = dict(structure.get("metadata", {}))
            structure["metadata"].update({
                "type": "uploaded_structure",
                "source_file": str(upload_path),
                "source": "web_xas_agent_upload",
            })
            substrate_n = len(structure["atoms"])
            paths = StructureGenerator().save_structure(structure, output_dir=p["output_dir"], name="uploaded_structure")
        else:
            structure, paths, substrate_n = generate_structure(
                structure_mode=p["structure_mode"],
                element=p["element"],
                element1=p["element1"],
                element2=p["element2"],
                facet=p["facet"],
                nx=int(p["nx"]),
                ny=int(p["ny"]),
                layers=int(p["layers"]),
                vacuum=float(p["vacuum"]),
                adsorbate=p["adsorbate"],
                height=float(p["height"]),
                output_dir=p["output_dir"],
                interface_match_mode=p["interface_match_mode"],
                element1_repeats=int(p["element1_repeats"]),
                element2_repeats=int(p["element2_repeats"]),
                interface_binding_element=p.get("interface_binding_element", p["element1"]),
            )
        st.session_state["last_structure"] = structure
        st.session_state["last_paths"] = paths
        st.session_state["last_substrate_n"] = substrate_n
        st.session_state["last_structure_error"] = None
        st.session_state["last_generated_package_kind"] = "single"
        st.session_state.pop("last_relax_result", None)
        st.session_state.pop("last_xas_result", None)
    except Exception as exc:
        st.session_state["last_structure_error"] = str(exc)
        st.session_state.pop("last_structure", None)
        st.exception(exc)

if st.session_state.get("last_structure_error"):
    st.error(st.session_state["last_structure_error"])

if st.session_state.get("last_structure") is not None:
    structure = st.session_state["last_structure"]
    paths = st.session_state["last_paths"]
    substrate_n = st.session_state["last_substrate_n"]
    atoms = structure["atoms"]
    positions = np.asarray(structure["positions"], dtype=float)
    metadata = structure.get("metadata", {})

    st.success("Structure generated.")

    if metadata.get("type") == "interface" or "interface" in metadata:
        show_interface_strain(metadata)

    contact_report = element_aware_minimum_pair(atoms, positions)
    if contact_report["severity"] == "error":
        st.error(contact_report["message"])
    elif contact_report["severity"] == "warning":
        st.warning(contact_report["message"])
    else:
        st.info(contact_report["message"])

    reports = adsorbate_geometry_report(structure, substrate_n)
    if reports:
        st.write("**Adsorbate geometry checks**")
        for name, value, unit, ok in reports:
            if ok:
                st.success(f"{name}: {value:.3f} {unit}")
            else:
                st.warning(f"{name}: {value:.3f} {unit}")

    st.subheader("Structure viewer")
    show_color_legend(atoms)
    view_structure_3dmol(atoms, positions, sorted(set(atoms)), substrate_n=substrate_n)

    with st.expander("Metadata", expanded=False):
        st.json(metadata)

    d1, d2 = st.columns(2)
    poscar_path = Path(paths["poscar"])
    if poscar_path.exists():
        with d1:
            st.download_button(
                "Download POSCAR",
                data=poscar_path.read_text(),
                file_name="POSCAR",
                mime="text/plain",
            )
    metadata_path = Path(paths["metadata"])
    if metadata_path.exists():
        with d2:
            st.download_button(
                "Download metadata JSON",
                data=metadata_path.read_text(),
                file_name="structure_info.json",
                mime="application/json",
            )

    st.divider()
    st.header("Structure relaxation")

    p = st.session_state.params
    p["xas_output_dir"] = st.text_input("Calculation package root directory", value=p.get("xas_output_dir", "generated_outputs/web_xas_agent"))

    st.subheader("VASP relaxation settings")
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        p["relax_encut"] = st.number_input("ENCUT", min_value=300, max_value=800, value=int(p.get("relax_encut", 520)), step=20)
        p["relax_ediff"] = st.text_input("EDIFF", value=str(p.get("relax_ediff", "1E-6")))
        p["relax_ediffg"] = st.text_input("EDIFFG", value=str(p.get("relax_ediffg", "-0.02")))
    with r2:
        p["relax_ibrion"] = st.selectbox("IBRION", [1, 2, 3], index=safe_index([1, 2, 3], int(p.get("relax_ibrion", 2)), 1))
        p["relax_nsw"] = st.number_input("NSW", min_value=0, max_value=1000, value=int(p.get("relax_nsw", 200)), step=20)
        p["relax_isif"] = st.selectbox("ISIF", [0, 2, 3], index=safe_index([0, 2, 3], int(p.get("relax_isif", 2)), 1))
    with r3:
        p["relax_ispin"] = st.selectbox("ISPIN", [1, 2], index=safe_index([1, 2], int(p.get("relax_ispin", 2)), 1))
        p["relax_ncore"] = st.number_input("NCORE", min_value=1, max_value=32, value=int(p.get("relax_ncore", 8)))
        p["relax_lreal"] = st.selectbox("LREAL", ["Auto", ".FALSE."], index=safe_index(["Auto", ".FALSE."], p.get("relax_lreal", "Auto")))
    with r4:
        p["relax_ismear"] = st.selectbox("ISMEAR", [-5, 0, 1, 2], index=safe_index([-5, 0, 1, 2], int(p.get("relax_ismear", 0)), 1))
        p["relax_sigma"] = st.text_input("SIGMA", value=str(p.get("relax_sigma", "0.05")))

    st.subheader("NERSC setup")
    n1, n2, n3, n4, n5 = st.columns([1.2, 1, 0.8, 1, 1.6])
    with n1:
        p["nersc_account"] = st.text_input("NERSC project/account", value=p.get("nersc_account", DEFAULT_NERSC_ACCOUNT), key="relax_account")
    with n2:
        p["nersc_queue"] = st.selectbox("Queue/QOS", ["regular", "debug", "premium", "shared"], index=safe_index(["regular", "debug", "premium", "shared"], p.get("nersc_queue", "regular")), key="relax_queue")
    with n3:
        p["nersc_nodes"] = st.number_input("Nodes", min_value=1, max_value=64, value=int(p.get("nersc_nodes", 1)), key="relax_nodes")
    with n4:
        p["nersc_walltime"] = st.text_input("Walltime", value=p.get("nersc_walltime", "05:00:00"), key="relax_walltime")
    with n5:
        p["nersc_email"] = st.text_input("Email, optional", value=p.get("nersc_email", ""), key="relax_email")

    relaxation_settings = relaxation_settings_from_params(p)

    if st.button("Generate VASP relaxation inputs", type="secondary"):
        try:
            relax_result = generate_relaxation_folder(
                structure=structure,
                output_dir=str(Path(p["xas_output_dir"]) / "01_structure"),
                relaxation_settings=relaxation_settings,
                potcar_dir=p.get("potcar_dir", DEFAULT_NERSC_POTCAR_DIR),
                account=p["nersc_account"],
                queue=p["nersc_queue"],
                nodes=p["nersc_nodes"],
                walltime=p["nersc_walltime"],
                email=p["nersc_email"],
            )
            st.session_state["last_relax_result"] = relax_result
            st.session_state["last_xas_dir"] = p["xas_output_dir"]
        except Exception as exc:
            st.exception(exc)

    if st.session_state.get("last_relax_result"):
        relax_result = st.session_state["last_relax_result"]
        st.success("VASP relaxation inputs generated.")
        with st.expander("Relaxation file list", expanded=False):
            relax_tree_dir = relax_result.get("directory") or relax_result.get("relax_folder") or str(Path(p["xas_output_dir"]) / "01_structure")
            st.code(file_tree(relax_tree_dir), language="text")
        vasp_folder = Path(relax_result["relax_folder"])
        if vasp_folder.exists():
            st.download_button(
                "Download VASP relaxation ZIP",
                data=zip_directory_bytes(vasp_folder),
                file_name="VASP_inputs_relaxation.zip",
                mime="application/zip",
                key="download_vasp_relax_zip",
            )

    st.divider()
    st.header("XAS calculation")
    st.info("XAS inputs are generated from the current structure. Relax the structure first if it hasn't been relaxed.")

    present_elements = list(dict.fromkeys(atoms))
    absorber_options = [el for el in present_elements if el not in {"H"}] or present_elements
    if p.get("xas_absorber") not in absorber_options:
        p["xas_absorber"] = "Cu" if "Cu" in absorber_options else absorber_options[0]

    st.subheader("Common XAS settings")
    c1, c2, c3 = st.columns(3)
    with c1:
        p["xas_absorber"] = st.selectbox("Absorber element", absorber_options, index=safe_index(absorber_options, p["xas_absorber"]))
    with c2:
        p["xas_edge"] = st.selectbox("Edge", ["K", "L3", "auto"], index=safe_index(["K", "L3", "auto"], p.get("xas_edge", "K")))
        resolved_edge = normalize_edge_for_absorber(p["xas_absorber"], p["xas_edge"])
    with c3:
        p["cluster_radius"] = st.number_input("Cluster radius, Å", min_value=2.0, max_value=12.0, value=float(p.get("cluster_radius", 6.0)), step=0.5)

    e1, e2, e3 = st.columns(3)
    with e1:
        p["fdmnes_energy_min"] = st.number_input("Energy min, eV", value=float(p.get("fdmnes_energy_min", -5.0)), step=1.0)
    with e2:
        p["fdmnes_energy_step"] = st.number_input("Energy step, eV", value=float(p.get("fdmnes_energy_step", 0.2)), step=0.1, format="%.3f")
    with e3:
        p["fdmnes_energy_max"] = st.number_input("Energy max, eV", value=float(p.get("fdmnes_energy_max", 50.0)), step=5.0)

    st.subheader("Software settings")
    vasp_col, fdmnes_col, feff_col = st.columns(3)
    with vasp_col:
        st.markdown("### VASP")
        p["vasp_method"] = st.selectbox("Method", ["PBE", "GW"], index=safe_index(["PBE", "GW"], p.get("vasp_method", "PBE")))
        p["potcar_dir"] = st.text_input("POTCAR root on NERSC", value=p.get("potcar_dir", DEFAULT_NERSC_POTCAR_DIR))
    with fdmnes_col:
        st.markdown("### FDMNES")
        p["fdmnes_method"] = st.selectbox("Method", ["Green", "FDM"], index=safe_index(["Green", "FDM"], p.get("fdmnes_method", "Green")))
        p["fdmnes_scf"] = st.checkbox("SCF", value=bool(p.get("fdmnes_scf", False)))
        p["fdmnes_quadrupole"] = st.checkbox("Quadrupole", value=bool(p.get("fdmnes_quadrupole", False)))
        p["fdmnes_spinorbit"] = st.checkbox("Spinorbit", value=bool(p.get("fdmnes_spinorbit", False)))
    with feff_col:
        st.markdown("### FEFF")

    st.subheader("NERSC setup")
    x1, x2, x3, x4, x5 = st.columns([1.2, 1, 0.8, 1, 1.6])
    with x1:
        p["nersc_account"] = st.text_input("NERSC project/account", value=p.get("nersc_account", DEFAULT_NERSC_ACCOUNT), key="xas_account")
    with x2:
        p["nersc_queue"] = st.selectbox("Queue/QOS", ["regular", "debug", "premium", "shared"], index=safe_index(["regular", "debug", "premium", "shared"], p.get("nersc_queue", "regular")), key="xas_queue")
    with x3:
        p["nersc_nodes"] = st.number_input("Nodes", min_value=1, max_value=64, value=int(p.get("nersc_nodes", 1)), key="xas_nodes")
    with x4:
        p["nersc_walltime"] = st.text_input("Walltime", value=p.get("nersc_walltime", "05:00:00"), key="xas_walltime")
    with x5:
        p["nersc_email"] = st.text_input("Email, optional", value=p.get("nersc_email", ""), key="xas_email")

    if st.button("Generate XAS calculation inputs", type="primary"):
        try:
            xas_dir = Path(p["xas_output_dir"]) / "02_XAS"
            xas_result = generate_xas_calculation_folders(
                structure=structure,
                output_dir=xas_dir,
                absorber=p["xas_absorber"],
                edge=resolved_edge,
                vasp_method=p["vasp_method"],
                cluster_radius=p["cluster_radius"],
                fdmnes_method=p["fdmnes_method"],
                fdmnes_scf=p["fdmnes_scf"],
                fdmnes_quadrupole=p["fdmnes_quadrupole"],
                fdmnes_spinorbit=p["fdmnes_spinorbit"],
                fdmnes_energy_min=p["fdmnes_energy_min"],
                fdmnes_energy_step=p["fdmnes_energy_step"],
                fdmnes_energy_max=p["fdmnes_energy_max"],
                potcar_dir=p["potcar_dir"],
                account=p["nersc_account"],
                queue=p["nersc_queue"],
                nodes=p["nersc_nodes"],
                walltime=p["nersc_walltime"],
                email=p["nersc_email"],
            )
            st.session_state["last_xas_result"] = xas_result
            st.session_state["last_xas_dir"] = str(xas_dir)
        except Exception as exc:
            st.exception(exc)

    if st.session_state.get("last_xas_result"):
        xas_result = st.session_state["last_xas_result"]
        xas_dir = Path(st.session_state["last_xas_dir"])
        st.success("XAS calculation input folders generated.")
        for warning in xas_result.get("warnings", []):
            st.warning(warning)

        with st.expander("XAS input summary", expanded=False):
            st.json(xas_result)

        tabs = st.tabs(["VASP", "FDMNES", "FEFF"])
        folder_map = {
            "VASP": xas_dir / "VASP",
            "FDMNES": xas_dir / "FDMNES",
            "FEFF": xas_dir / "FEFF",
        }
        for tab, name in zip(tabs, ["VASP", "FDMNES", "FEFF"]):
            with tab:
                folder = folder_map[name]
                st.write(f"**{name} folder**")
                tree = file_tree(folder)
                if tree:
                    st.code(tree, language="text")
                else:
                    st.info(f"{name} folder not found yet.")
                if folder.exists():
                    st.download_button(
                        f"Download {name} folder ZIP",
                        data=zip_directory_bytes(folder),
                        file_name=f"{name}_inputs_{p['xas_absorber']}_{resolved_edge}.zip",
                        mime="application/zip",
                        key=f"download_{name}_zip",
                    )

# =============================================================================


st.subheader("Batch generation")
with st.expander("Generate input packages for multiple structures or reaction pathways", expanded=False):
    p = st.session_state.params
    p["batch_source"] = st.radio(
        "Batch source",
        ["Generate pathway from current structure setup", "Upload structure files or ZIP"],
        index=safe_index(["Generate pathway from current structure setup", "Upload structure files or ZIP"], p.get("batch_source", "Generate pathway from current structure setup")),
        horizontal=True,
    )
    p["batch_output_dir"] = st.text_input("Batch output directory", value=p.get("batch_output_dir", "generated_outputs/web_xas_agent/batch"))

    st.markdown("#### NERSC project for generated batch scripts")
    bacc1, bacc2, bacc3 = st.columns([1.2, 1.0, 1.0])
    with bacc1:
        p["nersc_account"] = st.text_input(
            "NERSC project/account",
            value=p.get("nersc_account", DEFAULT_NERSC_ACCOUNT),
            key="batch_nersc_account",
            help="This value is written to #SBATCH -A in every generated submit script, including batch relaxation, VASP, FDMNES, and FEFF jobs.",
        ).strip() or DEFAULT_NERSC_ACCOUNT
    with bacc2:
        p["nersc_queue"] = st.selectbox(
            "Queue/QOS",
            ["regular", "debug", "premium", "shared"],
            index=safe_index(["regular", "debug", "premium", "shared"], p.get("nersc_queue", "regular")),
            key="batch_nersc_queue",
        )
    with bacc3:
        p["nersc_walltime"] = st.text_input(
            "Walltime",
            value=p.get("nersc_walltime", "05:00:00"),
            key="batch_nersc_walltime",
        )

    batch_uploaded_files = None
    batch_adsorbates = []
    if p["batch_source"] == "Generate pathway from current structure setup":
        pathway_defaults = [ads for ads in ["clean", "CO2", "CO", "CHO", "OCCO", "COCO", "OH", "H"] if ads in ADSORBATE_CHOICES]
        stored_batch_ads = [ads for ads in p.get("batch_adsorbates", pathway_defaults) if ads in ADSORBATE_CHOICES]
        batch_adsorbates = st.multiselect(
            "Adsorbates / pathway structures",
            ADSORBATE_CHOICES,
            default=stored_batch_ads or pathway_defaults,
        )
        p["batch_adsorbates"] = batch_adsorbates
        st.caption("Each selected adsorbate is generated on the current sidebar system and written as one batch item.")
    else:
        batch_uploaded_files = st.file_uploader(
            "Upload multiple structure files or one ZIP containing POSCAR/CONTCAR/CIF/XYZ files",
            accept_multiple_files=True,
            key="batch_structure_uploads",
        )
        st.caption("For a full local folder, zip the folder first and upload the ZIP.")

    b1, b2, b3 = st.columns(3)
    with b1:
        p["batch_include_relaxation"] = st.checkbox("Include VASP relaxation inputs", value=bool(p.get("batch_include_relaxation", True)))
    with b2:
        p["batch_include_xas"] = st.checkbox("Include XAS inputs", value=bool(p.get("batch_include_xas", True)))
    with b3:
        p["batch_absorber_mode"] = st.selectbox(
            "Absorber selection",
            ["auto", "Use selected absorber if available"],
            index=safe_index(["auto", "Use selected absorber if available"], p.get("batch_absorber_mode", "auto")),
        )

    if st.button("Generate batch input package", type="primary"):
        try:
            if not p.get("nersc_account") or p.get("nersc_account") == "mXXXX":
                raise ValueError("Please enter a valid NERSC project/account, for example m5268, before generating batch submit scripts.")
            workspace = Path(p["batch_output_dir"]) / "_workspace"
            if p["batch_source"] == "Generate pathway from current structure setup":
                if not batch_adsorbates:
                    raise ValueError("Select at least one adsorbate/pathway item.")
                batch_items = build_pathway_batch_items(p, batch_adsorbates, workspace)
            else:
                if not batch_uploaded_files:
                    raise ValueError("Upload at least one structure file or ZIP.")
                batch_items = build_uploaded_batch_items(batch_uploaded_files, workspace)

            batch_result = generate_batch_input_package(
                batch_items=batch_items,
                output_dir=p["batch_output_dir"],
                params=p,
                include_relaxation=p["batch_include_relaxation"],
                include_xas=p["batch_include_xas"],
                absorber_mode=p["batch_absorber_mode"],
            )
            st.session_state["last_batch_result"] = batch_result
            st.session_state["last_generated_package_kind"] = "batch"
        except Exception as exc:
            st.exception(exc)

    if st.session_state.get("last_batch_result"):
        batch_result = st.session_state["last_batch_result"]
        st.success(f"Batch package generated for {batch_result['n_structures']} structures.")
        with st.expander("Batch manifest", expanded=False):
            st.json(batch_result)
        run_dir = Path(batch_result["run_dir"])
        if run_dir.exists():
            st.download_button(
                "Download batch ZIP",
                data=zip_directory_bytes(run_dir),
                file_name=f"{run_dir.name}.zip",
                mime="application/zip",
                key="download_batch_zip",
            )




# NERSC job control and ISAAC record conversion
# =============================================================================

st.divider()
st.subheader("NERSC agent")

p = st.session_state.params
p["nersc_enable"] = st.checkbox(
    "Enable NERSC submission and monitoring",
    value=bool(p.get("nersc_enable", False)),
)

if p["nersc_enable"]:
    if not NERSC_JOB_CONTROL_AVAILABLE:
        st.error(f"NERSC job-control module could not be imported: {NERSC_JOB_CONTROL_IMPORT_ERROR}")
    else:
        st.caption(
            "This uses the NERSC Superfacility API. Keep private keys out of the repository; "
            "use a local key path or environment variables."
        )

        def _nersc_client_from_settings():
            return NERSCSFAPIClient(
                client_id=p.get("sfapi_client_id") or None,
                private_key_path=p.get("sfapi_private_key_path") or None,
                system=p.get("nersc_system", "perlmutter"),
            )

        def _local_folder_options():
            opts = {}
            xas_dir = p.get("xas_output_dir")
            if xas_dir:
                opts["Full package root"] = xas_dir
            if st.session_state.get("last_relax_result"):
                opts["VASP relaxation folder"] = st.session_state["last_relax_result"].get("relax_folder")
            if st.session_state.get("last_xas_dir"):
                opts["Last XAS input root"] = st.session_state.get("last_xas_dir")
            if st.session_state.get("last_batch_result"):
                opts["Last batch package"] = st.session_state["last_batch_result"].get("run_dir")
            opts["Custom local folder"] = p.get("nersc_custom_local_dir", "")
            return opts

        with st.expander("Authentication", expanded=False):
            a1, a2, a3 = st.columns([1.1, 1.6, 1.0])
            with a1:
                p["nersc_system"] = st.text_input("System", value=p.get("nersc_system", "perlmutter"))
            with a2:
                p["sfapi_client_id"] = st.text_input(
                    "SF API client ID",
                    value=p.get("sfapi_client_id", ""),
                    type="password",
                )
            with a3:
                if st.button("Check Perlmutter status"):
                    try:
                        client = _nersc_client_from_settings()
                        st.session_state["nersc_status_result"] = client.status()
                    except Exception as exc:
                        st.session_state["nersc_status_result"] = {"error": str(exc)}
            p["sfapi_private_key_path"] = st.text_input(
                "Private key path on this computer",
                value=p.get("sfapi_private_key_path", ""),
                help="Example: ~/.ssh/nersc_sfapi_private_key.pem. You can also set NERSC_SFAPI_PRIVATE_KEY_PATH.",
            )
            if st.session_state.get("nersc_status_result"):
                st.json(st.session_state["nersc_status_result"])


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

        st.subheader("Upload input package")
        folder_options = _local_folder_options()
        upload_labels = list(folder_options.keys())
        p["nersc_upload_source"] = st.selectbox(
            "Local folder to upload",
            upload_labels,
            index=safe_index(upload_labels, p.get("nersc_upload_source", "Full package root")),
        )
        if p["nersc_upload_source"] == "Custom local folder":
            p["nersc_custom_local_dir"] = st.text_input(
                "Custom local folder path",
                value=p.get("nersc_custom_local_dir", ""),
            )
        selected_local_dir = folder_options.get(p["nersc_upload_source"], "")
        if p["nersc_upload_source"] == "Custom local folder":
            selected_local_dir = p.get("nersc_custom_local_dir", "")

        u1, u2 = st.columns([2.0, 1.0])
        with u1:
            p["nersc_remote_dir"] = st.text_input(
                "Remote run directory",
                value=p.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR),
            )
        with u2:
            overwrite_remote = st.checkbox("Overwrite remote folder", value=True)

        st.write(f"Selected local folder: `{selected_local_dir}`")
        if selected_local_dir and Path(selected_local_dir).is_dir():
            submit_scripts_preview = discover_submit_scripts(selected_local_dir)
            with st.expander("Working folder structure", expanded=False):
                st.caption("This local structure is mirrored under the remote run directory after upload. Submit scripts should include the sample subfolder for batch packages.")
                st.code(compact_directory_tree(selected_local_dir), language="text")
                if submit_scripts_preview:
                    st.markdown("Detected submit scripts:")
                    st.code("\n".join(submit_scripts_preview[:80]), language="text")
                else:
                    st.warning("No submit.sh or submit_relax.sh files were found in this local folder.")

        if st.button("Upload selected folder to NERSC", type="secondary"):
            try:
                if not selected_local_dir or not Path(selected_local_dir).is_dir():
                    raise ValueError(f"Local folder does not exist: {selected_local_dir}")
                client = _nersc_client_from_settings()
                upload_result = client.upload_directory(
                    local_dir=selected_local_dir,
                    remote_dir=p["nersc_remote_dir"],
                    overwrite=overwrite_remote,
                    timeout_s=240,
                )
                st.session_state["nersc_last_upload"] = upload_result
                st.session_state["nersc_job_registry"] = st.session_state.get("nersc_job_registry", [])
                st.session_state["nersc_job_registry"].append({
                    "event": "upload",
                    "remote_dir": p["nersc_remote_dir"],
                    "local_dir": selected_local_dir,
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "result": upload_result,
                })
                p["nersc_download_remote_dir"] = p["nersc_remote_dir"]
            except Exception as exc:
                st.exception(exc)
        if st.session_state.get("nersc_last_upload"):
            upload_ok, upload_label = classify_upload_result(st.session_state["nersc_last_upload"])
            if upload_ok:
                st.success(upload_label)
            else:
                st.error(upload_label)
            with st.expander("Upload details", expanded=False):
                st.json(st.session_state["nersc_last_upload"])

        st.subheader("Submit job")
        detected_submit_scripts = discover_submit_scripts(selected_local_dir) if selected_local_dir else []
        submit_defaults = detected_submit_scripts or [
            "01_structure/submit_relax.sh",
            "02_XAS/VASP/submit.sh",
            "02_XAS/FDMNES/submit.sh",
            "02_XAS/FEFF/submit.sh",
        ]
        if "custom" not in submit_defaults:
            submit_defaults = submit_defaults + ["custom"]
        if detected_submit_scripts:
            st.info("Submit paths were detected from the selected local folder. For batch packages, choose one under a sample folder, e.g. Cu_111_CO/01_structure/submit_relax.sh.")
        submit_choice = st.selectbox(
            "Remote submit script",
            submit_defaults,
            index=safe_index(submit_defaults, p.get("nersc_remote_submit_script", submit_defaults[0]))
            if p.get("nersc_remote_submit_script") in submit_defaults else 0,
        )
        if submit_choice == "custom":
            p["nersc_remote_submit_script"] = st.text_input(
                "Remote submit script path or path relative to remote directory",
                value=p.get("nersc_remote_submit_script", "submit.sh"),
            )
        else:
            p["nersc_remote_submit_script"] = submit_choice
        remote_submit = p["nersc_remote_submit_script"]
        if not remote_submit.startswith("/") and not remote_submit.startswith("$"):
            remote_submit = p["nersc_remote_dir"].rstrip("/") + "/" + remote_submit
        st.write(f"Submit script resolved to: `{remote_submit}`")
        if detected_submit_scripts and "/" not in str(p.get("nersc_remote_submit_script", "")):
            st.warning("For a batch upload, the submit script must include the sample folder name. Use the detected list above instead of a root-level path.")

        s1, s2 = st.columns(2)
        with s1:
            if st.button("Submit selected job", type="primary"):
                try:
                    client = _nersc_client_from_settings()
                    submit_result = client.submit_and_wait_for_jobid(remote_submit, timeout_s=180)
                    st.session_state["nersc_last_submit"] = submit_result
                    if submit_result.get("task_id"):
                        p["nersc_task_id"] = str(submit_result["task_id"])
                    if submit_result.get("job_id"):
                        p["nersc_job_id"] = str(submit_result["job_id"])
                    p["nersc_last_submit_dir"] = str(PurePosixPath(remote_submit).parent)
                    p["nersc_download_remote_dir"] = p["nersc_last_submit_dir"]
                    st.session_state["nersc_job_registry"] = st.session_state.get("nersc_job_registry", [])
                    st.session_state["nersc_job_registry"].append({
                        "event": "submit",
                        "remote_submit": remote_submit,
                        "task_id": submit_result.get("task_id"),
                        "job_id": submit_result.get("job_id"),
                        "timestamp": datetime.now().isoformat(timespec="seconds"),
                        "result": submit_result,
                    })
                except Exception as exc:
                    st.exception(exc)
        with s2:
            if st.session_state.get("nersc_last_submit"):
                st.success("Submit request completed. Check Slurm job status below.")
        if st.session_state.get("nersc_last_submit"):
            with st.expander("Submit result", expanded=False):
                st.json(st.session_state["nersc_last_submit"])

        st.subheader("Monitor job and logs")
        m1, m2 = st.columns(2)
        with m1:
            p["nersc_task_id"] = st.text_input("SF API task ID", value=str(p.get("nersc_task_id", "")), help="Leave blank for a new submission. This is auto-filled after you click Submit selected job; paste an old task_id only when recovering a previous submission.")
            if st.button("Check task"):
                try:
                    client = _nersc_client_from_settings()
                    task_result = client.task(p["nersc_task_id"])
                    st.session_state["nersc_last_task"] = task_result
                    parsed = client.task_result_json(task_result)
                    if parsed.get("jobid"):
                        p["nersc_job_id"] = str(parsed["jobid"])
                except Exception as exc:
                    st.exception(exc)
        with m2:
            p["nersc_job_id"] = st.text_input("Slurm job ID", value=str(p.get("nersc_job_id", "")), help="Auto-filled from the SF API task result when Slurm accepts the job. You can also paste a known job ID manually.")
            if st.button("Check Slurm job"):
                try:
                    client = _nersc_client_from_settings()
                    st.session_state["nersc_last_slurm"] = client.slurm_status(p["nersc_job_id"], timeout_s=90)
                except Exception as exc:
                    st.exception(exc)
        if st.button("List my current/recent jobs"):
            try:
                client = _nersc_client_from_settings()
                st.session_state["nersc_user_jobs"] = client.list_user_jobs(username="hjia", timeout_s=90)
            except Exception as exc:
                st.exception(exc)
        if st.session_state.get("nersc_user_jobs"):
            with st.expander("Current/recent jobs for hjia", expanded=True):
                st.code(st.session_state["nersc_user_jobs"].get("output", json.dumps(st.session_state["nersc_user_jobs"], indent=2)), language="text")

        if st.session_state.get("nersc_last_task"):
            with st.expander("Task status", expanded=False):
                st.json(st.session_state["nersc_last_task"])
        if st.session_state.get("nersc_last_slurm"):
            st.code(st.session_state["nersc_last_slurm"].get("output", json.dumps(st.session_state["nersc_last_slurm"], indent=2)), language="text")

        st.subheader("Remote folder browser")
        submitted_dir_guess = p.get("nersc_last_submit_dir") or p.get("nersc_download_remote_dir") or p.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)
        p["nersc_preview_dir"] = st.text_input(
            "Remote folder to list/preview",
            value=p.get("nersc_preview_dir", submitted_dir_guess),
            help="After submission this defaults to the folder containing the selected submit script.",
        )
        if st.button("List remote folder"):
            try:
                client = _nersc_client_from_settings()
                st.session_state["nersc_last_ls"] = client.remote_ls(p["nersc_preview_dir"], timeout_s=90)
            except Exception as exc:
                st.exception(exc)
        if st.session_state.get("nersc_last_ls"):
            ls_output = st.session_state["nersc_last_ls"].get("output", json.dumps(st.session_state["nersc_last_ls"], indent=2))
            st.code(ls_output, language="text")
            file_choices = parse_remote_ls_files(ls_output)
            if file_choices:
                p["nersc_tail_file"] = st.selectbox(
                    "File to preview from listed folder",
                    file_choices,
                    index=safe_index(file_choices, p.get("nersc_tail_file", file_choices[0])),
                )
            else:
                p["nersc_tail_file"] = st.text_input("Remote log/output file to preview", value=p.get("nersc_tail_file", "vasp.out"))
        else:
            p["nersc_tail_file"] = st.text_input("Remote log/output file to preview", value=p.get("nersc_tail_file", "vasp.out"))

        tail_lines = st.number_input("Tail lines", min_value=20, max_value=500, value=80, step=20)
        remote_tail_file = p["nersc_tail_file"]
        if not remote_tail_file.startswith("/") and not remote_tail_file.startswith("$"):
            remote_tail_file = p.get("nersc_preview_dir", p["nersc_remote_dir"]).rstrip("/") + "/" + remote_tail_file
        st.write(f"Preview file resolved to: `{remote_tail_file}`")
        if st.button("Preview remote output"):
            try:
                client = _nersc_client_from_settings()
                st.session_state["nersc_last_tail"] = client.remote_tail(remote_tail_file, n_lines=int(tail_lines), timeout_s=90)
            except Exception as exc:
                st.exception(exc)
        if st.session_state.get("nersc_last_tail"):
            st.code(st.session_state["nersc_last_tail"].get("output", json.dumps(st.session_state["nersc_last_tail"], indent=2)), language="text")

        st.subheader("Download outputs")
        d1, d2 = st.columns([2, 2])
        with d1:
            p["nersc_download_remote_dir"] = st.text_input(
                "Remote output directory",
                value=p.get("nersc_download_remote_dir", p.get("nersc_remote_dir", DEFAULT_NERSC_REMOTE_RUN_DIR)),
            )
        with d2:
            p["nersc_download_glob"] = st.text_input(
                "Files to include",
                value=p.get("nersc_download_glob", "*.out OUTCAR OSZICAR CONTCAR vasprun.xml xmu.dat *_conv.txt CORE_DIELECTRIC_IMAG.dat isaac_run_metadata.json isaac_record_draft.json"),
            )
        if st.button("Fetch output archive from NERSC"):
            try:
                client = _nersc_client_from_settings()
                archive_bytes, archive_meta = client.download_remote_archive(
                    remote_dir=p["nersc_download_remote_dir"],
                    include_glob=p["nersc_download_glob"],
                    timeout_s=240,
                )
                st.session_state["nersc_download_archive_bytes"] = archive_bytes
                st.session_state["nersc_download_archive_meta"] = archive_meta
                local_extract_dir = p.get("isaac_local_output_dir", "generated_outputs/web_xas_agent/downloaded_outputs")
                extracted = extract_archive_bytes(archive_bytes, local_extract_dir, clean=True)
                st.session_state["nersc_last_download_extract_dir"] = extracted
                p["isaac_local_output_dir"] = extracted
            except Exception as exc:
                st.exception(exc)
        if st.session_state.get("nersc_download_archive_bytes"):
            st.download_button(
                "Download fetched output archive",
                data=st.session_state["nersc_download_archive_bytes"],
                file_name="nersc_outputs.tar.gz",
                mime="application/gzip",
            )
            with st.expander("Download metadata", expanded=False):
                st.json(st.session_state.get("nersc_download_archive_meta", {}))

        st.subheader("ISAAC record conversion")
        i1, i2, i3 = st.columns([2, 1, 1])
        with i1:
            p["isaac_local_output_dir"] = st.text_input(
                "Local output folder for ISAAC conversion",
                value=p.get("isaac_local_output_dir", "generated_outputs/web_xas_agent/downloaded_outputs"),
            )
        with i2:
            p["isaac_software"] = st.selectbox(
                "Software",
                ["auto", "VASP", "FDMNES", "FEFF"],
                index=safe_index(["auto", "VASP", "FDMNES", "FEFF"], p.get("isaac_software", "auto")),
            )
        with i3:
            p["isaac_sample_name"] = st.text_input("Sample name", value=p.get("isaac_sample_name", "not_specified"))
        if st.button("Convert outputs to ISAAC record"):
            try:
                record = build_isaac_record_from_outputs(
                    output_dir=p["isaac_local_output_dir"],
                    software=p["isaac_software"],
                    sample_name=p["isaac_sample_name"],
                    notes="Generated from CO2RR XAS Agent NERSC output folder.",
                )
                record_path = write_isaac_record(p["isaac_local_output_dir"], record)
                st.session_state["last_isaac_record"] = record
                st.session_state["last_isaac_record_path"] = record_path
            except Exception as exc:
                st.exception(exc)
        if st.session_state.get("last_isaac_record"):
            st.success("ISAAC record draft generated.")
            with st.expander("ISAAC record preview", expanded=False):
                st.json(st.session_state["last_isaac_record"])
            record_path = Path(st.session_state["last_isaac_record_path"])
            if record_path.exists():
                st.download_button(
                    "Download ISAAC record JSON",
                    data=record_path.read_text(),
                    file_name="isaac_record_draft.json",
                    mime="application/json",
                )

        if st.session_state.get("nersc_job_registry"):
            with st.expander("Local job registry", expanded=False):
                st.json(st.session_state["nersc_job_registry"])


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

