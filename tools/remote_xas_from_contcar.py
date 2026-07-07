#!/usr/bin/env python3
"""Regenerate XAS input folders from a relaxed CONTCAR on NERSC.

This script is intentionally non-Streamlit and can run inside an uploaded NERSC
workflow folder. It expects the CO2RR_XAS_agent repository to be available through
PYTHONPATH or CO2RR_AGENT_REPO. Typical workflow usage:

    python3 tools/remote_xas_from_contcar.py \
        --package-root . \
        --contcar 01_structure/CONTCAR \
        --absorber Cu \
        --edge K
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

REPO_ROOT = os.environ.get("CO2RR_AGENT_REPO")
if REPO_ROOT and REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
else:
    here = Path(__file__).resolve()
    repo = here.parents[1]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))

from tools.utils import read_structure_file, write_poscar  # noqa: E402
from tools.xas_input_generator import (  # noqa: E402
    FEFFInputGenerator,
    FDMNESInputGenerator,
    NERSCScriptGenerator,
    VASPXASInputGenerator,
)

DEFAULT_POTCAR_DIR = "/global/common/software/nersc9/vasp/dependencies/pseudopotentials/PBE/potpaw_PBE"


def sanitize_name(name: str, max_len: int = 64) -> str:
    text = re.sub(r"[^A-Za-z0-9_.+-]+", "_", str(name or "structure")).strip("._-")
    return (text or "structure")[:max_len]


def ordered_composition(atoms: Iterable[str]) -> Dict[str, int]:
    comp: Dict[str, int] = {}
    for atom in atoms:
        comp[atom] = comp.get(atom, 0) + 1
    return comp


def formula_from_composition(comp: Dict[str, int]) -> str:
    preferred = ["Cu", "Au", "Ag", "Ni", "Pt", "Pd", "Ir", "Rh", "Al", "C", "O", "H", "N"]
    keys = [k for k in preferred if k in comp] + sorted(k for k in comp if k not in preferred)
    parts = []
    for key in keys:
        count = comp[key]
        parts.append(f"{key}{count if count != 1 else ''}")
    return "".join(parts)


def load_structure_info(package_root: Path, structure: Dict) -> Dict:
    candidates = [
        package_root / "01_structure" / "structure_info.json",
        package_root / "structure_info.json",
    ]
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                pass
    atoms = structure.get("atoms", [])
    comp = ordered_composition(atoms)
    return {
        "structure_id": sanitize_name("post_relax_structure"),
        "composition": comp,
        "sample_formula": formula_from_composition(comp),
        "sample_form": "slab_model",
        "sample_notes": "Generated from relaxed CONTCAR by CO2RR XAS Agent workflow.",
        "n_atoms": len(atoms),
    }


def choose_absorber(structure: Dict, preferred: Optional[str]) -> str:
    atoms = list(dict.fromkeys(structure.get("atoms", [])))
    if preferred and preferred in atoms:
        return preferred
    for candidate in ["Cu", "Au", "Ni", "Pt", "Pd", "Ag", "Ir", "Rh", "C", "O"]:
        if candidate in atoms:
            return candidate
    non_h = [el for el in atoms if el != "H"]
    return non_h[0] if non_h else atoms[0]


def normalize_edge(absorber: str, edge: str) -> str:
    if edge and edge != "auto":
        return edge
    z = {
        "H": 1, "C": 6, "N": 7, "O": 8, "Al": 13, "Ni": 28, "Cu": 29,
        "Zn": 30, "Rh": 45, "Pd": 46, "Ag": 47, "Ir": 77, "Pt": 78, "Au": 79,
    }.get(absorber, 0)
    return "K" if z <= 30 else "L3"


def write_make_potcar(structure: Dict, output_dir: Path, potcar_dir: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    gen = VASPXASInputGenerator()
    lines = [
        "#!/bin/bash",
        "set -euo pipefail",
        f"POTCAR_DIR=\"${{CO2RR_POTCAR_DIR:-{potcar_dir}}}\"",
        "rm -f POTCAR",
    ]
    for symbol in gen.get_potcar_symbols(structure):
        lines.extend([
            f"test -f \"${{POTCAR_DIR}}/{symbol}/POTCAR\"",
            f"cat \"${{POTCAR_DIR}}/{symbol}/POTCAR\" >> POTCAR",
        ])
    lines.append("echo Wrote POTCAR")
    path = output_dir / "make_potcar.sh"
    path.write_text("\n".join(lines) + "\n")
    path.chmod(0o755)
    return path


def copy_structure_info(metadata: Dict, *dirs: Path) -> None:
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "structure_info.json").write_text(json.dumps(metadata, indent=2))


def patch_submit_metadata(submit_path: Path, metadata: Dict) -> None:
    if not submit_path.exists():
        return
    name = metadata.get("sample_name") or metadata.get("structure_id") or "post_relax_structure"
    form = metadata.get("sample_form") or "slab_model"
    formula = metadata.get("sample_formula") or formula_from_composition(metadata.get("composition", {})) or "not_specified"
    notes = metadata.get("sample_notes") or "Post-relaxation XAS input generated from CONTCAR."
    replacements = {
        "ISAAC_SAMPLE_NAME": name,
        "ISAAC_SAMPLE_FORM": form,
        "ISAAC_SAMPLE_NOTES": notes,
        "ISAAC_SAMPLE_FORMULA": formula,
    }
    text = submit_path.read_text()
    for key, value in replacements.items():
        safe = str(value).replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        text = re.sub(rf"^export\s+{key}=.*$", f'export {key}="${{{key}:-{safe}}}"', text, flags=re.M)
    submit_path.write_text(text)


def ensure_isaac_inline(submit_path: Path) -> None:
    # The existing app already injects a richer inline converter. This remote helper
    # only guarantees sample metadata. Record creation remains in generated scripts.
    if submit_path.exists():
        submit_path.chmod(0o755)


def regenerate_xas(args: argparse.Namespace) -> Dict:
    package_root = Path(args.package_root).resolve()
    contcar = (package_root / args.contcar).resolve() if not Path(args.contcar).is_absolute() else Path(args.contcar)
    if not contcar.exists():
        raise FileNotFoundError(f"Relaxed CONTCAR not found: {contcar}")

    structure = read_structure_file(str(contcar))
    structure.setdefault("metadata", {})
    metadata = load_structure_info(package_root, structure)
    metadata["xas_input_source"] = "post_relaxation_CONTCAR"
    metadata["relaxed_structure_path"] = str(contcar.relative_to(package_root) if contcar.is_relative_to(package_root) else contcar)
    metadata["composition"] = ordered_composition(structure.get("atoms", []))
    metadata["sample_formula"] = metadata.get("sample_formula") or formula_from_composition(metadata["composition"])
    metadata["n_atoms"] = len(structure.get("atoms", []))
    structure["metadata"].update(metadata)

    absorber = choose_absorber(structure, args.absorber)
    edge = normalize_edge(absorber, args.edge)
    xas_root = package_root / "02_XAS"
    if xas_root.exists() and not args.keep_existing:
        backup = package_root / f"02_XAS_pre_restart_backup"
        if backup.exists():
            shutil.rmtree(backup)
        shutil.move(str(xas_root), str(backup))
    xas_root.mkdir(parents=True, exist_ok=True)

    vasp_root = xas_root / "VASP"
    fdmnes_root = xas_root / "FDMNES"
    feff_root = xas_root / "FEFF"
    copy_structure_info(metadata, xas_root, vasp_root, fdmnes_root, feff_root)

    vasp = VASPXASInputGenerator()
    fdmnes = FDMNESInputGenerator()
    feff = FEFFInputGenerator()
    scripts = NERSCScriptGenerator()

    vasp_result = vasp.write_inputs(
        structure=structure,
        output_dir=str(vasp_root),
        absorber=absorber,
        method=args.vasp_method,
        potcar_dir=args.potcar_dir,
        account=args.account,
        queue=args.queue,
        nodes=args.nodes,
        walltime=args.walltime,
        email=args.email or None,
        job_name=sanitize_name(f"{metadata.get('structure_id', 'post_relax')}_XAS_VASP"),
    )
    for subdir in [vasp_root / "01_scf", vasp_root / "02_xas"]:
        write_make_potcar(structure, subdir, args.potcar_dir)
        spec = subdir / "POTCAR.spec"
        if spec.exists():
            spec.unlink()
    patch_submit_metadata(Path(vasp_result["submit"]), metadata)
    ensure_isaac_inline(Path(vasp_result["submit"]))

    fdmnes.write_input(
        structure=structure,
        output_dir=str(fdmnes_root),
        absorber=absorber,
        edge=edge,
        radius=args.cluster_radius,
        energy_range=(args.fdmnes_energy_min, args.fdmnes_energy_step, args.fdmnes_energy_max),
        options={"Green": args.fdmnes_method.lower().startswith("green"), "SCF": args.fdmnes_scf},
    )
    fdmnes_submit = Path(scripts.write_script(
        str(fdmnes_root),
        job_name=sanitize_name(f"{metadata.get('structure_id', 'post_relax')}_XAS_FDMNES"),
        software="FDMNES",
        account=args.account,
        queue=args.queue,
        nodes=args.nodes,
        walltime=args.walltime,
        email=args.email or None,
    ))
    patch_submit_metadata(fdmnes_submit, metadata)
    ensure_isaac_inline(fdmnes_submit)

    feff.write_input(structure=structure, output_dir=str(feff_root), absorber=absorber, edge=edge, radius=args.cluster_radius)
    feff_submit = Path(scripts.write_script(
        str(feff_root),
        job_name=sanitize_name(f"{metadata.get('structure_id', 'post_relax')}_XAS_FEFF"),
        software="FEFF",
        account=args.account,
        queue=args.queue,
        nodes=args.nodes,
        walltime=args.walltime,
        email=args.email or None,
    ))
    patch_submit_metadata(feff_submit, metadata)
    ensure_isaac_inline(feff_submit)

    marker = xas_root / ".post_relax_generated"
    marker.write_text(json.dumps({
        "status": "generated",
        "source": str(contcar),
        "absorber": absorber,
        "edge": edge,
        "xas_root": str(xas_root),
    }, indent=2))
    return {"status": "success", "xas_root": str(xas_root), "absorber": absorber, "edge": edge}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", default=".")
    p.add_argument("--contcar", default="01_structure/CONTCAR")
    p.add_argument("--absorber", default="")
    p.add_argument("--edge", default="K")
    p.add_argument("--vasp-method", default="PBE")
    p.add_argument("--cluster-radius", type=float, default=6.0)
    p.add_argument("--fdmnes-method", default="Green")
    p.add_argument("--fdmnes-scf", action="store_true")
    p.add_argument("--fdmnes-energy-min", type=float, default=-5.0)
    p.add_argument("--fdmnes-energy-step", type=float, default=0.2)
    p.add_argument("--fdmnes-energy-max", type=float, default=50.0)
    p.add_argument("--potcar-dir", default=DEFAULT_POTCAR_DIR)
    p.add_argument("--account", default=os.environ.get("CO2RR_NERSC_ACCOUNT", os.environ.get("NERSC_ACCOUNT", "m5268")))
    p.add_argument("--queue", default="regular")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--walltime", default="02:00:00")
    p.add_argument("--email", default="")
    p.add_argument("--keep-existing", action="store_true")
    args = p.parse_args()
    print(json.dumps(regenerate_xas(args), indent=2))


if __name__ == "__main__":
    main()
