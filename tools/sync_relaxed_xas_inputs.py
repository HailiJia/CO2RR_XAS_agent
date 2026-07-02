"""Synchronize XAS input folders with a completed relaxation CONTCAR.

The web app generates the initial layout as::

    sample/
      01_structure/       # VASP relaxation
      02_XAS/
        VASP/
        FDMNES/
        FEFF/

Because the XAS folders are generated before the relaxation finishes, they may
still contain the initial/unrelaxed structure. This module updates the XAS input
folders from ``01_structure/CONTCAR`` after the relaxation has completed.

Usage from the repository root::

    python -m tools.sync_relaxed_xas_inputs /path/to/sample/01_structure
    python -m tools.sync_relaxed_xas_inputs /path/to/sample

Updated outputs:
- ``02_XAS/VASP/01_scf`` and ``02_XAS/VASP/02_xas`` POSCAR/KPOINTS/INCAR are
  rewritten from the relaxed structure.
- ``02_XAS/FEFF`` HEADER/PARAMETERS/POTENTIALS/ATOMS/feff.inp are regenerated
  using the existing absorber/edge/radius parsed from the old input.
- ``02_XAS/FDMNES`` fdmfile.txt and ``*_in.txt`` are regenerated using the
  existing absorber/edge/radius/range/options parsed from the old input.
- ``02_XAS/relaxed_CONTCAR_for_xas`` records the exact relaxed structure used.
- ``02_XAS/xas_structure_sync.json`` records what was updated.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .utils import ATOMIC_NUMBERS, METALS_3D, METALS_4D, METALS_5D, read_poscar, write_poscar
from .xas_input_generator import FEFFInputGenerator, FDMNESInputGenerator, VASPXASInputGenerator

GENERATOR_VERSION = "v37_2026-07-02_post_relax_xas_structure_sync"
Z_TO_ELEMENT = {z: el for el, z in ATOMIC_NUMBERS.items()}
METAL_PRIORITY = ["Cu", "Au", "Ni", "Pt", "Pd", "Ag", "Ir", "Rh"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sample_dir_from_path(path: Path) -> Path:
    path = path.resolve()
    if path.name == "01_structure":
        return path.parent
    if path.name == "02_XAS":
        return path.parent
    if path.name in {"VASP", "FEFF", "FDMNES"} and path.parent.name == "02_XAS":
        return path.parent.parent
    return path


def choose_absorber(structure: Dict[str, Any], preferred: Optional[str] = None) -> str:
    atoms = list(structure.get("atoms", []))
    if preferred and preferred in atoms:
        return preferred
    for candidate in METAL_PRIORITY:
        if candidate in atoms:
            return candidate
    for atom in atoms:
        if atom in METALS_3D + METALS_4D + METALS_5D:
            return atom
    for atom in atoms:
        if atom != "H":
            return atom
    return atoms[0] if atoms else "Cu"


def default_edge_for_absorber(absorber: str) -> str:
    z = ATOMIC_NUMBERS.get(absorber, 0)
    return "K" if z <= 30 else "L3"


def backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".pre_relax_sync.bak") if path.suffix else path.with_name(path.name + ".pre_relax_sync.bak")


def maybe_backup(path: Path, enabled: bool) -> Optional[str]:
    if not enabled or not path.exists() or not path.is_file():
        return None
    dst = backup_path(path)
    shutil.copy2(path, dst)
    return str(dst)


def parse_feff_settings(feff_dir: Path, structure: Dict[str, Any]) -> Dict[str, Any]:
    text = ""
    for name in ["feff.inp", "HEADER", "PARAMETERS"]:
        p = feff_dir / name
        if p.exists():
            text += "\n" + p.read_text(errors="ignore")

    absorber = None
    edge = None
    radius = None

    match = re.search(r"absorber\s*=\s*([A-Z][a-z]?)", text)
    if match:
        absorber = match.group(1)
    match = re.search(r"edge\s*=\s*([A-Za-z0-9]+)", text)
    if match:
        edge = match.group(1).upper()
    match = re.search(r"radius\s*=\s*([-+0-9.]+)", text)
    if match:
        radius = float(match.group(1))

    # Fall back to FEFF cards if the TITLE line is missing.
    if edge is None:
        match = re.search(r"^\s*EDGE\s+([A-Za-z0-9]+)", text, flags=re.M)
        if match:
            edge = match.group(1).upper()
    if radius is None:
        for card in ["FMS", "SCF"]:
            match = re.search(rf"^\s*{card}\s+([-+0-9.]+)", text, flags=re.M)
            if match:
                radius = float(match.group(1))
                break

    absorber = choose_absorber(structure, absorber)
    edge = edge or default_edge_for_absorber(absorber)
    radius = radius or 6.0
    return {"absorber": absorber, "edge": edge, "radius": radius}


def parse_keyword_next_value(lines: List[str], keyword: str) -> Optional[str]:
    key = keyword.lower()
    for i, line in enumerate(lines):
        if line.strip().lower() == key:
            for j in range(i + 1, len(lines)):
                value = lines[j].strip()
                if value:
                    return value
    return None


def parse_fdmnes_settings(fdmnes_dir: Path, structure: Dict[str, Any]) -> Dict[str, Any]:
    input_file = None
    fdmfile = fdmnes_dir / "fdmfile.txt"
    if fdmfile.exists():
        entries = [x.strip() for x in fdmfile.read_text(errors="ignore").splitlines() if x.strip()]
        if len(entries) >= 2:
            input_file = entries[1]

    if not input_file:
        candidates = sorted(fdmnes_dir.glob("*_in.txt"))
        input_file = candidates[0].name if candidates else "Cu_in.txt"

    input_path = fdmnes_dir / input_file
    lines = input_path.read_text(errors="ignore").splitlines() if input_path.exists() else []
    text = "\n".join(lines)

    absorber = None
    z_text = parse_keyword_next_value(lines, "Z_Absorber")
    if z_text:
        try:
            absorber = Z_TO_ELEMENT.get(int(float(z_text.split()[0])))
        except Exception:
            absorber = None

    edge = parse_keyword_next_value(lines, "Edge") or None
    if edge:
        edge = edge.split()[0].upper()

    radius = 6.0
    radius_text = parse_keyword_next_value(lines, "Radius")
    if radius_text:
        try:
            radius = float(radius_text.split()[0])
        except Exception:
            pass

    energy_range = (-5.0, 0.2, 50.0)
    range_text = parse_keyword_next_value(lines, "Range")
    if range_text:
        parts = range_text.split()
        if len(parts) >= 3:
            try:
                energy_range = (float(parts[0]), float(parts[1]), float(parts[2]))
            except Exception:
                pass

    options = {}
    for key in FDMNESInputGenerator.OPTION_ORDER:
        options[key] = bool(re.search(rf"^\s*{re.escape(key)}\s*$", text, flags=re.M))

    absorber = choose_absorber(structure, absorber)
    edge = edge or default_edge_for_absorber(absorber)
    return {
        "absorber": absorber,
        "edge": edge,
        "radius": radius,
        "energy_range": energy_range,
        "options": options,
    }


def parse_vasp_xas_settings(vasp_dir: Path, structure: Dict[str, Any]) -> Dict[str, Any]:
    text = ""
    for p in [vasp_dir / "02_xas" / "INCAR", vasp_dir / "01_scf" / "INCAR"]:
        if p.exists():
            text += "\n" + p.read_text(errors="ignore")

    absorber = None
    method = "PBE"
    absorber_index = 0
    match = re.search(r"^\s*#\s*Absorber:\s*([A-Z][a-z]?)", text, flags=re.M)
    if match:
        absorber = match.group(1)
    match = re.search(r"step\s+2\s+([A-Za-z0-9]+)-based\s+XAS", text)
    if match:
        method = match.group(1).upper()
    match = re.search(r"^\s*CLNT\s*=\s*([0-9]+)", text, flags=re.M)
    if match:
        absorber_index = max(0, int(match.group(1)) - 1)

    absorber = choose_absorber(structure, absorber)
    return {"absorber": absorber, "method": method, "absorber_index": absorber_index}


def write_vasp_xas_from_relaxed(vasp_dir: Path, structure: Dict[str, Any], backup: bool) -> List[str]:
    if not vasp_dir.exists():
        return []
    gen = VASPXASInputGenerator()
    settings = parse_vasp_xas_settings(vasp_dir, structure)
    updated: List[str] = []

    for step_name, incar_content in [
        ("01_scf", gen.generate_incar_scf(structure, method=settings["method"])),
        ("02_xas", gen.generate_incar_xas(
            structure,
            absorber=settings["absorber"],
            method=settings["method"],
            absorber_index=settings["absorber_index"],
        )),
    ]:
        step_dir = vasp_dir / step_name
        if not step_dir.exists():
            continue
        for path in [step_dir / "POSCAR", step_dir / "KPOINTS", step_dir / "INCAR", step_dir / "POTCAR.spec", step_dir / "make_potcar.sh"]:
            maybe_backup(path, backup)
        write_poscar(structure["atoms"], structure["positions"], structure["cell"], str(step_dir / "POSCAR"), selective_dynamics=False)
        (step_dir / "KPOINTS").write_text(gen.generate_kpoints(structure) + "\n")
        (step_dir / "INCAR").write_text(incar_content + "\n")
        (step_dir / "POTCAR.spec").write_text(gen.generate_potcar_spec(structure, method=settings["method"]) + "\n")
        make_potcar = step_dir / "make_potcar.sh"
        make_potcar.write_text(gen.generate_make_potcar_script(structure, method=settings["method"]) + "\n")
        make_potcar.chmod(0o755)
        updated.extend(str(p) for p in [step_dir / "POSCAR", step_dir / "KPOINTS", step_dir / "INCAR", step_dir / "POTCAR.spec", make_potcar])

    return updated


def write_feff_from_relaxed(feff_dir: Path, structure: Dict[str, Any], backup: bool) -> List[str]:
    if not feff_dir.exists():
        return []
    settings = parse_feff_settings(feff_dir, structure)
    for name in ["HEADER", "PARAMETERS", "POTENTIALS", "ATOMS", "feff.inp"]:
        maybe_backup(feff_dir / name, backup)
    gen = FEFFInputGenerator()
    gen.write_input(
        structure=structure,
        output_dir=str(feff_dir),
        absorber=settings["absorber"],
        edge=settings["edge"],
        radius=settings["radius"],
    )
    return [str(feff_dir / name) for name in ["HEADER", "PARAMETERS", "POTENTIALS", "ATOMS", "feff.inp"]]


def write_fdmnes_from_relaxed(fdmnes_dir: Path, structure: Dict[str, Any], backup: bool) -> List[str]:
    if not fdmnes_dir.exists():
        return []
    settings = parse_fdmnes_settings(fdmnes_dir, structure)
    for path in list(fdmnes_dir.glob("*_in.txt")) + [fdmnes_dir / "fdmfile.txt"]:
        maybe_backup(path, backup)
    gen = FDMNESInputGenerator()
    files = gen.write_input(
        structure=structure,
        output_dir=str(fdmnes_dir),
        absorber=settings["absorber"],
        edge=settings["edge"],
        radius=settings["radius"],
        energy_range=settings["energy_range"],
        options=settings["options"],
    )
    return sorted(str(p) for p in files.values())


def sync_relaxed_xas_inputs(path: str, backup: bool = True) -> Dict[str, Any]:
    sample_dir = sample_dir_from_path(Path(path))
    structure_dir = sample_dir / "01_structure"
    xas_dir = sample_dir / "02_XAS"
    contcar = structure_dir / "CONTCAR"

    if not contcar.exists():
        raise FileNotFoundError(f"Relaxed CONTCAR not found: {contcar}")
    if not xas_dir.exists():
        raise FileNotFoundError(f"XAS folder not found: {xas_dir}")

    structure = read_poscar(str(contcar))
    structure.setdefault("metadata", {})["source_file"] = str(contcar)
    structure["metadata"]["source"] = "post_relax_CONTCAR_sync"

    relaxed_copy = xas_dir / "relaxed_CONTCAR_for_xas"
    maybe_backup(relaxed_copy, backup)
    shutil.copy2(contcar, relaxed_copy)

    updated: Dict[str, List[str]] = {
        "VASP": write_vasp_xas_from_relaxed(xas_dir / "VASP", structure, backup),
        "FEFF": write_feff_from_relaxed(xas_dir / "FEFF", structure, backup),
        "FDMNES": write_fdmnes_from_relaxed(xas_dir / "FDMNES", structure, backup),
    }

    manifest = {
        "status": "success",
        "generator": "CO2RR XAS Agent",
        "version": GENERATOR_VERSION,
        "updated_utc": utc_now(),
        "sample_dir": str(sample_dir),
        "relaxed_structure_source": str(contcar),
        "relaxed_structure_copy": str(relaxed_copy),
        "updated_files": updated,
        "note": "XAS input structures regenerated from 01_structure/CONTCAR after relaxation.",
    }
    manifest_path = xas_dir / "xas_structure_sync.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Update 02_XAS input structures from 01_structure/CONTCAR.")
    parser.add_argument("path", help="sample directory, 01_structure directory, or 02_XAS directory")
    parser.add_argument("--no-backup", action="store_true", help="do not create *.pre_relax_sync.bak backups before overwriting")
    args = parser.parse_args()
    manifest = sync_relaxed_xas_inputs(args.path, backup=not args.no_backup)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
