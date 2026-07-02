"""Schema-compliant ISAAC record builders for CO2RR XAS Agent outputs.

This module converts generated VASP relaxation folders and XAS output folders
into ISAAC evidence records that validate against ISAAC record version 1.05.

Typical use from a completed folder:

    python -m tools.isaac_record_builder /path/to/01_structure
    python -m tools.isaac_record_builder /path/to/02_XAS/FEFF

Relaxation records are DFT/computation evidence records with a non-null
``energy_raw`` descriptor. XAS records follow the simulation XAS evidence shape:
``measurement.series`` carries the spectrum, and ``descriptors.outputs`` carries
auto-extracted spectral features. Adsorption-energy descriptors are added only
for non-clean adsorbate systems when a matching clean relaxation energy can be
found.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ISAAC_RECORD_VERSION = "1.05"
GENERATOR_NAME = "CO2RR XAS Agent"
GENERATOR_VERSION = "v36_2026-07-02_schema_valid_isaac_records"
DATA_OWNER = "hailijia"


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "not_available"


def media_type(path: Path) -> str:
    lower = path.name.lower()
    if lower.endswith(".json"):
        return "application/json"
    if lower.endswith(".xml"):
        return "application/xml"
    if lower.endswith((".tar.gz", ".tgz")):
        return "application/gzip"
    if lower.endswith(".zip"):
        return "application/zip"
    if lower.endswith(".xyz"):
        return "chemical/x-xyz"
    return "text/plain"


def parse_first_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        return float(str(value).replace(",", " ").split()[0])
    except Exception:
        return default


def parse_first_int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        return int(float(str(value).replace(",", " ").split()[0]))
    except Exception:
        return default


def vasp_bool(value: Any) -> bool:
    return str(value).strip().upper() in {".TRUE.", "TRUE", "T"}


def parse_incar(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in read_text(path).splitlines():
        line = line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip().upper()] = value.strip()
    return data


def read_kpoints(path: Path) -> str:
    lines = [x.strip() for x in read_text(path).splitlines() if x.strip()]
    if len(lines) >= 4:
        mesh = lines[3].split()
        if len(mesh) >= 3:
            style = lines[2].split()[0] if len(lines) >= 3 else "Gamma"
            return f"{style} {mesh[0]}x{mesh[1]}x{mesh[2]}"
    return "not_specified"


def smearing_name(ismear: Optional[int]) -> str:
    return {
        -5: "tetrahedron",
        -1: "Fermi",
        0: "Gaussian",
        1: "Methfessel-Paxton",
        2: "Methfessel-Paxton",
    }.get(ismear, "not_specified")


def dispersion_name(incar: Dict[str, str]) -> str:
    ivdw = parse_first_int(incar.get("IVDW"), None)
    if ivdw in {11, 12}:
        return "D3"
    if ivdw is None:
        return "none"
    # The current ISAAC vocabulary is closed; keep unknown IVDW values out of the
    # vocabulary field and preserve details in computation.method.notes if needed.
    return "none"


def sample_name_from_folder(root: Path) -> str:
    if root.name in {"01_structure", "02_XAS", "VASP", "FDMNES", "FEFF", "01_scf", "02_xas"}:
        return root.parent.name if root.name == "01_structure" else root.parents[1].name if len(root.parents) > 1 else root.name
    return root.name


def structure_info_candidates(root: Path) -> Iterable[Path]:
    yield root / "structure_info.json"
    for parent in [root.parent, root.parent.parent, root.parent.parent.parent]:
        yield parent / "01_structure" / "structure_info.json"
        yield parent / "structure_info.json"


def load_structure_info(root: Path) -> Dict[str, Any]:
    for path in structure_info_candidates(root):
        info = read_json(path)
        if info:
            return info
    return {}


def formula_from_structure_info(info: Dict[str, Any]) -> str:
    catalyst = info.get("catalyst") if isinstance(info.get("catalyst"), dict) else {}
    if catalyst.get("composition"):
        return str(catalyst["composition"])
    if info.get("element"):
        return str(info["element"])
    comp = info.get("composition") if isinstance(info.get("composition"), dict) else {}
    if comp:
        parts = []
        for elem, count in comp.items():
            try:
                n = int(count)
            except Exception:
                n = count
            parts.append(str(elem) if n == 1 else f"{elem}{n}")
        return "".join(parts)
    return "not_specified"


def adsorbate_identity(info: Dict[str, Any]) -> str:
    ads = info.get("adsorbate_metadata") if isinstance(info.get("adsorbate_metadata"), dict) else {}
    return str(ads.get("identity") or "clean")


def final_energy_from_outputs(root: Path) -> Tuple[float, str]:
    energy = None
    outcar = root / "OUTCAR"
    for line in read_text(outcar).splitlines():
        if "free  energy   TOTEN" in line:
            try:
                energy = float(line.split("=")[1].split()[0])
            except Exception:
                pass
    if energy is not None:
        return energy, "OUTCAR"

    oszicar = root / "OSZICAR"
    for line in reversed(read_text(oszicar).splitlines()):
        match = re.search(r"F=\s*([-+0-9.Ee]+)", line)
        if match:
            return float(match.group(1)), "OSZICAR"

    raise RuntimeError(f"Could not parse final VASP energy from {root}/OUTCAR or OSZICAR")


def timestamps_from_metadata(root: Path) -> Dict[str, str]:
    metadata = read_json(root / "isaac_run_metadata.json")
    timestamps = metadata.get("timestamps") if isinstance(metadata.get("timestamps"), dict) else {}
    created = utc_now()
    return {
        "created_utc": created,
        "acquired_start_utc": timestamps.get("acquired_start_utc", created),
        "acquired_end_utc": timestamps.get("acquired_end_utc", created),
    }


def sanitize_id_token(text: Any, length: int) -> str:
    token = re.sub(r"[^0-9A-Z]+", "", str(text).upper())
    return token[:length].ljust(length, "0")


def relaxation_record_id(info: Dict[str, Any], sequence: int = 1) -> str:
    element = info.get("element") or formula_from_structure_info(info)
    facet = info.get("facet") or (info.get("catalyst") or {}).get("surface_facet") or "000"
    system_token = sanitize_id_token(f"{element}{facet}", 5)
    record_id = f"01JRGDFTEVID00{system_token}RELX{sequence:03d}"
    if not re.match(r"^[0-9A-Z]{26}$", record_id):
        raise ValueError(f"Generated invalid relaxation record_id: {record_id}")
    return record_id


def xas_record_id(sequence: int = 1) -> str:
    record_id = f"01JRG{'SIMXAS'}{sequence:015d}"
    if not re.match(r"^[0-9A-Z]{26}$", record_id):
        raise ValueError(f"Generated invalid XAS record_id: {record_id}")
    return record_id


def base_evidence_header(record_id: str, timestamps: Dict[str, str]) -> Dict[str, Any]:
    return {
        "isaac_record_version": ISAAC_RECORD_VERSION,
        "record_id": record_id,
        "record_type": "evidence",
        "record_domain": "simulation",
        "source_type": "computation",
        "timestamps": timestamps,
        "attribution": {"contributors": [{"name": DATA_OWNER, "role": "data_owner"}]},
    }


def computation_method(root: Path, incar: Dict[str, str], energy_source: str = "OUTCAR") -> Dict[str, Any]:
    ediffg = parse_first_float(incar.get("EDIFFG"), None)
    return {
        "family": "DFT",
        "functional_class": "GGA",
        "functional_name": "PBE",
        "basis_type": "planewave",
        "pseudopotential": "PAW",
        "cutoff_eV": parse_first_float(incar.get("ENCUT"), None),
        "spin_treatment": "collinear" if parse_first_int(incar.get("ISPIN"), 1) == 2 else "none",
        "dispersion": dispersion_name(incar),
        "kpoints": read_kpoints(root / "KPOINTS"),
        "smearing": {
            "method": smearing_name(parse_first_int(incar.get("ISMEAR"), None)),
            "width_eV": parse_first_float(incar.get("SIGMA"), None),
        },
        "convergence": {
            "energy_eV": parse_first_float(incar.get("EDIFF"), None),
            "force_eV_per_A": abs(ediffg) if ediffg is not None and ediffg < 0 else ediffg,
        },
        "notes": f"Energy source file: {energy_source}.",
    }


def slab_model(info: Dict[str, Any], incar: Dict[str, str]) -> Dict[str, Any]:
    supercell = info.get("supercell", "not_specified")
    if isinstance(supercell, list):
        supercell = "x".join(str(x) for x in supercell)
    return {
        "surface_facet": str(info.get("facet") or (info.get("catalyst") or {}).get("surface_facet") or "not_specified"),
        "supercell": str(supercell),
        "layers": info.get("layers", "not_specified"),
        "fixed_layers": "bottom 2",
        "vacuum_A": info.get("vacuum", "not_specified"),
        "lattice_constant_A": info.get("lattice_constant", "not_specified"),
        "dipole_correction": vasp_bool(incar.get("LDIPOL", "False")),
    }


def contcar_asset(root: Path) -> List[Dict[str, Any]]:
    contcar = root / "CONTCAR"
    if not contcar.exists():
        return []
    return [{
        "asset_id": "asset_000_CONTCAR",
        "content_role": "input_structure",
        "uri": "CONTCAR",
        "media_type": media_type(contcar),
        "sha256": sha256_file(contcar),
        "notes": "Relaxed VASP structure from CONTCAR.",
    }]


# -----------------------------------------------------------------------------
# Relaxation record
# -----------------------------------------------------------------------------


def build_relaxation_record(output_dir: str, sequence: int = 1) -> Dict[str, Any]:
    root = Path(output_dir).resolve()
    info = load_structure_info(root)
    incar = parse_incar(root / "INCAR")
    energy_eV, energy_source = final_energy_from_outputs(root)
    timestamps = timestamps_from_metadata(root)

    record = base_evidence_header(relaxation_record_id(info, sequence), timestamps)
    record.update({
        "sample": {
            "material": {
                "name": sample_name_from_folder(root),
                "formula": formula_from_structure_info(info),
                "provenance": "theoretical",
            },
            "sample_form": "slab_model",
        },
        "context": {
            "environment": "in_silico",
            "temperature_K": 0,
            "simulation_assumptions": {"structural_model": "Static 0 K electronic structure."},
        },
        "system": {
            "domain": "computational",
            "technique": "DFT",
            "facility": {"facility_name": "NERSC", "organization": "ANL", "cluster": "Perlmutter"},
            "instrument": {
                "instrument_type": "simulation_engine",
                "instrument_name": "VASP_Standard",
                "vendor_or_project": "VASP",
            },
            "configuration": {
                "code_version": "VASP 6.4.3",
                "software_module": "vasp/6.4.3-cpu",
                "compute_architecture": "CPU",
                "generator_name": GENERATOR_NAME,
                "generator_version": GENERATOR_VERSION,
                "metadata_source": "structure_info.json + INCAR + KPOINTS + OUTCAR + CONTCAR",
                "energy_source_file": energy_source,
            },
        },
        "computation": {
            "output_quantity": "energy_raw",
            "method": computation_method(root, incar, energy_source),
            "slab_model": slab_model(info, incar),
        },
        "descriptors": {
            "outputs": [{
                "label": "energy_raw",
                "generated_utc": timestamps["created_utc"],
                "generated_by": {"agent": GENERATOR_NAME, "version": GENERATOR_VERSION},
                "method": f"Automatically parsed final VASP raw total energy from {energy_source}.",
                "descriptors": [{
                    "name": "energy_raw",
                    "kind": "absolute",
                    "source": "auto",
                    "value": energy_eV,
                    "unit": "eV",
                    "definition": "Raw VASP total energy; no thermodynamic, solvation, entropy, or reference-state corrections applied.",
                    "uncertainty": {"basis": "none"},
                }],
            }]
        },
        "assets": contcar_asset(root),
        "links": [],
        "tags": ["CO2RR_XAS_agent", "DFT_relaxation"],
    })
    return record


# -----------------------------------------------------------------------------
# XAS record
# -----------------------------------------------------------------------------


def spectrum_files(root: Path) -> List[Path]:
    candidates = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in {"xmu.dat", "CORE_DIELECTRIC_IMAG.dat"} or path.name.endswith("_conv.txt"):
            candidates.append(path)
    return candidates


def parse_spectrum(path: Path) -> Tuple[List[float], List[float], str, str]:
    rows: List[List[float]] = []
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "*", "!", "%")):
            continue
        nums: List[float] = []
        for part in stripped.replace(",", " ").split():
            try:
                value = float(part)
                if math.isfinite(value):
                    nums.append(value)
            except Exception:
                pass
        if len(nums) >= 2:
            rows.append(nums)
    if not rows:
        raise RuntimeError(f"No numeric spectrum rows found in {path}")

    max_cols = max(len(r) for r in rows)
    if path.name == "xmu.dat":
        x_index, y_index = 0, 2 if max_cols >= 3 else 1
        return [r[x_index] for r in rows if len(r) > y_index], [r[y_index] for r in rows if len(r) > y_index], "energy", "absorption_cross_section"
    return [r[0] for r in rows if len(r) > 1], [r[1] for r in rows if len(r) > 1], "energy", "absorption_cross_section"


def edge_position_first_derivative(energy: List[float], intensity: List[float]) -> Optional[float]:
    if len(energy) < 3 or len(intensity) < 3:
        return None
    best_i = None
    best_d = None
    for i in range(1, len(energy)):
        dx = energy[i] - energy[i - 1]
        if dx == 0:
            continue
        deriv = (intensity[i] - intensity[i - 1]) / dx
        if best_d is None or deriv > best_d:
            best_d = deriv
            best_i = i
    return energy[best_i] if best_i is not None else None


def white_line_intensity(intensity: List[float]) -> Optional[float]:
    return max(intensity) if intensity else None


def asset_role(path: Path) -> str:
    name = path.name
    if name in {"POSCAR", "CONTCAR"} or name.lower().endswith((".cif", ".xyz")):
        return "input_structure"
    if name in {"INCAR", "KPOINTS", "submit.sh", "submit_relax.sh", "make_potcar.sh", "feff.inp", "fdmfile.txt"} or name.endswith("_in.txt"):
        return "workflow_recipe"
    if name in {"xmu.dat", "CORE_DIELECTRIC_IMAG.dat"} or name.endswith("_conv.txt"):
        return "reduction_product"
    if name in {"OUTCAR", "vasprun.xml", "OSZICAR", "vasp.out", "fdmnes.out", "feff.out"}:
        return "raw_data_pointer"
    return "other"


def collect_xas_assets(root: Path) -> List[Dict[str, Any]]:
    include = {"POSCAR", "CONTCAR", "INCAR", "KPOINTS", "submit.sh", "submit_relax.sh", "make_potcar.sh", "feff.inp", "fdmfile.txt", "xmu.dat", "CORE_DIELECTRIC_IMAG.dat", "OUTCAR", "OSZICAR", "vasp.out", "fdmnes.out", "feff.out"}
    assets: List[Dict[str, Any]] = []
    for idx, path in enumerate(p for p in sorted(root.rglob("*")) if p.is_file()):
        if path.name not in include and not path.name.endswith(("_in.txt", "_conv.txt")):
            continue
        rel = path.relative_to(root).as_posix()
        assets.append({
            "asset_id": f"asset_{idx:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', rel)[:80]}",
            "content_role": asset_role(path),
            "uri": rel,
            "media_type": media_type(path),
            "sha256": sha256_file(path),
            "notes": f"File collected from {rel}.",
        })
    return assets


def find_matching_clean_dir(root: Path, info: Dict[str, Any]) -> Optional[Path]:
    element = str(info.get("element") or formula_from_structure_info(info))
    facet = str(info.get("facet") or (info.get("catalyst") or {}).get("surface_facet") or "111")
    clean_name = f"{element}_{facet}_clean"
    for parent in [root.parent, root.parent.parent, root.parent.parent.parent, root.parent.parent.parent.parent]:
        candidate = parent / clean_name / "01_structure"
        if (candidate / "OUTCAR").exists() or (candidate / "OSZICAR").exists():
            return candidate
    return None


def find_adsorbate_relax_dir(root: Path) -> Optional[Path]:
    for parent in [root, root.parent, root.parent.parent, root.parent.parent.parent]:
        candidate = parent / "01_structure"
        if (candidate / "OUTCAR").exists() or (candidate / "OSZICAR").exists():
            return candidate
    return None


def adsorption_energy_descriptor(root: Path, info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ads = adsorbate_identity(info)
    if ads.lower() in {"clean", "none", "null", ""}:
        return None
    ads_dir = find_adsorbate_relax_dir(root)
    clean_dir = find_matching_clean_dir(root, info)
    if not ads_dir or not clean_dir:
        return None
    try:
        e_ads, _ = final_energy_from_outputs(ads_dir)
        e_clean, _ = final_energy_from_outputs(clean_dir)
    except Exception:
        return None
    value = e_ads - e_clean
    return {
        "name": f"adsorption_energy.{ads}",
        "kind": "theoretical_metric",
        "source": "auto",
        "value": value,
        "unit": "eV",
        "definition": "Clean-slab-referenced adsorption energy from relaxed adsorbate and clean-surface calculations: E(slab+adsorbate) - E(clean slab).",
        "uncertainty": {"basis": "none"},
    }


def build_xas_record(output_dir: str, sequence: int = 1) -> Dict[str, Any]:
    root = Path(output_dir).resolve()
    info = load_structure_info(root)
    incar = parse_incar(root / "INCAR")
    timestamps = timestamps_from_metadata(root)

    spec_paths = spectrum_files(root)
    if not spec_paths:
        raise RuntimeError(f"No XAS spectrum file found under {root}")
    spec_path = spec_paths[0]
    energy, intensity, x_name, y_name = parse_spectrum(spec_path)

    descriptors = [
        {
            "name": "xanes.edge_position",
            "kind": "absolute",
            "source": "auto",
            "value": edge_position_first_derivative(energy, intensity),
            "unit": "eV",
            "definition": "First-derivative maximum after broadening.",
            "uncertainty": {"basis": "none"},
        },
        {
            "name": "xanes.white_line_intensity",
            "kind": "absolute",
            "source": "auto",
            "value": white_line_intensity(intensity),
            "unit": "arb",
            "definition": "Peak intensity of the simulated absorption spectrum.",
            "uncertainty": {"basis": "none"},
        },
    ]
    ads_desc = adsorption_energy_descriptor(root, info)
    if ads_desc is not None:
        descriptors.append(ads_desc)

    ads = adsorbate_identity(info)
    notes = f"{info.get('layers', 'not_specified')}-layer {formula_from_structure_info(info)}({info.get('facet', 'not_specified')}) slab."
    if ads.lower() not in {"clean", "none", "null", ""}:
        notes += f" Adsorbate: {ads}."
    notes += " Static 0 K electronic structure."

    record = base_evidence_header(xas_record_id(sequence), timestamps)
    record.update({
        "sample": {
            "material": {
                "name": sample_name_from_folder(root),
                "formula": formula_from_structure_info(info),
                "provenance": "theoretical",
                "notes": notes,
            },
            "sample_form": "slab_model",
        },
        "context": {"environment": "in_silico", "temperature_K": 0},
        "system": {
            "domain": "computational",
            "technique": "DFT",
            "facility": {"facility_name": "NERSC", "organization": "ANL", "cluster": "Perlmutter"},
            "instrument": {"instrument_type": "simulation_engine", "instrument_name": "VASP_Standard", "vendor_or_project": "VASP"},
            "configuration": {"code_version": "VASP 6.4.3", "compute_architecture": "CPU", "generator_name": GENERATOR_NAME, "generator_version": GENERATOR_VERSION},
        },
        "computation": {"method": {"family": "DFT", "functional_class": "GGA", "functional_name": "PBE", "core_hole_treatment": "final_state_rule", "notes": computation_method(root, incar).get("notes", "Specific functional settings.")}},
        "measurement": {
            "processing": {"type": "simulated_spectrum_broadened", "recipe_link": {"rel": "processing_recipe", "target": "repo://CO2RR_XAS_agent/xas_simulation"}},
            "series": [{
                "series_id": "computed_xas_spectrum",
                "independent_variables": [{"name": x_name, "unit": "eV", "values": energy}],
                "channels": [{"name": y_name, "unit": "arb", "role": "simulated_observable", "values": intensity}],
            }],
            "qc": {"status": "valid"},
        },
        "assets": collect_xas_assets(root),
        "links": [],
        "descriptors": {"outputs": [{"label": "auto_features_v1", "generated_utc": timestamps["created_utc"], "generated_by": {"agent": GENERATOR_NAME, "version": GENERATOR_VERSION}, "descriptors": descriptors}]},
        "tags": ["CO2RR_XAS_agent", "XAS_simulation"],
    })
    return record


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def detect_mode(root: Path) -> str:
    if root.name == "01_structure" or ((root / "OUTCAR").exists() and not spectrum_files(root)):
        return "relaxation"
    return "xas"


def build_record(output_dir: str, mode: str = "auto", sequence: int = 1) -> Dict[str, Any]:
    root = Path(output_dir).resolve()
    if mode == "auto":
        mode = detect_mode(root)
    if mode == "relaxation":
        return build_relaxation_record(str(root), sequence=sequence)
    if mode == "xas":
        return build_xas_record(str(root), sequence=sequence)
    raise ValueError(f"Unknown ISAAC record mode: {mode}")


def write_record(record: Dict[str, Any], output_dir: str, filename: str = "isaac_record_draft.json") -> str:
    path = Path(output_dir).resolve() / filename
    path.write_text(json.dumps(record, indent=2) + "\n")
    return str(path)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Build schema-compliant ISAAC evidence records from CO2RR XAS Agent outputs.")
    parser.add_argument("output_dir", help="Completed relaxation or XAS output directory.")
    parser.add_argument("--mode", choices=["auto", "relaxation", "xas"], default="auto")
    parser.add_argument("--sequence", type=int, default=1)
    parser.add_argument("--filename", default="isaac_record_draft.json")
    args = parser.parse_args(argv)

    record = build_record(args.output_dir, mode=args.mode, sequence=args.sequence)
    out = write_record(record, args.output_dir, filename=args.filename)
    print(f"Wrote {out}")
    print(f"record_id = {record.get('record_id')}")


if __name__ == "__main__":
    main()
