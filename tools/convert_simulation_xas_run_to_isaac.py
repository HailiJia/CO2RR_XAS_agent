#!/usr/bin/env python3
"""Convert a completed FEFF/FDMNES/VASP XAS run folder to an ISAAC 1.05 record.

This is intended for repeated use from generated ``submit.sh`` scripts. It writes
an ISAAC evidence record aligned with the official simulation XAS example:

  isaac_record_version = 1.05
  record_type          = evidence
  record_domain        = simulation
  source_type          = computation
  measurement.series   = numeric XAS spectrum
  assets               = input/output files with sha256

The converter reads ``isaac_run_metadata.json`` when present. That file can be
created by the run script before/after the calculation and should contain
``timestamps``, ``sample``, ``context``, ``system``, ``run``, and ``assets`` if
available. Missing fields are filled from environment variables and parsed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ISAAC_RECORD_VERSION = "1.05"
GENERATOR_NAME = "CO2RR XAS Agent"
GENERATOR_VERSION = "v45_2026-07-03_reusable_simulation_xas_converter"
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"

SPECTRUM_NAMES = {"xmu.dat", "CORE_DIELECTRIC_IMAG.dat"}
INPUT_NAMES = {
    "POSCAR", "CONTCAR", "INCAR", "KPOINTS", "POTCAR", "POTCAR.spec",
    "make_potcar.sh", "submit.sh", "submit_relax.sh", "feff.inp", "HEADER",
    "PARAMETERS", "POTENTIALS", "ATOMS", "fdmfile.txt",
}
OUTPUT_NAMES = {
    "OUTCAR", "vasprun.xml", "OSZICAR", "vasp.out", "fdmnes.out", "feff.out",
    "chi.dat", "paths.dat", "files.dat", "log1.dat", "CORE_DIELECTRIC_IMAG.dat",
    "xmu.dat", "isaac_run_metadata.json", "isaac_record_draft.json",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def generate_record_id() -> str:
    ts_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    value = (ts_ms << 80) | (uuid.uuid4().int >> 48)
    chars = []
    for _ in range(26):
        chars.append(CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def read_json(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def read_text(path: Path) -> str:
    try:
        return path.read_text(errors="ignore")
    except Exception:
        return ""


def sha256_file(path: Path) -> str:
    try:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "not_available"


def media_type(path: Path) -> str:
    name = path.name.lower()
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".xml"):
        return "application/xml"
    if name.endswith((".tar.gz", ".tgz")):
        return "application/gzip"
    if name.endswith(".zip"):
        return "application/zip"
    if name.endswith(".xyz"):
        return "chemical/x-xyz"
    if name.endswith(".cif"):
        return "chemical/x-cif"
    return "text/plain"


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def first_present(*values: Any, default: Any = "not_specified") -> Any:
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return default


def parse_numeric_table(path: Path) -> List[List[float]]:
    rows: List[List[float]] = []
    for line in read_text(path).splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "*", "!", "%")):
            continue
        nums: List[float] = []
        for token in stripped.replace(",", " ").split():
            try:
                value = float(token)
                if math.isfinite(value):
                    nums.append(value)
            except Exception:
                pass
        if len(nums) >= 2:
            rows.append(nums)
    return rows


def find_spectrum_files(root: Path) -> List[Path]:
    paths: List[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SPECTRUM_NAMES or path.name.endswith("_conv.txt"):
            paths.append(path)
    return paths


def parse_spectrum(path: Path) -> Tuple[List[float], List[float], str, str, str]:
    rows = parse_numeric_table(path)
    if not rows:
        raise RuntimeError(f"No numeric spectrum rows found in {path}")
    max_cols = max(len(row) for row in rows)
    if path.name == "xmu.dat":
        y_index = 2 if max_cols >= 3 else 1
        return [r[0] for r in rows if len(r) > y_index], [r[y_index] for r in rows if len(r) > y_index], "energy", "absorption_cross_section", "FEFF"
    if path.name.endswith("_conv.txt"):
        return [r[0] for r in rows if len(r) > 1], [r[1] for r in rows if len(r) > 1], "energy", "absorption_cross_section", "FDMNES"
    if path.name == "CORE_DIELECTRIC_IMAG.dat":
        return [r[0] for r in rows if len(r) > 1], [r[1] for r in rows if len(r) > 1], "energy", "absorption_cross_section", "VASP"
    return [r[0] for r in rows if len(r) > 1], [r[1] for r in rows if len(r) > 1], "energy", "absorption_cross_section", "unknown"


def parse_key_value_file(path: Path) -> Dict[str, str]:
    data: Dict[str, str] = {}
    for line in read_text(path).splitlines():
        line = line.split("#", 1)[0].split("!", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            data[key.strip().upper()] = value.strip()
    return data


def detect_software(root: Path, parsed_software: str, metadata: Dict[str, Any]) -> str:
    run = metadata.get("run") if isinstance(metadata.get("run"), dict) else {}
    if run.get("software"):
        return str(run["software"])
    system = metadata.get("system") if isinstance(metadata.get("system"), dict) else {}
    instrument = system.get("instrument") if isinstance(system.get("instrument"), dict) else {}
    if instrument.get("vendor_or_project"):
        return str(instrument["vendor_or_project"])
    if parsed_software != "unknown":
        return parsed_software
    names = {p.name for p in root.rglob("*") if p.is_file()}
    if "xmu.dat" in names or "feff.inp" in names:
        return "FEFF"
    if "fdmfile.txt" in names or any(name.endswith("_conv.txt") for name in names):
        return "FDMNES"
    if "CORE_DIELECTRIC_IMAG.dat" in names or "OUTCAR" in names:
        return "VASP"
    return env("ISAAC_SOFTWARE", "not_specified")


def code_version_for(software: str, metadata: Dict[str, Any]) -> str:
    system = metadata.get("system") if isinstance(metadata.get("system"), dict) else {}
    config = system.get("configuration") if isinstance(system.get("configuration"), dict) else {}
    existing = config.get("code_version")
    if existing:
        return str(existing)
    value = env("ISAAC_CODE_VERSION")
    if value:
        return value
    defaults = {"FEFF": "FEFF 10", "FDMNES": "FDMNES", "VASP": "VASP 6.4.3"}
    return defaults.get(software.upper(), "not_specified")


def normalize_edge(edge: str) -> str:
    text = str(edge or "").upper().replace("_EDGE", "").replace("-EDGE", "")
    match = re.search(r"\b(K|L[1-3]|M[1-5])\b", text)
    return match.group(1) if match else text.strip() or "not_specified"


def infer_absorber_edge(root: Path, metadata: Dict[str, Any], energy: List[float]) -> Tuple[str, str]:
    system = metadata.get("system") if isinstance(metadata.get("system"), dict) else {}
    config = system.get("configuration") if isinstance(system.get("configuration"), dict) else {}
    absorber = first_present(
        env("ISAAC_ABSORBER"),
        system.get("absorber"),
        config.get("absorber"),
        metadata.get("absorber"),
        default="",
    )
    edge = first_present(
        env("ISAAC_EDGE"),
        system.get("edge"),
        config.get("edge"),
        metadata.get("edge"),
        default="",
    )
    text = " ".join([str(absorber), str(edge), root.as_posix()])
    match = re.search(r"(?<![A-Za-z0-9])([A-Z][a-z]?)[_\-\s]*(K|L[1-3]|M[1-5])(?:[_\-\s]*edge)?(?![A-Za-z0-9])", text, re.I)
    if match:
        absorber = absorber or match.group(1).capitalize()
        edge = edge or match.group(2).upper()
    return str(absorber or "not_specified"), normalize_edge(str(edge or "not_specified"))


def edge_position_first_derivative(energy: List[float], intensity: List[float]) -> Optional[float]:
    if len(energy) < 3 or len(intensity) < 3:
        return None
    best_idx = None
    best_val = None
    for i in range(1, min(len(energy), len(intensity))):
        dx = energy[i] - energy[i - 1]
        if dx == 0:
            continue
        val = (intensity[i] - intensity[i - 1]) / dx
        if best_val is None or val > best_val:
            best_val = val
            best_idx = i
    return energy[best_idx] if best_idx is not None else None


def collect_assets(root: Path) -> List[Dict[str, Any]]:
    assets: List[Dict[str, Any]] = []
    for idx, path in enumerate(p for p in sorted(root.rglob("*")) if p.is_file()):
        if path.name not in INPUT_NAMES and path.name not in OUTPUT_NAMES and not path.name.endswith(("_in.txt", "_conv.txt", "_bav.txt")):
            continue
        rel = path.relative_to(root).as_posix()
        lower = path.name.lower()
        if path.name in {"POSCAR", "CONTCAR"} or lower.endswith((".cif", ".xyz")):
            role = "input_structure"
        elif path.name in INPUT_NAMES or path.name.endswith("_in.txt"):
            role = "workflow_recipe"
        elif path.name in SPECTRUM_NAMES or path.name.endswith("_conv.txt"):
            role = "reduction_product"
        else:
            role = "raw_data_pointer"
        assets.append({
            "asset_id": f"asset_{idx:03d}_{re.sub(r'[^A-Za-z0-9]+', '_', rel)[:80]}",
            "content_role": role,
            "uri": rel,
            "media_type": media_type(path),
            "sha256": sha256_file(path),
            "notes": f"Collected from {rel}.",
        })
    return assets


def sample_from_metadata(root: Path, metadata: Dict[str, Any]) -> Dict[str, Any]:
    sample = metadata.get("sample") if isinstance(metadata.get("sample"), dict) else {}
    material = sample.get("material") if isinstance(sample.get("material"), dict) else {}
    return {
        "material": {
            "name": first_present(material.get("name"), env("ISAAC_SAMPLE_NAME"), root.parent.name, default="not_specified"),
            "formula": first_present(material.get("formula"), env("ISAAC_SAMPLE_FORMULA"), default="not_specified"),
            "provenance": first_present(material.get("provenance"), env("ISAAC_MATERIAL_PROVENANCE"), default="theoretical"),
            "notes": first_present(material.get("notes"), env("ISAAC_SAMPLE_NOTES"), default="Static 0 K electronic structure unless otherwise specified."),
        },
        "sample_form": first_present(sample.get("sample_form"), env("ISAAC_SAMPLE_FORM"), default="slab_model"),
    }


def context_from_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    context = metadata.get("context") if isinstance(metadata.get("context"), dict) else {}
    out = dict(context)
    out.setdefault("environment", first_present(env("ISAAC_CONTEXT_ENVIRONMENT"), default="in_silico"))
    try:
        out.setdefault("temperature_K", float(first_present(env("ISAAC_TEMPERATURE_K"), default=0)))
    except Exception:
        out.setdefault("temperature_K", 0)
    if env("ISAAC_CONTEXT_NOTES"):
        out.setdefault("notes", env("ISAAC_CONTEXT_NOTES"))
    return out


def system_from_metadata(metadata: Dict[str, Any], software: str, absorber: str, edge: str) -> Dict[str, Any]:
    system = metadata.get("system") if isinstance(metadata.get("system"), dict) else {}
    facility = system.get("facility") if isinstance(system.get("facility"), dict) else {}
    instrument = system.get("instrument") if isinstance(system.get("instrument"), dict) else {}
    config = system.get("configuration") if isinstance(system.get("configuration"), dict) else {}
    software_upper = software.upper()
    instrument_name = first_present(
        instrument.get("instrument_name"), env("ISAAC_INSTRUMENT_NAME"),
        {"FEFF": "FEFF10", "FDMNES": "FDMNES", "VASP": "VASP_Standard"}.get(software_upper),
        default="simulation_engine",
    )
    vendor = first_present(instrument.get("vendor_or_project"), env("ISAAC_VENDOR_OR_PROJECT"), software, default="not_specified")
    configuration = dict(config)
    configuration.update({
        "code_version": code_version_for(software, metadata),
        "compute_architecture": first_present(config.get("compute_architecture"), env("ISAAC_COMPUTE_ARCHITECTURE"), default="CPU"),
        "absorber": absorber,
        "edge": edge,
        "generator_name": GENERATOR_NAME,
        "generator_version": GENERATOR_VERSION,
    })
    if env("ISAAC_SOFTWARE_MODULE"):
        configuration["software_module"] = env("ISAAC_SOFTWARE_MODULE")
    return {
        "domain": "computational",
        "technique": first_present(env("ISAAC_SCHEMA_TECHNIQUE"), system.get("technique"), default="XAS"),
        "absorber": absorber,
        "edge": edge,
        "facility": {
            "facility_name": first_present(facility.get("facility_name"), env("ISAAC_FACILITY_NAME"), default="NERSC"),
            "organization": first_present(facility.get("organization"), env("ISAAC_ORGANIZATION"), default="LBNL"),
            "cluster": first_present(facility.get("cluster"), env("ISAAC_CLUSTER"), default="Perlmutter"),
        },
        "instrument": {
            "instrument_type": "simulation_engine",
            "instrument_name": instrument_name,
            "vendor_or_project": vendor,
        },
        "configuration": configuration,
    }


def computation_block(root: Path, software: str, absorber: str, edge: str) -> Dict[str, Any]:
    incar = parse_key_value_file(root / "INCAR")
    software_upper = software.upper()
    family = "DFT" if software_upper == "VASP" else "multiple_scattering_xas"
    method: Dict[str, Any] = {
        "family": family,
        "core_hole_treatment": "not_specified",
        "notes": f"Simulated {absorber} {edge}-edge XAS using {software}.",
    }
    if software_upper == "VASP" or incar:
        method.update({
            "functional_class": "GGA",
            "functional_name": "PBE",
            "basis_type": "planewave",
            "pseudopotential": "PAW",
        })
        if incar.get("ENCUT"):
            try:
                method["cutoff_eV"] = float(str(incar["ENCUT"]).split()[0])
            except Exception:
                pass
    return {"method": method}


def build_record(root: Path, *, record_id: str = "") -> Dict[str, Any]:
    metadata = read_json(root / "isaac_run_metadata.json")
    timestamps = metadata.get("timestamps") if isinstance(metadata.get("timestamps"), dict) else {}
    created = utc_now()
    spec_paths = find_spectrum_files(root)
    if not spec_paths:
        raise RuntimeError(f"No XAS spectrum file found under {root}")
    spec_path = spec_paths[0]
    energy, intensity, x_name, y_name, parsed_software = parse_spectrum(spec_path)
    software = detect_software(root, parsed_software, metadata)
    absorber, edge = infer_absorber_edge(root, metadata, energy)
    record = {
        "isaac_record_version": ISAAC_RECORD_VERSION,
        "record_id": record_id or env("ISAAC_RECORD_ID") or generate_record_id(),
        "record_type": "evidence",
        "record_domain": "simulation",
        "source_type": "computation",
        "timestamps": {
            "acquired_start_utc": first_present(timestamps.get("acquired_start_utc"), env("ISAAC_START_UTC"), default=created),
            "acquired_end_utc": first_present(timestamps.get("acquired_end_utc"), env("ISAAC_END_UTC"), default=created),
            "created_utc": first_present(timestamps.get("created_utc"), default=created),
        },
        "sample": sample_from_metadata(root, metadata),
        "context": context_from_metadata(metadata),
        "system": system_from_metadata(metadata, software, absorber, edge),
        "computation": computation_block(root, software, absorber, edge),
        "measurement": {
            "processing": {
                "type": first_present(env("ISAAC_PROCESSING_TYPE"), default="simulated_spectrum"),
                "recipe_link": {
                    "rel": "processing_recipe",
                    "target": first_present(env("ISAAC_RECIPE_TARGET"), default="repo://CO2RR_XAS_agent/xas_simulation"),
                },
            },
            "series": [{
                "series_id": f"computed_{str(absorber).lower()}_{str(edge).lower()}_edge" if absorber != "not_specified" and edge != "not_specified" else "computed_xas_spectrum",
                "independent_variables": [{"name": x_name or "energy", "unit": "eV", "values": energy}],
                "channels": [{"name": y_name or "absorption_cross_section", "unit": "arb", "role": "simulated_observable", "values": intensity}],
            }],
            "qc": {"status": "valid" if str(env("ISAAC_EXIT_CODE", "0")) == "0" else "failed"},
        },
        "assets": collect_assets(root),
        "links": [],
        "descriptors": {
            "outputs": [{
                "label": "auto_features_v1",
                "generated_utc": created,
                "generated_by": {"agent": GENERATOR_NAME, "version": GENERATOR_VERSION},
                "descriptors": [
                    {
                        "name": "xanes.edge_position",
                        "kind": "absolute",
                        "source": "auto",
                        "value": edge_position_first_derivative(energy, intensity),
                        "unit": "eV",
                        "definition": "First-derivative maximum of the simulated XAS spectrum.",
                        "uncertainty": {"basis": "none"},
                    },
                    {
                        "name": "xanes.white_line_intensity",
                        "kind": "absolute",
                        "source": "auto",
                        "value": max(intensity) if intensity else None,
                        "unit": "arb",
                        "definition": "Maximum intensity of the simulated absorption spectrum.",
                        "uncertainty": {"basis": "none"},
                    },
                ],
            }],
        },
        "tags": ["CO2RR_XAS_agent", "XAS_simulation", str(software)],
    }
    return record


def post_json(url: str, api_key: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default=".", help="Completed XAS run directory.")
    parser.add_argument("--output", default="isaac_record_draft.json", help="Output ISAAC JSON path.")
    parser.add_argument("--record-id", default="", help="Optional ISAAC record_id. Defaults to generated ULID-like ID or ISAAC_RECORD_ID env var.")
    parser.add_argument("--validate", action="store_true", help="Validate with ISAAC Portal using ISAAC_URL/ISAAC_KEY.")
    parser.add_argument("--upload", action="store_true", help="Validate then upload with ISAAC Portal using ISAAC_URL/ISAAC_KEY.")
    args = parser.parse_args(argv)

    root = Path(args.input_dir).resolve()
    record = build_record(root, record_id=args.record_id)
    out = Path(args.output)
    if not out.is_absolute():
        out = root / out
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"Wrote {out}")
    print(f"record_id = {record.get('record_id')}")
    print(f"series_count = {len(record.get('measurement', {}).get('series', []))}")
    print(f"asset_count = {len(record.get('assets', []))}")

    if args.validate or args.upload:
        isaac_url = env("ISAAC_URL", "https://isaac.slac.stanford.edu/portal/api").rstrip("/")
        isaac_key = env("ISAAC_KEY")
        if not isaac_key:
            raise SystemExit("Set ISAAC_KEY before using --validate or --upload.")
        validation = post_json(f"{isaac_url}/validate", isaac_key, record)
        print(json.dumps({"validation": validation}, indent=2))
        if not validation.get("valid"):
            raise SystemExit("Validation failed; not uploading.")
        if args.upload:
            created = post_json(f"{isaac_url}/records", isaac_key, record)
            print(json.dumps({"create": created}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
