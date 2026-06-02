"""
Result Parser Tool for CO2RR XAS Agent
ISAAC-style simulation XAS record generator (v1.05 compliant).
"""

import os
import json
import re
import hashlib
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timezone

import numpy as np

from .utils import (
    ensure_dir,
    read_structure_file,
    generate_uuid,
    ATOMIC_NUMBERS,
    get_edge_for_element,
)


# =============================================================================
# General helpers
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _read_text_if_exists(path: str) -> Optional[str]:
    if path and os.path.exists(path):
        with open(path, "r", errors="ignore") as f:
            return f.read()
    return None


def _next_nonempty_token(lines: List[str], start_idx: int) -> Optional[str]:
    for line in lines[start_idx + 1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "!", "*")):
            continue
        parts = stripped.split()
        if parts:
            return parts[0]
    return None


def _normalize_edge(edge: Optional[str]) -> Optional[str]:
    if not edge:
        return None
    edge = str(edge).upper().replace("-EDGE", "").replace("_EDGE", "").strip()
    return edge


def _symbol_from_atomic_number(z_value: str) -> Optional[str]:
    try:
        z = int(float(str(z_value).strip()))
    except (TypeError, ValueError):
        return None
    inverse = {v: k for k, v in ATOMIC_NUMBERS.items()}
    return inverse.get(z)


def sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return "not_available"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _media_type_for_file(path: str) -> str:
    name = Path(path).name.lower()
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".gz") or name.endswith(".tgz"):
        return "application/gzip"
    if name.endswith(".xml"):
        return "application/xml"
    if name in {"poscar", "contcar"} or name.endswith(".vasp"):
        return "chemical/x-poscar"
    if name.endswith(".cif"):
        return "chemical/x-cif"
    if name.endswith(".xyz"):
        return "chemical/x-xyz"
    if name.endswith(".xdi"):
        return "text/plain"
    return "text/plain"


def _asset_uri(path: str, asset_base_uri: Optional[str] = None, repo_root: Optional[str] = None) -> str:
    """Return either a repository URI for a local asset or its filesystem path."""
    p = Path(path)
    if asset_base_uri:
        base = asset_base_uri.rstrip("/")
        root = Path(repo_root).resolve() if repo_root else Path.cwd().resolve()
        try:
            rel = p.resolve().relative_to(root) if p.exists() else p.relative_to(root)
            return f"{base}/{rel.as_posix()}"
        except (ValueError, RuntimeError):
            return f"{base}/{p.name}"
    return str(p.resolve()) if p.exists() else str(p)


def _asset(
    path: str,
    asset_id: str,
    content_role: str,
    notes: str = "",
    asset_base_uri: Optional[str] = None,
    repo_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Create an ISAAC-compliant asset entry."""
    p = Path(path)
    return {
        "asset_id": asset_id,
        "content_role": content_role,
        "uri": _asset_uri(str(p), asset_base_uri=asset_base_uri, repo_root=repo_root),
        "media_type": _media_type_for_file(str(p)),
        "sha256": sha256_file(str(p)),
        "notes": notes,
    }


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def find_run_metadata(output_dir: str) -> Dict[str, Any]:
    """Search output_dir and parents for isaac_run_metadata.json."""
    current = Path(output_dir).resolve()
    candidates = [current / "isaac_run_metadata.json"]
    candidates.extend(parent / "isaac_run_metadata.json" for parent in current.parents[:4])
    for path in candidates:
        data = _load_json(path)
        if isinstance(data, dict):
            data.setdefault("_metadata_file", str(path))
            return data
    return {}


def deep_get(data: Dict[str, Any], keys: List[str], default: Any = None) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _formula_from_atoms(atoms: List[str]) -> str:
    if not atoms:
        return "not_specified"
    counts: Dict[str, int] = {}
    order: List[str] = []
    for atom in atoms:
        if atom not in counts:
            counts[atom] = 0
            order.append(atom)
        counts[atom] += 1
    return "".join(f"{el}{counts[el] if counts[el] > 1 else ''}" for el in order)


def _reduced_formula_from_atoms(atoms: List[str]) -> str:
    """Return a reduced chemical formula inferred from input-file atoms."""
    if not atoms:
        return "not_specified"
    counts: Dict[str, int] = {}
    order: List[str] = []
    for atom in atoms:
        if atom not in counts:
            counts[atom] = 0
            order.append(atom)
        counts[atom] += 1

    divisor = 0
    for count in counts.values():
        divisor = count if divisor == 0 else math.gcd(divisor, count)
    divisor = max(divisor, 1)

    return "".join(
        f"{el}{(counts[el] // divisor) if (counts[el] // divisor) > 1 else ''}"
        for el in order
    )


def _formula_from_feff_input(output_dir: str) -> Optional[str]:
    """Read reduced formula metadata from FEFF input titles when present."""
    for inp in [os.path.join(output_dir, "feff.inp"), os.path.join(os.path.dirname(output_dir), "feff.inp")]:
        text = _read_text_if_exists(inp)
        if not text:
            continue
        match = re.search(r"^TITLE\s+Reduced formula:\s*(\S+)", text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        match = re.search(r"^TITLE\s+Structure Summary:\s*(\S+)", text, re.MULTILINE | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _infer_sample_from_inputs(
    structure: Dict[str, Any],
    output_dir: str,
    structure_file: str,
    software: str,
    absorber: str,
    edge: str,
) -> Dict[str, str]:
    """Infer sample metadata from simulation input files, not hardcoded labels."""
    atoms = structure.get("atoms", []) or []
    formula = _formula_from_feff_input(output_dir) or _reduced_formula_from_atoms(atoms)
    if not formula or formula == "not_specified":
        formula = _formula_from_atoms(atoms)

    unique_atoms = []
    for atom in atoms:
        if atom not in unique_atoms:
            unique_atoms.append(atom)

    sample_form = "bulk_model"
    cell = structure.get("cell") or []
    try:
        lengths = [float(np.linalg.norm(np.array(v, dtype=float))) for v in cell]
        if len(lengths) == 3:
            median = float(np.median(lengths))
            if median > 0 and max(lengths) / median > 1.8:
                sample_form = "slab_model"
    except Exception:
        sample_form = "model"

    source_name = Path(structure_file).name if structure_file else "input structure"
    composition_note = (
        f"contains only {unique_atoms[0]}" if len(unique_atoms) == 1 else f"contains elements {', '.join(unique_atoms)}"
    )
    return {
        "name": f"{formula} reference model for {absorber} {edge}-edge XAS",
        "formula": formula,
        "sample_form": sample_form,
        "notes": f"Sample metadata inferred from {source_name} and {software} input files; {composition_note}.",
    }


def _default_code_version(software: str) -> str:
    s = software.upper()
    if s.startswith("VASP"):
        return "6.3.2"
    if s.startswith("FEFF"):
        return "10"
    if s.startswith("FDMNES"):
        return "2025.11"
    return "not_specified"


def _default_instrument(software: str) -> Dict[str, str]:
    s = software.upper()
    if s.startswith("VASP"):
        return {
            "instrument_type": "simulation_engine",
            "instrument_name": "VASP_Standard",
            "vendor_or_project": "VASP"
        }
    if s.startswith("FEFF"):
        return {
            "instrument_type": "simulation_engine",
            "instrument_name": "FEFF10",
            "vendor_or_project": "FEFF"
        }
    if s.startswith("FDMNES"):
        return {
            "instrument_type": "simulation_engine",
            "instrument_name": "FDMNES",
            "vendor_or_project": "FDMNES"
        }
    return {
        "instrument_type": "simulation_engine",
        "instrument_name": software,
        "vendor_or_project": software
    }


# =============================================================================
# XAS metadata inference (unchanged from your version)
# =============================================================================


def _infer_from_path(output_dir: str) -> Dict[str, Optional[str]]:
    meta = {"absorber": None, "edge": None, "source": None}
    joined = "/".join(os.path.normpath(output_dir).split(os.sep))
    match = re.search(r"\b([A-Z][a-z]?)_(K|L2|L3|L23)_edge\b", joined, re.IGNORECASE)
    if match:
        meta["absorber"] = match.group(1).capitalize()
        meta["edge"] = _normalize_edge(match.group(2))
        meta["source"] = "path"
    return meta


def _infer_from_feff_input(output_dir: str) -> Dict[str, Optional[str]]:
    meta = {"absorber": None, "edge": None, "source": None}
    for inp in [
        os.path.join(output_dir, "feff.inp"),
        os.path.join(os.path.dirname(output_dir), "feff.inp")
    ]:
        text = _read_text_if_exists(inp)
        if not text:
            continue
        match = re.search(r"^\s*EDGE\s+([A-Za-z0-9]+)", text, re.MULTILINE | re.IGNORECASE)
        if match:
            meta["edge"] = _normalize_edge(match.group(1))
        lines = text.splitlines()
        in_potentials = False
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if re.match(r"^POTENTIALS\b", stripped, re.IGNORECASE):
                in_potentials = True
                continue
            if in_potentials and re.match(r"^(ATOMS|END)\b", stripped, re.IGNORECASE):
                break
            if in_potentials and not stripped.startswith(("#", "!", "*")):
                parts = stripped.split()
                if len(parts) >= 3:
                    try:
                        ipot = int(parts[0])
                    except ValueError:
                        continue
                    if ipot == 0:
                        symbol = parts[2]
                        meta["absorber"] = symbol if re.match(r"^[A-Z][a-z]?$", symbol) else _symbol_from_atomic_number(parts[1])
                        break
        if not meta["absorber"]:
            match = re.search(r"Absorber\s*:\s*([A-Z][a-z]?)", text, re.IGNORECASE)
            if match:
                meta["absorber"] = match.group(1).capitalize()
        if not meta["edge"]:
            match = re.search(r"Edge\s*:\s*(K|L2|L3|L23)", text, re.IGNORECASE)
            if match:
                meta["edge"] = _normalize_edge(match.group(1))
        if meta["absorber"] or meta["edge"]:
            meta["source"] = inp
            return meta
    return meta


def _infer_from_fdmnes_input(output_dir: str) -> Dict[str, Optional[str]]:
    meta = {
        "absorber": None,
        "edge": None,
        "source": None,
        "absorber_state": None,
        "core_hole_treatment": None,
        "solver": None
    }
    candidates: List[str] = []
    fdmfile = os.path.join(output_dir, "fdmfile.txt")
    if os.path.exists(fdmfile):
        with open(fdmfile, "r", errors="ignore") as f:
            for line in f:
                stripped = line.strip()
                if stripped and not stripped.isdigit():
                    candidates.append(os.path.join(output_dir, stripped))
    if os.path.isdir(output_dir):
        for name in os.listdir(output_dir):
            if name.endswith("_in.txt") or name.endswith(".in"):
                candidates.append(os.path.join(output_dir, name))
            m = re.match(r"^([A-Z][a-z]?)_conv\.txt$", name)
            if m and not meta["absorber"]:
                meta["absorber"] = m.group(1)

    for inp in dict.fromkeys(candidates):
        text = _read_text_if_exists(inp)
        if not text:
            continue
        lines = text.splitlines()
        
        # Determine solver
        meta["solver"] = "multiple_scattering_green_muffin_tin" if re.search(
            r"^\s*green\b", text, re.I | re.M
        ) else "finite_difference_full_potential"
        
        # Determine core hole treatment
        if re.search(r"^\s*nonexc\b", text, re.I | re.M):
            meta["absorber_state"] = "nonexcited_absorber"
            meta["core_hole_treatment"] = "no_core_hole_no_screening_nonexc"
        elif re.search(r"^\s*excited\b", text, re.I | re.M):
            meta["absorber_state"] = "excited_absorber"
            meta["core_hole_treatment"] = "excited_absorber_core_hole"
        else:
            meta["absorber_state"] = "default_excited_absorber"
            meta["core_hole_treatment"] = "default_excited_absorber_core_hole_for_K_L1_M1_N1_edges"

        for i, line in enumerate(lines):
            if re.match(r"^\s*Z_Absorber\b", line, re.IGNORECASE):
                token = _next_nonempty_token(lines, i)
                symbol = _symbol_from_atomic_number(token) if token is not None else None
                if symbol:
                    meta["absorber"] = symbol
            elif re.match(r"^\s*Edge\b", line, re.IGNORECASE):
                token = _next_nonempty_token(lines, i)
                if token:
                    meta["edge"] = _normalize_edge(token)
            elif re.match(r"^\s*Filout\b", line, re.IGNORECASE) and not meta["absorber"]:
                token = _next_nonempty_token(lines, i)
                if token and re.match(r"^[A-Z][a-z]?$", token):
                    meta["absorber"] = token
        meta["source"] = inp
        break

    if meta.get("absorber") and not meta.get("edge"):
        meta["edge"] = get_edge_for_element(meta["absorber"])[0]
        meta["source"] = meta.get("source") or "fdmnes_filename_or_default_edge"
    return meta


def _infer_from_vasp_input(
    output_dir: str,
    structure_file: Optional[str] = None
) -> Dict[str, Optional[str]]:
    meta = {"absorber": None, "edge": None, "source": None}
    candidates = [
        os.path.join(output_dir, "INCAR"),
        os.path.join(output_dir, "02_xas", "INCAR"),
        os.path.join(os.path.dirname(output_dir), "INCAR"),
        os.path.join(os.path.dirname(output_dir), "02_xas", "INCAR"),
    ]
    for incar in candidates:
        text = _read_text_if_exists(incar)
        if not text:
            continue
        match = re.search(r"Absorber\s*:\s*([A-Z][a-z]?)", text, re.IGNORECASE)
        if match:
            meta["absorber"] = match.group(1).capitalize()
        match = re.search(r"Edge\s*:\s*(K|L2|L3|L23)", text, re.IGNORECASE)
        if match:
            meta["edge"] = _normalize_edge(match.group(1))
        if meta["absorber"] or meta["edge"]:
            meta["source"] = incar
            break
    
    path_meta = _infer_from_path(output_dir)
    meta["absorber"] = meta["absorber"] or path_meta.get("absorber")
    meta["edge"] = meta["edge"] or path_meta.get("edge")
    meta["source"] = meta["source"] or path_meta.get("source")
    
    if not meta["absorber"] and structure_file and os.path.exists(structure_file):
        try:
            structure = read_structure_file(structure_file)
            unique = []
            for atom in structure.get("atoms", []):
                if atom not in unique:
                    unique.append(atom)
            if len(unique) == 1:
                meta["absorber"] = unique[0]
                meta["source"] = "single_element_structure"
        except Exception:
            pass
    
    if meta.get("absorber") and not meta.get("edge"):
        meta["edge"] = get_edge_for_element(meta["absorber"])[0]
        meta["source"] = meta.get("source") or "default_edge_from_absorber"
    return meta


def infer_xas_metadata(
    output_dir: str,
    software: str,
    structure_file: Optional[str] = None
) -> Dict[str, Optional[str]]:
    software_upper = software.upper()
    if software_upper == "FEFF":
        meta = _infer_from_feff_input(output_dir)
    elif software_upper.startswith("FDMNES"):
        meta = _infer_from_fdmnes_input(output_dir)
    elif software_upper.startswith("VASP"):
        meta = _infer_from_vasp_input(output_dir, structure_file=structure_file)
    else:
        meta = {"absorber": None, "edge": None, "source": None}
    
    path_meta = _infer_from_path(output_dir)
    meta["absorber"] = meta.get("absorber") or path_meta.get("absorber")
    meta["edge"] = meta.get("edge") or path_meta.get("edge")
    meta["source"] = meta.get("source") or path_meta.get("source")
    
    if meta.get("absorber") and not meta.get("edge"):
        meta["edge"] = get_edge_for_element(meta["absorber"])[0]
        meta["source"] = meta.get("source") or "default_edge_from_absorber"
    return meta


# =============================================================================
# Output parsers
# =============================================================================


class FEFFParser:
    def parse(self, output_dir: str) -> Dict[str, Any]:
        xmu_file = os.path.join(output_dir, "xmu.dat")
        if not os.path.exists(xmu_file):
            return {"status": "error", "message": f"xmu.dat not found in {output_dir}"}
        energy, mu = [], []
        with open(xmu_file, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        energy.append(float(parts[0]))
                        mu.append(float(parts[2]))
                    except ValueError:
                        continue
        return {
            "status": "success",
            "software": "FEFF",
            "source_file": xmu_file,
            "spectrum": {
                "energy": energy,
                "intensity": mu,
                "energy_unit": "eV",
                "intensity_unit": "arb. units",
                "spectrum_type": "XANES"
            }
        }


class FDMNESParser:
    def parse(self, output_dir: str) -> Dict[str, Any]:
        conv_file = None
        for filename in os.listdir(output_dir):
            if filename.endswith("_conv.txt"):
                conv_file = os.path.join(output_dir, filename)
                break
        if not conv_file:
            return {"status": "error", "message": f"No *_conv.txt file found in {output_dir}"}
        energy, intensity = [], []
        with open(conv_file, "r", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("!") or not line:
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                    except ValueError:
                        continue
        return {
            "status": "success",
            "software": "FDMNES",
            "source_file": conv_file,
            "spectrum": {
                "energy": energy,
                "intensity": intensity,
                "energy_unit": "eV",
                "intensity_unit": "arb. units",
                "spectrum_type": "XANES"
            }
        }


class VASPXASParser:
    def parse(self, output_dir: str, mode: str = "trace") -> Dict[str, Any]:
        outcar_file = os.path.join(output_dir, "OUTCAR")
        if not os.path.exists(outcar_file):
            return {"status": "error", "message": f"OUTCAR not found in {output_dir}"}
        energy, intensity = self._extract_dielectric(outcar_file, mode)
        if not energy:
            return {"status": "error", "message": "Could not extract dielectric function from OUTCAR"}
        return {
            "status": "success",
            "software": "VASP",
            "source_file": outcar_file,
            "spectrum": {
                "energy": energy,
                "intensity": intensity,
                "energy_unit": "eV",
                "intensity_unit": "arb. units",
                "spectrum_type": "XANES",
                "extraction_mode": mode
            }
        }

    def _extract_dielectric(self, outcar_file: str, mode: str) -> Tuple[List[float], List[float]]:
        energy, intensity = [], []
        content = Path(outcar_file).read_text(errors="ignore")
        match = re.search(r"frequency dependent IMAGINARY DIELECTRIC FUNCTION", content)
        if not match:
            return [], []
        lines = content[match.end():].split("\n")
        data_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                e = float(parts[0])
                data_started = True
            except (ValueError, IndexError):
                if data_started:
                    break
                continue
            if len(parts) < 7:
                continue
            try:
                xx, yy, zz = float(parts[1]), float(parts[2]), float(parts[3])
                xy, yz, zx = float(parts[4]), float(parts[5]), float(parts[6])
                if any(np.isnan([xx, yy, zz, xy, yz, zx])):
                    continue
                if mode == "parallel":
                    val = xx + yy
                elif mode == "normal":
                    val = zz
                elif mode == "tauc":
                    val = np.sqrt(e * e * (xx + yy + zz))
                else:
                    val = xx + yy + zz
                energy.append(e)
                intensity.append(val)
            except (ValueError, IndexError):
                continue
        return energy, intensity


# =============================================================================
# ISAAC record generation (UPDATED for full compliance)
# =============================================================================


class ISAACRecordGenerator:
    """Generate ISAAC v1.05 compliant records."""
    
    def _default_method(
        self,
        software: str,
        metadata: Dict[str, Any],
        inferred: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate computation.method block based on software."""
        s = software.upper()
        
        if s.startswith("VASP"):
            return {
                "family": "DFT",
                "functional_class": parameters.get("functional_class", "GGA"),
                "functional_name": parameters.get("functional_name", "PBE"),
                "core_hole_treatment": parameters.get("core_hole_treatment", "final_state_rule"),
                "pseudopotential": parameters.get("pseudopotential", {
                    "applicability": "applicable",
                    "set": parameters.get("potcar_set", "not_specified"),
                    "notes": "VASP POTCAR/pseudopotential information, if available."
                }),
                "notes": parameters.get("method_notes", "Specific functional settings."),
            }
        
        if s.startswith("FEFF"):
            return {
                "family": "real_space_multiple_scattering",
                "functional_class": "not_applicable",
                "functional_name": "FEFF",
                "core_hole_treatment": parameters.get("core_hole_treatment", "corehole_rpa"),
                "pseudopotential": {
                    "applicability": "not_applicable",
                    "set": "not_applicable",
                    "notes": "FEFF does not use VASP-style pseudopotentials/POTCAR files."
                },
                "notes": parameters.get("method_notes", "FEFF real-space multiple-scattering XAS simulation."),
            }
        
        if s.startswith("FDMNES"):
            return {
                "family": "xas_simulation",
                "solver": inferred.get("solver") or parameters.get("solver", "finite_difference_or_green_function"),
                "functional_class": "not_applicable",
                "functional_name": "FDMNES",
                "absorber_state": inferred.get("absorber_state") or parameters.get("absorber_state", "not_specified"),
                "core_hole_treatment": inferred.get("core_hole_treatment") or parameters.get("core_hole_treatment", "not_specified"),
                "pseudopotential": {
                    "applicability": "not_applicable",
                    "set": "not_applicable",
                    "notes": "FDMNES does not use VASP-style pseudopotentials/POTCAR files."
                },
                "notes": parameters.get("method_notes", "FDMNES XAS simulation."),
            }
        
        return {
            "family": "xas_simulation",
            "functional_class": "not_specified",
            "functional_name": software,
            "core_hole_treatment": "not_specified",
            "notes": "not_specified"
        }

    def _compute_descriptors(
        self,
        spectrum: Dict[str, Any],
        absorber: str,
        edge: str
    ) -> Dict[str, Any]:
        """
        Compute auto-extracted descriptors from spectrum.
        Returns ISAAC-compliant descriptors block with numeric uncertainties.
        """
        energy = np.array(spectrum.get("energy", []), dtype=float)
        intensity = np.array(spectrum.get("intensity", []), dtype=float)
        descriptors_list = []
        
        if len(energy) > 2 and len(intensity) == len(energy):
            try:
                # Compute first derivative
                deriv = np.gradient(intensity, energy)
                
                # Edge position: first-derivative maximum
                edge_idx = int(np.nanargmax(deriv))
                edge_position = float(energy[edge_idx])
                
                # Estimate uncertainty from energy step
                energy_step = np.median(np.diff(energy)) if len(energy) > 1 else 0.5
                edge_uncertainty = float(energy_step)
                
                # White line intensity: maximum intensity
                max_intensity = float(np.nanmax(intensity))
                
                # Estimate intensity uncertainty (5% of max or noise estimate)
                intensity_std = float(np.std(intensity[:min(10, len(intensity))])) if len(intensity) > 10 else max_intensity * 0.05
                intensity_uncertainty = max(intensity_std, max_intensity * 0.02)
                
                descriptors_list.extend([
                    {
                        "name": "xanes.edge_position",
                        "kind": "absolute",
                        "source": "auto",
                        "value": round(edge_position, 2),
                        "unit": "eV",
                        "definition": "First-derivative maximum after broadening.",
                        "uncertainty": {
                            "sigma": round(edge_uncertainty, 2),
                            "unit": "eV"
                        }
                    },
                    {
                        "name": "xanes.white_line_intensity",
                        "kind": "absolute",
                        "source": "auto",
                        "value": round(max_intensity, 4),
                        "unit": spectrum.get("intensity_unit", "arb"),
                        "definition": "Peak intensity.",
                        "uncertainty": {
                            "sigma": round(intensity_uncertainty, 4),
                            "unit": spectrum.get("intensity_unit", "arb")
                        }
                    },
                ])
            except Exception:
                pass
        
        return {
            "outputs": [
                {
                    "label": "auto_features_v1",
                    "generated_utc": utc_now(),
                    "generated_by": {
                        "agent": "co2rr_xas_agent",
                        "version": "1.0"
                    },
                    "descriptors": descriptors_list
                }
            ]
        }

    def _build_assets(
        self,
        output_dir: str,
        structure_file: str,
        parsed_source_file: Optional[str],
        run_metadata: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Build ISAAC-compliant assets list."""
        # Prefer explicitly supplied metadata assets unless the caller asks to
        # regenerate them from files in output_dir (useful for example records
        # that should use repository/GitHub links rather than local run paths).
        assets = run_metadata.get("assets") if isinstance(run_metadata.get("assets"), list) else []
        if assets and not parameters.get("prefer_generated_assets", False):
            return assets

        asset_base_uri = parameters.get("asset_base_uri")
        repo_root = parameters.get("repo_root")
        
        out = []
        
        # Input structure
        if structure_file and os.path.exists(structure_file):
            out.append(_asset(
                structure_file,
                "structure_relaxed",
                "input_structure",
                "Relaxed geometry used for XAS simulation. Contains full atomic coordinates.",
                asset_base_uri=asset_base_uri,
                repo_root=repo_root,
            ))
        
        # Workflow recipe files
        recipe_files = [
            ("feff.inp", "feff_inputs", "FEFF input deck."),
            ("fdmfile.txt", "fdmnes_inputs", "FDMNES input file list."),
            ("INCAR", "vasp_incar", "VASP INCAR file."),
            ("KPOINTS", "vasp_kpoints", "VASP KPOINTS file."),
            ("POTCAR.spec", "vasp_potcar_spec", "VASP POTCAR specification."),
            ("submit.sh", "submit_script", "Job submission script."),
            ("feff.pbs", "feff_submit_script", "PBS job submission script for the FEFF example run."),
            ("fdm.pbs", "fdmnes_submit_script", "PBS job submission script for the FDMNES example run."),
            ("vasp.pbs", "vasp_submit_script", "PBS job submission script for the VASP example run."),
        ]
        
        for filename, asset_id, notes in recipe_files:
            p = Path(output_dir) / filename
            if p.exists():
                out.append(_asset(str(p), asset_id, "workflow_recipe", notes, asset_base_uri=asset_base_uri, repo_root=repo_root))
        
        # Also check for *_in.txt files (FDMNES)
        if os.path.isdir(output_dir):
            for name in os.listdir(output_dir):
                if name.endswith("_in.txt"):
                    p = Path(output_dir) / name
                    out.append(_asset(str(p), f"fdmnes_{name}", "workflow_recipe", "FDMNES input file.", asset_base_uri=asset_base_uri, repo_root=repo_root))
                    break  # Only add one
        
        # Reduction product (parsed spectrum)
        if parsed_source_file and os.path.exists(parsed_source_file):
            out.append(_asset(
                parsed_source_file,
                "xas_fullres",
                "reduction_product",
                "Full-resolution simulated spectrum (energy grid + mu).",
                asset_base_uri=asset_base_uri,
                repo_root=repo_root,
            ))
        
        return out

    def _build_links(
        self,
        parameters: Dict[str, Any],
        run_metadata: Dict[str, Any],
        absorber: str,
        edge: str
    ) -> List[Dict[str, Any]]:
        """Build ISAAC-compliant links list."""
        # Use run_metadata links if available
        links = run_metadata.get("links") if isinstance(run_metadata.get("links"), list) else []
        if links:
            return links
        
        # Use parameters links if provided
        param_links = parameters.get("links", [])
        if param_links:
            return param_links
        
        # Default: empty but with schema-compliant structure available
        # User can add comparison targets via parameters
        comparison_target = parameters.get("comparison_target_id")
        if comparison_target:
            return [
                {
                    "rel": "intended_comparison_target",
                    "target": comparison_target,
                    "basis": "same_absorber_edge",
                    "notes": f"Compare simulated {absorber} {edge}-edge spectrum against measurement record."
                }
            ]
        
        return []

    def create_record(
        self,
        structure: Dict[str, Any],
        software: str,
        spectrum: Dict[str, Any],
        parameters: Dict[str, Any],
        absorber: str,
        edge: str,
        output_dir: str,
        structure_file: str,
        parsed_source_file: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Create ISAAC v1.05 compliant record.
        
        Field order matches ISAAC example:
        isaac_record_version, record_id, record_type, record_domain, source_type,
        timestamps, sample, context, system, computation, measurement, assets,
        links, descriptors
        """
        parameters = dict(parameters or {})
        run_metadata = find_run_metadata(output_dir)
        inferred = parameters.get("metadata_inference", {})
        
        # Extract atoms and infer sample metadata from input files.
        atoms = structure.get("atoms", []) or []
        input_sample = _infer_sample_from_inputs(
            structure=structure,
            output_dir=output_dir,
            structure_file=structure_file,
            software=software,
            absorber=absorber,
            edge=edge,
        )
        formula = parameters.get("formula") or input_sample.get("formula") or _formula_from_atoms(atoms)
        
        # Extract metadata from run_metadata
        sample_meta = run_metadata.get("sample", {}) if isinstance(run_metadata.get("sample"), dict) else {}
        material_meta = sample_meta.get("material", {}) if isinstance(sample_meta.get("material"), dict) else {}
        timestamps_meta = run_metadata.get("timestamps", {}) if isinstance(run_metadata.get("timestamps"), dict) else {}
        
        instrument_default = _default_instrument(software)
        instrument_meta = deep_get(run_metadata, ["system", "instrument"], {}) or {}
        facility_meta = deep_get(run_metadata, ["system", "facility"], {}) or {}
        config_meta = deep_get(run_metadata, ["system", "configuration"], {}) or {}
        
        measurement_meta = run_metadata.get("measurement", {}) if isinstance(run_metadata.get("measurement"), dict) else {}
        processing_meta = measurement_meta.get("processing", {}) if isinstance(measurement_meta.get("processing"), dict) else {}
        qc_meta = measurement_meta.get("qc", {}) if isinstance(measurement_meta.get("qc"), dict) else {}

        # Determine technique
        technique = deep_get(run_metadata, ["system", "technique"], None)
        if not technique:
            technique = "DFT" if software.upper().startswith("VASP") else "XAS simulation"

        # Series ID
        series_id = f"computed_{edge.lower()}_edge"
        
        # Build record in ISAAC field order
        record = {
            "isaac_record_version": "1.05",
            "record_id": parameters.get("record_id", generate_uuid()),
            "record_type": "evidence",
            "record_domain": "simulation",
            "source_type": "computation",
            
            "timestamps": {
                "acquired_start_utc": parameters.get("acquired_start_utc") or timestamps_meta.get("acquired_start_utc", "not_specified"),
                "acquired_end_utc": parameters.get("acquired_end_utc") or timestamps_meta.get("acquired_end_utc", "not_specified"),
                "created_utc": utc_now(),
            },
            
            "sample": {
                "material": {
                    "name": parameters.get("sample_name") or input_sample.get("name") or material_meta.get("name") or f"{formula} model for {absorber} {edge}-edge XAS",
                    "formula": parameters.get("formula") or input_sample.get("formula") or material_meta.get("formula") or formula,
                    "provenance": material_meta.get("provenance", "theoretical"),
                    "notes": parameters.get("sample_notes") or input_sample.get("notes") or material_meta.get("notes") or description or "not_specified",
                },
                "sample_form": parameters.get("sample_form") or input_sample.get("sample_form") or sample_meta.get("sample_form", "slab_model"),
            },
            
            "context": {
                "environment": deep_get(run_metadata, ["context", "environment"], "in_silico"),
                "temperature_K": deep_get(run_metadata, ["context", "temperature_K"], 0),
                "notes": deep_get(run_metadata, ["context", "notes"], "Static 0 K electronic structure."),
            },
            
            "system": {
                "domain": "computational",
                "technique": technique,
                "facility": {
                    "facility_name": facility_meta.get("facility_name", parameters.get("facility_name", "NERSC")),
                    "organization": facility_meta.get("organization", parameters.get("organization", "LBNL")),
                    "cluster": facility_meta.get("cluster", parameters.get("cluster", "Perlmutter")),
                },
                "instrument": {
                    "instrument_type": instrument_meta.get("instrument_type", instrument_default["instrument_type"]),
                    "instrument_name": instrument_meta.get("instrument_name", instrument_default["instrument_name"]),
                    "vendor_or_project": instrument_meta.get("vendor_or_project", instrument_default["vendor_or_project"]),
                },
                "configuration": {
                    "code_version": config_meta.get("code_version", parameters.get("code_version", _default_code_version(software))),
                    "compute_architecture": config_meta.get("compute_architecture", parameters.get("compute_architecture", "GPU")),
                },
            },
            
            "computation": {
                "method": self._default_method(software, run_metadata, inferred, parameters),
                "run": run_metadata.get("run", parameters.get("run", {})),
            },
            
            "measurement": {
                "processing": {
                    "type": processing_meta.get("type", parameters.get("processing_type", "simulated_spectrum_broadened")),
                    "recipe_link": processing_meta.get("recipe_link", {
                        "rel": "processing_recipe",
                        "target": parameters.get("recipe_target", "not_specified")
                    }),
                },
                "series": [
                    {
                        "series_id": series_id,
                        "independent_variables": [
                            {
                                "name": "energy",
                                "unit": spectrum.get("energy_unit", "eV"),
                                "values": spectrum.get("energy", [])
                            }
                        ],
                        "channels": [
                            {
                                "name": "absorption_cross_section",
                                "unit": spectrum.get("intensity_unit", "arb. units"),
                                "role": "simulated_observable",
                                "values": spectrum.get("intensity", [])
                            }
                        ],
                    }
                ],
                "qc": {
                    "status": qc_meta.get("status", parameters.get("qc_status", "valid"))
                },
            },
            
            "assets": self._build_assets(output_dir, structure_file, parsed_source_file, run_metadata, parameters),
            
            "links": self._build_links(parameters, run_metadata, absorber, edge),
            
            "descriptors": run_metadata.get("descriptors") or self._compute_descriptors(spectrum, absorber, edge),
        }
        
        return record

    def save_record(self, record: Dict[str, Any], output_path: str) -> str:
        """Save record to JSON file."""
        ensure_dir(os.path.dirname(output_path))
        with open(output_path, "w") as f:
            json.dump(record, f, indent=2)
        return output_path


# =============================================================================
# Main parser interface
# =============================================================================


class ResultParser:
    """Main interface for parsing XAS results and creating ISAAC records."""
    
    def __init__(self):
        self.feff_parser = FEFFParser()
        self.fdmnes_parser = FDMNESParser()
        self.vasp_parser = VASPXASParser()
        self.record_gen = ISAACRecordGenerator()

    def parse_results(
        self,
        output_dir: str,
        software: str,
        vasp_mode: str = "trace"
    ) -> Dict[str, Any]:
        """Parse results from specified software."""
        software_upper = software.upper()
        if software_upper == "FEFF":
            return self.feff_parser.parse(output_dir)
        if software_upper.startswith("FDMNES"):
            return self.fdmnes_parser.parse(output_dir)
        if software_upper.startswith("VASP"):
            return self.vasp_parser.parse(output_dir, mode=vasp_mode)
        return {"status": "error", "message": f"Unknown software: {software}"}

    def parse_and_create_record(
        self,
        output_dir: str,
        software: str,
        structure_file: str,
        absorber: Optional[str] = None,
        edge: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        vasp_mode: str = "trace",
        description: str = "",
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Parse results and create ISAAC-compliant record.
        
        Args:
            output_dir: Directory containing simulation output
            software: Software used (FEFF, FDMNES, VASP)
            structure_file: Path to structure file
            absorber: Absorbing element (inferred if not provided)
            edge: Absorption edge (inferred if not provided)
            parameters: Additional parameters for record generation
            vasp_mode: VASP extraction mode (trace, parallel, normal, tauc)
            description: Record description
            tags: Record tags
            
        Returns:
            Dictionary with status, spectrum, isaac_record, and record_path
        """
        results = self.parse_results(output_dir, software, vasp_mode)
        if results["status"] != "success":
            return results
        
        inferred = infer_xas_metadata(output_dir, software, structure_file=structure_file)
        absorber = absorber or inferred.get("absorber")
        edge = edge or inferred.get("edge")
        
        if not absorber or not edge:
            return {
                "status": "error",
                "message": (
                    "Could not determine absorber/edge from input files. "
                    "Pass absorber=... and edge=... explicitly, or keep "
                    "FEFF feff.inp / FDMNES *_in.txt / VASP INCAR in the output directory."
                ),
                "inferred_metadata": inferred
            }
        
        parameters = dict(parameters or {})
        parameters.setdefault("metadata_inference", inferred)
        
        structure = read_structure_file(structure_file)
        
        record = self.record_gen.create_record(
            structure=structure,
            software=software,
            spectrum=results["spectrum"],
            parameters=parameters,
            absorber=absorber,
            edge=edge,
            output_dir=output_dir,
            structure_file=structure_file,
            parsed_source_file=results.get("source_file"),
            description=description,
            tags=tags,
        )
        
        record_path = os.path.join(output_dir, "isaac_record.json")
        self.record_gen.save_record(record, record_path)
        
        return {
            "status": "success",
            "spectrum": results["spectrum"],
            "isaac_record": record,
            "record_path": record_path
        }


def execute_result_parsing(
    output_dir: str,
    software: str,
    structure_file: str,
    absorber: Optional[str] = None,
    edge: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    vasp_mode: str = "trace",
    description: str = "",
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Execute result parsing and ISAAC record generation.
    
    This is the main entry point for the result parsing skill.
    """
    parser = ResultParser()
    return parser.parse_and_create_record(
        output_dir=output_dir,
        software=software,
        structure_file=structure_file,
        absorber=absorber,
        edge=edge,
        parameters=parameters,
        vasp_mode=vasp_mode,
        description=description,
        tags=tags,
    )