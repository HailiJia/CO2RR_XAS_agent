"""Utilities for converting ISAAC records into XAS spectral rows.

These functions are shared by the Streamlit ML page and can also be reused by
small command-line summaries. The record-level XAS summary follows the same
logic as the lightweight `simple_xas_summary.py` helper used for local JSON
folders: read formula/material/technique, absorber/edge metadata, and infer the
edge from the energy axis when metadata is missing.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
import zipfile
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

ENERGY_NAMES = ("energy", "incident_energy", "photon_energy", "mono_energy", "e0", "energy_ev", "e_ev")
ENERGY_UNITS = ("ev", "electronvolt", "electron_volt", "kev")
SIGNAL_NAMES = ("absorption", "absorbance", "mu", "xanes", "xas", "tey", "tfy", "pfy", "herfd", "intensity", "fluorescence", "delta_absorption")
EDGE_WINDOWS = [
    ("C", "K", 260, 330), ("O", "K", 500, 560),
    ("Cu", "L3", 920, 970), ("Ni", "L3", 830, 890),
    ("Cu", "K", 8850, 9100), ("Ni", "K", 8200, 8500),
    ("Ir", "L3", 11100, 11350), ("Pt", "L3", 11400, 11650),
    ("Au", "L3", 11800, 12080), ("Bi", "L3", 13300, 13550),
]
XAS_TECHNIQUE_RE = re.compile(r"\b(XAS|XANES|EXAFS|XAFS)\b|x-ray absorption|xray absorption|absorption spectroscopy", re.I)
NON_XAS_TECHNIQUE_RE = re.compile(r"\b(XRD|XPS|UPS|XRF|XRR|PDF|Raman|FTIR|IR|SEM|TEM|STM|STS)\b", re.I)
ABS_EDGE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z][a-z]?)[_\-\s]*(K|L[1-3]|M[1-5])(?:[_\-\s]*edge)?(?![A-Za-z0-9])", re.I)
EDGE_RE = re.compile(r"(?<![A-Za-z0-9])(K|L[1-3]|M[1-5])(?:[_\-\s]*edge)?(?![A-Za-z0-9])", re.I)


def safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def lower(x: Any) -> str:
    return safe_str(x).strip().lower()


def is_number(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except Exception:
        return False


def integrate(y: np.ndarray, x: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def as_array(values: Any) -> Optional[np.ndarray]:
    if values is None:
        return None
    try:
        out = np.asarray(values, dtype=float)
    except Exception:
        return None
    if out.ndim != 1 or out.size == 0 or not np.all(np.isfinite(out)):
        return None
    return out


def get_path(data: Dict[str, Any], dotted: str) -> Any:
    node: Any = data
    for part in dotted.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def flatten_scalars(data: Any, prefix: str = "", depth: int = 5) -> Dict[str, Any]:
    if depth < 0:
        return {}
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if str(key).startswith("_"):
                continue
            child = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_scalars(value, child, depth - 1))
    elif isinstance(data, list):
        if len(data) <= 8 and all(not isinstance(v, (dict, list)) for v in data):
            out[prefix] = ", ".join(safe_str(v) for v in data)
    elif isinstance(data, (str, int, float, bool)) or data is None:
        out[prefix] = data
    return out


def find_first_key(data: Any, names: Sequence[str], depth: int = 6) -> Any:
    keys = {lower(k) for k in names}
    if depth < 0:
        return None
    if isinstance(data, dict):
        for key, value in data.items():
            if lower(key) in keys and not isinstance(value, (dict, list)):
                return value
        for value in data.values():
            found = find_first_key(value, names, depth - 1)
            if found not in [None, ""]:
                return found
    elif isinstance(data, list) and len(data) <= 30:
        for value in data:
            found = find_first_key(value, names, depth - 1)
            if found not in [None, ""]:
                return found
    return None


def descriptor_values(record: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for output in (record.get("descriptors") or {}).get("outputs") or []:
        label = output.get("label") or output.get("name") or "descriptor_output"
        for desc in output.get("descriptors") or []:
            name = desc.get("name")
            if name:
                out[f"descriptors.{name}"] = desc.get("value")
                out[f"descriptors.{label}.{name}"] = desc.get("value")
    return out


def canonical_element(x: Any) -> str:
    text = safe_str(x)
    match = ABS_EDGE_RE.search(text)
    if match:
        token = match.group(1)
    else:
        tokens = re.findall(r"[A-Z][a-z]?", text)
        if not tokens:
            return ""
        token = tokens[0]
    return token.upper() if len(token) == 1 else token.capitalize()


def canonical_edge(x: Any) -> str:
    text = safe_str(x)
    match = EDGE_RE.search(text)
    return match.group(1).upper() if match else ""


def parse_absorber_edge_text(*values: Any) -> Tuple[str, str]:
    for value in values:
        text = safe_str(value)
        match = ABS_EDGE_RE.search(text)
        if match:
            absorber = canonical_element(match.group(1))
            edge = canonical_edge(match.group(2))
            return absorber, edge
    return "", ""


def infer_edge(x: Optional[np.ndarray]) -> Tuple[str, str, str]:
    if x is None or not x.size:
        return "unknown", "unknown", "not inferred"
    xmin, xmax = float(np.min(x)), float(np.max(x))
    mid = 0.5 * (xmin + xmax)
    for el, edge, lo, hi in EDGE_WINDOWS:
        if lo <= mid <= hi or xmin <= lo <= xmax or xmin <= hi <= xmax:
            return el, edge, "inferred from energy range"
    return "unknown", "unknown", "not inferred"


def absorber_edge(record: Dict[str, Any], x: Optional[np.ndarray]) -> Tuple[str, str, str]:
    raw_abs = (
        get_path(record, "system.absorber")
        or get_path(record, "system.absorbing_element")
        or get_path(record, "system.configuration.absorber")
        or get_path(record, "measurement.absorber")
        or find_first_key(record, ["absorber", "absorbing_element", "absorber_element"])
    )
    raw_edge = (
        get_path(record, "system.edge")
        or get_path(record, "system.configuration.edge")
        or get_path(record, "measurement.edge")
        or find_first_key(record, ["edge", "absorption_edge"])
    )
    parsed_abs, parsed_edge = parse_absorber_edge_text(raw_abs, raw_edge)
    abs_el = canonical_element(raw_abs) or parsed_abs
    edge = canonical_edge(raw_edge) or canonical_edge(raw_abs) or parsed_edge
    if abs_el and edge:
        return abs_el, edge, "metadata"
    inferred_abs, inferred_edge, source = infer_edge(x)
    return abs_el or inferred_abs, edge or inferred_edge, source


def edge_group(absorber: str, edge: str) -> str:
    if not absorber or absorber == "unknown" or not edge or edge == "unknown":
        return "Unknown absorber/edge"
    return f"{absorber} {edge}-edge"


def classify_series(x_name: str, x_unit: str, y_name: str) -> Tuple[bool, str, str]:
    xn, xu, yn = lower(x_name), lower(x_unit), lower(y_name)
    if xn in {"q", "qz"} or "angstrom^-1" in xu or "1/angstrom" in xu:
        return False, "not_xas_scattering", "x-axis is q/scattering vector"
    if "2theta" in xn or "two theta" in xn or "diffraction" in xn:
        return False, "not_xas_diffraction", "x-axis is diffraction/scattering angle"
    if "binding" in xn or "photoelectron" in xn or "xps" in xn or "photoelectron" in yn:
        return False, "not_xas_photoelectron", "x-axis/channel indicates XPS/photoelectron spectroscopy"
    has_energy_x = any(k in xn for k in ENERGY_NAMES) or xn in {"e", "energy_ev"}
    has_energy_unit = any(u == xu or u in xu for u in ENERGY_UNITS)
    has_signal = any(k in yn for k in SIGNAL_NAMES)
    if has_energy_x and has_energy_unit and has_signal:
        return True, "xas_like", "energy-axis XAS/XANES-like spectrum with eV/keV unit"
    if has_energy_x and has_energy_unit:
        return True, "xas_like_uncertain_y", "energy-axis spectrum with eV/keV unit; y name is not standard XAS"
    return False, "not_xas_unknown_axis", "no incident-energy x-axis with eV/keV units"


def read_records_from_uploads(files: Sequence[Any]) -> List[Tuple[str, Dict[str, Any]]]:
    records: List[Tuple[str, Dict[str, Any]]] = []
    for uploaded in files or []:
        name = getattr(uploaded, "name", "uploaded")
        payload = uploaded.getvalue()
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                for member in sorted(zf.namelist()):
                    if not member.lower().endswith(".json") or member.endswith("/"):
                        continue
                    try:
                        data = json.loads(zf.read(member).decode("utf-8"))
                    except Exception:
                        continue
                    if isinstance(data, dict):
                        records.append((member, data))
        elif name.lower().endswith(".json"):
            data = json.loads(payload.decode("utf-8"))
            if isinstance(data, dict):
                records.append((name, data))
    return records


def choose_channel(channels: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not channels:
        return {}
    for channel in channels:
        if lower(channel.get("role")) == "primary_signal":
            return channel
    for channel in channels:
        if any(token in lower(channel.get("name")) for token in SIGNAL_NAMES):
            return channel
    return channels[0]


def record_technique(record: Dict[str, Any]) -> str:
    return safe_str(
        get_path(record, "system.technique")
        or get_path(record, "system.configuration.technique")
        or get_path(record, "measurement.technique")
        or find_first_key(record, ["technique", "method"])
    )


def make_row(source: str, record: Dict[str, Any], series_id: str, x: Optional[np.ndarray], y: Optional[np.ndarray], x_name: str, x_unit: str, y_name: str, y_unit: str) -> Dict[str, Any]:
    absorber, edge, edge_source = absorber_edge(record, x)
    group = edge_group(absorber, edge)
    technique = record_technique(record)
    if x is None or y is None or x.size != y.size:
        is_xas, classification, reason = False, "not_plottable_invalid_xy", "missing/non-numeric x or y, or length mismatch"
    else:
        is_xas, classification, reason = classify_series(x_name, x_unit, y_name)
    if NON_XAS_TECHNIQUE_RE.search(technique) and not XAS_TECHNIQUE_RE.search(technique):
        is_xas, classification, reason = False, "not_xas_technique", f"record technique is {technique}, not XAS"
    metadata = {
        **flatten_scalars(record),
        **descriptor_values(record),
        "series.series_id": series_id,
        "series.x_name": x_name,
        "series.x_unit": x_unit,
        "series.y_name": y_name,
        "series.y_unit": y_unit,
        "derived.absorber": absorber,
        "derived.edge": edge,
        "derived.edge_group": group,
        "derived.n_points": int(x.size if x is not None else 0),
    }
    return {
        "source_name": source,
        "record_id": record.get("record_id"),
        "series_id": series_id,
        "technique": technique,
        "formula": get_path(record, "sample.material.formula") or get_path(record, "sample.material.name"),
        "absorber": absorber,
        "edge": edge,
        "edge_group": group,
        "edge_source": edge_source,
        "x_name": x_name,
        "x_unit": x_unit,
        "y_name": y_name,
        "y_unit": y_unit,
        "n_points": int(x.size if x is not None else 0),
        "is_xas": is_xas,
        "classification": classification,
        "reason": reason,
        "record": record,
        "metadata": metadata,
        "x": x,
        "y": y,
    }


def rows_from_records(records: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source, record in records:
        added = False
        for i, series in enumerate((record.get("measurement") or {}).get("series") or []):
            xobj = (series.get("independent_variables") or [{}])[0]
            yobj = choose_channel(series.get("channels") or [])
            rows.append(make_row(
                source, record, safe_str(series.get("series_id")) or f"series_{i}",
                as_array(xobj.get("values")), as_array(yobj.get("values")),
                safe_str(xobj.get("name")), safe_str(xobj.get("unit")),
                safe_str(yobj.get("name")), safe_str(yobj.get("unit")),
            ))
            added = True
        for i, out in enumerate((record.get("descriptors") or {}).get("outputs") or []):
            x = as_array(out.get("energy") or out.get("energy_eV") or out.get("x") or out.get("grid"))
            y = as_array(out.get("mu") or out.get("absorption") or out.get("intensity") or out.get("y") or out.get("values"))
            if x is not None and y is not None and x.size == y.size:
                rows.append(make_row(
                    source, record, safe_str(out.get("label") or out.get("name") or f"descriptor_output_{i}"),
                    x, y, "energy", "eV",
                    safe_str(out.get("channel") or out.get("name") or "absorption"),
                    safe_str(out.get("unit") or "a.u."),
                ))
                added = True
        if not added:
            rows.append(make_row(source, record, "", None, None, "", "", "", ""))
    return rows


def summary_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    keys = ["source_name", "record_id", "series_id", "technique", "formula", "absorber", "edge", "edge_group", "edge_source", "x_name", "x_unit", "y_name", "y_unit", "n_points", "is_xas", "classification", "reason"]
    return [{key: row.get(key) for key in keys} for row in rows]


def rows_to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    out = io.StringIO()
    writer = csv.DictWriter(out, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def minmax(y: np.ndarray) -> np.ndarray:
    span = float(np.max(y) - np.min(y))
    return np.zeros_like(y) if span == 0 else (y - float(np.min(y))) / span


def target_value(row: Dict[str, Any], path: str) -> Any:
    if path.startswith(("derived.", "series.", "descriptors.")):
        return (row.get("metadata") or {}).get(path)
    value = get_path(row.get("record") or {}, path)
    if value is not None:
        return value
    return (row.get("metadata") or {}).get(path)


def label_fields(rows: Sequence[Dict[str, Any]]) -> List[str]:
    preferred = ["sample.material.formula", "sample.material.name", "derived.absorber", "derived.edge", "derived.edge_group", "system.technique", "source_type", "record_domain"]
    keys = set()
    for row in rows:
        for key, value in (row.get("metadata") or {}).items():
            if value is not None and not (isinstance(value, str) and len(value) > 160):
                keys.add(key)
    return [key for key in preferred if key in keys] + sorted(keys - set(preferred))
