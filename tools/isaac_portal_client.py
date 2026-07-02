"""ISAAC Portal API helpers used by the CO2RR XAS Streamlit app.

Credentials are intentionally read from environment variables so API keys are
not stored in the app state, logs, or repository:

    export ISAAC_URL="https://isaac.slac.stanford.edu/portal/api"
    export ISAAC_KEY="..."

The client implements the portal endpoints currently used by the UI:
GET /records, GET /records/{record_id}, POST /validate, and POST /records.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_ISAAC_URL = "https://isaac.slac.stanford.edu/portal/api"

EDGE_E0_EV = {
    "C": {"K": 284}, "N": {"K": 400}, "O": {"K": 532},
    "Ti": {"K": 4966, "L3": 455},
    "V": {"K": 5465, "L3": 512},
    "Cr": {"K": 5989, "L3": 575},
    "Mn": {"K": 6539, "L3": 640},
    "Fe": {"K": 7112, "L3": 708},
    "Co": {"K": 7709, "L3": 779},
    "Ni": {"K": 8333, "L3": 852},
    "Cu": {"K": 8979, "L3": 932},
    "Zn": {"K": 9659, "L3": 1022},
    "Pt": {"L3": 11564, "L2": 13273},
    "Au": {"L3": 11919, "L2": 13734},
    "Ir": {"L3": 11215, "L2": 12824},
}


class ISAACPortalError(RuntimeError):
    """Raised when the ISAAC Portal request or payload handling fails."""


@dataclass
class ISAACPortalResponse:
    payload: Any
    status_code: int
    headers: Dict[str, str]

    @property
    def total_count(self) -> Optional[int]:
        raw = self.headers.get("X-Total-Count") or self.headers.get("x-total-count")
        try:
            return int(raw) if raw is not None else None
        except Exception:
            return None


def get_path(data: Any, path: Sequence[str], default: Any = "") -> Any:
    node = data
    for part in path:
        if not isinstance(node, dict):
            return default
        node = node.get(part, default)
    return node


def elements_from_formula(formula: Any) -> List[str]:
    return re.findall(r"[A-Z][a-z]?", str(formula or ""))


def energy_values(record: Dict[str, Any]) -> List[float]:
    vals: List[float] = []
    for series in get_path(record, ["measurement", "series"], []) or []:
        for iv in series.get("independent_variables", []) or []:
            if "energy" in str(iv.get("name", "")).lower():
                for value in iv.get("values", []) or []:
                    try:
                        vals.append(float(value))
                    except Exception:
                        pass
    for output in get_path(record, ["descriptors", "outputs"], []) or []:
        for key in ["energy", "energy_eV", "x", "grid"]:
            raw = output.get(key)
            if isinstance(raw, list):
                for value in raw:
                    try:
                        vals.append(float(value))
                    except Exception:
                        pass
                break
    return vals


def infer_absorber_edge(record: Dict[str, Any], formula: str) -> Tuple[str, str]:
    vals = energy_values(record)
    if not vals:
        return "", ""
    emin, emax = min(vals), max(vals)
    for elem in elements_from_formula(formula):
        for edge, e0 in EDGE_E0_EV.get(elem, {}).items():
            if emin - 100 <= float(e0) <= emax + 100:
                return elem, edge
    return "", ""


def record_xas_summary(record: Dict[str, Any], source_name: str = "") -> Dict[str, Any]:
    formula = get_path(record, ["sample", "material", "formula"], "")
    material_name = get_path(record, ["sample", "material", "name"], "")
    technique = (
        get_path(record, ["system", "technique"], "")
        or get_path(record, ["system", "configuration", "technique"], "")
        or get_path(record, ["measurement", "technique"], "")
    )
    absorber = (
        get_path(record, ["system", "absorber"], "")
        or get_path(record, ["system", "absorbing_element"], "")
        or get_path(record, ["system", "configuration", "absorber"], "")
        or get_path(record, ["measurement", "absorber"], "")
    )
    edge = (
        get_path(record, ["system", "edge"], "")
        or get_path(record, ["system", "configuration", "edge"], "")
        or get_path(record, ["measurement", "edge"], "")
    )
    if not absorber or not edge:
        inferred_abs, inferred_edge = infer_absorber_edge(record, str(formula))
        absorber = absorber or inferred_abs
        edge = edge or inferred_edge
    is_xas = str(technique).strip().upper() == "XAS" or bool(absorber and edge and energy_values(record))
    return {
        "source_name": source_name,
        "record_id": record.get("record_id", ""),
        "created_at": record.get("created_at", ""),
        "record_domain": record.get("record_domain", ""),
        "record_type": record.get("record_type", ""),
        "source_type": record.get("source_type", ""),
        "formula": formula,
        "material_name": material_name,
        "absorber": absorber,
        "edge": edge,
        "technique": technique,
        "is_xas": bool(is_xas),
    }


def xas_summary_table(records: Sequence[Tuple[str, Dict[str, Any]]], xas_only: bool = True) -> List[Dict[str, Any]]:
    rows = [record_xas_summary(record, source_name=source) for source, record in records]
    if xas_only:
        rows = [row for row in rows if row.get("is_xas")]
    return rows


class ISAACPortalClient:
    """Small JSON client for ISAAC Portal record endpoints."""

    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None, timeout_s: int = 60):
        self.base_url = (base_url or os.environ.get("ISAAC_URL") or DEFAULT_ISAAC_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("ISAAC_KEY") or ""
        self.timeout_s = int(timeout_s)
        if not self.api_key:
            raise ISAACPortalError("Missing ISAAC_KEY environment variable.")

    def _request(self, method: str, path: str, payload: Any = None, params: Optional[Dict[str, Any]] = None) -> ISAACPortalResponse:
        url = self.base_url + "/" + path.lstrip("/")
        if params:
            clean = {k: v for k, v in params.items() if v not in [None, ""]}
            if clean:
                url += "?" + urlencode(clean)
        body = None
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"}
        if payload is not None:
            body = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"
        req = Request(url, data=body, headers=headers, method=method.upper())
        try:
            with urlopen(req, timeout=self.timeout_s) as resp:
                text = resp.read().decode("utf-8")
                parsed = json.loads(text) if text.strip() else None
                return ISAACPortalResponse(parsed, int(resp.status), dict(resp.headers.items()))
        except HTTPError as exc:
            text = exc.read().decode("utf-8", errors="replace")
            try:
                parsed = json.loads(text) if text.strip() else {"text": text}
            except Exception:
                parsed = {"text": text}
            raise ISAACPortalError(f"ISAAC Portal HTTP {exc.code} for {method} {path}: {parsed}") from exc
        except URLError as exc:
            raise ISAACPortalError(f"ISAAC Portal request failed for {method} {path}: {exc}") from exc

    def list_records(self, limit: int = 50, offset: int = 0, **filters: Any) -> Dict[str, Any]:
        params = {"limit": int(limit), "offset": int(offset)}
        params.update(filters)
        response = self._request("GET", "records", params=params)
        records = response.payload if isinstance(response.payload, list) else []
        return {"records": records, "total_count": response.total_count, "status_code": response.status_code}

    def fetch_record(self, record_id: str) -> Dict[str, Any]:
        if not record_id:
            raise ISAACPortalError("Missing record_id.")
        response = self._request("GET", f"records/{record_id}")
        if not isinstance(response.payload, dict):
            raise ISAACPortalError(f"Unexpected record payload for {record_id}: {type(response.payload).__name__}")
        return response.payload

    def fetch_records(self, record_ids: Iterable[str]) -> List[Tuple[str, Dict[str, Any]]]:
        out: List[Tuple[str, Dict[str, Any]]] = []
        for record_id in record_ids:
            rid = str(record_id or "").strip()
            if rid:
                out.append((rid, self.fetch_record(rid)))
        return out

    def validate_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", "validate", payload=record)
        if not isinstance(response.payload, dict):
            raise ISAACPortalError(f"Unexpected validation payload: {type(response.payload).__name__}")
        return response.payload

    def create_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", "records", payload=record)
        if not isinstance(response.payload, dict):
            raise ISAACPortalError(f"Unexpected create payload: {type(response.payload).__name__}")
        return response.payload

    def validate_then_create(self, record: Dict[str, Any]) -> Dict[str, Any]:
        validation = self.validate_record(record)
        if not validation.get("valid"):
            return {"success": False, "reason": "validation_failed", "validation": validation}
        created = self.create_record(record)
        return {"success": bool(created.get("success", True)), "validation": validation, "create": created, "record_id": created.get("record_id") or record.get("record_id")}
