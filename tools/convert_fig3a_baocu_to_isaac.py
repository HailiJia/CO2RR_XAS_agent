#!/usr/bin/env python3
"""Convert BaO/Cu Fig. 3a XANES data to an ISAAC XAS record.

Source data:
  https://github.com/onealshu/CO2_alcohol_BaOCu/blob/main/exp_Data/fig3a.dat

Paper:
  Xu et al., Nature Catalysis 5, 1081-1088 (2022)
  https://www.nature.com/articles/s41929-022-00880-6

The input table is organized as repeated (energy, intensity) column pairs.
The header labels are the series labels from Fig. 3a:
    0, 3, 6, 9, 15, 18, 21, 24, 30, 36, 42, 51, 60

The script interprets those labels as time labels in minutes because Fig. 3a is
an operando time-series XANES panel. If you confirm a different unit from the
caption/source metadata, pass --series-label-unit accordingly.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import re
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, List, Tuple

DEFAULT_URL = "https://raw.githubusercontent.com/onealshu/CO2_alcohol_BaOCu/main/exp_Data/fig3a.dat"
PAPER_URL = "https://www.nature.com/articles/s41929-022-00880-6"
REPO_FILE_URL = "https://github.com/onealshu/CO2_alcohol_BaOCu/blob/main/exp_Data/fig3a.dat"
CROCKFORD = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"


def now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def generate_record_id() -> str:
    """Return a ULID-like 26-character Crockford-base32 identifier."""
    ts_ms = int(_dt.datetime.now(_dt.timezone.utc).timestamp() * 1000)
    value = (ts_ms << 80) | uuid.uuid4().int >> 48
    chars = []
    for _ in range(26):
        chars.append(CROCKFORD[value & 31])
        value >>= 5
    return "".join(reversed(chars))


def read_text(source: str) -> Tuple[str, str]:
    """Return (text, source_kind)."""
    if source.startswith(("http://", "https://")):
        req = urllib.request.Request(source, headers={"User-Agent": "CO2RR-XAS-agent/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8"), "url"
    return Path(source).read_text(encoding="utf-8"), "file"


def parse_fig3a_dat(text: str) -> Tuple[List[str], List[List[float]], List[List[float]]]:
    """Parse repeated (energy, signal) pairs."""
    raw_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not raw_lines:
        raise ValueError("Input data file is empty.")

    labels = re.split(r"\s+", raw_lines[0].strip())
    labels = [label for label in labels if label != ""]
    if not labels:
        raise ValueError("Could not parse series labels from the first line.")

    n_series = len(labels)
    energies_by_series: List[List[float]] = [[] for _ in labels]
    signal_by_series: List[List[float]] = [[] for _ in labels]

    for lineno, line in enumerate(raw_lines[1:], start=2):
        toks = [tok for tok in re.split(r"\s+", line.strip()) if tok]
        if len(toks) != 2 * n_series:
            raise ValueError(
                f"Line {lineno} has {len(toks)} numeric fields; expected {2*n_series} "
                f"for {n_series} repeated energy/signal pairs."
            )
        vals = [float(tok) for tok in toks]
        for i in range(n_series):
            energies_by_series[i].append(vals[2 * i])
            signal_by_series[i].append(vals[2 * i + 1])

    ref = energies_by_series[0]
    for i, grid in enumerate(energies_by_series[1:], start=1):
        if len(grid) != len(ref) or any(abs(a - b) > 1e-8 for a, b in zip(ref, grid)):
            raise ValueError(f"Energy grid for series {labels[i]} does not match series {labels[0]}.")

    return labels, energies_by_series, signal_by_series


def build_record(
    text: str,
    labels: List[str],
    energies_by_series: List[List[float]],
    signal_by_series: List[List[float]],
    *,
    record_id: str,
    created_utc: str,
    series_label_unit: str,
    source_uri: str,
    source_sha256: str,
) -> Dict[str, Any]:
    series = []
    for label, energy, signal in zip(labels, energies_by_series, signal_by_series):
        safe_label = str(label).replace(".", "p").replace("-", "m")
        series.append(
            {
                "series_id": f"fig3a_{safe_label}_{series_label_unit}",
                "independent_variables": [
                    {
                        "name": "incident_energy",
                        "unit": "eV",
                        "values": energy,
                    }
                ],
                "channels": [
                    {
                        "name": "absorption",
                        "unit": "arb",
                        "role": "primary_signal",
                        "values": signal,
                    }
                ],
            }
        )

    return {
        "isaac_record_version": "1.05",
        "record_id": record_id,
        "record_type": "evidence",
        "record_domain": "characterization",
        "source_type": "facility",
        "tags": [
            "co2rr",
            "operando-xanes",
            "cu-k-edge",
            "baocu",
            "fig3a",
            "nature-catalysis-2022",
        ],
        "timestamps": {
            "created_utc": created_utc,
        },
        "sample": {
            "material": {
                "name": "BaO/Cu electrocatalyst",
                "formula": "BaOCu",
                "provenance": "synthesized",
                "notes": (
                    "Cu catalyst decorated with barium oxide. Public processed/digitized Fig. 3a "
                    "XANES data from Xu et al., Nature Catalysis 5, 1081-1088 (2022)."
                ),
            },
            "sample_form": "electrode",
        },
        "system": {
            "domain": "experimental",
            "technique": "XAS",
            "absorber": "Cu",
            "edge": "K",
            "configuration": {
                "measurement_mode": "operando XANES",
                "absorber": "Cu",
                "edge": "K",
                "source_panel": "Fig. 3a",
                "series_label_unit": series_label_unit,
                "series_labels": labels,
            },
        },
        "context": {
            "environment": "operando",
            "electrochemistry": {
                "reaction": "CO2RR",
                "notes": (
                    "Paper reports Cu catalysts decorated with alkaline-earth metal oxides and "
                    "uses operando X-ray absorption spectroscopy to probe surface states under "
                    "CO2RR conditions. Exact potential/electrolyte for this extracted data file "
                    "is not encoded in fig3a.dat."
                ),
            },
        },
        "measurement": {
            "processing": {
                "type": "processed_figure_data",
                "notes": (
                    "Converted from exp_Data/fig3a.dat. The table stores repeated energy/signal "
                    "pairs for each header label. Signal values are stored as arbitrary-unit "
                    "absorption because the source data file does not provide a channel unit."
                ),
            },
            "series": series,
            "qc": {
                "status": "usable",
                "notes": (
                    "Automated conversion checked that all spectra share the same incident-energy "
                    "grid and that energy units are eV. Scientific interpretation should verify "
                    "the figure caption and original processing details."
                ),
            },
        },
        "assets": [
            {
                "asset_id": "source_fig3a_dat",
                "content_role": "source_data",
                "uri": source_uri,
                "media_type": "text/plain",
                "sha256": source_sha256,
                "notes": "Public processed data file used to generate this ISAAC record.",
            },
            {
                "asset_id": "source_publication",
                "content_role": "source_publication",
                "uri": PAPER_URL,
                "media_type": "text/html",
                "notes": "Nature Catalysis article associated with the data.",
            },
            {
                "asset_id": "source_github_file",
                "content_role": "source_repository_file",
                "uri": REPO_FILE_URL,
                "media_type": "text/html",
                "notes": "GitHub blob page for the source data file.",
            },
        ],
        "links": [
            {
                "rel": "derived_from",
                "target": REPO_FILE_URL,
                "basis": "public_processed_data_file",
                "notes": "Converted directly from fig3a.dat in the public GitHub repository.",
            },
            {
                "rel": "described_by",
                "target": PAPER_URL,
                "basis": "source_publication",
                "notes": "Publication describes BaO/Cu CO2RR catalyst and operando X-ray absorption spectroscopy context.",
            },
        ],
    }


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
            "User-Agent": "CO2RR-XAS-agent/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default=DEFAULT_URL, help="Local fig3a.dat path or URL.")
    parser.add_argument("--output", default="baocu_fig3a_isaac_xas_record.json", help="Output ISAAC JSON path.")
    parser.add_argument("--record-id", default="", help="Optional record_id. If omitted, a ULID-like ID is generated.")
    parser.add_argument("--created-utc", default="", help="created_utc timestamp. Defaults to current UTC time.")
    parser.add_argument("--series-label-unit", default="min", help="Unit for header labels, encoded in series_id/configuration.")
    parser.add_argument("--validate", action="store_true", help="Validate against ISAAC Portal using ISAAC_URL/ISAAC_KEY.")
    parser.add_argument("--upload", action="store_true", help="Validate then upload to ISAAC Portal using ISAAC_URL/ISAAC_KEY.")
    args = parser.parse_args(argv)

    text, source_kind = read_text(args.input)
    labels, energies_by_series, signal_by_series = parse_fig3a_dat(text)
    sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()
    source_uri = args.input if source_kind == "url" else str(Path(args.input).resolve())

    record = build_record(
        text,
        labels,
        energies_by_series,
        signal_by_series,
        record_id=args.record_id or generate_record_id(),
        created_utc=args.created_utc or now_utc(),
        series_label_unit=args.series_label_unit,
        source_uri=source_uri,
        source_sha256=sha256,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(f"Wrote {out_path}")
    print(f"record_id: {record['record_id']}")
    print(f"series: {len(record['measurement']['series'])}")
    print(f"points per series: {len(record['measurement']['series'][0]['independent_variables'][0]['values'])}")

    if args.validate or args.upload:
        isaac_url = os.environ.get("ISAAC_URL", "https://isaac.slac.stanford.edu/portal/api").rstrip("/")
        isaac_key = os.environ.get("ISAAC_KEY", "")
        if not isaac_key:
            raise SystemExit("Set ISAAC_KEY before using --validate or --upload.")
        validation = post_json(f"{isaac_url}/validate", isaac_key, record)
        print(json.dumps({"validation": validation}, indent=2))
        if not validation.get("valid"):
            raise SystemExit("ISAAC validation failed; not uploading.")
        if args.upload:
            created = post_json(f"{isaac_url}/records", isaac_key, record)
            print(json.dumps({"create": created}, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
