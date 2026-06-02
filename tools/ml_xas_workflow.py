"""CO2RR ISAAC XAS ML dataset assembly and record helpers.

This module implements the deterministic, lightweight part of the
CO2RR_ISAAC_XAS_ML skill: ingest ISAAC XAS records, assemble spectrum- and
condition-level tables, apply XAS-aware preprocessing/featurization, and write
ISAAC-compatible ML training/prediction record scaffolds. Heavy model training
(backends such as XGBoost, LightGBM, PyTorch) can consume these outputs.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tools.utils import ensure_dir, generate_uuid

SUPPORTED_EDGE_LABELS = {
    ("Cu", "K"),
    ("Ni", "K"),
    ("Au", "L3"),
    ("Ir", "L3"),
    ("Pt", "L3"),
    ("C", "K"),
}
DEFAULT_GROUP_KEYS = [
    "condition_id",
    "structure_id",
    "sample_id",
    "catalyst_composition",
    "surface_facet",
    "adsorbate_identity",
    "adsorption_site",
    "experimental_condition",
]
DEFAULT_MODELS = ["PLS", "SVM", "XGBoost", "LightGBM", "MLP", "1D-CNN"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"ISAAC record must be a JSON object: {path}")
    return data


def load_isaac_records(paths: Sequence[str]) -> List[Dict[str, Any]]:
    """Load one or more ISAAC record JSON files or directories containing them."""
    records: List[Dict[str, Any]] = []
    for item in paths:
        path = Path(item)
        if path.is_dir():
            json_paths = sorted(path.rglob("*.json"))
        else:
            json_paths = [path]
        for json_path in json_paths:
            if not json_path.exists():
                raise FileNotFoundError(f"ISAAC record not found: {json_path}")
            record = _load_json(json_path)
            if "measurement" in record and "record_id" in record:
                record["_source_path"] = str(json_path)
                records.append(record)
    return records


def _first_values(series: Dict[str, Any], key: str) -> Tuple[List[float], str]:
    containers = series.get(key) or []
    if not containers:
        return [], "not_specified"
    first = containers[0]
    return list(first.get("values") or []), first.get("unit", "not_specified")


def _infer_edge(series: Dict[str, Any], record: Dict[str, Any]) -> str:
    text = " ".join([
        str(series.get("series_id", "")),
        json.dumps(record.get("computation", {})),
        json.dumps(record.get("measurement", {}).get("processing", {})),
    ]).lower()
    if "l3" in text or "l_3" in text or "l-edge" in text:
        return "L3"
    if "l2" in text:
        return "L2"
    if "l23" in text or "l2,3" in text:
        return "L23"
    return "K"


def _infer_absorber(record: Dict[str, Any], edge: str, series: Optional[Dict[str, Any]] = None) -> str:
    sample = record.get("sample", {}) if isinstance(record.get("sample"), dict) else {}
    catalyst = sample.get("catalyst", {}) if isinstance(sample.get("catalyst"), dict) else {}
    elements = catalyst.get("elements") or []
    series_text = json.dumps(series or {}).lower()

    # Carbon K-edge spectra are adsorbate-sensitive; only infer C when the
    # spectrum-level text indicates a carbon edge to avoid mislabeling metal K
    # spectra from catalyst records that also contain carbonaceous adsorbates.
    if edge == "K" and any(token in series_text for token in ["c_k", "c k", "carbon", "c-edge", "c_edge"]):
        return "C"

    if elements:
        for element in elements:
            if (element, edge) in SUPPORTED_EDGE_LABELS:
                return element
        return str(elements[0])

    formula = (sample.get("material", {}) or {}).get("formula", "")
    for element in ["Cu", "Ni", "Au", "Ir", "Pt", "C"]:
        if element in formula and (element, edge) in SUPPORTED_EDGE_LABELS:
            return element

    adsorbate_formula = (sample.get("adsorbate", {}) or {}).get("formula", "")
    if edge == "K" and "C" in adsorbate_formula:
        return "C"

    # Last resort for legacy records with no structured catalyst metadata.
    return "Cu"


def _labels_from_record(record: Dict[str, Any]) -> Dict[str, Any]:
    sample = record.get("sample", {}) if isinstance(record.get("sample"), dict) else {}
    catalyst = sample.get("catalyst", {}) if isinstance(sample.get("catalyst"), dict) else {}
    adsorbate = sample.get("adsorbate", {}) if isinstance(sample.get("adsorbate"), dict) else {}
    ml_labels = sample.get("ml_labels", {}) if isinstance(sample.get("ml_labels"), dict) else {}
    motif = ml_labels.get("adsorption_motif") or adsorbate.get("binding_mode") or adsorbate.get("adsorption_site")
    return {
        "adsorbate_identity": ml_labels.get("adsorbate_identity") or adsorbate.get("identity"),
        "adsorption_motif": motif,
        "oxidation_coordination_class": ml_labels.get("oxidation_coordination_class"),
        "catalyst_composition": catalyst.get("composition"),
        "surface_facet": catalyst.get("surface_facet"),
        "site_type": catalyst.get("site_type"),
    }


def build_spectrum_table(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build one row per ISAAC measurement.series spectrum."""
    rows: List[Dict[str, Any]] = []
    for record in records:
        measurement = record.get("measurement", {}) if isinstance(record.get("measurement"), dict) else {}
        sample = record.get("sample", {}) if isinstance(record.get("sample"), dict) else {}
        catalyst = sample.get("catalyst", {}) if isinstance(sample.get("catalyst"), dict) else {}
        adsorbate = sample.get("adsorbate", {}) if isinstance(sample.get("adsorbate"), dict) else {}
        labels = _labels_from_record(record)
        for series in measurement.get("series", []) or []:
            energy, energy_unit = _first_values(series, "independent_variables")
            intensity, intensity_unit = _first_values(series, "channels")
            edge = _infer_edge(series, record)
            absorber = _infer_absorber(record, edge, series)
            condition_id = (
                series.get("condition_id")
                or sample.get("condition_id")
                or sample.get("structure_id")
                or catalyst.get("structure_id")
                or record.get("record_id")
            )
            rows.append({
                "record_id": record.get("record_id"),
                "series_id": series.get("series_id"),
                "condition_id": condition_id,
                "record_domain": record.get("record_domain"),
                "source_type": record.get("source_type"),
                "absorbing_element": absorber,
                "edge": edge,
                "edge_key": f"{absorber}_{edge}",
                "energy": energy,
                "intensity": intensity,
                "energy_units": energy_unit,
                "intensity_units": intensity_unit,
                "normalization_metadata": measurement.get("normalization", {}),
                "processing_metadata": measurement.get("processing", {}),
                "labels": labels,
                "descriptors": record.get("descriptors", {}),
                "catalyst": catalyst,
                "adsorbate": adsorbate,
                "structure_descriptors": sample.get("structure_descriptors", {}),
                "experiment": sample.get("experiment", {}),
                "source_path": record.get("_source_path"),
                "missing_required_fields": [],
            })
    return rows


def validate_spectrum_row(row: Dict[str, Any], min_points: int = 5) -> List[str]:
    missing = []
    energy = np.asarray(row.get("energy") or [], dtype=float)
    intensity = np.asarray(row.get("intensity") or [], dtype=float)
    if energy.size < min_points:
        missing.append("energy")
    if intensity.size < min_points:
        missing.append("intensity")
    if energy.size != intensity.size:
        missing.append("energy_intensity_length_match")
    if energy.size and not np.all(np.isfinite(energy)):
        missing.append("finite_energy")
    if intensity.size and not np.all(np.isfinite(intensity)):
        missing.append("finite_intensity")
    for field in ["record_id", "series_id", "condition_id", "absorbing_element", "edge"]:
        if not row.get(field):
            missing.append(field)
    return missing


def preprocess_spectrum(
    energy: Sequence[float],
    intensity: Sequence[float],
    grid_points: int = 256,
    normalization: str = "minmax",
) -> Dict[str, Any]:
    """Interpolate, normalize, and derive CDF/first-derivative representations."""
    x = np.asarray(energy, dtype=float)
    y = np.asarray(intensity, dtype=float)
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    grid = np.linspace(float(x.min()), float(x.max()), int(grid_points))
    interp = np.interp(grid, x, y)
    if normalization == "area":
        area = float(np.trapezoid(np.abs(interp), grid)) or 1.0
        norm = interp / area
    else:
        span = float(interp.max() - interp.min()) or 1.0
        norm = (interp - float(interp.min())) / span
    derivative = np.gradient(norm, grid).tolist()
    cdf = np.cumsum(np.maximum(norm, 0.0))
    cdf = (cdf / (float(cdf[-1]) or 1.0)).tolist()
    return {
        "grid": grid.tolist(),
        "normalized": norm.tolist(),
        "derivative": derivative,
        "cdf": cdf,
        "recipe": {
            "interpolation": "linear",
            "grid_points": int(grid_points),
            "normalization": normalization,
            "derivative": True,
            "cdf": True,
        },
    }


def featurize_preprocessed(preprocessed: Dict[str, Any]) -> Dict[str, float]:
    grid = np.asarray(preprocessed["grid"], dtype=float)
    y = np.asarray(preprocessed["normalized"], dtype=float)
    dy = np.asarray(preprocessed["derivative"], dtype=float)
    cdf = np.asarray(preprocessed["cdf"], dtype=float)
    peak_idx = int(np.argmax(y))
    features = {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "max": float(np.max(y)),
        "min": float(np.min(y)),
        "area": float(np.trapezoid(y, grid)),
        "peak_energy": float(grid[peak_idx]),
        "peak_intensity": float(y[peak_idx]),
        "derivative_max": float(np.max(dy)),
        "derivative_min": float(np.min(dy)),
        "cdf_mid_energy": float(grid[int(np.argmin(np.abs(cdf - 0.5)))]),
    }
    # Three stable window-integral descriptors across the normalized energy span.
    bins = np.array_split(np.arange(len(grid)), 3)
    for idx, bin_idx in enumerate(bins, start=1):
        features[f"window_{idx}_area"] = float(np.trapezoid(y[bin_idx], grid[bin_idx])) if len(bin_idx) > 1 else 0.0
    return features


def preprocess_and_featurize_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    processed: List[Dict[str, Any]] = []
    for row in rows:
        row = dict(row)
        row["missing_required_fields"] = validate_spectrum_row(row)
        if row["missing_required_fields"]:
            row["valid_for_ml"] = False
            processed.append(row)
            continue
        prep = preprocess_spectrum(row["energy"], row["intensity"])
        row["valid_for_ml"] = True
        row["preprocessing"] = prep["recipe"]
        row["common_grid"] = prep["grid"]
        row["normalized_intensity"] = prep["normalized"]
        row["derivative"] = prep["derivative"]
        row["cdf"] = prep["cdf"]
        row["features"] = featurize_preprocessed(prep)
        processed.append(row)
    return processed


def build_condition_table(spectrum_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Merge spectra into condition/structure-level samples."""
    grouped: Dict[str, Dict[str, Any]] = {}
    for row in spectrum_rows:
        condition_id = str(row.get("condition_id") or row.get("record_id"))
        entry = grouped.setdefault(condition_id, {
            "condition_id": condition_id,
            "structure_id": row.get("catalyst", {}).get("structure_id") or row.get("condition_id"),
            "sample_id": row.get("record_id"),
            "catalyst_composition": row.get("labels", {}).get("catalyst_composition"),
            "surface_facet": row.get("labels", {}).get("surface_facet"),
            "adsorbate_identity": row.get("labels", {}).get("adsorbate_identity"),
            "adsorption_site": row.get("adsorbate", {}).get("adsorption_site"),
            "experimental_condition": row.get("experiment", {}),
            "target_labels": row.get("labels", {}),
            "domain": row.get("record_domain"),
            "group_id": condition_id,
            "edge_coverage": [],
            "metadata_features": {},
        })
        edge_key = row.get("edge_key")
        if edge_key:
            entry["edge_coverage"].append(edge_key)
            if row.get("features"):
                entry[f"{edge_key}_features"] = row["features"]
    for entry in grouped.values():
        entry["edge_coverage"] = sorted(set(entry["edge_coverage"]))
    return list(grouped.values())


def create_ml_training_record(
    spectrum_rows: Sequence[Dict[str, Any]],
    condition_rows: Sequence[Dict[str, Any]],
    task_name: str = "adsorbate_classification",
) -> Dict[str, Any]:
    class_names = sorted({
        row.get("labels", {}).get("adsorbate_identity")
        for row in spectrum_rows
        if row.get("labels", {}).get("adsorbate_identity")
    })
    sim_ids = sorted({row.get("record_id") for row in spectrum_rows if row.get("record_domain") == "simulation" and row.get("record_id")})
    exp_ids = sorted({row.get("record_id") for row in spectrum_rows if row.get("record_domain") == "experiment" and row.get("record_id")})
    return {
        "isaac_record_version": "1.05",
        "record_id": generate_uuid(),
        "record_type": "ml_training",
        "record_domain": "machine_learning",
        "created_utc": _utc_now(),
        "input_records": {
            "simulation_record_ids": sim_ids,
            "experimental_record_ids": exp_ids,
        },
        "task": {
            "name": task_name,
            "task_type": "classification" if "classification" in task_name else "regression_or_classification",
            "target_field": "ml_labels.adsorbate_identity" if task_name == "adsorbate_classification" else "task_specific",
            "class_names": class_names,
        },
        "dataset": {
            "sample_unit": "measurement.series",
            "fusion_unit": "condition_or_structure",
            "elements": ["Cu", "Au", "Ir", "Ni", "Pt", "C"],
            "edges": ["Cu K", "Ni K", "Au L3", "Ir L3", "Pt L3", "C K"],
            "simulation_record_count": len(sim_ids),
            "experimental_record_count": len(exp_ids),
            "condition_count": len(condition_rows),
            "spectrum_count": len(spectrum_rows),
            "group_keys": DEFAULT_GROUP_KEYS,
        },
        "preprocessing": {
            "interpolation_grid": "edge_specific_common_grid",
            "normalization": "minmax",
            "alignment": "record_energy_axis",
            "broadening": None,
            "smoothing": None,
            "derivative": True,
            "cdf": True,
        },
        "features": {
            "raw_spectrum": True,
            "derivative": True,
            "cdf": True,
            "peak_window_descriptors": True,
            "pca": "planned_grid_search",
            "pls": "planned_grid_search",
        },
        "fusion": {
            "single_edge_models": True,
            "early_fusion": True,
            "late_fusion": True,
            "neural_multibranch": False,
        },
        "models_tested": DEFAULT_MODELS,
        "hpo": {
            "grid_search": {"used_for": ["PCA", "PLS", "SVM", "simple_MLP"]},
            "bayesian_optimization": {"used_for": ["XGBoost", "LightGBM", "MLP", "1D-CNN"]},
        },
        "evaluation": {
            "split_strategy": "grouped_cv",
            "metrics": {},
            "robustness_tests": {},
            "edge_ablation": {},
        },
        "best_model": {
            "model_family": None,
            "feature_set": None,
            "hyperparameters": {},
            "validation_score": None,
        },
        "outputs": {
            "trained_model_asset": None,
            "metrics_asset": None,
            "report_asset": None,
            "feature_importance_asset": None,
        },
    }


def create_ml_prediction_record(
    experimental_rows: Sequence[Dict[str, Any]],
    simulation_rows: Sequence[Dict[str, Any]],
    linked_training_record: Optional[str] = None,
) -> Dict[str, Any]:
    predictions = []
    sim_ids = sorted({row.get("record_id") for row in simulation_rows if row.get("record_id")})
    for row in experimental_rows:
        predictions.append({
            "condition_id": row.get("condition_id"),
            "experimental_record_id": row.get("record_id"),
            "edge_coverage": [row.get("edge_key")] if row.get("edge_key") else [],
            "predicted_class": None,
            "class_probability": None,
            "top_k_classes": [],
            "nearest_simulated_records": [],
            "domain_of_applicability": {
                "status": "not_evaluated",
                "score": None,
                "method": "nearest_simulated_feature_distance",
            },
            "warning_flags": ["model_not_trained"] if not linked_training_record else [],
            "important_energy_regions": [],
        })
    return {
        "isaac_record_version": "1.05",
        "record_id": generate_uuid(),
        "record_type": "ml_prediction",
        "record_domain": "machine_learning",
        "created_utc": _utc_now(),
        "linked_training_record": linked_training_record,
        "input_records": {
            "experimental_record_ids": sorted({row.get("record_id") for row in experimental_rows if row.get("record_id")}),
            "simulation_reference_record_ids": sim_ids,
        },
        "prediction_task": {
            "name": "adsorbate_classification",
            "task_type": "classification",
            "target_field": "ml_labels.adsorbate_identity",
        },
        "predictions": predictions,
        "assets": {
            "prediction_table": None,
            "prediction_report": None,
            "figures": [],
        },
    }


def execute_ml_dataset_assembly(
    isaac_record_paths: Sequence[str],
    output_dir: str,
    mode: str = "simulation_only_training",
    task_name: str = "adsorbate_classification",
) -> Dict[str, Any]:
    """Assemble ML-ready tables and ISAAC-compatible ML record scaffolds."""
    ensure_dir(output_dir)
    records = load_isaac_records(isaac_record_paths)
    spectrum_rows = preprocess_and_featurize_rows(build_spectrum_table(records))
    condition_rows = build_condition_table(spectrum_rows)

    spectrum_path = Path(output_dir) / "spectrum_table.json"
    condition_path = Path(output_dir) / "condition_table.json"
    training_record_path = Path(output_dir) / "ml_training_record.json"
    prediction_record_path = Path(output_dir) / "ml_prediction_record.json"

    spectrum_path.write_text(json.dumps(spectrum_rows, indent=2) + "\n")
    condition_path.write_text(json.dumps(condition_rows, indent=2) + "\n")

    training_record = create_ml_training_record(spectrum_rows, condition_rows, task_name=task_name)
    training_record_path.write_text(json.dumps(training_record, indent=2) + "\n")

    prediction_record = None
    if mode in {"simulation_to_experiment", "sim_to_exp", "prediction"}:
        experimental = [row for row in spectrum_rows if row.get("record_domain") == "experiment"]
        simulation = [row for row in spectrum_rows if row.get("record_domain") == "simulation"]
        prediction_record = create_ml_prediction_record(experimental, simulation, linked_training_record=training_record["record_id"])
        prediction_record_path.write_text(json.dumps(prediction_record, indent=2) + "\n")

    return {
        "status": "success",
        "mode": mode,
        "task": task_name,
        "record_count": len(records),
        "spectrum_count": len(spectrum_rows),
        "condition_count": len(condition_rows),
        "files": {
            "spectrum_table": str(spectrum_path),
            "condition_table": str(condition_path),
            "ml_training_record": str(training_record_path),
            "ml_prediction_record": str(prediction_record_path) if prediction_record else None,
        },
        "ml_training_record": training_record,
        "ml_prediction_record": prediction_record,
        "warnings": [
            "Model training backends are intentionally not run by dataset assembly; consume these tables with the CO2RR_ISAAC_XAS_ML skill plan."
        ],
    }
