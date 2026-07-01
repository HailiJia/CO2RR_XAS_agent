# web_app/pages/2_Machine_Learning.py

from __future__ import annotations

import csv
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import streamlit as st

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional runtime dependency
    pd = None

try:
    from sklearn.base import clone
    from sklearn.decomposition import PCA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import learning_curve, train_test_split
    from sklearn.neural_network import MLPClassifier, MLPRegressor
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.svm import SVC, SVR

    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - optional runtime dependency
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

ML_PAGE_UPDATE_TAG = "v34_2026-07-01_isaac_xas_ml_page"

ENERGY_AXIS_NAMES = ("energy", "incident_energy", "photon_energy", "mono_energy", "e0")
ENERGY_UNITS = ("ev", "electronvolt", "electron_volt", "kev")
ABSORPTION_CHANNEL_NAMES = (
    "absorption",
    "absorbance",
    "mu",
    "mu_normalized",
    "xanes",
    "xas",
    "tey",
    "tfy",
    "pfy",
    "herfd",
    "intensity",
    "fluorescence",
    "delta_absorption",
)
NON_XAS_X_NAMES = ("q", "time", "wavelength", "two_theta", "angle")
NON_SPECTRAL_Y_NAMES = ("oxidation_state", "coordination_number", "charge", "descriptor")


st.set_page_config(page_title="CO2RR XAS Agent | ML", layout="wide")
st.title("Machine learning from ISAAC spectra")
st.caption(f"Update tag: `{ML_PAGE_UPDATE_TAG}`")


# -----------------------------
# Generic ISAAC parsing helpers
# -----------------------------


def _safe_str(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _lower(value: Any) -> str:
    return _safe_str(value).strip().lower()


def _is_number(value: Any) -> bool:
    try:
        value = float(value)
    except Exception:
        return False
    return math.isfinite(value)


def _as_float_array(values: Any) -> Optional[np.ndarray]:
    if not isinstance(values, list) or len(values) == 0:
        return None
    try:
        arr = np.asarray(values, dtype=float)
    except Exception:
        return None
    if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
        return None
    return arr


def _get_path(data: Dict[str, Any], dotted_path: str) -> Any:
    node: Any = data
    for part in dotted_path.split("."):
        if isinstance(node, dict) and part in node:
            node = node[part]
        else:
            return None
    return node


def _flatten_scalar_paths(data: Any, prefix: str = "", max_depth: int = 5) -> Dict[str, Any]:
    """Return scalar leaf values that are reasonable ML label/metadata candidates."""
    if max_depth < 0:
        return {}
    out: Dict[str, Any] = {}
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("_"):
                continue
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_scalar_paths(value, child_prefix, max_depth=max_depth - 1))
    elif isinstance(data, list):
        # Avoid flattening long spectra. Only expose short lists of scalars.
        if len(data) <= 8 and all(not isinstance(v, (dict, list)) for v in data):
            out[prefix] = ", ".join(_safe_str(v) for v in data)
    elif isinstance(data, (str, int, float, bool)) or data is None:
        out[prefix] = data
    return out


def _descriptor_values(record: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {}
    descriptor_outputs = ((record.get("descriptors") or {}).get("outputs") or [])
    for output in descriptor_outputs:
        label = output.get("label") or "descriptor_output"
        for descriptor in output.get("descriptors") or []:
            name = descriptor.get("name")
            if not name:
                continue
            values[f"descriptors.{name}"] = descriptor.get("value")
            values[f"descriptors.{label}.{name}"] = descriptor.get("value")
    return values


def _material_formula(record: Dict[str, Any]) -> str:
    return _safe_str(_get_path(record, "sample.material.formula")) or _safe_str(_get_path(record, "sample.material.name"))


def _record_display_name(record: Dict[str, Any], source_name: str, series_id: str = "") -> str:
    formula = _material_formula(record)
    technique = _safe_str(_get_path(record, "system.technique")) or _safe_str(_get_path(record, "system.configuration.technique"))
    record_id = _safe_str(record.get("record_id")) or source_name
    parts = [part for part in [record_id, formula, technique, series_id] if part]
    return " | ".join(parts)


def _read_uploaded_json_records(uploaded_files: Sequence[Any]) -> List[Tuple[str, Dict[str, Any]]]:
    """Read JSON records from uploaded .json files or .zip archives."""
    records: List[Tuple[str, Dict[str, Any]]] = []
    for uploaded in uploaded_files or []:
        name = getattr(uploaded, "name", "uploaded")
        payload = uploaded.getvalue()
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                for member in sorted(zf.namelist()):
                    if member.endswith("/") or not member.lower().endswith(".json"):
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


def _choose_primary_channel(channels: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not channels:
        return None
    for channel in channels:
        if _lower(channel.get("role")) == "primary_signal":
            return channel
    for channel in channels:
        name = _lower(channel.get("name"))
        if any(token in name for token in ABSORPTION_CHANNEL_NAMES):
            return channel
    return channels[0]


def _classify_series(record: Dict[str, Any], x_name: str, x_unit: str, y_name: str, y_unit: str) -> Tuple[bool, str, str]:
    """Classify from the actual data axes, not only metadata technique."""
    x_name_l = _lower(x_name)
    x_unit_l = _lower(x_unit)
    y_name_l = _lower(y_name)
    technique = _lower(_get_path(record, "system.technique")) or _lower(_get_path(record, "system.configuration.technique"))

    if x_name_l in {"q", "qz"} or "angstrom^-1" in x_unit_l or "1/angstrom" in x_unit_l:
        return False, "not_xas_reflectivity_or_scattering", "x-axis is q/scattering vector, not incident energy"
    if "time" in x_name_l or x_unit_l in {"s", "sec", "second", "min", "minute", "h", "hour"}:
        return False, "not_xas_time_series", "x-axis is time, not incident energy"
    if any(token in y_name_l for token in NON_SPECTRAL_Y_NAMES):
        return False, "not_xas_descriptor_series", "y-axis is a descriptor/time trace rather than spectral intensity"

    has_energy_x = any(token in x_name_l for token in ENERGY_AXIS_NAMES) or x_name_l in {"e", "energy_eV".lower()}
    has_energy_unit = any(unit == x_unit_l or unit in x_unit_l for unit in ENERGY_UNITS)
    has_xas_y = any(token in y_name_l for token in ABSORPTION_CHANNEL_NAMES)

    if has_energy_x and has_energy_unit and has_xas_y:
        if "delta" in y_name_l or "difference" in y_name_l:
            return True, "xas_derived_difference_spectrum", "energy-axis differential XAS spectrum"
        if "herfd" in technique:
            return True, "herfd_xas_like", "energy-axis HERFD-XAS/HERFD-XANES-like spectrum"
        if "xanes" in technique:
            return True, "xanes_like", "energy-axis XANES-like spectrum"
        return True, "xas_like", "energy-axis XAS/XANES-like spectrum"

    if has_energy_x and has_energy_unit:
        return True, "xas_like_uncertain_y", "energy-axis spectrum, but y channel name is not a standard XAS signal"

    return False, "not_xas_unknown_axis", "does not have an incident-energy x-axis with eV/keV units"


def _build_series_rows(records: Sequence[Tuple[str, Dict[str, Any]]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for source_name, record in records:
        measurement = record.get("measurement") or {}
        series_list = measurement.get("series") or []
        if not series_list:
            rows.append({
                "source_name": source_name,
                "record_id": record.get("record_id"),
                "series_id": "",
                "has_series": False,
                "is_xas": False,
                "classification": "not_plottable_no_measurement_series",
                "reason": "no measurement.series found",
                "x_name": "",
                "x_unit": "",
                "y_name": "",
                "y_unit": "",
                "n_points": 0,
                "technique": _get_path(record, "system.technique") or _get_path(record, "system.configuration.technique"),
                "formula": _material_formula(record),
                "record": record,
                "metadata": {**_flatten_scalar_paths(record), **_descriptor_values(record)},
                "x": None,
                "y": None,
            })
            continue

        for idx, series in enumerate(series_list):
            independent = series.get("independent_variables") or []
            channels = series.get("channels") or []
            x_obj = independent[0] if independent else {}
            y_obj = _choose_primary_channel(channels) or {}
            x = _as_float_array(x_obj.get("values"))
            y = _as_float_array(y_obj.get("values"))
            x_name = _safe_str(x_obj.get("name"))
            x_unit = _safe_str(x_obj.get("unit"))
            y_name = _safe_str(y_obj.get("name"))
            y_unit = _safe_str(y_obj.get("unit"))
            series_id = _safe_str(series.get("series_id")) or f"series_{idx}"
            metadata = {**_flatten_scalar_paths(record), **_descriptor_values(record)}
            metadata["series.series_id"] = series_id
            metadata["series.x_name"] = x_name
            metadata["series.x_unit"] = x_unit
            metadata["series.y_name"] = y_name
            metadata["series.y_unit"] = y_unit
            metadata["derived.n_points"] = int(x.size if x is not None else 0)

            if x is None or y is None or x.size != y.size:
                is_xas, classification, reason = False, "not_plottable_invalid_xy", "missing/non-numeric x or y values, or length mismatch"
            else:
                is_xas, classification, reason = _classify_series(record, x_name, x_unit, y_name, y_unit)

            rows.append({
                "source_name": source_name,
                "record_id": record.get("record_id"),
                "series_id": series_id,
                "has_series": True,
                "is_xas": is_xas,
                "classification": classification,
                "reason": reason,
                "x_name": x_name,
                "x_unit": x_unit,
                "y_name": y_name,
                "y_unit": y_unit,
                "n_points": int(x.size if x is not None else 0),
                "technique": _get_path(record, "system.technique") or _get_path(record, "system.configuration.technique"),
                "formula": _material_formula(record),
                "record": record,
                "metadata": metadata,
                "x": x,
                "y": y,
            })
    return rows


def _summary_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "source_name": row["source_name"],
            "record_id": row.get("record_id"),
            "series_id": row.get("series_id"),
            "technique": row.get("technique"),
            "formula": row.get("formula"),
            "x": row.get("x_name"),
            "x_unit": row.get("x_unit"),
            "y": row.get("y_name"),
            "y_unit": row.get("y_unit"),
            "n_points": row.get("n_points"),
            "is_xas": row.get("is_xas"),
            "classification": row.get("classification"),
            "reason": row.get("reason"),
        }
        for row in rows
    ]


def _rows_to_csv(rows: Sequence[Dict[str, Any]]) -> str:
    if not rows:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()))
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


# -----------------------------
# Feature construction and ML
# -----------------------------


def _minmax(y: np.ndarray) -> np.ndarray:
    span = float(np.max(y) - np.min(y))
    if span == 0.0:
        return np.zeros_like(y, dtype=float)
    return (y - float(np.min(y))) / span


def _area_norm(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    area = float(np.trapz(np.abs(y), x))
    if area == 0.0:
        return y.astype(float)
    return y / area


def _common_grid(rows: Sequence[Dict[str, Any]], n_grid: int) -> np.ndarray:
    xmin = max(float(np.min(row["x"])) for row in rows if row.get("x") is not None)
    xmax = min(float(np.max(row["x"])) for row in rows if row.get("x") is not None)
    if not math.isfinite(xmin) or not math.isfinite(xmax) or xmin >= xmax:
        xmin = min(float(np.min(row["x"])) for row in rows if row.get("x") is not None)
        xmax = max(float(np.max(row["x"])) for row in rows if row.get("x") is not None)
    return np.linspace(xmin, xmax, int(n_grid))


def _numeric_descriptor_features(row: Dict[str, Any]) -> Tuple[List[str], List[float]]:
    names: List[str] = []
    values: List[float] = []
    for key, value in sorted((row.get("metadata") or {}).items()):
        if key.startswith("descriptors.") and _is_number(value):
            names.append(key)
            values.append(float(value))
    return names, values


def _target_value(row: Dict[str, Any], path: str) -> Any:
    if path.startswith("descriptors.") or path.startswith("series.") or path.startswith("derived."):
        return (row.get("metadata") or {}).get(path)
    value = _get_path(row.get("record") or {}, path)
    if value is not None:
        return value
    return (row.get("metadata") or {}).get(path)


def _candidate_label_fields(rows: Sequence[Dict[str, Any]]) -> List[str]:
    keys = set()
    preferred = [
        "sample.material.formula",
        "sample.material.name",
        "sample.ml_labels.adsorbate_identity",
        "sample.ml_labels.adsorption_motif",
        "sample.ml_labels.oxidation_coordination_class",
        "sample.catalyst.composition",
        "sample.catalyst.surface_facet",
        "system.technique",
        "source_type",
        "record_domain",
    ]
    for row in rows:
        for key, value in (row.get("metadata") or {}).items():
            if value is None:
                continue
            if isinstance(value, str) and len(value) > 160:
                continue
            keys.add(key)
    ordered = [key for key in preferred if key in keys]
    ordered.extend(sorted(keys - set(ordered)))
    return ordered


def _build_feature_matrix(
    rows: Sequence[Dict[str, Any]],
    target_path: str,
    feature_kinds: Sequence[str],
    normalization: str,
    n_grid: int,
) -> Tuple[np.ndarray, np.ndarray, List[str], List[Dict[str, Any]], np.ndarray]:
    valid = [row for row in rows if row.get("x") is not None and row.get("y") is not None and row.get("n_points", 0) >= 5]
    valid = [row for row in valid if _target_value(row, target_path) not in [None, ""]]
    if len(valid) < 2:
        raise ValueError("Need at least two labeled spectra with valid x/y arrays.")

    grid = _common_grid(valid, n_grid)
    feature_names: List[str] = []
    X_rows: List[List[float]] = []
    y_values: List[Any] = []
    used_rows: List[Dict[str, Any]] = []

    for row in valid:
        x = np.asarray(row["x"], dtype=float)
        y = np.asarray(row["y"], dtype=float)
        order = np.argsort(x)
        x = x[order]
        y = y[order]
        yi = np.interp(grid, x, y)
        if normalization == "minmax":
            yi = _minmax(yi)
        elif normalization == "area":
            yi = _area_norm(grid, yi)
        elif normalization == "zscore":
            std = float(np.std(yi)) or 1.0
            yi = (yi - float(np.mean(yi))) / std

        row_features: List[float] = []
        local_feature_names: List[str] = []
        if "raw" in feature_kinds:
            row_features.extend(yi.tolist())
            local_feature_names.extend([f"raw_{i:03d}" for i in range(len(grid))])
        if "derivative" in feature_kinds:
            dy = np.gradient(yi, grid)
            row_features.extend(dy.tolist())
            local_feature_names.extend([f"d1_{i:03d}" for i in range(len(grid))])
        if "cdf" in feature_kinds:
            cdf = np.cumsum(np.maximum(yi, 0.0))
            cdf = cdf / (float(cdf[-1]) or 1.0)
            row_features.extend(cdf.tolist())
            local_feature_names.extend([f"cdf_{i:03d}" for i in range(len(grid))])
        if "peak_window" in feature_kinds:
            peak_idx = int(np.argmax(yi))
            peak_features = [
                float(grid[peak_idx]),
                float(yi[peak_idx]),
                float(np.trapz(yi, grid)),
                float(np.max(np.gradient(yi, grid))),
                float(np.min(np.gradient(yi, grid))),
            ]
            bins = np.array_split(np.arange(len(grid)), 4)
            for idx in bins:
                peak_features.append(float(np.trapz(yi[idx], grid[idx])) if len(idx) > 1 else 0.0)
            peak_names = ["peak_x", "peak_y", "area", "derivative_max", "derivative_min"] + [f"window_{i}_area" for i in range(1, 5)]
            row_features.extend(peak_features)
            local_feature_names.extend(peak_names)
        if "descriptor_scalars" in feature_kinds:
            desc_names, desc_values = _numeric_descriptor_features(row)
            row_features.extend(desc_values)
            local_feature_names.extend(desc_names)

        if not feature_names:
            feature_names = local_feature_names
        elif len(local_feature_names) != len(feature_names):
            # Most often caused by descriptor fields being present only in some records.
            # For the first pass, descriptor-scalar features are only safe when all rows share them.
            raise ValueError("Selected feature set produced inconsistent feature lengths. Try disabling descriptor scalar features.")

        X_rows.append(row_features)
        y_values.append(_target_value(row, target_path))
        used_rows.append(row)

    X = np.asarray(X_rows, dtype=float)
    y_arr = np.asarray(y_values, dtype=object)
    return X, y_arr, feature_names, used_rows, grid


def _make_model(task: str, model_name: str, use_pca: bool, n_components: int):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn is not available: {SKLEARN_IMPORT_ERROR}")

    if task == "classification":
        if model_name == "Logistic regression":
            model = LogisticRegression(max_iter=2000, class_weight="balanced")
        elif model_name == "Random forest":
            model = RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced")
        elif model_name == "SVM":
            model = SVC(kernel="rbf", C=10.0, gamma="scale", probability=True, class_weight="balanced")
        elif model_name == "MLP":
            model = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=7)
        else:
            raise ValueError(f"Unknown classification model: {model_name}")
    else:
        if model_name == "Ridge":
            model = Ridge(alpha=1.0)
        elif model_name == "Random forest":
            model = RandomForestRegressor(n_estimators=300, random_state=7)
        elif model_name == "SVR":
            model = SVR(kernel="rbf", C=10.0, gamma="scale")
        elif model_name == "MLP":
            model = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=7)
        else:
            raise ValueError(f"Unknown regression model: {model_name}")

    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=max(1, int(n_components)), random_state=7)))
    steps.append(("model", model))
    return Pipeline(steps)


def _train_and_score(X: np.ndarray, y: np.ndarray, task: str, model_name: str, use_pca: bool, n_components: int) -> Dict[str, Any]:
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn is not available: {SKLEARN_IMPORT_ERROR}")
    if len(y) < 4:
        raise ValueError("Need at least four labeled spectra for a train/test split. Upload more records or use this page for plotting only.")

    model = _make_model(task, model_name, use_pca, n_components)
    result: Dict[str, Any] = {}

    if task == "classification":
        labels, counts = np.unique(y.astype(str), return_counts=True)
        if len(labels) < 2:
            raise ValueError("Classification needs at least two label classes.")
        stratify = y.astype(str) if np.min(counts) >= 2 else None
        X_train, X_test, y_train, y_test = train_test_split(X, y.astype(str), test_size=0.25, random_state=7, stratify=stratify)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        result["metrics"] = {
            "train_accuracy": float(accuracy_score(y_train, pred_train)),
            "test_accuracy": float(accuracy_score(y_test, pred_test)),
            "train_balanced_accuracy": float(balanced_accuracy_score(y_train, pred_train)),
            "test_balanced_accuracy": float(balanced_accuracy_score(y_test, pred_test)),
            "test_macro_f1": float(f1_score(y_test, pred_test, average="macro")),
        }
        result["prediction_table"] = [
            {"true": str(t), "predicted": str(p)} for t, p in zip(y_test, pred_test)
        ]
        result["confusion_matrix"] = {
            "labels": labels.astype(str).tolist(),
            "matrix": confusion_matrix(y_test, pred_test, labels=labels.astype(str)).tolist(),
        }
        scoring = "balanced_accuracy"
    else:
        numeric_y = np.asarray([float(v) for v in y], dtype=float)
        X_train, X_test, y_train, y_test = train_test_split(X, numeric_y, test_size=0.25, random_state=7)
        model.fit(X_train, y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        result["metrics"] = {
            "train_RMSE": float(mean_squared_error(y_train, pred_train, squared=False)),
            "test_RMSE": float(mean_squared_error(y_test, pred_test, squared=False)),
            "test_MAE": float(mean_absolute_error(y_test, pred_test)),
            "test_R2": float(r2_score(y_test, pred_test)) if len(y_test) > 1 else float("nan"),
        }
        result["prediction_table"] = [
            {"true": float(t), "predicted": float(p)} for t, p in zip(y_test, pred_test)
        ]
        scoring = "neg_root_mean_squared_error"

    # Learning curve: use a small CV count that respects small datasets.
    curve_model = clone(model)
    cv = min(5, max(2, len(y) // 3))
    try:
        train_sizes, train_scores, valid_scores = learning_curve(
            curve_model,
            X,
            y.astype(str) if task == "classification" else np.asarray([float(v) for v in y], dtype=float),
            train_sizes=np.linspace(0.4, 1.0, 4),
            cv=cv,
            scoring=scoring,
            error_score=np.nan,
        )
        if task == "regression":
            train_mean = -np.nanmean(train_scores, axis=1)
            valid_mean = -np.nanmean(valid_scores, axis=1)
            y_label = "RMSE"
        else:
            train_mean = np.nanmean(train_scores, axis=1)
            valid_mean = np.nanmean(valid_scores, axis=1)
            y_label = "Balanced accuracy"
        result["learning_curve"] = [
            {"train_size": int(size), "train": float(tr), "validation": float(val)}
            for size, tr, val in zip(train_sizes, train_mean, valid_mean)
        ]
        result["learning_curve_y_label"] = y_label
    except Exception as exc:
        result["learning_curve_error"] = str(exc)

    return result


# -----------------------------
# Streamlit UI
# -----------------------------


st.markdown(
    """
This page is the first-pass ML front end for ISAAC spectra. It parses uploaded ISAAC JSON records, checks whether each record is actually XAS-like from the x/y axes, plots spectra with units, and can train a simple scikit-learn model once you choose an input representation and label.
""".strip()
)

uploaded_files = st.file_uploader(
    "Upload ISAAC JSON files or a ZIP folder of ISAAC records",
    type=["json", "zip"],
    accept_multiple_files=True,
)

if not uploaded_files:
    st.info("Upload one or more ISAAC JSON records, or a ZIP containing JSON records.")
    st.stop()

try:
    records = _read_uploaded_json_records(uploaded_files)
    rows = _build_series_rows(records)
except Exception as exc:
    st.exception(exc)
    st.stop()

summary_rows = _summary_table(rows)
plottable_rows = [row for row in rows if row.get("x") is not None and row.get("y") is not None and row.get("n_points", 0) > 0]
xas_rows = [row for row in plottable_rows if row.get("is_xas")]
non_xas_rows = [row for row in rows if not row.get("is_xas")]
unit_missing_rows = [
    row for row in plottable_rows
    if not _safe_str(row.get("x_unit")).strip() or not _safe_str(row.get("y_unit")).strip()
]

m1, m2, m3, m4 = st.columns(4)
m1.metric("JSON records", len(records))
m2.metric("Series", len(rows))
m3.metric("XAS-like", len(xas_rows))
m4.metric("Missing x/y unit", len(unit_missing_rows))

if unit_missing_rows:
    st.warning("Some plottable series are missing x or y units. See the table below.")
else:
    st.success("All plottable series have both x and y units.")

with st.expander("Series classification table", expanded=True):
    if pd is not None:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)
    else:
        st.dataframe(summary_rows, use_container_width=True)
    st.download_button(
        "Download classification CSV",
        data=_rows_to_csv(summary_rows),
        file_name="isaac_series_classification.csv",
        mime="text/csv",
    )

if non_xas_rows:
    with st.expander("Non-XAS or questionable records", expanded=True):
        st.write("These are excluded by default from ML training because their actual axes/channels are not incident-energy XAS spectra.")
        if pd is not None:
            st.dataframe(pd.DataFrame(_summary_table(non_xas_rows)), use_container_width=True)
        else:
            st.dataframe(_summary_table(non_xas_rows), use_container_width=True)

st.divider()
st.header("Plot spectra")

plot_scope = st.radio(
    "Records to plot",
    ["XAS-like only", "All plottable series", "Non-XAS only"],
    horizontal=True,
)
if plot_scope == "XAS-like only":
    candidate_plot_rows = xas_rows
elif plot_scope == "Non-XAS only":
    candidate_plot_rows = [row for row in non_xas_rows if row.get("x") is not None and row.get("y") is not None]
else:
    candidate_plot_rows = plottable_rows

labels = [_record_display_name(row["record"], row["source_name"], row["series_id"]) for row in candidate_plot_rows]
selected_labels = st.multiselect(
    "Select series",
    labels,
    default=labels[: min(len(labels), 12)],
)
selected_set = set(selected_labels)
selected_rows = [row for row, label in zip(candidate_plot_rows, labels) if label in selected_set]

if selected_rows:
    if pd is None:
        st.warning("pandas is not available, so plotting is disabled. Install pandas or use the classification table.")
    else:
        plot_records = []
        max_points = int(st.slider("Maximum points per series for display", 50, 2000, 800, step=50))
        for row in selected_rows:
            x = np.asarray(row["x"], dtype=float)
            y = np.asarray(row["y"], dtype=float)
            label = _record_display_name(row["record"], row["source_name"], row["series_id"])
            if len(x) > max_points:
                idx = np.linspace(0, len(x) - 1, max_points).astype(int)
                x = x[idx]
                y = y[idx]
            for xi, yi in zip(x, y):
                plot_records.append({"x": float(xi), "y": float(yi), "label": label})
        plot_df = pd.DataFrame(plot_records)
        x_label = selected_rows[0].get("x_name") or "x"
        x_unit = selected_rows[0].get("x_unit") or ""
        y_label = selected_rows[0].get("y_name") or "y"
        y_unit = selected_rows[0].get("y_unit") or ""
        st.caption(f"x: {x_label} ({x_unit})    y: {y_label} ({y_unit})")
        st.line_chart(plot_df, x="x", y="y", color="label", use_container_width=True)
else:
    st.info("No plottable series selected.")

st.divider()
st.header("Train a first-pass ML model")
st.caption("This is intended for quick checks. For publication-quality benchmarking, use grouped validation and a larger external test set.")

training_scope = st.radio(
    "Training records",
    ["XAS-like only", "All plottable series"],
    horizontal=True,
)
training_rows = xas_rows if training_scope == "XAS-like only" else plottable_rows

if len(training_rows) < 2:
    st.info("Need at least two plottable labeled spectra for ML setup.")
    st.stop()

label_candidates = _candidate_label_fields(training_rows)
if not label_candidates:
    st.warning("No scalar label fields found in these ISAAC records.")
    st.stop()

default_label = "sample.material.formula" if "sample.material.formula" in label_candidates else label_candidates[0]
target_path = st.selectbox("Target / label field", label_candidates, index=label_candidates.index(default_label))

feature_kinds = st.multiselect(
    "Input features",
    ["raw", "derivative", "cdf", "peak_window", "descriptor_scalars"],
    default=["raw", "peak_window"],
    help="Raw/derivative/CDF features are interpolated onto a common grid. Descriptor scalars come from descriptors.outputs when available.",
)
normalization = st.selectbox("Spectrum normalization", ["minmax", "area", "zscore", "none"], index=0)
n_grid = st.slider("Common-grid points", 32, 1024, 256, step=32)

label_values = [_target_value(row, target_path) for row in training_rows]
label_values = [value for value in label_values if value not in [None, ""]]
can_be_regression = label_values and all(_is_number(value) for value in label_values)

task = st.radio(
    "Task type",
    ["classification", "regression"],
    index=1 if can_be_regression and target_path.startswith("descriptors.") else 0,
    horizontal=True,
)

if task == "classification":
    model_name = st.selectbox("Model", ["Logistic regression", "Random forest", "SVM", "MLP"], index=0)
else:
    model_name = st.selectbox("Model", ["Ridge", "Random forest", "SVR", "MLP"], index=0)

use_pca = st.checkbox("Use PCA before model", value=False)
n_components = 2
if use_pca:
    n_components = st.slider("PCA components", 1, min(32, max(1, len(training_rows) - 1)), min(4, max(1, len(training_rows) - 1)))

if not SKLEARN_AVAILABLE:
    st.warning(f"scikit-learn is not available, so training is disabled. Import error: {SKLEARN_IMPORT_ERROR}")

if st.button("Train model", type="primary", disabled=not SKLEARN_AVAILABLE):
    if not feature_kinds:
        st.error("Select at least one feature type.")
    else:
        try:
            X, y, feature_names, used_rows, grid = _build_feature_matrix(
                training_rows,
                target_path=target_path,
                feature_kinds=feature_kinds,
                normalization=normalization,
                n_grid=n_grid,
            )
            if task == "regression" and not all(_is_number(v) for v in y):
                raise ValueError("Regression target must be numeric. Choose a numeric descriptor or switch to classification.")
            if use_pca and n_components >= min(X.shape[0], X.shape[1]):
                raise ValueError("PCA components must be smaller than min(n_samples, n_features).")

            result = _train_and_score(X, y, task, model_name, use_pca, n_components)
            st.success(f"Trained {model_name} on {X.shape[0]} spectra with {X.shape[1]} features.")

            st.subheader("Metrics")
            metric_cols = st.columns(max(1, len(result["metrics"])))
            for col, (key, value) in zip(metric_cols, result["metrics"].items()):
                col.metric(key, f"{value:.4g}" if isinstance(value, float) and math.isfinite(value) else str(value))

            st.subheader("Held-out predictions")
            if pd is not None:
                st.dataframe(pd.DataFrame(result["prediction_table"]), use_container_width=True)
            else:
                st.dataframe(result["prediction_table"], use_container_width=True)

            if task == "classification" and "confusion_matrix" in result:
                st.subheader("Confusion matrix")
                cm = result["confusion_matrix"]
                if pd is not None:
                    st.dataframe(pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"]), use_container_width=True)
                else:
                    st.write(cm)

            if result.get("learning_curve"):
                st.subheader("Learning / performance curve")
                if pd is not None:
                    lc_df = pd.DataFrame(result["learning_curve"])
                    st.caption(f"y-axis: {result.get('learning_curve_y_label', 'score')}")
                    st.line_chart(lc_df, x="train_size", y=["train", "validation"], use_container_width=True)
                else:
                    st.write(result["learning_curve"])
            elif result.get("learning_curve_error"):
                st.warning(f"Learning curve could not be computed: {result['learning_curve_error']}")

            with st.expander("Training details", expanded=False):
                st.write({
                    "target_path": target_path,
                    "task": task,
                    "model": model_name,
                    "feature_kinds": list(feature_kinds),
                    "normalization": normalization,
                    "n_grid": n_grid,
                    "n_samples": int(X.shape[0]),
                    "n_features": int(X.shape[1]),
                    "x_grid_min": float(grid[0]),
                    "x_grid_max": float(grid[-1]),
                    "first_20_feature_names": feature_names[:20],
                })
        except Exception as exc:
            st.exception(exc)
