"""Small ML utilities for the ISAAC XAS Streamlit page."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tools.xas_record_utils import integrate, is_number, minmax, target_value

try:
    from sklearn.decomposition import PCA
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.model_selection import KFold, StratifiedKFold, cross_validate, learning_curve, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on runtime env
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc


PEAK_DESCRIPTOR_FEATURE = "peak_descriptors"
LEGACY_PEAK_DESCRIPTOR_FEATURE = "peak_window"


def _uses_peak_descriptors(kinds: Sequence[str]) -> bool:
    """Accept the renamed feature and the legacy token used by older settings."""
    return PEAK_DESCRIPTOR_FEATURE in kinds or LEGACY_PEAK_DESCRIPTOR_FEATURE in kinds


def feature_matrix(rows: Sequence[Dict[str, Any]], target: str, kinds: Sequence[str], norm: str, n_grid: int):
    valid = [
        row for row in rows
        if row.get("x") is not None
        and row.get("y") is not None
        and row.get("n_points", 0) >= 5
        and target_value(row, target) not in [None, ""]
    ]
    if len(valid) < 2:
        raise ValueError("Need at least two labeled spectra with valid x/y arrays.")
    xmin = max(float(np.min(row["x"])) for row in valid)
    xmax = min(float(np.max(row["x"])) for row in valid)
    if xmin >= xmax:
        xmin = min(float(np.min(row["x"])) for row in valid)
        xmax = max(float(np.max(row["x"])) for row in valid)
    grid = np.linspace(xmin, xmax, int(n_grid))
    X, y, names = [], [], []
    for row in valid:
        order = np.argsort(row["x"])
        x = np.asarray(row["x"])[order]
        yi = np.interp(grid, x, np.asarray(row["y"])[order])
        if norm == "minmax":
            yi = minmax(yi)
        elif norm == "area":
            yi = yi / (abs(integrate(np.abs(yi), grid)) or 1.0)
        elif norm == "zscore":
            yi = (yi - float(np.mean(yi))) / (float(np.std(yi)) or 1.0)
        feats, local_names = [], []
        if "raw" in kinds:
            feats.extend(yi.tolist())
            local_names.extend([f"raw_{i:03d}" for i in range(len(grid))])
        if "derivative" in kinds:
            dy = np.gradient(yi, grid)
            feats.extend(dy.tolist())
            local_names.extend([f"d1_{i:03d}" for i in range(len(grid))])
        if "cdf" in kinds:
            cdf = np.cumsum(np.maximum(yi, 0))
            cdf = cdf / (float(cdf[-1]) or 1.0)
            feats.extend(cdf.tolist())
            local_names.extend([f"cdf_{i:03d}" for i in range(len(grid))])
        if _uses_peak_descriptors(kinds):
            dy = np.gradient(yi, grid)
            pk = int(np.argmax(yi))
            vals = [float(grid[pk]), float(yi[pk]), integrate(yi, grid), float(np.max(dy)), float(np.min(dy))]
            vals.extend([integrate(yi[idx], grid[idx]) if len(idx) > 1 else 0.0 for idx in np.array_split(np.arange(len(grid)), 4)])
            feats.extend(vals)
            local_names.extend(["peak_x", "peak_y", "area", "derivative_max", "derivative_min", "window_1_area", "window_2_area", "window_3_area", "window_4_area"])
        if not names:
            names = local_names
        X.append(feats)
        y.append(target_value(row, target))
    return np.asarray(X, dtype=float), np.asarray(y, dtype=object), names, grid


def model_pipeline(task: str, name: str, use_pca: bool, n_comp: int):
    if not SKLEARN_AVAILABLE:
        raise RuntimeError(f"scikit-learn is not available: {SKLEARN_IMPORT_ERROR}")
    if task == "classification":
        models = {
            "Dummy baseline": DummyClassifier(strategy="most_frequent"),
            "Logistic regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
            "Random forest": RandomForestClassifier(n_estimators=300, random_state=7, class_weight="balanced"),
            "SVM": SVC(kernel="rbf", C=10.0, gamma="scale", class_weight="balanced"),
        }
    else:
        models = {
            "Dummy baseline": DummyRegressor(strategy="mean"),
            "Ridge": Ridge(alpha=1.0),
            "Random forest": RandomForestRegressor(n_estimators=300, random_state=7),
            "SVR": SVR(kernel="rbf", C=10.0, gamma="scale"),
        }
    steps = [("scale", StandardScaler())]
    if use_pca:
        steps.append(("pca", PCA(n_components=max(1, int(n_comp)), random_state=7)))
    steps.append(("model", models[name]))
    return Pipeline(steps)


def cv_splitter(task: str, y: np.ndarray):
    if len(y) < 4:
        return None
    if task == "classification":
        labels, counts = np.unique(y.astype(str), return_counts=True)
        if len(labels) >= 2 and int(np.min(counts)) >= 2:
            return StratifiedKFold(n_splits=min(5, int(np.min(counts)), len(y)), shuffle=True, random_state=7)
    return KFold(n_splits=min(5, len(y)), shuffle=True, random_state=7)


def train_manual(X, y, task, model_name, use_pca, n_comp):
    if task == "classification":
        ym = y.astype(str)
        labels, counts = np.unique(ym, return_counts=True)
        if len(labels) < 2:
            model_name = "Dummy baseline"
        model = model_pipeline(task, model_name, use_pca, n_comp)
        if len(ym) < 4:
            model.fit(X, ym)
            pred = model.predict(X)
            return {
                "metrics": {"train_balanced_accuracy": float(balanced_accuracy_score(ym, pred)), "note": "train-only; fewer than four spectra"},
                "table": [{"true": str(t), "predicted": str(p)} for t, p in zip(ym, pred)],
                "model": model,
                "y_model": ym,
            }
        strat = ym if len(labels) >= 2 and np.min(counts) >= 2 else None
        Xtr, Xte, ytr, yte = train_test_split(X, ym, test_size=0.25, random_state=7, stratify=strat)
        model.fit(Xtr, ytr)
        ptr, pte = model.predict(Xtr), model.predict(Xte)
        return {
            "metrics": {
                "train_balanced_accuracy": float(balanced_accuracy_score(ytr, ptr)),
                "test_balanced_accuracy": float(balanced_accuracy_score(yte, pte)),
                "test_macro_f1": float(f1_score(yte, pte, average="macro")),
            },
            "table": [{"true": str(t), "predicted": str(p)} for t, p in zip(yte, pte)],
            "model": model,
            "y_model": ym,
        }
    if not all(is_number(v) for v in y):
        raise ValueError("Regression target must be numeric.")
    ym = np.asarray([float(v) for v in y], dtype=float)
    model = model_pipeline(task, model_name, use_pca, n_comp)
    if len(ym) < 4:
        model.fit(X, ym)
        pred = model.predict(X)
        return {
            "metrics": {
                "train_RMSE": float(np.sqrt(mean_squared_error(ym, pred))),
                "train_MAE": float(mean_absolute_error(ym, pred)),
                "note": "train-only; fewer than four spectra",
            },
            "table": [{"true": float(t), "predicted": float(p)} for t, p in zip(ym, pred)],
            "model": model,
            "y_model": ym,
        }
    Xtr, Xte, ytr, yte = train_test_split(X, ym, test_size=0.25, random_state=7)
    model.fit(Xtr, ytr)
    ptr, pte = model.predict(Xtr), model.predict(Xte)
    return {
        "metrics": {
            "train_RMSE": float(np.sqrt(mean_squared_error(ytr, ptr))),
            "test_RMSE": float(np.sqrt(mean_squared_error(yte, pte))),
            "test_MAE": float(mean_absolute_error(yte, pte)),
            "test_R2": float(r2_score(yte, pte)) if len(yte) > 1 else float("nan"),
        },
        "table": [{"true": float(t), "predicted": float(p)} for t, p in zip(yte, pte)],
        "model": model,
        "y_model": ym,
    }


def learning_curve_rows(model, X: np.ndarray, y_model: np.ndarray, task: str) -> Tuple[str, List[Dict[str, float]], str]:
    cv = cv_splitter(task, y_model)
    if cv is None or X.shape[0] < 4:
        return "", [], "Training curve needs at least four spectra. Showing train-only result above."
    scoring = "balanced_accuracy" if task == "classification" else "neg_root_mean_squared_error"
    sizes, train_scores, val_scores = learning_curve(model, X, y_model, cv=cv, scoring=scoring, train_sizes=np.linspace(0.4, 1.0, 5), error_score=np.nan)
    train_mean = np.nanmean(train_scores, axis=1)
    val_mean = np.nanmean(val_scores, axis=1)
    if task == "regression":
        train_mean = -train_mean
        val_mean = -val_mean
        ylabel = "RMSE (lower is better)"
    else:
        ylabel = "Balanced accuracy (higher is better)"
    rows = [
        {"training_size": int(n), "train": float(t), "validation": float(v)}
        for n, t, v in zip(sizes, train_mean, val_mean)
        if math.isfinite(float(t)) and math.isfinite(float(v))
    ]
    return ylabel, rows, ""


def auto_advisor(rows, target, task, n_grid, max_rows):
    feature_sets = [[PEAK_DESCRIPTOR_FEATURE], ["raw", PEAK_DESCRIPTOR_FEATURE], ["derivative", PEAK_DESCRIPTOR_FEATURE], ["cdf", PEAK_DESCRIPTOR_FEATURE], ["raw"], ["raw", "derivative"], ["raw", "cdf"]]
    models = ["Dummy baseline", "Logistic regression", "SVM", "Random forest"] if task == "classification" else ["Dummy baseline", "Ridge", "SVR", "Random forest"]
    results, failures = [], []
    for fs in feature_sets:
        for norm in ["minmax", "area", "zscore"]:
            try:
                X, y, _, _ = feature_matrix(rows, target, fs, norm, n_grid)
            except Exception as exc:
                failures.append(f"{'+'.join(fs)} / {norm}: {exc}")
                continue
            for model_name in models:
                for n_comp in [None] + [n for n in [2, 4, 8, 16] if n < min(X.shape)]:
                    try:
                        if task == "classification":
                            ym = y.astype(str)
                            if len(np.unique(ym)) < 2 and model_name != "Dummy baseline":
                                continue
                            scoring = "balanced_accuracy"
                        else:
                            if not all(is_number(v) for v in y):
                                raise ValueError("Regression target must be numeric.")
                            ym = np.asarray([float(v) for v in y], dtype=float)
                            scoring = "neg_root_mean_squared_error"
                        cv = cv_splitter(task, ym)
                        model = model_pipeline(task, model_name, n_comp is not None, n_comp or 0)
                        if cv is None:
                            model.fit(X, ym)
                            pred = model.predict(X)
                            if task == "classification":
                                valid = float(balanced_accuracy_score(ym, pred)); train = valid; rank = valid; metric = "validation_balanced_accuracy"
                            else:
                                valid = float(np.sqrt(mean_squared_error(ym, pred))); train = valid; rank = -valid; metric = "validation_RMSE"
                            evaluation = "train-only (<4 samples)"
                        else:
                            scores = cross_validate(model, X, ym, cv=cv, scoring=scoring, return_train_score=True, error_score=np.nan)
                            tr = float(np.nanmean(scores["train_score"])); va = float(np.nanmean(scores["test_score"]))
                            if not math.isfinite(tr) or not math.isfinite(va):
                                raise ValueError("non-finite CV score")
                            if task == "classification":
                                valid = va; train = tr; rank = va; metric = "validation_balanced_accuracy"
                            else:
                                valid = -va; train = -tr; rank = va; metric = "validation_RMSE"
                            evaluation = "cross-validation"
                        gap = train - valid if task == "classification" else valid - train
                        results.append({
                            "feature_set": "+".join(fs),
                            "normalization": norm,
                            "dimension_reduction": "none" if n_comp is None else f"PCA({n_comp})",
                            "model": model_name,
                            metric: valid,
                            "train_score_same_metric": train,
                            "overfit_gap": gap,
                            "rank_score": rank,
                            "evaluation": evaluation,
                            "n_samples": int(X.shape[0]),
                            "n_features": int(X.shape[1]),
                        })
                    except Exception as exc:
                        failures.append(f"{'+'.join(fs)} / {norm} / {model_name} / {n_comp}: {exc}")
    results = sorted(results, key=lambda r: (r["rank_score"], -abs(r.get("overfit_gap", 0.0))), reverse=True)
    for i, row in enumerate(results, 1):
        row["rank"] = i
    return results[:max_rows], failures
