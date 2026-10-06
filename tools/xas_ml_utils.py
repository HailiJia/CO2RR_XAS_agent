"""Small ML utilities for the ISAAC XAS Streamlit page."""

from __future__ import annotations

import math
import io
import hashlib
import json
import platform
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from tools.xas_record_utils import integrate, is_number, minmax, target_value
from tools.ml_splits import groups_for_rows, grouped_holdout

try:
    import joblib
    import sklearn
    from sklearn.decomposition import PCA
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score
    from sklearn.base import BaseEstimator, TransformerMixin, clone
    from sklearn.model_selection import GroupKFold, cross_validate
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC, SVR
    SKLEARN_AVAILABLE = True
    SKLEARN_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - depends on runtime env
    SKLEARN_AVAILABLE = False
    SKLEARN_IMPORT_ERROR = exc
    class BaseEstimator:
        pass
    class TransformerMixin:
        pass


PEAK_DESCRIPTOR_FEATURE = "peak_descriptors"
LEGACY_PEAK_DESCRIPTOR_FEATURE = "peak_window"


def _uses_peak_descriptors(kinds: Sequence[str]) -> bool:
    """Accept the renamed feature and the legacy token used by older settings."""
    return PEAK_DESCRIPTOR_FEATURE in kinds or LEGACY_PEAK_DESCRIPTOR_FEATURE in kinds


def feature_matrix(rows: Sequence[Dict[str, Any]], target: str, kinds: Sequence[str], norm: str, n_grid: int, grid=None):
    valid = [
        row for row in rows
        if row.get("x") is not None
        and row.get("y") is not None
        and row.get("n_points", 0) >= 5
        and (target is None or target_value(row, target) not in [None, ""])
    ]
    if not valid or (target is not None and len(valid) < 2):
        raise ValueError("Need valid spectra with x/y arrays and the selected labels.")
    if target is None and len(valid) != len(rows):
        raise ValueError("Every prediction row must have valid spectral arrays.")
    if norm not in {"minmax", "area", "zscore", "none"}:
        raise ValueError("Unknown spectrum normalization.")
    if int(n_grid) < 5:
        raise ValueError("The common energy grid needs at least five points.")
    if set(kinds) - {"raw", "derivative", "cdf", PEAK_DESCRIPTOR_FEATURE, LEGACY_PEAK_DESCRIPTOR_FEATURE}:
        raise ValueError("Unknown feature family.")
    if not kinds:
        raise ValueError("Select at least one feature family.")
    xmin = max(float(np.min(row["x"])) for row in valid)
    xmax = min(float(np.max(row["x"])) for row in valid)
    if xmin >= xmax:
        xmin = min(float(np.min(row["x"])) for row in valid)
        xmax = max(float(np.max(row["x"])) for row in valid)
    grid = np.linspace(xmin, xmax, int(n_grid)) if grid is None else np.asarray(grid, float)
    X, y, names = [], [], []
    for row in valid:
        order = np.argsort(row["x"])
        x = np.asarray(row["x"])[order]
        if len(x) != len(row["y"]) or not np.all(np.isfinite(x)) or not np.all(np.isfinite(row["y"])) or np.any(np.diff(x) <= 0):
            raise ValueError("Spectra need finite, equal-length x/y arrays and unique energy points.")
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
        y.append(target_value(row, target) if target is not None else None)
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



class SpectralFeatures(BaseEstimator, TransformerMixin):
    """A fitted energy grid and deterministic per-spectrum feature recipe."""
    def __init__(self, kinds=("raw",), normalization="minmax", n_grid=256):
        self.kinds = kinds
        self.normalization = normalization
        self.n_grid = n_grid

    def fit(self, X, y=None):
        if not X:
            raise ValueError("No training spectra.")
        xmin = max(float(np.min(row["x"])) for row in X)
        xmax = min(float(np.max(row["x"])) for row in X)
        if xmin >= xmax:
            raise ValueError("Training spectra have no shared energy interval.")
        self.grid_ = np.linspace(xmin, xmax, int(self.n_grid))
        _, _, self.feature_names_, _ = feature_matrix(X, None, self.kinds, self.normalization, self.n_grid, self.grid_)
        return self

    def transform(self, X):
        # Reject extrapolation rather than pad unknown spectral support silently.
        for row in X:
            if float(np.min(row["x"])) > self.grid_[0] or float(np.max(row["x"])) < self.grid_[-1]:
                raise ValueError("Spectrum does not cover the fitted energy grid.")
        return feature_matrix(X, None, self.kinds, self.normalization, self.n_grid, self.grid_)[0]


def spectral_pipeline(task, model_name, kinds, normalization, n_grid, use_pca=False, n_comp=2):
    model = model_pipeline(task, model_name, use_pca, n_comp)
    return Pipeline([("features", SpectralFeatures(tuple(kinds), normalization, n_grid))] + model.steps)


def cv_splitter(task, y, groups=None):
    if groups is None:
        raise ValueError("Cross-validation requires configuration groups.")
    count = len(np.unique(groups))
    return GroupKFold(n_splits=min(5, count)) if count >= 2 else None


def _target_array(y, task):
    if task not in {"classification", "regression"}:
        raise ValueError("Task must be classification or regression.")
    if task == "classification":
        return np.asarray(y, dtype=str)
    if not all(is_number(v) for v in y):
        raise ValueError("Regression target must be numeric.")
    return np.asarray(y, dtype=float)


def _metrics(true, pred, task, prefix):
    if task == "classification":
        return {prefix + "balanced_accuracy": float(balanced_accuracy_score(true, pred)),
                prefix + "macro_f1": float(f1_score(true, pred, average="macro", zero_division=0))}
    result = {prefix + "RMSE": float(np.sqrt(mean_squared_error(true, pred))),
              prefix + "MAE": float(mean_absolute_error(true, pred))}
    if len(true) > 1:
        result[prefix + "R2"] = float(r2_score(true, pred))
    return result


def _fit_holdout(X, y, groups, task, model, row_ids=None, split_manifest=None):
    groups = np.asarray(groups, str)
    if len(groups) != len(y) or len(X) != len(y):
        raise ValueError("Features, labels and group IDs must be aligned.")
    train, test, manifest = grouped_holdout(groups, row_ids, manifest=split_manifest)
    ym = _target_array(y, task)
    if task == "classification" and len(np.unique(ym[train])) < 2:
        if not isinstance(model.steps[-1][1], DummyClassifier):
            raise ValueError("Training groups contain fewer than two classes. Add independent labeled configurations or choose the dummy baseline.")
    subset = lambda ids: [X[i] for i in ids] if isinstance(X, list) else X[ids]
    model.fit(subset(train), ym[train])
    ptr, pte = model.predict(subset(train)), model.predict(subset(test))
    metrics = {**_metrics(ym[train], ptr, task, "train_"), **_metrics(ym[test], pte, task, "test_")}
    if task == "classification":
        metrics["unseen_test_classes"] = sorted(set(ym[test]) - set(ym[train]))
        metrics["test_class_counts"] = {label: int(np.sum(ym[test] == label)) for label in sorted(set(ym))}
    table = [{"row_id": manifest["test_row_ids"][k], "group": str(groups[i]),
              "true": ym[i].item(), "predicted": pte[k].item()} for k, i in enumerate(test)]
    return {"metrics": metrics, "table": table, "model": model, "y_model": ym,
            "split_manifest": manifest, "train_indices": train, "test_indices": test,
            "groups": groups, "X": X, "task": task}


def train_manual(X, y, task, model_name, use_pca, n_comp, groups=None, row_ids=None, split_manifest=None):
    """Array API: callers must supply true configuration groups."""
    if groups is None:
        raise ValueError("Supply configuration groups; random per-spectrum evaluation is disabled.")
    return _fit_holdout(np.asarray(X), y, groups, task, model_pipeline(task, model_name, use_pca, n_comp), row_ids, split_manifest)


def _valid_rows(rows, target):
    valid = [r for r in rows if r.get("x") is not None and r.get("y") is not None and r.get("n_points", 0) >= 5 and target_value(r, target) not in [None, ""]]
    if len(valid) < 2:
        raise ValueError("Need at least two labeled spectra.")
    edges = {(r.get("absorber"), r.get("edge")) for r in valid if r.get("absorber")}
    if len(edges) > 1:
        raise ValueError("Train each absorber/edge separately or define an explicit multi-edge representation.")
    return valid


def _row_ids(rows):
    # Distinguish multiple channels/files while retaining record provenance.
    return [f"{r.get('record_id') or r.get('source_name') or 'row'}:{r.get('series_id') or 'series'}:{i}" for i, r in enumerate(rows)]


def train_manual_rows(rows, target, task, model_name, kinds=("raw",), normalization="minmax", n_grid=256,
                      use_pca=False, n_comp=2, group_field=None, split_manifest=None):
    valid = _valid_rows(rows, target)
    groups, sources = groups_for_rows(valid, group_field)
    y = [target_value(r, target) for r in valid]
    result = _fit_holdout(valid, y, groups, task, spectral_pipeline(task, model_name, kinds, normalization, n_grid, use_pca, n_comp),
                          _row_ids(valid), split_manifest)
    result["split_manifest"]["group_sources"] = sources
    result["target"] = target
    snapshot = [{"row_id": row_id, "configuration_group": group, "target": str(label),
                 "energy": np.asarray(row["x"], float).tolist(), "signal": np.asarray(row["y"], float).tolist()}
                for row_id, group, label, row in zip(_row_ids(valid), groups, y, valid)]
    result["dataset_sha256"] = hashlib.sha256(json.dumps(snapshot, sort_keys=True).encode()).hexdigest()
    result["recipe"] = {"feature_set": list(kinds), "normalization": normalization, "n_grid": int(n_grid),
                        "model": model_name, "pca_components": int(n_comp) if use_pca else None}
    return result


def learning_curve_rows(model, X, y_model, task, groups=None, train_indices=None):
    """Whole-group learning curves using only outer training groups."""
    if groups is None or train_indices is None:
        return "", [], "Learning curves require configuration groups and the outer training indices."
    X = [X[i] for i in train_indices] if isinstance(X, list) else np.asarray(X)[train_indices]
    y = np.asarray(y_model)[train_indices]
    groups = np.asarray(groups)[train_indices]
    cv = cv_splitter(task, y, groups)
    if cv is None:
        return "", [], "Insufficient training groups for a learning curve."
    samples = []
    for fraction in (0.5, 0.75, 1.0):
        scores, sizes = [], []
        for tr, va in cv.split(np.zeros((len(y), 1)), y, groups):
            available = sorted(set(groups[tr]))
            chosen = available[:max(1, int(np.ceil(len(available) * fraction)))]
            subset = tr[np.isin(groups[tr], chosen)]
            if task == "classification" and len(set(y[subset])) < 2 and not isinstance(model.steps[-1][1], DummyClassifier):
                continue
            take = lambda ids: [X[i] for i in ids] if isinstance(X, list) else X[ids]
            try:
                fitted = clone(model).fit(take(subset), y[subset])
                metric = "balanced_accuracy" if task == "classification" else "RMSE"
                scores.append((_metrics(y[subset], fitted.predict(take(subset)), task, "")[metric],
                               _metrics(y[va], fitted.predict(take(va)), task, "")[metric]))
                sizes.append(len(chosen))
            except ValueError:
                continue
        if scores:
            samples.append({"training_groups": float(np.mean(sizes)), "train": float(np.mean([v[0] for v in scores])),
                            "validation": float(np.mean([v[1] for v in scores]))})
    return "Balanced accuracy" if task == "classification" else "RMSE", samples, "" if samples else "No feasible grouped learning-curve folds."


def auto_advisor(rows, target, task, n_grid, max_rows, group_field=None, split_manifest=None, candidate_configs=None):
    """Rank by inner grouped CV only; never evaluate the outer test spectra."""
    valid = _valid_rows(rows, target)
    groups, sources = groups_for_rows(valid, group_field)
    y = _target_array([target_value(r, target) for r in valid], task)
    tr, te, manifest = grouped_holdout(groups, _row_ids(valid), manifest=split_manifest)
    manifest["group_sources"] = sources
    training = [valid[i] for i in tr]
    ytr, gtr = y[tr], groups[tr]
    cv = cv_splitter(task, ytr, gtr)
    if cv is None:
        raise ValueError("Advisor needs at least two outer training groups for grouped CV.")
    folds = list(cv.split(np.zeros((len(tr), 1)), ytr, gtr))
    feature_sets = [[PEAK_DESCRIPTOR_FEATURE], ["raw", PEAK_DESCRIPTOR_FEATURE], ["derivative", PEAK_DESCRIPTOR_FEATURE],
                    ["cdf", PEAK_DESCRIPTOR_FEATURE], ["raw"], ["raw", "derivative"], ["raw", "cdf"]]
    models = ["Dummy baseline", "Logistic regression", "SVM", "Random forest"] if task == "classification" else ["Dummy baseline", "Ridge", "SVR", "Random forest"]
    min_fold = min(len(f[0]) for f in folds)
    configs = candidate_configs if candidate_configs is not None else [
        {"feature_set": fs, "normalization": norm, "model": name, "pca_components": comp}
        for fs in feature_sets for norm in ("minmax", "area", "zscore") for name in models
        for comp in [None] + [n for n in (2, 4, 8, 16) if n < min_fold and n <= (9 if fs == [PEAK_DESCRIPTOR_FEATURE] else n_grid)]
    ]
    results, failures = [], []
    for config in configs:
        fs, norm, name, comp = config["feature_set"], config["normalization"], config["model"], config.get("pca_components")
        try:
            pipeline = spectral_pipeline(task, name, fs, norm, n_grid, comp is not None, comp or 2)
            scores = cross_validate(pipeline, training, ytr, cv=folds,
                                    scoring="balanced_accuracy" if task == "classification" else "neg_root_mean_squared_error",
                                    return_train_score=True, error_score="raise")
            va, train = float(np.mean(scores["test_score"])), float(np.mean(scores["train_score"]))
            if not math.isfinite(va) or not math.isfinite(train):
                raise ValueError("Non-finite grouped CV score.")
            value, train_value = (va, train) if task == "classification" else (-va, -train)
            results.append({"feature_set": "+".join(fs), "normalization": norm,
                            "dimension_reduction": "none" if comp is None else f"PCA({comp})", "pca_components": comp,
                            "model": name, "validation_balanced_accuracy" if task == "classification" else "validation_RMSE": value,
                            "validation_std": float(np.std(scores["test_score"])), "train_score_same_metric": train_value,
                            "overfit_gap": train_value - value if task == "classification" else value - train_value,
                            "rank_score": va, "evaluation": "inner grouped cross-validation", "n_samples": len(tr),
                            "n_training_groups": len(np.unique(gtr)), "n_test_groups": len(np.unique(groups[te])),
                            "test_evaluated": False, "split_manifest": manifest,
                            "cv_folds": [{"train_groups": sorted(set(gtr[a])), "validation_groups": sorted(set(gtr[b]))} for a, b in folds]})
        except Exception as exc:
            failures.append(f"{'+'.join(fs)} / {norm} / {name} / {comp}: {exc}")
    results.sort(key=lambda r: (r["rank_score"], -abs(r["overfit_gap"])), reverse=True)
    for i, result in enumerate(results, 1):
        result["rank"] = i
    return results[:max_rows], failures


def run_auto_advisor(rows, target, task, n_grid, max_rows=15, **kwargs):
    """Select by grouped CV, then fit/evaluate the winner exactly once."""
    recommendations, failures = auto_advisor(rows, target, task, n_grid, max_rows, **kwargs)
    if not recommendations:
        raise ValueError("No feasible grouped-CV candidates. " + "; ".join(failures[:3]))
    best = recommendations[0]
    result = train_manual_rows(rows, target, task, best["model"], best["feature_set"].split("+"),
                               best["normalization"], n_grid, best["pca_components"] is not None,
                               best["pca_components"] or 2, kwargs.get("group_field"), best["split_manifest"])
    return {"recommendations": recommendations, "failures": failures, "result": result}


def export_model(result):
    """Serialize the fitted feature/scaler/PCA/estimator pipeline and provenance."""
    buffer = io.BytesIO()
    bundle = {"artifact_version": "2.0", "pipeline": result["model"], "target": result.get("target"),
              "task": result["task"], "recipe": result.get("recipe"), "split_manifest": result["split_manifest"],
              "metrics": result["metrics"], "dataset_sha256": result.get("dataset_sha256"), "versions": {"python": platform.python_version(), "numpy": np.__version__,
                                                       "sklearn": sklearn.__version__, "xas_ml_workflow": "2.0"}}
    joblib.dump(bundle, buffer)
    return buffer.getvalue()
