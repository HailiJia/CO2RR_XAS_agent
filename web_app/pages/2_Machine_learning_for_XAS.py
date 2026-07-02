# web_app/pages/2_Machine_learning_for_XAS.py
from __future__ import annotations

import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import streamlit as st

try:
    import pandas as pd
except Exception:
    pd = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.isaac_portal_client import ISAACPortalClient, xas_summary_table
from tools.xas_ml_utils import (
    SKLEARN_AVAILABLE,
    SKLEARN_IMPORT_ERROR,
    auto_advisor,
    feature_matrix,
    learning_curve_rows,
    train_manual,
)
from tools.xas_record_utils import (
    label_fields,
    minmax,
    read_records_from_uploads,
    rows_from_records,
    rows_to_csv,
    safe_str,
    summary_table,
    target_value,
    is_number,
)

ML_PAGE_UPDATE_TAG = "v43_2026-07-02_isaac_portal_paged_keyword_xas"
ISAAC_PORTAL_URL = "https://isaac.slac.stanford.edu/portal/"
SIMPLE_XAS_SUMMARY_COLUMNS = ["record_id", "record_domain", "formula", "material_name", "absorber", "edge", "technique"]
XAS_TEXT_RE = re.compile(r"\bxas\b|xanes|exafs|xafs|x-ray absorption|xray absorption|absorption spectroscopy", re.IGNORECASE)

st.set_page_config(page_title="CO2RR XAS Agent | Machine learning for XAS", layout="wide")
st.title("Agentic machine learning for XAS")
st.caption(f"Update tag: `{ML_PAGE_UPDATE_TAG}`")
st.markdown(f"""
**Goal.** Turn ISAAC-format XAS records into validated spectral datasets, model inputs,
trained ML models, performance diagnostics, training curves, and suggested next actions.

**Data source.** Upload local ISAAC JSON/ZIP records or retrieve XAS records directly from the
[ISAAC portal]({ISAAC_PORTAL_URL}).
""".strip())


def display_table(rows: Sequence[Dict[str, Any]], *, use_container_width: bool = True):
    if pd is not None:
        st.dataframe(pd.DataFrame(rows), use_container_width=use_container_width)
    else:
        st.table(rows)


def simple_xas_summary_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{key: row.get(key, "") for key in SIMPLE_XAS_SUMMARY_COLUMNS} for row in rows]


def record_has_xas_text(record: Dict[str, Any]) -> bool:
    def walk(obj: Any):
        if isinstance(obj, dict):
            for value in obj.values():
                yield from walk(value)
        elif isinstance(obj, list):
            for value in obj:
                yield from walk(value)
        elif isinstance(obj, str):
            yield obj
    return any(XAS_TEXT_RE.search(text or "") for text in walk(record))


def scan_portal_stubs(client: ISAACPortalClient, page_size: int, max_records: int) -> Dict[str, Any]:
    page_size = max(1, min(int(page_size), 500))
    max_records = max(1, int(max_records))
    all_stubs: List[Dict[str, Any]] = []
    offset = 0
    total_count = None
    pages = 0
    while len(all_stubs) < max_records:
        listed = client.list_records(limit=page_size, offset=offset)
        stubs = listed.get("records", [])
        if total_count is None:
            total_count = listed.get("total_count")
        pages += 1
        if not stubs:
            break
        remaining = max_records - len(all_stubs)
        all_stubs.extend(stubs[:remaining])
        if total_count is not None and offset + len(stubs) >= int(total_count):
            break
        if len(stubs) < page_size:
            break
        offset += page_size
    return {"records": all_stubs, "total_count": total_count, "pages": pages, "scanned": len(all_stubs), "page_size": page_size}


def records_from_portal_ui() -> List[Tuple[str, Dict[str, Any]]]:
    st.subheader("Retrieve XAS records from ISAAC Portal")
    env_url = os.environ.get("ISAAC_URL", "")
    env_key = os.environ.get("ISAAC_KEY", "")
    c1, c2 = st.columns(2)
    c1.caption(f"ISAAC_URL: `{env_url or 'not set; using default portal API URL'}`")
    c2.caption("ISAAC_KEY: " + ("found" if env_key else "missing"))
    if not env_key:
        st.warning("Set `ISAAC_KEY` in the terminal before launching Streamlit. Optionally set `ISAAC_URL`.")
        return st.session_state.get("isaac_portal_records", [])

    f1, f2, f3 = st.columns([1, 1, 2])
    page_size = f1.number_input("Page size", min_value=1, max_value=500, value=int(st.session_state.get("isaac_portal_page_size", 500)), step=50)
    max_scan = f2.number_input("Max records to scan", min_value=1, max_value=10000, value=int(st.session_state.get("isaac_portal_max_scan", 2500)), step=500)
    domain = f3.selectbox("Domain filter", ["all", "characterization", "simulation"], index=1)
    st.session_state["isaac_portal_page_size"] = int(page_size)
    st.session_state["isaac_portal_max_scan"] = int(max_scan)

    st.caption("Page size is capped at 500 by the portal. To scan more than 500 records, this app requests multiple pages with offsets 0, 500, 1000, ... and then applies the domain/XAS filters locally.")

    if st.button("Scan portal record stubs", type="secondary"):
        try:
            client = ISAACPortalClient()
            scanned = scan_portal_stubs(client, page_size=int(page_size), max_records=int(max_scan))
            st.session_state["isaac_portal_list"] = scanned
        except Exception as exc:
            st.error(f"Failed to scan ISAAC Portal records: {exc}")

    listed = st.session_state.get("isaac_portal_list")
    displayed_stubs: List[Dict[str, Any]] = []
    if listed:
        all_stubs = listed.get("records", [])
        displayed_stubs = [r for r in all_stubs if domain == "all" or r.get("record_domain") == domain]
        st.caption(
            f"Scanned {len(all_stubs)} stubs across {listed.get('pages')} page(s) with page size {listed.get('page_size')}. "
            f"Displaying {len(displayed_stubs)} after domain filter `{domain}`. "
            f"Total count header: {listed.get('total_count')} (API-wide count, not domain/XAS-filtered)."
        )
        if displayed_stubs:
            display_table(displayed_stubs)
        else:
            st.info("No scanned stubs match the selected domain. Increase max records to scan or change the domain.")

    if displayed_stubs:
        record_ids = [r.get("record_id", "") for r in displayed_stubs if r.get("record_id")]
        c_fetch1, c_fetch2 = st.columns([1, 1])
        fetch_mode = c_fetch1.radio("Full-record fetch mode", ["Fetch all displayed stubs", "Fetch selected stubs"], horizontal=False)
        max_fetch_default = min(max(len(record_ids), 1), int(st.session_state.get("isaac_portal_max_fetch", min(len(record_ids), 2000))))
        max_fetch = c_fetch2.number_input("Max full records to fetch", min_value=1, max_value=10000, value=max_fetch_default, step=100)
        st.session_state["isaac_portal_max_fetch"] = int(max_fetch)
        if fetch_mode == "Fetch selected stubs":
            default_ids = record_ids[: min(len(record_ids), 100)]
            selected_ids = st.multiselect("Records to fetch in full", record_ids, default=default_ids)
        else:
            selected_ids = record_ids[: int(max_fetch)]
            st.caption(f"Will fetch {len(selected_ids)} full record(s) from the {len(record_ids)} displayed stub(s).")
        if st.button("Fetch full records and find XAS", type="primary", disabled=not selected_ids):
            try:
                client = ISAACPortalClient()
                records = client.fetch_records(selected_ids)
                st.session_state["isaac_portal_records"] = records
                st.success(f"Fetched {len(records)} full record(s) from ISAAC Portal.")
            except Exception as exc:
                st.error(f"Failed to fetch ISAAC Portal records: {exc}")

    records = st.session_state.get("isaac_portal_records", [])
    if not records:
        return []

    records_after_domain = [(source, rec) for source, rec in records if domain == "all" or rec.get("record_domain") == domain]
    record_summary = xas_summary_table(records_after_domain, xas_only=False)
    records_by_id = {rec.get("record_id"): rec for _, rec in records_after_domain}
    for row in record_summary:
        rec = records_by_id.get(row.get("record_id"), {})
        row["xas_text_match"] = record_has_xas_text(rec)
        row["is_xas"] = bool(row.get("is_xas") or row.get("xas_text_match"))
    xas_summary = [r for r in record_summary if r.get("is_xas")]
    xas_ids = {r.get("record_id") for r in xas_summary}
    simple_summary = simple_xas_summary_rows(xas_summary)
    st.success(f"Fetched-cache records after domain filter: {len(record_summary)}. XAS matches: {len(xas_summary)}.")
    with st.expander("Simple XAS summary table", expanded=True):
        display_table(simple_summary)
        st.download_button("Download simple XAS summary CSV", data=rows_to_csv(simple_summary), file_name="isaac_portal_simple_xas_summary.csv", mime="text/csv")

    if not xas_summary:
        st.warning("No XAS records were detected in the fetched full records. XAS detection uses string keyword matching plus absorber/edge/energy fallback.")
        return []

    use_all_xas = st.checkbox("Use all XAS records in the summary table", value=True)
    absorber_options = sorted({safe_str(r.get("absorber")) for r in xas_summary if safe_str(r.get("absorber"))})
    edge_options = sorted({safe_str(r.get("edge")) for r in xas_summary if safe_str(r.get("edge"))})
    c1, c2, c3 = st.columns(3)
    sel_abs = c1.multiselect("Absorber filter", absorber_options, default=absorber_options, disabled=use_all_xas)
    sel_edge = c2.multiselect("Edge filter", edge_options, default=edge_options, disabled=use_all_xas)
    material_query = c3.text_input("Material/formula contains", value="", disabled=use_all_xas)

    if use_all_xas:
        selected_ids = set(xas_ids)
    else:
        selected_ids = {
            r.get("record_id")
            for r in xas_summary
            if (not sel_abs or r.get("absorber") in sel_abs)
            and (not sel_edge or r.get("edge") in sel_edge)
            and (
                not material_query
                or material_query.lower() in safe_str(r.get("formula")).lower()
                or material_query.lower() in safe_str(r.get("material_name")).lower()
            )
        }
    filtered = [(source, rec) for source, rec in records_after_domain if rec.get("record_id") in selected_ids]
    st.info(f"Using {len(filtered)} XAS record(s) for plotting/training.")
    return filtered


data_source = st.radio("Data source", ["Upload local ISAAC files", "Retrieve from ISAAC Portal"], horizontal=True)

if data_source == "Upload local ISAAC files":
    uploaded = st.file_uploader("Upload ISAAC record JSON files or a ZIP folder of ISAAC records", type=["json", "zip"], accept_multiple_files=True)
    if not uploaded:
        st.info("Upload one or more ISAAC JSON records, or a ZIP containing JSON records.")
        st.stop()
    try:
        records = read_records_from_uploads(uploaded)
    except Exception as exc:
        st.error(f"Failed to load records: {exc}")
        st.stop()
else:
    records = records_from_portal_ui()
    if not records:
        st.info("Scan portal stubs, fetch full records, and keep XAS filters broad enough to include at least one record.")
        st.stop()

try:
    rows = rows_from_records(records)
except Exception as exc:
    st.error(f"Failed to parse records into spectra: {exc}")
    st.stop()

if data_source == "Retrieve from ISAAC Portal":
    for row in rows:
        row["is_xas"] = True
        if row.get("classification") == "not_plottable_invalid_xy":
            continue
        if row.get("classification", "").startswith("not_xas"):
            row["classification"] = "xas_keyword_match"
            row["reason"] = "record matched XAS keyword search in full ISAAC JSON"

st.success(f"Loaded {len(records)} JSON record(s) and detected {len(rows)} spectral/record row(s).")
plottable = [r for r in rows if r.get("x") is not None and row.get("y") is not None and r.get("n_points", 0) > 0]
xas_rows = [r for r in plottable if r.get("is_xas")]
non_xas = [r for r in rows if not r.get("is_xas")]
missing_units = [r for r in plottable if not safe_str(r.get("x_unit")).strip() or not safe_str(r.get("y_unit")).strip()]
groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in xas_rows})

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("JSON records", len(records))
c2.metric("Rows", len(rows))
c3.metric("XAS-like", len(xas_rows))
c4.metric("Absorber/edge groups", len(groups))
c5.metric("Missing x/y unit", len(missing_units))
if missing_units:
    st.warning("Some plottable series are missing x or y units.")
else:
    st.success("All plottable series have both x and y units.")

summary_rows = summary_table(rows)
with st.expander("Series classification table", expanded=True):
    display_table(summary_rows)
    st.download_button("Download classification CSV", data=rows_to_csv(summary_rows), file_name="isaac_series_classification.csv", mime="text/csv")
if non_xas:
    with st.expander("Non-XAS or questionable records", expanded=True):
        st.write("These rows are excluded by default from XAS ML because their axes/channels are not incident-energy XAS spectra.")
        display_table(summary_table(non_xas))

st.divider()
st.header("Plot spectra by absorber and edge")
plot_scope = st.radio("Records to plot", ["XAS-like only", "All plottable series", "Non-XAS only"], horizontal=True)
if plot_scope == "XAS-like only":
    plot_rows = xas_rows
elif plot_scope == "Non-XAS only":
    plot_rows = [r for r in non_xas if r.get("x") is not None and r.get("y") is not None]
else:
    plot_rows = plottable
plot_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in plot_rows})
sel_groups = st.multiselect("Absorber/edge groups", plot_groups, default=([g for g in plot_groups if g != "Unknown absorber/edge"] or plot_groups)[:4])
plot_mode = st.selectbox("Display mode", ["raw", "min-max normalized", "min-max normalized with vertical offset"], index=1)
if pd is None:
    st.warning("pandas is not available, so plotting is disabled.")
else:
    for group in sel_groups:
        group_rows = [r for r in plot_rows if r.get("edge_group") == group]
        with st.expander(f"{group} ({len(group_rows)} series)", expanded=len(sel_groups) <= 2):
            plot_records = []
            for offset, row in enumerate(group_rows[:20]):
                x = np.asarray(row["x"], dtype=float)
                y = np.asarray(row["y"], dtype=float)
                if plot_mode.startswith("min-max"):
                    y = minmax(y)
                if plot_mode.endswith("vertical offset"):
                    y = y + offset * 1.15
                label = " | ".join(v for v in [safe_str(row.get("record_id")) or safe_str(row.get("source_name")), safe_str(row.get("formula")), safe_str(row.get("series_id"))] if v)
                plot_records.extend([{"x": float(a), "y": float(b), "label": label} for a, b in zip(x, y)])
            if plot_records:
                st.line_chart(pd.DataFrame(plot_records), x="x", y="y", color="label", use_container_width=True)

st.divider()
st.header("Train ML model")
train_scope = st.radio("Training records", ["XAS-like only", "All plottable series"], horizontal=True)
train_rows = xas_rows if train_scope == "XAS-like only" else plottable
train_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in train_rows})
sel_train_groups = st.multiselect("Absorber/edge groups for training", train_groups, default=[g for g in train_groups if g != "Unknown absorber/edge"] or train_groups)
train_rows = [r for r in train_rows if r.get("edge_group") in sel_train_groups]
if len(train_rows) < 2:
    st.info("Need at least two plottable labeled spectra for ML setup.")
    st.stop()

label_candidates = label_fields(train_rows)
if not label_candidates:
    st.warning("No scalar label fields found in these ISAAC records.")
    st.stop()
target = st.selectbox("Target / label field", label_candidates, index=label_candidates.index("sample.material.formula") if "sample.material.formula" in label_candidates else 0)
label_vals = [v for v in [target_value(r, target) for r in train_rows] if v not in [None, ""]]
reg_ok = bool(label_vals) and all(is_number(v) for v in label_vals)
task = st.radio("Task type", ["classification", "regression"], index=1 if reg_ok and target.startswith("descriptors.") else 0, horizontal=True)
if not SKLEARN_AVAILABLE:
    st.warning(f"scikit-learn is not available, so training is disabled. Import error: {SKLEARN_IMPORT_ERROR}")

st.subheader("Auto-advisor")
st.write("Run a deterministic comparison of feature sets, normalization choices, PCA/no-PCA, and simple models.")
a1, a2 = st.columns(2)
advisor_grid = a1.slider("Auto-advisor common-grid points", 32, 512, 128, step=32)
advisor_rows = a2.slider("Configurations to show", 5, 40, 15, step=5)
if st.button("Run Auto-advisor comparison", disabled=not SKLEARN_AVAILABLE, type="secondary"):
    try:
        res, failures = auto_advisor(train_rows, target, task, advisor_grid, advisor_rows)
        if res:
            st.success("Auto-advisor finished. Use the recommendation as a starting point, then verify with grouped validation.")
            best = res[0]
            st.markdown(f"**Recommendation:** `{best['feature_set']}` features, `{best['normalization']}` normalization, `{best['dimension_reduction']}`, `{best['model']}`.")
            if pd is not None:
                df = pd.DataFrame(res)
                st.dataframe(df.drop(columns=["rank_score"], errors="ignore"), use_container_width=True)
                st.download_button("Download Auto-advisor CSV", data=df.to_csv(index=False), file_name="xas_ml_auto_advisor_results.csv", mime="text/csv")
            else:
                st.table(res)
        else:
            st.warning("Auto-advisor could not rank any configuration. Check labels, record count, and task type.")
        if failures:
            with st.expander("Skipped/failed configurations", expanded=False):
                st.write(failures[:80])
    except Exception as exc:
        st.error(f"Auto-advisor failed: {exc}")

st.subheader("Manual model run")
feature_kinds = st.multiselect("Input features", ["raw", "derivative", "cdf", "peak_window"], default=["raw", "peak_window"])
norm = st.selectbox("Spectrum normalization", ["minmax", "area", "zscore", "none"], index=0)
n_grid = st.slider("Common-grid points", 32, 1024, 256, step=32)
model_options = ["Dummy baseline", "Logistic regression", "Random forest", "SVM"] if task == "classification" else ["Dummy baseline", "Ridge", "Random forest", "SVR"]
model_name = st.selectbox("Model", model_options, index=1 if len(model_options) > 1 else 0)
use_pca = st.checkbox("Use PCA before model", value=False)
if use_pca:
    n_comp = st.slider("PCA components", 1, min(32, max(1, len(train_rows) - 1)), min(4, max(1, len(train_rows) - 1)))
else:
    n_comp = 2

if st.button("Train model", type="primary", disabled=not SKLEARN_AVAILABLE):
    try:
        X, y, feature_names, grid = feature_matrix(train_rows, target, feature_kinds, norm, n_grid)
        result = train_manual(X, y, task, model_name, use_pca, n_comp)
        st.success(f"Trained {model_name} on {X.shape[0]} spectra with {X.shape[1]} features.")
        metric_items = list(result["metrics"].items())
        cols = st.columns(max(1, len(metric_items)))
        for col, (key, value) in zip(cols, metric_items):
            if isinstance(value, float) and math.isfinite(value):
                col.metric(key, f"{value:.4g}")
            else:
                col.metric(key, str(value))
        st.subheader("Held-out predictions")
        if pd is not None:
            st.dataframe(pd.DataFrame(result["table"]), use_container_width=True)
        else:
            st.table(result["table"])
        st.subheader("Training curve")
        try:
            ylabel, curve_rows, note = learning_curve_rows(result["model"], X, result["y_model"], task)
            if note:
                st.info(note)
            elif curve_rows and pd is not None:
                st.caption(ylabel)
                curve_df = pd.DataFrame(curve_rows)
                st.line_chart(curve_df, x="training_size", y=["train", "validation"], use_container_width=True)
                st.dataframe(curve_df, use_container_width=True)
            elif curve_rows:
                st.table(curve_rows)
            else:
                st.warning("Training curve could not be computed with finite scores for this small dataset.")
        except Exception as exc:
            st.warning(f"Training curve skipped: {exc}")
    except Exception as exc:
        st.error(f"Training failed: {exc}")
