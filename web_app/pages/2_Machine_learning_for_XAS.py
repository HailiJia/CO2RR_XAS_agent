from __future__ import annotations

import sys
from pathlib import Path


APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from top_navigation import install_top_navigation

install_top_navigation(active_page="ml")

_IMPL_PATH = APP_DIR / "ml_pages" / "machine_learning_for_XAS_impl.py"
_source = _IMPL_PATH.read_text()
_source = _source.replace('"peak_window"', '"peak_descriptors"')
_source = _source.replace(
    'ML_PAGE_UPDATE_TAG = "v46_2026-07-03_xas_ml_target_cleanup"',
    'ML_PAGE_UPDATE_TAG = "v49_2026-08-14_signal_filter_downloads"',
)

_loaded_old = '''st.success(f"Loaded {len(records)} JSON record(s) and detected {len(rows)} spectral/record row(s).")
plottable = [r for r in rows if r.get("x") is not None and r.get("y") is not None and r.get("n_points", 0) > 0]'''
_loaded_new = '''st.success(f"Loaded {len(records)} JSON record(s) and detected {len(rows)} spectral/record row(s).")

def _xas_signal_type(row):
    text = " ".join(
        safe_str(row.get(key))
        for key in ("y_name", "series_id")
        if safe_str(row.get(key))
    ).lower()
    text = re.sub(r"[\\s_-]+", " ", text)
    if any(token in text for token in ("delta absorption", "difference absorption", "delta mu", "dmu")):
        return "Delta absorption"
    if any(
        token in text
        for token in (
            "absorption", "absorbance", "mu", "xanes", "xas",
            "tey", "tfy", "pfy", "herfd", "fluorescence", "intensity",
        )
    ):
        return "Absorption"
    return "Other"

for _row in rows:
    _row["signal_type"] = _xas_signal_type(_row)

_signal_counts = {}
for _row in rows:
    _signal_type = _row.get("signal_type", "Other")
    _signal_counts[_signal_type] = _signal_counts.get(_signal_type, 0) + 1
if _signal_counts:
    st.caption(
        "Spectral rows by signal type: "
        + ", ".join(f"{key}: {value}" for key, value in sorted(_signal_counts.items()))
    )

plottable = [r for r in rows if r.get("x") is not None and r.get("y") is not None and r.get("n_points", 0) > 0]'''
if _loaded_old not in _source:
    raise RuntimeError("Signal-type patch could not find the loaded-records block.")
_source = _source.replace(_loaded_old, _loaded_new)

_adsorbate_old = '''        metadata.update(parsed_structure_labels(record))'''
_adsorbate_new = '''        metadata.update(parsed_structure_labels(record))
        explicit_adsorbate = (
            get_path(record, "system.configuration.adsorbate.identity")
            or get_path(record, "system.configuration.adsorbate_identity")
            or get_path(record, "sample.adsorbate.identity")
        )
        if explicit_adsorbate:
            metadata["structure.adsorbate"] = safe_str(explicit_adsorbate)
        else:
            material_name = safe_str(get_path(record, "sample.material.name"))
            adsorbate_match = re.search(r"\\bwith\\s+([A-Za-z0-9_+\\-]+)\\s+adsorbate\\b", material_name, re.I)
            if adsorbate_match:
                metadata["structure.adsorbate"] = adsorbate_match.group(1)
            elif material_name and "slab" in material_name.lower() and "adsorbate" not in material_name.lower():
                metadata["structure.adsorbate"] = "clean"'''
if _adsorbate_old not in _source:
    raise RuntimeError("Adsorbate-target patch could not find the ML metadata block.")
_source = _source.replace(_adsorbate_old, _adsorbate_new)
_source = _source.replace(
    'for key in ["structure.facet", "structure.adsorption_site"]:',
    'for key in ["structure.adsorbate", "structure.facet", "structure.adsorption_site"]:',
)
_source = _source.replace(
    'Only chemically useful XAS ML targets are shown: material formula, non-null descriptors, and optional parsed facet/site labels.',
    'Only chemically useful XAS ML targets are shown: material formula, non-null descriptors, and optional parsed adsorbate/facet/site labels.',
)

_plot_old = '''else:
    plot_rows = plottable
plot_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in plot_rows})'''
_plot_new = '''else:
    plot_rows = plottable

_plot_signal_types = sorted({r.get("signal_type", "Other") for r in plot_rows})
_plot_signal_options = [
    value for value in ("Absorption", "Delta absorption", "Other")
    if value in _plot_signal_types
]
_plot_signal_options.append("All")
plot_signal_filter = st.radio(
    "Signal type",
    _plot_signal_options,
    index=0,
    horizontal=True,
    help="Keep absolute absorption and delta-absorption spectra separate for plotting and comparison.",
)
if plot_signal_filter != "All":
    plot_rows = [r for r in plot_rows if r.get("signal_type", "Other") == plot_signal_filter]

prepare_ml_target_metadata(plot_rows)
plot_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in plot_rows})'''
if _plot_old not in _source:
    raise RuntimeError("Signal-type patch could not find the plot filter block.")
_source = _source.replace(_plot_old, _plot_new)

_download_anchor = '''plot_mode = st.selectbox("Display mode", ["raw", "min-max normalized", "min-max normalized with vertical offset"], index=1)
if pd is None:'''
_download_insert = '''plot_mode = st.selectbox("Display mode", ["raw", "min-max normalized", "min-max normalized with vertical offset"], index=1)

selected_plot_rows = [r for r in plot_rows if r.get("edge_group") in sel_groups]
st.subheader("Download data")
st.caption("Export the currently selected signal type and absorber/edge groups for publication-quality plotting in Python, Origin, MATLAB, or other tools.")
if pd is None:
    st.warning("pandas is required to prepare the spectral download.")
elif not selected_plot_rows:
    st.info("Select at least one absorber/edge group above.")
else:
    download_points = []
    download_series = []
    for row in selected_plot_rows:
        record = row.get("record") or {}
        material_name = safe_str(get_path(record, "sample.material.name"))
        adsorbate = safe_str(target_value(row, "structure.adsorbate"))
        site = safe_str(target_value(row, "structure.adsorption_site"))
        facet = safe_str(target_value(row, "structure.facet"))
        series_label_parts = [adsorbate or material_name or safe_str(row.get("formula")), site]
        series_label = " | ".join(v for v in series_label_parts if v)
        if not series_label:
            series_label = safe_str(row.get("record_id")) or safe_str(row.get("source_name")) or safe_str(row.get("series_id"))
        x = np.asarray(row["x"], dtype=float)
        y = np.asarray(row["y"], dtype=float)
        download_series.append({
            "label": series_label,
            "record_id": safe_str(row.get("record_id")),
            "series_id": safe_str(row.get("series_id")),
            "material_name": material_name,
            "formula": safe_str(row.get("formula")),
            "adsorbate": adsorbate,
            "adsorption_site": site,
            "facet": facet,
            "absorber": safe_str(row.get("absorber")),
            "edge": safe_str(row.get("edge")),
            "signal_type": safe_str(row.get("signal_type")),
            "x_name": safe_str(row.get("x_name")),
            "x_unit": safe_str(row.get("x_unit")),
            "y_name": safe_str(row.get("y_name")),
            "y_unit": safe_str(row.get("y_unit")),
            "n_points": int(row.get("n_points") or len(x)),
        })
        for xv, yv in zip(x, y):
            download_points.append({
                "label": series_label,
                "record_id": safe_str(row.get("record_id")),
                "series_id": safe_str(row.get("series_id")),
                "material_name": material_name,
                "formula": safe_str(row.get("formula")),
                "adsorbate": adsorbate,
                "adsorption_site": site,
                "facet": facet,
                "absorber": safe_str(row.get("absorber")),
                "edge": safe_str(row.get("edge")),
                "signal_type": safe_str(row.get("signal_type")),
                "energy": float(xv),
                "signal": float(yv),
                "energy_unit": safe_str(row.get("x_unit")),
                "signal_unit": safe_str(row.get("y_unit")),
            })
    download_df = pd.DataFrame(download_points)
    series_df = pd.DataFrame(download_series)
    st.write(f"Selected `{len(series_df)}` spectral series and `{len(download_df)}` data points.")
    display_table(download_series)
    c_dl1, c_dl2 = st.columns(2)
    c_dl1.download_button(
        "Download spectra CSV",
        data=download_df.to_csv(index=False),
        file_name="isaac_xas_selected_spectra.csv",
        mime="text/csv",
        use_container_width=True,
    )
    c_dl2.download_button(
        "Download series metadata CSV",
        data=series_df.to_csv(index=False),
        file_name="isaac_xas_selected_series_metadata.csv",
        mime="text/csv",
        use_container_width=True,
    )

if pd is None:'''
if _download_anchor not in _source:
    raise RuntimeError("Download-data patch could not find the plotting anchor.")
_source = _source.replace(_download_anchor, _download_insert)

_label_old = '''                label = " | ".join(v for v in [safe_str(row.get("record_id")) or safe_str(row.get("source_name")), safe_str(row.get("formula")), safe_str(row.get("series_id"))] if v)'''
_label_new = '''                adsorbate = safe_str(target_value(row, "structure.adsorbate"))
                site = safe_str(target_value(row, "structure.adsorption_site"))
                material_name = safe_str(get_path(row.get("record") or {}, "sample.material.name"))
                label_parts = [adsorbate or material_name or safe_str(row.get("formula")), site]
                label = " | ".join(v for v in label_parts if v)
                if not label:
                    label = safe_str(row.get("record_id")) or safe_str(row.get("source_name")) or safe_str(row.get("series_id"))'''
if _label_old not in _source:
    raise RuntimeError("Plot-label patch could not find the original label expression.")
_source = _source.replace(_label_old, _label_new)

_train_old = '''train_scope = st.radio("Training records", ["XAS-like only", "All plottable series"], horizontal=True)
train_rows = xas_rows if train_scope == "XAS-like only" else plottable
train_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in train_rows})'''
_train_new = '''train_scope = st.radio("Training records", ["XAS-like only", "All plottable series"], horizontal=True)
train_rows = xas_rows if train_scope == "XAS-like only" else plottable

_train_signal_types = sorted({r.get("signal_type", "Other") for r in train_rows})
_train_signal_options = [
    value for value in ("Absorption", "Delta absorption", "Other")
    if value in _train_signal_types
]
_train_signal_options.append("All")
train_signal_filter = st.radio(
    "Training signal type",
    _train_signal_options,
    index=0,
    horizontal=True,
    help="Do not mix absolute absorption and delta-absorption spectra in the same ML dataset unless that is intentional.",
)
if train_signal_filter != "All":
    train_rows = [r for r in train_rows if r.get("signal_type", "Other") == train_signal_filter]

train_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in train_rows})'''
if _train_old not in _source:
    raise RuntimeError("Signal-type patch could not find the training filter block.")
_source = _source.replace(_train_old, _train_new)

exec(compile(_source, str(_IMPL_PATH), "exec"), globals())
