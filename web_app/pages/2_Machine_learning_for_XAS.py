from __future__ import annotations

from pathlib import Path


_IMPL_PATH = Path(__file__).resolve().parents[1] / "ml_pages" / "machine_learning_for_XAS_impl.py"
_source = _IMPL_PATH.read_text()
_source = _source.replace('"peak_window"', '"peak_descriptors"')
_source = _source.replace(
    'ML_PAGE_UPDATE_TAG = "v46_2026-07-03_xas_ml_target_cleanup"',
    'ML_PAGE_UPDATE_TAG = "v48_2026-08-13_signal_type_filter"',
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

plot_groups = sorted({r.get("edge_group", "Unknown absorber/edge") for r in plot_rows})'''
if _plot_old not in _source:
    raise RuntimeError("Signal-type patch could not find the plot filter block.")
_source = _source.replace(_plot_old, _plot_new)

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
