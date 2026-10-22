# Reproducible Cu K-edge identity and depth study

The manuscript source and five-figure package in [`manuscript_package/`](manuscript_package/) are the authoritative revised package. The executable analysis, exact plotted data, and outputs remain alongside it. The archived benchmark under `CO2RR_ML/updated_dataset_20261020/` is preserved; the primary nested selector stays distinct from its exploratory fixed winner.

The checked-in package reproduces figures and saved-OOF metrics without the full local simulation tree. A complete 87-configuration export is available locally for a fresh audit and fit; it is intentionally not included in this repository. The notebook and shared CLIs support relocating that export.

## Reproduce figures and archived predictions

From the repository root, install `study/requirements-replay.txt`, then run:

```bash
python -m tools.study_cli replay --output-dir /tmp/co2rr_oof_replay
python study/manuscript_package/code/verify_package.py
python study/manuscript_package/code/make_figures.py
```

This recomputes metrics from saved OOF truth/prediction arrays and checks per-species tables and confusion matrices. It is figure/OOF reproduction; it does not rerun the structural or layer-population audit.

## Audit and fit from a complete local export

```bash
python -m tools.study_cli verify-export --export-root /path/to/cohort_export --output-dir /tmp/export_check
python -m tools.study_cli audit --data-root /path/to/cohort_export/data --split-manifest /path/to/cohort_export/splits.csv --output-dir /tmp/audit
python -m tools.audited_training --data-root /path/to/cohort_export/data --split-manifest /path/to/cohort_export/splits.csv --output-dir /path/to/new-empty-run
```

`audited_training` verifies byte hashes, audits final coordinates and populations, checks current-geometry fold groups, executes the archived nested-selector workflow in a fresh output/cache, exports fold models, and verifies held-out predictions after import. `training_only_global_threshold` is a separate new recipe, not a reproduction of the archived selector. The natural-language agent and multilabel web route call this same API.

## Tests and manuscript build status

Install `study/requirements-tests.txt` for the focused scientific and NERSC workflow tests. The local-fixture scheduler is explicitly simulated and cannot establish production retry, intervention, cost, or runtime rates. See [`EVIDENCE.md`](EVIDENCE.md) for measured, implementation-tested, planned, and unavailable items.

The revised manuscript source is `manuscript_package/manuscript.tex`; its five figures retain PDF/SVG/PNG plus exact plotted data. A TeX engine and SI source are not present in this environment, so the editable source is authoritative and final PDF compilation/page inspection remain outstanding. The supplied pre-edit PDF is retained only as an evidence artifact, outside the manuscript release.
