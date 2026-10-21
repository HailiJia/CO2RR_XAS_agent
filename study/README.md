# Reproducible Cu K-edge study and implementation reconciliation

Reviewed upstream commit: **71bb0a20b369faba842b413a09544dd66c127a78**, checked 21 October 2026. There were no newer commits and no repository `AGENTS.md` files. Work is isolated on `study/reproducible-xas-reconciliation`; the requested deliverable is a reviewable PR, not a merge or data deposition.

The authoritative archived result remains in [`CO2RR_ML/updated_dataset_20261020`](../CO2RR_ML/updated_dataset_20261020/README.md). Its saved OOF arrays, executed historical notebook and numerical results are preserved. The portable source entry points write to a **new** output directory. See [EVIDENCE.md](EVIDENCE.md), [CHANGE_REPORT.md](CHANGE_REPORT.md), [captions](captions.md), and the [working manuscript](manuscript.tex).

The named supplied manuscript package was not accessible. This directory is an explicitly separate working package based on the accessible two-page outline and benchmark artifacts. It does not pretend to be the supplied draft or replace unmeasured panels with invented results. `provenance/figure_sources.json` identifies every panel's evidence; PDF/SVG/PNG and plotted CSV/JSON data are retained.

## Environment

Python **3.11** was used. Install from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r study/requirements-replay.txt
# Additional dependencies for generator/integration tests:
python -m pip install -r study/requirements-tests.txt
```

Exact tested versions and the actual existing-environment test setup are recorded in `provenance/dependencies.json`. A fresh dependency installation could not be exercised here because pip network resolution was unavailable; the scientific runs used the installed pinned scientific environment and an isolated exported dataset, without old fit caches. No HPC execution or portal requests were used to validate this package.

## Figure-only and archived OOF reproduction

This mode needs no local simulation tree and performs no fitting or structural audit:

```bash
python -m tools.study_cli replay --output-dir /tmp/co2rr_oof_replay
python study/code/prepare_figure_data.py --output-dir /tmp/co2rr_figure_data
python study/code/make_figures.py \
  --data-dir /tmp/co2rr_figure_data --output-dir /tmp/co2rr_figures
```

The scripts replay all 28 nested workflow/objective outputs plus the exploratory fixed entry, verify per-species metrics, confusion matrices and pure/mixed recall against the archived CSVs, and derive training-only constant baselines on the same outer folds. The majority binary constant predicts no species here (there are no clean controls): micro/macro F1 and exact accuracy are zero. The training-most-common-set baseline gives micro/macro F1 **0.3225806452 / 0.0466200466**, exact accuracy **0.2413793103**.

Committed `figure_data/layer_spectra.csv` contains actual normalized layer means and configuration SD, allowing figure-only replay without raw spectra. This is distinct from validating or recomputing those values from records. Supply `--data-root` to `prepare_figure_data.py` to extract them from raw saved records. The five new manuscript-role figures do not replace the five historical standalone-analysis figure roles.

## Exact 87-configuration export

Configure paths rather than editing source:

```bash
DATA=/path/to/full/CO2RR
ANALYSIS=CO2RR_ML/updated_dataset_20261020
COHORT=/path/to/new/cohort_export_v1_20261021
python -m tools.study_cli export \
  --data-root "$DATA" --analysis-dir "$ANALYSIS" \
  --split-manifest "$ANALYSIS/cv_manifest.csv" --output-dir "$COHORT"
python -m tools.study_cli verify-export \
  --export-root "$COHORT" --output-dir /tmp/co2rr_export_check
```

The export is opt-in and refuses to overwrite a populated destination. It copies final calculation coordinates, source/initial/final alias coordinates, immutable audited records, tables, main/resume/final convolution decks, final and attempt logs, broadened arrays, every required site spectrum, frozen folds and target-correction provenance. Potential files are excluded; potential identifiers and SHA256 checksums are recorded. Credentials, executables, wavefunctions and restricted potentials are not copied. Source files are never edited by export.

`dataset_manifest.json` records source and stored hashes, aliases, folds, initial/final hashes, settings/versions, correction metadata, and the full byte manifest. Large text logs use lossless gzip by default; verification checks both compressed bytes and the recovered original bytes. `--full-logs` copies uncompressed logs instead. The tested export contains 87 configurations, 18,700 hash-listed files (excluding the self-referential manifest), about 23.49 GB of original scientific bytes and 6.64 GB of stored bytes. The separate verification report records the manifest's SHA256. The export remains local and unpublished.

The exact-cohort export deliberately excludes duplicate/excluded records and unfinished structures' full files. Their frozen inventory/audit metadata remains in `protocol/`; this permits inventory plots without pretending those absent coordinates were re-audited. Figure-only reconstruction uses archived full-inventory provenance. Full scientific audit and fitting operate on the 87 eligible configurations.

## Relocated full scientific audit and fitting

The shared utility reads compressed logs directly; decompression is not required:

```bash
python -m tools.study_cli audit \
  --data-root "$COHORT/data" --split-manifest "$COHORT/splits.csv" \
  --output-dir /tmp/co2rr_relocated_audit
python -m tools.study_cli model-replay \
  --data-root "$COHORT/data" --split-manifest "$COHORT/splits.csv" \
  --output-dir /tmp/co2rr_saved_models
```

Model replay refits the five archived primary outer-fold models using their archived training-selected settings and thresholds, saves/imports them, and predicts from raw saved records. It checks prediction identity against the archived held-out arrays. It is not new independent confirmation. Model artifacts contain energy grid, recipe, physical populations, class ordering, scaler/estimators, thresholds, training/split/data linkage, dependency/code fingerprints and the prediction function. `tools.spectral_multilabel.predict_records(bundle, records, populations)` rejects incompatible grids and requires explicitly supplied represented populations.

To repeat the complete 179-recipe inner/outer fitting and figure generation in a new directory:

```bash
RUN=/path/to/new/portable_run
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python CO2RR_ML/updated_dataset_20261020/precompute_fit_cache.py \
  --data-root "$COHORT/data" --output-dir "$RUN" \
  --split-manifest "$COHORT/splits.csv"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
python CO2RR_ML/updated_dataset_20261020/build_and_execute.py \
  --data-root "$COHORT/data" --output-dir "$RUN" \
  --split-manifest "$COHORT/splits.csv"
```

The parallel cache helper is optional; the notebook builder can perform every fit itself. `analysis.py` accepts the same arguments and `--audit-only`. The builder supports `--audit-only` and `--build-only`. In an interactive notebook set `CO2RR_DATA_ROOT`, `CO2RR_OUTPUT_DIR`, and optionally `CO2RR_SPLIT_MANIFEST`; set `CO2RR_REPO_ROOT` if the notebook is opened outside the checkout. The canonical entry point locates its protocol/source within the repository. It refuses to overwrite the archived directory.

Fit-cache keys include record/split manifests, parameters, source-code hashes, executed recipe signatures and dependency versions. Cached train/test indices and fingerprints are verified before reuse. The current-geometry grouping audit enforces linkage across aliases even with explicit group fields. Changes to frozen calculation coordinates or record hashes require an explicit new version/comparison; they cannot silently alter this benchmark. The legacy hash is used only to verify archived source linkage; new geometry grouping and connectivity use in-plane PBC with nonperiodic vacuum.

Portable archived validation works without old caches or absolute backup paths:

```bash
python CO2RR_ML/updated_dataset_20261020/validate_outputs.py \
  --output-dir /tmp/co2rr_validation \
  --split-manifest "$COHORT/splits.csv"
python CO2RR_ML/updated_dataset_20261020/audit_two_CO_population_records.py \
  --data-root "$COHORT/data" --output-dir /tmp/co2rr_population_check --audit-only
```

## Shared chemistry, populations and application route

`tools/component_audit.py` discloses fallback cutoffs, atom indices, requested mapping/formulas, observed components and ±10% sensitivity. It shares `minimum_image_vectors` with the geometry checker, including skew cells and a nonperiodic vacuum. Mapped formulas that disagree and sensitivity-dependent/unresolved species remain review cases. General generation/conversion routes attach the audit; standalone copied converters without the shared module report that verification is unavailable. The actual legacy layer-record producer (`CO2RR XAS Agent layer converter`, version `2026-10-19.1` in provenance) was not found; this PR does not claim that the root converter owns it.

`tools/layer_population.py` prefers final convolution logs (`fdmnes.out` or named `fdmnes_convolution-*.out`) over obsolete subset recalculations. It checks unique full physical layer indices, Cu elements/heights, actual full-cell input coordinates, source site coverage, E_cut agreement within **0.00051 eV**, grids and final outputs. Source symmetry multiplicities are recorded, not confused with the unique physical layer census. The two recovered CO cases retain 36 represented sites in layers 4 and 5; `uniform2` still fails the grid check. Layer averages precede normalization. Four available layers are not the entire five-layer slab or a bulk experiment.

The application's **Audited adsorbate presence (multilabel)** choice uses the exported audit/splits and shared modules. It does not string-coerce CO_H into one species class. Class order is explicit; clean is all-zero; unseen labels and missing positive training classes are reported. Candidate ranking and thresholds use outer training groups only. General categorical classification/regression remains separate. The UI states primary versus post-hoc scores explicitly; the old advisor is not described as validated by the standalone benchmark. Reliable parent lineage is not available for a separate independent-parent claim; a one-parent holdout is infeasible rather than an extra source of independence.

## Metadata, execution and portal boundaries

The pinned [ISAAC schema](https://github.com/dimosthenisSLAC/isaac-ai-ready-record/blob/main/schema/isaac_record_v1.json) has no multiple-scattering family and makes family optional. The proposed representative correction omits the false semi-empirical bucket and unsupported unknown core-hole value, preserves unresolved physics in notes, distinguishes NERSC/Lawrence Berkeley National Laboratory from author affiliation, and retains unmapped legacy metadata in a sidecar. The local proposal has zero errors against the pinned schema; it is not portal validation/deposition and no original record was changed. Input/version evidence is still required to establish the actual core-hole treatment. Corrections require explicit `apply=True`, preserve measurement arrays, back up bytes and log hashes.

The implemented batch CLI defaults to local checks. External validation/deposition flags and round-trip safeguards are documented in the root README; deposition requires separate authorization. `tools.isaac_roundtrip` tests stable IDs, measurement/label hashes and prediction identity using fixtures only. No portal receipts are claimed.

`tools.execution_evidence` retains 735 surviving stdout paths, deduplicated to 651 scheduler job IDs across 93 canonical folder identifiers. These are not a reconciled attempted-run denominator. Unknown terminal/numerical/record/ML/restart/manual-action fields remain explicit. Set `CO2RR_EXECUTION_EVENT_LOG`, `CO2RR_WORKFLOW_ID` and `CO2RR_CONFIGURATION_ID` for optional prospective local submission receipts; this does not submit anything by itself. New events are hash chained; historical backfill and prospective logs are separate formats. Runtime/cost/recovery rates remain unmeasured.

The matched quartet is prepared under `prepared_jobs/` without POTCAR or submissions. It shares a cell, parent and 2 C + 2 O loading. The denominator is 36 top Cu sites; molecular coverage differs between two CO and one OCCO. Final trajectories are linked by initial-geometry hashes, not similar directory names. Verified matched relaxed states/edges are still needed before coupling contrasts or discrimination can be reported.

## Manuscript build and exact remaining inputs

```bash
python study/code/compile_manuscript.py --output-dir /tmp/co2rr_paper_build
```

This command runs pdfLaTeX twice and renders every page when the engine and PyMuPDF are available. The current environment has no TeX engine; the failed availability check is recorded with zero completed passes. The five figure renders were inspected. No compiled-paper or all-page-inspection claim is made. The file described as an RSC template was not used as a template.

Minimum remaining inputs:

1. The actual manuscript package: `manuscript.tex`, its bibliography/assets, original `provenance/figure_sources.json`, `figure_data/`, figure scripts and five captions; and the named `CuO_Kedge_benchmark_source.pdf` for context.
2. The actual legacy layer-record converter source/version, not an unrelated root converter.
3. For new science: verified final trajectories for the four exact initial-geometry hashes, matched final Cu/C/O spectra plus site/population/numerical settings; raw experimental Cu/Cu2O/CuO traces with source/license and allowed alignment protocol. Existing CuO raw simulated site traces are inventoried locally; they are not an experimental benchmark or a verified broadened ensemble.
4. For execution rates: reconciled workflow manifests/submissions and scheduler accounting (job IDs, Submit/Start/End, states/exit codes, allocations and CPU/node time), plus restart/manual-action history. Surviving logs alone cannot supply the denominator or intervention intent.
5. A genuinely new frozen confirmation set and experiment-grounded perturbation ranges before robustness/calibration/OOD measurements; confirmed author/funding/conflict/DOI/AI-disclosure text; and a TeX engine for compilation/page review.

No entire HPC filesystem is requested. Geometry/coverage performance, matched coupling spectra, reference metrics, execution-rate improvements and remote portal reuse remain unmeasured rather than copied from identity F1.
