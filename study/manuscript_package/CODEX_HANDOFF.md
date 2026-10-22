# Complete the CO2RR XAS analysis and reconcile the implementation

Work in `https://github.com/HailiJia/CO2RR_XAS_agent`. Begin by reading repository instructions and recording the current commit. This manuscript package was prepared against **71bb0a20b369faba842b413a09544dd66c127a78**, reviewed on 21 October 2026. Compare newer commits before changing anything. Use an isolated branch and produce a reviewable PR; do not merge, publish records, or submit expensive HPC jobs without the user's existing authorization for that action.

The goal is a reproducible Digital Discovery study connecting structure generation, relaxation, XAS, ISAAC records, and inference. Read `manuscript.tex`, `provenance/figure_sources.json`, and the five figure captions in the supplied package. Treat the draft's unmeasured tasks as a plan, not as existing results. Preserve its completed identity/depth results unless an explicitly documented correction changes them.

## Authoritative current scientific results

Use `CO2RR_ML/updated_dataset_20261020/`, especially its executed notebook, `analysis.py`, `README.md`, `record_audit.csv`, `layer_population_audit.csv`, `cv_manifest.csv`, `run_config.json`, and saved OOF predictions. Historical exclusions and uncorrected record identities are not authoritative.

- 172 source locations; 155 unique structures including unfinished structures and CuO.
- 102 record files → remove 14 duplicates and one grid-incompatible case → **87 independent configurations / 348 spectra**.
- Layer 5 is the physical top; available layers are 2–5. The four-layer representation is **not** the entire five-layer slab or a bulk experiment.
- Primary independent-species selector: micro/macro F1 **0.7790697674 / 0.6382765883**, exact set accuracy **0.6896551724**.
- Fixed-depth nested top layer: **0.7861271676 / 0.6428295066**; top two **0.7444444444 / 0.6253068197**; four available layers **0.6593406593 / 0.5082757013**.
- Exploratory best fixed entry: `extra_trees`, `derivative`, `layer_5`: **0.7978142077 / 0.6597402597**, exact set accuracy **0.7011494253**. It was selected after comparing outer scores. Do not promote it to independent confirmation.
- Nine mixed configurations: CO+H (5), CHO+CO+H (3), CO+OH (1). There are 78 single-species configurations. Species supports overlap and sum to 99.
- Rare supports: CO2=3, COCHO=1, OCCOH=4. No eligible clean controls or intact OCCO positives.
- Seven label/count corrections and two population-eligibility corrections changed metadata, **not measurement arrays**.
- The two recovered `CO/CO_0p11/{interface_biased,interior_biased}` cases have final 36-site convolutions. Do not exclude them on the basis of three-site subset main decks or intermediate logs. `CO/CO_0p11/uniform2` remains excluded for an incompatible layer-5 grid.

## 1. Make the complete cohort accessible and portable

The GitHub copy explicitly omits the full local simulation dataset. Its original path is `/Users/hjia/Downloads/CO2RR`. If that path is accessible in your environment, inventory it read-only first. If it is not accessible, finish all work that can be done from the repository and the supplied package, then report the exact minimum missing artifacts; do not fabricate data or ask for the entire HPC filesystem indiscriminately.

Create an export command that packages the exact 87-configuration cohort: final coordinates used by XAS; audited ISAAC records; absorber tables; main and final convolution decks/logs; convolved arrays; site data required to reproduce population verification; frozen fold assignments; corrected target provenance; software settings and versions. Preserve aliases and source/initial/final hashes. Exclude credentials, licensed executables and restricted potential files; use potential identifiers/checksums where appropriate. Keep source files immutable by default. Record hashes of all exported bytes and a dataset manifest.

Remove machine-specific paths in `analysis.py`, `build_and_execute.py`, `validate_outputs.py`, and `audit_two_CO_population_records.py`. Add `--data-root`, `--output-dir`, `--split-manifest`, and `--audit-only` or equivalent configuration. Validation must work on a relocated export, without old fit caches or local backup paths. Include source-code/version and dependency fingerprints in fit-cache keys; the current run tag includes parameters and input manifest but does not protect edits to feature/model implementation by itself. Verify cached train/test isolation.

Acceptance: a clean environment can replay exported predictions and rebuild derived tables/figures; with the complete export it can rerun the audit and fitting. Publish nothing yet. Clearly separate figure-only reproduction from full scientific reproduction.

## 2. Reconcile chemical labeling and population verification upstream

Inspect `tools/dataset_labels.py`, `tools/contact_checks.py`, `tools/isaac_record_builder.py`, `tools/convert_simulation_xas_run_to_isaac.py`, `tools/result_parser.py`, generation/job conversion routes, and the legacy layer-record converter used on the local dataset. Locate the actual producer of the seven wrong component labels. Do not assume the root converter owns a layer-specific converter absent from the repo.

Implement one reusable, read-only component-label audit shared by record generation and the standalone analysis. Use the actual calculation-input coordinates, consistent atom indices, formulas, connectivity, and requested-versus-observed provenance. Distinguish CO+H from CHO, COH from CHO, OCHO from COOH, coupled OCCO from two CO, and COCHO from OCCOH. Use molecule mappings when available and a disclosed, sensitivity-tested fallback for legacy data. Preserve ambiguous cases for review. Handle skew cells correctly in plane and keep the vacuum direction nonperiodic; the standalone `components()` currently rounds fractional displacements in all three dimensions and is not the same algorithm as the general geometry checker. Test boundaries rather than silently changing old labels.

Extract final-layer population verification into a reusable utility. Prefer the **actual final convolution population** over obsolete subset input decks, and retain evidence linking every represented absorber/site file to that output. Test full-layer, valid subset recalculation + full final convolution, truly partial population, duplicate sites, wrong elements/heights, unmatched E_cut, and incompatible grids. Any tolerance must be documented; preserve the existing approximately 0.00051 eV matching criterion unless evidence supports changing it. Keep population weighting before per-spectrum normalization.

Metadata corrections must be opt-in, reversible, and hash-logged. Do not modify spectral arrays when fixing identities, populations, facility names, or schema fields. Resolve the representative record's `semi_empirical` method-family bucket and `not_reported` core-hole entry with the current ISAAC schema rather than inventing a scientific method label. Distinguish NERSC facility organization from the author's affiliation.

Acceptance: all seven corrected examples reproduce their current component labels and counts; the two recovered CO cases remain valid; the grid-incompatible record remains excluded; any scientifically justified changed cohort produces a transparent before/after report and new version, not an overwrite of the old benchmark.

## 3. Integrate the scientific multilabel recipe with the application

Inspect `tools/xas_ml_utils.py`, `tools/xas_record_utils.py`, `tools/ml_splits.py`, the ML web pages, and `tools/ml_xas_workflow.py`. The current general classification path coerces targets to strings and uses categorical balanced accuracy; it is not the 11-output multilabel workflow in `analysis.py`. A string such as `CO_H` must not become a single mutually exclusive class in a species-presence task.

Factor audited cohort loading, population-aware depth aggregation, feature recipes, explicit species ordering, binary multi-output fitting, threshold selection, and saved preprocessing into shared importable modules. Retain general categorical classification/regression as separate task choices. For multilabel inference support `adsorbate_counts` and explicit identity sets, with verified legacy parsing. Do not feed labels, geometry, counts, filenames, or status fields into spectral predictors.

Preserve a fixed configuration-level outer test protocol, with identical groups linked by current geometry hash. Fit all learned preprocessing and all selection on training data only. Grouped cross-validation must keep layer/site/backend variants of the same physical configuration together. Parent holdouts are a separate stricter question and may be infeasible if there is only one parent; report that rather than inventing independence. Missing positive training classes must be reported.

Export the energy grid, processing recipe, populations, class order, scaler, estimator(s), thresholds, training/split/data fingerprints, dependency versions, and prediction function. Reject incompatible grids instead of extrapolating silently. Provide a replay check from raw saved records to the same multilabel predictions. Maintain the primary nested selector versus post-hoc fixed-winner distinction in the UI and report.

Acceptance: replay the supplied OOF artifacts and metrics first; then demonstrate saved-model prediction identity on a held-out subset after export/import. Add meaningful tests for mixtures, species ordering, all-zero clean labels, unseen labels, no-positive training folds, and group leakage. Ensure a modified outer test set cannot alter advisor ranking. Do not describe the old app's advisor as validated by the standalone benchmark until this integration passes.

## 4. Finish scientifically justified analyses without inventing new results

1. Recompute all reported metrics directly from saved OOF truth/prediction arrays and reconcile per-species CSVs, exact match, confusion matrices, and pure/mixed recall. Supply training-only constant and most-common-set baselines with the same outer folds. State repeated use of this cohort and reserve genuinely new configurations for confirmation.
2. Preserve the full planned structural matrix separately from completion: pure Cu(111)/Au(111), lateral stripe ratios 1:1, 2:1, 1:2, 3:1, and 1:3 with explicit row counts; interface/near-interface/terrace sites; coverage/distribution; coadsorbates; C1/C2 species. Reconcile this prior plan against the actual inventory; flag any requested cell absent from source rather than claiming it exists.
3. Generate a missing-calculation priority table. Highest priorities are clean controls; independent CO2, COCHO, OCCOH positives; matched CO/CO+H/CO+OH states; and verified intact OCCO/2CO pairs. Count independent geometries, not repeated layer/site spectra. Do not choose exact new sample counts solely to inflate apparent dataset size; explain balance and holdout feasibility.
4. If verified geometry/coverage targets and enough independent groups exist, evaluate them separately. Otherwise export their support and missing-label audit, and leave performance unmeasured. Do not copy classification F1 into geometry or coverage results.
5. Once a frozen confirmation set exists, test energy shifts, noise, and broadening perturbations with predetermined ranges grounded in the intended experiment. Record both perturbation protocol and whether models were trained with augmentation. Report calibration or out-of-distribution behavior only if actually measured. Bootstrap intervals from existing OOF predictions are conditional, not retraining confidence intervals.

## 5. Complete the matched 2CO/OCCO test and reference benchmarks

Use the actual repository input quartet:

```
data/CuAu_dataset/2CO_adjacent/2CO_adjacent_0p056_interface/POSCAR
data/CuAu_dataset/2CO_adjacent/2CO_adjacent_0p056_interior/POSCAR
data/CuAu_dataset/OCCO/OCCO_0p056_interface/POSCAR
data/CuAu_dataset/OCCO/OCCO_0p056_interior/POSCAR
```

These are input structures, with minimum C–C distances approximately 2.52, 2.46, 1.45, and 1.45 Å. Verify shared cell/parent, 2 C + 2 O loading, actual coverage denominator, matched numerical settings and final molecular connectivity. Inspect relaxed results before generating new work. Record OCCO dissociation as a chemical result; do not relabel dissociated states as coupled positives. Prepare missing jobs and a job manifest if running is not already authorized. After completion, compute matched Cu, C and O edge contrasts, including layer and population effects, and the interface-minus-terrace difference in coupling contrast. Only then evaluate coupling-state discrimination using independent matched groups. No activation-barrier or selectivity claim follows from spectral contrast alone.

Consolidate Cu/Cu2O/CuO trace files and experimental provenance, beginning with the supplied `CuO_Kedge_benchmark_source.pdf` for visual context. That image cannot support exact numerical metrics. For every numeric comparison record source/license, backend/mode/version, structure, core hole, radius or cell/band convergence, broadening, energy calibration, fitting interval, and permitted shift/scale/offset. Report normalized RMSE, Pearson r and edge/peak displacement under a fixed, disclosed protocol, with unshifted and aligned results distinguished. Do not digitize a plot and present its metrics as raw-data precision. Verify actual C/O absorber support for each backend with generated inputs and output parsers before describing a universal edge matrix.

## 6. Quantify execution and portal reuse

Create an execution-event log that captures workflow/configuration IDs, attempt IDs, submissions, start/end times, resource use, terminal states, output validators, restart reason/action and manual intervention. Separate scheduler success, numerical output validity, record creation, and ML eligibility. Compute first-pass success, recovery success, retries/cost, and intervention rates only against a reconciled attempted-run denominator. Backfill only events supported by surviving logs.

Check the README command `batch_validate_upload_isaac_records.py`: that file is absent from this snapshot. Either provide and test an implemented batch CLI consistent with the portal client or correct the documentation. Also repair stale repository-layout and patch-script references after checking which entry points remain active. Do not consolidate parallel generator/orchestrator paths without regression evidence that both routes preserve relaxed geometry and labels.

Prepare an opt-in ISAAC round-trip test: validate a small representative record set, upload only when authorized, retrieve stable IDs, compare measurement and label hashes, and replay saved predictions. An upload client alone is not evidence of successful deposition or independent reuse. Use local fixtures to validate logic without external writes.

## 7. Rebuild the manuscript package

Preserve the five-outline figure roles:

1. Framework architecture and scope of autonomy.
2. Interfaces, controls, full planned inventory versus completion.
3. Numerical reference comparisons when available, complementary edges, real layer spectra.
4. Primary grouped inference, per-species support, layer dependence, exploratory model comparison.
5. Matched relaxed 2CO/OCCO structures and measured spectral contrasts, once available.

The supplied `code/make_figures.py` rebuilds the present figures from `figure_data/`, and `code/prepare_figure_data.py` documents their extraction. Keep PDF/SVG/PNG and exact plotted data. Replace pending panels only after real results exist. Use explicit status labels where unfinished. Keep the primary numerical result distinct from exploratory winners and the general application's advisor.

Update the three manuscript tables: inventory/labels/coverage; backend/edge capability and reference metrics; workflow outputs, retries and interventions. Replace bracketed author, conflict, funding, raw-data-DOI and AI-disclosure fields only with confirmed information. Retrieve the correct RSC template if desired: the file named `Royal_Society_of_Chemistry_article_template.pdf` in this session actually contains the two-page outline, so it must not be treated as a journal template.

Run the applicable existing focused tests after implementation, including geometry, grouped ML, labels, peak descriptors and workflow integration. Record current failures rather than quoting an earlier PR's test counts as new validation. Add only tests that cover new scientific or integration behavior. Verify figure scripts, compare metrics with OOF arrays, compile the paper twice, inspect all rendered pages, and check every caption against its data.

Deliver a concise change report, reviewable code changes, reproducible environment/CLI instructions, updated analysis/figures/manuscript, and an evidence table showing completed, implementation-tested, planned, blocked, and unavailable items. If raw data or HPC access is missing, complete the code/export/replay/figure work that is possible and report exactly what remains blocked. Never invent result values, portal receipts, runtime savings, or autonomous-execution rates.


**Superseding task:** use `CO2RR_Codex_PR18_Followup.md` and the latest full manuscript package. The earlier instructions above are historical context.
