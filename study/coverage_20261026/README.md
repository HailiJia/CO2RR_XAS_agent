# New record audit and CO coverage experiment — 26 October 2026

126 records were converted from 135 Green folders; 9 lack a usable canonical spectrum. Complete four-layer cases use the author's `convert_layers_to_isaac.sh`, which calls `tools/convert_simulation_xas_run_to_isaac.py`. Other available layer sets use that generic converter plus an explicit canonical-series merge. No layers or spectra are synthesized. A Mac Bash empty-array/`set -u` failure was repaired. The author's revised ISAAC converter is retained, including unreported temperature metadata instead of an invented 0 K.

Thirty-six previously missing local record files were written; existing author records were preserved. The canonical audit versions are in this study.

All original records/raw exports and the archived 87/90 benchmarks are preserved. `records/` holds the **new version**; `converter_original.json.gz` freezes each pre-audit converted record. `record_geometry_audit.json/csv` compares source labels, converter labels, full molecular graphs, root CONTCAR, and every actual FDMNES Crystal input, with one-based atom mapping, hashes, exclusions and ±10% molecular-cutoff sensitivity. `molecule_audit.json` gives measured C–H/O–H distances, bonds and metal contacts. Filenames do not define chemical labels.

## Audit and inclusion

There are 99 usable Cu–Au configurations and 17 Cu100 input-geometry configurations. Cu100 CONTCAR files are absent, but the author located the **original ASE snapshots in `Cu-Ox_updated_structures/100`**. Each of the 17 calculated cases matches its named original snapshot through element-aware coordinate assignment. Stored free-atom forces and constraints are reported; production relaxation stopping/termination evidence is not supplied. Geometry labels are checked against both original snapshots and actual calculation decks. Five Cu100 carbonate/water-containing cases are preserved as records/graphs but excluded from the present 11-species targets because water/CO3 are outside the archived classifier target set. An explicit graph extension verifies two O–H bonds for water and three C–O bonds for CO3; it does not infer formal charge. They are not silently relabeled as CO/OH-only.

Existing supplied labels agree with the successfully traced, stable XAS-input graphs. Unresolved root/XAS mismatches remain excluded; agreement is not asserted there. The historical H coordinate discrepancy is retained only when the exact old root/copy/output hashes match archived lineage; this is not a new geometry match or a widened tolerance. Newly usable aliases and any old exclusions are listed in `comparison_with_90.csv`.

The final-layer population audit retains the strict existing Cu–Au verifier. Cu100 checks all four physical Cu sites per layer, actual input order, explicit convolution source order or complete aggregate log, and raw-source E_cut values within 0.00051 eV. A sorted-index assumption would incorrectly reject one reordered calculation. Raw source files remain author-local; their hashes and recorded log mapping are preserved. Inputs, canonical convolved spectra, geometry copies, absorber tables and available separate convolution logs are captured losslessly in the source archive. Verbose raw/runtime logs and raw site arrays remain local with full hash manifests. Proprietary potentials/binaries are excluded.

## Coverage definition and protocol

The coverage model is conditional on a known **CO-only** configuration. Its target counts intact disconnected CO molecules using the bond graph, not carbon atoms, nominal names or a class count. There are **25 independent Cu–Au CO-only configurations**; molecules and layers do not increase sample size. Cu–Au exposes 36 Cu and 32 Au sites: the primary target is N(CO)/68 top-surface metal sites. The legacy Cu-only loading N(CO)/36 and molecules/nm² are saved separately. Each element's highest physical layer defines its exposed sites, so a height threshold anchored to the higher Au terrace cannot discard most Cu sites. Cu100 has four top-surface Cu sites. `CO_coverage_support.csv` makes all target levels explicit.

Only spectral arrays enter predictors. The population-weighted native top-two mean uses Cu–Au layers 5+4 and Cu100 layers 4+3. Archived minimum-subtracted unit-area normalization, smoothing and derivatives are unchanged. Native lifetime broadening is preserved; no extra broadening, alignment or stress tuning is applied. A fixed common supported 8974.0–9029.0 eV grid (276 points) is linearly interpolated without extrapolation.

The compact declared set is Combined spectrum/CDF/d1/d2 and DCT8/16/24, each with Extra Trees leaf2/4 or standardized Ridge alpha10/100. Ridge scaling is fitted only on each training partition. Three inner grouped folds minimize coverage MAE. Five outer configuration folds, separate leave-coverage-level-out and contact-motif-group-held-out evaluations are saved, with training-mean/median baselines and a fixed Combined/leaf2 control. The contact motif is an audit/splitting label, never a spectral predictor. These evaluations are exploratory reused-configuration tests, not independent prospective confirmation or proven parent transfer.

## Measured coverage results

| Protocol | Model | N | MAE (CO / all surface sites) | RMSE | R² |
|---|---|---:|---:|---:|---:|
| coverage_configuration_folds | nested | 25 | 0.0138 | 0.0212 | 0.943 |
| coverage_configuration_folds | fixed_Combined_leaf2 | 25 | 0.0165 | 0.0312 | 0.876 |
| coverage_configuration_folds | training_mean | 25 | 0.0658 | 0.0930 | -0.097 |
| coverage_configuration_folds | training_median | 25 | 0.0609 | 0.0980 | -0.218 |
| coverage_level_held_out | nested | 25 | 0.0143 | 0.0207 | 0.946 |
| coverage_level_held_out | fixed_Combined_leaf2 | 25 | 0.0256 | 0.0392 | 0.806 |
| coverage_level_held_out | training_mean | 25 | 0.0753 | 0.0995 | -0.254 |
| coverage_level_held_out | training_median | 25 | 0.0735 | 0.1029 | -0.341 |
| contact_motif_held_out | nested | 25 | 0.0147 | 0.0250 | 0.921 |
| contact_motif_held_out | fixed_Combined_leaf2 | 25 | 0.0161 | 0.0318 | 0.872 |
| contact_motif_held_out | training_mean | 25 | 0.0740 | 0.0969 | -0.190 |
| contact_motif_held_out | training_median | 25 | 0.0753 | 0.1032 | -0.349 |
| Cu100_input_geometry_external | CuAu_CO_only_nested_recipe | 4 | 0.2942 | 0.3335 | -0.424 |
| CuAu_CO_mixtures_diagnostic | CuAu_CO_only_nested_recipe | 12 | 0.0280 | 0.0296 | -3.851 |

The configuration-fold MAE is **0.0138** versus the training-mean baseline **0.0658**. The conditional paired MAE-difference interval is -0.0762 to -0.0320; it resamples configurations with predictions fixed and excludes refitting/selection uncertainty. Full held-out arrays, inner selection ranks, group memberships, model files and target-range flags are included. No best outer-score feature is promoted. Negative/out-of-range Ridge predictions, if any, are retained rather than clipped.

Cu100 is a **separate zero-shot domain diagnostic**, with no Cu100 fitting, threshold choice or calibration. Its four CO cases differ in facet, composition, cell size, layer populations and loading range. Three of the four exceed the Cu–Au all-site coverage range. The transfer result is substantially weaker than within-Cu–Au evaluation and does not validate a universal coverage estimator. Molecules/nm² and explicit range flags are saved so normalization conventions are visible. CO-containing mixtures are a separate diagnostic: a CO-only model is not validated for coadsorbate interference.

## Identity and local geometry

The new grouped, fixed-top-two Combined identity evaluation has N=99, micro/macro F1 **0.7500/0.5711**, exact set **0.6263**. It uses training-only thresholds and newly declared grouped folds; it is **not a same-fold update** of the archived primary/fixed-top-two scores. All 11 per-label supports and single/mixed recall are saved. Identity estimators use the archived seed 20261020; coverage/contact regressors use 20261026. Cu100 identity transfer has N=12, micro F1 0.4211; report its supported-label table rather than treating absent-label macro zeros as tests of rare species. Probabilities are uncalibrated.

The CO local-geometry model predicts **configuration-mean Cu and Au contact counts within 2.25 Å**, not a unique site for every molecule or electronic bond order. Per-molecule targets and 2.15/2.35 Å sensitivity are saved. The measured hollow-series molecule support is {'Cu3Au0': 25, 'Cu2Au1': 2, 'Cu1Au2': 0, 'Cu0Au3': 0}. In particular, **CuAu2 and Au3 CO environments have zero support**. The current data cannot validate a continuous Cu3→Au3 inference claim. Multiple molecules in one spectrum do not count as independent training examples. Contact errors and mean baselines are in `contact_support.json`.

## Best use of Cu100 and next data

Keep Cu100 as a facet/composition transfer panel and a controlled CO loading series, rather than pooling it into the frozen Cu–Au folds. Original ASE snapshots are now matched; recover the production relaxation stopping/termination record before calling them verified converged calculations. Add independent Cu100 configurations at matched coverages/motifs and independent Cu–Au interface CO configurations, then lock a new prospective transfer set before inspection. The four existing CO loadings alone are insufficient to validate within-facet coverage training. Water/CO3 graphs are now explicitly labeled, but adding them to an identity model requires an extended target schema and sufficient independent support before retraining. Au-rich CO motifs and matched CO/CO+H/CO+OH controls address gaps that more Cu-only examples cannot fill.

A clean Cu–Au top-two record is newly available (`plain`, layers 4+5). `parent_control_inventory.json` finds 34 exact initial-metal matches to its saved relaxed coordinates, but coordinate families are not a verified count of independent ancestral parents. This coverage experiment does not perform a physical clean subtraction or equate independently normalized spectra with adsorbate-induced amplitude. The 66 exact initial-metal coordinate families are an inventory, not 66 independent-parent validations.

## Reproduce and inspect

```sh
python -m tools.coverage_study convert --data-root /path/to/export --output-dir /new/version
python -m tools.coverage_study audit --data-root /path/to/export --output-dir /new/version
python -m tools.coverage_inference --output-dir /new/version
```

The same audit→fit operation is exposed through `run_audited_training(..., recipe='coverage_v1')`, the agent parser/planner and UI. `tools.coverage_inference.predict` reloads a coverage model and requires verified physical populations and full trained energy support. No production scheduler, HPC or portal write is involved. No revised manuscript/SI is claimed in this versioned report; archived manuscript and predictions remain untouched. Figures are generated from saved arrays by `study/code/report_coverage.py` and exported as PDF/SVG/350-dpi PNG.

Method implementation references: [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [training-only mean/median baselines](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). Runtime versions are pinned in `protocol.json`; online documentation version is not substituted for the installed environment.
