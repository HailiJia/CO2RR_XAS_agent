# Current count-schema audit and ML retraining

Open **adsorbate_identity_depth_updated.ipynb**. It is the canonical executed notebook for the current data/schema and contains all scientific code: structural connectivity and optional record correction, full inventory, completed-record/spectrum audits, folds, features, model fitting, nested selection, figure generation and bullet-point interpretation. Earlier notebooks in the parent directory are historical and use older targets/schemas.

## Deliverables

- `ANALYSIS.md`: concise bullet points: features/models tried, provisional best, limitations, mixture errors and improvement plan.
- `DETAILED_ANALYSIS.md`: longer methods/results notes.
- `figures/01_all_structures_by_adsorbate.{png,pdf}`: all unique structures, including unfinished cases; exclusive mixture combinations plus nonexclusive species presence.
- `figures/02_training_testing_structure_counts.{png,pdf}`: inventory eligibility, per-fold train/test structure counts and fold-1 species counts.
- `figures/03_layer_sensitivity.{png,pdf}`: all four individual layers and top-two/top-three/four-layer averages, identical folds.
- `figures/04_adsorbate_identity_performance.{png,pdf}`: one held-out F1 bar per adsorbate for the provisional best fixed configuration, labeled with positive structure counts; rare species flagged. Full precision/recall and mixture breakdowns remain in CSV.
- `figures/05_feature_model_comparison.{png,pdf}`: fair model/feature heatmaps at top-layer and top-two-average depths.
- `requested_figures.zip`: all five requested PNG/PDF pairs plus the supplemental figures.
- Three supplementary PNG/PDF figures show an energy-resolved identity contrast heatmap, separate species mean/variability spectra, and automatic-selector binary confusion matrices.

## Current audited cohort

There are 172 source locations (including the standalone CuO reference), 155 unique physical structures, and 102 completed-record files. Removing 14 duplicate records and excluding one unsuitable paired-layer record leaves **87 independent configurations / 348 spectra**. Source locations with identical atomic coordinates are counted once; all aliases remain in the CSV audit. Calculational copies of a structure inside XAS/layer folders are not new source structures. Empty CONTCAR files fall back to POSCAR for inventory. Folder names alone do not define a structure. All unfinished configurations with a readable structural source are counted. CuO is included in the inventory, excluded from identity ML.

All 85 configurations from the preceding analysis retain their folds. Two CO cases were recovered after checking the final convolution logs: layers 4 and 5 use 36 sites despite main decks and `_bav.txt` logs describing three-site subset recalculations. Every final convolution site is matched to its raw spectrum by E_cut (within 0.00051 eV rounding), exact layer atom coverage and compatible grids. Only `CO/CO_0p11/uniform2` remains excluded for an incompatible layer-5 grid. See `record_audit.csv`, `layer_population_audit.csv` and `two_CO_population_audit.json`. The earlier exclusion/QC annotation was incorrect and has been superseded; spectra were never changed.

Seven regenerated count-schema records mislabeled molecular components (for example CO and H both called CHO). Identity/count fields and site species labels were corrected after formula/connectivity checks. Measurements are unchanged. `record_corrections.json` logs all changes and hashes; the pre-edit files are in `records_before_count_correction_*.tar.gz`. The notebook's correction function defaults to read-only on rerun and is self-contained. Original directory/material names remain provenance. Site geometry is associated with the correct molecular identity; site-type descriptors are not reclassified or predicted.

Source CONTCAR and calculation-input CONTCAR can have small coordinate differences. Training identities/counts must agree, and spectra/decks are verified against their actual simulation input. Such source-coordinate differences are flagged in the record audit. POSCAR versus relaxed identities/counts are audited separately; relaxation can dissociate an intended intermediate.

## Results and interpretation

The provisional best observed fixed configuration is **extra_trees + derivative + layer_5**, with micro/macro F1 **0.798/0.660**. It was selected after comparing CV entries, so this ranking is exploratory. The predeclared automatic independent-model selector scored **0.779/0.638**. Within each fixed depth, nested feature selection gave top-layer **0.786/0.643**, top-two **0.744/0.625**, and all-four **0.659/0.508**. No independent confirmation claim is made. The cohort correction changes the observed rankings; the best workflow is not stable enough to claim a universal winner.

Use `provisional_best_model.json`, `provisional_best_per_species.csv`, `provisional_best_errors.csv` and `oof_provisional_best.npz` for the provisional fixed winner. `nested_performance.csv`, `per_species.csv`, `confusion_matrices.csv` and `oof_<selector>_<objective>.npz` describe the declared selectors. The distinction is intentional; do not confuse a winner chosen after comparison with an unbiased automatic-selection result.

## Portable replay and scientific rerun

The reviewed executed notebook and OOF arrays in this directory are immutable benchmark artifacts. The refactored `analysis.py`, builder, validator and two-CO audit use shared repository modules and configurable paths. A reproduced portable notebook is saved separately; historical source cells in the reviewed notebook document the implementation at the time of the benchmark.

See [the study reproduction instructions](../../study/README.md) for pinned dependencies, figure-only replay, exact-cohort export, relocated audit/fitting, saved-model inference, and the difference between primary nested results and exploratory winners. No local absolute backup path or old fit cache is needed for archived replay or export validation.

```bash
python -m tools.study_cli replay --output-dir /tmp/co2rr_replay
python CO2RR_ML/updated_dataset_20261020/validate_outputs.py \
  --output-dir /tmp/co2rr_validation
python CO2RR_ML/updated_dataset_20261020/analysis.py \
  --data-root /path/to/CO2RR --output-dir /path/to/new_run \
  --split-manifest CO2RR_ML/updated_dataset_20261020/cv_manifest.csv --audit-only
```

The builder accepts the same path options, plus `--build-only` or `--audit-only`. Every rerun must use a new output directory. The primary archived result remains micro/macro F1 0.7790697674/0.6382765883. Metadata corrections are opt-in, reversible and hash logged; measurement arrays must not change. The actual legacy layer-record producer was not found and is not conflated with the repository's general converter.
