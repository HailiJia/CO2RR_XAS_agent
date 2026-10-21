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

## Setup and rerun

Use Python 3.11+ and install:

```sh
python -m pip install numpy pandas scipy scikit-learn matplotlib nbformat ipykernel ipython
```

Exact tested versions are in `environment.json`. This session used `/Users/hjia/miniconda3/envs/EELS/bin/python`.

Open the notebook in Jupyter from this folder and Run All. Set DATA and OUT in its first cell if relocating files. Keep `prior_fold_assignments.csv` and `record_corrections.json` beside it. The former is data (configuration/fold pairs), not an external code dependency. Matching caches save recomputation. **Delete cache/ after modifying feature/model implementations**, so edited code cannot accidentally use old fits. Input record hashes and cohort/parameter settings enter the run tag.

The helper below executes all notebook cells and captures rich figures without Jupyter sockets, which are blocked in this sandbox:

```sh
MPLCONFIGDIR=/tmp/co2rr_mpl IPYTHONDIR=/tmp/co2rr_ipython \
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
/Users/hjia/miniconda3/envs/EELS/bin/python build_and_execute.py
```

`analysis.py` is the source embedded in the notebook. `build_and_execute.py --build-only` regenerates an unexecuted notebook. `validate_outputs.py` checks inventory totals, record backups/unchanged measurements, frozen folds, cached train/test isolation, reported metrics and figure/notebook completeness.

The new count and geometry fields are used for auditing only. Predictors come exclusively from spectra. Targets are multilabel species presence; neither coverage/count nor geometry prediction is performed.

## GitHub copy and local data requirements

This copy includes the executed notebook, all current result tables, held-out predictions, PNG/PDF figures and reversible record-correction backups. Fit caches, temporary logs, OS metadata and historical analysis snapshots are omitted. The full POSCAR/CONTCAR, ISAAC records, FDMNES decks and site spectra used by the audit remain in the local simulation dataset. The repository structure-input examples alone are insufficient to rerun the completed-spectrum audit.

To rerun elsewhere, set `DATA` in the first notebook cell to the full dataset root and set `OUT` to this analysis directory. The shipped notebook records the original local paths. Source-script execution uses the same settings near the top of `analysis.py`; `audit_two_CO_population_records.py` also has a dataset-root constant. `validate_outputs.py` currently assumes the original dataset/analysis directory relationship and uses fit caches when present. On a relocated copy, update its `DATA` assignment and the backup paths in the provenance JSON files before running it. The recorded metrics and figures can be inspected without any raw data or caches.

For the tested dependency versions, see `environment.json`. This workflow uses NumPy 2 (`numpy.trapezoid`); a suitable starting environment is:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install 'numpy>=2' pandas scipy scikit-learn matplotlib nbformat ipykernel ipython jupyterlab
jupyter lab adsorbate_identity_depth_updated.ipynb
```

Choose **Run All** after setting the paths. Fitting regenerates the omitted caches. Setting up the application from the repository root and running this standalone analysis are separate entry points.
