# Historical automatic-depth export v3 — 22 October 2026

**Protocol correction:** this run selected depth automatically. Its 0.7821 result does not update the manuscript’s fixed-top-two benchmark. See [EVALUATION_CORRECTION.md](EVALUATION_CORRECTION.md) and the paired fixed-top-two study in `../top_two_20261023`.

The updated local export was audited and refitted through `CO2RRXASAgent → tools.audited_training.run_audited_training(recipe='updated_data_v3')`. The original 87-configuration/348-spectrum benchmark, its predictions/models, reference benchmarks and manuscript package remain unchanged.

## What changed

The versioned dataset contains **90 configurations / 360 layer spectra**. All 87 prior configurations retain their target sets, geometry groups, original fold assignments and byte-identical four-layer convolved outputs. Nineteen retained aliases changed (six COH and thirteen H paths). The COH identities remain O–H bonded; folder spelling does not determine targets.

Two newly completed supplied records are adjacent 2CO at the Au–Cu interface and 0.167 uniform loading. A third CO configuration, `CO/CO_0p11/uniform`, was recovered from existing complete raw Green-function output with no AI-ready record. Its derived record is explicitly marked as recovery, not a newly calculated spectrum. All three additions are CO-only configurations; they add no independent CHO/COH/H-mixture positives.

One `CO/CO_0p11/uniform2` record remains excluded for partial/duplicate final convolution-site evidence. The raw finite-difference sibling has incompatible site grids and is not substituted into the Green cohort. All 90 selected configurations share the same declared Green/SCF/Perdew/radius-6 Å controls and full 8974–9029 eV / 276-point grid.

One H configuration has a 0.00447 Å maximum difference between the root CONTCAR and exact XAS-used geometry. This difference already existed in the verified historical export. Its root geometry, XAS geometry and all four native output byte hashes match that archive exactly. The row is retained as the historical XAS calculation, with a visible lineage warning; no root/XAS equality is asserted and no distance tolerance was loosened. Novel mismatches without this exact historical evidence are rejected. Graphs and targets use the actual XAS geometry. The failed-relaxation guard remains intact; POSCAR cannot replace a missing relaxed geometry.

## Evaluation

The independent binary Extra Trees selector is refitted with the same 37 feature/depth candidates, 64-tree settings, training-label-balanced inner split search, and global/shrunk micro-F1 threshold decoder as the archived primary. Five outer configuration folds preserve all old assignments; only the three additions are assigned deterministically. All selection uses outer-training data only. Counts, paths, graphs and structural metadata do not enter predictors. Native broadening and minimum-subtracted unit-area ML preprocessing are unchanged.

| Evaluation | Configurations | Micro F1 | Macro F1 | Exact set |
|---|---:|---:|---:|---:|
| Frozen primary | 87 | 0.7791 | 0.6383 | 0.6897 |
| Updated refit, all | 90 | 0.7821 | 0.6448 | 0.6778 |
| Updated refit, matched old subset | 87 | 0.7816 | 0.6458 | 0.6782 |
| New/recovered configurations held out from all training/selection | 3 | 0.8000 | Not informative (CO-only) | 0.6667 |

For this subset, CO precision is 1.0, recall 2/3, and F1 0.8. Other species have no positive cases and their recovery was not tested. The legacy all-eleven-label macro 0.0727 is merely 0.8/11 and remains only in saved provenance.

The 90- and 87-configuration totals are not directly comparable. On the same old 87, the micro-F1 change is +0.00254, with conditional paired 95% interval −0.04236 to +0.04897. All three metric intervals include zero and exclude refitting uncertainty. This is not an established overall improvement or independent confirmation. Existing folds have been examined repeatedly.

Mixture recall changes from H 0/8 to 1/8 and CO 5/9 to 7/9; single-species CO recall is 22/24 in the expanded set. The three-configuration holdout correctly returns CO for the uniform 2CO and recovered CO cases but predicts an empty identity set for the new Au–Cu-interface 2CO case. The holdout was not prospectively registered, is tiny, and tests only CO presence. It does not validate sparse intermediates, coupling mechanisms, parent transfer or experimental performance.

Seven models are supplied: five evaluation-fold models, a separate old-only/new-configuration holdout model, and a full-cohort deployment model. The deployment model is trained on all 90 and has **no held-out performance claim**. Tree scores are uncalibrated. Every model replays identically through the shared `predict_records` function.

## Data and reproduction

- `change_summary.json`, `source_changes.json`, `record_comparison.json`: discovery, renames, measured numeric changes and earlier inventory scope.
- `refit/dataset/`: complete four-layer numerical snapshot, fixed class order, groups/folds, target and molecule/lineage audits, recovered record, exact original supplied records/geometry/decks/tables/convolved outputs and convolution logs in lossless `source_evidence.tar`. `source_manifest.json` maps every archive object to its source path, bytes and SHA-256. `snapshot_hashes.json` hashes the dataset files. Site-level numerical files and very large native calculation logs remain in the author's export; their hashes/evidence are retained. POTCAR and executable binaries are not exported.
- `refit/models/`: frozen OOF scores/predictions, all 37-candidate inner scores and rankings, exact inner splits/missing-positive supports, seven models and their hashes, per-species and single/mixed metrics.
- `before_after_predictions.csv`, `paired_old_subset_comparison.csv`, `new_configuration_predictions.csv`, `numerical_control_comparison.csv`: exact comparisons supporting this report.
- `figures/updated_refit_summary.{pdf,svg,png}`: figures generated from saved arrays, with no fitting during plotting.
- `verification/`: syntax/tests/model/snapshot replay evidence. The SHA-specific clean-checkout result is recorded in the draft PR body after push.

Fresh live-data audit and refit, writing a new directory:

```sh
python -m tools.audited_training --data-root /path/to/updated/export --output-dir /new/refit --recipe updated_data_v3
```

Portable refit from the uploaded snapshot, without the original simulation tree:

```sh
python -m tools.audited_training --data-root study/retraining_20261022/refit/dataset --output-dir /new/snapshot_refit --recipe updated_data_v3
python study/code/verify_updated_data_refit.py
python study/code/report_updated_data_refit.py
python study/code/plot_updated_data_refit.py
```

Exact dependency/code fingerprints are in `refit/models/protocol.json`. No new HPC jobs or portal writes were made. The earlier manuscript continues to describe the archived 87-configuration study; this versioned refit is reported separately rather than silently replacing its primary scores.
