# Corrected fixed-top-two evaluation — 23 October 2026

**The historical v3 score 0.7821 came from automatic depth selection and does not update the manuscript's fixed-top-two benchmark.** This study fixes every classifier to the raw population-weighted mean of layers **5+4**, followed by the archived per-spectrum minimum subtraction and unit-area normalization. No surface-only, deeper-layer or depth-channel alternative can be selected.

The author specified **Combined spectrum/CDF/derivatives** as the fixed-feature anchor. It concatenates the smoothed normalized spectrum, CDF, d1 and d2, with the archived smoothing/grid settings. The provisional d1 controls remain secondary. Author choice is recorded in `author_anchor.json`; it was not chosen from outer scores. All candidate scores were frozen before extracting Combined controls. Identical deterministic Combined outer models were saved and checked against those frozen scores.

## Controlled comparisons

Both cohorts use the same dependencies and original 87 outer-fold assignments. The 87 rows are ordered exactly as the archived manifest; the three additions are appended. Old inner-fold memberships are also held identical across 87/90 and across seeds. Added outer-training rows enter inner folds by deterministic geometry-hash assignment. No held-out labels choose splits, features, leaves or thresholds.

Three declared tree seeds (20261020/21/22) vary estimator randomness only. No seed is selected as a winner. Layers and seed replicates are not new independent configurations. Inputs are the same audited 90-configuration snapshot; all original 87 geometries/labels/folds and native outputs remain unchanged.

1. **Fixed Combined, threshold 0.5:** hold the representation, depth, leaf size 1, estimator and decoder fixed to isolate training-data changes.
2. **Fixed Combined, transferred old-87 thresholds:** learn each seed/fold threshold vector using old outer-training data only, then apply it unchanged to that seed/fold's 90-cohort model.
3. **Fixed Combined, retuned thresholds:** keep Combined/leaf1/top-two fixed; repeat threshold selection within each outer-training partition.
4. **Original feature selection:** with depth fixed, select among normalized spectrum, CDF, d1, Combined and DCT24 at leaf1, using inner-training scores/thresholds only.
5. **Compact regularized selection:** select among d1, Combined and DCT8/16/24/32 and minimum leaf sizes 1/2/4 inside training. The union is 20 distinct candidates (18 regularized candidates plus normalized/CDF leaf1 controls), not an architecture search.

Every estimator uses 64 independent per-species Extra Trees, balanced class weights and sqrt feature sampling; single-class training labels use constants. Tree scores are uncalibrated. Fixed feature/leaf ablations, per-species precision/recall/FP/FN, single/mixed recall, missing-positive supports, all inner predictions, ranks, thresholds, exact splits and outer predictions are saved.

## Results

The historical-seed original top-two feature-selection run reproduces the archived **prediction array exactly**, giving micro/macro F1 **0.7444/0.6253**, exact-set **0.6207** on 87 configurations. This verifies the appropriate fixed-depth baseline, rather than the 0.7791 automatic-depth historical primary.

For seed 20261020:

| Protocol | Cohort | Micro F1 | Macro F1 | Exact set |
|---|---:|---:|---:|---:|
| Combined, fixed threshold 0.5 | 87 | 0.7738 | 0.5920 | 0.6667 |
| Combined, fixed threshold 0.5 | 90 | 0.7746 | 0.6050 | 0.6667 |
| Combined, transferred 87 thresholds | 87 | 0.7861 | 0.6516 | 0.6897 |
| Combined, transferred 87 thresholds | 90 | 0.7571 | 0.5962 | 0.6667 |
| Combined, retuned thresholds | 90 | 0.7821 | 0.6335 | 0.6889 |
| Original feature selection, depth fixed | 87 | 0.7444 | 0.6253 | 0.6207 |
| Original feature selection, depth fixed | 90 | 0.7435 | 0.6110 | 0.6889 |
| Compact regularized selection | 87 | 0.7528 | 0.6230 | 0.6552 |
| Compact regularized selection | 90 | 0.7821 | 0.6126 | 0.6667 |

The repeated 0.7821 values are **different protocols with different macro/exact-set results**, not a revalidation of the earlier automatic-depth run. The seed above was declared because it is the archived seed, not selected after comparison.

Across the three seeds, full-cohort mean micro F1 is:

| Protocol | 87 | 90 |
|---|---:|---:|
| Combined, threshold 0.5 | 0.7725 | 0.7539 |
| Combined, transferred 87 thresholds | 0.7795 | 0.7587 |
| Combined, retuned thresholds | 0.7795 | 0.7686 |
| Original feature selection, depth fixed | 0.7604 | 0.7525 |
| Compact regularized selection | 0.7618 | 0.7621 |

These cohort totals have different evaluation composition; `paired_figure_arrays.csv` and `paired_old_configuration_intervals.csv` compare the same original 87. Fixed-prediction bootstrap intervals exclude refitting/selection uncertainty. Some individual conditional comparisons favor or penalize settings; no multiple-comparison adjustment or independent confirmation is claimed. Seed variation is separately reported as refitting variability. It must not be pooled as 261/270 independent observations.

Regularization gives **no consistent overall improvement across seeds and metrics**. On the 90 cohort its mean micro F1 is 0.7621 versus 0.7525 for original feature selection, but mean macro F1 is essentially unchanged (0.6165 versus 0.6166) and mean exact-set accuracy is lower (0.6444 versus 0.6556). One seed improves micro F1 substantially, another decreases it, and the third is unchanged. Exact identity sets vary across seeds for 19/90 configurations under original feature selection and 16/90 under regularized selection. Training-only leaf/feature frequencies are provided rather than promoting an outer-score winner.

## Missed interface 2CO case

The historical miss is confirmed: the original v3 old-only model selected **multiscale depth channels**, with CO score **0.15625** below threshold **0.35**. Corrected Combined/top-two models trained only on the old 87 also miss it for all three seeds, with CO scores **0, 0.0625 and 0.03125**, against thresholds **0.4, 0.425 and 0.5**. The other two added/recovered CO cases are detected. This is not a near-threshold failure.

The query contains two unbonded CO molecules (coverage 2/36 Cu surface sites). At the explicit 2.25 Å contact cutoff one carbon has three Cu contacts (2.029–2.042 Å), and the other one Au contact (2.012 Å). Neither of the two old CO-only configurations at the same loading has this exact contact signature. Contact counts are geometric guides, not electronic bond-order assignments or independent proof of a new parent domain.

Yet nearest-distance ranks are ordinary: about 21st percentile for the top-two normalized spectrum, 15th for d1 and 11th for the failed model's actual multiscale representation, relative to old-training leave-one-out nearest distances. Closest spectral neighbors include non-CO species. Spectral similarity/label overlap and a sparsely sampled contact combination are observable; neither weak Cu-edge perturbation nor unfamiliarity is established as the cause. A compatible clean-parent spectrum remains absent, so adsorbate-induced amplitude cannot be isolated. The independently unit-area-normalized query-neighbor differences are descriptive, **not adsorbate-minus-clean physical contrasts**.

`diagnosis/` includes every training-neighbor distance, same-loading/contact comparisons, initial/relaxed hashes and coordinates, interface-distance and contact-cutoff sensitivity, enlarged local views with contact keys, exact plotted arrays and fixed-top-two old-only diagnostic models.

## Corrected new-only reporting and historical error audit

Only three configurations were added/recovered, all CO-only. The twelve stored layer spectra represent three structures (only six of those layer inputs enter the top-two averages). CHO, OCHO, COH and mixture-positive support did not increase.

For the historical three-case v3 holdout, report **CO precision 1.0, recall 2/3, F1 0.8**, and exact-set 2/3. The other ten labels have no positives, so their recovery was not tested. The saved macro F1 0.0727 is simply 0.8/11 because the absent labels receive zero. It remains in historical provenance, not the new-only headline. Any negative-only false positives remain explicitly counted in new subset reports.

The historical old-87 prediction audit confirms four wrong→correct and five correct→wrong identity sets; two CO-only cases gained false H/OH. All five selected representations changed in v3. Those verified case tables explain the F1/exact-set tradeoff without treating v3 as the corrected fixed-top-two result.

## Artifacts and reproduction

- `experiments/`: declared protocol, all candidate inner/outer arrays, per-seed metrics/selections/supports, and saved evaluation models. Every saved model fixes `depth=top_2`; there is **no new production/deployment winner**.
- `metrics.csv`, `seed_summary.csv`, `feature_leaf_ablation.csv`, `per_species.csv`, `single_mixed_recall.csv`, `new_only_CO_metrics.csv`: complete numerical results.
- `Combined_control_thresholds.csv`, `paired_old_configuration_intervals.csv`, `paired_regularization_comparison.csv`, `seed_identity_stability.csv`: paired controls and uncertainty/stability.
- `historical_v3_prediction_changes.csv`, `historical_depth_selector_changes.csv`, `historical_v3_reporting_correction.json`: verified correction audit.
- `proposed_additions.json`: ranked interface CO, matched CO/CO+H/CO+OH and independent CHO/OCHO environments. These are **proposals only**; none was generated/submitted or credited with improved performance. Lock new independent confirmation cases before inspection.
- `figures/`: PDF/SVG/high-resolution PNG, with seed SD distinguished from confidence intervals and exact saved plot arrays.

From a repository checkout with matching dependencies, the shared agent/UI recipe is `fixed_top_two_v4`, pinned to the audited 87/90 snapshot. CLI:

```sh
python -m tools.audited_training --data-root study/retraining_20261022/refit/dataset --output-dir /new/paired_experiment --recipe fixed_top_two_v4
```

The generic callable computes the core study and author-specified Combined report; the delivered experiment is under `experiments/`, with the report in this directory. To replay/rebuild delivered artifacts:

```sh
python study/code/diagnose_interface_2co.py
python study/code/extend_interface_diagnosis.py
python study/code/holdout_top_two_diagnostic.py
python study/code/report_fixed_top_two.py
python study/code/plot_fixed_top_two.py
python study/code/verify_fixed_top_two.py
```

The original archives and revised-manuscript prose are not overwritten. Available local manuscript copies predate the author's fixed-top-two correction; this versioned report follows the current author instruction and is not presented as a newly compiled manuscript. No HPC jobs, portal writes, synthetic independent positives, clean-reference substitutes or mechanistic claims were introduced.
