# Full-cohort model adjustments: no difficult-case trimming

**Every comparison evaluates ALL 141 supported configurations in the unchanged dataset (135 near-geometry groups). No test record is removed, no abstention filter is applied, and no new spectra, labels, depth choices or energy windows are introduced.** The earlier 130-query comparison was an additional training-size control, not the full dataset result.

The existing full 141 common-threshold baseline is **0.7992 micro F1**, not the 0.7904 previously emphasized for the130-query comparison. This distinction matters. All seeds are reported, without choosing the best.

## Training includes every permitted view

The previous guarded baseline already tested all 141 records but used one canonical training view per near group. This new controlled arm instead retains **every training record outside the held-out geometry groups**, weights each group to total base weight1, and balances binary class masses inside training. Inner validation also includes every permitted view and blocks related geometries. Holding out test groups is required for evaluation; it is not trimming based on difficulty. No metadata/count/site/coverage predictor is introduced.

After evaluation, three separately saved final models are fitted to **all 141 records together**. Their hyperparameters and common thresholds are selected using grouped training-only validation with the same locked candidate set; no independent test score is claimed for those full-data fits.

## Declared model comparison

Fixed signed-shape top-two representation. Seven candidates: Combined or DCT16 × minimum leaf size 1/2/4 with 64 trees, plus Combined/leaf 1/256 trees. Eleven independent classifiers; single-class training partitions use constants. Training-only threshold choices 0.3/0.4/0.5/0.6 have a restraint toward 0.5. The selected recipe maximizes grouped inner quality; outer scores never choose recipes, seeds or thresholds. All three predeclared seeds 20261020/21/22 are run. The original baseline stays unchanged.

**Same141 held-out records; equal total weight per geometry group; mean across all three seeds:**

| Model protocol | Threshold | Micro F1 | Macro F1 | Exact-set accuracy | Queries |
|---|---|---:|---:|---:|---:|
|Prior canonical-view64-tree control|fixed0p5|0.7631|0.5956|0.6605|141|
|Prior canonical-view64-tree control|common|0.7992|0.6630|0.7037|141|
|All-view weighted64-tree control|fixed0p5|0.7460|0.5850|0.6333|141|
|All-view weighted64-tree control|common|0.7947|0.6685|0.7000|141|
|All-view weighted256-tree control|fixed0p5|0.7510|0.5767|0.6444|141|
|All-view weighted256-tree control|common|0.7886|0.6510|0.6988|141|
|Seven-candidate nested selection|fixed0p5|0.7587|0.5903|0.6395|141|
|Seven-candidate nested selection|common|0.7917|0.6439|0.7025|141|

**The tested model adjustments do not establish an improvement.** The primary nested-selection common-threshold mean is 0.7917, versus 0.7992 for the previous control. Do not replace the prior benchmark with a more favorable seed or a different subset.

| Common-threshold modification vs prior control | Micro-F1 change | Conditional95% paired interval |
|---|---:|---:|
|All-view weighted64-tree control|-0.0045|[-0.0216,0.0125]|
|All-view weighted256-tree control|-0.0106|[-0.0277,0.0067]|
|Seven-candidate nested selection|-0.0076|[-0.0255,0.0095]|

Every interval above includes zero. These paired near-group bootstrap intervals condition on saved predictions; refit/seed SD is separately saved. They are not experimental or new-parent intervals. More trees, stronger regularization or a compressed representation are not guaranteed to help this small cohort. This finite declared experiment ends here; unsuccessful controls are preserved rather than repeatedly searching the same outer results until a target score appears.

## All seeds and mixture errors

| Model | Seed | Common-threshold micro F1 | Complete mixtures /22 |
|---|---:|---:|---:|
|Prior canonical-view64-tree control|20261020|0.8126|9/22|
|Prior canonical-view64-tree control|20261021|0.7890|10/22|
|Prior canonical-view64-tree control|20261022|0.7961|10/22|
|All-view weighted64-tree control|20261020|0.7961|9/22|
|All-view weighted64-tree control|20261021|0.7906|8/22|
|All-view weighted64-tree control|20261022|0.7974|10/22|
|All-view weighted256-tree control|20261020|0.8032|10/22|
|All-view weighted256-tree control|20261021|0.7738|9/22|
|All-view weighted256-tree control|20261022|0.7889|9/22|
|Seven-candidate nested selection|20261020|0.8013|8/22|
|Seven-candidate nested selection|20261021|0.7730|8/22|
|Seven-candidate nested selection|20261022|0.8006|10/22|

A single seed over 0.8 is not selected as a reproducible overall result. Every omitted and false-positive mixture species is saved in `mixture_predictions.csv`. Per-species and domain-specific common-four/full-eleven scores are in `per_label.csv` and `metrics.csv`; record-weighted metrics are kept separately, not substituted for the group-weighted headline. No synthetic/repeated view counts as a new independent structure.

## Complete inventory, not score-based exclusions

The source dataset has 156 calculation folders and 152 with usable native series. Existing audits define 148 geometry/population-eligible configurations, of which 141 are in the supported 11-label identity task and 7 have unsupported water/carbonate components. Those 7 remain explicitly inventoried for review, not relabeled as free H/O or dropped after examining scores. The clean Cu100 reference is separately pinned to its previous verified complete sources; its current redundant raw absorber is incomplete. Missing top layers, unresolved geometry lineage or incomplete absorber populations cannot be repaired by model tuning and retain their original exclusion reasons. No new data exclusion is made by this study.

For the requested model adjustment, the141-record cohort, targets, energy grid, physical top layers and source weights are byte-identical to the November update. The use of all training views is recorded separately from model hyperparameter changes, with the64-tree all-view control isolating that intervention.

## Final full-data models and shared route

The final models in `final_models/` each train on all 141 records and135 near groups. Their training-only selection arrays, split memberships, thresholds, references, source hashes and reload evidence are saved. These are final inference candidates, not additional held-out evaluations or an outer-selected upgrade. The earlier validated control is not overwritten.

Both evaluation and final fitting use the same audited callable as the agent/UI: recipes `full_cohort_tuning_v1` and `full_cohort_final_v1`. The explicit agent recipes and shared signed-difference inference route are tested. Signed inputs, compatible energy support, documented normalization/SNR and substrate-specific reference conventions remain required. Identity still does not imply calibrated absolute CO coverage.

No manuscript, SI, captions, bibliography or publication figures were changed. No new electronic/HPC calculation, merge, force-push or portal write occurred. These are conditional simulation results, not measured experimental accuracy, unseen-parent transfer or a guaranteed0.8 score.
