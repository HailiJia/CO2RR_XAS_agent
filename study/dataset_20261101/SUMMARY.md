# CO2RR_11101: guarded signed-shape retraining

No manuscript, SI, bibliography, captions or publication figures were edited. Previous studies/predictions remain unchanged. No new electronic/HPC calculation or portal write occurred.

## Dataset and provenance

- Supported identity cohort: **130 → 141 records**, **124 → 135 near-geometry groups**. Cu–Au:118→129; Cu(100):12 unchanged. These are not independent clean-parent families.
- All130 earlier labels, inherited outer folds and differential arrays are unchanged; arrays are bitwise identical. Hydrogen folders were renamed (H/H_0p25/... → H_0p25/...), matched by geometry rather than counted as new cases.
- Eleven added configurations: one CHO/CO/H mixture, two pure COH, one CO/COH mixture, two CO/H mixtures, two CO/OH mixtures and three pure O configurations. Names do not establish these labels; the actual XAS geometry graph does. The new CHO bidentate seed is **3CHO+CO+H**, not pure CHO.
- CHO support9→10 positive records (8→9 groups), but **pure CHO remains2**. COH10→13 positives (pure10→12), CO48→54 (pure32 unchanged), H33→36, O8→11, OH16→18. COCHO remains1.
- Source inventory:156 calculation folders,152 with usable native series. Generic/four-layer conversions and element-aware CONTCAR/XAS graph audits are preserved; OH/interface-Au uses verified top layers only, excluding inconsistent deeper layers. Unresolved geometry mismatch, missing top layer and incomplete absorber population remain excluded. Water/carbonate queries stay outside the eleven-label task.
- The current clean Cu(100) layer3 absorber11 raw file has143 points ending9002.4eV, versus276 points through9029eV required. It is **not extrapolated**. The pinned previous clean reference is reused with matching full geometry and exact hashes of complete symmetry representatives; the newly supplied four-source clean aggregate is not asserted validated. Unchanged corrections are reused only for exact geometry/native-array matches; all new cases are locally reconvolved with verified physical source weights.

## Fixed model and fair comparison

Population-weighted top two layers remain fixed (Cu–Au4+5, Cu1003+4); signed-shape **Combined=[smoothed signed spectrum, cumulative difference, d1,d2]**. Independent balanced64-tree, leaf1 anchor; regularized C1 logistic baseline. Seeds20261020/21/22 all reported. No automatic depth selector or outer-test model/seed choice. Counts, filenames, geometry/site/contact labels and loading never enter spectral predictors.

Every outer/inner training partition excludes all query-related near groups and retains one canonical training record per group. Three threshold protocols are separately reported: fixed0.5, previous training-only common threshold transferred, and restrained common threshold selected within current inner training.

**Equal-near-group weighting; mean across the three seeds:**

| Model / threshold | Same130: previous micro F1 | Same130: updated micro F1 | Paired change [conditional95% interval] |
|---|---:|---:|---:|
| trees / fixed0p5 | 0.7466 | 0.7637 | +0.0170 [-0.0038, 0.0377] |
| trees / transferred | 0.7692 | 0.7870 | +0.0178 [-0.0054, 0.0407] |
| trees / common | 0.7692 | 0.7904 | +0.0212 [-0.0073, 0.0499] |
| logistic / fixed0p5 | 0.6500 | 0.6396 | -0.0104 [-0.0270, 0.0062] |
| logistic / transferred | 0.6844 | 0.6657 | -0.0187 [-0.0387, 0.0012] |
| logistic / common | 0.6844 | 0.6657 | -0.0187 [-0.0387, 0.0012] |

**All paired micro-F1 intervals include zero.** F1 improvement is not established. Logistic declines are also inconclusive. Bootstrap intervals condition on fixed held-out predictions and jointly sample geometry groups across seeds; seed/refit variation is separately saved. They do not include shared-reference, new-parent or experimental uncertainty. Logistic fits are deterministic across these seeds.

| Tree / fixed0.5 query scope | Micro F1 | Macro F1 | Exact-set accuracy |
|---|---:|---:|---:|
| Previous training, same130 queries | 0.7466 | 0.5864 | 0.6304 |
| Updated training, same130 queries | 0.7637 | 0.5999 | 0.6626 |
| Updated training, all141 queries | 0.7631 | 0.5956 | 0.6605 |

On identical130 queries, exact-set gain is +0.0323, conditional95% interval[0.0027,0.0672]. This positive fixed-prediction interval does not establish a refit-stable or independent-parent improvement. Do not compare full141 with old130 as a paired improvement. The previously highlighted .7602 was **one seed**, whereas .7466 here is the **three-seed baseline mean**; seed20261020 archived scores reproduce bitwise.

## Mixtures, domain and rare-species limits

| Seed | Earlier training, old16 mixtures | Updated training, same16 | Updated full22 mixtures |
|---|---:|---:|---:|
|20261020|1/16|3/16|7/22|
|20261021|1/16|4/16|6/22|
|20261022|1/16|4/16|7/22|

Mixture recovery improves on identical cases, but most complete sets are still missed. Exact omitted and false-positive species are in `mixture_predictions.csv`; neither oversampling nor molecular counts create additional independent configurations.

Current141, tree/fixed0.5, three-seed mean **record-weighted** per-label scores (separately labeled from group-weighted main metrics):

| Label | Positive records | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
|CHO|10|0.5556|0.1333|0.2137|
|CO|54|0.9415|0.7840|0.8553|
|CO2|5|1.0000|0.6000|0.7500|
|COCHO|1|0.0000|0.0000|0.0000|
|COH|13|1.0000|0.7692|0.8696|
|COOH|9|0.6111|0.2222|0.3248|
|H|36|0.9082|0.6389|0.7500|
|O|11|1.0000|0.6667|0.7992|
|OCCOH|5|1.0000|0.4000|0.5714|
|OCHO|8|1.0000|0.5000|0.6667|
|OH|18|1.0000|0.7037|0.8258|

Current tree/fixed0.5 equal-group domain means:

| Domain | N | Micro F1 | Macro F1 (11 labels) | Exact sets |
|---|---:|---:|---:|
|CuAu|129|0.7717|0.6025|0.6707|
|Cu100|12|0.6556|0.1815|0.5556|

Cu(100) common-four macro F1 is .4990 (micro .6556); full-eleven macro includes seven unsupported-on-this-domain labels. Twelve Cu100 configurations do not establish new-facet or independent-parent transfer. Per-domain common-four/full-eleven and record/group metrics are saved separately. COCHO remains unlearnable in the fold without positive training support. CHO recall remains poor; more CHO-containing mixtures do not fill the pure-CHO gap.

**CO coverage:** the32 CO-only records (28 Cu–Au+4 Cu100) are unchanged; no new CO-only configurations were added. This identity retraining supplies no new CO-only coverage evidence. Signed-shape identity is not an absolute-loading calibration. Hollow composition and interface proximity remain separate questions; no missing Au-rich support is invented.

## Reproduction and verification

Preparation: `code/prepare.py`, `code/recover_top_layers.py`, `code/scale_pairs.py`, `code/groups.py`; portable source archives retain selected actual bytes. `local_export/` is an ignored local staging map including the separately supplied original Cu100 ASE snapshots. Conversion inputs are read-only. The reusable record fix initializes missing generic composition metadata from the actual audited graph.

Actual fit used the shared audited callable:

```sh
python -m tools.audited_training --data-root study/dataset_20261101 --output-dir /tmp/new_11101_fit --recipe guarded_dataset_update_v1
```

Use a new empty output directory. `tools/guarded_dataset_update.py` saves all90 models, folds/inner memberships, training supports, thresholds, predictions, model/source hashes and exact reload evidence. Models and record archives are lossless; `code/package.py --unpack` restores them with member-hash checks. `code/verify.py` replays numerical checks without fitting. `metrics.csv`, `per_label.csv`, `NUMBERS.csv`, `paired_intervals.csv` and `label_support.csv` provide all numbers including record-weighted controls.

No manuscript integration was performed. These remain conditional simulation results, not measured experimental accuracy, calibrated detection limits, independently confirmed kinetics or new-parent validation.

## Toward micro F1 above 0.8

The same 130-query tree/common-threshold scores are **0.8052, 0.7792 and 0.7868** for seeds20261020/21/22; the mean is **0.7904**. Its added-data paired interval includes zero. The0.8052 seed exceeds0.8, but must not be selected as a headline from the outer-test scores. Macro F1 remains much lower; a micro threshold target does not establish rare-species reliability.

The next restrained experiment is a training-only, group-nested comparison of signed Combined versus low-order DCT and tree minimum leaf sizes1/2/4, preserving all existing query folds/seeds and reporting all results. These feature families have already been studied; re-evaluating them on the larger cohort is not a new method or a guaranteed gain. Keep threshold selection restrained and inside training. No such additional search is claimed in this update.

The more important data additions are independent pure CHO, more than one independent COCHO environment, and matched CO+H/CHO/COH or CO+OH compositions at fixed loading/parent/motif. Mixture support is still small and incomplete. Repeated similar CO structures, oversampling, selecting the best seed or testing many settings against the same outer queries cannot establish improvement. Abstention may improve retained-query performance only with its coverage/full-query errors disclosed; it does not raise the directly comparable full-query benchmark. Reserve genuinely new configurations for a locked prospective evaluation.
