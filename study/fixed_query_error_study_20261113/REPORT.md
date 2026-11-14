# Fixed-query data addition and recurring-error study

**Added training data did not establish an improvement on the fixed original queries.** Thirteen usable configurations (twelve new near-geometry groups, plus one additional view of an old group) change three-seed group-weighted micro F1 from **0.7992 to 0.7972**. Paired change **-0.0020**, conditional 95% geometry-group bootstrap interval **[-0.0221, +0.0184]**. CHO and OH omissions decrease, but additional CHO/CO false positives and other omissions offset that benefit. H-containing mixture recovery remains poor; complete CHO+CO+H recovery remains 1/8 for every seed.

**Single most informative next calculation:** calculate the CO-only counterpart of `CO_H/CO_H_0p083_2CO_1H` on the same parent with **2 CO molecules**, preserving its CO coordinates/sites for the first vertical diagnostic and removing only isolated H. Then obtain a separately relaxed counterpart. The existing CO+H query misses H in all three expanded-model seeds and has no adequate matched CO-only control. Do not use that already-seen anchor as prospective independent validation; reserve a wholly new matched family for that purpose. No jobs were submitted.

## Provenance and scope

- Repository: `HailiJia/CO2RR_XAS_agent`, draft PR #18, verified base head `12d9409c6be60492d925356e7deaa7431806330a`. This study is new work above that head; it does not replace an archived benchmark.
- Frozen reference cohort: `study/dataset_20261101`, 141 configurations / 135 conservative near-geometry groups. Training-array SHA256: `84439cdcd899ad4451d11c1176411d8a8407fb92995b4d1b9ecc29ed267d44e3`.
- November 13 model-comparison report was found locally in `CO2RR_11103/model_comparison_20261113/REPORT.md`, outside the checkout and absent from the base PR head. Report, producing scripts and all 18 family/seed prediction files are preserved in `prior_local_evidence/`; `prior_prediction_verification.csv` recomputes their group scores. Its scores concern unchanged-cohort architecture selection, **not** adding data. The earlier first rerun used different inner membership and record weighting; it is not the fixed-protocol comparison here.
- New source: local author export `CO2RR_1113`. The raw full dataset was not present at the base PR commit. Selected geometry/spectrum/source evidence hashes are in `source_manifest.json`; `inputs.npz` and full manifests contain the actual saved analysis inputs. Raw large electronic logs/wavefunctions are not duplicated.

## 1. What the data added

The audit produces **154 configurations / 147 near groups** in total. Of the 13 added records, **9** already had complete top-two spectra in the November 3 export but were absent from the frozen November 1 cohort; **4** first acquired both top-two spectra in the November 13 export. The latter are CHO/CHO_0p111_interface_Cu, CO_OH/CO_OH_0p194_4CO_3OH_AuCu2, OCCO_CO/OCCO_CO_0p111_AuCu_interface and OH/OH_0p25_interface_biased. All added configurations are CuAu; **zero new Cu100 training configurations** pass this task audit.

There are **18 additional convolved layer spectra** relative to November 3, of which **8** supply those four newly complete top-two pairs. Other deeper layers are extra observations of existing physical configurations, not independent training cases. `COOH/COOH_0p111_Cu_uniform` is within RMS 0.01398 Å / max 0.09377 Å of an existing COOH near group, so it adds a record view, not a new group, and the canonical one-view training rule retains the original view.

| category | value | configurations | groups | new_groups |
| --- | --- | --- | --- | --- |
| species | CHO | 2 | 2 | 2 |
| species | CO | 10 | 10 | 10 |
| species | CO2 | 1 | 1 | 1 |
| species | COOH | 1 | 1 | 0 |
| species | H | 5 | 5 | 5 |
| species | OH | 5 | 5 | 5 |
| species_set | CHO+CO+H | 2 | 2 | 2 |
| species_set | CO | 1 | 1 | 1 |
| species_set | CO+H | 3 | 3 | 3 |
| species_set | CO+OH | 4 | 4 | 4 |
| species_set | CO2 | 1 | 1 | 1 |
| species_set | COOH | 1 | 1 | 0 |
| species_set | OH | 1 | 1 | 1 |
| substrate | CuAu | 13 | 13 | 12 |

Exact molecular stoichiometries, separate from species-presence sets, are in `new_counts_by_substrate_species_mixture.csv` and `new_cohort_counts.csv`. Species totals overlap because mixtures contain multiple labels. Groups are geometry clusters, not independent clean-parent families.

**Geometry/label checks:** all 141 frozen XAS geometries match their supplied top-layer input geometry, and all 141 frozen label sets agree with the observed XAS-input graph. A newer root CONTCAR is never substituted into a frozen query. The historical H_0p25/interface_biased2 root/XAS discrepancy is retained as originally audited and explicitly listed. For new cases, require stable graph labels under the existing ±10% bond-cutoff check, an exact root-relaxed CONTCAR/XAS mapping, and agreement between top-two input geometries. CHO is identified by C–H and COH by O–H connectivity, not folder name. The new CHO folders contain 2CHO+2CO+2H; the OCCO_CO folder contains 4CO, not intact OCCO. Initial-versus-XAS connectivity and supplied-label disagreements are recorded per case in `audit_inventory.json`.

**Reference and population checks:** every newly admitted layer covers all 36 physical Cu sites; top-two populations are 36+36. Raw source grids are complete, site/log E_cut mappings pass, and all new raw source symmetry multiplicities equal one. Thus native aggregate divided by physical N is already the per-Cu scale; no guessed aggregate symmetry correction is needed. Independent reconvolution was not run in this study. Pin the previously audited clean physical curves only after source hashes match; reject cross-cell/composition/settings/grids. CuAu and Cu100 keep separate clean references. The incomplete redundant Cu100 clean native source is not used. Neither CONTCAR availability nor complete spectra establishes a relaxation stopping criterion; no new convergence or independent-parent-transfer claim is made.

**Exclusions:** `exclusions.csv` lists every non-old candidate that fails the task audit, including missing top-two spectra, root/XAS lineage mismatch, unsupported H2O/CO3 chemistry and an invalid OH layer population. The clean Cu100 entry is a pinned reference, not an added independent query. None of the original 141 queries is removed because of model errors.

| configuration | status | reason |
| --- | --- | --- |
| CO/CO_0p11/uniform2 | excluded new candidate | Missing physical top-two convolved spectra |
| CO2/CO2_0p084_interface_biased_activated | excluded new candidate | Missing physical top-two convolved spectra |
| COOH/COOH_0p111_mixed_Au_Cu | excluded new candidate | Missing physical top-two convolved spectra |
| CO_H/CO_H_0p139_3CO_2H_interface_biased | excluded new candidate | No exact relaxed root CONTCAR/XAS lineage for a new case |
| CO_OH/CO_OH_0p139_3CO_2OH_AuCu2 | excluded new candidate | Missing physical top-two convolved spectra |
| Cu100/Cu100_2x2x4_1CO3_1H2O_hollow_flat-1HB | excluded new candidate | Unsupported identity outside frozen eleven-label task: CO3,H2O |
| Cu100/Cu100_2x2x4_1CO3_hollow | excluded new candidate | Unsupported identity outside frozen eleven-label task: CO3 |
| Cu100/Cu100_2x2x4_1H2O_1CO_H2Otop_CObridge | excluded new candidate | Unsupported identity outside frozen eleven-label task: CO,H2O |
| Cu100/Cu100_2x2x4_1H2O_1OH_H2Otop_OHbridge | excluded new candidate | Unsupported identity outside frozen eleven-label task: H2O,OH |
| Cu100/Cu100_2x2x4_1H2O_2OH_H2Otop_OHbridge-top | excluded new candidate | Unsupported identity outside frozen eleven-label task: H2O,OH |
| Cu100/Cu100_2x2x4_clean | pinned reference; not a new query | No exact relaxed root CONTCAR/XAS lineage for a new case |
| OCCO_CO/OCCO_plus_CO_0p333_AuCu_interface | excluded new candidate | Missing physical top-two convolved spectra |
| OCCO_H2O/OCCO_2H2O | excluded new candidate | Unsupported identity outside frozen eleven-label task: CO,H2O |
| OCCO_H2O/OCCO_2H2O_atoo_bridge_interior | excluded new candidate | Unsupported identity outside frozen eleven-label task: CO,H2O |
| OCCO_H2O/OCCO_2H2O_atop_bridge | excluded new candidate | Unsupported identity outside frozen eleven-label task: CO,H2O |
| OH/OH_0p50_uniform | excluded new candidate | Layer table does not cover the full physical Cu layer |

## 2. Adding training data on fixed queries

Both arms retain the exact original 141 spectra, labels, test-query fold assignments and original seed/fold thresholds. Model: independent balanced 64-tree ExtraTrees, leaf size 1, max_features=sqrt; verified population-weighted physical top-two Cu K-edge reaction-minus-clean spectra, training-only signed RMS floor, Combined spectrum/integral/first/second derivative features. Seeds: 20261020/21/22. Every old-only probability and binary prediction reproduces the verified November 13 local baseline exactly.

Five inherited near groups have related query views assigned to different original outer folds. Moving them would violate the fixed-query-fold requirement. We therefore keep those query folds and **purge the entire held-out related group from every training fit**, exactly as the canonical baseline did. No outer or inner fit has a shared train/validation/test geometry family. Related new views inherit old grouping; new unrelated groups use a deterministic geometry-group hash fold. The inherited conflict inventory and exact memberships are saved; these folds must not be described as disjoint test-group partitions. Bootstrap resamples whole near groups, preserving their correlated views.

New data enter only permitted training folds. One canonical view per group prevents the additional COOH view from silently increasing its weight. Original fold-specific thresholds are transferred unchanged to both arms. Separate common-threshold recalibration uses inner grouped predictions only; it does not enter the primary added-data comparison.

| protocol | micro_f1 | macro_f1 | exact_set_accuracy |
| --- | --- | --- | --- |
| expanded_H_threshold | 0.8008 | 0.6610 | 0.6901 |
| expanded_locked | 0.7972 | 0.6594 | 0.6827 |
| expanded_recalibrated | 0.8035 | 0.6558 | 0.6889 |
| expanded_residual | 0.7946 | 0.6575 | 0.6827 |
| old_only_locked | 0.7992 | 0.6630 | 0.7037 |
| old_only_recalibrated | 0.7992 | 0.6630 | 0.7037 |

All figures in the table are three-seed means, with each geometry group carrying total evaluation weight one. Macro F1 includes all eleven frozen labels, including unsupported-in-a-fold positives; exact-set accuracy requires every label to match. Seed variation and majority voting are separate:

| protocol | seed | micro_f1 | macro_f1 | exact_set_accuracy |
| --- | --- | --- | --- | --- |
| old_only_locked | 20261020 | 0.8126 | 0.6766 | 0.7148 |
| old_only_locked | 20261021 | 0.7890 | 0.6615 | 0.6852 |
| old_only_locked | 20261022 | 0.7961 | 0.6509 | 0.7111 |
| old_only_locked | voting | 0.8099 | 0.6700 | 0.7222 |
| expanded_locked | 20261020 | 0.8122 | 0.6707 | 0.7000 |
| expanded_locked | 20261021 | 0.7859 | 0.6704 | 0.6556 |
| expanded_locked | 20261022 | 0.7934 | 0.6373 | 0.6926 |
| expanded_locked | voting | 0.8046 | 0.6561 | 0.7037 |

Voting probabilities are not averaged and rethresholded: voting means at least two of three predeclared binary predictions. The saved mean score is diagnostic only. `paired_bootstrap.csv` reports paired micro/macro/exact-set intervals, conditional on fixed OOF predictions; these intervals do not incorporate retraining, independent-parent or measured-experiment uncertainty.

**New queries, evaluated separately:** thirteen new records form thirteen test groups within that subset (one group is shared with the old cohort). Their difficulty does not change the primary test population. Macro F1 still averages eleven labels, so absent-label zeros lower it; it is not directly comparable with full-cohort macro F1.

| protocol | micro_f1 | macro_f1 | exact_set_accuracy |
| --- | --- | --- | --- |
| old_only_locked | 0.8301 | 0.3746 | 0.5385 |
| expanded_locked | 0.8471 | 0.3878 | 0.6154 |

## 3. Errors before targeted changes

| protocol | seed | H_mixture_omissions | H_containing_mixtures | CHO_CO_H_exact | CHO_CO_H_sets | false_H_CO_only |
| --- | --- | --- | --- | --- | --- | --- |
| old_only_locked | 20261020 | 8 | 16 | 1 | 8 | 1 |
| old_only_locked | 20261021 | 8 | 16 | 1 | 8 | 4 |
| old_only_locked | 20261022 | 8 | 16 | 1 | 8 | 1 |
| old_only_locked | voting | 8 | 16 | 1 | 8 | 1 |
| expanded_locked | 20261020 | 9 | 16 | 1 | 8 | 2 |
| expanded_locked | 20261021 | 8 | 16 | 1 | 8 | 5 |
| expanded_locked | 20261022 | 7 | 16 | 1 | 8 | 2 |
| expanded_locked | voting | 8 | 16 | 1 | 8 | 2 |
| expanded_H_threshold | 20261020 | 9 | 16 | 1 | 8 | 1 |
| expanded_H_threshold | 20261021 | 8 | 16 | 1 | 8 | 2 |
| expanded_H_threshold | 20261022 | 8 | 16 | 1 | 8 | 2 |
| expanded_H_threshold | voting | 8 | 16 | 1 | 8 | 2 |
| expanded_residual | 20261020 | 9 | 16 | 1 | 8 | 2 |
| expanded_residual | 20261021 | 8 | 16 | 1 | 8 | 5 |
| expanded_residual | 20261022 | 7 | 16 | 1 | 8 | 2 |
| expanded_residual | voting | 8 | 16 | 1 | 8 | 2 |

The original pattern is reproduced exactly: each seed omits H in 8/16 H-containing mixtures and recovers only 1/8 CHO+CO+H sets. Expanded training gives 9/8/7 H omissions and still 1/8 complete triple-mixture recovery. False H on 32 CO-only original queries rises from 1/4/1 to 2/5/2. Voting omits H in 8/16 mixtures and still recovers only 1/8 triple mixtures.

Mean original-query CHO omissions decrease from 5.67 to 4.33 per seed and OH omissions from 4.00 to 2.33, but CHO false positives increase from 1.33 to 3.33 and CO false positives from 8.67 to 10.00. O, OCHO and COOH omissions also increase slightly. `per_case_predictions_errors.csv`, `errors_long.csv`, `error_transitions.csv` and the four `error_breakdown_*.csv` files preserve per-species, exact stoichiometry/species set, substrate and pure/mixed detail.

H has training positives in every outer fold (25–34 positive training groups in expanded fits). Expanded inner H AUC is 0.818–0.918 (mean 0.872), so common-cutoff calibration is a plausible limited test, but CO-containing mixture scores still overlap CO-only negatives. Original-query OOF H AUC is diagnostic, not a selection criterion: all-case AUC improves from 0.917 to 0.928; CO-containing AUC from 0.831 to 0.836. Those aggregate rankings do not imply complete mixture recovery at acceptable false-positive rates.

There remain only two pure CHO configurations, and the new CHO positives are mixtures rather than new pure CHO controls. COCHO still has one positive geometry group; its held-out-positive fold has zero COCHO training positives in both arms, so zero held-out COCHO recall cannot establish a learned positive class. Missing supports and per-fold rare-label limitations are saved in `training_label_support.csv`.

## 4. Two targeted changes, selected within training only

**H-specific cutoff:** keep all other labels at the original common threshold. Permit H offsets ±0.05/0.10 (including unchanged), require inner H ranking/support, optimize group-weighted inner H F1, and allow at most one extra group-weighted CO-only false H call in inner validation. The selected thresholds frequently increase rather than decrease: the training objective favors specificity. On original queries this corrects **zero H omissions** and adds **zero false H calls**; it removes five false H calls across the three seed/case evaluations (one on CO-only in seed20 and three on CO-only in seed21) but introduces one H omission in seed22. It therefore does not repair the recurring missed-mixture mechanism. Voting H-mixture omissions, CO-only false calls and triple-mixture exact recovery are unchanged.

**CO-template residual for H/OH:** if CO-conditioned H inner AUC is below 0.80, form a substrate-matched pure-CO mean template from training groups only. Fit a nonnegative scalar to each common-scale signed spectrum, subtract the CO template, then apply training-floor signed RMS and Combined features. Each training example uses templates excluding its entire geometry group; inner validation templates exclude all validation groups. Require at least two pure-CO training groups for a template; unsupported cases retain original scores rather than being dropped. Fit the same 64-tree estimator for H/OH and retain the original cutoff. Switch a label to residual scores only if inner group-weighted label F1 improves by at least .02, with the H false-call constraint. This is a test of an additive template hypothesis on actual relaxed mixtures, not synthetic sums.

The overlap gate permits residual evaluation in 10/15 seed/fold fits. H residual replacement is selected in **0** fits; OH in **7** fits. The selected residual arm does not improve original-query micro F1 or H-mixture recovery. An unselected residual-candidate diagnostic, with original-score fallback and all queries retained, is saved separately; it is not used to choose an outer-test winner:

| species | original_F1 | residual_candidate_F1 | omissions | false_positives |
| --- | --- | --- | --- | --- |
| H | 0.7516 | 0.7817 | 8.3333 | 6.6667 |
| OH | 0.8909 | 0.8697 | 3.0000 | 1.6667 |

Paired changes below are relative to the expanded locked-threshold arm; recalibration is explicitly separate from the data-addition effect:

| before | after | difference | lower | upper |
| --- | --- | --- | --- | --- |
| expanded_locked | expanded_recalibrated | 0.0063 | -0.0035 | 0.0162 |
| expanded_locked | expanded_H_threshold | 0.0036 | 0.0001 | 0.0079 |
| expanded_locked | expanded_residual | -0.0026 | -0.0082 | 0.0000 |

Both changes are exploratory targeted analyses prompted by known recurring errors, not prospective confirmation. No architecture sweep or outer-score optimization was added.

## 5. Matched contrasts and ranked calculations

We searched **44 actual same-substrate/same-CO-count CO-only versus CO+H candidate pairs** among 33 CO-only and 11 CO+H configurations. Four share an initial metal-parent hash, but none simultaneously matches parent, environment, contact composition and relaxed CO-site distances (carbon RMS≤0.5 Å and max≤1 Å). There are **zero adequate existing matched pairs** under those disclosed criteria; initial parent hashing itself is not a trajectory-level lineage proof.

For the proposed anchor, the best parent-prioritized available CO-only comparison is `2CO_adjacent/2CO_adjacent_0p056_interface`. Its CO-site RMS mismatch is **7.45 Å**, so its shared-scale spectral difference **0.003668 RMS** cannot be attributed uniquely to H. Comparable-loading/environment CO-only variation is **0.002457 median RMS**. The saved hypothetical ±0.1/0.2 eV registration scenarios reach **0.007989 RMS** for that CO-only curve; a 1% gain scenario is also saved. These are explicitly simulated sensitivities, **not measured noise or energy-registration estimates**. The 0.2 eV energy grid is sampling, not an uncertainty estimate. Actual pair curves and signed differences are saved before unit-RMS normalization in `matched_pair_spectra.npz`; nearest unmatched controls are never called H-only contrasts.

The calculation list is prepared only; no scheduler, electronic calculation, remote portal, manuscript, SI or publication-figure update was invoked.

### Priority 1: CO-only counterpart of an existing repeatedly missed relaxed CO+H configuration

**Hold fixed:** Same clean parent, cell, Cu/Au ordering, electronic/broadening settings, top-two absorber populations and CO count/contact motif as CO_H/CO_H_0p083_2CO_1H.

**Vary:** Remove only isolated H from the XAS-used CO+H snapshot for an initial vertical diagnostic; then separately relax the CO-only counterpart under the same slab constraints. Label vertical and relaxed comparisons separately..

**Decision:** If adding H gives a reproducible signed contribution above matched CO-only site variation and independently estimated measurement/registration uncertainty, pursue residual/contrast features; if not, collect complementary H-sensitive observables rather than lower the cutoff.

**Prospective split:** Reserve the entire new clean/CO-only/CO+H family and all its layer/site views for prospective checks; never split a family between training and evaluation.

### Priority 2: Replicate CO-only/CO+H matched families in both interface and Cu-rich environments

**Hold fixed:** Within each family: parent, CO count, CO sites/contact compositions, coverage denominator, reference and numerical settings.

**Vary:** H occupancy/location at fixed CO; then change clean parent independently. At least one interface and one Cu-rich family, with independent-parent replication..

**Decision:** If H contrast generalizes across parents/environment and survives CO-only motif variation, add matched families to training; if environment-specific, acquire environment-stratified controls before claiming general H inference.

**Prospective split:** Reserve the entire new clean/CO-only/CO+H family and all its layer/site views for prospective checks; never split a family between training and evaluation.

### Priority 3: Pure CHO and matched CHO/COH connectivity controls

**Hold fixed:** Same parent, coverage, comparable C/O contact sites, top-two physical populations, reference, spectral settings.

**Vary:** C-bound H (CHO) versus O-bound H (COH); include isolated CHO and coadsorbed CHO+CO+H. Verify final C-H/O-H connectivity in the exact XAS geometry..

**Decision:** Stable distinct pure-CHO versus COH signatures plus improved held-out pure-CHO recovery justify extending the claim; isomerization/dissociation requires relabeling actual products and narrower claims.

**Prospective split:** Reserve the entire new clean/CO-only/CO+H family and all its layer/site views for prospective checks; never split a family between training and evaluation.

### Priority 4: COOH/OCHO matched controls if directional errors persist

**Hold fixed:** Same substrate parent, coverage, binding/contact composition and common intensity scale.

**Vary:** O-bound versus C-bound H at fixed CO2H stoichiometry; comparable motif/coverage sweep.

**Decision:** Persistent directional confusion despite adequate training support motivates matched pairs; otherwise prioritize H and pure CHO first.

**Prospective split:** Reserve the entire new clean/CO-only/CO+H family and all its layer/site views for prospective checks; never split a family between training and evaluation.

### Priority 5: Additional independent COCHO positives if COCHO remains central

**Hold fixed:** Verified C-C/C-O/C-H graph; matched clean reference and converged complete top-two spectra.

**Vary:** Independent geometries/parents, coverage and motif; acquire several positive groups so each training fold contains positives.

**Decision:** If independent positive groups allow nonzero held-out recall, assess COCHO inference prospectively; a sole positive group cannot support a general learned-recognition claim.

**Prospective split:** Reserve the entire new clean/CO-only/CO+H family and all its layer/site views for prospective checks; never split a family between training and evaluation.

COOH/OCHO currently suffers primarily from omissions: expanded voting omits COOH in 6/9 positives and OCHO in 4/8. Direct cross-label confusion is rarer (one false OCHO call among COOH positives, zero reverse calls in voting). `directional_label_confusions.csv` separates these mechanisms, so these controls remain below H and pure CHO in priority. COCHO additions are conditional on that species being central to the claim; a sole positive group does not justify broad identification claims.

## Reproduction and saved evidence

Use the repository at the recorded base commit plus this study. Python 3.13, NumPy 2.3.5 and scikit-learn 1.8.0 were used; exact dependency versions are in `requirements.txt` and `evaluation_protocol.json`. No network access is needed to refit from the saved audited arrays.

```sh
python -m pip install -r study/fixed_query_error_study_20261113/requirements.txt
python study/fixed_query_error_study_20261113/code/train.py
python study/fixed_query_error_study_20261113/code/analyze.py
python study/fixed_query_error_study_20261113/code/matched_pairs.py
python study/fixed_query_error_study_20261113/code/report.py
python study/fixed_query_error_study_20261113/code/verify.py
```

These commands overwrite only this new study’s regenerated outputs; preserve a copy first if changing inputs/code. To repeat source admission with both raw exports present, run `code/audit.py --old-raw /path/CO2RR_11103 --new-raw /path/CO2RR_1113` first. The saved immutable query inputs avoid substituting changed root geometries. Models, held-out probabilities/labels, per-case errors, inner scores, fold/template memberships, source hashes, original thresholds, dependency versions and ranked calculation JSON/CSV are included. Every model is reloaded and its held-out scores replayed exactly. `verification.json` checks original input identity, all fold/template group guards, selected source hashes, and unchanged original tracked manuscript/SI/figure/benchmark files. `ARTIFACTS.json` records this study’s saved files.
