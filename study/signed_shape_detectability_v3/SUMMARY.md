# Signed-shape identity: detectability and guarded training-size study v3

The shared inference route now separates normalization from detectability. Documented normalized signed differences need no absolute per-Cu calibration, surface fraction or raw pair. Compatible-unit noise or documented pre-normalization SNR supplies detectability evidence; arbitrary positive divisors do not invoke a simulation-amplitude floor. Unknown detectability returns an explicit review status without a class set. Exact zeros and low-SNR inputs abstain before shape normalization. The illustrative SNR rule (3; boundary reviewed) is not a measured detection limit. Reaction-minus-reference signs, full energy support, reference chemistry and double-subtraction guards remain. Absolute coverage is not inferred.

**Saved simulated predictions are unchanged:** all 260 held-out queries across the archived fixed joint relative-delta and signed-shape models reproduce bitwise scores, sets and statuses when explicitly using their audited simulation-scale convention. Unknown-quality experimental inputs intentionally receive review instead of an implied simulation-derived detection limit. See `saved_simulation_inference_comparison.json` and shared-route tests.

## Added training data: identical old queries, not independent confirmation

Fixed population-weighted physical top two layers, source-corrected arrays, labels, energy grid and inherited outer queries remain unchanged. There are 130 identity records/124 near-geometry groups; old99 queries span 96 groups. Added Cu–Au training records contribute 19 records/16 groups. Every query-related group is excluded from training; one canonical available record per remaining group is retained. Inner validation applies the same exclusion. These groups are not independent clean-parent families.

Fixed Combined signed-shape features; independent balanced 64-tree/leaf1 anchor and C1 balanced logistic baseline. All predeclared seeds 20261020/21/22 are included. Thresholds are fixed 0.5, transferred old-training common thresholds, or restrained common thresholds chosen only in the corresponding training partitions. No depth selector, outer-query tuning or new spectra.

Mean micro F1 with **equal geometry-group weights**, on identical old99 queries:

| Estimator / threshold | Old training | Expanded training | Paired difference [conditional 95% interval] |
|---|---:|---:|---:|
| Trees / 0.5 | 0.7417 | 0.7591 | +0.0173 [−0.0092, 0.0490] |
| Trees / transferred | 0.7580 | 0.7785 | +0.0205 [−0.0149, 0.0543] |
| Trees / training-selected common | 0.7580 | 0.7767 | +0.0187 [−0.0175, 0.0552] |
| Logistic / 0.5 | 0.6170 | 0.6418 | +0.0249 [−0.0087, 0.0599] |
| Logistic / transferred or common | 0.6504 | 0.6667 | +0.0163 [−0.0109, 0.0481] |

**Every added-data interval includes zero.** No statistically established improvement is claimed. Paired group bootstrap intervals condition on fixed predictions; separate seed/refit SD is saved. Logistic refits are deterministic here, so identical seeds do not provide additional independent evidence. Record-weighted metrics, all seed results, macro F1, exact sets and per-label precision/recall/support are separate CSV rows. Threshold protocols are not selected by outer results.

The label-blind nested group learning curve gives tree mean micro F1 0.6712/0.7299/0.7591 at 50/75/100% of available training groups; logistic gives 0.5359/0.6070/0.6418. Actual training group counts and memberships are saved for every fold/seed. Curves describe this conditional protocol, not prospective or new-parent validation.

## What signed shape improves—and what it does not

Reanalysis of the **already committed fixed joint tree predictions**, not a new model search:

| Corrected representation | Group-weighted micro F1 | Macro F1 | Exact sets |
|---|---:|---:|---:|
| Corrected-normalized raw | 0.7155 | 0.4675 | 0.5968 |
| Corrected common-scale raw | 0.6989 | 0.4733 | 0.5806 |
| Relative delta | 0.6940 | 0.4424 | 0.5565 |
| Signed shape | 0.7602 | 0.6163 | 0.6532 |

Native-normalized raw remains a separately labeled historical weighting/normalization control in the machine tables. Source correction is a physical consistency correction, not a guaranteed accuracy gain.

Relative delta → signed shape improves fixed joint micro F1 by 0.0662, with conditional group-bootstrap interval [0.0222, 0.1135]. This excludes zero **only for these fixed predictions**; it excludes refit uncertainty and says nothing about experimental or independent-parent accuracy. Cu–Au micro F1 changes 0.7089→0.7634; Cu(100) common-four micro F1 changes 0.5263→0.7273, but its paired interval [−0.0043, 0.4652] includes zero. Full-eleven-label and common-four-label metrics are both saved by domain; Cu(100) has only 12 supported identity cases and four loading cases.

The exact-set gain is entirely in single-species records: Cu–Au has 14 wrong→correct and 3 correct→wrong; Cu(100) has 3 wrong→correct and none in reverse. Cu–Au recoveries include COH, O, OCCOH, COOH, CO2, OCHO and CO; the gain is not simply frequent CO recovery. Cu(100) gains one CO, one H and one OH case. One H case acquires a false CO. This supports improved conditional recovery in several classes/domains, not a proven physical mechanism.

**Cu–Au mixture exact recovery remains 1/16 records (15 geometry groups) for both representations.** Detailed identity sets, omissions, false positives, coverage, supported contact-count proxies and geometry-audited interface distances are saved. Contact-count proxies are not automatically hollow-site classifications. Coverage and motif change together in the four Cu(100) CO cases. Hollow composition and interface proximity are separate descriptors; absent CuAu2/Au3 support is not invented.

CHO has nine positive records/eight near groups but only two pure cases; molecules are separately counted. COCHO has one record/group. Its held-out fold has no positive training example; general held-out recognition is unsupported. CO/CHO/COH per-case calls and missing-positive inner/outer folds are explicit. Oversampling is not used.

One predeclared shrunk pair-dependence decoder (training-only pair prior, strengths 0/0.25; grouped inner choice) selects independence in four folds. It lowers group-weighted micro F1 0.7602→0.7551 and exact sets 0.6532→0.6452, with unchanged mixture recovery and false-positive count; no benefit is claimed. Fixed strength 0.25 is a separately labeled unsuccessful sensitivity, not an outer-selected method.

## Detectability, robustness and next data

Reused DCT16/logistic robustness models are a separate control, not the Combined/tree training-size anchor. Saved paired perturbations cover shared/residual +0.2-eV registration error, additional Gaussian sigma 0.25 eV, common/sample-only 5% gain and 1% independent or reused-reference noise. These are illustrative synthetic ranges, not measured instrument/noise information. No perturbation is tuned on outer labels. Native lifetime broadening is retained; all perturbed views remain in their source groups.

The corrected shared route reports retained-query fractions, retained error and full-query model error before the gate. With the chosen 1%-of-absorption noise, **no queries pass the SNR gate** in either noise scenario; performance after rejection is unavailable, not perfect. Unknown detectability also yields no automatic decisions. Zero and noisy near-clean controls abstain consistently across positive rescalings. Sample-only 5% gain produces severe errors even when a signal is detectable. Training augmentation does not consistently improve full-query errors and is not adopted as a primary replacement. A common positive scale cannot correct relative sample/reference gain mismatch.

Ranked additions (recommendations only): (1) independently relaxed pure CHO environments and controlled same-parent CO+H/CHO/COH alternatives; (2) mixture composition sweeps at fixed loading/motif, especially omitted H; (3) rare species, particularly multiple independent COCHO and OCHO environments; (4) interface CO motif/coverage sweeps and supported CO/OCHO contact/location controls; (5) independent clean/covered parents and measured paired null/reference repeats to quantify gain, registration and shared-reference uncertainty. Expand Cu(100) motif/loading combinations before transfer claims. Dissociated OCCO/water remains CO plus water. No jobs were submitted.

## Artifacts and verification scope

- `ml/`: frozen inputs, targets/folds, thresholds, inner scores, all held-out arrays, learning memberships and source hashes. `verification/effective_inner_splits.json` records the actual canonical training subsets used by fits (the as-run splitter also saves pre-canonical candidate indices). Model archives contain all 150 saved models; each reload reproduces scores exactly.
- `code/`: analysis/plotting scripts, pre-fix inference source and as-run training dependency snapshot. Training transforms are AST-identical after inference-only fixes. `tools/signed_shape_size.py` is exposed through the shared audited-training API and explicit agent/UI recipe; export integrity is checked before dispatch.
- `figures/`: PNG-only analysis exports with CSV/NPZ inputs alongside; no publication performance edits.
- `MANUSCRIPT_INTEGRATION.md`: specific superseded statements and limits for subsequent integration. Older studies/predictions are untouched.
- Exact author-supplied baseline is `../../manuscript_reviewed/`; inventory has 303 file hashes. After the author's explicit “Fix both” reply, **only** author order/affiliations and Figure 2b's Cu(100) numeric column are corrected in `../manuscript_reviewed_scoped_fixes_v1/`. Main/SI scientific prose, captions, bibliography, other figures and benchmark arrays/CPU values are unchanged. New ML results are not inserted into either publication source.

Current verification receipts and command list are in `verification/`. Exact pushed-SHA clean-checkout results are recorded in PR18; precommit evidence is labeled accordingly. No experimental accuracy, calibrated detection limit, absolute surface loading, new-parent transfer or production scheduler result is established.
