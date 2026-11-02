# Controlled relative-difference identity study

**Signed-shape normalization helps the regularized logistic identity baseline on identical cases. Subtraction alone does not consistently help. More Cu–Au training data help the fixed differential anchor, conditional on these reused folds. Neither result establishes measured experimental accuracy, unseen-parent transfer or absolute coverage.**

The supplied reviewed paper is uploaded separately, with author order Haili Jia, Pooja Basera, Dimosthenis Sokaras and Maria K. Y. Chan (corresponding). Its circular Figure1, oblique structures, explicit Cu100 count column and Figure3 spectra/alignment/broadening/metrics/CPU data are preserved. No new outer-score winner replaces a historical paper benchmark. The following new study is versioned alongside those scopes.

## Input convention and actual route

`experimental_pair(...,task="identity")`, `differential_route.infer`, the agent recipe `relative_identity_v2` and the UI now support a documented precomputed signed difference without raw absolute spectra, an absolute per-Cu basis or a surface fraction. Alternatively, a compatible relative reaction/reference pair uses one reference-RMS divisor. Positive common gain cancels; independent arbitrary pair normalization is rejected. Energy registration, reaction-minus-reference sign, reference chemistry, normalization, domain, processing and noise status remain explicit.

Models preserve signed response. The optional identity shape transform divides by RMS only after a training-defined floor. Null/near-null queries abstain. Already unit-RMS inputs require a documented pre-normalization SNR>=3: a scalar suffices, without raw absolute arrays. This is an unvalidated detection heuristic, not reliable experimental calibration. Unknown residual coverage changes the interpretation to a change relative to reference. Unknown gain or active fraction prevents absolute loading; the calibrated coverage route remains separate.

`shared_route_trace.json` records one actual held-out simulated query through the shared inference callable and model reload. The fit used `run_audited_training`; agent natural-language parsing selects the same recipe. No production scheduling, new spectra or labeled measured experiment is implied.

## Representation: same configurations, folds and fixed recipe

All five controls use the same130 geometry groups, current labels/grid and population-weighted top two layers (CuAu4+5; Cu1003+4). No depth selection or structural predictors. Existing source-weight corrections, clean Cu100 symmetry multiplicities and lineage exclusions remain unchanged. The following fixed Combined/.5 controls remove feature-selection and threshold changes from the representation comparison:

| Representation | CuAu118 logistic micro F1 | CuAu118 trees micro F1 | Joint130 logistic | Joint130 trees |
|---|---:|---:|---:|---:|
| Native-normalized raw |0.6413|0.7667|0.5530|0.7149|
| Corrected-normalized raw |0.6396|0.7373|0.5527|0.7244|
| Corrected common-scale raw |0.6556|0.7554|0.5701|0.7120|
| Relative signed delta |0.6556|0.7500|0.5885|0.7200|
| Signed shape, guarded floor |0.7059|0.7647|0.6789|0.7645|

Logistic C=1 with training-fitted standardization; independent balanced64-tree leaf1 anchor; threshold.5. On CuAu, signed-shape versus relative-delta logistic improves micro F1 by0.0502, conditional95% interval[0.0059,0.0902]. Its tree difference0.0147 has interval[-0.0252,0.0607]. Joint differences are0.0904[0.0480,0.1326] for logistic and0.0445[0.0029,0.0893] for trees. These are configuration bootstrap intervals conditional on fixed predictions, without multiplicity adjustment or refit uncertainty; the folds have been repeatedly examined.

Correcting source weights is an evidence correction, not a guaranteed score improvement: fixed-tree corrected-normalized raw drops on CuAu and rises slightly in joint training. Common-scale raw and delta give identical CuAu logistic predictions because subtraction of one fixed reference offset is removed by training centering. The small tree differences are not evidence of a new independent physical signal. Full paired intervals, including null/negative comparisons, are saved.

A separate equal-budget nested comparison selects signed spectrum/DCT16/Combined inside training for each estimator family. Its best-looking outer scores are **not selected as a replacement**. Complete micro/macro/exact, precision/recall/support, single/mixed, domain-common-four/full-eleven, coverage and connectivity-derived strata are in the CSVs. All six candidates and unsuccessful outcomes are retained; no classifier chain or neural architecture search was added.

## Training size: identical old99 queries

Relative-delta Combined/tree anchor, same three predeclared seeds and preserved old inner assignments:

| Threshold rule | Old99 training mean micro F1 | Expanded118 training, same99 queries | Paired mean difference95% interval |
|---|---:|---:|---|
| Fixed.5 |0.6962|0.7517|+0.0556[0.0173,0.0998]|
| Transferred old-training common threshold |0.7163|0.7612|+0.0449[0.0094,0.0838]|
| New training-only restrained common threshold |0.7163|0.7752|+0.0589[0.0059,0.1102]|

The common rule searches only.3/.4/.5/.6 with a penalty toward.5. Sparse labels do not get independent unrestricted thresholds. The table's intervals resample configurations jointly across the three **fixed** seed-prediction arrays; seeds are not3× independent samples. Individual-seed intervals sometimes cross zero. Seed/refit SD of the difference is0.0243/0.0057/0.0089 respectively. This is stronger conditional evidence than the earlier raw-control change0.7434→0.7519, but neither is prospective confirmation.

Grouped learning curves use the same old99 outer queries, label-blind nested subsets of expanded training groups and fixed.5. Mean micro F1 for50/75/100% of available outer-training groups is0.6509/0.6941/0.7478; refit/seed SD0.0152/0.0124/0.0090. Membership and missing-positive training labels are saved. Full118 scores are not treated as paired old99 improvement.

## Domain, mixture, null and robustness limits

Cu100 remains a conditional domain with12 supported identity cases and four CO-only loading cases. One reference family/domain prevents leave-parent-out estimates. Entire-domain-held-out DCT16/logistic tests perform poorly: CuAu→Cu100 signed-shape common-four micro F1=0.2667 (full-eleven0.1538); Cu100→CuAu common-four=0.2500 (full-eleven0.2122). Relative-delta CuAu→Cu100 common-four0.6667 coexists with full-eleven0.3390 and zero full-set accuracy. These are stress tests, not validated domain transfer.

COCHO has one positive, CHO nine positives but only two pure configurations. Missing-positive folds, CO/CHO/COH omissions, mixture false positives and closest actual training spectra are saved. Neither class weighting nor repeated layers creates chemical diversity. Six water/carbonate queries remain unsupported chemistry, with review-only predictions. Dissociated OCCO/water remains CO plus water. Contact compositions and interface distances are reporting strata, never spectral predictors; contact counts are not automatically hollow labels or missing CuAu2/Au3 evidence.

There is only one independently held-out real clean negative in the historical cohort; it cannot establish specificity. Null/near-null synthetic queries and clean false positives are explicitly reported separately from independent sample counts. Review curves retain the denominator and all-query error; uncalibrated score margins are not reliable probabilities.

Preferred pair-first robustness uses the identical8975–9028eV supported window for raw/delta/shape controls, with perturbations before subtraction, shared-reference noise and independent noise. Thirty saved models and48 scenario arrays include training-only paired augmentation. Common acquisition gain cancels. Residual sample-only5% gain can severely corrupt shape inference (native micro0.6667→0.0737 without augmentation). A reference-only0.2eV registration error and1% pair noise also hurt. Augmentation trades clean shape F1 0.6667→0.5908 for noise F1 0.2816→0.4363 and residual-shift F1 0.5188→0.5714. This is a tradeoff, not uniform improvement. The ranges are illustrative because measured resolution/noise metadata for a labeled query were not supplied.

The initial finite-window/direct-signal stress artifacts are also retained with their explicit boundary convention; they must not be interpreted as global gain cancellation tests. Synthetic surface dilution0.1/0.25/0.5/1 is separate from actual coverage calibration. No new loading model or experimental coverage accuracy is claimed.

## Deliverables and next data

341 models are archived with hashes:311 original/fixed controls and30 pair-first robustness models. Held-out arrays, source/record/model hashes, transforms, signal floors, folds, threshold rules, all seeds, candidate selections and reload checks are saved. The PR receipt ties clean-checkout checks to the exact uploaded commit.

`proposed_additions.json` ranks independent rare/pure CHO and COCHO environments, matched CO+H/CHO/COH controls, composition at matched coverage, independent clean/covered parent families, supported CO/OCHO geometry/proximity sweeps and Cu100 loading/motif controls. Connectivity must be rechecked after relaxation. These are data/input recommendations; no calculations were submitted and no improvement from them is claimed.
