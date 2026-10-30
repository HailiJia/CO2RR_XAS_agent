# Guarded differential-XAS identity: current evidence

**Normalized signed differences now support identity inference without an absolute per-Cu scale or surface fraction. Signed-shape normalization helps the joint-domain controls. Added-data improvement is inconclusive after protecting near-duplicate geometries. No experimental accuracy, absolute coverage or new-parent transfer is established.**

This version supersedes the pre-near-geometry interpretation in `../relative_identity_v2` while preserving all its files and predictions. Historical87/348 and October29 archives remain unchanged. The supplied revised manuscript is uploaded with Haili Jia, Pooja Basera, Dimosthenis Sokaras and Maria K. Y. Chan (corresponding), CNM/Argonne and SLAC affiliations. Its circular Figure1, oblique structure views and Figure3 arrays/alignment/broadening/similarity/CPU values are preserved. The supplied PNG lacked the stated explicit Cu100 count column; it is added from the unchanged saved inventory counts, and its axis now says Configuration records. The added main-text qualification and SI section distinguish the guarded study from older scopes.

## Input and scope

`experimental_pair(...,task="identity")`, the agent/UI recipe `relative_identity_v2` and `differential_route.infer` accept a documented precomputed normalized signed difference without requiring unavailable absolute raw spectra. Alternatively a compatible reaction/reference pair uses a shared reference-RMS divisor; common positive gain cancels. Sign, registration, reference chemistry, normalization, domain, processing and noise status are explicit. Independent arbitrary pair normalization is rejected. Signed shape is an identity-only control, never an absolute loading calibration.

The RMS transform follows a training-defined signal floor. Null/near-null inputs abstain; already unit-RMS inputs require pre-normalization SNR>=3 without requiring the raw pair. This is a heuristic rather than a measured detection limit. Unknown gain, active fraction or residual reference adsorbates limits interpretation to relative response/change. The simulation's short near-edge divisor is not a validated experimental full edge jump. Coverage retains its separate calibrated contract.

## Why the near-geometry guard matters

The audit compares137 same-composition/connectivity/cell pairs by element-aware Hungarian atom correspondence, using in-plane minimum images and no vacuum wrapping. Six pairs satisfy RMS<=0.02A and maximum displacement<=0.2A; three also satisfy maximum<=0.05A. Two H-labelled0p75 cases are nearly identical to0p5 cases, both with18 actual H atoms. A CHO-containing mixture pair also crosses inherited folds. Folder names neither establish chemistry nor independent data.

The130 identity records span124 conservative near-geometry groups:118 CuAu records span112 groups;12 Cu100 records span12. The old99 query records span96 groups. Thus19 added CuAu records contribute16 groups. These are geometry groups, not124 independent clean-parent families. CHO support is9 positive records/8 groups and2 pure cases; H30/27; CO44/43; CO2 5/4; COOH9/8; OCCOH5/4. COCHO remains1. Molecular counts and layer spectra are not independent examples.

To keep the requested outer query folds unchanged, any group represented among an outer fold's queries is completely excluded from that fold's training. Training retains one canonical available member per remaining group, preferring an older audited record. Inner validation has the same protection; inherited old-training assignments are retained. Every saved model records train/test families. Repeated query views are reported as records but get one total group weight in the primary comparisons and bootstrap. No target labels, source weighting, energy grid or query membership changed.

## Representation control, identical cases and recipes

Fixed Combined, threshold0.5, regularized independent logistic C1 or independent balanced64-tree leaf1 anchor. These are equal-near-group-weight micro F1 values, distinct from the saved record-weighted scores:

| Input | CuAu logistic | CuAu trees | Joint logistic | Joint trees |
|---|---:|---:|---:|---:|
| Native-normalized raw | 0.6266 | 0.7123 | 0.5611 | 0.7167 |
| Corrected-normalized raw | 0.6319 | 0.7045 | 0.5457 | 0.7155 |
| Corrected common-scale raw | 0.6344 | 0.7153 | 0.5671 | 0.6989 |
| Signed relative delta | 0.6344 | 0.7153 | 0.5788 | 0.6940 |
| Guarded signed shape | 0.6647 | 0.7500 | 0.6500 | 0.7602 |

Exact values, macro F1, exact sets, per-label precision/recall/support, single/mixed, coverage, contact/proximity and domain metrics are in the CSV/NPZ files. Within CuAu, common-scale raw and delta give identical centered-logistic predictions: subtracting one constant reference offset cannot add information after training centering. Source-weight correction is an evidence correction, not a guaranteed F1 gain.

Signed shape versus relative delta improves joint logistic micro F1 by0.0712, conditional95% group-bootstrap interval[0.0323,0.1094]; trees improve0.0662[0.0231,0.1171]. CuAu-only differences0.0302 and0.0347 have intervals including zero. All intervals condition on fixed predictions, with no multiplicity adjustment or prospective confirmation. The separate small nested feature comparison uses the same signed/DCT16/Combined budget per estimator and training-only common thresholds; its best-looking outer score is not selected as a replacement.

## Added-data control and learning curves

Relative-delta Combined/tree anchor, three predeclared seeds, identical old99 query records (96 groups), unchanged inner assignments for retained old training records:

| Threshold | Old-training mean F1 | Expanded-training mean on same queries | Paired difference95% interval |
|---|---:|---:|---|
| Fixed0.5 |0.7130|0.7252|+0.0121[-0.0234,0.0495]|
| Transferred old-training common threshold |0.7191|0.7295|+0.0104[-0.0190,0.0425]|
| Training-only restrained common threshold |0.7191|0.7311|+0.0120[-0.0157,0.0415]|

The common threshold searches only0.3/0.4/0.5/0.6 with a penalty toward0.5. Rare labels do not receive unrestricted individual tuning. Bootstrap resampling is by geometry group, jointly across fixed seed predictions; refit/seed difference SD is separately0.0151/0.0148/0.0297. Full118 scores are not paired old99 improvement. Grouped learning curves retain all seeds and query membership, with label-blind training subsets and saved actual training-group counts.

Geometry-cutoff sensitivity is consequential. Grouping only the three pairs below maximum0.05A gives fixed0.5 F1 0.7007→0.7431 on98 old query groups, with conditional difference interval[0.0061,0.0815]. The conservative six-pair result is weaker and inconclusive. Both settings were defined from geometry, not selected by outer scores. The earlier unguarded positive interval must not be treated as independent evidence.

## Chemical, domain and robustness limits

Missing-positive training labels, exact mixture recovery/false positives, omitted CO/CHO/COH and nearest actual training spectra are saved. Nine CHO records do not provide nine independent environments, and two pure cases do not establish broad pure-CHO inference. Oversampling cannot repair that diversity gap. Contact compositions and interface proximity are separate audit targets; no absent CuAu2/Au3 hollow support is invented. Unsupported water/carbonate queries stay review-only; dissociated OCCO/water remains CO plus water.

Cu100 supplies12 supported identity cases and four CO-loading cases. Per-domain common-four and full-eleven metrics, plus entire-domain-held-out stresses, remain separate. One clean-reference family per domain prevents leave-parent-out estimates. No pooled-R2 facet-transfer or experimental coverage claim is made.

Only one real held-out clean negative is available; specificity cannot be established from it. Null/near-null synthetic tests, clean false positives, abstention/retained-query fractions and full-query errors remain explicit. Classifier scores and margin curves are uncalibrated.

Pair-first robustness perturbs reaction/reference arrays before subtraction on the identical supported8975–9028eV window. Common acquisition gain and its shared divisor cancel; residual sample gain, residual registration error and pair noise do not. Shared-reference and independent noise are separate components. Training-only paired augmentation trades clean performance against some stressed scenarios; no favorable setting is selected as proof of improvement. Direct-signal finite-window experiments remain a secondary, explicitly limited scenario. Illustrative dilution fractions0.1/0.25/0.5/1 do not determine a physical active fraction or coverage.

## Delivered evidence

416 new models:311 main/fixed/learning controls,30 pair-first robustness models,30 strict-cutoff sensitivities and45 genuinely group-sampled learning models, all with held-out reload evidence and hashes. The341 pre-guard models are preserved and numerically replayed with their saved as-run fitting-source hashes. Every fold, transformation, threshold, target, source/reference/model hash, seed, unsuccessful comparison and scientific limitation remains inspectable.

The shared agent/UI callable now defaults to the guarded recipe. A sanitized trace demonstrates actual held-out simulated inference through the shared route without an absolute scale or raw experimental pair. It is not a production scheduler or measured-spectrum demonstration. Proposed independent additions prioritize pure/rare chemistry, matched CO+H/CHO/COH, mixture composition at matched loading, independent clean/covered parents, supported CO/OCHO geometry/proximity and Cu100 loading/motif sweeps. No HPC jobs, portal writes, merge or force-push occurred.

The paired synthetic broadening adds Gaussian sigma0.25eV (FWHM0.5887eV) while retaining native lifetime broadening. It is a stress scenario, not a fitted physical remainder or measured instrument uncertainty.

A separate shared-reference residual-noise test reuses one noisy clean-reference vector across every query in each outer fold, rather than silently treating reused-reference errors as independent. Six extra scenario arrays and known-injected-noise abstention fractions are saved. Common incident noise cancels separately; residual shared reference noise does not. No model was tuned to these tests.
