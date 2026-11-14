# Scientific review and manuscript revision — 29 October 2026

Reviewed source: draft [PR18](https://github.com/HailiJia/CO2RR_XAS_agent/pull/18), commit `798e2c81b7a9afa684dc2a5625607ec866430163`. The PR was open, draft and unmerged when inspected. This package is prepared for integration; this review made no GitHub writes, HPC submissions or new model fits.

## What the new results actually show

The controlled fixed-top-two Combined experiment gives a modest gain on identical old99 queries: mean micro F1 **0.7434 → 0.7519**, macro **0.5795 → 0.5812**, and exact-set **0.5993 → 0.6229**. Two of three seeds improve micro F1 and one declines. Each conditional paired retuned micro-F1 interval crosses zero. The full118 score has a different query set and cannot be used as a paired improvement claim. Earlier automatic-depth, frozen87,90 and matched34 scores are different experiments.

CO-only loading MAE on the same25 queries falls **0.013971 → 0.011444**, an approximately18% point reduction. Its paired conditional difference interval, **[-0.006057,+0.000344]**, includes no improvement. The full28 result changes the queries. These intervals resample fitted predictions and exclude model-refit/new-parent uncertainty.

Transferring baseline training-derived thresholds gives mean same99 micro F1 **0.7608**, higher than the retuned point estimate. Threshold estimation therefore affects the apparent gain. This diagnostic cannot justify choosing thresholds against outer-test results.

## Where the gaps are

1. **The controlled learning-size test still uses native-normalized raw XAS.** The differential study is a separate six-candidate single-seed analysis. A same-fold, fixed-recipe differential learning curve remains necessary.
2. **Weighting and normalization are confounded.** Native-normalized raw starts from native convolutions, while corrected raw and delta use reconstructed source weights. Corrected-and-normalized raw is the missing control. Unit-area shapes differ detectably for8/118 CuAu and6/12 Cu100 cases, with maximum integrated absolute differences0.00185/0.00790. Most differences are small; their causal contribution to scores is not established.
3. **New data do not fill the rare-label gaps.** CO grows37→44, H25→30, OH6→12 and COH7→10. OCHO remains8, COCHO1, OCCOH5, COOH9 and O7. CHO grows7→9, but only2 remain pure CHO; the other7 contain CO/H. When the sole COCHO positive is held out, that fold has no positive training example. Its zero F1 is not evidence of intrinsic spectral indistinguishability.
4. **Mixtures remain poorly resolved.** CuAu-only delta gives micro F1.767 on101 single-species cases versus.586 on16 mixtures, with1/16 exact sets. Joint delta improves the corresponding mixture result to.633 and2/16 exact sets. Additional common-species single configurations cannot alone resolve chemical interference.
5. **Cu100 is a small conditional domain.** Joint delta gives common-four-label micro F1.817 on118 CuAu queries and.476 on12 Cu100 queries. Its4 CO loading cases confound loading with motif. One clean family per domain cannot establish independent parent/facet/experimental transfer. Pooled R² can be inflated by domain loading differences.
6. **Null-signal behavior needs validation.** The CuAu-only delta model assigns COOH/OCCOH to the one clean negative; the joint delta model correctly outputs the empty set. This is a model-specific one-case diagnostic, not an estimated false-positive rate. Very weak differences should not be forced into an identity prediction or unit-normalized into apparent signal.
7. **The inspected experimental adapter has an inappropriate identity gate.** `tools/differential_xas.py::experimental_pair` requires `Cu_basis == same independently calibrated per-Cu basis` and `surface_fraction` before returning a pair. Normalized signed delta identity does not require either absolute quantity. This is an experimental deployment-contract issue, not a cause of the simulated F1. The shared agent/UI wiring should be audited before claiming every route uses this gate.

## Experimental interpretation incorporated into the paper

The intended input is normalized signed relative delta, or a compatible pair from which it is formed. A shared multiplicative factor cancels in `(c*mu_ads-c*mu_ref)/(c*S_ref)`. Simulation-side per-absorber reconstruction makes site/layer contributions comparable; it does not require experimental absolute per-Cu absorption. Energy registration, normalization convention, reference chemistry and noise still need compatibility checks. Identity, relative change and calibrated absolute coverage are distinct targets. Unknown active fraction/gain specifically limits quantitative coverage and may also shift the identity model's input distribution; measured accuracy is not claimed.

The current simulation's9018–9025 minus8974–8977 clean-window scale is a near-edge proxy on a short8974–9029 grid. It is not a validated experimental full edge jump. Normalization-range limitations are supported by Weng et al., JSR2005, DOI10.1107/S0909049504034193, and Bugarin et al., JSR2024, DOI10.1107/S1600577524005307. The latter evaluates HERFD spectra and does not validate this Cu simulation window.

## Manuscript and figure changes

- Revised abstract, introduction study outline, results and conclusions around relative differential inputs and the current evidence. Kept the catalytic-hypothesis motivation, authors, affiliation, NERSC capabilities, automatic lattice matching, fixed-layer sensitivity and DCT definitions.
- Figure1 is unchanged. Figure2 retains the preferred structure rendering and gray/blue nested bars, updated to135 adsorbed configurations:118 CuAu and17 Cu100. Figure4b uses118 closed-label CuAu cases (117 adsorbed plus the clean negative), so its118 membership differs from Figure2's118 adsorbed cases.
- Figure3 keeps the previously requested rigid CuPBE first-small-peak alignment at8982eV, fixed broadening, recomputed similarity metrics, latest XAS-stage CPU values and external differential-layer legend. CuO fdm is129600CPU s. Cu2O PBE/GW remain included; incomplete CuOPBE remains explicitly blank.
- Figure4 retains the87-case fixed-depth/feature controls; its current differential per-species panel uses118 cases. The actual-clean heat map now uses33 matched singles from37 matched adsorbed cases, with relative-delta labels.
- Figure5 separates representation controls, domain-resolved identity, differential CO loading and the raw training-size control. No automatic depth selection or per-Cu experimental requirement is introduced.
- SI preserves older benchmark/paired studies as historical scopes and appends current record definitions, normalization, model protocols, support, complete scores and paired controls. The cleanCu100 reference is now reconstructed rather than missing. The newer OCCO/water spectral case is2CO+2H2O, not an intactOCCO positive; the earlier unfinished checkpoints remain historical.

## Verification and limits

The review script checks array/manifest/fold and label correspondence for25 saved prediction files, recomputes their scores, verifies signed subtraction and common-scale cancellation, and recomputes28 benchmark metric triplets (maximum numerical discrepancy5.6e-16). Corrected raw/delta identity predictions on CuAu are identical; coverage predictions agree within1e-12. Figure2/4 memberships and support were reconciled. Source audit CSVs accompany the package; the complete553-series source restoration and full geometry calculations were not rerun here.

Both final PDFs are compiled twice and visually inspected; final page counts and hashes are in `build_receipt.json`. Figure images are PNG only and plotted arrays plus code are included. Source training used Python3.11.9/NumPy2.4.6/scikit-learn1.9.0; this editing environment independently checks predictions without reloading that full model export. Source-reported153 tests and110 replays are not represented as new local runs.
