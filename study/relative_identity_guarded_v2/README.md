# Guarded relative differential identity study

This is the current controlled study. `../relative_identity_v2` remains the archived pre-near-geometry run at commit67a9437. Its positive training-size interval is not the current guarded conclusion. Source records, targets, grid and inherited outer query folds remain unchanged. Six conservative geometry-near pairs form124 groups from130 records; one canonical training member per group is retained and every query-related group is excluded from outer and inner training. This is not124 independent clean-parent families.

Run the shared agent/UI training recipe from the repository root:

```bash
python -m tools.audited_training --data-root study/dataset_20261029 --output-dir /tmp/new_guarded_fit --recipe relative_identity_v2
```

The geometry/source manifest hashes must match the audited export. Original old-training inner assignments are preserved for retained records. The guarded study contains the same fixed/nested representations, three-seed old99/expanded118 comparison, grouped learning curve, domain-held-out and synthetic scenarios. `near_group_*` tables give equal group weights and grouped conditional bootstrap intervals; record-weighted tables are retained separately. The stricter geometry cutoff is an explicit sensitivity, not a selected outer-score winner.

Additional controls were run through these saved scripts (their outputs target this versioned path):

```bash
PYTHONPATH=. python study/relative_identity_guarded_v2/code/fixed_controls.py
PYTHONPATH=. python study/relative_identity_guarded_v2/code/paired_robustness.py
PYTHONPATH=. python study/relative_identity_guarded_v2/code/strict_sensitivity.py
PYTHONPATH=. python study/relative_identity_guarded_v2/code/cluster_intervals.py
```

For a clean checkout, restore the lossless multipart model archive and replay without fitting:

```bash
python study/relative_identity_guarded_v2/code/package.py --unpack
PYTHONPATH=. python study/relative_identity_guarded_v2/code/verify.py
```

416 new saved models include311 main/control models,30 pair-first robustness models,30 strict-cutoff sensitivities and45 genuinely group-sampled learning models. The original341 pre-guard models remain archived unchanged. To replay that archived scope under current prediction code, restore its own archive and use `code/verify_archived_original.py`; original training-source hashes are checked against its full saved as-run source snapshots rather than pretending the current guarded fitting code is identical.

The relative identity contract and examples remain in `../relative_identity_v2/input_contract_examples.json`: precomputed signed difference or compatible relative pair; no absolute per-Cu scale or surface fraction for identity. Signed shape requires signal-floor handling. Already unit-RMS inputs need a pre-normalization SNR declaration. Absolute coverage stays separately calibrated. Unknown morphology/reference chemistry and sparse-label confidence require review. Actual measured experimental accuracy is unvalidated.

Every layer/view belongs to its physical configuration and near-geometry group. Grouped bootstrap intervals condition on fixed predictions, not refitting or prospective validation. One clean reference family/domain prevents unseen-parent validation; Cu100 has12 supported-label cases and four CO-loading cases. The unsupported water/carbonate queries remain review-only. Contact counts and interface distances are audit/reporting variables, never spectral predictors.

Primary learning curves are in `group_learning`: canonical near-groups are formed and protected before label-blind subset sampling. The earlier record-subset-then-dedup curves remain a secondary control in `results`. The shared callable adds the primary group-sampled curve automatically inside a requested output directory; the stored study uses its explicit sibling `group_learning` folder.

The paired synthetic broadening adds Gaussian sigma0.25eV (FWHM0.5887eV) while retaining native lifetime broadening. It is a stress scenario, not a fitted physical remainder or measured instrument uncertainty.

`paired_robustness/shared_reference_noise_*` adds a residual noisy reference vector reused across all queries in a fold. This is distinct from common incident noise that cancels, and from separately measured noisy references per query. Six additional scenarios reuse the fixed models; the review table reports all-query errors and retained fractions under a known-injected-noise3-sigma policy. This is synthetic, not measured SNR.

The preferred score review table is `review_policy_by_decision_threshold.csv`: distance is to each actual inner-training decision threshold, with all-query errors, retained denominators and false-positive costs. The earlier0.5-score-distance table remains a supplementary diagnostic. Neither is a calibrated posterior probability; sparse classes have no reliable calibration support.
