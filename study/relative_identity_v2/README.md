# Relative signed differential identity v2

A new exploratory study on the frozen October29 source/record audit. The historical87/348 benchmark, October29 study, reference benchmark processing and manuscript Figure3 are unchanged. No prospective or measured experimental accuracy is established.

## Reproduce

From the repository root, with the scientific dependencies recorded in `results/protocol.json`:

```bash
python -m tools.audited_training --data-root study/dataset_20261029 --output-dir /tmp/new_relative_identity_fit --recipe relative_identity_v2
```

`results` is the saved actual run. To regenerate its separately fixed Combined/.5 controls and diagnostics, use a disposable checkout and run:

```bash
PYTHONPATH=. python study/relative_identity_v2/code/fixed_controls.py
PYTHONPATH=. python study/relative_identity_v2/code/diagnostics.py
PYTHONPATH=. python study/relative_identity_v2/code/finalize.py
```

The latter scripts target this versioned study path. They do not write source records or archived studies. The multipart logical `models.tar.gz` archive restores to `results/models`; every member is listed and hashed. The clean-checkout receipt in the PR names the exact tested commit. All source records and raw/convolution inputs remain in the uploaded October29/pinned source packages.

## Input contract

The UI and `tools.differential_route.infer` dispatch identity-v2 models to the same `tools.differential_identity.identity_input` contract used by `experimental_pair(...,task="identity")`. See `input_contract_examples.json`. A documented precomputed signed difference needs no raw pair, absolute per-Cu calibration or surface fraction. A relative pair uses one reference-RMS divisor over the documented supported interval; a positive common gain cancels. Independent arbitrary normalization of the pair is rejected. Sign is reaction minus reference. Metadata must state energy registration, reference chemistry, processing, domain and noise (including explicit unknown noise).

Signed-shape models divide signed differences by RMS **only after** a training-defined signal floor. Null/near-null queries abstain. A pre-unit-RMS query requires a documented pre-normalization SNR>=3 because its original signal strength was discarded; this can be a supplied scalar and needs no raw absolute spectra. Sparse/noisy signals and unknown morphology/reference changes remain review cases. The floor is a heuristic, not an experimentally validated detection limit. Relative-amplitude models require their trained near-edge divisor; arbitrary positive normalization is supported through shape models, not silently substituted into amplitude-sensitive models.

The simulation divisor uses short8974–8977/9018–9025eV windows: it is a near-edge contrast scale, **not a validated full experimental edge jump**. Absolute loading requires a separate gain/active-fraction/coverage calibration. Neither identity normalization nor surface subtraction supplies it. The legacy calibrated-coverage route retains its stricter requirements; identity is separate.

## Protocol and limits

Same physical top-two populations,130 unique geometry groups, energy grid, labels and original query folds. Four source/normalization controls plus signed shape. `fixed_representation_metrics.csv` uses exactly Combined, fixedC1 logistic or balanced64-tree leaf1 anchor and threshold.5 across all representations. The nested comparison selects only signed/DCT16/Combined inside training, separately for each estimator family with equal budgets. A common threshold in{.3,.4,.5,.6}, penalized toward.5, replaces unsupported per-label tuning. All feature scaling, signal floors and selections use training partitions.

Relative-delta old99 versus expanded118 uses all three fixed seeds, identical old99 queries and paired original inner assignments. Threshold.5, transferred old-training common threshold and newly fitted common thresholds are all retained. Grouped learning curves use label-blind nested subsets and fixed.5; saved membership includes missing-positive training labels. Bootstrap intervals condition on fixed predictions; seed/refit SD is separately reported. These reused folds are not independent confirmation.

Synthetic perturbations are scenarios, not measured instrument uncertainty. Training augmentation is one grouped copy with±.2eV shift, Gaussian sigma.25eV,±5% gain,1% noise. Stress sensitivity perturbs the supported interior and explicitly keeps native1eV boundary margins; it is **not** a complete full-window instrumental shift/broadening measurement. Noise includes a shared reference component and independent components. The residual5% gain is a processed-scale uncertainty, not a claimed failure of common gain cancellation. Dilution fractions.1/.25/.5/1 are illustrative, not inferred particle fractions. All clean and perturbed query outputs are retained.

One reference family per domain prevents parent-held-out inference claims. Cu100 has12 supported identity and four CO-loading configurations; unsupported water/carbonate queries stay review-only. Geometry descriptors stratify errors but never enter spectral predictors. Contact counts are not automatically hollow labels; proximity to the interface is a separate descriptor. Dissociated OCCO/water remains CO plus water.

## Restore and verify saved models

```bash
python study/relative_identity_v2/code/package.py --unpack
PYTHONPATH=. python study/relative_identity_v2/code/finalize.py
```

`all_model_replay.json` replays all311 original/fixed-control models against their saved held-out scores. A further30 pair-first robustness models and48 held-out scenario arrays are in `paired_robustness`; this is the preferred noise/gain/shift sensitivity. It trains and queries on the same explicit8975–9028eV interior, with actual paired perturbation before subtraction. Common acquisition gain and its divisor cancel; residual sample-only gain is separate and does not cancel. Its matched additional broadening retains native lifetime treatment. The earlier finite-window stress artifacts remain available with their original boundary convention and direct-signal augmentation; they are not global-gain or full measurement models. Neither scenario set is measured experimental validation.
