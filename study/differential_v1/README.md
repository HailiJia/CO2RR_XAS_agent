# Differential XAS study v1

This is a new study; archived cohorts, predictions and source ISAAC records remain unchanged. Sources (including unfinished checkpoints, actual per-site electronic spectra and current VASP outputs) are under `sources/`, with member and archive SHA-256. Checkpoint identities use current CONTCAR, not the intended seed.

## Reproduce

From repository root, use the recorded scientific dependencies (see `ml/protocol.json` and benchmark executed notebook):

```sh
python study/code/package_differential_artifacts.py --unpack
python study/code/verify_differential_artifacts.py
python -m pytest -q tests/test_differential_xas.py tests/test_reference_v3.py
python study/code/differential_publication_figures.py
```

The lossless `artifacts/` archives contain 272 saved models, all inner selection arrays, robustness predictions and reconstruction logs/arrays. Their manifest maps each member back to its original relative path, size and SHA-256. Unpacking verifies every byte and refuses to replace changed files. These are derived outputs, separate from immutable source bytes. Training a new study requires a new empty output directory through `tools.audited_training.run_audited_training(..., recipe='differential_v1')`. The agent and UI call this same route. Model inference in `tools.differential_route.infer` requires explicit compatible paired inputs; there is no borrowed cross-facet reference.

## Findings and limits

* 232/232 native layer convolutions numerically reconstructed. Inverse recorded symmetry multiplicities correct individual sources already covering all physical atoms; clean-layer sums 51/52 are not 36-site populations. The correction is energy dependent.
* 34 primary initial-metal-model-matched Cu–Au configurations; 64 weaker same-substrate pairs remain a sensitivity set. Initial-coordinate equality alone is not proven historical lineage. One shared clean-model family.
* Identical physical raw / Δμ / [clean, Δμ] predictions: primary identity micro F1 0.7941, exact-set 0.6765; seven CO-only coverage cases MAE 0.006545 CO/all surface metal sites. Subtraction adds no information beyond physically scaled raw for this fixed-reference affine control. Conditional intervals do not include refitting/parent variation.
* No completed compatible clean Cu100 spectrum was located in the current supplied roots. Joint differential/facet transfer is not claimed. Proposed inputs are in `ready_inputs/`; no jobs submitted. Raw Cu100 controls are explicitly separate.
* Synthetic paired robustness/dilution is not experimental validation. Surface fraction, detection and paired calibration conventions must be independently specified for experimental queries.
* Three actual OCCO–water CONTCAR checkpoints are dissociated 2CO and unfinished; 15 seed-only alternatives are not relaxed evidence or ML examples.

The actual updated editable paper and compiled main/SI PDFs are in `../manuscript_differential_v1`; current numeric reference processing and executed notebook are in `../reference_benchmark_v3`.
