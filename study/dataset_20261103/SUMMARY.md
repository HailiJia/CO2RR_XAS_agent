# November 3 audit and retraining status

PR: https://github.com/HailiJia/CO2RR_XAS_agent/pull/18

Reviewed PR head: `5cbfa58151aabfc6daf6275e011087020e1e1e1e`.

## Source findings

- Captured 5,727 selected current source files, totaling 71,472,736 bytes, in a hash-verified archive.
- Checked 643 convolved spectra: all contain finite numeric values and strictly increasing energy grids. These are file-format checks, not full physical-provenance approval.
- All 2,164 files listed in the November 1 selected-source manifest exist locally and have identical SHA256 hashes. This comparison covers that manifest only; it does not establish that every current input is unchanged, that additional files are new configurations, or that the current training cohort remains exactly 141.
- Recorded SHA256 hashes for all 1,090 files in the latest `CO2RR_Agent_MS` package. No manuscript, SI, figure, or package file was edited.

## Existing numbers reviewed — not new retraining

| Existing analysis | Population | Result |
|---|---|---|
| Canonical identity, common training-selected threshold, mean of three seeds | 141 configurations / 135 geometry groups | micro F1 0.7992354 |
| Same identity control, seed 20261020 | 141 / 135 | micro F1 0.8126036 |
| Same identity control, seed 20261021 | 141 / 135 | micro F1 0.7890499 |
| Same identity control, seed 20261022 | 141 / 135 | micro F1 0.7960526 |
| Three-seed identity voting | 141 / 135 | micro F1 0.8099174; macro F1 0.6699631; exact-set accuracy 0.7222222 |
| Four-layer control, top-two average, three-seed voting | 110 / 104 | micro F1 0.7494553 |
| Four-layer control, fourth layer, three-seed voting | 110 / 104 | micro F1 0.3076923 |
| Au-associated CO fraction, CO-only spectral-ratio model | 28 / 28 | MAE 0.0716070 (7.16 percentage points); R2 0.9100516 |
| Au-associated CO fraction, all-CO spectral-ratio model | 50 / 49 | MAE 0.1016133 (10.16 percentage points); R2 0.5552505 |

The manuscript's headline 0.813 matches seed 20261020 rounded to three decimals; it is not the three-seed mean. The archived raw-layer Figure4c CSV differs from the latest differential four-layer analysis; use `depth_four_layer/training/depth_metrics.csv` for that analysis. The older supplemental CO-fraction study also differs from the current `local_CO_environment/regression_metrics.csv`. These distinctions are reported here without editing the manuscript.

PR #18's previous full-cohort nested model selection reported mean micro F1 0.7917 versus the canonical control's 0.7992. Its conditional paired interval included zero; the prior work did not establish an improvement.

## Blocker and remaining work

Fresh training has NOT run. This new Mac lacks a working system Python/Git installation and scientific Python libraries. An existing application-bundled Python 3.13.9 can run standard-library audits when configured, but has no NumPy, SciPy, pandas, scikit-learn, joblib or pip. Shell networking is explicitly disabled (`CODEX_SANDBOX_NETWORK_DISABLED`); GitHub and PyPI hostname resolution fail. User authorization to install is recorded, but cannot override the enforced network restriction.

A project-local setup script and pinned dependency list are supplied. Once shell networking is enabled, install a standalone Python with uv and run setup. Then rebuild the chemically audited dataset from current sources, preserve geometry-group isolation and declared seeds, fit models, verify model reloads and held-out predictions, and replace this status with new numbers. Original source-weight/convolution evidence and Cu100 lineage/reference gaps must be resolved or explicitly retained from verified pinned sources, not silently inferred from the selected archive.

Older versions remain pending validated replacement; deleting them now would remove provenance dependencies and leave no verified new trained model. No merge or history rewrite was performed.
