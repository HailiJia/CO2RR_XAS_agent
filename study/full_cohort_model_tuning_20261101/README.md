# Full-cohort model tuning

SUMMARY.md reports all 141 query records, controls, model changes and unsuccessful results. `metrics.csv` and `NUMBERS.csv` separate near-group and record metrics. No difficult-query filtering or abstention is used.

Run from repository root, using the recorded scientific environment:

```sh
python study/full_cohort_model_tuning_20261101/code/package.py --unpack
PYTHONPATH=. python study/full_cohort_model_tuning_20261101/code/verify.py
python -m pytest -q tests/test_full_cohort_tuning.py
```

Use **new empty directories** to repeat fitting:

```sh
python -m tools.audited_training --data-root study/dataset_20261101 --output-dir /tmp/new_full_cohort_cv --recipe full_cohort_tuning_v1
python -m tools.audited_training --data-root study/dataset_20261101 --output-dir /tmp/new_full_cohort_final --recipe full_cohort_final_v1
```

The final recipe fits each model to all 141 records after grouped training-only hyperparameter/threshold selection. Its selection scores are not independent test evidence. Three seeds are saved; none is selected using an outer score. Both explicit recipes are exposed in the agent/UI. New models use the existing normalized signed-difference inference route.

Base sample weight is inverse number of permitted views in a geometry group. Binary class balancing is computed from the resulting training class masses; each class receives equal total weight when both are present. ExtraTrees supports sample weights ([official documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)); the actual fitted dependency is scikit-learn 1.9.0, recorded in verification/environment.txt. Minimum leaf size still counts record views, and is labeled accordingly. Every test record is retained; all outer/inner query-related groups are blocked from their training partition.

Protocol/source/input hashes, candidate inner predictions, fold memberships, missing-positive supports, thresholds, reference identities and model replay evidence are retained. The lossless multipart archive contains evaluation models and final full-data models. It changes no source dataset or manuscript. Unsupported water/carbonate configurations remain explicitly inventoried in unsupported_inventory.json and in the original source audit, rather than assigned incorrect 11-label targets.
