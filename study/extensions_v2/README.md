# Scientific extensions v2

The frozen 87-configuration / 348-spectrum benchmark, its primary predictions, and reference-v1 remain unchanged. This directory contains new audits and exploratory evaluations of the same configurations; none is independent confirmation.

- `chemical_audit/`: initial → relaxed → actual XAS-copy/deck → record/target lineage, per-molecule distances/graphs, hashes, cutoff sensitivity and H-visible local views. No retained CHO/COH error was demonstrated. Five retained CHO positives comprise two CHO-only configurations and three mixtures; all six COH cases have O–H connectivity despite CHO basenames. Individual molecule counts are separate from configuration counts.
- `controlled_comparisons/`: expanded quartet recovery, clean-parent search, seven rejected equal-formula hydrogenation pairs, saved-run-log search, exact missing artifacts and unsubmitted prepared controls. The completed terrace 2CO record is recovered but differs from the pinned initial version. No valid controlled spectral pair or clean subtraction was established. Not located does not mean never run.
- `design_reconciled/`: evidence-derived replacement of the earlier exact-hash-only lookup. Equivalent duplicate relaxed copies count as one observed outcome.
- `ml/`: declared nested recipes, all held-out/stress/ablation arrays, 40 hashed/reloaded fold models, nearest training examples, per-species and single/mixed results, conditional paired intervals, harder-coordinate grouping, learning curves and review policy. `invocation_trace.json` and `heldout_review_demo.json` document the actual local agent training and held-out review. This uses saved spectra and no scheduler/portal calls.
- `input_package.json`: complete authoritative ZIP inventory and SHA-256.

Reproduce the read-only audits from the repository:

```sh
python -m tools.chemical_lineage --data-root /path/to/export --output-dir /new/audit
python -m tools.controlled_comparisons --data-root /path/to/export --output-dir /new/controls
python study/code/plot_chemical_audit_v2.py
```

The fit uses the shared agent/UI API, an empty new output directory and audited cohort loading:

```sh
python -m tools.audited_training --data-root /path/to/export --output-dir /new/fit --recipe scientific_extensions_v2
```

Required packages include numpy, scipy, pandas, scikit-learn, joblib, matplotlib, ase/pymatgen, and xgboost; exact executed versions are in `ml/protocol.json`. No binary dependencies are vendored. The new fit does not overwrite archived results. `tools.audited_training.review_extension_model` exposes the same saved-model review callable for agent/UI consumers; only load trusted model artifacts.

The nearest-neighbor mixture gains have costs: H false positives increase 1→4 and CO 0→2; single-species CO detections fall 21/21→19/21. No new recipe improves overall micro F1. Synthetic stress settings are scenarios, not measured experimental uncertainties. The 56 coordinate groups are not verified independent ancestral parents. Unsupported sparse-label calibration and domain flags remain visible. Scientific controls and existing-case disagreement reviews are separately ranked; no proposed calculation is credited with improved performance.
