# Reproduce and inspect

Run from the repository root with the dependency versions in `verification/environment.txt`.

```sh
python study/relative_identity_guarded_v2/code/package.py --unpack
python study/signed_shape_detectability_v3/code/package_models.py --unpack
PYTHONPATH=. python study/signed_shape_detectability_v3/code/verify_artifacts.py
PYTHONPATH=. python study/signed_shape_detectability_v3/code/check_simulated_predictions.py
sh study/signed_shape_detectability_v3/verification/focused_command.sh
```

Saved-prediction analysis and PNGs (no fitting):

```sh
PYTHONPATH=. python study/signed_shape_detectability_v3/code/diagnose_existing.py
PYTHONPATH=. python study/signed_shape_detectability_v3/code/summarize_size.py
PYTHONPATH=. python study/signed_shape_detectability_v3/code/robustness.py
PYTHONPATH=. python study/signed_shape_detectability_v3/code/finish_analysis.py
```

The bounded label-dependence script refits only inner estimators; its declared protocol and unsuccessful results are preserved separately. To rerun the new training experiment, use a **new empty directory**, not the frozen outputs:

```sh
python -m tools.audited_training --data-root study/dataset_20261029 --output-dir /tmp/new_signed_shape_size_run --recipe signed_shape_size_v3
```

The shared API validates input-array/manifest hashes and geometry-audit correspondence. The new explicit recipe is available through the agent/UI. `verification/effective_inner_splits.json` describes actual canonical fit subsets; `ml/splits.json` also retains the splitter's pre-canonical candidates for traceability. Nothing in these commands edits the manuscript. The separate user-authorized scoped publication correction and editable ZIP contain only author/affiliation and Figure2b changes.
