# November data update

See SUMMARY.md and NUMBERS.csv for the results. The manuscript is unchanged.

Restore expanded versioned records and models (hash-checked archive parts):

```sh
python study/dataset_20261101/code/package.py --unpack
PYTHONPATH=. python study/dataset_20261101/code/verify.py
python -m pytest -q tests/test_november_update.py
```

`selected_sources.tar.gz` stores actual author inputs/spectra for added configurations and clean references, plus all available original author records. `selected_sources_manifest.json` verifies each member. It excludes potentials, binaries and wavefunctions. Existing source corrections and Cu100 reference are pinned to the previous study only when geometry/source evidence matches; see scale/source_weight_audit.json. `source_evidence.tar` inside the expanded artifact archive retains the complete current audit's selected convolved spectra, geometries, input decks and logs.

For full conversion/audit from the supplied export, construct an ignored local staging directory `local_export/` containing its top-level folders and the separately supplied `Cu-Ox_updated_structures/100` original ASE snapshots. Run `code/prepare.py`, `code/recover_top_layers.py`, `code/scale_pairs.py`, then `code/groups.py` in that order. Scale pairs reuses pinned valid reference data explicitly and rejects incomplete sources; do not invoke the older generic scale routine on the incomplete four-source clean Cu100 snapshot.

The actual shared-callable retraining command is in SUMMARY.md. Use a new empty output directory. All seeds, thresholds, folds, missing-positive training supports, unsuccessful logistic results and held-out predictions are saved. Independent parents and measured experimental performance remain untested.
