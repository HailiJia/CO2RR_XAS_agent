# CO2RR_1029: records and model refresh

This version updates scientific data, records, code and analysis summaries only. Manuscript/SI and archived studies are unchanged. No merge, force-push, HPC submission or portal write.

`SUMMARY.md` gives the comparison and limitations. `record_corrections.csv/json`, `record_series_validation.csv/json` and `record_geometry_audit.csv/json` retain supplied-label versus observed-graph checks, exact structure correspondence, source hashes and exclusions. Original author record bytes remain in source archives. Corrected full available-layer records are in `records`; independently source-scaled top-two records are in `scale/corrected_records`.

## Reproduce checks and training

From repository root with the recorded Python/scientific dependencies:

```sh
python study/code/package_refresh_artifacts.py --unpack
python study/code/verify_dataset_20261029.py
python -m pytest -q tests/test_dataset_refresh.py
python -m tools.dataset_refresh_sources --output /tmp/CO2RR_1029_sources
python study/code/check_refresh_record_sources.py /tmp/CO2RR_1029_sources
```

Sources are exact bytes in new archives or pinned reused archives. The source restoration verifies every member. Supplementary partial-run logs are split into practical parts, reassembled and verified automatically. Original Cu100 ASE snapshots are independently restored from their already-uploaded archive. No licensed potentials, executable binaries or wavefunction scratch are included.

Actual retraining used the shared agent/UI callable:

```python
from tools.audited_training import run_audited_training
run_audited_training('study/dataset_20261029', '/tmp/new_refresh_fit',
                     recipe='dataset_refresh_v1')
```

Use a new empty fit directory. This recipe requires the prepared, converted, geometry-audited/source-scaled snapshot, not an arbitrary raw folder. The agent defaults updated-data requests to this fixed-top-two recipe; the old automatic-depth v3 route requires an explicit historical recipe request. Full recipe, dependencies, code hashes, candidate selection, thresholds, fold/group assignments, missing-positive supports, models and all held-out arrays are saved in `ml`. Frozen old99 folds are retained; additional configurations receive deterministic geometry-hash folds. Counts, graph labels, path/site labels and true loading never enter spectral predictors.

Preparation used both the author four-layer shell converter and generic Python converter, followed by graph/CONTCAR/ASE audits. The reusable functions are `tools.coverage_study.convert/audit`, `tools.dataset_refresh.fix_records/scale`; pass `extended_graph=True` when auditing water/carbonate. The one top-two recovery and incomplete-layer provenance checks are saved as explicit code/evidence. Local convolution of saved electronic spectra uses an externally supplied FDMNES executable, never a new electronic/HPC calculation.

`artifacts` contains lossless saved model and reconstruction outputs, with member hashes and a safe verified unpacker. Signed differences preserve amplitude and both signs, use substrate-specific clean references and shared clean-derived scales, and do not independently align pair members. Clean Cu100 is locally convolved from actual supplied raw sources and exact layer CONTCAR, with termination limitations recorded. A numeric-only CO/uniform2 candidate remains excluded for unresolved full-run lineage. Partial or incompatible records are not silently promoted to complete training examples.
