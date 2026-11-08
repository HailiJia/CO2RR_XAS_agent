# November 3 source refresh

Status: source upload and audit completed locally; fresh training is blocked by the local environment. Existing scores are not new retraining results.

Concatenate the ordered parts listed in `archive_manifest.json`, then extract the resulting `latest_sources.tar.xz` into an empty directory. Paths inside the archive match the author export. The archive contains 5,727 selected files (71,472,736 uncompressed bytes): convolved spectra, FDMNES input decks, absorber tables, available geometries, supplied records and source metadata. Raw unconvolved per-absorber curves, full electronic-structure logs, wavefunctions and executables are excluded. This capture alone does not establish chemical eligibility, full absorber-population validity or convergence.

`audit_evidence.tar.xz` contains per-file SHA256 manifests, comparisons with the November 1 source manifest, spectrum checks and copies of the current manuscript's numerical CSV evidence. These CSVs are existing results, not a new fit. `manuscript_unchanged_manifest.json` inside it records all 1,090 local manuscript-package files before any work.

The 26 data parts must be concatenated in filename order. Verify the concatenated archive SHA256 and every member against the manifest before using it. The read-only capture script `audit_and_package.py` documents the selection rules. It requires the previous source manifest from the evidence archive and Python standard-library modules only.

No previous study should be interpreted as superseded by a newly trained model from this upload. No new model was fitted. Historical versions have not been deleted because replacement training/validation has not completed and existing code depends on their provenance files.
