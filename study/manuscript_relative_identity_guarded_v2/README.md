# Reviewed manuscript and Figures 2–5

Revision based on draft PR18 at `798e2c81b7a9afa684dc2a5625607ec866430163`, 29 October 2026.

The paper uses normalized signed differential XAS as the intended experimental identity input. Absolute per-Cu absorption is not required. Absolute surface coverage has additional calibration requirements. Current-cohort results and historical controls are identified separately.

- `manuscript.pdf` and `manuscript.tex`: main paper.
- `Supporting_Information.pdf`, its TeX, `si/expanded_differential_study.tex`, and `references.tex`: full SI and bibliography.
- `figures/`: PNG-only figure images; Figure 1 remains unchanged.
- `figure_data/dataset_20261029/`: pinned current arrays, compact manifests, held-out predictions, protocol, audit tables and reviewed metrics.
- Older arrays remain historical controls in the other figure-data paths. They are not the latest training cohort.
- `REVIEW_AND_CHANGES.md`: scientific assessment, changes and limitations.
- `CODEX_HANDOFF.md`: integration instructions and prioritized follow-up analysis.

## Rebuild without fitting

With NumPy, pandas, SciPy, scikit-learn and Matplotlib installed, run from this extracted directory:

```
python code/review_expanded_results.py
python code/rebuild_requested_figures.py
python code/write_expanded_SI.py
python code/check_lattice_matching.py
python code/compare_fixed_layers.py
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error Supporting_Information.tex
pdflatex -interaction=nonstopmode -halt-on-error Supporting_Information.tex
```

These commands use supplied arrays and coordinates. They do not train models, change spectral widths or submit jobs. The Cu PBE benchmark alignment and latest timing are preserved. The optional `update_reference_alignment.py` reproduces the earlier specified first-feature correction and its scores. Archive scripts retain older cohort definitions and should not overwrite current outputs.

Current source-code snapshots under `figure_data/dataset_20261029/source_code` require the full repository. They document the inspected training/experimental-input implementation, rather than providing a standalone training installation. Full simulation source packages and fitted model binaries remain at the pinned GitHub commit; this ZIP is the complete manuscript/figure package, not a duplicate of the full simulation export.

This review independently recalculates 25 saved OOF files and 28 benchmark metric triplets. The full repository test suite and all saved-model reloads were not rerun here. The source report's 153 tests and 110 reloads are attributed to that repository verification. `build_receipt.json` records final local PDF and figure checks. No GitHub mutation, HPC execution or new electronic spectrum occurred in this review.
