# Current PR18 integration

The revised sources and designs in this directory came from the supplied extracted package. Numeric Cu/Cu2O/CuO data now replace the graphical CuO display-path comparison. Both paper and SI sources/PDFs are present and compiled twice; landscape SI grids preserve readability. Processing code, inputs, original and executed notebook, provenance, parameters, metrics and exact arrays are in `../reference_benchmark/`. See `../EVIDENCE.md` and build receipts for current evidence. The historical package handoff/review notes below describe earlier states; they do not supersede these new outputs.

# Digital Discovery manuscript package

`manuscript.pdf` is the revised draft by Haili Jia and Maria K. Y. Chan. `Supporting_Information.pdf` contains detailed computational, chemical-label and statistical methods. Both have editable LaTeX sources. Figures appear near their first substantive discussion.

- `figures/`: five redesigned figures as vector PDF/SVG and 450-dpi PNG.
- `figure_data/`: plotted tables, held-out arrays, layer spectra, starting structures and molecular templates.
- `code/make_figures.py`: figure-generation entry point.
- `code/make_publication_figures.py`: current shaded-atom renderer and plotting implementation.
- `code/prepare_figure_data.py`: original extraction from the pinned repository.
- `code/verify_package.py`: prediction and cohort verification.
- `references.tex`, `si/references.tex`: references in citation order for each document.
- `si/per_species_table.tex`: full per-species primary results.
- `CO2RR_PR18_Review.md`: latest PR review, independently run checks and committed-source blockers.
- `CO2RR_Codex_PR18_Followup.md`: targeted prompt for source repair, reference-benchmark upload/replotting, and manuscript integration.
- `CODEX_HANDOFF.md`: original broader analysis and integration plan.
- `REVISION_NOTES.md`: revisions and author/release tasks kept outside the paper.
- `SOURCE_AUDIT.md` and `provenance/`: scientific evidence and source hashes.

## Rebuild

Use Python with NumPy, pandas, matplotlib, scikit-learn and PyMuPDF. PyMuPDF is needed only to regenerate the CuO vector-path CSV when that CSV is absent. The verification environment is recorded in `provenance/plot_environment.json`.

```bash
python code/verify_package.py
python code/make_figures.py
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error Supporting_Information.tex
pdflatex -interaction=nonstopmode -halt-on-error Supporting_Information.tex
```

The main paper uses a single-column review layout. Structural renderings preserve source coordinates; local coordination cartoons and molecular templates are construction illustrations. CuO curves restyled from the supplied vector plot are graphical rendering data, not raw experimental measurements for quantitative fitting.

This bundle supports figure reproduction and prediction-summary verification. Repeating the full structural/spectral audit and model fitting requires the local dataset described in the handoff. Repository snapshot: `71bb0a20b369faba842b413a09544dd66c127a78`. No new model fitting or HPC calculation was performed for the revision.

## Earlier PR review (superseded by the latest review below)

The current manuscript incorporates the corrected stripe design (including 3:1 and 1:3), catalytic coupling hypothesis, metric equations and explicit NERSC Perlmutter workflow. Figures include the full molecular gallery, mixed Cu/Au motifs, common-scale contrast curves, separate individual-layer/cumulative-depth trends, and a gray-to-green model comparison. `figure_data/adsorbate_illustrations.json` specifies schematic bond graphs; these drawings are not optimized molecular geometries.

PR #18 was reviewed at `68953df0128b730e1b357b8698da3075590b0e07`. Archived prediction replay passed. Independent tests gave 84 passed / 5 failed plus 19 additional passes; the same five NERSC tests fail on the base. Two additional graph/cell validation counterexamples were reproduced. PR evidence and review logs are retained in `provenance/pr18_review/`. The full raw export and complete cold-cache refit were not repeated in this environment. The new manual web training route is distinguished from an agent-callable implementation and from the archived primary selector.

## Earlier figure clarity revision

Figure 2a now uses a common viewport and uniform atomic symbol radii for every molecular illustration. Figure 2c replaces isolated coordination fragments with one extended ideal Cu–Au surface carrying five labeled CO adsorption sites, including a Cu–Au bridge across the boundary. Its exact schematic coordinates and contacts are saved in `figure_data/interface_site_illustration.json`. Figure 4c separates individual layers from population-weighted averages of the top n layers into two plots with explicit axes and different colors. Shape, CDF, d1, Combined and DCT24 are defined in the main text and caption, with numerical definitions in SI. The abstract leads with catalyst selectivity and competing pathways. All scientific prediction values are unchanged.

## Latest PR and benchmark revision

The reviewed head is `518c686ccdaf79e24eca7577d72b5b9448987631`. Three committed Python files are truncated and fail syntax checks; the full focused suite stops during collection. An importable four-file subset gives 76 passes and two failures, one from the agent syntax error and one from the truncated final assertion `assert summ`. The shared graph and FDMNES cell checks pass. `provenance/pr18_review_latest/independent_review.json` records the exact Git blobs and findings. Archived PR reports are labeled `pr_reported_*`, distinct from independently run checks.

The manuscript now keeps the abstract general until CO2RR is introduced after the workflow sentence, improves the introductory argument and Section 2.2, and explains DCT24 with its source and equation. Figure 2 uses exclusive single/coadsorbed bins and seven consistent-projection sites including CuCuCu/CuAuAu hollows. `figure_data/adsorbate_configuration_counts.csv` and `interface_site_illustration.json` provide the exact counts/coordinates. The numerical ML scores are unchanged. SI reports 140/169 exact broader OOF files rather than universal agreement, and distinguishes recorded agent/model replay from production execution.

The new numeric `benchmark` folder and `Cu_XAS.ipynb` exist in the separate Codex working folder, not in this package. Their upload, physical processing, three-compound comparison, and resulting FDMNES Green selection rationale are specified in `CO2RR_Codex_PR18_Followup.md`. The current CuO panel remains explicitly qualitative graphical data until replaced with those raw numeric references. Both manuscript and SI have been compiled and rendered here; SI sources are included.
