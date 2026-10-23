# Digital Discovery manuscript package

Current manuscript and Supporting Information by Haili Jia and Maria K. Y. Chan, updated using PR #18 at `390b1b899c4dca43685fda1380a494b1d5019fcd`.

The main paper preserves the general agent scope and uses CO2RR as the demonstration. It includes the numeric Cu/Cu2O/CuO reference comparison, interface/site/coverage controls, adsorbate inference, depth sensitivity, and the proposed matched 2CO/OCCO test. The SI contains detailed computational/statistical methods, reference processing, metric tables, and four landscape comparison grids.

## Contents

- `manuscript.tex`, `references.tex`, `manuscript.pdf`: main paper.
- `Supporting_Information.tex`, `si/`, `Supporting_Information.pdf`: complete SI sources and compiled PDF.
- `figures/`: five main figures and four SI grids as vector PDF/SVG and 450-dpi PNG.
- `figure_data/`: exact ML predictions, spectral arrays, starting structures, molecular illustrations, and reference display arrays.
- `code/make_publication_figures.py`: main native figure source; `make_figures.py` is its entry point.
- `code/plot_reference_grids.py`: SI grid presentation from recorded arrays, with no refitting.
- `reference_benchmark/`: complete uploaded numeric benchmark, original/executed notebooks, portable processing, raw inputs/logs, provenance, fixed settings, metrics, and exact plotted arrays.
- `CO2RR_PR18_Review.md`: current review and remaining scientific limits.
- `CO2RR_Codex_PR18_Followup.md`: remaining scientific work, including mandatory CHO/COH connectivity and missing-output provenance audits before further fitting; the previous source-repair prompt is historical.
- `provenance/`: hashes, scientific sources, independent checks, and explicitly historical earlier reviews.

## Reproduce figures and compile

Use Python with NumPy, pandas, SciPy, matplotlib, and scikit-learn, plus a LaTeX engine with the packages declared in the sources.

```bash
python code/verify_package.py
python code/make_figures.py
python code/plot_reference_grids.py
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error manuscript.tex
pdflatex -interaction=nonstopmode -halt-on-error Supporting_Information.tex
pdflatex -interaction=nonstopmode -halt-on-error Supporting_Information.tex
```

`reference_benchmark/process.py` recomputes the original reference-v1 processing from the included raw inputs. Its own grids retain the original PR styling. `code/plot_reference_grids.py` restyles those exact arrays for this manuscript. Main Figure 3 reads the frozen copy in `figure_data/reference_benchmark/plotted_arrays.csv`; after deliberately rerunning processing, synchronize that CSV before rebuilding. Keep a versioned comparison rather than silently replacing archived inputs/results.

## Interpretation and review scope

Three numeric VASP slots are absent and remain blank. The VASP “GW” name is a source alias, not verified many-body physics. The CuO FDM series lacks completed native final-state processing. Short normalization windows, nonidentical broadening, and variable fitting intersections limit method ranking; see SI. All common-offset metrics use the full 201-point comparison window; the shorter separately aligned CuO FDM entry is flagged explicitly.

The archived ML cohort remains 87 independent configurations and 348 spectra. Saved predictions reproduce primary micro/macro F1 0.7791/0.6383 and exact-set 0.6897. The broader recorded refit has 140/169 exact OOF files; it was not rerun here. No new spectral calculation or model fit was performed. The full simulation export required for audit/refitting is not included; the raw reference benchmark is included.

Independent review at the repaired head confirms 106 compile checks, six imports, 101 focused passes, 165 benchmark hashes, and 46 RMSE arithmetic checks. Raw-input reprocessing also reproduces all 46 errors and fitted offsets to numerical precision. The supplied Figure 4 plotting source was corrected to reproduce its saved panel arrangement. Figure illustrations remain distinct from relaxed scientific structures.

## Figure interpretation and next scientific tests

The latest figures use the complete set of available computational reference curves in the main panel, with labels FDMNES green, FDMNES fdm, FEFF, VASP PBE, and VASP GW. Reference traces are scaled to unit maximum for display and labeled Intensity (a.u.); benchmark metrics retain their earlier local edge-jump normalization. Figure 2a now shows the substrate alone on the left and schematic adsorbates on the right, with a shared element legend. Figure 3c compares mean spectra for configurations containing each species against configurations without that species, not against a clean surface. A common maximum absolute contrast scales every curve to the same −1 to 1 range; it is not a percentage absorption change. Exact display values/divisors are saved in figure_data.

The uncompleted 2CO/OCCO test has moved from Results to an explicit Outlook. The next Codex task prioritizes a genuinely matched chemical comparison and coadsorbate/robustness validation, followed by a compact model and feature ablation. No new model was fitted for this figure revision.

The current manifest contains five CHO-containing configurations (two CHO-only and three mixtures), versus thirteen across the full inventory. These are configuration counts from existing labels, not a new molecular-connectivity audit. Six retained COH entries have CHO in their basenames; the updated Codex task requests initial/relaxed/XAS-used geometry reconciliation before any relabeling. It also requests reconciliation of the missing-quartet report with a completed terrace-2CO record of the same configuration name; exact matched input lineage remains unverified. See `provenance/CHO_COH_audit_request.json`. The manuscript and scientific labels have not been changed on the basis of filenames alone.

The updated benchmark task also requires a source-level VASP broadening/energy-axis audit, correction of negative processed CuO fdm intensity, consistent 0–1 normalization for every calculation and experimental trace, and visible cosine/Pearson/Spearman tables computed from the plotted arrays. The packaged figures remain the earlier version pending that reprocessing; no new corrected benchmark scores are claimed. See `provenance/benchmark_processing_audit_request.json` and `provenance/benchmark_display_range_audit.csv`.

Figure 3c is also scheduled for replacement by verified matched adsorbate-minus-clean-parent spectra; its current species-absent mean is not a clean-surface reference. The prompt specifies shared intensity scaling for differences, separate relaxed/frozen-metal controls, and distinguishes potentially negative difference signals from nonnegative absorption. Actual clean-parent availability must be verified against the full export.
