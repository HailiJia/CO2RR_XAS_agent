# Codex task: complete a chemical comparison and strengthen the XAS ML evidence

Continue PR #18 in `HailiJia/CO2RR_XAS_agent`; last reviewed head `390b1b899c4dca43685fda1380a494b1d5019fcd`. Read repository instructions and newer changes first. Preserve the archived 87-configuration/348-spectrum benchmark and its frozen predictions. Use the accompanying latest manuscript package: it adds the available VASP curves to main Figure 3, uses Intensity (a.u.), clarifies species-present versus species-absent contrasts, and moves the uncompleted carbon-coupling design into an Outlook section. Keep the PR a draft; do not merge or force-push.

The repaired source already passes 106 compile checks, six imports, and 101 focused tests. The old source-repair prompt is complete. The tasks below are new scientific extensions, not claims that those results already exist.

## Immediate prerequisite: audit CHO/COH labels and recover apparently missing outputs

The author reports more than two CHO structures and asks whether CHO and COH have been confused. Resolve this **before further model fitting or claims about rare-species support**. The supplied manifests currently give the following configuration counts; these are existing labels, not an independent geometry validation:

| Inventory | CHO only | CHO in mixtures | Total CHO-containing | Total COH-containing |
| --- | ---: | ---: | ---: | ---: |
| Retained ML cohort (87 configurations) | 2 | 3 | 5 | 6 |
| All unique structures (155 configurations) | 2 | 11 | 13 | 15 |

The three retained CHO mixtures have labels `CHO;CO;H`: `CHO/CHO_0p056_interface_Cu`, `CHO/CHO_0p056_interior_Cu`, and `CHO/CHO_0p111_uniform`. The two CHO-only entries are `CHO/CHO_0p056_AuCu_bidentate` and `CHO/CHO_0p056_CuCu_bidentate`. Thus a bar showing two CHO-only configurations does not establish that there are only two CHO positives. Distinguish configurations, individual adsorbate molecules, single-species configurations, mixtures, and completed usable records throughout the figures, captions, support counts, and manuscript.

All six retained COH entries have CHO in their basename: `COH/CHO_0p028_interface_Cu3`, `COH/CHO_0p056_AuCuCu`, `COH/CHO_0p056_interface_Cu3`, `COH/CHO_0p056_interior`, `COH/CHO_0p056_uniform`, and `COH/CHO_0p167_uniform`. This is a concrete audit target, **not proof that the current chemical labels are wrong**. Search the full local export for every candidate named or classified as CHO/COH, including excluded, incomplete, and duplicate structures; do not limit the audit to these retained examples.

Trace initial structure → relaxed structure → exact structure supplied to each XAS calculation → AI-ready record → training target. Verify atom correspondence and any atom-index reordering. Identify CHO by H bonded to C, COH by H bonded to O, and CO+H by disconnected CO and H components, using the full molecular bond graph and minimum-image distances in the periodic surface directions. Treat the vacuum direction appropriately. Check hydrogen migration/dissociation and distance-cutoff sensitivity; show ambiguous cases for review rather than resolving them from directory names or elemental formulas. Supply per-molecule and per-configuration audit tables with paths/hashes, C–H/O–H distances, connectivity, intended/relaxed/XAS-used identities, record labels, local metal contacts, and inclusion/exclusion reasons. Save enlarged local geometry views with H and relevant bonds visible, plus their plotting code.

If incorrect labels are demonstrated, preserve the archived benchmark as an historical snapshot, issue a versioned corrected dataset and before/after label table, and rerun all affected feature/model selection and held-out evaluation on the same declared grouping protocol. Update species counts, supports, metrics, figures, manuscript, SI, and agent/UI labels consistently. Do not silently overwrite old results or relabel based on a folder-name mismatch alone.

**Recheck the missing-output search for the former Section 3.6.** `study/results/design/matched_quartet.json` marks all four matched cases `relaxed_available: false`. However, both `figure_data/cv_manifest.csv` and `figure_data/all_unique_structures.csv` contain a completed usable record for `2CO_adjacent/2CO_adjacent_0p056_interior`, with the record at `XAS/FDMNES/green/isaac_record_layers.json`. This apparent contradiction requires a provenance check: the same configuration name does not prove the same input version, parent, or numerical settings.

Inspect `tools/study_design.py`: its current reconciliation first requires an exact initial-geometry hash match, then checks only `CONTCAR` at that alias; multiple candidates also produce no unique observed audit. The contrast-status sentence is hard-coded. Search actual records, XAS input geometries, relaxed-output copies, alternate paths, archived runs, job logs, and recorded lineage before declaring a calculation unavailable. Distinguish not located by the current lookup, never run, unfinished, completed but incompatible, completed and matched, ambiguous lineage, and dissociated. Derive the status from evidence rather than a fixed sentence. Retain justified grid/absorber exclusions, and test the reconciliation against recovered records and equivalent duplicate candidates.

Return a verified status table for each member of the 2CO/OCCO interface/terrace quartet, with exact missing artifacts and recoverable completed outputs. A valid Cu K-edge pair in one region already supports a scoped coupled-versus-uncoupled contrast; the full quartet supports an interface-versus-terrace difference of those contrasts. Additional C/O edges are extensions, not prerequisites for a first valid Cu K-edge result. If an intended OCCO structure relaxes to another connectivity, report the actual relaxation outcome separately rather than treating it as an intact-OCCO spectrum or an established thermodynamic conclusion. Restore a Results subsection as soon as a verified completed comparison supports it; the earlier move to Outlook is not a claim that such results are impossible.

## 1. Highest priority: produce one genuinely controlled chemical result

The previous Section 3.6 documented starting structures and an analysis design, without a verified matched relaxed spectral comparison. This is not evidence that the full local export lacks usable outputs; perform the reconciliation above first. Search for an actual comparison with the same metal parent, elemental loading, absorber definition, and numerical settings:

- First choice: two neighboring unbonded CO molecules versus intact OCCO, each at interface and Cu-terrace locations.
- If those results do not exist, inspect whether a matched CO+H versus CHO/COH hydrogenation comparison is already complete. Equal formula alone is insufficient: verify molecular counts, surface loading, parent geometry, adsorption region, and relaxed connectivity.
- Recover the corresponding clean parent spectrum when available. It is required for an actual adsorbate-minus-clean contrast. A mean over configurations lacking one species is not a clean reference.

For every candidate, report structure/output paths, hashes, observed connectivity, local metal contacts, interface distance, atom counts and absorber populations. Reject unmatched or dissociated cases as evidence for the intended intact intermediate. Compute and plot matched differences only for valid completed pairs. For a complete 2CO/OCCO quartet, evaluate the interface-minus-terrace difference of coupling contrasts, with consistent Cu/C/O edge processing where available.

If required outputs are missing, create a precise missing-results table and ready-to-run inputs. Do not invent spectra, relabel dissociated OCCO, or substitute linear combinations of unrelated spectra. Do not launch new HPC jobs unless the user has authorized them in this Codex session. The manuscript should retain an Outlook until the chemical result actually exists.

## 2. Target the main ML failure: coadsorbate recovery

From the primary saved OOF predictions, H is detected in **14/14 single-species H cases but 0/8 H-containing mixtures**; CO is detected in **21/21 single-species cases and 5/9 mixtures**. The exploratory fixed winner has different values and must remain separate. Reproduce these counts before extending the analysis; use `figure_data/primary_single_mixed_recall.csv` from the supplied package as a cross-check.

Keep the present primary selector as a frozen baseline. Add a transparent spectral nearest-neighbor baseline (cosine/Euclidean distance; k=1/3/5 selected inside training) and return the closest training examples for review. Evaluate a small, declared candidate set on identical configuration folds:

1. **Independent binary XGBoost classifiers**, one per species, with a compact regularized search. Use constant predictors when an outer-training partition has no positive or no negative examples. Early stopping must use inner-training validation, never an outer test fold.
2. **Classifier chains using a compact tree estimator**, compared directly with independent classifiers using the same features and tuning budget. Choose any chain order only inside training. Use group-aware out-of-fold predecessor predictions for training-chain features; never supply held-out true labels at inference. Do not assume co-occurrence learned from nine mixtures will generalize.

Select features, hyperparameters, calibration, and decision thresholds using only inner grouped folds. Report micro/macro F1, exact-set accuracy, per-species precision/recall and support, and separate single/mixed recall. Use paired configuration-level comparisons with the baseline. Improved mixture recall at the cost of many new false positives is a tradeoff, not an unqualified improvement. No synthetic examples count as new independent positives.

## 3. Test compact feature hypotheses, rather than a large architecture search

Audit existing code first: normalized spectra, CDF, derivatives, their combinations, DCT24, multiscale derivatives, and surface/depth contrasts have already been tried. Do not describe them as new.

Add a compact ablation with:

- Fixed-window areas, centroids, first-derivative summaries, and near-/post-edge peak descriptors. Define windows from physical/literature assumptions before looking at test labels, or learn them exclusively inside training folds. **Do not use the full-dataset, label-conditioned Figure 3c curves to choose supervised feature windows before cross-validation.**
- A small DCT dimension sweep, e.g. 8/16/24/32 coefficients, selected in the inner loop. The original 24 is a fixed candidate dimension, not an established physical cutoff.
- A small PCA baseline, with its transform fitted solely on each training partition, to test whether compression improves small-sample generalization.

Keep spectral predictors separate from structural metadata. Counts, identity strings, filenames, molecular graphs, site labels, and interface distance can define targets, groups, or audits; they must not enter an adsorbate-from-XAS classifier as predictors. With 87 independent configurations, prioritize this controlled ablation over a broad CNN/transformer search.

## 4. Test robustness and the actual domain of generalization

Evaluate clean held-out spectra and explicitly synthetic stress tests for energy shifts, extra instrumental broadening, and noise. A compact starting grid is shifts ±0.2/0.5/1/2 eV, extra Gaussian sigma 0.25/0.5/1 eV, and noise amplitudes 1/2/5% of peak intensity. These are test scenarios, not measured instrument uncertainty. Preserve native lifetime broadening. Interpolate only on common supported energy ranges; report any crop.

Compare unaugmented training with training-only augmentation. Every augmented version and every layer representation from a physical configuration stays in that configuration's fold. Freeze perturbation settings and random seeds before examining outer-test results; do not tune models against the stress-test curves.

Inventory independent parents, compositions, coverage regimes, and adsorption motifs. Where there is sufficient support, run a genuinely harder parent- or motif-held-out evaluation. If there is only one parent, state that parent transfer cannot be estimated. Report missing-positive training labels in every split. Keep this new protocol separate from the existing frozen-fold benchmark.

Add configuration-level learning curves and paired uncertainty intervals. Distinguish intervals conditional on fixed predictions from variation caused by refitting. Existing test folds have already been examined repeatedly; reserve genuinely new configurations for a locked prospective confirmation test rather than calling another refit independent validation.

## 5. Connect model uncertainty to the agent's next useful action

Using inner-training calibration only, evaluate a simple review/abstention policy and a coverage-versus-error curve. For sparse labels without calibration support, say so rather than fabricating reliable probabilities. Surface the predicted identity set, relevant uncertainty, and whether the query lies outside the trained energy/depth/parent domain.

Have the agent rank a small set of proposed additional configurations that address observable failures: matched CO/H mixtures, more independent hydrogenated C1/C2 examples, or missing clean/matched coupling references. Distinguish model-disagreement candidates from scientific-priority controls. Do not claim these proposed calculations have already improved performance.

Expose each new supported analysis through the same scientific callable used by the agent/UI. Demonstrate one actual natural-language request on the available local export: audit → fit a declared recipe → save/reload → predict on held-out records → generate a reviewable report. Save data/code hashes, protocol, outputs, and a sanitized trace. No production scheduler or new-spectrum claim should be inferred from a local training/inference demonstration.

## 6. Benchmark and manuscript consistency

**Author priority: repair benchmark processing and show similarity metrics before interpreting method accuracy.** The VASP broadening/alignment is disputed. The main intensity plot must use one declared normalization rule for every calculation and experimental reference, contain no negative displayed absorption, and be accompanied by cosine similarity, Pearson correlation, and Spearman rank correlation. The existing RMSE-only presentation is insufficient. Do this alongside the label/provenance audit above, before presenting additional ML extensions as the main improvement.

The main figure now includes all available VASP PBE and VASP GW traces. The uploaded snapshot still lacks numeric spectra for Cu2O PBE, Cu2O GW, and CuO PBE. Search the user's working folder for newly available outputs and verify their lineage; do not assume they do not exist merely because the previous upload lacked them. If found, version the provenance and results. Otherwise retain missing slots explicitly.

Use the requested labels FDMNES green, FDMNES fdm, FEFF, VASP PBE, VASP GW, and Experiment; label the ordinate Intensity (a.u.). Keep method settings precise: folder names alone do not establish a GW/BSE treatment. Keep reference-v1 unchanged as an archive; make a new version for the corrected processing, normalized arrays, metrics, and figures, rather than mixing new figure scaling with old scores.

### 6.1 Trace and correct the energy axis and broadening

Reconcile `raw/Cu_XAS.ipynb` with every parsed VASP source. The notebook uses Gaussian sigma = 1 eV and exploratory +120/+125 eV offsets. Current `process.py` instead applies an additional Lorentzian FWHM = 1.55 eV to VASP in the primary view, treats Gaussian sigma = 1 eV as a sensitivity, uses nominal +120/+130 eV offsets to select normalization windows, and fits one backend shift shared by PBE and GW. These are consequential choices, not validated equivalences. Reproduce the notebook route on its actual input files where recoverable, then provide a side-by-side raw → native-broadened → additional-broadened → aligned comparison and explain any justified change. An empty or unverified notebook input must not be silently substituted.

Verify the actual VASP tags and output quantity, diagonal tensor/polarization averaging, energy-zero convention, and supplied PAW-potential lineage. Account for broadening already present in each raw spectrum, distinguishing Gaussian sigma/FWHM from Lorentzian HWHM/FWHM. Add only justified remaining lifetime/instrumental broadening; preserve nonnegative intensities with positive normalized convolution kernels and handle energy-grid spacing and finite-domain edges explicitly. Do not smooth to improve apparent agreement without declaring the fitted degrees of freedom. A native CH_SIGMA of 0.1 eV does not by itself justify omitting the notebook's instrumental Gaussian or adding a universal Lorentzian to every method.

Assess separate Cu calibration offsets for PBE and the GW-labeled dataset; a shared software name alone does not establish a common energy zero. Use one transferable offset across compounds within each declared method for the primary comparison. Show separately fitted per-compound alignment only as a shape-comparison diagnostic because it removes chemical-shift information. Use a fixed, declared alignment objective and physically justified bounds; report fitted shifts, widths, units, boundary hits, and uncertainties/sensitivity. Require the same full comparison energy window at every trial shift, not a variable intersection that changes the objective. Report missing support instead of extrapolating or rewarding a shorter window.

### 6.2 Repair negative intensities and normalize every series consistently

The supplied display audit shows all Figure 3a curves already have maximum 1, but CuO FDMNES fdm reaches about -0.209. Its raw calculated values are positive: the negative portion appears after the existing local-pre-edge background subtraction. Thus this is not fixed by changing the y-axis limit or dividing by the maximum. The small negative values in the experimental raw export are a separate baseline/noise issue; preserve those raw data too.

Locate the first processing stage introducing each negative value. Check whether the supposed pre-edge window is actually pre-edge after the method-specific energy convention/alignment is resolved, particularly CuO fdm and VASP. Correct unsupported background subtraction and normalization windows. Do not take absolute values, indiscriminately clip negative values, or merely hide them below the axis. Any roundoff-only tolerance must be explicit; substantive negative processed absorption is a failed processing check requiring investigation.

Use one clearly stated final 0–1 intensity normalization for **all** methods and Experiment on the same declared energy interval: after justified background correction, `I_norm = (I - I_min)/(I_max - I_min)` with each trace's extrema evaluated over that identical interval. Save all offsets/divisors, reject flat/invalid traces, and show raw/pre-normalization diagnostics separately. This is a relative-intensity comparison and does not establish absolute cross-section accuracy. The same final arrays and energy grid must feed both the main plotted comparison and its reported similarity scores. If an independently justified edge-jump comparison is retained as a sensitivity, apply it to every eligible series, label it separately, and never mix its metrics with the 0–1 display. Min–max scaling must not be used to conceal a faulty physical baseline.

### 6.3 Show the requested quantitative metrics

For each available material/method pair against Experiment, report cosine similarity, Pearson r, and Spearman rho on the identical fixed energy grid/window and normalized arrays used for the main comparison. Compute cosine as the normalized dot product without extra mean centering; Pearson uses centered values, and Spearman correlates intensity ranks with correct tie handling. Handle zero norms, constant traces, nonfinite values, and inadequate overlap as explicit unavailable cases, not zero similarity. Do not report per-energy-point correlation p-values as independent-sample evidence because adjacent spectral samples are correlated.

Provide a readable table or heatmap in the manuscript/benchmark figure, a complete SI table, and a machine-readable CSV with material, method, processing version, comparison type, normalization, energy limits, point count, shift, Gaussian sigma/FWHM, Lorentzian HWHM/FWHM, cosine similarity, Pearson r, Spearman rho, and optional RMSE. Keep missing method/material cells blank and explain why. Distinguish Cu-calibrated transferable alignment from separately shape-aligned diagnostics. A high correlation alone is not proof of accurate chemical shifts or a physical mechanism; discuss metric agreement/disagreement briefly. Include metric definitions and processing order in Methods/SI and use official SciPy implementations where applicable.

Return corrected benchmark code and an executed notebook, frozen intermediate/final arrays, before/after processing audit, metric tables, regenerated publication-quality figures, and compiled/visually inspected manuscript/SI. Add focused checks that normalized intensities lie in [0,1], all methods and Experiment use the same rule and full energy window, metric rows reproduce the plotted arrays, and identity comparisons have cosine/Pearson/Spearman equal to 1 when defined. These deliverables must be visibly included, not merely mentioned in the completion summary.

### 6.4 Replace Figure 3c with matched clean-surface contrasts

The author explicitly asks why the reference is not the clean surface. The current Figure 3c is a descriptive difference between mean spectra for species-present and species-absent configurations. The latter group is not clean: it includes other adsorbates and may differ in site, coverage, parent, and relaxation. It cannot isolate the spectral change caused by one adsorbate. Do not defend or retain this as an adsorbate-induced response merely by changing the caption.

Search the full export for actual clean-parent Cu K-edge spectra and their relaxed structures. For each comparison, verify the same metal composition, cell/parent lineage, absorber site or population weighting, layer definition, numerical settings, polarization treatment, broadening, and energy convention. Construct `Delta mu(E) = mu_adsorbed(E) - mu_clean,parent(E)` on a common supported energy grid, using a physically consistent shared per-absorber intensity scale. Do not independently min–max normalize the two spectra before subtracting: that would confound amplitude changes with rescaling. The all-series 0–1 convention above applies to the reference-material overlay/its metrics, not to physically scaled adsorbate-minus-clean differences. Report any common display divisor for the difference explicitly.

The supplied inventory already lists clean relaxed structures at `plain/CONTCAR` and `plain/2fixed/CONTCAR`, both flagged `no completed record`. Start the search there and reconcile those flags with actual XAS outputs elsewhere; a missing AI-ready record is not proof that no spectrum was calculated. Verify which clean structure matches each adsorbed parent and fixed-layer convention.

Distinguish a separately relaxed clean parent, which measures the combined adsorbate and substrate-response contrast, from a frozen-metal clean control made by removing adsorbates from the adsorbed geometry, which tests a different controlled comparison. Label those two comparisons separately if both exist. Do not call a frozen-metal calculation the relaxed clean reference. For a coadsorbate configuration, a clean-parent difference measures the total mixture effect; it does not isolate CHO, CO, or H individually. A species-specific claim needs an additional matched coadsorbate control or should be framed as a configuration-level response.

Use available valid pairs to replace main Figure 3c, showing representative adsorbates and mixtures with support/coverage and reference identity clearly stated. Difference spectra can legitimately be negative; the nonnegative-intensity requirement concerns absorption overlays, not Delta mu. If a dimensionless relative change is desired, define a stable shared clean edge-jump or common intensity denominator; do not divide pointwise by near-zero clean absorption or call arbitrary rescaling a percentage. Include clean/adsorbed overlays in SI, exact paired arrays and provenance, and saved plotting code. If clean spectra are absent or incompatible, report exactly which controls are missing and prepare their inputs; move the current unmatched descriptive statistic to SI or omit it rather than presenting it as the requested physical contrast. Do not synthesize a clean reference from the species-absent mean.

**Figure legend defect reported by the author:** Figure 3c in the version they are viewing has dashed lines but no dashed-line key. Inspect the exact latest compiled PDF and its generating source; do not guess their meaning or dismiss the report because another package differs. In the supplied `code/make_publication_figures.py`, current panel 3c uses solid contrast curves and a gray zero reference, whereas panel 3a explicitly draws CuO FDMNES fdm dashed while the shared method legend contains only solid handles. Reconcile this version/panel mismatch. Every solid/dashed/dotted curve must have a matching visible key or direct label identifying its scientific meaning, and shaded areas/reference lines must be explained. If a dash denotes an incomplete or provisional series, state exactly that rather than implying a second method. Build legends from the actual plotted artists or explicit validated handles; do not use a generic solid handle for a dashed trace. Check all Figure 3 panels at publication size in the final PDF, regenerate all exported formats from the saved source, and keep caption and manuscript references synchronized.

Only add new scientific results to the paper after completing their evaluation. Preserve the distinction between completed identity/depth results and the proposed chemical comparison. Return a concise model/feature/robustness table, held-out arrays, saved recipes/models, reproducible code, exact figure data, and updated compiled manuscript/SI. Do not replace a negative or inconclusive result with more confident prose.

Implementation references:

- XGBoost per-target multilabel support: https://xgboost.readthedocs.io/en/stable/tutorials/multioutput.html
- Classifier chains: https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.ClassifierChain.html
- Leakage-safe preprocessing: https://scikit-learn.org/stable/common_pitfalls.html
- Robust XANES featurization: https://doi.org/10.1021/acs.chemmater.3c02584
- VASP core-hole workflow and native broadening: https://vasp.at/wiki/SCH_calculations and https://vasp.at/wiki/CH_SIGMA
- Pearson and Spearman definitions/implementations: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.pearsonr.html and https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html
