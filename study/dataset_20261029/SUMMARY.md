# CO2RR_1029 record and model refresh

The added data give a modest improvement on the same old configurations, with substantial uncertainty and mixed outcomes across seeds. They do not establish a uniform improvement, independent facet transfer or experimental accuracy. Manuscript and SI were not changed.

## Uploaded data and record audit

The primary scientific-source manifest covers26,198 current files:4,549 newly uploaded files and21,649 exact-byte matches to already uploaded, pinned source archives. Seven additional surviving partial-layer BAV logs are uploaded losslessly in practical multipart archives because they are needed to assess incomplete/mixed-run provenance. Eighteen original Cu100 ASE snapshots remain hash-verified in their pinned source archive. All selected bytes can be restored with the supplied source tool. Wavefunctions, licensed potentials, executables and unrelated/redundant scratch are not uploaded; this is not a copy of all37GB of scratch.

There are146 XAS folders,145 with readable supplied canonical convolved data and121 with supplied layer records. Both supplied conversion routes were used: the four-layer shell wrapper and generic Python converter with explicit available-layer merging. The resulting145 records plus one locally derived clean Cu100 reference give146 versioned records. Original author files are preserved in the source packages; corrected records are separate.

Every one of553 retained native measurement series exactly matches its numeric source, and every corresponding layer CONTCAR matches the actual FDMNES input geometry with element-aware atom correspondence, full cell metric and in-plane minimum-image checks. Labels use molecular connectivity, including H–C versus H–O, not basenames. Formula/cell/site metadata and generic Cu100 placeholders were corrected; site descriptions now expose actual contacts rather than equating pure-Cu atop/bridge environments with Cu3 hollows. The clean reference is explicitly clean with Cu16 formula and correct source assets.

Two root CONTCAR/XAS-copy discrepancies remain explicit. `H/H_0p25/interface_biased2` retains only the previously established exact archived geometry/output lineage, without claiming the root geometry was used. `CO_H/CO_H_0p139_3CO_2H_interface_biased` lacks that provenance and is excluded. Records describe the actual XAS-used geometry rather than attaching a new root geometry to old spectra. No relaxed-geometry guard was bypassed.

`OH/OH_0p125_interface_Au` is recovered for layers4+5 only: both required layers match CONTCAR and full populations, while incompatible layers2/3 are explicitly removed from the versioned measurement. `OH/OH_0p50_uniform` remains outside the fixed-layer study because strong roughening splits the height-based Cu-layer partition; its table cannot silently override that ambiguity. Seven other configurations lack a valid required layer. One has36 complete numeric raw sources and a saved numeric convolution candidate, but its surviving BAV proves only a three-site recalculation, leaving full-source geometry lineage unresolved; it is not included in ML. Other incomplete layers have missing sites and/or truncated ranges. See the exact per-case/atom tables rather than interpreting these statuses as never-run claims.

There are137 source-scaled configurations, including136 geometry/top-layer-eligible supplied cases and the derived clean Cu100 reference. Closed-label fitting uses118 Au–Cu cases (including the existing clean negative) and12 supported-label Cu100 cases. Six water/carbonate cases remain audited interference/review queries, not silently relabeled examples. The clean Cu100 reference is an input, not a new adsorbate example.

| Au–Cu positive configurations | Old99 | New118 |
|---|---:|---:|
| CHO |7|9|
| COH |7|10|
| OCHO |8|8|
| CO |37|44|
| H |25|30|
| OH |6|12|

The19 additional independent Au–Cu configurations include three CO-only cases. Layer spectra and molecules are not independent additions. Two CHO-only configurations still coexist with additional CHO-containing mixtures; these counts must not be conflated.

## Controlled identity comparison

The primary anchor fixes the population-weighted physical top two layers and **Combined spectrum/CDF/first/second derivatives**, with64 independent balanced ExtraTrees classifiers, leaf1 and sqrt feature sampling. Three fixed seeds are all reported; no seed was selected by its test score. Original99 outer folds are preserved, additions receive deterministic geometry-hash folds, and original inner-training assignments remain paired across extension. Fixed0.5, baseline-transferred and training-only retuned thresholds are saved separately. No automatic depth selector was rerun.

All198 old99 top-two canonical source files are byte-identical to the earlier export, and labels/groups are unchanged. The archived model-input matrices used a floating-point `arange` interpolation grid: current direct-grid arrays differ by at most4.8e-12 in relative intensity and2.1e-10eV in grid coordinates. That harmless sampling difference is reported rather than asserting bitwise matrix identity. Both newly run cohorts use the same current arrays, dependencies and paired folds. The baseline seed20261020 reproduces the prior99-case0.7500/0.5711/exact0.6263 values.

| Fixed Combined, inner-training thresholds | Training cohort | Query cohort | Mean micro F1 | SD across3 seeds | Mean exact-set |
|---|---:|---:|---:|---:|---:|
| Baseline |99 Au–Cu|same99|0.7434|0.0165|0.5993|
| Expanded |118 Au–Cu|same99|0.7519|0.0047|0.6229|
| Expanded |118 Au–Cu|all118|0.7645|0.0094|0.6299|

Mean macro F1 on the same99 is0.5795→0.5812. With retuned thresholds, micro F1 improves for two seeds and declines for one. All three paired conditional95% micro-F1 intervals cross zero; these intervals resample fixed configuration predictions and omit refitting/parent-transfer uncertainty. Thus the point estimates are encouraging but inconclusive. At the first predeclared seed, six wrong identity sets become correct and six correct sets become wrong on the same99. H-mixture recall improves4/11→6/11, while CO-mixture recall remains9/12. These are the new99-case control, not replacements for the frozen87-case primary counts.

Holding the old training-only thresholds fixed as well gives same99 micro F1 0.7434→0.7608 and exact-set0.5993→0.6364. With a universal fixed0.5 threshold, micro F1 changes0.7209→0.7550. All three threshold stages were declared in advance and are saved; retuning does not always help. These controls isolate the added-training-data effect from threshold reselection, without choosing a threshold by outer-test performance.

The full118 score is not itself a paired improvement because the query cohort changed. The frozen87/348 and90-configuration studies, predictions, scores and manuscript remain unchanged. Reusing these folds is not prospective confirmation.

## Physically scaled and differential controls

Local reconvolution reproduces272 supplied native top-layer outputs. For full explicit site lists, inverse convolution multiplicities give one contribution per physical site. The new clean Cu100 uses three nonoverlapping symmetry representatives covering four sites; a verified x/y reflection maps the multiplicity-two representative onto its partner. Its multiplicity must be retained rather than divided out. Two clean layers are reconstructed from actual saved electronic spectra, with declared matching broadening and no new electronic/HPC calculation. A CONTCAR or complete raw table alone is not claimed to certify relaxation/production termination.

Au–Cu uses layers4+5 and its clean interface; Cu100 uses3+4 and its own clean reference. Shared clean-derived edge-step scales preserve signed differential amplitude, with no per-difference min/area/norm scaling or independent peak alignment. The compact ablation uses the same configurations/folds and a predeclared six-candidate, training-only feature/model/threshold search for all three representations. It is separate from the fixed Combined primary control.

| Compact ablation | Normalized raw micro F1 | Physical raw micro F1 | Signed delta micro F1 |
|---|---:|---:|---:|
| Au–Cu118 |0.7717|0.7194|0.7194|
| Joint130 |0.7123|0.7170|0.7361|

The composition/cell/physical-layer checks are strict, but most Au–Cu pairs have weaker initial-parent or constraint evidence; this differential table is a same-substrate/reference sensitivity study, not a claim of universally matched clean parents. The separate reference_compatibility table records initial-metal/constraint matches. All12 supported Cu100 cases match the clean bottom-eight-fixed convention; initial ancestry remains separately qualified. The symmetry representative check requires an actual cell-metric isometry and rejects rectangular-cell x/y swaps.

With one fixed Au–Cu clean reference, subtraction is an affine translation and supplies no new information beyond physically scaled raw in this control. Joint delta gives a higher point score than joint raw here, but it does not outperform every Au–Cu-only control or establish substrate invariance. On the four common labels CO/H/O/OH, compact joint delta gives Cu100 micro F1 of0.4762 on12 cases; common-label and all-eleven-label metrics are both saved. Equal-domain aggregates are separate from sample-weighted pooled scores. Cu100 macro zeros for untested labels are not evidence about those species. One reference family per domain cannot estimate new-parent transfer.

## CO loading

The primary target is N(CO)/all top-surface metal sites. Cu-normalized loading and CO/nm2 are distinct, explicitly saved deterministic unit conversions using known substrate populations/area; Cu loading is not literal Cu-site occupancy. No molecular counts or structural metadata enter spectral predictors.

Using identical old25 query cases, the compact normalized-raw coverage control changes MAE0.013971→0.011444 after adding the three CO-only configurations. The paired conditional95% MAE-difference interval is[−0.006057,+0.000344], again crossing zero. On all28 Au–Cu CO-only cases the MAE is0.010785. This is not a direct replacement for the earlier differently grouped frozen coverage score.

Joint32-case coverage must be read by domain: normalized raw gives Au–Cu MAE0.010930 and Cu1000.044054; delta gives0.015878 and0.041227, respectively. Only four Cu100 CO configurations exist, and loading changes with motif. Pooled R² can be inflated by the between-domain loading range and is not a transfer headline. Sixteen CO-containing mixtures are evaluated separately with frozen CO-only ensembles; joint delta MAE0.014745 there is an interference diagnostic, not a validated general mixture-loading model.

## Reproducible delivery and remaining limits

The shared audited callable, agent schema/parser/planner and UI expose `dataset_refresh_v1` for a prepared versioned export. Implicit updated-data requests now use this fixed-top-two route; the historical automatic-depth v3 recipe requires explicit selection. Models, predictions, folds, thresholds, source hashes, per-species/single/mixed/domain supports, missing-positive training labels, paired intervals, unit conversions, unsupported queries and a review figure are saved. All110 saved models reproduce their held-out scores/predictions after reload.

Local focused checks pass153 tests, including the original eight-file suite and four new refresh checks. Every tracked Python source is compiled and imported checks are repeated at the exact pushed commit in a clean worktree; the PR body records that SHA and final counts. All source/derived archive members are hash verified. No hosted-CI success, production scheduler execution, new electronic spectra, calibrated experimental accuracy or independently confirmed transfer is inferred from these local checks. Manuscript/SI PDFs, editable sources and their archives are unchanged.
