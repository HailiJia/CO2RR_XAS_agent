"""Render exact saved coverage/identity/contact arrays and scientific report."""
from pathlib import Path
import sys,json,collections
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import dump
B=ROOT/'study/coverage_20261026';F=B/'figures';F.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})
def save(fig,name):
 for ext in ['pdf','svg','png']:fig.savefig(F/f'{name}.{ext}',bbox_inches='tight',dpi=350)
 plt.close(fig)
m=pd.read_csv(B/'coverage_metrics.csv');p=pd.read_csv(B/'coverage_heldout_predictions.csv');e=pd.read_csv(B/'coverage_external_predictions.csv');a=json.loads((B/'record_geometry_audit.json').read_text());manifest=json.loads((B/'model_configuration_manifest.json').read_text());proto=json.loads((B/'protocol.json').read_text());q=p[(p.protocol=='coverage_configuration_folds')&(p.model=='nested')]
fig,ax=plt.subplots(1,3,figsize=(13,3.8));ax[0].scatter(q.truth_ML,q.prediction_ML,s=40,color='#32784e',label='25 held-out configurations');upper=max(q.truth_ML.max(),q.prediction_ML.max())*1.1;ax[0].plot([0,upper],[0,upper],color='.45',lw=1,label='Identity');ax[0].set(xlabel='Observed CO / top-surface metal sites',ylabel='Predicted CO / top-surface metal sites',title='Cu–Au, configuration folds');ax[0].legend(fontsize=8)
for j,task in enumerate(['coverage_configuration_folds','coverage_level_held_out','contact_motif_held_out']):
 r=m[m.protocol==task].set_index('model');ax[1].bar(j-.18,r.loc['nested','MAE'],.34,color='#32784e',label='Nested spectra model' if j==0 else None);ax[1].bar(j+.18,r.loc['training_mean','MAE'],.34,color='.65',label='Training-mean baseline' if j==0 else None)
ax[1].set(xticks=[0,1,2],xticklabels=['Configuration','Coverage level','Contact motif'],ylabel='MAE (CO / top-surface metal sites)',title='Grouped extrapolation checks');ax[1].tick_params(axis='x',labelsize=8);ax[1].legend(fontsize=8)
c=e[e.domain=='Cu100_input_geometry_external'];ax[2].axvspan(q.truth_ML.min(),q.truth_ML.max(),color='#a1d99b',alpha=.3,label='Cu–Au training range');ax[2].plot([0,1],[0,1],color='.45',lw=1,label='Identity');ax[2].scatter(c.truth_ML,c.prediction_ML,c='#a34c37',s=45,label='Cu100 input geometries');ax[2].set(xlabel='Observed CO / Cu100 surface sites',ylabel='Predicted coverage',title='Cu100: separate transfer diagnostic',xlim=(0,1.05),ylim=(0,1.05));ax[2].legend(fontsize=7,loc='upper left');fig.tight_layout();save(fig,'CO_coverage')
contact=json.loads((B/'contact_predictions.json').read_text());fig,ax=plt.subplots(1,2,figsize=(8,3.5))
for k,name in enumerate(['Cu','Au']):
 x=np.array([r[f'truth_mean_{name}'] for r in contact]);y=np.array([r[f'prediction_mean_{name}'] for r in contact]);ax[k].scatter(x,y,c='#32784e');hi=max(x.max(),y.max())*1.1;ax[k].plot([0,hi],[0,hi],c='.5',label='Identity');ax[k].set(xlabel=f'Observed mean CO–{name} contacts',ylabel=f'Predicted mean CO–{name} contacts',title=f'{name} contacts within 2.25 Å');ax[k].legend(fontsize=8)
fig.tight_layout();save(fig,'CO_contacts')
# Identity per-label and single/mixed support remains separate from loading.
for name in ['identity_oof','identity_Cu100_external']:
 z=np.load(B/f'{name}.npz');pr,re,f1,s=precision_recall_fscore_support(z['truth'],z['prediction'],zero_division=0);pd.DataFrame({'species':z['species'],'precision':pr,'recall':re,'F1':f1,'positive_configurations':s,'predicted_positive':z['prediction'].sum(axis=0)}).to_csv(B/f'{name}_per_species.csv',index=False)
 if name=='identity_oof':
  mixed=z['truth'].sum(axis=1)>1;rr=[]
  for label,mask in [('single',~mixed),('mixed',mixed)]:
   for j,sp in enumerate(z['species']):
    n=int(z['truth'][mask,j].sum());tp=int((z['truth'][mask,j]*z['prediction'][mask,j]).sum());rr.append({'stratum':label,'species':sp,'support_configurations':n,'true_positive':tp,'recall':tp/n if n else None})
  pd.DataFrame(rr).to_csv(B/'identity_single_mixed_recall.csv',index=False)
# Source/archived comparisons by current alias; no claims based on folder names.
old=pd.read_csv(ROOT/'study/retraining_20261022/refit/dataset/cv_manifest.csv');now={r['configuration']:r for r in manifest if r['domain']=='CuAu'};delta=[]
for r in old.to_dict('records'):
 n=now.get(r['configuration']);delta.append({'configuration':r['configuration'],'status':'retained' if n else 'excluded_or_missing','old_labels':r['labels'],'current_labels':';'.join(n['identity_labels']) if n else None})
for n in now.values():
 if n['configuration'] not in set(old.configuration):delta.append({'configuration':n['configuration'],'status':'newly_usable','current_labels':';'.join(n['identity_labels'])})
pd.DataFrame(delta).to_csv(B/'comparison_with_90.csv',index=False)
for sp in ['CHO','COH','OCHO','CO','H']:
 for domain in ['CuAu','Cu100']:
  selected=[r for r in manifest if r['domain']==domain and sp in r['identity_labels']];print(sp,domain,'positive configurations',len(selected),'molecules',sum(r['observed_counts'][sp] for r in selected))
cs=json.loads((B/'contact_support.json').read_text());im=json.loads((B/'identity_metrics.json').read_text());em=json.loads((B/'identity_Cu100_external_metrics.json').read_text());unc=json.loads((B/'uncertainty.json').read_text());base=m[(m.protocol=='coverage_configuration_folds')&(m.model=='training_mean')].iloc[0];res=m[(m.protocol=='coverage_configuration_folds')&(m.model=='nested')].iloc[0]
missing=[r for r in a if not r['eligible']];dump(B/'missing_or_review_required.json',missing)
coverage_counts=collections.Counter(r['n_CO'] for r in manifest if r['domain']=='CuAu' and r['CO_only']);pd.DataFrame([{'n_CO':k,'configuration_support':v,'coverage_per_surface_Cu':k/36,'coverage_per_all_surface_metals':k/68} for k,v in sorted(coverage_counts.items())]).to_csv(B/'CO_coverage_support.csv',index=False)
text=f'''# New record audit and CO coverage experiment — 26 October 2026

126 records were converted from 135 Green folders; 9 lack a usable canonical spectrum. Complete four-layer cases use the author's `convert_layers_to_isaac.sh`, which calls `tools/convert_simulation_xas_run_to_isaac.py`. Other available layer sets use that generic converter plus an explicit canonical-series merge. No layers or spectra are synthesized. A Mac Bash empty-array/`set -u` failure was repaired. The author's revised ISAAC converter is retained, including unreported temperature metadata instead of an invented 0 K.

Thirty-six previously missing local record files were written; existing author records were preserved. The canonical audit versions are in this study.

All original records/raw exports and the archived 87/90 benchmarks are preserved. `records/` holds the **new version**; `converter_original.json.gz` freezes each pre-audit converted record. `record_geometry_audit.json/csv` compares source labels, converter labels, full molecular graphs, root CONTCAR, and every actual FDMNES Crystal input, with one-based atom mapping, hashes, exclusions and ±10% molecular-cutoff sensitivity. `molecule_audit.json` gives measured C–H/O–H distances, bonds and metal contacts. Filenames do not define chemical labels.

## Audit and inclusion

There are {sum(r['domain']=='CuAu' for r in manifest)} usable Cu–Au configurations and {sum(r['domain']=='Cu100' for r in manifest)} Cu100 input-geometry configurations. Cu100 CONTCAR files are absent, but the author located the **original ASE snapshots in `Cu-Ox_updated_structures/100`**. Each of the 17 calculated cases matches its named original snapshot through element-aware coordinate assignment. Stored free-atom forces and constraints are reported; production relaxation stopping/termination evidence is not supplied. Geometry labels are checked against both original snapshots and actual calculation decks. Five Cu100 carbonate/water-containing cases are preserved as records/graphs but excluded from the present 11-species targets because water/CO3 are outside the archived classifier target set. An explicit graph extension verifies two O–H bonds for water and three C–O bonds for CO3; it does not infer formal charge. They are not silently relabeled as CO/OH-only.

Existing supplied labels agree with the successfully traced, stable XAS-input graphs. Unresolved root/XAS mismatches remain excluded; agreement is not asserted there. The historical H coordinate discrepancy is retained only when the exact old root/copy/output hashes match archived lineage; this is not a new geometry match or a widened tolerance. Newly usable aliases and any old exclusions are listed in `comparison_with_90.csv`.

The final-layer population audit retains the strict existing Cu–Au verifier. Cu100 checks all four physical Cu sites per layer, actual input order, explicit convolution source order or complete aggregate log, and raw-source E_cut values within 0.00051 eV. A sorted-index assumption would incorrectly reject one reordered calculation. Raw source files remain author-local; their hashes and recorded log mapping are preserved. Inputs, canonical convolved spectra, geometry copies, absorber tables and available separate convolution logs are captured losslessly in the source archive. Verbose raw/runtime logs and raw site arrays remain local with full hash manifests. Proprietary potentials/binaries are excluded.

## Coverage definition and protocol

The coverage model is conditional on a known **CO-only** configuration. Its target counts intact disconnected CO molecules using the bond graph, not carbon atoms, nominal names or a class count. There are **{len(q)} independent Cu–Au CO-only configurations**; molecules and layers do not increase sample size. Cu–Au exposes 36 Cu and 32 Au sites: the primary target is N(CO)/68 top-surface metal sites. The legacy Cu-only loading N(CO)/36 and molecules/nm² are saved separately. Each element's highest physical layer defines its exposed sites, so a height threshold anchored to the higher Au terrace cannot discard most Cu sites. Cu100 has four top-surface Cu sites. `CO_coverage_support.csv` makes all target levels explicit.

Only spectral arrays enter predictors. The population-weighted native top-two mean uses Cu–Au layers 5+4 and Cu100 layers 4+3. Archived minimum-subtracted unit-area normalization, smoothing and derivatives are unchanged. Native lifetime broadening is preserved; no extra broadening, alignment or stress tuning is applied. A fixed common supported {proto['energy_window_eV'][0]:.1f}–{proto['energy_window_eV'][1]:.1f} eV grid ({proto['energy_points']} points) is linearly interpolated without extrapolation.

The compact declared set is Combined spectrum/CDF/d1/d2 and DCT8/16/24, each with Extra Trees leaf2/4 or standardized Ridge alpha10/100. Ridge scaling is fitted only on each training partition. Three inner grouped folds minimize coverage MAE. Five outer configuration folds, separate leave-coverage-level-out and contact-motif-group-held-out evaluations are saved, with training-mean/median baselines and a fixed Combined/leaf2 control. The contact motif is an audit/splitting label, never a spectral predictor. These evaluations are exploratory reused-configuration tests, not independent prospective confirmation or proven parent transfer.

## Measured coverage results

| Protocol | Model | N | MAE (CO / all surface sites) | RMSE | R² |
|---|---|---:|---:|---:|---:|
'''
for r in m.to_dict('records'):
 text+=f"| {r['protocol']} | {r['model']} | {r['n']} | {r['MAE']:.4f} | {r['RMSE']:.4f} | {r['R2']:.3f} |\n"
text+=f'''
The configuration-fold MAE is **{res.MAE:.4f}** versus the training-mean baseline **{base.MAE:.4f}**. The conditional paired MAE-difference interval is {unc['percentile95'][0]:.4f} to {unc['percentile95'][1]:.4f}; it resamples configurations with predictions fixed and excludes refitting/selection uncertainty. Full held-out arrays, inner selection ranks, group memberships, model files and target-range flags are included. No best outer-score feature is promoted. Negative/out-of-range Ridge predictions, if any, are retained rather than clipped.

Cu100 is a **separate zero-shot domain diagnostic**, with no Cu100 fitting, threshold choice or calibration. Its four CO cases differ in facet, composition, cell size, layer populations and loading range. Three of the four exceed the Cu–Au all-site coverage range. The transfer result is substantially weaker than within-Cu–Au evaluation and does not validate a universal coverage estimator. Molecules/nm² and explicit range flags are saved so normalization conventions are visible. CO-containing mixtures are a separate diagnostic: a CO-only model is not validated for coadsorbate interference.

## Identity and local geometry

The new grouped, fixed-top-two Combined identity evaluation has N={im['n']}, micro/macro F1 **{im['micro_f1']:.4f}/{im['macro_f1']:.4f}**, exact set **{im['exact_set_accuracy']:.4f}**. It uses training-only thresholds and newly declared grouped folds; it is **not a same-fold update** of the archived primary/fixed-top-two scores. All 11 per-label supports and single/mixed recall are saved. Identity estimators use the archived seed 20261020; coverage/contact regressors use 20261026. Cu100 identity transfer has N={em['n']}, micro F1 {em['micro_f1']:.4f}; report its supported-label table rather than treating absent-label macro zeros as tests of rare species. Probabilities are uncalibrated.

The CO local-geometry model predicts **configuration-mean Cu and Au contact counts within 2.25 Å**, not a unique site for every molecule or electronic bond order. Per-molecule targets and 2.15/2.35 Å sensitivity are saved. The measured hollow-series molecule support is {cs['hollow_series_molecule_support']}. In particular, **CuAu2 and Au3 CO environments have zero support**. The current data cannot validate a continuous Cu3→Au3 inference claim. Multiple molecules in one spectrum do not count as independent training examples. Contact errors and mean baselines are in `contact_support.json`.

## Best use of Cu100 and next data

Keep Cu100 as a facet/composition transfer panel and a controlled CO loading series, rather than pooling it into the frozen Cu–Au folds. Original ASE snapshots are now matched; recover the production relaxation stopping/termination record before calling them verified converged calculations. Add independent Cu100 configurations at matched coverages/motifs and independent Cu–Au interface CO configurations, then lock a new prospective transfer set before inspection. The four existing CO loadings alone are insufficient to validate within-facet coverage training. Water/CO3 graphs are now explicitly labeled, but adding them to an identity model requires an extended target schema and sufficient independent support before retraining. Au-rich CO motifs and matched CO/CO+H/CO+OH controls address gaps that more Cu-only examples cannot fill.

## Reproduce and inspect

```sh
python -m tools.coverage_study convert --data-root /path/to/export --output-dir /new/version
python -m tools.coverage_study audit --data-root /path/to/export --output-dir /new/version
python -m tools.coverage_inference --output-dir /new/version
```

The same audit→fit operation is exposed through `run_audited_training(..., recipe='coverage_v1')`, the agent parser/planner and UI. `tools.coverage_inference.predict` reloads a coverage model and requires verified physical populations and full trained energy support. No production scheduler, HPC or portal write is involved. No revised manuscript/SI is claimed in this versioned report; archived manuscript and predictions remain untouched. Figures are generated from saved arrays by `study/code/report_coverage.py` and exported as PDF/SVG/350-dpi PNG.

Method implementation references: [grouped cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data), [training-only mean/median baselines](https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyRegressor.html). Runtime versions are pinned in `protocol.json`; online documentation version is not substituted for the installed environment.
'''
text+='\nA clean Cu–Au top-two record is newly available (`plain`, layers 4+5). `parent_control_inventory.json` finds 34 exact initial-metal matches to its saved relaxed coordinates. The 66 exact initial-metal coordinate families are an inventory, not independent ancestral parents. No physical clean subtraction is performed in this coverage experiment.\n'
(B/'README.md').write_text(text)
