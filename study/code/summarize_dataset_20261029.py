"""Summarize fixed protocols and paired old-case comparisons, never choose a seed."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import f1_score,accuracy_score,mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import dump
from tools.spectral_multilabel import SPECIES,f1_metrics
from tools.differential_training import cls_score,features
B=ROOT/'study/dataset_20261029';M=B/'ml';rows=json.load(open(B/'training_manifest.json'));z=np.load(B/'training_arrays.npz');metrics=pd.read_csv(M/'metrics.csv');subs=pd.read_csv(M/'identity_subsets.csv');per=pd.read_csv(M/'per_species_single_mixed.csv');seed_summary=subs[(subs.stage=='inner_retuned_threshold')&(subs.evaluation_subset.isin(['all','old99','new_only','CuAu','Cu100']))].groupby(['cohort','evaluation_subset']).agg(n=('n','first'),micro_F1_mean=('micro_f1','mean'),micro_F1_sd=('micro_f1','std'),macro_F1_mean=('macro_f1','mean'),exact_mean=('exact_set_accuracy','mean')).reset_index();seed_summary.to_csv(B/'seed_summary.csv',index=False)
old_manifest=json.load(open(ROOT/'study/coverage_20261026/model_configuration_manifest.json'));old_arrays=np.load(ROOT/'study/coverage_20261026/model_inputs.npz');ix={r['group']:j for j,r in enumerate(old_manifest)};floats=[]
for i,r in enumerate(rows):
 if r['old99']:
  a=z['raw_native'][i];b=old_arrays['raw'][ix[r['group']]];floats.append({'configuration':r['configuration'],'max_relative_intensity_difference':np.max(abs(a-b))/np.max(abs(b)),'max_energy_grid_difference_eV':np.max(abs(z['energy']-old_arrays['energy'])),'explanation':'old archived model_inputs had interpolation onto np.arange grid; current direct native grids differ only at floating-point accumulation scale'})
pd.DataFrame(floats).to_csv(B/'baseline_float_grid_comparison.csv',index=False)
# Paired bootstrap on the SAME old99 query configurations, fixed predictions.
ci=[];rng=np.random.default_rng(20261029)
for stage in ['threshold0p5','transferred_baseline_threshold','inner_retuned_threshold']:
 for seed in [20261020,20261021,20261022]:
  a=np.load(M/f'identity_baseline99_CuAu_normalized_raw_{stage}_{seed}.npz');b=np.load(M/f'identity_refresh_CuAu_normalized_raw_{stage}_{seed}.npz');lookup={v:j for j,v in enumerate(b['configuration'])};pred=b['prediction'][[lookup[v] for v in a['configuration']]];truth=a['truth'];base=a['prediction'];values=[]
  for _ in range(1000):
   idx=rng.integers(len(truth),size=len(truth));values.append(f1_score(truth[idx],pred[idx],average='micro',zero_division=0)-f1_score(truth[idx],base[idx],average='micro',zero_division=0))
  ci.append({'stage':stage,'seed':seed,'n_same_old_cases':len(truth),'delta_micro_F1':f1_score(truth,pred,average='micro',zero_division=0)-f1_score(truth,base,average='micro',zero_division=0),'conditional_95_low':np.quantile(values,.025),'conditional_95_high':np.quantile(values,.975),'wrong_to_correct':int(np.sum(np.any(base!=truth,axis=1)&np.all(pred==truth,axis=1))),'correct_to_wrong':int(np.sum(np.all(base==truth,axis=1)&np.any(pred!=truth,axis=1))),'scope':'paired fixed predictions; excludes refitting/parent transfer uncertainty'})
pd.DataFrame(ci).to_csv(B/'paired_old99_comparison.csv',index=False)
# Four supported common labels and equal-domain aggregates; other targets not silently counted as tested.
domain=[]
for path in M.glob('identity_*.npz'):
 q=np.load(path);ids=q['pair_indices'];bits=[SPECIES.index(s) for s in ['CO','H','O','OH']];doms={}
 for dom in ['CuAu','Cu100']:
  mask=np.array([rows[i]['domain']==dom for i in ids]);
  if not mask.any():continue
  yy=q['truth'][mask][:,bits];pp=q['prediction'][mask][:,bits];r={'artifact':path.name,'domain':dom,'n':int(mask.sum()),'labels':'CO;H;O;OH',**f1_metrics(yy,pp)};domain.append(r);doms[dom]=r
 if len(doms)==2:domain.append({'artifact':path.name,'domain':'equal_domain_mean','n':len(ids),'labels':'CO;H;O;OH',**{k:(doms['CuAu'][k]+doms['Cu100'][k])/2 for k in ['micro_f1','macro_f1','exact_set_accuracy']}})
pd.DataFrame(domain).to_csv(B/'common_label_domain_metrics.csv',index=False)
# Covariate/interference diagnostics: saved CO-only fold ensemble on mixtures, no mixture labels in any fit.
mixtures=[]
for cohort in ['refresh_CuAu','refresh_joint']:
 for rep in ['normalized_raw','physical_raw','delta']:
  from tools.dataset_refresh_training import feature_bank
  bank=feature_bank(z['raw_native'],z['energy']) if rep=='normalized_raw' else features(z['raw_physical'] if rep=='physical_raw' else z['delta'],z['energy'])
  if rep=='normalized_raw':
   from scipy.fft import dct
   bank={'Combined':bank['shape_CDF_derivatives'],'DCT16':dct(bank['normalized'],type=2,norm='ortho',axis=1)[:,:16]}
  ids=np.array([i for i,r in enumerate(rows) if r['n_CO']>0 and not r['CO_only'] and (cohort=='refresh_joint' or r['domain']=='CuAu')]);pred=[]
  for path in sorted((M/'models').glob(f'coverage_{cohort}_{rep}_fold*.joblib')):
   m=joblib.load(path);assert not set(m['train_groups'])&{rows[i]['group'] for i in ids};pred.append(m['model'].predict(bank[m['recipe'][0]][ids]))
  pp=np.mean(pred,axis=0);truth=np.array([rows[i]['coverage_ML'] for i in ids]);mixtures.append({'cohort':cohort,'representation':rep,'n_CO_containing_mixtures':len(ids),'MAE_CO_per_all_surface_metals':mean_absolute_error(truth,pp),'interpretation':'frozen CO-only ensemble; mixture interference diagnostic, no mixture-trained coverage claim'})
  np.savez_compressed(M/f'mixture_{cohort}_{rep}.npz',pair_indices=ids,truth=truth,prediction=pp,fold_model_predictions=np.array(pred))
pd.DataFrame(mixtures).to_csv(B/'mixture_metrics.csv',index=False)
# Simple review figure; no manuscript/SI modification.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,2,figsize=(9,3.6));fixed=seed_summary[(seed_summary.evaluation_subset=='old99')&(seed_summary.cohort.isin(['baseline99_CuAu','refresh_CuAu']))];ax[0].bar(np.arange(len(fixed)),fixed.micro_F1_mean,yerr=fixed.micro_F1_sd,color=['#777777','#25834e'],capsize=3);ax[0].set(xticks=np.arange(len(fixed)),xticklabels=['Old99 training','Expanded Au–Cu training'],ylim=(0,1),ylabel='Micro F1 on identical old99 queries',title='Fixed top-two Combined\nRetuned thresholds; three seeds')
a=metrics[(metrics.task=='CO_coverage')&(metrics.stage=='compact_nested')&(metrics.representation=='normalized_raw')];ax[1].bar(np.arange(len(a)),a.MAE,color=['#777777','#25834e','#447aa1']);ax[1].set(xticks=np.arange(len(a)),xticklabels=['Old25 CO','New28 CO','Joint32 CO'],ylabel='MAE (CO/all surface metal sites)',title='Different cohort sizes; paired errors also saved');fig.tight_layout();fig.savefig(B/'summary.png',dpi=300);plt.close(fig)
print(seed_summary.to_string(index=False))
# Paired CO-only errors and explicit denominator conversions, no new fit.
a=np.load(M/'CO_coverage_baseline99_CuAu_normalized_raw_compact_nested_20261027.npz');b=np.load(M/'CO_coverage_refresh_CuAu_normalized_raw_compact_nested_20261027.npz');lookup={v:i for i,v in enumerate(b['configuration'])};p=b['prediction'][[lookup[v] for v in a['configuration']]];pd.DataFrame({'configuration':a['configuration'],'truth_CO_per_surface_metal':a['truth'],'baseline_prediction':a['prediction'],'expanded_prediction':p,'baseline_absolute_error':abs(a['truth']-a['prediction']),'expanded_absolute_error':abs(a['truth']-p)}).to_csv(B/'paired_coverage_old25.csv',index=False)
rng=np.random.default_rng(20261030);difference=abs(p-a['truth'])-abs(a['prediction']-a['truth']);values=[np.mean(difference[rng.integers(len(difference),size=len(difference))]) for _ in range(1000)];dump(B/'paired_coverage_interval.json',{'n_same_old_configurations':len(difference),'baseline_MAE':float(np.mean(abs(a['prediction']-a['truth']))),'expanded_MAE':float(np.mean(abs(p-a['truth']))),'MAE_difference':float(np.mean(difference)),'conditional_95_interval':np.quantile(values,[.025,.975]).tolist(),'scope':'fixed predictions, configuration resampling; excludes refitting and parent transfer uncertainty'})
conversions=[]
for path in M.glob('CO_coverage_*.npz'):
 q=np.load(path)
 for i,y,pred in zip(q['pair_indices'],q['truth'],q['prediction']):
  r=rows[int(i)];conversions.append({'artifact':path.name,'configuration':r['configuration'],'domain':r['domain'],'truth_CO_per_surface_metal':float(y),'prediction_CO_per_surface_metal':float(pred),'truth_CO_per_surface_Cu_loading':r['coverage_per_surface_Cu'],'prediction_CO_per_surface_Cu_loading':float(pred)*r['surface_sites']/r['surface_Cu'],'truth_CO_per_nm2':r['coverage_CO_per_nm2'],'prediction_CO_per_nm2':float(pred)*r['surface_sites']/(r['cell_area_A2']/100),'note':'same model, deterministic unit/denominator conversion using known substrate population/area; Cu loading is not literal Cu-site occupancy'})
pd.DataFrame(conversions).to_csv(B/'coverage_heldout_units.csv',index=False)

fixed=subs[(subs.evaluation_subset=="old99")&subs.cohort.isin(["baseline99_CuAu","refresh_CuAu"])&subs.stage.isin(["threshold0p5","transferred_baseline_threshold","inner_retuned_threshold"])];fixed.groupby(["stage","cohort"])[["micro_f1","macro_f1","exact_set_accuracy"]].agg(["mean","std"]).to_csv(B/"fixed_threshold_comparison.csv")
