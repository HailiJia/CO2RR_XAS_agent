from pathlib import Path
import json,hashlib,gzip,tarfile,sys
import numpy as np,pandas as pd
repo=Path(__file__).resolve().parents[2];root=repo/'study/retraining_20261022';b=root/'refit';d=b/'dataset';models=b/'models';sys.path.insert(0,str(repo))
from tools.spectral_multilabel import f1_metrics,SPECIES
from tools.scientific_extensions import single_mixed
from tools.controlled_comparisons import deck_settings
cv=pd.read_csv(d/'cv_manifest.csv');oldcv=pd.read_csv(repo/'study/manuscript_package/figure_data/cv_manifest.csv');old=np.load(repo/'study/manuscript_package/figure_data/oof_independent_selected_micro.npz');new=np.load(models/'oof.npz');hist=json.load(open(d/'historical_lineage.json'))['records_by_legacy_geometry'];sources={r['path']:r for r in json.load(open(d/'source_manifest.json'))};rows=[];ii=[];jj=[]
for i,row in cv.iterrows():
 if pd.isna(row.prior_configuration):continue
 j=oldcv.index[oldcv.configuration==row.prior_configuration][0];ii.append(i);jj.append(j);previous=hist[row.structure_hash];same=[]
 for l in [2,3,4,5]:same.append(sources[row.configuration+f'/XAS/FDMNES/green/layer_{l}/Cu_layer{l}_conv.txt']['sha256']==previous['convolved_output_sha256'][str(l)])
 assert all(same) and np.array_equal(new['truth'][i],old['truth'][j])
 rows.append({'current_configuration':row.configuration,'prior_configuration':row.prior_configuration,'alias_renamed':row.configuration!=row.prior_configuration,'fold_preserved':int(row.fold)==int(oldcv.iloc[j].fold),'target_identical':True,'all_four_native_outputs_byte_identical':True,'old_prediction':';'.join(np.array(SPECIES)[old['prediction'][j].astype(bool)]),'new_prediction':';'.join(np.array(SPECIES)[new['prediction'][i].astype(bool)])})
pd.DataFrame(rows).to_csv(root/'before_after_predictions.csv',index=False)
y=new['truth'][ii];p=new['prediction'][ii];baseline=old['prediction'][jj];rng=np.random.default_rng(20261022);keys=['micro_f1','macro_f1','exact_set_accuracy'];bootstrap=[]
for _ in range(1000):
 ix=rng.integers(0,len(y),len(y));a=f1_metrics(y[ix],p[ix]);c=f1_metrics(y[ix],baseline[ix]);bootstrap.append([a[k]-c[k] for k in keys])
a=f1_metrics(y,p);c=f1_metrics(y,baseline);bootstrap=np.array(bootstrap);pd.DataFrame([{'metric':k,'updated_old_subset':a[k],'frozen_primary':c[k],'difference':a[k]-c[k],'conditional_95_low':float(np.quantile(bootstrap[:,j],.025)),'conditional_95_high':float(np.quantile(bootstrap[:,j],.975)),'scope':'paired configuration bootstrap of fixed predictions on same 87 old configurations; excludes refit/selection uncertainty'} for j,k in enumerate(keys)]).to_csv(root/'paired_old_subset_comparison.csv',index=False)
recalls=single_mixed(old['truth'],old['prediction'],'frozen_primary_87')+single_mixed(new['truth'],new['prediction'],'updated_refit_90');pd.DataFrame(recalls).to_csv(root/'single_mixed_comparison.csv',index=False)
h=np.load(models/'new_configuration_holdout.npz');pd.DataFrame([{'configuration':config,'truth':';'.join(np.array(SPECIES)[t.astype(bool)]),'prediction':';'.join(np.array(SPECIES)[p.astype(bool)]) or '(none)','prediction_set_json':json.dumps(np.array(SPECIES)[p.astype(bool)].tolist()),'exact_set':bool(np.array_equal(t,p)),'interpretation':'not prospectively locked; three CO-only configurations do not test rare/intermediate recovery'} for config,t,p in zip(h['configuration'],h['truth'],h['prediction'])]).to_csv(root/'new_configuration_predictions.csv',index=False)
settings=[]
with tarfile.open(d/'source_evidence.tar') as tar:
 for row in cv.itertuples():
  controls=[]
  for layer in [2,3,4,5]:
   key=row.configuration+f'/XAS/FDMNES/green/layer_{layer}/Cu_layer{layer}_in.txt';sha=sources[key]['sha256'];text=gzip.decompress(tar.extractfile('source_objects/'+sha+'.gz').read()).decode();before=[];skip=False
   for line in text.splitlines():
    t=line.strip()
    if t.lower()=='crystal':break
    if skip:skip=False;continue
    if t.lower()=='filout':skip=True;continue
    if t and not t.startswith('!'):before.append(t.lower())
   controls.append(before)
  settings.append({'configuration':row.configuration,'settings_sha256':hashlib.sha256(json.dumps(controls,sort_keys=True).encode()).hexdigest(),'controls':json.dumps(controls)})
assert len({r['settings_sha256'] for r in settings})==1;pd.DataFrame(settings).to_csv(root/'numerical_control_comparison.csv',index=False)
summary=json.load(open(d/'summary.json'));summary.update(prior_output_arrays_byte_identical=87,prior_folds_preserved=87,renamed_retained_aliases=sum(r['alias_renamed'] for r in rows),same_declared_FDMNES_controls_for_all_configurations=True,raw_recovered_vs_new='two newly completed supplied 2CO records and one recovered existing raw CO calculation; no new HPC work by agent',new_supports=dict(zip(SPECIES,map(int,new['truth'].sum(axis=0)))))
(root/'change_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2));print(pd.read_csv(root/'paired_old_subset_comparison.csv').to_string(index=False));print(pd.read_csv(root/'new_configuration_predictions.csv').to_string(index=False))
