"""Author-specified Combined control and paired reports, no test-score selection."""
from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.metrics import precision_recall_fscore_support
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fixed_top_two import load_data,fit,CANDIDATES,SEEDS,STAGES,digest
from tools.spectral_multilabel import SPECIES,binary_scores,f1_metrics
from tools.scientific_extensions import single_mixed
B=ROOT/'study/top_two_20261023';E=B/'experiments'
ANCHORS=('fixed_Combined_threshold0p5','fixed_Combined_transferred87_threshold','fixed_Combined_retuned_threshold')


def subset_report(y,p):
 j=SPECIES.index('CO');tp=int(((y[:,j]==1)&(p[:,j]==1)).sum());fp=int(((y[:,j]==0)&(p[:,j]==1)).sum());fn=int(((y[:,j]==1)&(p[:,j]==0)).sum())
 return {'configurations':len(y),'CO_support':int(y[:,j].sum()),'CO_TP':tp,'CO_FP':fp,'CO_FN':fn,'CO_precision':tp/(tp+fp) if tp+fp else None,'CO_recall':tp/(tp+fn) if tp+fn else None,'CO_f1':2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,'exact_set_accuracy':float(np.all(y==p,axis=1).mean()),'other_positive_species_tested':False,'other_species_false_positives':int(p[:,np.arange(len(SPECIES))!=j].sum()),'macro_F1_display':'not reported: ten labels have no positives'}


def historical_correction():
 old=np.load(ROOT/'study/manuscript_package/figure_data/oof_independent_selected_micro.npz');v=ROOT/'study/retraining_20261022';new=np.load(v/'refit/models/oof.npz');cv=pd.read_csv(v/'refit/dataset/cv_manifest.csv');oldcv=pd.read_csv(ROOT/'study/manuscript_package/figure_data/cv_manifest.csv');lookup={r.prior_configuration:i for i,r in cv.iterrows() if isinstance(r.prior_configuration,str)};ix=np.array([lookup[c] for c in oldcv.configuration]);y=old['truth'];p=old['prediction'];q=new['prediction'][ix];rows=[]
 for i,c in enumerate(oldcv.configuration):
  before=bool(np.array_equal(y[i],p[i]));after=bool(np.array_equal(y[i],q[i]));rows.append({'configuration':c,'truth':json.dumps(np.array(SPECIES)[y[i].astype(bool)].tolist()),'old_prediction':json.dumps(np.array(SPECIES)[p[i].astype(bool)].tolist()),'v3_prediction':json.dumps(np.array(SPECIES)[q[i].astype(bool)].tolist()),'old_exact':before,'v3_exact':after,'new_false_H_or_OH':bool((y[i].sum()==1) and y[i,SPECIES.index('CO')] and any(q[i,SPECIES.index(s)] and not p[i,SPECIES.index(s)] for s in ['H','OH']))})
 pd.DataFrame(rows).to_csv(B/'historical_v3_prediction_changes.csv',index=False);h=np.load(v/'refit/models/new_configuration_holdout.npz');report=subset_report(h['truth'],h['prediction']);report.update(scope='historical v3 old-only/new-three holdout; not the fixed-top-two result',became_correct=sum(not r['old_exact'] and r['v3_exact'] for r in rows),became_wrong=sum(r['old_exact'] and not r['v3_exact'] for r in rows),CO_only_cases_acquiring_false_H_OH=sum(r['new_false_H_or_OH'] for r in rows),legacy_eleven_label_macro_F1=0.8/11,macro_explanation='CO F1=0.8 divided by all 11 labels gives 0.0727 because the ten unrepresented labels receive zero; this does not test their recovery')
 (B/'historical_v3_reporting_correction.json').write_text(json.dumps(report,indent=2)+'\n')
 original=pd.read_csv(ROOT/'CO2RR_ML/updated_dataset_20261020/nested_selections.csv').query("selector=='independent_selected' and objective=='micro'").set_index('fold');updated=json.load(open(v/'refit/models/selections.json'));changes=[]
 for r in updated:
  if not r['partition'].startswith('outer_fold_'):continue
  k=int(r['partition'].rsplit('_',1)[1]);previous=original.loc[k];changes.append({'fold':k,'old_depth':previous.depth,'old_feature':previous.feature,'v3_depth':r['candidate'].split('::')[0],'v3_feature':r['candidate'].split('::')[1],'representation_changed':previous.depth+'::'+previous.feature!=r['candidate']})
 pd.DataFrame(changes).to_csv(B/'historical_depth_selector_changes.csv',index=False)


def main(experiment_dir=None,report_dir=None):
 global B,E
 if experiment_dir is not None:E=Path(experiment_dir)
 if report_dir is not None:B=Path(report_dir)
 B.mkdir(parents=True,exist_ok=True)
 data=load_data(ROOT);anchor=CANDIDATES.index(('shape_CDF_derivatives',1));metrics=[];selection=[];candidate_rows=[];subsets=[];recalls=[];per=[];paired=[];stability=[];saved=[];rng=np.random.default_rng(20261023)
 for n in [87,90]:
  for seed in SEEDS:
   d=E/f'cohort{n}_seed{seed}';z=np.load(d/'candidate_oof.npz');scores=z['score'][anchor];y=z['truth'];fold=z['fold'];preds={stage:np.zeros_like(y) for stage in ANCHORS}
   for k in range(5):
    te=np.flatnonzero(fold==k);tr=np.flatnonzero(fold!=k);current=np.load(d/f'fold{k}/inner.npz')['thresholds'][anchor];old=np.load(E/f'cohort87_seed{seed}/fold{k}/inner.npz')['thresholds'][anchor]
    for stage in ANCHORS:
     thresholds=np.full(len(SPECIES),.5) if stage.endswith('threshold0p5') else old if 'transferred87' in stage else current;preds[stage][te]=scores[te]>=thresholds;selection.append({'cohort':n,'seed':seed,'fold':k,'stage':stage,'feature':'shape_CDF_derivatives','leaf':1,'thresholds':json.dumps(thresholds.tolist()),'train_indices':json.dumps(tr.tolist()),'test_indices':json.dumps(te.tolist())})
    model=fit(data['bank']['shape_CDF_derivatives'][tr],y[tr],1,seed);observed=binary_scores(model,data['bank']['shape_CDF_derivatives'][te]);assert np.array_equal(observed,scores[te]);p=d/f'fold{k}/author_Combined_leaf1.joblib';bundle={'format':'co2rr-multilabel-v1','fit':model,'energy':data['energy'],'species':list(SPECIES),'depth':'top_2','feature':'shape_CDF_derivatives','thresholds':current,'provenance':{'cohort':n,'seed':seed,'fold':k,'training_indices':tr.tolist(),'test_indices':te.tolist(),'source_snapshot_indices':data['source_indices'].tolist(),'candidate':anchor,'experiment':'author Combined fixed-top-two control'},'normalization':'raw top-two population average before unit-area normalization'};joblib.dump(bundle,p,compress=3);saved.append({'path':str(p.relative_to(E)),'sha256':digest(p),'replay_identical':True})
   for stage,p in preds.items():np.savez_compressed(d/(stage+'.npz'),truth=y,prediction=p,score=scores,fold=fold,configuration=data['configuration'][:n],species=SPECIES)
   for stage in list(ANCHORS)+['original_feature_selection','regularized_selection']:
    p=np.load(d/(stage+'.npz'))['prediction'];metrics.append({'cohort':n,'seed':seed,'stage':stage,**f1_metrics(y,p)});recalls.extend([{'cohort':n,'seed':seed,**r} for r in single_mixed(y,p,stage)]);pr,re,ff,su=precision_recall_fscore_support(y,p,zero_division=0);per.extend([{'cohort':n,'seed':seed,'stage':stage,'species':s,'precision':pr[j],'recall':re[j],'f1':ff[j],'support':int(su[j]),'FP':int(((y[:,j]==0)&(p[:,j]==1)).sum()),'FN':int(((y[:,j]==1)&(p[:,j]==0)).sum())} for j,s in enumerate(SPECIES)])
    if n==90:subsets.append({'seed':seed,'stage':stage,**subset_report(y[87:],p[87:])})
   for j,(feature,leaf) in enumerate(CANDIDATES):candidate_rows.append({'cohort':n,'seed':seed,'feature':feature,'leaf':leaf,**f1_metrics(y,z['prediction'][j])})
 for seed in SEEDS:
  for stage in list(ANCHORS)+['original_feature_selection','regularized_selection']:
   a=np.load(E/f'cohort87_seed{seed}'/(stage+'.npz'));b=np.load(E/f'cohort90_seed{seed}'/(stage+'.npz'));y=a['truth'];p=a['prediction'];q=b['prediction'][:87];delta={k:f1_metrics(y,q)[k]-f1_metrics(y,p)[k] for k in f1_metrics(y,p)};sample=[]
   for _ in range(1000):
    ix=rng.integers(0,87,87);v=f1_metrics(y[ix],q[ix]);w=f1_metrics(y[ix],p[ix]);sample.append([v[k]-w[k] for k in delta])
   sample=np.array(sample)
   for j,key in enumerate(delta):paired.append({'seed':seed,'stage':stage,'metric':key,'difference90_minus87_on_same87':delta[key],'conditional95_low':np.quantile(sample[:,j],.025),'conditional95_high':np.quantile(sample[:,j],.975),'scope':'fixed-prediction paired bootstrap; excludes refit uncertainty'})
 for n in [87,90]:
  for stage in list(ANCHORS)+['original_feature_selection','regularized_selection']:
   ps=np.array([np.load(E/f'cohort{n}_seed{seed}'/(stage+'.npz'))['prediction'] for seed in SEEDS]);variation=np.any(ps!=ps[0],axis=(0,2));stability.append({'cohort':n,'stage':stage,'configurations_with_any_seed_identity_change':int(variation.sum()),'total_configurations':n,'seed_variation_is_refit_not_independent_validation':True})
 pd.DataFrame(metrics).to_csv(B/'metrics.csv',index=False);pd.DataFrame(candidate_rows).to_csv(B/'feature_leaf_ablation.csv',index=False);pd.DataFrame(subsets).to_csv(B/'new_only_CO_metrics.csv',index=False);pd.DataFrame(per).to_csv(B/'per_species.csv',index=False);pd.DataFrame(recalls).to_csv(B/'single_mixed_recall.csv',index=False);pd.DataFrame(selection).to_csv(B/'Combined_control_thresholds.csv',index=False);pd.DataFrame(paired).to_csv(B/'paired_old_configuration_intervals.csv',index=False);pd.DataFrame(stability).to_csv(B/'seed_identity_stability.csv',index=False);(B/'Combined_models.json').write_text(json.dumps(saved,indent=2)+'\n');historical_correction()
 table=pd.DataFrame(metrics).groupby(['cohort','stage'])[['micro_f1','macro_f1','exact_set_accuracy']].agg(['mean','std','min','max']);table.to_csv(B/'seed_summary.csv');print(pd.DataFrame(metrics).to_string(index=False))
if __name__=='__main__':main()
