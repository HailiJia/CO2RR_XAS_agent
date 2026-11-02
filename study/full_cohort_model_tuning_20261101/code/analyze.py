"""Report every full-cohort result and fixed-prediction paired group intervals."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tools.spectral_multilabel import SPECIES
B=Path(__file__).resolve().parents[1];R=B.parents[1];D=R/'study/dataset_20261101';O=B/'results';rows=json.load(open(D/'training_manifest.json'));fm=json.load(open(D/'family_map.json'))['families'];fam=np.array([fm[a['configuration']] for a in rows]);domains=np.array([a['domain'] for a in rows]);metric=[];per=[];mixtures=[];predictions={}
def weight(ids):return np.array([1/sum(fam[ids]==v) for v in fam[ids]])
def metrics(y,p,w):
 pr,re,ff,su=precision_recall_fscore_support(y,p,average=None,sample_weight=w,zero_division=0);den=((y+p)*w[:,None]).sum();return dict(micro_f1=2*(y*p*w[:,None]).sum()/den if den else 0,macro_f1=ff.mean(),exact_set_accuracy=np.average(np.all(y==p,axis=1),weights=w),false_positive_labels=((1-y)*p*w[:,None]).sum(),omitted_labels=(y*(1-p)*w[:,None]).sum())
for seed in [20261020,20261021,20261022]:
 for protocol in ['canonical64','all_views64','all_views256','nested']:
  for stage in ['fixed0p5','common']:
   p=D/f'ml/trees_{seed}_updated_full_cohort_{stage}.npz' if protocol=='canonical64' else O/f'{protocol}_{seed}_{stage}.npz';z=np.load(p);ids=z['indices'];y=z['truth'];prediction=z['prediction'];assert np.array_equal(ids,np.arange(141));predictions[protocol,seed,stage]=(y,prediction)
   subsets={'all':np.ones(141,bool),'CuAu':domains=='CuAu','Cu100':domains=='Cu100','single':y.sum(axis=1)==1,'mixed':y.sum(axis=1)>1,'clean':y.sum(axis=1)==0}
   for subset,mask in subsets.items():
    ids=np.flatnonzero(mask)
    for weighting,w in [('record',np.ones(len(ids))),('near_group',weight(ids))]:
     meta=dict(protocol=protocol,seed=seed,threshold=stage,subset=subset,weighting=weighting,n_records=len(ids),n_groups=len(set(fam[ids])));metric.append({**meta,**metrics(y[ids],prediction[ids],w)});pr,re,ff,su=precision_recall_fscore_support(y[ids],prediction[ids],average=None,sample_weight=w,zero_division=0)
     for j,sp in enumerate(SPECIES):per.append({**meta,'species':sp,'precision':pr[j],'recall':re[j],'f1':ff[j],'support':su[j]})
     if subset in ['CuAu','Cu100']:
      jj=[SPECIES.index(sp) for sp in ['CO','H','O','OH']];metric.append({**{**meta,'subset':subset+'_common4'},**metrics(y[ids][:,jj],prediction[ids][:,jj],w)})
   for i in np.flatnonzero(y.sum(axis=1)>1):mixtures.append(dict(protocol=protocol,seed=seed,threshold=stage,configuration=rows[i]['configuration'],near_group=fam[i],truth=';'.join(np.array(SPECIES)[y[i]>0]),prediction=';'.join(np.array(SPECIES)[prediction[i]>0]),omitted=';'.join(np.array(SPECIES)[(y[i]>0)&(prediction[i]==0)]),false_positive=';'.join(np.array(SPECIES)[(y[i]==0)&(prediction[i]>0)]),exact=bool(np.array_equal(y[i],prediction[i]))))
pd.DataFrame(metric).to_csv(B/'metrics.csv',index=False);pd.DataFrame(per).to_csv(B/'per_label.csv',index=False);pd.DataFrame(mixtures).to_csv(B/'mixture_predictions.csv',index=False)
def contribution(y,p):
 c=[];ex=[]
 for g in sorted(set(fam)):
  mask=fam==g;c.append(np.stack([(y[mask]*p[mask]).mean(axis=0),y[mask].mean(axis=0),p[mask].mean(axis=0)]));ex.append(np.all(y[mask]==p[mask],axis=1).mean())
 return np.array(c),np.array(ex)
def value(c,ex):
 tp=c[:,0].sum(axis=0);den=c[:,1:].sum(axis=(0,1));return np.array([2*tp.sum()/max(1e-12,den.sum()),np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(),ex.mean()])
interval=[]
for stage in ['fixed0p5','common']:
 for protocol in ['all_views64','all_views256','nested']:
  pair=[(contribution(*predictions['canonical64',seed,stage]),contribution(*predictions[protocol,seed,stage])) for seed in [20261020,20261021,20261022]];delta=np.array([value(*b)-value(*a) for a,b in pair]);rng=np.random.default_rng(11012026);bs=[]
  for _ in range(2000):
   ix=rng.integers(0,135,135);bs.append(np.mean([value(b[0][ix],b[1][ix])-value(a[0][ix],a[1][ix]) for a,b in pair],axis=0))
  for j,k in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):interval.append(dict(protocol=protocol,threshold=stage,metric=k,baseline=np.mean([value(*a)[j] for a,b in pair]),modified=np.mean([value(*b)[j] for a,b in pair]),difference=delta[:,j].mean(),lower=np.quantile(bs,.025,axis=0)[j],upper=np.quantile(bs,.975,axis=0)[j],seed_refit_difference_SD=delta[:,j].std(ddof=1),scope='same141 queries; paired near-group bootstrap conditional on fixed seed predictions, not independent-parent/refit/experimental uncertainty'))
pd.DataFrame(interval).to_csv(B/'paired_intervals.csv',index=False)
d=pd.DataFrame(metric);d=d.query("subset=='all' and weighting=='near_group'");d.groupby(['protocol','threshold'])[['micro_f1','macro_f1','exact_set_accuracy']].agg(['mean','std']).to_csv(B/'NUMBERS.csv');print(d.groupby(['protocol','threshold'])[['micro_f1','macro_f1','exact_set_accuracy']].mean().to_string());print(pd.DataFrame(interval).query("threshold=='common' and metric=='micro_f1'").to_string(index=False))
