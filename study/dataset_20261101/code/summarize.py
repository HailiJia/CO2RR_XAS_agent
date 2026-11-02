"""Equal-group and record metrics; paired conditional bootstrap on identical queries."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tools.spectral_multilabel import SPECIES
B=Path(__file__).resolve().parents[1];r=json.load(open(B/'training_manifest.json'));fm=json.load(open(B/'family_map.json'))['families'];fam=np.array([fm[a['configuration']] for a in r]);dom=np.array([a['domain'] for a in r]);rows=[];per=[];mix=[]
def weights(ids):return np.array([1/sum(fam[ids]==g) for g in fam[ids]])
def metrics(y,p,w):
 pr,re,ff,su=precision_recall_fscore_support(y,p,sample_weight=w,average=None,zero_division=0);den=((y+p)*w[:,None]).sum();return dict(micro_f1=2*(y*p*w[:,None]).sum()/den if den else 0,macro_f1=ff.mean(),exact_set_accuracy=np.average(np.all(y==p,axis=1),weights=w),false_positive_labels=((1-y)*p*w[:,None]).sum(),omitted_labels=(y*(1-p)*w[:,None]).sum())
for path in (B/'ml').glob('*.npz'):
 q=np.load(path);ids=q['indices'];y=q['truth'];p=q['prediction'];parts=path.stem.split('_');kind=parts[0];seed=int(parts[1]);stage=parts[-1];protocol='_'.join(parts[2:-1]);meta=dict(kind=kind,seed=seed,stage=stage,protocol=protocol)
 subsets={'all':np.ones(len(ids),bool),'CuAu':dom[ids]=='CuAu','Cu100':dom[ids]=='Cu100','single':y.sum(axis=1)==1,'mixed':y.sum(axis=1)>1,'clean':y.sum(axis=1)==0,'added':np.array([r[i]['old_index'] is None for i in ids])}
 for subset,mask in subsets.items():
  if not mask.any():continue
  ii=ids[mask];yy=y[mask];pp=p[mask]
  for weighting,w in [('record',np.ones(len(ii))),('near_group',weights(ii))]:
   row={**meta,'subset':subset,'weighting':weighting,'n_records':len(ii),'n_groups':len(set(fam[ii])),**metrics(yy,pp,w)};rows.append(row);pr,re,ff,su=precision_recall_fscore_support(yy,pp,sample_weight=w,average=None,zero_division=0)
   for j,sp in enumerate(SPECIES):per.append({**meta,'subset':subset,'weighting':weighting,'species':sp,'precision':pr[j],'recall':re[j],'f1':ff[j],'support':su[j]})
   if subset in ['CuAu','Cu100']:
    common=[SPECIES.index(s) for s in ['CO','H','O','OH']];rows.append({**meta,'subset':subset+'_common4','weighting':weighting,'n_records':len(ii),'n_groups':len(set(fam[ii])),**metrics(yy[:,common],pp[:,common],w)})
 if protocol=='updated_full_cohort':
  for j,i in enumerate(ids):
   if y[j].sum()>1:mix.append({**meta,'configuration':r[i]['configuration'],'near_group':fam[i],'true_set':';'.join(np.array(SPECIES)[y[j]>0]),'prediction':';'.join(np.array(SPECIES)[p[j]>0]),'omitted':';'.join(np.array(SPECIES)[(y[j]>0)&(p[j]==0)]),'false_positive':';'.join(np.array(SPECIES)[(y[j]==0)&(p[j]>0)]),'exact':bool(np.array_equal(y[j],p[j]))})
pd.DataFrame(rows).to_csv(B/'metrics.csv',index=False);pd.DataFrame(per).to_csv(B/'per_label.csv',index=False);pd.DataFrame(mix).to_csv(B/'mixture_predictions.csv',index=False)
# Paired group contribution bootstrap; jointly resample the same groups across all seeds.
def contributions(q):
 ids=q['indices'];y=q['truth'];p=q['prediction'];c=[];ex=[]
 for g in sorted(set(fam[ids])):
  m=fam[ids]==g;c.append(np.stack([(y[m]*p[m]).mean(axis=0),y[m].mean(axis=0),p[m].mean(axis=0)]));ex.append(np.all(y[m]==p[m],axis=1).mean())
 return np.array(c),np.array(ex)
def value(c,ex):
 tp=c[:,0].sum(axis=0);den=c[:,1:].sum(axis=(0,1));return np.array([2*tp.sum()/max(1e-12,den.sum()),np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(),ex.mean()])
ci=[]
for kind in ['trees','logistic']:
 for stage in ['fixed0p5','transferred','common']:
  pairs=[]
  for seed in [20261020,20261021,20261022]:
   a=np.load(B/f'ml/{kind}_{seed}_previous_training_same_queries_{stage}.npz');b=np.load(B/f'ml/{kind}_{seed}_updated_training_same_queries_{stage}.npz');assert np.array_equal(a['indices'],b['indices']);pairs.append((contributions(a),contributions(b)))
  delta=np.array([value(*b)-value(*a) for a,b in pairs]);rng=np.random.default_rng(11101);boot=[]
  for _ in range(2000):
   ix=rng.integers(0,len(pairs[0][0][0]),len(pairs[0][0][0]));boot.append(np.mean([value(b[0][ix],b[1][ix])-value(a[0][ix],a[1][ix]) for a,b in pairs],axis=0))
  for j,k in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):ci.append(dict(kind=kind,stage=stage,metric=k,old=np.mean([value(*a)[j] for a,b in pairs]),updated_same_queries=np.mean([value(*b)[j] for a,b in pairs]),difference=delta[:,j].mean(),lower=np.quantile(boot,.025,axis=0)[j],upper=np.quantile(boot,.975,axis=0)[j],seed_refit_SD=delta[:,j].std(ddof=1),scope='fixed-prediction near-group bootstrap; not refit/experimental/new-parent uncertainty'))
pd.DataFrame(ci).to_csv(B/'paired_intervals.csv',index=False)
# Records, groups, pure cases and molecules are distinct support units.
support=[]
for sp in SPECIES:
 for cohort,pool in [('previous',[a for a in r if a['old_index'] is not None]),('current',r)]:
  positive=[a for a in pool if sp in a['identity_labels']];pure=[a for a in positive if a['identity_labels']==[sp]];support.append(dict(species=sp,cohort=cohort,positive_records=len(positive),positive_groups=len({fm[a['configuration']] for a in positive}),pure_records=len(pure),individual_molecules=sum(a['observed_counts'].get(sp,0) for a in positive)))
pd.DataFrame(support).to_csv(B/'label_support.csv',index=False);print(pd.DataFrame(ci).query("metric=='micro_f1'").to_string(index=False));print(pd.DataFrame(rows).query("weighting=='near_group' and subset=='all' and stage=='fixed0p5'").groupby(['kind','protocol'])[['micro_f1','macro_f1','exact_set_accuracy']].mean().to_string())
