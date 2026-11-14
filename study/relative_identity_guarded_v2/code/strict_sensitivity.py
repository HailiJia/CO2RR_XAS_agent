"""Geometry-only grouping cutoff sensitivity; fixed delta Combined/trees/.5."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from tools.identity_comparison import transform,fit_model,score_model,SEEDS
from tools.differential_identity import shape_floor,VERSION
from tools.spectral_multilabel import SPECIES,f1_metrics
from tools.coverage_study import dump,sha
B=Path('study/relative_identity_guarded_v2');I=B/'results';O=B/'strict_sensitivity';(O/'models').mkdir(parents=True,exist_ok=True);r=json.load(open(I/'source_manifest.json'));z=np.load(I/'inputs.npz');t=np.load(I/'targets_folds.npz');y=t['truth'];e=z['energy'];parent={a['configuration']:a['configuration'] for a in r}
def root(a):
 while parent[a]!=a:a=parent[a]
 return a
for _,p in pd.read_csv(B/'near_geometry_pairs.csv').iterrows():
 if p.RMS_A<=.02 and p.max_A<=.05:
  a,b=root(p.configuration_a),root(p.configuration_b);parent[max(a,b)]=min(a,b)
fam=np.array([root(a['configuration']) for a in r]);dump(O/'family_map.json',dict(zip([a['configuration'] for a in r],fam)));rows=[];replay=[];preds={}
for cohort,ids in [('old99',np.flatnonzero(t['old99'])),('expanded118',np.flatnonzero(t['domain']=='CuAu'))]:
 for seed in SEEDS:
  scores=np.zeros_like(y[ids],float)
  for f in range(5):
   tr=ids[t['fold'][ids]!=f];te=ids[t['fold'][ids]==f];pos=np.flatnonzero(t['fold'][ids]==f);blocked=set(fam[te]);keep=[];seen=set()
   for i in sorted(tr,key=lambda i:(not r[i]['old99'],r[i]['configuration'])):
    if fam[i] not in blocked and fam[i] not in seen:keep.append(int(i));seen.add(fam[i])
   tr=np.array(sorted(keep));floor=shape_floor(z['delta'][tr]);X=transform(z['delta'],e,'relative_delta','Combined',floor);m=fit_model(X[tr],y[tr],'trees',seed);s=score_model(m,X[te]);scores[pos]=s;p=O/'models'/f'{cohort}_{seed}_{f}.joblib';bundle=dict(model=m,representation='relative_delta',feature='Combined',floor=floor,energy=e,train_indices=tr,test_indices=te,train_families=fam[tr].tolist(),test_families=fam[te].tolist(),thresholds=np.full(len(SPECIES),.5),input_contract=VERSION,species=SPECIES,domains=['CuAu'],source_hash=sha(Path(__file__)));joblib.dump(bundle,p,compress=3);assert np.array_equal(s,score_model(joblib.load(p)['model'],X[te]));replay.append({'model':str(p.relative_to(O)),'sha256':sha(p),'exact_reload':True})
  pp=(scores>=.5).astype(int);np.savez_compressed(O/f'{cohort}_{seed}.npz',indices=ids,truth=y[ids],prediction=pp,score=scores,fold=t['fold'][ids]);preds[cohort,seed]=(ids,pp);rows.append(dict(cohort=cohort,seed=seed,**f1_metrics(y[ids],pp)))
pd.DataFrame(rows).to_csv(O/'record_metrics.csv',index=False);dump(O/'model_replay.json',replay)
def contributions(ids,p):
 out=[]
 for family in sorted(set(fam[ids])):
  m=fam[ids]==family;yy=y[ids][m];pp=p[m];out.append(np.array([np.sum(yy*pp),np.sum(yy),np.sum(pp)],float)/m.sum())
 return np.array(out)
def micro(c):return 2*c[:,0].sum()/max(1e-20,c[:,1:].sum())
a=[];b=[]
for seed in SEEDS:
 ids,p=preds['old99',seed];bi,bp=preds['expanded118',seed];lookup={int(i):j for j,i in enumerate(bi)};bp=bp[[lookup[int(i)] for i in ids]];a.append(contributions(ids,p));b.append(contributions(ids,bp))
rng=np.random.default_rng(1330);bs=[]
for _ in range(2000):
 ix=rng.integers(0,len(a[0]),len(a[0]));bs.append(np.mean([micro(bb[ix])-micro(aa[ix]) for aa,bb in zip(a,b)]))
out={'criterion':'RMS<=.02 A and max<=.05 A, three near pairs','old_query_groups':len(a[0]),'old_mean':np.mean([micro(c) for c in a]),'expanded_mean_same_old':np.mean([micro(c) for c in b]),'conditional_lower':np.quantile(bs,.025),'conditional_upper':np.quantile(bs,.975),'scope':'cutoff sensitivity, not selected by test scores'};dump(O/'size_summary.json',out);print(out)
