"""Equal near-geometry-group metrics and conditional family bootstrap intervals."""
from pathlib import Path
import json
import numpy as np,pandas as pd
B=Path('study/relative_identity_guarded_v2');O=B/'results';r=json.load(open(O/'source_manifest.json'));mapping=json.load(open(B/'near_duplicate_family_map.json'))['families'];family=np.array([mapping[a['configuration']] for a in r]);metrics=[];ci=[]
def contributions(ids,y,p):
 groups=sorted(set(family[ids]));out=[];exact=[]
 for g in groups:
  mask=family[ids]==g;yy=y[mask];pp=p[mask];out.append(np.array([np.sum(yy*pp),np.sum(yy),np.sum(pp)],float)/mask.sum());exact.append(float(np.all(yy==pp,axis=1).mean()))
 return np.array(out),np.array(exact)
def micro(c):return 2*c[:,0].sum()/max(1e-20,c[:,1:].sum())
for p in O.glob('*.npz'):
 q=np.load(p)
 if not {'indices','truth','prediction'}<=set(q.files):continue
 ids=q['indices'];y=q['truth'];pred=q['prediction'];c,ex=contributions(ids,y,pred);w=np.array([1/np.sum(family[ids]==f) for f in family[ids]])[:,None];tp=np.sum(y*pred*w,axis=0);den=np.sum((y+pred)*w,axis=0);macro=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean();metrics.append({'analysis':p.stem,'n_query_records':len(ids),'n_near_geometry_groups':len(c),'group_equal_micro_f1':micro(c),'group_equal_macro_f1':macro,'group_equal_exact':ex.mean()})
pd.DataFrame(metrics).to_csv(B/'near_group_metrics.csv',index=False)
for cohort in ['expanded118','joint130']:
 for estimator in ['logistic','trees']:
  for a,b in [('native_normalized','corrected_normalized'),('corrected_normalized','common_raw'),('common_raw','relative_delta'),('relative_delta','signed_shape')]:
   aa=np.load(O/f'fixed_rep_{cohort}_{a}_{estimator}.npz');bb=np.load(O/f'fixed_rep_{cohort}_{b}_{estimator}.npz');ca,ea=contributions(aa['indices'],aa['truth'],aa['prediction']);cb,eb=contributions(bb['indices'],bb['truth'],bb['prediction']);rng=np.random.default_rng(840);v=[]
   for _ in range(2000):
    ix=rng.integers(0,len(ca),len(ca));v.append(micro(cb[ix])-micro(ca[ix]))
   ci.append(dict(cohort=cohort,estimator=estimator,baseline=a,intervention=b,baseline_micro=micro(ca),intervention_micro=micro(cb),difference=micro(cb)-micro(ca),lower=np.quantile(v,.025),upper=np.quantile(v,.975),n_near_groups=len(ca)))
pd.DataFrame(ci).to_csv(B/'near_group_representation_intervals.csv',index=False)
rows=[]
for stage in ['fixed0p5','transferred','common']:
 ca=[];cb=[]
 for seed in [20261020,20261021,20261022]:
  a=np.load(O/f'size_old99_{seed}_{stage}.npz');b=np.load(O/f'size_expanded118_{seed}_{stage}.npz');lookup={int(i):j for j,i in enumerate(b['indices'])};ix=[lookup[int(i)] for i in a['indices']];ca.append(contributions(a['indices'],a['truth'],a['prediction'])[0]);cb.append(contributions(a['indices'],b['truth'][ix],b['prediction'][ix])[0])
 differences=np.array([micro(b)-micro(a) for a,b in zip(ca,cb)]);rng=np.random.default_rng(1330);v=[]
 for _ in range(2000):
  ix=rng.integers(0,len(ca[0]),len(ca[0]));v.append(np.mean([micro(b[ix])-micro(a[ix]) for a,b in zip(ca,cb)]))
 rows.append(dict(threshold=stage,old_mean=np.mean([micro(a) for a in ca]),expanded_same_old_queries_mean=np.mean([micro(b) for b in cb]),mean_difference=differences.mean(),lower=np.quantile(v,.025),upper=np.quantile(v,.975),seed_difference_SD=differences.std(ddof=1),n_query_near_groups=len(ca[0]),uncertainty='near-geometry group bootstrap jointly across fixed seed predictions; equal group weights; refit/seed SD separate'))
pd.DataFrame(rows).to_csv(B/'near_group_training_size_intervals.csv',index=False)
# Species support is record support versus geometry-family support, not molecule count.
from tools.spectral_multilabel import SPECIES
support=[]
for domain in ['CuAu','Cu100']:
 for sp in SPECIES:
  ids=[i for i,a in enumerate(r) if a['domain']==domain and sp in a['identity_labels']];pure=[i for i in ids if r[i]['identity_labels']==[sp]];support.append({'domain':domain,'species':sp,'positive_records':len(ids),'positive_near_groups':len(set(family[ids])),'pure_records':len(pure),'pure_near_groups':len(set(family[pure]))})
pd.DataFrame(support).to_csv(B/'near_group_species_support.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
