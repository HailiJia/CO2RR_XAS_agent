from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from tools.identity_comparison import transform,fit_model,score_model,REPS,SEEDS
from tools.differential_identity import shape_floor,VERSION
from tools.spectral_multilabel import SPECIES,encode_labels,f1_metrics
from tools.coverage_study import dump,sha
B=Path('study/relative_identity_guarded_v2/results');z=np.load(B/'inputs.npz');r=json.load(open(B/'source_manifest.json'));t=np.load(B/'targets_folds.npz');fold=t['fold'];y=t['truth'];dom=t['domain'];e=z['energy'];rows=[];replay=[]
families=np.array([json.load(open(B/'family_map.json'))['families'][a['configuration']] for a in r]) if (B/'family_map.json').exists() else np.array([json.load(open(B.parent/'near_duplicate_family_map.json'))['families'][a['configuration']] for a in r])
def guard(tr,te):
 blocked=set(families[te]);seen=set();keep=[]
 for i in sorted(tr,key=lambda i:(not r[i]['old99'],r[i]['configuration'])):
  if families[i] not in blocked and families[i] not in seen:keep.append(int(i));seen.add(families[i])
 return np.array(sorted(keep),int)
for cohort,ids in [('expanded118',np.flatnonzero(dom=='CuAu')),('joint130',np.arange(len(r)))]:
 for rep in REPS:
  a=z['raw_native'] if rep=='native_normalized' else z['raw_physical'] if rep in ['corrected_normalized','common_raw'] else z['delta']
  for family in ['logistic','trees']:
   scores=np.zeros_like(y[ids],float)
   for f in range(5):
    tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);tr=guard(tr,te);floor=shape_floor(z['delta'][tr]);m=fit_model(transform(a[tr],e,rep,'Combined',floor),y[tr],family,SEEDS[0]);s=score_model(m,transform(a[te],e,rep,'Combined',floor));scores[pos]=s
    p=B/'models'/f'fixed_rep_{cohort}_{rep}_{family}_{f}.joblib';bundle=dict(input_contract=VERSION,representation=rep,feature='Combined',kind=family,model=m,floor=floor,thresholds=np.full(len(SPECIES),.5),energy=e,species=SPECIES,domains=sorted(set(dom[tr])),train_indices=tr,test_indices=te,train_groups=t['groups'][tr].tolist(),test_groups=t['groups'][te].tolist(),train_families=families[tr].tolist(),test_families=families[te].tolist(),reference_ids=sorted({r[i]['reference_id'] for i in tr}),normalization='fixed Combined representation control',seed=SEEDS[0]);joblib.dump(bundle,p,compress=3);assert np.array_equal(score_model(joblib.load(p)['model'],transform(a[te],e,rep,'Combined',floor)),s);replay.append({'path':str(p.relative_to(B)),'sha256':sha(p),'exact_reload':True})
   pred=(scores>=.5).astype(int);np.savez_compressed(B/f'fixed_rep_{cohort}_{rep}_{family}.npz',indices=ids,truth=y[ids],score=scores,prediction=pred,fold=fold[ids]);rows.append(dict(cohort=cohort,representation=rep,estimator=family,feature='Combined',threshold=.5,**f1_metrics(y[ids],pred)));print(cohort,rep,family,flush=True)
pd.DataFrame(rows).to_csv(B/'fixed_representation_metrics.csv',index=False);dump(B/'fixed_representation_models.json',replay)
