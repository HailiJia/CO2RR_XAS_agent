"""Learning subsets sampled from protected geometry groups, never from record views."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from .identity_comparison import transform,fit_model,score_model,SEEDS
from .differential_identity import shape_floor,VERSION
from .spectral_multilabel import SPECIES,f1_metrics
from .coverage_study import dump,sha

def canonical_training(rows,families,train,test):
 blocked=set(families[test]);seen=set();keep=[]
 for i in sorted(train,key=lambda i:(not rows[i]['old99'],rows[i]['configuration'])):
  if families[i] not in blocked and families[i] not in seen:keep.append(int(i));seen.add(families[i])
 return np.array(sorted(keep),int)

def run(output_dir,target=None):
 I=Path(output_dir);O=Path(target) if target is not None else I/'group_learning';(O/'models').mkdir(parents=True,exist_ok=True);r=json.load(open(I/'source_manifest.json'));z=np.load(I/'inputs.npz');t=np.load(I/'targets_folds.npz');e=z['energy'];y=t['truth'];mapping=json.load(open(I/'family_map.json'))['families'];fam=np.array([mapping[a['configuration']] for a in r]);base=np.flatnonzero(t['domain']=='CuAu');query=np.flatnonzero(t['old99']);membership=[];metrics=[];replay=[]
 for seed in SEEDS:
  for fraction in [.5,.75,1.]:
   score=np.zeros_like(y[query],float)
   for f in range(5):
    te=query[t['fold'][query]==f];pos=np.flatnonzero(t['fold'][query]==f);available=canonical_training(r,fam,base[t['fold'][base]!=f],te);rng=np.random.default_rng(seed+f);tr=np.sort(rng.permutation(available)[:max(3,int(len(available)*fraction))]);floor=shape_floor(z['delta'][tr]);X=transform(z['delta'],e,'relative_delta','Combined',floor);m=fit_model(X[tr],y[tr],'trees',seed);s=score_model(m,X[te]);score[pos]=s;p=O/'models'/f'fraction{fraction}_{seed}_{f}.joblib';bundle=dict(model=m,input_contract=VERSION,representation='relative_delta',feature='Combined',floor=floor,energy=e,train_indices=tr,test_indices=te,train_groups=t['groups'][tr].tolist(),test_groups=t['groups'][te].tolist(),train_families=fam[tr].tolist(),test_families=fam[te].tolist(),thresholds=np.full(len(SPECIES),.5),species=SPECIES,domains=['CuAu'],reference_ids=sorted({r[i]['reference_id'] for i in tr}),reference_record_hashes=sorted({r[i]['reference_record_sha256'] for i in tr}),source_hash=sha(Path(__file__)),seed=seed,fraction=fraction);joblib.dump(bundle,p,compress=3);assert np.array_equal(s,score_model(joblib.load(p)['model'],X[te]));replay.append({'model':str(p.relative_to(O)),'sha256':sha(p),'exact_reload':True});membership.append(dict(seed=seed,fraction=fraction,fold=f,available_groups=len(available),sampled_groups=len(tr),train=tr.tolist(),test=te.tolist(),missing_positive=[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]))
   pred=(score>=.5).astype(int);np.savez_compressed(O/f'fraction{fraction}_{seed}.npz',indices=query,truth=y[query],prediction=pred,score=score,fold=t['fold'][query]);metrics.append(dict(seed=seed,fraction=fraction,**f1_metrics(y[query],pred)))
 pd.DataFrame(metrics).to_csv(O/'record_metrics.csv',index=False);dump(O/'membership.json',membership);dump(O/'model_replay.json',replay);dump(O/'protocol.json',{'subset_unit':'canonical near-geometry group, after outer query exclusion; one view/group; label blind nested seeded subsets','threshold':.5,'feature':'Combined','estimator':'balanced64-tree leaf1','seeds':SEEDS,'source_sha256':sha(Path(__file__))});return {'group_learning_models':len(replay),'output':str(O)}
