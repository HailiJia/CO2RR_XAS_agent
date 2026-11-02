"""Fixed signed-shape update with protected groups and identical previous queries."""
from pathlib import Path
import hashlib,json
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from .identity_comparison import transform,fit_model,score_model,common_threshold,SEEDS
from .grouped_identity_learning import canonical_training
from .differential_identity import shape_floor,VERSION
from .spectral_multilabel import SPECIES,encode_labels
from .coverage_study import dump,sha

def train(data_root,output_dir):
 B=Path(data_root);O=Path(output_dir);O.mkdir(parents=True,exist_ok=True)
 if any(O.iterdir()):raise ValueError('Use a new empty output directory')
 r=json.load(open(B/'training_manifest.json'));z=np.load(B/'training_arrays.npz');fm=json.load(open(B/'family_map.json'));assert sha(B/'training_manifest.json')==fm['training_manifest_sha256'];fam=np.array([fm['families'][a['configuration']] for a in r]);y=encode_labels([a['identity_labels'] for a in r]);f=np.array([a['fold'] for a in r]);e=z['energy'];old=np.array([a['old_index'] is not None for a in r]);allids=np.arange(len(r));oldids=np.flatnonzero(old);old99=np.array([a['old99'] for a in r]);(O/'models').mkdir();splits=[];thresholds=[];reload=[];transfer={}
 def inner(tr,fold):
  oo=np.flatnonzero(old99&(f!=fold));assignment={}
  for k,(_,v) in enumerate(GroupKFold(3).split(oo,groups=np.array([a['group'] for a in r])[oo])):
   for i in oo[v]:assignment[int(i)]=k
  for i in tr:
   if int(i) not in assignment:assignment[int(i)]=int(hashlib.sha256(r[i]['group'].encode()).hexdigest()[:8],16)%3
  return [(canonical_training(r,fam,np.array([i for i in tr if assignment[int(i)]!=k]),np.array([i for i in tr if assignment[int(i)]==k])),np.array([i for i in tr if assignment[int(i)]==k])) for k in range(3)]
 for kind in ['trees','logistic']:
  for seed in SEEDS:
   for protocol,base,query in [('previous_training_same_queries',oldids,oldids),('updated_training_same_queries',allids,oldids),('updated_full_cohort',allids,allids)]:
    score=np.zeros_like(y[query],float);pred={s:np.zeros_like(y[query]) for s in ['fixed0p5','transferred','common']}
    for fold in range(5):
     te=query[f[query]==fold];pos=np.flatnonzero(f[query]==fold);tr=canonical_training(r,fam,base[f[base]!=fold],te);ins=inner(tr,fold);ip=np.zeros_like(y[tr],float);loc={int(i):j for j,i in enumerate(tr)}
     for a,v in ins:
      floor=shape_floor(z['delta'][a]);X=transform(z['delta'][a],e,'signed_shape','Combined',floor);V=transform(z['delta'][v],e,'signed_shape','Combined',floor);ip[[loc[int(i)] for i in v]]=score_model(fit_model(X,y[a],kind,seed),V)
     th=common_threshold(y[tr],ip)
     if protocol=='previous_training_same_queries':transfer[kind,seed,fold]=th
     floor=shape_floor(z['delta'][tr]);X=transform(z['delta'][tr],e,'signed_shape','Combined',floor);V=transform(z['delta'][te],e,'signed_shape','Combined',floor);model=fit_model(X,y[tr],kind,seed);s=score_model(model,V);score[pos]=s
     for stage,t in [('fixed0p5',.5),('transferred',transfer[kind,seed,fold]),('common',th)]:pred[stage][pos]=s>=t
     bundle=dict(model=model,input_contract=VERSION,representation='signed_shape',feature='Combined',floor=floor,energy=e,species=SPECIES,domains=sorted({a['domain'] for a in r}),thresholds=np.full(len(SPECIES),th),train_indices=tr,test_indices=te,train_families=fam[tr].tolist(),test_families=fam[te].tolist(),reference_ids=sorted({r[i]['reference_id'] for i in tr}),reference_hashes=sorted({r[i]['reference_record_sha256'] for i in tr}),source_sha256=sha(Path(__file__)),seed=seed);path=O/'models'/f'{kind}_{seed}_{protocol}_fold{fold}.joblib';joblib.dump(bundle,path,compress=3);np.testing.assert_array_equal(s,score_model(joblib.load(path)['model'],V));reload.append({'path':str(path.relative_to(O)),'sha256':sha(path),'exact_reload':True});splits.append({'protocol':protocol,'kind':kind,'seed':seed,'fold':fold,'train':tr.tolist(),'test':te.tolist(),'inner':[{'train':a.tolist(),'validation':v.tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[a].sum(axis=0)==0)]} for a,v in ins],'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]});thresholds.append(dict(protocol=protocol,kind=kind,seed=seed,fold=fold,common=th,transferred=transfer[kind,seed,fold]))
    for stage,p in pred.items():np.savez_compressed(O/f'{kind}_{seed}_{protocol}_{stage}.npz',indices=query,truth=y[query],prediction=p,score=score,fold=f[query])
    print('FIT',kind,seed,protocol,len(query),flush=True)
 dump(O/'splits.json',splits);dump(O/'thresholds.json',thresholds);dump(O/'reload.json',reload);dump(O/'protocol.json',{'primary':'signed-shape Combined, fixed population-weighted top-two, independent balanced64-tree leaf1; logistic C1 balanced baseline','seeds':SEEDS,'thresholds':'fixed.5; previous-training common transferred; common {.3,.4,.5,.6}, penalty .02(t-.5)^2 selected only inner-training','queries':'previous available configurations with unchanged inputs; updated full cohort is separate, not a paired improvement','groups':'one canonical train record per near group; outer and inner query-related groups excluded; prior clusters/folds inherited','features':'spectra only, no names/counts/loading/site metadata','source_sha256':sha(Path(__file__)),'training_arrays_sha256':sha(B/'training_arrays.npz'),'no_depth_selector':True});return {'status':'success','models':len(reload),'output':str(O)}
