"""Fixed signed-shape old99/expanded118 comparison; protected geometry groups."""
from pathlib import Path
import json,hashlib,platform
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from .identity_comparison import transform,fit_model,score_model,common_threshold,SEEDS
from .grouped_identity_learning import canonical_training
from .differential_identity import shape_floor,VERSION
from .spectral_multilabel import SPECIES,encode_labels
from .coverage_study import dump,sha

def train(data_root,output_dir):
 D=Path(data_root);O=Path(output_dir);O.mkdir(parents=True,exist_ok=True)
 if any(O.iterdir()):raise ValueError('Use a new empty study output')
 M=O/'models';M.mkdir();r=json.load(open(D/'training_manifest.json'));z=np.load(D/'training_arrays.npz');e=z['energy'];y=encode_labels([a['identity_labels'] for a in r]);fold=np.array([a['fold'] for a in r]);old=np.array([a['old99'] for a in r]);groups=np.array([a['group'] for a in r]);root=Path(__file__).resolve().parents[1];mapping=json.load(open(root/'study/relative_identity_guarded_v2/near_duplicate_family_map.json'));assert mapping['training_manifest_sha256']==sha(D/'training_manifest.json');fam=np.array([mapping['families'][a['configuration']] for a in r]);query=np.flatnonzero(old);cohorts={'old99':query,'expanded118':np.array([i for i,a in enumerate(r) if a['domain']=='CuAu'])};assert len(set(fam))==124 and len(set(fam[query]))==96
 dump(O/'source_manifest.json',r);dump(O/'family_map.json',mapping);np.savez_compressed(O/'inputs.npz',**{k:z[k] for k in z.files});np.savez_compressed(O/'targets_folds.npz',truth=y,fold=fold,groups=groups,families=fam,old99=old);splits=[];thresholds=[];replay=[];learn=[];transfer={}
 def inner(tr,f):
  oldtrain=np.flatnonzero(old&(fold!=f));assignment={}
  for k,(_,v) in enumerate(GroupKFold(3).split(oldtrain,groups=groups[oldtrain])):
   for i in oldtrain[v]:assignment[int(i)]=k
  for i in tr:
   if int(i) not in assignment:assignment[int(i)]=int(hashlib.sha256(groups[i].encode()).hexdigest()[:8],16)%3
  return [(np.array([i for i in tr if assignment[int(i)]!=k]),np.array([i for i in tr if assignment[int(i)]==k])) for k in range(3)]
 def fit(tr,te,kind,seed):
  tr=canonical_training(r,fam,tr,te);floor=shape_floor(z['delta'][tr]);x=transform(z['delta'][tr],e,'signed_shape','Combined',floor);xt=transform(z['delta'][te],e,'signed_shape','Combined',floor);m=fit_model(x,y[tr],kind,seed);return m,score_model(m,xt),floor,tr
 def save(name,m,s,floor,tr,te,kind,seed,threshold):
  assert not set(fam[tr])&set(fam[te]) and len(set(fam[tr]))==len(tr)
  bundle=dict(model=m,input_contract=VERSION,detectability_policy='v3 compatible-unit SNR; arbitrary divisors do not use simulation floor',representation='signed_shape',feature='Combined',floor=floor,energy=e,species=SPECIES,domains=['CuAu'],thresholds=np.full(len(SPECIES),threshold),train_indices=tr,test_indices=te,train_groups=groups[tr].tolist(),test_groups=groups[te].tolist(),train_families=fam[tr].tolist(),test_families=fam[te].tolist(),reference_ids=sorted({r[i]['reference_id'] for i in tr}),reference_record_hashes=sorted({r[i]['reference_record_sha256'] for i in tr}),kind=kind,seed=seed,source_sha256=sha(Path(__file__)))
  p=M/(name+'.joblib');joblib.dump(bundle,p,compress=3);mm=joblib.load(p);actual=score_model(mm['model'],transform(z['delta'][te],e,'signed_shape','Combined',floor));np.testing.assert_array_equal(s,actual);replay.append({'path':str(p.relative_to(O)),'sha256':sha(p),'exact_reload':True})
 for kind in ['trees','logistic']:
  for seed in SEEDS:
   for cohort,base in cohorts.items():
    scores=np.zeros_like(y[query],float);pp={k:np.zeros_like(y[query]) for k in ['fixed0p5','transferred','common']}
    for f in range(5):
     te=query[fold[query]==f];pos=np.flatnonzero(fold[query]==f);tr=canonical_training(r,fam,base[fold[base]!=f],te);ip=np.zeros_like(y[tr],float);loc={int(i):j for j,i in enumerate(tr)};ins=inner(tr,f)
     for aa,vv in ins:
      aa=canonical_training(r,fam,aa,vv);m,ss,fl,aa=fit(aa,vv,kind,seed);ip[[loc[int(i)] for i in vv]]=ss
     th=common_threshold(y[tr],ip)
     if cohort=='old99':transfer[kind,seed,f]=th
     m,s,fl,tr=fit(tr,te,kind,seed);scores[pos]=s
     for stage,t in [('fixed0p5',.5),('transferred',transfer[kind,seed,f]),('common',th)]:pp[stage][pos]=s>=t
     name=f'size_{kind}_{cohort}_{seed}_fold{f}';save(name,m,s,fl,tr,te,kind,seed,th);np.savez_compressed(O/(name+'_inner.npz'),indices=tr,truth=y[tr],score=ip);splits.append({'name':name,'train':tr.tolist(),'test':te.tolist(),'inner':[{'train':aa.tolist(),'validation':vv.tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[aa].sum(axis=0)==0)]} for aa,vv in ins],'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]});thresholds.append({'kind':kind,'cohort':cohort,'seed':seed,'fold':f,'common':th,'transferred':transfer[kind,seed,f]})
    for stage,pred in pp.items():np.savez_compressed(O/f'size_{kind}_{cohort}_{seed}_{stage}.npz',indices=query,truth=y[query],prediction=pred,score=scores,fold=fold[query])
    print('size complete',kind,seed,cohort,flush=True)
   # Group-sampled subsets, label blind and nested within each seed/fold.
   for fraction in [.5,.75,1.]:
    score=np.zeros_like(y[query],float)
    for f in range(5):
     te=query[fold[query]==f];pos=np.flatnonzero(fold[query]==f);available=canonical_training(r,fam,cohorts['expanded118'][fold[cohorts['expanded118']]!=f],te);rng=np.random.default_rng(seed+f);tr=np.sort(rng.permutation(available)[:max(3,int(fraction*len(available)))]);m,s,fl,tr=fit(tr,te,kind,seed);score[pos]=s;save(f'curve_{kind}_{fraction}_{seed}_fold{f}',m,s,fl,tr,te,kind,seed,.5);learn.append({'kind':kind,'fraction':fraction,'seed':seed,'fold':f,'available_groups':len(available),'training_groups':len(tr),'train':tr.tolist(),'test':te.tolist(),'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]})
    np.savez_compressed(O/f'curve_{kind}_{fraction}_{seed}.npz',indices=query,truth=y[query],prediction=(score>=.5).astype(int),score=score,fold=fold[query])
 dump(O/'splits.json',splits);dump(O/'thresholds.json',thresholds);dump(O/'learning_membership.json',learn);dump(O/'model_replay.json',replay);dump(O/'protocol.json',{'version':'signed-shape-size-v3','estimator_recipes':{'trees':'independent balanced64-tree, leaf1,sqrt','logistic':'independent C1 balanced liblinear; training-only StandardScaler'},'feature':'fixed Combined signed spectrum/integral/d1/d2 after training-floor RMS normalization','seeds':SEEDS,'query':'identical old99 records/96 near groups; inherited outer and retained old inner assignments','guard':'exclude every query-related family; one canonical old-preferred train view/group; same inside inner validation','threshold':'fixed.5; old-partition common threshold transferred; enlarged training-only common threshold {.3,.4,.5,.6} with .02(t-.5)^2 penalty','learning':'label-blind canonical-group subsets before fitting; fixed.5; same old queries','uncertainty':'geometry-group bootstrap conditional on fixed predictions, separate from seed/refit variability','source_hashes':{str(p):sha(p) for p in [D/'training_arrays.npz',D/'training_manifest.json',Path(__file__),Path(__file__).with_name('differential_identity.py'),Path(__file__).with_name('identity_comparison.py')]},'python':platform.python_version(),'not_independent_confirmation':True});return {'status':'success','recipe':'signed_shape_size_v3','output_dir':str(O),'metrics':'see versioned summary/group tables','trace':str(O/'protocol.json'),'saved_models':len(replay),'prediction_replay_identical':True}
