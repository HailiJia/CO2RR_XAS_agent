"""One bounded dependence control motivated by missing mixture constituents.
Declared before these fits: fixed existing joint signed-shape Combined/64-tree
outer models, seed20261020, threshold.5, pair prior50, clip+-2, lambda{0,.25}.
Select lambda only by grouped inner micro F1; ties choose independence. No order,
new spectra, estimator grid or outer-score winner. Prior studies stay read-only.
"""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from tools.identity_comparison import fit_model,score_model,transform
from tools.differential_identity import shape_floor
from tools.grouped_identity_learning import canonical_training
from tools.spectral_multilabel import SPECIES,f1_metrics
from tools.spectral_label_decoder import pair_lift,decode
from tools.coverage_study import dump,sha
B=Path('study/signed_shape_detectability_v3');I=Path('study/relative_identity_guarded_v2/results');O=B/'label_dependence';O.mkdir(exist_ok=True);r=json.load(open(I/'source_manifest.json'));z=np.load(I/'inputs.npz');t=np.load(I/'targets_folds.npz');mapping=json.load(open(I/'family_map.json'))['families'];fam=np.array([mapping[a['configuration']] for a in r]);y=t['truth'];fold=t['fold'];g=t['groups'];q=np.load(I/'fixed_rep_joint130_signed_shape_trees.npz');selected=np.zeros_like(y);fixed=np.zeros_like(y);log=[]
for f in range(5):
 model_path=I/'models'/f'fixed_rep_joint130_signed_shape_trees_{f}.joblib';m=joblib.load(model_path);tr=m['train_indices'];te=m['test_indices'];assert not set(fam[tr])&set(fam[te]);oldtrain=np.flatnonzero(t['old99']&(fold!=f));assignment={}
 for k,(_,v) in enumerate(GroupKFold(3).split(oldtrain,groups=g[oldtrain])):
  for i in oldtrain[v]:assignment[int(i)]=k
 import hashlib
 for i in tr:
  if int(i) not in assignment:assignment[int(i)]=int(hashlib.sha256(g[i].encode()).hexdigest()[:8],16)%3
 pred={a:np.zeros_like(y[tr]) for a in [0.,.25]};inner=[]
 for k in range(3):
  a=np.array([i for i in tr if assignment[int(i)]!=k]);v=np.array([i for i in tr if assignment[int(i)]==k]);a=canonical_training(r,fam,a,v);floor=shape_floor(z['delta'][a]);est=fit_model(transform(z['delta'][a],z['energy'],'signed_shape','Combined',floor),y[a],'trees',20261020);s=score_model(est,transform(z['delta'][v],z['energy'],'signed_shape','Combined',floor));J=pair_lift(y[a]);loc=[list(tr).index(i) for i in v]
  for strength in pred:pred[strength][loc]=decode(s,J,strength)
  inner.append({'train':a.tolist(),'validation':v.tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[a].sum(axis=0)==0)]})
 quality={a:f1_metrics(y[tr],p)['micro_f1'] for a,p in pred.items()};win=.25 if quality[.25]>quality[0.] else 0.;J=pair_lift(y[tr]);selected[te]=decode(q['score'][te],J,win);fixed[te]=decode(q['score'][te],J,.25);p=O/f'decoder_fold{f}.npz';np.savez_compressed(p,interaction=J,strength=win,train_indices=tr,test_indices=te);reload=np.load(p);np.testing.assert_array_equal(selected[te],decode(q['score'][te],reload['interaction'],float(reload['strength'])));log.append({'fold':f,'selected_strength':win,'inner_micro':quality,'inner_splits':inner,'outer_model_sha256':sha(model_path),'decoder_sha256':sha(p),'exact_reload':True})
np.savez_compressed(O/'predictions.npz',indices=q['indices'],truth=y,independent=q['prediction'],inner_selected=selected,fixed_strength_sensitivity=fixed,score=q['score'],fold=fold);dump(O/'selection.json',log);dump(O/'protocol.json',{'settings':'pair-log-lift from train labels only; independent smoothed prior strength50, clip+-2; lambda0/.25 selected by inner micro F1; fixed.25 sensitivity never used to select a primary method','seed':20261020,'not_independent_confirmation':True,'source_sha256':sha(Path(__file__)),'decoder_source_sha256':sha(Path('tools/spectral_label_decoder.py'))});print('Selected strengths',[a['selected_strength'] for a in log])
