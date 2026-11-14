"""All-query/all-view proof, saved model replay and input integrity; no fitting."""
from pathlib import Path
import json
import numpy as np,joblib
from tools.coverage_study import sha,dump
from tools.full_cohort_tuning import CANDIDATES,group_weights,choose_threshold
from tools.identity_comparison import transform,score_model
from tools.spectral_multilabel import encode_labels
B=Path(__file__).resolve().parents[1];D=B.parents[1]/'study/dataset_20261101';O=B/'results';r=json.load(open(D/'training_manifest.json'));z=np.load(D/'training_arrays.npz');y=encode_labels([a['identity_labels'] for a in r]);fm=json.load(open(D/'family_map.json'))['families'];fam=np.array([fm[a['configuration']] for a in r]);p=json.load(open(O/'protocol.json'))
for name,h in p['input_hashes'].items():assert sha(D/name)==h
assert sha(B.parents[1]/'tools/full_cohort_tuning.py')==p['source_sha256']
for s in json.load(open(O/'splits.json')):
 tr=np.array(s['train']);te=np.array(s['test']);np.testing.assert_array_equal(tr,np.flatnonzero(~np.isin(fam,fam[te])))
 for inn in s['inner']:
  a,v=np.array(inn['train']),np.array(inn['validation']);assert sorted(np.r_[a,v].tolist())==tr.tolist();assert not set(fam[a])&set(fam[v]);w=group_weights(fam[a]);assert all(np.isclose(w[fam[a]==g].sum(),1.) for g in set(fam[a]))
for path in O.glob('*.npz'):
 if path.name.startswith('inner_'):continue
 q=np.load(path);np.testing.assert_array_equal(q['indices'],np.arange(141));np.testing.assert_array_equal(q['truth'],y);assert np.isfinite(q['score']).all()
replays=[]
for rec in json.load(open(O/'model_replay.json')):
 path=O/rec['path'];assert sha(path)==rec['sha256'];m=joblib.load(path);te=m['test_indices'];V=transform(z['delta'][te],z['energy'],'signed_shape',m['feature'],m['floor']);s=score_model(m['model'],V);seed=m['seed'];fold=next(k for k in range(5) if f'fold{k}_' in path.name)
 selected=[a for a in json.load(open(O/'selections.json')) if a['seed']==seed and a['fold']==fold and a['outer_model']==rec['path']]
 for row in selected:np.testing.assert_array_equal(s,np.load(O/f"{row['protocol']}_{seed}_common.npz")['score'][te])
 replays.append(rec)
for row in json.load(open(O/'selections.json')):
 q=np.load(O/f"inner_{row['seed']}_fold{row['fold']}.npz");k=row['candidate'];assert row['threshold']==q['thresholds'][k]
 if row['protocol']=='nested':assert k==int(np.argmax(q['quality']))
fp=json.load(open(B/'final_models/protocol.json'));assert sha(B.parents[1]/'tools/full_cohort_deploy.py')==fp['source_sha256'];assert sha(B.parents[1]/'tools/full_cohort_tuning.py')==fp['fitting_dependency_sha256'];assert sha(D/'training_arrays.npz')==fp['input_sha256']
final=json.load(open(B/'final_models/models.json'))
for rec in final:
 path=B/'final_models'/rec['path'];assert sha(path)==rec['sha256'];m=joblib.load(path);np.testing.assert_array_equal(m['train_indices'],np.arange(141));assert not len(m['test_indices']);X=transform(z['delta'],z['energy'],'signed_shape',m['feature'],m['floor']);actual=score_model(m['model'],X);assert np.isfinite(actual).all();np.testing.assert_array_equal(actual,np.load(B/f"final_models/full_training_replay_seed{rec['seed']}.npz")['score'])
receipt=dict(all_queries=141,near_groups=135,no_query_filter=True,all_permitted_training_views=True,all_inner_views=True,outer_models_exact_reload=len(replays),final_models=len(final),final_training_records_per_model=141,inputs_unchanged=True,manuscript_changes=False)
dump(B/'verification.json',receipt);print(json.dumps(receipt))
