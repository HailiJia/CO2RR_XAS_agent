"""Verify saved numerical replay, effective grouped splits and as-run sources; no fit."""
from pathlib import Path
import ast,hashlib,json
import numpy as np,joblib
from tools.identity_comparison import transform,score_model
from tools.grouped_identity_learning import canonical_training
from tools.spectral_multilabel import SPECIES
B=Path('study/signed_shape_detectability_v3');O=B/'ml';r=json.load(open(O/'source_manifest.json'));z=np.load(O/'inputs.npz');t=np.load(O/'targets_folds.npz');fam=t['families'];actual=[]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for row in json.load(open(O/'model_replay.json')):
 p=O/row['path'];assert sha(p)==row['sha256'];m=joblib.load(p);tr=m['train_indices'];te=m['test_indices'];assert not set(fam[tr])&set(fam[te]);assert len(set(fam[tr]))==len(tr);s=score_model(m['model'],transform(z['delta'][te],z['energy'],'signed_shape','Combined',m['floor']))
 name=p.stem;parts=name.split('_');q=np.load(O/('_'.join(parts[:-1])+('_common' if parts[0]=='size' else '')+'.npz'));positions=[list(q['indices']).index(i) for i in te];np.testing.assert_array_equal(s,q['score'][positions]);actual.append({'model':row['path'],'sha256':sha(p),'exact_reload':True,'training_groups':len(tr),'test_groups':len(set(fam[te]))})
effective=[]
for row in json.load(open(O/'splits.json')):
 for k,inn in enumerate(row['inner']):
  a=canonical_training(r,fam,np.array(inn['train']),np.array(inn['validation']));v=np.array(inn['validation']);assert not set(fam[a])&set(fam[v]);assert len(set(fam[a]))==len(a);effective.append({'outer':row['name'],'inner':k,'train':a.tolist(),'validation':v.tolist(),'training_groups':len(a),'missing_positive':[SPECIES[j] for j in np.flatnonzero(t['truth'][a].sum(axis=0)==0)]})
curve=json.load(open(O/'learning_membership.json'))
for kind in ['trees','logistic']:
 for seed in [20261020,20261021,20261022]:
  for f in range(5):
   q=sorted([a for a in curve if a['kind']==kind and a['seed']==seed and a['fold']==f],key=lambda a:a['fraction']);assert set(q[0]['train'])<=set(q[1]['train'])<=set(q[2]['train'])
# Source dependency was recorded at fit time; subsequent fixes concern inference.
snap=B/'code/training_source_snapshots/differential_identity.py';expected=[v for k,v in json.load(open(O/'protocol.json'))['source_hashes'].items() if k.endswith('/differential_identity.py')][0];assert sha(snap)==expected
for name in ['shape_floor','signed_shape']:
 def node(p):return ast.dump(next(n for n in ast.parse(Path(p).read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
 assert node(snap)==node('tools/differential_identity.py')
# No historical study or predictions were rewritten.
import subprocess
changed=subprocess.check_output(['git','diff','--name-only','638e3cb87cd1aac40157db38d52a5fac00aec56d','--','study']).decode().splitlines();assert all(a.startswith(('study/signed_shape_detectability_v3/','study/manuscript_reviewed_scoped_fixes_v1/','study/Reviewed_Manuscript_Scoped_Fixes_v1.zip')) for a in changed),changed
(B/'verification/effective_inner_splits.json').write_text(json.dumps(effective,indent=2)+'\n');(B/'verification/model_replay_verified.json').write_text(json.dumps({'models':actual,'total_exact':len(actual),'training_transforms_AST_unchanged':True,'as_run_dependency_sha256':expected,'nested_learning_subsets':True},indent=2)+'\n');print('Verified',len(actual),'saved models,',len(effective),'effective inner splits; nested curves and unchanged training transforms')
