"""No-refit clean-checkout verification of all models, arrays and input contracts."""
from pathlib import Path
import ast,json,hashlib
import numpy as np,joblib
from scipy.ndimage import gaussian_filter1d
from tools.identity_comparison import score_model,transform
from tools.coverage_study import sha
B=Path('study/relative_identity_v2');O=B/'results';z=np.load(O/'inputs.npz');t=np.load(O/'targets_folds.npz');e=z['energy'];mask=(e>=e[0]+1)&(e<=e[-1]-1);ec=e[mask];count=0
for row in json.load(open(B/'all_model_replay.json')):
 p=O/row['model'];assert sha(p)==row['sha256'];m=joblib.load(p);ids=m['test_indices'];rep=m['representation'];x=z['raw_native'] if rep=='native_normalized' else z['raw_physical'] if rep in ['corrected_normalized','common_raw'] else z['delta']
 if p.stem.startswith('augment_') and rep in ['relative_delta','signed_shape']:
  x=(z['raw_physical']-z['clean']).copy();x[:,~mask]=z['delta'][:,~mask]
 s=score_model(m['model'],transform(x[ids],e,rep,m['feature'],m['floor']));q=np.load(O/row['expected']);lookup={int(v):j for j,v in enumerate(q['indices'])};expected=q['score'][[lookup[int(v)] for v in ids]];np.testing.assert_array_equal(s,expected);assert not set(m['train_groups'])&set(m['test_groups']);count+=1
# Extract only the pure pair perturbation function, never execute a fitting script.
script=B/'code/paired_robustness.py';tree=ast.parse(script.read_text());fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='perturb');exec(compile(ast.Module(body=[fun],type_ignores=[]),str(script),'exec'))
P=B/'paired_robustness';nscenario=0
for row in json.load(open(P/'model_replay.json')):
 p=P/row['model'];assert sha(p)==row['sha256'];m=joblib.load(p);ids=m['test_indices'];rep=m['representation'];fold=int(p.stem.rsplit('fold',1)[1]);assert np.array_equal(m['energy'],ec)
 for scenario in json.load(open(P/'protocol.json'))['scenarios']:
  aa,bb,scale=perturb(z['raw_physical'][ids],z['clean'][ids],scenario,np.random.default_rng(624+fold));x=aa/scale if rep=='corrected_normalized' else (aa-bb)/scale;s=score_model(m['model'],transform(x,ec,rep,'DCT16',m['floor']));q=np.load(P/f"{rep}_aug{m['augmentation']}_{scenario}.npz");lookup={int(v):j for j,v in enumerate(q['indices'])};np.testing.assert_array_equal(s,q['score'][[lookup[int(v)] for v in ids]]);nscenario+=1
 assert not set(m['train_groups'])&set(m['test_groups']);count+=1
 # Common positive acquisition gain cannot change the normalized input observable.
 aa,bb,sc=perturb(z['raw_physical'][ids],z['clean'][ids],'common_acquisition_gain5pct',np.random.default_rng(0));np.testing.assert_allclose((aa-bb)/sc,(z['raw_physical'][ids]-z['clean'][ids])[:,mask],atol=2e-15)
for name,h in json.load(open(O/'protocol.json'))['source_hashes'].items():
 p=Path('tools')/Path(name).name if str(name).endswith('.py') else Path('study/dataset_20261029')/Path(name).name
 if sha(p)!=h and str(name).endswith('.py'):p=B/'code/training_source_snapshots'/Path(name).name
 assert sha(p)==h,(p,'archived as-run source hash mismatch')
np.testing.assert_allclose(z['raw_physical']-z['clean'],z['delta'],atol=1e-14);assert len(set(t['groups']))==130
print(json.dumps({'saved_models':count,'original_heldout_models':311,'paired_robustness_models':30,'paired_robustness_fold_scenarios':nscenario,'exact_scores':True,'group_disjoint':True,'source_hashes_match':True,'not_refitting':True}))
