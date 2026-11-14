"""Replay all saved refresh models and physical/source invariants, without refitting."""
from pathlib import Path
import sys,json,re
import numpy as np,pandas as pd,joblib
from scipy.fft import dct
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spectral_multilabel import feature_bank,binary_scores
from tools.differential_xas import features
from tools.differential_training import cls_score
from tools.coverage_study import sha
B=ROOT/'study/dataset_20261029';O=B/'ml';rows=json.load(open(B/'training_manifest.json'));z=np.load(B/'training_arrays.npz');norm=feature_bank(z['raw_native'],z['energy']);banks={'normalized_raw':{'Combined':norm['shape_CDF_derivatives'],'DCT16':dct(norm['normalized'],type=2,norm='ortho',axis=1)[:,:16]},'physical_raw':features(z['raw_physical'],z['energy']),'delta':features(z['delta'],z['energy'])};receipt=[]
for p in sorted((O/'models').glob('*.joblib')):
 m=joblib.load(p);ids=m['test_indices'];assert not set(m['train_groups'])&set(m['test_groups']);assert all(rows[i]['group']in m['test_groups'] for i in ids)
 if m['kind']=='fixed_Combined_independent_trees':
  c,seed,fold=re.fullmatch(r'fixed_(.+)_(\d+)_fold(\d+)',p.stem).groups();expected=np.load(O/f'identity_{c}_normalized_raw_inner_retuned_threshold_{seed}.npz');score=binary_scores(m['model'],banks['normalized_raw']['Combined'][ids]);pred=(score>=m['thresholds']).astype(int)
 elif m['kind']=='compact_nested_identity':
  c,rep,fold=re.fullmatch(r'nested_(refresh_CuAu|refresh_joint)_(normalized_raw|physical_raw|delta)_fold(\d+)',p.stem).groups();expected=np.load(O/f'identity_{c}_{rep}_compact_nested_20261027.npz');score=cls_score(m['model'],banks[rep][m['recipe'][0]][ids]);pred=(score>=m['thresholds']).astype(int)
  for h in m['reference_hashes']:assert h in {r['reference_record_sha256'] for r in rows}
 else:
  c,rep,fold=re.fullmatch(r'coverage_(baseline99_CuAu|refresh_CuAu|refresh_joint)_(normalized_raw|physical_raw|delta)_fold(\d+)',p.stem).groups();expected=np.load(O/f'CO_coverage_{c}_{rep}_compact_nested_20261027.npz');pred=m['model'].predict(banks[rep][m['recipe'][0]][ids]);score=None
 lookup={int(v):j for j,v in enumerate(expected['pair_indices'])};ii=[lookup[int(v)] for v in ids];np.testing.assert_array_equal(pred,expected['prediction'][ii])
 if score is not None:np.testing.assert_array_equal(score,expected['score'][ii])
 receipt.append({'model':p.name,'sha256':sha(p),'heldout_reload_match':True,'training_test_groups_disjoint':True})
assert len(receipt)==110
for r in rows:
 assert sha(B/'scale'/r['corrected_record'])==r['sample_record_sha256']
for k,v in json.load(open(O/'protocol.json'))['source_hashes'].items():assert sha(ROOT/'tools'/k)==v,(k,'code changed after fit')
np.testing.assert_allclose(z['raw_physical']-z['clean'],z['delta'],atol=1e-14);scale=json.load(open(B/'scale/source_weight_audit.json'));assert sum(r['native_reproduced'] is True for r in scale)==272 and sum(r['native_reproduced'] is None for r in scale)==2
print(json.dumps({'models':receipt,'saved_models_replayed':len(receipt),'native_layer_reconstructions':272,'new_clean_layers_from_actual_saved_raw':2,'signed_difference_and_source_hash_checks':True},indent=2))
