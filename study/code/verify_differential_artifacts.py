"""Replay saved held-out models without refitting; source archives checked separately."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.fft import dct
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.differential_training import banks,predict_bundle,cls_score
from tools.differential_robustness import perturb,feature_banks,SEED
from tools.spectral_multilabel import feature_bank
B=ROOT/'study/differential_v1';z=np.load(B/'paired_arrays.npz');bf=banks(z);receipts=[]
def receipt(p):receipts.append({'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'heldout_reload_match':True})
for p in sorted((B/'ml/models').glob('*.joblib')):
 m=joblib.load(p);ids=np.array(m['test_pair_indices']);out=predict_bundle(m,{r:{f:a[ids] for f,a in bank.items()} for r,bank in bf.items()});pred=out[0] if m['task']=='identity' else out;q=np.load(B/'ml'/f"{m['scope']}_{m['task']}_{m['representation']}_oof.npz");look={int(v):j for j,v in enumerate(q['pair_indices'])};np.testing.assert_array_equal(pred,q['prediction'][[look[int(i)] for i in ids]]);receipt(p)
for p in sorted((B/'geometry/models').glob('*.joblib')):
 m=joblib.load(p);ids=np.array(m['test_pair_indices']);q=np.load(B/'geometry'/f"{m['scope']}_{m['representation']}_oof.npz");look={int(v):j for j,v in enumerate(q['pair_indices'])};np.testing.assert_array_equal(m['model'].predict(bf[m['representation']][m['recipe'][0]][ids]),q['prediction'][[look[int(i)] for i in ids]]);receipt(p)
protocol=json.load(open(B/'robustness/predeclared_protocol.json'));scenarios=protocol['scenarios'];stress_checks=0
for p in sorted((B/'robustness/models').glob('*.joblib')):
 m=joblib.load(p);ids=m['test_pair_indices']
 for si,sc in enumerate(scenarios):
  for rep in range(3 if 'noise' in sc['kind'] else 1):
   shared=np.random.default_rng(SEED+si*100+rep).normal(size=z['energy'].shape);a,b,e=perturb(z['raw_physical'],z['clean'],z['energy'],sc,SEED+si*100+rep,shared);X=feature_banks(a,b,e)[m['representation']]['DCT16'][ids];pred=m['model'].predict(X) if m['task']=='CO_coverage' else (cls_score(m['model'],X)>=m['thresholds']).astype(int);q=np.load(B/'robustness'/f"{m['task']}_{m['training_protocol']}_{m['representation']}_{sc['name']}_rep{rep}.npz");look={int(v):j for j,v in enumerate(q['pair_indices'])};np.testing.assert_array_equal(pred,q['prediction'][[look[int(i)] for i in ids]]);stress_checks+=1
 receipt(p)
old=ROOT/'study/coverage_20261026';rr=json.load(open(old/'model_configuration_manifest.json'));zz=np.load(old/'model_inputs.npz');X=dct(feature_bank(zz['raw'],zz['energy'])['normalized'],type=2,norm='ortho',axis=1)[:,:16];table=pd.read_csv(B/'raw_augmentation/heldout.csv')
for p in sorted((B/'raw_augmentation').glob('*.joblib')):
 m=joblib.load(p);ids=m['test_indices'];pred=m['model'].predict(X[ids]);expected=[table[(table.configuration==rr[i]['configuration'])&(table.training==('CuAu_plus_Cu100' if 'plus_Cu100' in p.name else 'CuAu_only'))].iloc[0].prediction_Cu_loading for i in ids];np.testing.assert_allclose(pred,expected,atol=1e-15);receipt(p)
for p in sorted((B/'raw_identity_augmentation').glob('*.joblib')):
 m=joblib.load(p);ids=m['test_indices'];name=p.stem.rsplit('_fold',1)[0];q=np.load(B/'raw_identity_augmentation'/f'{name}_oof.npz');look={v:j for j,v in enumerate(q['configuration'])};pred=(cls_score(m['model'],X[ids])>=m['thresholds']).astype(int);np.testing.assert_array_equal(pred,q['prediction'][[look[rr[i]['configuration']] for i in ids]]);receipt(p)
assert len(receipts)==272,len(receipts)
result={'models_replayed':len(receipts),'stress_fold_scenario_replays':stress_checks,'not_independent_confirmation':'saved predictions conditional on existing cohorts; no model refitting','models':receipts};print(json.dumps(result,indent=2))
