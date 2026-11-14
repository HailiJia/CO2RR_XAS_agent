"""Old/new shared inference decisions on the committed fixed joint models."""
from pathlib import Path
import importlib.util,json
import numpy as np,joblib
from tools.differential_route import infer
from tools.coverage_study import dump
B=Path('study/signed_shape_detectability_v3');I=Path('study/relative_identity_guarded_v2/results');z=np.load(I/'inputs.npz');r=json.load(open(I/'source_manifest.json'));spec=importlib.util.spec_from_file_location('tools._pre_fix_identity',B/'code/pre_fix_differential_identity.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old);records=[]
for rep in ['relative_delta','signed_shape']:
 for f in range(5):
  p=I/'models'/f'fixed_rep_joint130_{rep}_trees_{f}.joblib';m=joblib.load(p)
  for i in m['test_indices']:
   c=dict(input_kind='precomputed_difference',sign='reaction-minus-reference',energy_registration='audited simulation common photon-energy convention',normalization='shared_reference_near_edge_scale',reference_chemistry='actual substrate-specific clean simulation',domain=r[i]['domain'],noise='no measured noise; controlled simulation',processing='audited fixed top2 and common near-edge divisor',input_energy_eV=z['energy'].tolist(),detectability_basis='audited_simulation_scale');a=old.infer_identity(p,z['delta'][i],None,z['energy'],c);b=infer(p,z['delta'][i],None,{},{},r[i]['domain'],r[i]['domain'],experimental_convention=c);assert a['predicted_identity_set']==b['predicted_identity_set'];assert a['status']==b['status'];error=0.
   if 'scores' in a:error=max(abs(a['scores'][k]-b['scores'][k]) for k in a['scores']);assert error==0
   records.append({'representation':rep,'configuration':r[i]['configuration'],'fold':f,'old_status':a['status'],'new_status':b['status'],'identity_set_unchanged':True,'max_score_change':error})
dump(B/'saved_simulation_inference_comparison.json',{'queries':records,'n':len(records),'decision_changes':0,'max_score_change':max(a['max_score_change'] for a in records),'scope':'declared original audited simulation units; unknown experimental detectability deliberately not claimed'})
print('260 saved-simulation query decisions unchanged; scores bitwise unchanged')
