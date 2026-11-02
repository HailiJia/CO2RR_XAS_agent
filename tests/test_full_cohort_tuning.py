from pathlib import Path
import json
import numpy as np
from tools.full_cohort_tuning import group_weights,balanced_weights
R=Path(__file__).resolve().parents[1];B=R/'study/full_cohort_model_tuning_20261101';D=R/'study/dataset_20261101'

def test_duplicate_views_keep_one_group_mass_and_balanced_binary_mass():
 g=np.array(['a','a','b','c','c']);w=group_weights(g);target=np.array([1,1,0,1,1]);assert all(np.isclose(w[g==v].sum(),1.) for v in set(g));bw=balanced_weights(target,w);assert np.isclose(bw[target==0].sum(),bw[target==1].sum())

def test_all_query_records_and_every_permitted_train_view_are_used():
 r=json.load(open(D/'training_manifest.json'));mapping=json.load(open(D/'family_map.json'))['families'];fam=np.array([mapping[a['configuration']] for a in r])
 for s in json.load(open(B/'results/splits.json')):
  tr=np.array(s['train']);te=np.array(s['test']);np.testing.assert_array_equal(tr,np.flatnonzero(~np.isin(fam,fam[te])))
  for q in s['inner']:assert sorted(q['train']+q['validation'])==tr.tolist() and not set(fam[q['train']])&set(fam[q['validation']])
 for p in (B/'results').glob('*.npz'):
  if not p.name.startswith('inner_'):np.testing.assert_array_equal(np.load(p)['indices'],np.arange(141))

def test_inner_selection_and_final_full_data_fits_are_not_outer_score_choices():
 import joblib
 for s in json.load(open(B/'results/selections.json')):
  p=np.load(B/f"results/inner_{s['seed']}_fold{s['fold']}.npz")
  if s['protocol']=='nested':assert s['candidate']==int(np.argmax(p['quality']))
 for rec in json.load(open(B/'final_models/models.json')):
  m=joblib.load(B/'final_models'/rec['path']);np.testing.assert_array_equal(m['train_indices'],np.arange(141));assert len(m['test_indices'])==0

def test_actual_new_model_uses_shared_signed_inference_route():
 from tools.differential_route import infer
 import joblib
 rec=json.load(open(B/'results/model_replay.json'))[0];path=B/'results'/rec['path'];m=joblib.load(path);z=np.load(D/'training_arrays.npz');rows=json.load(open(D/'training_manifest.json'));i=next(int(k) for k in m['test_indices'] if np.linalg.norm(z['delta'][k])>m['floor']);domain=rows[i]['domain']
 c=dict(input_kind='precomputed_difference',sign='reaction-minus-reference',energy_registration='audited simulation photon energy',normalization='shared_reference_near_edge_scale',reference_chemistry='audited compatible clean reference',domain=domain,noise='no measured instrument noise',processing='native simulation and signed subtraction',input_energy_eV=z['energy'].tolist(),detectability_basis='audited_simulation_scale')
 a=infer(path,z['delta'][i],None,{},{},domain,domain,experimental_convention=c);assert a['status']=='conditional_prediction' and not a['absolute_coverage_available']
 sel=next(v for v in json.load(open(B/'results/selections.json')) if v['outer_model']==rec['path']);expected=np.load(B/f"results/{sel['protocol']}_{sel['seed']}_common.npz")['score'][i];np.testing.assert_array_equal([a['scores'][sp] for sp in m['species']],expected)

def test_explicit_full_cohort_agent_recipes_reach_training_contract():
 from agent.co2rr_xas_agent import LocalIntentParser
 for recipe in ['full_cohort_tuning_v1','full_cohort_final_v1']:
  assert LocalIntentParser().parse('Train '+recipe+' on prepared data').parameters['training_recipe']==recipe
