import json
from pathlib import Path
import numpy as np
import pytest
from tools.dataset_refresh import symmetry_rule
from tools.coverage_study import deck_structure
from agent.co2rr_xas_agent import LocalIntentParser
B=Path(__file__).resolve().parents[1]/'study/dataset_20261029'
def test_updated_export_route_does_not_select_historical_depth_selector():
 p=LocalIntentParser()
 for text in ['retrain adsorbate identity with updated data','train an identity model using dataset_refresh_v1','retrain model on CO2RR_1029']:
  assert p.parse(text).parameters['training_recipe']=='dataset_refresh_v1'
 assert p.parse('train adsorbate identity with updated_data_v3').parameters['training_recipe']=='updated_data_v3'
def test_full_source_and_symmetry_representative_rules_are_distinct():
 s={'atoms':['Cu']*4,'positions':np.array([[0,0,0],[.5,0,0],[0,.5,0],[.5,.5,0]]),'cell':np.eye(3)}
 full=[{'atom_index':i,'natomsym_f':2. if i in [2,3] else 1.} for i in [1,2,3,4]];rule,w,_=symmetry_rule(s,[1,2,3,4],full);assert rule=='all_physical_sites' and w[2]==.5
 reps=[full[i] for i in [0,1,3]];rule,w,ev=symmetry_rule(s,[1,2,3,4],reps);assert rule=='nonoverlapping_symmetry_representatives' and w[2]==1 and ev['orbits']['2']==[2,3]
 with pytest.raises(ValueError):symmetry_rule(s,[1,2,3,4],full[:1])
def test_versioned_records_and_clean_reference_are_consistent():
 p=B/'records/Cu100/Cu100_2x2x4_clean/isaac_record_layers.json';d=json.load(open(p));assert d['sample']['material']['formula']=='Cu16' and d['sample']['composition']['adsorbate_counts']=={}
 assert all(v['channels'][0]['unit']=='Mbarn_per_Cu' for v in d['measurement']['series'])
 for p in (B/'records').rglob('isaac_record_layers.json'):
  d=json.load(open(p));assert d['sample']['material']['formula']!='not_specified'
 r=json.load(open(B/'top_two_recovery.json'));assert all(r['layer_comparison'][str(l)]['status']=='matched' for l in [4,5]);assert any(r['layer_comparison'][str(l)]['status']!='matched' for l in [2,3])
def test_training_preserves_groups_labels_and_delta_amplitude():
 rows=json.load(open(B/'training_manifest.json'));z=np.load(B/'training_arrays.npz');assert sum(r['old99'] for r in rows)==99
 assert len({r['group'] for r in rows})==len(rows)
 np.testing.assert_allclose(z['raw_physical']-z['clean'],z['delta'],atol=1e-14)
 assert z['delta'].min()<0<z['delta'].max()
 for r in rows:
  assert set(r['identity_labels'])<=set(['CHO','CO','CO2','COCHO','COH','COOH','H','O','OCCOH','OCHO','OH'])

def test_representative_symmetry_requires_an_actual_cell_isometry():
 s={'atoms':['Cu']*4,'positions':np.array([[0,0,0],[1,0,0],[0,.5,0],[1,.5,0]]),'cell':np.diag([2.,1.,1.])}
 reps=[{'atom_index':1,'natomsym_f':1.},{'atom_index':2,'natomsym_f':2.},{'atom_index':4,'natomsym_f':1.}]
 with pytest.raises(ValueError,match='isometry'):symmetry_rule(s,[1,2,3,4],reps)
 with pytest.raises(ValueError,match='multiplicity'):symmetry_rule(s,[1],[{'atom_index':1,'natomsym_f':0.}])
