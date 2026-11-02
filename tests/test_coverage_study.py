"""Scientific guards for variable-layer, coverage and separate-domain use."""
from pathlib import Path
import json,copy
import numpy as np
import pytest,joblib
from tools.coverage_study import native
from tools.coverage_inference import predict,RECIPES
from tools.cu100_lineage import array
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'study/coverage_20261026'

def test_declared_compact_search_no_structural_features():
 assert len(RECIPES)==16
 assert {r[0] for r in RECIPES}=={'Combined','DCT8','DCT16','DCT24'}
 assert {r[2] for r in RECIPES if r[1]=='trees'}=={2,4}

def test_native_rejects_incomplete_nonfinite_and_noncanonical(tmp_path):
 p=tmp_path/'layer_4';p.mkdir()
 np.savetxt(p/'Cu_layer4_conv.txt',np.c_[np.arange(60),np.ones(60)],header='Energy intensity',comments='')
 np.savetxt(p/'Cu_layer4_resume_conv.txt',np.c_[np.arange(60),np.ones(60)],header='Energy intensity',comments='')
 assert set(native(tmp_path))=={4}
 a=np.c_[np.arange(60),np.ones(60)];a[3,1]=np.nan;np.savetxt(p/'Cu_layer4_conv.txt',a,header='Energy intensity',comments='')
 assert not native(tmp_path)

def test_ASE_array_preserves_original_atom_order():
 d={'__ndarray__':[[2,3],'float64',[1,2,3,4,5,6]]}
 assert np.array_equal(array(d),[[1,2,3],[4,5,6]])

def test_actual_predict_requires_populations_and_full_energy_support():
 m=joblib.load(B/'models/coverage_fold0.joblib');rows=json.loads((B/'model_configuration_manifest.json').read_text());r=next(r for r in rows if r['group'] in m['test_groups']);d=json.loads((B/'records'/r['configuration']/'isaac_record_layers.json').read_text());pops={int(k):v['population'] for k,v in r['layer_evidence'].items()}
 assert np.isfinite(predict(d,m,pops)).all()
 with pytest.raises(ValueError,match='populations'):predict(d,m,{})
 bad=copy.deepcopy(d)
 for s in bad['measurement']['series']:s['independent_variables'][0]['values']=s['independent_variables'][0]['values'][10:];s['channels'][0]['values']=s['channels'][0]['values'][10:]
 with pytest.raises(ValueError,match='support'):predict(bad,m,pops)

def test_external_Cu100_never_enters_CuAu_training():
 m=joblib.load(B/'models/CuAu_only_external_Cu100.joblib');rows=json.loads((B/'model_configuration_manifest.json').read_text());domain={r['group']:r['domain'] for r in rows}
 assert {domain[g] for g in m['train_groups']}=={'CuAu'}
 assert {domain[g] for g in m['test_groups']}=={'Cu100'}
 assert not set(m['train_groups'])&set(m['test_groups'])


def test_coverage_denominators_and_missing_hollow_support():
 rows=json.loads((B/'model_configuration_manifest.json').read_text())
 for r in rows:
  assert r['coverage_ML']==r['n_CO']/r['surface_sites']
  assert r['surface_sites']==(68 if r['domain']=='CuAu' else 4)
  assert r['surface_Cu']==(36 if r['domain']=='CuAu' else 4)
 s=json.loads((B/'contact_support.json').read_text())['hollow_series_molecule_support']
 assert s['Cu1Au2']==0 and s['Cu0Au3']==0

def test_original_snapshots_match_Cu100_but_do_not_claim_termination():
 rows=json.loads((B/'record_geometry_audit.json').read_text());cu=[r for r in rows if r['domain']=='Cu100'];assert len(cu)==17
 for r in cu:
  a=r['ASE_lineage'];assert a['status']=='matched_original_ASE_snapshot'
  assert a['atom_mapping']['max_distance_A']<2e-3
  assert 'not supplied' in a['relaxation_status']

def test_missing_physical_top_layer_is_never_replaced_by_deeper_layer():
 m=joblib.load(B/'models/coverage_fold0.joblib');rows=json.loads((B/'model_configuration_manifest.json').read_text());r=next(r for r in rows if r['group'] in m['test_groups']);d=json.loads((B/'records'/r['configuration']/'isaac_record_layers.json').read_text());d['measurement']['series']=[s for s in d['measurement']['series'] if s['series_id']!='Cu_K_layer_5']
 with pytest.raises(ValueError,match='physical top-two'):predict(d,m,{3:36,4:36})

def test_shared_recipe_guards_archived_output_and_parser(tmp_path):
 from tools.audited_training import run_audited_training
 from agent.co2rr_xas_agent import LocalIntentParser
 (tmp_path/'archive').write_text('unchanged')
 with pytest.raises(ValueError,match='empty'):run_audited_training('/data/export',tmp_path,recipe='coverage_v1')
 p=LocalIntentParser().parse('Train an XAS model for CO coverage using data root /data/export')
 assert p.parameters['training_recipe']=='coverage_v1'
