from pathlib import Path
import copy,json
import numpy as np
import pytest
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import cosine
from tools.differential_xas import features,difference,pair,clean_scale,ReferenceError,experimental_pair
from tools.differential_training import SupportedColumns,reg_estimator
from tools.prepare_clean_Cu100 import prepare_xas
ROOT=Path(__file__).resolve().parents[1]

def test_signed_linear_features_keep_amplitude_and_DC():
 e=np.linspace(8974,9029,276);x=np.sin((e-8990)/3)+.2
 a=features(x,e);b=features(4.2*x,e)
 for key in a:np.testing.assert_allclose(b[key],4.2*a[key],atol=2e-12)
 assert a['signed'].min()<0<a['signed'].max()
 assert abs(a['DCT8'][0,0]-x.sum()/np.sqrt(len(x)))<1e-12
 assert all(np.count_nonzero(v)==0 for v in features(difference(x,x,2),e).values())

def test_cumulative_difference_has_correct_units_and_can_decrease():
 e=np.linspace(0,10,101);a=features(e-5,e);expected=.5*(e-5)**2-12.5
 np.testing.assert_allclose(a['cumulative_difference'][0],expected,atol=1e-12)
 assert np.any(np.diff(a['cumulative_difference'][0])<0)
 np.testing.assert_allclose(a['derivative'][0,6:-6],1,atol=1e-12)

def fixture_records():
 e=np.linspace(8974,9029,276);clean=.02+1/(1+np.exp(-(e-8983)));ads=clean+.02*np.sin(e-8980)
 def r(y,rid):return {'record_id':rid,'sample':{'composition':{'adsorbate_counts':{}}},'source_scale_correction':{'isotropic_cross_section_unit':'Mbarn_per_Cu','native_reconstruction_verified':True,'physical_domain':'CuAu','physical_top_layers':[4,5],'populations':{4:36,5:36}},'measurement':{'series':[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'unit':'eV','values':e.tolist()}],'channels':[{'unit':'Mbarn_per_Cu','values':y.tolist()}]} for l in [4,5]]}}
 return e,r(ads,'ads'),r(clean,'clean')

def test_pair_units_parent_population_and_missing_reference_guards():
 e,a,b=fixture_records();p=pair(a,b,e,{4:36,5:36},{4:36,5:36},[4,5],'CuAu','CuAu');assert p['delta'].min()<0<p['delta'].max()
 with pytest.raises(ReferenceError):pair(a,None,e,{4:36,5:36},{4:36,5:36},[4,5],'CuAu','CuAu')
 with pytest.raises(ReferenceError,match='Cross-facet'):pair(a,b,e,{4:36,5:36},{4:36,5:36},[4,5],'Cu100','CuAu')
 bad=copy.deepcopy(b);bad['sample']['composition']['adsorbate_counts']={'H':1}
 with pytest.raises(ReferenceError,match='clean'):pair(a,bad,e,{4:36,5:36},{4:36,5:36},[4,5],'CuAu','CuAu')
 bad=copy.deepcopy(a);bad['measurement']['series'][0]['channels'][0]['unit']='arb'
 with pytest.raises(ReferenceError,match='units'):pair(bad,b,e,{4:36,5:36},{4:36,5:36},[4,5],'CuAu','CuAu')
 with pytest.raises(ReferenceError,match='populations'):pair(a,b,e,{},{4:36,5:36},[4,5],'CuAu','CuAu')

def test_shared_gain_cancels_only_with_shared_physical_calibration():
 e,a,b=fixture_records();first=pair(a,b,e,{4:36,5:36},{4:36,5:36},[4,5],'CuAu','CuAu')
 for r in [a,b]:
  for s in r['measurement']['series']:s['channels'][0]['values']=(np.array(s['channels'][0]['values'])*7).tolist()
 second=pair(a,b,e,{4:36,5:36},{4:36,5:36},[4,5],'CuAu','CuAu');np.testing.assert_allclose(first['delta'],second['delta'],atol=1e-14)


def test_one_reference_offset_and_constant_padding_cannot_add_information():
 rng=np.random.default_rng(8);x=rng.normal(size=(15,32));clean=rng.normal(size=32);y=rng.normal(size=15);tr=np.arange(10);te=np.arange(10,15);recipe=('DCT16','ridge',10)
 p=reg_estimator(recipe).fit(x[tr],y[tr]).predict(x[te]);d=reg_estimator(recipe).fit(x[tr]-clean,y[tr]).predict(x[te]-clean);padded=np.hstack([np.tile(clean,(15,1)),x-clean]);q=reg_estimator(recipe).fit(padded[tr],y[tr]).predict(padded[te]);np.testing.assert_allclose(p,d,atol=1e-12);np.testing.assert_allclose(p,q,atol=1e-12)


def test_real_source_reconstructions_and_nonconstant_weight_correction():
 p=ROOT/'study/differential_v1/scale';r=json.load(open(p/'source_weight_audit.json'));assert len(r)==232 and all(x['native_reproduced'] for x in r)
 for l,total in [(4,51),(5,52)]:
  row=next(x for x in r if x['configuration']=='plain' and x['layer']==l);assert row['source_multiplicity_sum']==total and row['physical_Cu_sites']==36
 import pandas as pd
 csv=p/'reconvolution/plain/layer_4/scale_comparison.csv'
 if csv.exists():d=pd.read_csv(csv)
 else:
  import tarfile
  with tarfile.open(p.parent/'artifacts/reconvolution.tar.gz') as tf:d=pd.read_csv(tf.extractfile('scale/reconvolution/plain/layer_4/scale_comparison.csv'))
 ratio=d.corrected_per_Cu/d.native_aggregate_per_unique_Cu_without_symmetry_correction;assert np.ptp(ratio)>.001


def test_calibrated_coverage_query_cannot_ignore_reference_or_surface_fraction():
 e,a,b=fixture_records();y=np.array(b['measurement']['series'][0]['channels'][0]['values'])
 with pytest.raises(ReferenceError):experimental_pair(y,None,e,{},task="coverage")
 convention={k:'declared' for k in ['incident_intensity_correction','thickness_concentration','self_absorption','detection_geometry','reference_residual_coverage']};convention.update(Cu_basis='same independently calibrated per-Cu basis',edge_step_rule='shared reference-derived scale with fixed documented windows')
 with pytest.raises(ReferenceError,match='fraction'):experimental_pair(y,y,e,convention,task="coverage")
 convention['surface_fraction']=.2;assert np.all(experimental_pair(y,y,e,convention,task="coverage")['delta']==0)

def test_XAS_preparation_cannot_bypass_failed_relaxation(tmp_path):
 (tmp_path/'OUTCAR').write_text('Electronic iterations only; unfinished relaxation')
 with pytest.raises(ValueError,match='Successful relaxation'):prepare_xas(tmp_path,tmp_path/'XAS')

def test_new_natural_language_route():
 from agent.co2rr_xas_agent import LocalIntentParser
 for text in ['Train a differential XAS model','Fit a clean-referenced XAS model']:
  assert LocalIntentParser().parse(text).parameters['training_recipe']=='differential_v1'
