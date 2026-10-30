import numpy as np,pytest
from tools.differential_identity import identity_input,signed_shape,shape_floor
from tools.differential_xas import experimental_pair,ReferenceError

def conv(kind='reaction_reference_pair'):
 return dict(input_kind=kind,sign='reaction-minus-reference',energy_registration='common photon eV; external calibration',normalization='shared_reference_rms',reference_chemistry='same catalyst; residual coverage unknown',domain='CuAu',noise='not measured',processing='compatible background',relative_gain='common_or_independently_corrected',background='documented_compatible')

def test_relative_identity_needs_neither_absolute_Cu_nor_surface_fraction():
 e=np.linspace(8974,9029,276);b=2+np.tanh((e-8984)/4);a=b+.04*np.sin(e)
 p=experimental_pair(a,b,e,conv());q=experimental_pair(9*a,9*b,e,conv());np.testing.assert_allclose(p['delta'],q['delta'],atol=1e-15);assert p['delta'].min()<0<p['delta'].max();assert not p['absolute_coverage_available']
 d=identity_input(p['delta'],None,e,conv('precomputed_difference'));np.testing.assert_array_equal(d['delta'],p['delta'])
 with pytest.raises(ReferenceError):experimental_pair(a,b,e,conv(),task='coverage')

def test_contract_rejects_double_subtraction_arbitrary_pair_normalization_and_invalid_axes():
 e=np.linspace(1,10,50);v=np.sin(e);c=conv()
 for change in [{'sign':'reference-minus-reaction'},{'normalization':'independent_minmax'},{'relative_gain':'unknown'}]:
  with pytest.raises(ReferenceError):identity_input(v,v,e,{**c,**change})
 with pytest.raises(ReferenceError):identity_input(v,None,e[::-1],conv('precomputed_difference'))
 with pytest.raises(ReferenceError):identity_input(v,v,e,conv('precomputed_difference'))

def test_noise_floor_precedes_shape_scaling_and_preserves_sign():
 x=np.array([np.zeros(50),np.ones(50)*1e-14,np.sin(np.arange(50))]);s,valid=signed_shape(x,.001);assert valid.tolist()==[False,False,True];assert np.count_nonzero(s[:2])==0;assert s[2].min()<0<s[2].max();np.testing.assert_allclose(s[2],signed_shape(7*x,.001)[0][2]);assert shape_floor(x[:2])>=1e-10

def test_real_shared_route_rejects_missing_energy_and_abstains_null(tmp_path):
 import joblib
 from tools.differential_identity import VERSION
 from tools.differential_route import infer
 e=np.linspace(8974,9029,276);p=tmp_path/'model.joblib';joblib.dump(dict(input_contract=VERSION,energy=e,representation='signed_shape',floor=.001),p)
 r=infer(p,np.zeros(len(e)),None,{},{},'CuAu','CuAu',experimental_convention={**conv('precomputed_difference'),'input_energy_eV':e.tolist()});assert r['status']=='abstain_low_signal';assert r['predicted_identity_set'] is None
 with pytest.raises(ReferenceError,match='support'):infer(p,np.zeros(275),None,{},{},'CuAu','CuAu',experimental_convention={**conv('precomputed_difference'),'input_energy_eV':e[1:].tolist()})
 with pytest.raises(ReferenceError,match='Cross-domain'):infer(p,np.ones(len(e)),np.ones(len(e)),{},{},'CuAu','Cu100',experimental_convention=conv())
 with pytest.raises(ReferenceError,match='disagree'):infer(p,np.zeros(len(e)),None,{},{},'Cu100','Cu100',experimental_convention=conv('precomputed_difference'))

def test_agent_selects_explicit_relative_identity_recipe():
 from agent.co2rr_xas_agent import LocalIntentParser
 for text in ['Train relative_identity_v2 on normalized signed differential XAS','Fit normalized signed identity model']:
  assert LocalIntentParser().parse(text).parameters['training_recipe']=='relative_identity_v2'

def test_unit_signed_query_preserves_identity_scores_but_needs_signal_evidence(tmp_path):
 import joblib
 from tools.identity_comparison import fit_model,transform
 from tools.differential_identity import infer_identity,VERSION
 e=np.linspace(8974,9029,276);xx=np.array([np.sin(e+i/4) for i in range(8)]);yy=np.array([[i%2] for i in range(8)]);m=fit_model(transform(xx,e,'signed_shape','DCT16',1e-6),yy,'logistic',17);p=tmp_path/'m.joblib';joblib.dump(dict(input_contract=VERSION,energy=e,representation='signed_shape',feature='DCT16',floor=1e-6,model=m,thresholds=np.array([.5]),species=['CO'],domains=['CuAu']),p)
 c=conv('precomputed_difference');a=infer_identity(p,xx[0],None,e,c);b=infer_identity(p,19*xx[0],None,e,c);assert a['predicted_identity_set']==b['predicted_identity_set'];np.testing.assert_allclose(list(a['scores'].values()),list(b['scores'].values()),atol=1e-12)
 c['normalization']='signed_rms';assert infer_identity(p,xx[0],None,e,c)['status']=='abstain_unresolved_signal_floor';c['pre_normalization_signal_to_noise']=10;assert 'scores' in infer_identity(p,xx[0],None,e,c)

def test_shared_training_rejects_modified_prepared_export_before_fit(tmp_path):
 import json,hashlib
 from tools.audited_training import run_audited_training
 (tmp_path/'training_arrays.npz').write_bytes(b'changed')
 (tmp_path/'artifact_sha256.json').write_text(json.dumps({'training_arrays.npz':{'sha256':hashlib.sha256(b'original').hexdigest()}}))
 with pytest.raises(ValueError,match='hash mismatch'):run_audited_training(tmp_path,tmp_path/'out',recipe='relative_identity_v2')
