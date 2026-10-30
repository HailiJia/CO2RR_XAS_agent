import numpy as np,pytest,joblib
from tools.differential_route import infer
from tools.differential_identity import VERSION,stable_rms
from tools.identity_comparison import fit_model,transform
from tools.differential_xas import ReferenceError

@pytest.fixture
def model(tmp_path):
 e=np.linspace(8974,9029,276);x=np.array([np.sin((e-8990)/3+i/3) for i in range(10)]);y=np.array([[i%2] for i in range(10)]);fit=fit_model(transform(x,e,'signed_shape','Combined',.05),y,'logistic',11);p=tmp_path/'model.joblib';joblib.dump(dict(input_contract=VERSION,energy=e,representation='signed_shape',feature='Combined',floor=.05,model=fit,thresholds=np.array([.5]),species=['CO'],domains=['CuAu']),p);return p,e,x[0]

def convention(e):return dict(input_kind='precomputed_difference',sign='reaction-minus-reference',energy_registration='external common photon-energy registration',normalization='documented_positive_divisor',normalization_details='documented positive relative scale, no per-Cu calibration',reference_chemistry='same catalyst reference, residual coverage unknown',domain='CuAu',noise='unknown',processing='signed response retained',input_energy_eV=e.tolist())
def call(p,e,x,c,reference=None):return infer(p,x,reference,{},{},'CuAu','CuAu',experimental_convention=c)

def test_compatible_signal_noise_rescalings_preserve_scores_and_decisions(model):
 p,e,x=model;results=[]
 for factor in [1e-100,1e-8,1.,1e8,1e100]:
  c=convention(e);c['noise']={'rms_in_input_units':factor*stable_rms(x)/20};r=call(p,e,factor*x,c);assert r['status']=='conditional_prediction';results.append(r)
 for r in results[1:]:np.testing.assert_allclose(list(r['scores'].values()),list(results[0]['scores'].values()),atol=2e-12);assert r['predicted_identity_set']==results[0]['predicted_identity_set']

def test_unknown_detectability_zero_and_low_snr_do_not_become_apparent_signal(model):
 p,e,x=model
 for factor in [1e-8,1.,1e8]:
  c=convention(e);assert call(p,e,factor*x,c)['status']=='review_required_detectability';assert call(p,e,np.zeros(len(e)),c)['status']=='abstain_low_signal'
  c['noise']={'rms_in_input_units':factor*stable_rms(x)/2};r=call(p,e,factor*x,c);assert r['status']=='abstain_low_signal' and r['predicted_identity_set'] is None

def test_pre_normalization_snr_requires_no_raw_pair_and_signed_values_survive(model):
 p,e,x=model;unit=x/stable_rms(x);c=convention(e);c.update(normalization='signed_rms',pre_normalization_signal_to_noise=12);r=call(p,e,unit,c);assert 'scores' in r;assert unit.min()<0<unit.max();negative=call(p,e,-unit,c);assert not np.allclose(list(r['scores'].values()),list(negative['scores'].values()))
 with pytest.raises(ReferenceError,match='contradicts'):call(p,e,unit*.1,c)

def test_registration_sign_noise_basis_and_double_subtraction_guards(model):
 p,e,x=model;c=convention(e);c['pre_normalization_signal_to_noise']=10
 with pytest.raises(ReferenceError,match='subtracted twice'):call(p,e,x,c,reference=x)
 with pytest.raises(ReferenceError,match='support'):call(p,e[1:],x[1:],{**c,'input_energy_eV':e[1:].tolist()})
 with pytest.raises(ReferenceError):call(p,e,x,{**c,'sign':'reference-minus-reaction'})
 with pytest.raises(ReferenceError,match='contradictory'):call(p,e,x,{**c,'noise':{'rms_in_input_units':stable_rms(x)}})

def test_raw_relative_pair_gain_and_pair_noise_cancel_without_absolute_basis(model):
 p,e,x=model;b=2+np.tanh((e-8980)/4);a=b+.03*x;out=[]
 for k in [1e-8,1.,1e8]:
  c=convention(e);c.update(input_kind='reaction_reference_pair',normalization='shared_reference_rms',relative_gain='common_or_independently_corrected',background='documented_compatible',noise={'rms_difference_in_pair_units':k*.03*stable_rms(x)/20});out.append(call(p,e,k*a,c,reference=k*b))
 for r in out:assert r['status']=='conditional_prediction'
 for r in out[1:]:np.testing.assert_allclose(list(out[0]['scores'].values()),list(r['scores'].values()),atol=2e-12)

def test_unit_normalization_is_checked_on_declared_input_grid_not_model_crop(model):
 p,e,x=model;full=np.linspace(8960,9040,401);signal=np.sin((full-8990)/3);signal/=stable_rms(signal);c=convention(full);c.update(normalization='signed_rms',pre_normalization_signal_to_noise=10);r=call(p,full,signal,c);assert r['status']=='conditional_prediction'

def test_explicit_training_size_recipe_reaches_shared_agent_contract():
 from agent.co2rr_xas_agent import LocalIntentParser
 for text in ['Train signed_shape_size_v3 on audited data','Fit signed-shape training-size study']:
  assert LocalIntentParser().parse(text).parameters['training_recipe']=='signed_shape_size_v3'

def test_snr_boundary_is_reviewed_consistently_under_rescaling(model):
 p,e,x=model
 for k in [1e-100,1e-8,1.,1e8,1e100]:
  c=convention(e);c['noise']={'rms_in_input_units':k*stable_rms(x)/3};assert call(p,e,k*x,c)['status']=='review_required_detectability'
