"""Reuse committed robustness models/scenarios; apply v3 unit-aware shared gate."""
from pathlib import Path
import ast,importlib.util,json
import numpy as np,pandas as pd,joblib
from scipy.ndimage import gaussian_filter1d
from tools.differential_route import infer
from tools.differential_identity import stable_rms
from tools.coverage_study import dump,sha
B=Path('study/signed_shape_detectability_v3');D=Path('study/relative_identity_guarded_v2');I=D/'results';P=D/'paired_robustness';r=json.load(open(I/'source_manifest.json'));z=np.load(I/'inputs.npz');t=np.load(I/'targets_folds.npz');e=z['energy'];mask=(e>=e[0]+1)&(e<=e[-1]-1);ec=e[mask];ids=np.flatnonzero(t['domain']=='CuAu');y=t['truth'];script=D/'code/paired_robustness.py';node=next(n for n in ast.parse(script.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='perturb');exec(compile(ast.Module(body=[node],type_ignores=[]),str(script),'exec'));rows=[];query=[]
scenarios=['native','shared_shift0p2','residual_shift0p2','gaussian_sigma0p25','common_acquisition_gain5pct','relative_sample_gain5pct','pair_noise1pct','shared_reference_noise1pct','unknown_detectability']
for aug in [False,True]:
 for scenario in scenarios:
  retained=np.zeros(len(ids),bool);correct=np.zeros(len(ids),bool);full=np.zeros(len(ids),bool);scores=np.full((len(ids),len(t['truth'][0])),np.nan)
  for f in range(5):
   te=ids[t['fold'][ids]==f];pos=np.flatnonzero(t['fold'][ids]==f);mp=P/'models'/f'signed_shape_aug{aug}_fold{f}.joblib';m=joblib.load(mp);a=z['raw_physical'][te];b=z['clean'][te];noise=None
   if scenario=='shared_reference_noise1pct':
    rng=np.random.default_rng(921+f);sa=.01*np.max(abs(a),axis=1);sb=.01*np.max(abs(b));common=rng.normal(0,sb,len(e));referror=rng.normal(0,sb,len(e));aa=a+common+rng.normal(size=a.shape)*sa[:,None];bb=b+common+referror;delta=(aa-bb)[:,mask];noise=np.sqrt(sa**2+sb**2);q=np.load(P/f'signed_shape_aug{aug}_shared_reference_noise1pct.npz')
   else:
    name='native' if scenario=='unknown_detectability' else scenario;aa,bb,scale=perturb(a,b,name,np.random.default_rng(624+f));delta=(aa-bb)/scale;q=np.load(P/f'signed_shape_aug{aug}_{name}.npz')
    if scenario=='pair_noise1pct':noise=np.sqrt((.01*np.max(abs(a),axis=1))**2+(.01*np.max(abs(b),axis=1))**2)
   lookup={int(i):j for j,i in enumerate(q['indices'])};full[pos]=np.all(q['prediction'][[lookup[int(i)] for i in te]]==y[te],axis=1)
   for j,i in enumerate(te):
    c=dict(input_kind='precomputed_difference',sign='reaction-minus-reference',energy_registration='declared synthetic registration scenario; no fitted alignment',normalization='shared_reference_near_edge_scale',reference_chemistry='audited same CuAu clean reference',domain='CuAu',noise='no measured instrument noise',processing='fixed supported interior; native lifetime retained',input_energy_eV=ec.tolist())
    if noise is not None:c['noise']={'rms_in_input_units':float(noise[j])}
    elif scenario!='unknown_detectability':c['detectability_basis']='audited_simulation_scale'
    answer=infer(mp,delta[j],None,{},{},'CuAu','CuAu',experimental_convention=c);keep=answer['status']=='conditional_prediction';retained[pos[j]]=keep
    if keep:
     pred=np.array([sp in answer['predicted_identity_set'] for sp in m['species']]);correct[pos[j]]=np.array_equal(pred,y[i]);scores[pos[j]]=[answer['scores'][sp] for sp in m['species']]
    query.append({'augmented_training':aug,'scenario':scenario,'configuration':r[i]['configuration'],'fold':f,'status':answer['status'],'retained':keep,'correct_if_retained':bool(correct[pos[j]]) if keep else None,'signal_to_noise':answer.get('signal_to_noise'),'model_sha256':sha(mp)})
  np.savez_compressed(B/f'robustness_aug{aug}_{scenario}.npz',indices=ids,truth=y[ids],retained=retained,retained_correct=correct,score=scores,full_model_correct_before_gate=full);rows.append({'augmented_training':aug,'scenario':scenario,'n_queries':len(ids),'retained_fraction':float(retained.mean()),'retained_exact_error':float(1-correct[retained].mean()) if retained.any() else None,'full_query_model_exact_error_before_gate':float(1-full.mean()),'correct_automatic_fraction_all_queries':float(correct.mean()),'uncertainty':'synthetic scenarios, not measured instrument uncertainty or calibrated detection limits'})
# Known clean self-differences and noisy near-clean controls are not new samples.
controls=[];mp=P/'models/signed_shape_augFalse_fold0.joblib';rng=np.random.default_rng(337)
for amp in [0.,1e-10,1e-4]:
 x=rng.normal(0,amp,len(ec)) if amp else np.zeros(len(ec))
 for factor in [1e-8,1.,1e8]:
  c=dict(input_kind='precomputed_difference',sign='reaction-minus-reference',energy_registration='synthetic clean/null control',normalization='documented_positive_divisor',normalization_details='positive factor applies to signal and noise',reference_chemistry='self-subtraction of the same clean state',domain='CuAu',noise={'rms_in_input_units':factor*amp},processing='synthetic; not independent clean measurements',input_energy_eV=ec.tolist());a=infer(mp,x*factor,None,{},{},'CuAu','CuAu',experimental_convention=c);controls.append({'noise_rms':amp,'positive_scale':factor,'status':a['status'],'predicted_set':a['predicted_identity_set']})
pd.DataFrame(rows).to_csv(B/'robustness_review_metrics.csv',index=False);pd.DataFrame(query).to_csv(B/'robustness_query_decisions.csv',index=False);dump(B/'clean_null_controls.json',controls);dump(B/'robustness_protocol.json',{'source_models':'committed guarded paired robustness DCT16/C1 logistic; no refits','source_script_sha256':sha(script),'source_existing_protocol_sha256':sha(P/'protocol.json'),'added_gaussian_sigma_eV':.25,'Gaussian_FWHM_eV':.588705,'noise':'independent pair noise or one reference error reused across queries per fold; common incident component separately cancels','detectability':'known synthetic noise in difference units; noiseless synthetic cases explicitly declare audited simulation scale; unknown quality yields review. Noise0 is not automatically infinite SNR.','window':[float(ec[0]),float(ec[-1])],'shift_sign':'sample mu(E+0.2), features move-0.2eV on fixed grid; common versus sample-only shift separate','not_experimental_validation':True});print(pd.DataFrame(rows).to_string(index=False))
