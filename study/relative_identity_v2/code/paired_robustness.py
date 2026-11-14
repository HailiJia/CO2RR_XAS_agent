"""Pair-first synthetic sensitivity; fixed interior window, paired training augmentation."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from scipy.ndimage import gaussian_filter1d
from tools.identity_comparison import transform,fit_model,score_model
from tools.differential_identity import shape_floor,VERSION
from tools.spectral_multilabel import SPECIES,f1_metrics
from tools.coverage_study import dump,sha
B=Path('study/relative_identity_v2');I=B/'results';O=B/'paired_robustness';O.mkdir(exist_ok=True);(O/'models').mkdir(exist_ok=True);z=np.load(I/'inputs.npz');r=json.load(open(I/'source_manifest.json'));t=np.load(I/'targets_folds.npz');e=z['energy'];mask=(e>=e[0]+1)&(e<=e[-1]-1);ec=e[mask];ids=np.flatnonzero(t['domain']=='CuAu');y=t['truth'];rows=[];replays=[];scenarios=['native','shared_shift0p2','residual_shift0p2','gaussian_sigma0p25','pair_noise1pct','common_acquisition_gain5pct','relative_sample_gain5pct','processed_scale_error5pct']
def perturb(a,b,scenario,rng):
 a=a.copy();b=b.copy();scale=1.
 if scenario=='augmentation':
  shift=float(rng.choice([-.2,.2]));relative=float(rng.choice([-.2,.2]));aa=np.array([np.interp(ec+shift+relative,e,x) for x in a]);bb=np.array([np.interp(ec+shift,e,x) for x in b]);aa=gaussian_filter1d(aa,.25/np.median(np.diff(ec)),axis=1);bb=gaussian_filter1d(bb,.25/np.median(np.diff(ec)),axis=1);aa*=rng.uniform(.95,1.05);shared=rng.normal(size=len(ec))*.01*max(b.max(),1e-12);aa+=shared+rng.normal(size=aa.shape)*(.01*np.max(abs(a),axis=1))[:,None];bb+=shared+rng.normal(size=bb.shape)*(.01*np.max(abs(b),axis=1))[:,None];return aa,bb,1.
 if scenario=='shared_shift0p2':return np.array([np.interp(ec+.2,e,x) for x in a]),np.array([np.interp(ec+.2,e,x) for x in b]),1.
 if scenario=='residual_shift0p2':return np.array([np.interp(ec+.2,e,x) for x in a]),b[:,mask],1.
 if scenario=='gaussian_sigma0p25':a=gaussian_filter1d(a,.25/np.median(np.diff(e)),axis=1,mode='reflect');b=gaussian_filter1d(b,.25/np.median(np.diff(e)),axis=1,mode='reflect')
 if scenario=='pair_noise1pct':
  shared=rng.normal(size=len(e))*.01*max(b.max(),1e-12);a+=shared+rng.normal(size=a.shape)*(.01*np.max(abs(a),axis=1))[:,None];b+=shared+rng.normal(size=b.shape)*(.01*np.max(abs(b),axis=1))[:,None]
 if scenario=='common_acquisition_gain5pct':a*=1.05;b*=1.05;scale=1.05
 if scenario=='relative_sample_gain5pct':a*=1.05
 if scenario=='processed_scale_error5pct':scale=1/1.05
 return a[:,mask],b[:,mask],scale
for rep in ['corrected_normalized','relative_delta','signed_shape']:
 for aug in [False,True]:
  scores={s:np.zeros_like(y[ids],float) for s in scenarios}
  for fold in range(5):
   tr=ids[t['fold'][ids]!=fold];te=ids[t['fold'][ids]==fold];pos=np.flatnonzero(t['fold'][ids]==fold);a=z['raw_physical'][tr];b=z['clean'][tr];delta=(a-b)[:,mask];floor=shape_floor(delta);x=a[:,mask] if rep=='corrected_normalized' else delta;yy=y[tr]
   if aug:
    aa,bb,scale=perturb(a,b,'augmentation',np.random.default_rng(20261020+fold));extra=aa/scale if rep=='corrected_normalized' else (aa-bb)/scale;x=np.vstack([x,extra]);yy=np.vstack([yy,yy])
   m=fit_model(transform(x,ec,rep,'DCT16',floor),yy,'logistic',20261020);bundle=dict(input_contract=VERSION,representation=rep,feature='DCT16',kind='logistic',model=m,floor=floor,thresholds=np.full(len(SPECIES),.5),energy=ec,species=SPECIES,domains=['CuAu'],train_indices=tr,test_indices=te,train_groups=t['groups'][tr].tolist(),test_groups=t['groups'][te].tolist(),reference_ids=sorted({r[i]['reference_id'] for i in tr}),reference_record_hashes=sorted({r[i]['reference_record_sha256'] for i in tr}),training_source=sha(Path(__file__)),augmentation=aug);p=O/'models'/f'{rep}_aug{aug}_fold{fold}.joblib';joblib.dump(bundle,p,compress=3);reload=joblib.load(p)
   for scenario in scenarios:
    aa,bb,scale=perturb(z['raw_physical'][te],z['clean'][te],scenario,np.random.default_rng(624+fold));xx=aa/scale if rep=='corrected_normalized' else (aa-bb)/scale;X=transform(xx,ec,rep,'DCT16',floor);s=score_model(m,X);assert np.array_equal(s,score_model(reload['model'],X));scores[scenario][pos]=s
   replays.append({'model':str(p.relative_to(O)),'sha256':sha(p),'all_scenarios_exact_reload':True,'groups_disjoint':not bool(set(t['groups'][tr])&set(t['groups'][te]))})
  for scenario,s in scores.items():
   pred=(s>=.5).astype(int);np.savez_compressed(O/f'{rep}_aug{aug}_{scenario}.npz',indices=ids,truth=y[ids],score=s,prediction=pred,fold=t['fold'][ids]);rows.append(dict(representation=rep,training_pair_augmentation=aug,scenario=scenario,n=len(ids),**f1_metrics(y[ids],pred)))
pd.DataFrame(rows).to_csv(O/'metrics.csv',index=False);dump(O/'model_replay.json',replays);dump(O/'protocol.json',dict(scope='synthetic paired stress; no measured experimental uncertainty supplied',energy_eV=[float(ec[0]),float(ec[-1])],n_points=len(ec),crop_eV=1,normalization='same corrected-normalized raw vs relative delta vs signed-shape conventions, recomputed on same supported interior window',training='DCT16 C1 logistic, threshold.5; one paired augmentation, all views in original group',noise='1% of each query peak; one shared reference vector plus independent sample/reference noise; shared component cancels in subtraction',shared_reference='same simulated reference per domain; noise common across test queries within fold; not independent references',gain='common acquisition gain changes shared divisor too and cancels; sample-only residual gain and processed divisor error separate',broadening='additional Gaussian sigma.25 eV on native pair, reflect finite-grid kernel; no removal of lifetime broadening',augmentation='shared±.2 eV plus residual±.2 eV; extra Gaussian sigma.25 eV; sample gain±5%; shared+independent noise1%; fixed illustrative settings',scenarios=scenarios,source_sha256=sha(Path(__file__))));print('Pair-first robustness saved;30 models,48 query scenarios')
