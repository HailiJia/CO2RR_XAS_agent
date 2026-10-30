"""Predeclared simulated paired-error tests, not experimental validation."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from scipy.ndimage import gaussian_filter1d
from scipy.fft import dct
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from .differential_training import reg_estimator,cls_fit,cls_score,SEED
from .differential_xas import features
from .spectral_multilabel import feature_bank,encode_labels,f1_metrics
from .updated_data_training import threshold_fit
from .progress_snapshots import write_json


def scenarios():
 r=[{'name':'clean','kind':'clean','value':0.}]
 for v in [.2,.5,1.,2.]:
  for sign in [-1,1]:r.append({'name':f'shared_shift_{sign*v:g}','kind':'shared_shift','value':sign*v})
 for v in [.2,.5,1.]:
  for sign in [-1,1]:r.append({'name':f'relative_reference_shift_{sign*v:g}','kind':'relative_shift','value':sign*v})
 for kind in ['shared_gain_recalibrated','relative_adsorbate_gain','edge_step_error']:
  for v in [-.05,-.02,.02,.05]:r.append({'name':kind+f'_{v:g}','kind':kind,'value':v})
 for v in [.25,.5,1.]:r.append({'name':f'extra_Gaussian_sigma_{v:g}','kind':'broadening','value':v})
 for kind in ['independent_pair_noise','shared_reference_noise']:
  for v in [.01,.02,.05]:r.append({'name':kind+f'_{v:g}','kind':kind,'value':v})
 for v in [.1,.25,.5,1.]:r.append({'name':f'surface_fraction_{v:g}','kind':'surface_fraction','value':v})
 return r

def perturb(a,b,e,scenario,seed,shared_ref=None):
 """Arrays are on a clean-derived common scale before the fixed interior crop."""
 rng=np.random.default_rng(seed);kind=scenario['kind'];v=scenario['value'];aa=a.copy();bb=b.copy();mask=(e>=8976-1e-8)&(e<=9027+1e-8);grid=e[mask];scale=1.
 if kind in ['shared_shift','relative_shift']:
  if kind=='shared_shift':aa=np.array([np.interp(grid-v,e,x) for x in a])
  else:aa=a[:,mask]
  bb=np.array([np.interp(grid-v,e,x) for x in b]);return aa,bb,grid
 if kind=='shared_gain_recalibrated':aa*=1+v;bb*=1+v;scale=1+v
 elif kind=='relative_adsorbate_gain':aa*=1+v
 elif kind=='edge_step_error':scale=1+v
 elif kind=='broadening':
  width=v/(e[1]-e[0]);mass=gaussian_filter1d(np.ones_like(e),width,mode='constant',cval=0,truncate=5);aa=gaussian_filter1d(aa,width,axis=1,mode='constant',cval=0,truncate=5)/mass;bb=gaussian_filter1d(bb,width,axis=1,mode='constant',cval=0,truncate=5)/mass
 elif kind in ['independent_pair_noise','shared_reference_noise']:
  aa+=rng.normal(size=a.shape)*v*np.max(abs(a),axis=1)[:,None]
  bb+=rng.normal(size=b.shape)*v*np.max(abs(b),axis=1)[:,None] if kind=='independent_pair_noise' else np.tile(shared_ref if shared_ref is not None else rng.normal(size=e.shape), (len(b),1))*v*np.max(abs(b),axis=1)[:,None]
 elif kind=='surface_fraction':aa=b+v*(a-b)
 return aa[:,mask]/scale,bb[:,mask]/scale,grid

def training_view(a,b,e,fold,j):
    rng=np.random.default_rng(SEED+100*fold+j);grid=e[(e>=8976-1e-8)&(e<=9027+1e-8)];shift=(-.5,-.2,.2,.5)[j];relative=.2*(-1)**j;width=.25/(e[1]-e[0]);mass=gaussian_filter1d(np.ones_like(e),width,mode='constant',cval=0,truncate=5)
    aa=gaussian_filter1d(a,width,axis=1,mode='constant',cval=0,truncate=5)/mass;bb=gaussian_filter1d(b,width,axis=1,mode='constant',cval=0,truncate=5)/mass
    aa=np.array([np.interp(grid-shift,e,x) for x in aa]);bb=np.array([np.interp(grid-shift-relative,e,x) for x in bb]);aa*=1+rng.normal(0,.01);bb*=1+rng.normal(0,.01);aa+=rng.normal(0,.01*np.max(abs(aa)),aa.shape);bb+=rng.normal(0,.01*np.max(abs(bb)),bb.shape)
    return aa,bb,grid

def feature_banks(a,b,e):
 norm=feature_bank(a,e);return {'raw_normalized':{'DCT16':dct(norm['normalized'],type=2,norm='ortho',axis=1)[:,:16]},'raw_physical':{'DCT16':features(a,e)['DCT16']},'delta':{'DCT16':features(a-b,e)['DCT16']},'clean_plus_delta':{'DCT16':np.hstack([features(b,e)['DCT16'],features(a-b,e)['DCT16']])}}

def run(study):
 B=Path(study);O=B/'robustness';O.mkdir(exist_ok=True);z=np.load(B/'paired_arrays.npz');m=sorted([r for r in json.loads((B/'pair_manifest.json').read_text()) if 'pair_index'in r],key=lambda r:r['pair_index']);ids=np.array([i for i,r in enumerate(m) if r['eligible_primary']]);groups=np.array([r['group'] for r in m]);declared=scenarios();write_json(O/'predeclared_protocol.json',{'seed':SEED,'scenarios':declared,'common_crop_eV':[8976.,9027.],'interpolation':'linear, within full 8974–9029 source support; no extrapolation','features':'fixed DCT16; all representations on identical cropped grid','coverage_model':'training StandardScaler + Ridge alpha10','identity_model':'independent balanced Logistic C1; constant columns removed in training; inner grouped thresholds','augmentation':'four deterministic training-only variants per physical pair; shared shifts ±0.5 eV, relative shifts ±0.2 eV, gain 1%, noise 1%, Gaussian sigma0.25 eV; settings frozen before these stress outputs','noise_replicates':3,'noise_model':'additive independent Gaussian per energy point; amplitude fractions of peak intensity; shared-reference scenarios reuse one noise curve across all query pairs; not a measured instrument model','calibration':'reference-derived common scale fixed independently; shared multiplicative gains cancel when its same measured gain is included; edge-step uncertainty is separate','surface_fraction':'stable-background assumption a_obs=b+f*(a-b); f*theta is a response proxy, not unique experimental coverage','native_lifetime_broadening':'preserved; Gaussian is extra instrumental scenario','no_experimental_accuracy_claim':True})
 a=z['raw_physical'];b=z['clean'];e=z['energy'];clean_s={'name':'clean','kind':'clean','value':0};ca,cb,ee=perturb(a,b,e,clean_s,SEED);base=feature_banks(ca,cb,ee);results=[];examples=[]
 for task,chosen in [('identity',ids),('CO_coverage',np.array([i for i in ids if m[i]['CO_only']]))]:
  y=encode_labels([m[i]['labels'] for i in chosen]) if task=='identity' else np.array([m[i]['coverage_metal'] for i in chosen]);folds=list(GroupKFold(min(5,len(chosen))).split(chosen,groups=groups[chosen]));allpred={}
  for training in ['unaugmented','training_only_augmented']:
   bundles={}
   for fold,(tr0,te0) in enumerate(folds):
    tr=chosen[tr0];te=chosen[te0];trainbanks={rep:base[rep]['DCT16'][tr] for rep in base};trainy=y[tr0]
    if training=='training_only_augmented':
     batches=[trainbanks];yy=[trainy]
     for j in range(4):
      pa,pb,g=training_view(a[tr],b[tr],e,fold,j);fb=feature_banks(pa,pb,g);batches.append({rep:fb[rep]['DCT16'] for rep in base});yy.append(trainy)
     trainbanks={rep:np.vstack([x[rep] for x in batches]) for rep in base};trainy=np.concatenate(yy)
    for rep in base:
     rec=('DCT16','ridge',10.) if task=='CO_coverage' else ('DCT16','logistic',1.)
     if task=='CO_coverage':model=reg_estimator(rec).fit(trainbanks[rep],trainy);th=None
     else:
      # Calibration uses ORIGINAL inner-training/validation groups. No outer
      # labels and no augmented copy enters validation with its own parent in fit.
      inner_score=np.zeros_like(y[tr0],float)
      for it,iv in GroupKFold(min(3,len(tr))).split(tr,groups=groups[tr]):
       XX=base[rep]['DCT16'][tr[it]];YY=y[tr0[it]]
       if training=='training_only_augmented':
        extra=[]
        for j in range(4):
         pa,pb,g=training_view(a[tr[it]],b[tr[it]],e,fold,j);extra.append(feature_banks(pa,pb,g)[rep]['DCT16'])
        XX=np.vstack([XX]+extra);YY=np.tile(YY,(5,1))
       im=cls_fit(XX,YY,rec);inner_score[iv]=cls_score(im,base[rep]['DCT16'][tr[iv]])
      _,th,_=threshold_fit(y[tr0],inner_score);model=cls_fit(trainbanks[rep],trainy,rec)
     bundles[(fold,rep)]=(model,th)
     path=O/'models'/f'{task}_{training}_{rep}_fold{fold}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump({'model':model,'thresholds':th,'task':task,'representation':rep,'energy':ee,'training_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'test_pair_indices':te,'training_protocol':training},path)
   for sc_index,sc in enumerate(declared):
    repetitions=3 if 'noise' in sc['kind'] else 1
    for repno in range(repetitions):
     rng=np.random.default_rng(SEED+sc_index*100+repno);shared=rng.normal(size=e.shape);pa,pb,g=perturb(a,b,e,sc,SEED+sc_index*100+repno,shared);fb=feature_banks(pa,pb,g)
     for rep in base:
      prediction=np.zeros_like(y,float)
      for fold,(tr0,te0) in enumerate(folds):
       model,th=bundles[(fold,rep)];XX=fb[rep]['DCT16'][chosen[te0]];prediction[te0]=model.predict(XX) if task=='CO_coverage' else (cls_score(model,XX)>=th).astype(int)
      metric={'MAE':float(mean_absolute_error(y,prediction))} if task=='CO_coverage' else f1_metrics(y,prediction)
      row={'task':task,'training':training,'representation':rep,'scenario':sc['name'],'kind':sc['kind'],'value':sc['value'],'replicate':repno,'n_configurations':len(chosen),**metric}
      if task=='CO_coverage' and sc['kind']=='surface_fraction':row['MAE_vs_response_proxy_f_times_theta']=float(mean_absolute_error(y*sc['value'],prediction))
      results.append(row)
      np.savez_compressed(O/(task+'_'+training+'_'+rep+'_'+sc['name']+f'_rep{repno}.npz'),truth=y,prediction=prediction,pair_indices=chosen)
  write_json(O/(task+'_folds.json'),[{'fold':i,'train_groups':groups[chosen[tr]].tolist(),'test_groups':groups[chosen[te]].tolist()} for i,(tr,te) in enumerate(folds)])
 pd.DataFrame(results).to_csv(O/'metrics.csv',index=False);return len(results)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();print(run(a.study))
