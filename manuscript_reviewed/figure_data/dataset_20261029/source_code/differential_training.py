"""Matched-cohort raw/amplitude/difference controls with nested grouped folds."""
from pathlib import Path
import json,hashlib,platform
import numpy as np,pandas as pd,joblib,sklearn
from scipy.fft import dct
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor,ExtraTreesClassifier
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,precision_recall_fscore_support
from .differential_xas import features,VERSION
from .spectral_multilabel import feature_bank,encode_labels,SPECIES,f1_metrics
from .updated_data_training import threshold_fit
from .progress_snapshots import write_json

SEED=20261027
REPS=('raw_normalized','corrected_normalized','raw_physical','delta','clean_plus_delta')
FEATURES=('Combined','DCT8','DCT16','DCT24')
REG=tuple((f,'ridge',a) for f in FEATURES for a in [1.,10.,100.])+tuple((f,'trees',2) for f in ['Combined','DCT16'])
CLS=tuple((f,'logistic',a) for f in FEATURES for a in [.1,1.])+tuple((f,'trees',2) for f in ['Combined','DCT16'])

def banks(z):
 e=z['energy'];norm=feature_bank(z['raw_native'],e);f={'Combined':norm['shape_CDF_derivatives'],**{f'DCT{k}':dct(norm['normalized'],type=2,norm='ortho',axis=1)[:,:k] for k in [8,16,24]}};corrected_norm=feature_bank(z['raw_physical'],e);cn={'Combined':corrected_norm['shape_CDF_derivatives'],**{f'DCT{k}':dct(corrected_norm['normalized'],type=2,norm='ortho',axis=1)[:,:k] for k in [8,16,24]}};p=features(z['raw_physical'],e);delta=features(z['delta'],e);clean=features(z['clean'],e)
 return {'raw_normalized':f,'corrected_normalized':cn,'raw_physical':p,'delta':delta,'clean_plus_delta':{k:np.hstack([clean[k],delta[k]]) for k in FEATURES}}

class SupportedColumns(BaseEstimator,TransformerMixin):
 def fit(self,X,y=None):
  X=np.asarray(X);self.mask_=np.ptp(X,axis=0)>1e-12*np.maximum(1,np.max(abs(X),axis=0))
  if not self.mask_.any():raise ValueError('No varying training features')
  return self
 def transform(self,X):return np.asarray(X)[:,self.mask_]

def reg_estimator(recipe):
 f,kind,a=recipe
 return make_pipeline(SupportedColumns(),StandardScaler(),Ridge(alpha=a)) if kind=='ridge' else make_pipeline(SupportedColumns(),StandardScaler(),ExtraTreesRegressor(n_estimators=64,min_samples_leaf=int(a),max_features=1.,random_state=SEED,n_jobs=1))
def cls_fit(X,Y,recipe):
 f,kind,a=recipe;columns=SupportedColumns().fit(X);XX=columns.transform(X);scale=StandardScaler().fit(XX);v=scale.transform(XX);models=[]
 for j in range(Y.shape[1]):
  y=Y[:,j]
  if len(np.unique(y))<2:models.append(int(y[0]));continue
  model=LogisticRegression(C=a,class_weight='balanced',solver='liblinear',max_iter=2000,random_state=SEED) if kind=='logistic' else ExtraTreesClassifier(n_estimators=64,min_samples_leaf=int(a),class_weight='balanced',random_state=SEED,n_jobs=1)
  models.append(model.fit(v,y))
 return {'columns':columns,'scaler':scale,'estimators':models}
def cls_score(model,X):
 v=model['scaler'].transform(model['columns'].transform(X))
 return np.column_stack([np.full(len(v),m) if isinstance(m,int) else m.predict_proba(v)[:,list(m.classes_).index(1)] for m in model['estimators']])
def regression_metrics(y,p):return {'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None}

def train_study(study):
 B=Path(study);out=B/'ml';out.mkdir(exist_ok=True);z=np.load(B/'paired_arrays.npz');manifest=json.loads((B/'pair_manifest.json').read_text());paired=sorted([r for r in manifest if 'pair_index'in r],key=lambda r:r['pair_index']);group=np.array([r['group'] for r in paired]);bf=banks(z);coverage=[];identities=[];splitslog=[];ranklog=[];models=[];full_predictions=[];rng=np.random.default_rng(SEED)
 protocol={'version':VERSION,'seed':SEED,'representations':REPS,'features':FEATURES,'regression_candidates':REG,'identity_candidates':CLS,'outer':'5-fold GroupKFold, same configuration assignments for all representations','inner':'3-fold GroupKFold confined to each outer training partition','pair_population':'one reference/model family in CuAu; shared-reference conditional inference, not unseen-parent validation','primary':'initial-metal model/constraints matched subset','secondary':'same-substrate reference sensitivity, initial-parent lineage unproven','Cu100_joint_differential':'not available: no compatible completed clean Cu100 reference located','coverage_targets':['CO/all_top_surface_metals','CO/surface_Cu_loading','CO_per_nm2'],'negative_delta_retained':True,'per_spectrum_delta_normalization':False,'training_only_constant_columns':'drop training peak-to-peak <=1e-12 relative to max(1,absolute feature magnitude); fixed reference blocks cannot act as random-feature padding','tunable_nuisance_parameters':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('differential_xas.py')]},'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'confirmation':'exploratory reused configurations; not prospective or experimental validation'};write_json(out/'protocol.json',protocol)
 for scope,base in [('matched',np.array([i for i,r in enumerate(paired) if r['eligible_primary']],int)),('same_substrate_sensitivity',np.arange(len(paired)))]:
  for task,ids in [('identity',base),('CO_coverage',np.array([i for i in base if paired[i]['CO_only']],int))]:
   outer=list(GroupKFold(min(5,len(set(group[ids])))).split(ids,groups=group[ids]));truth=encode_labels([paired[i]['labels'] for i in ids]) if task=='identity' else np.array([paired[i]['coverage_metal'] for i in ids]);store={rep:{'prediction':np.zeros_like(truth,float),'score':np.zeros_like(truth,float)} for rep in REPS};baseline=np.zeros_like(truth,float)
   for fold,(tr0,te0) in enumerate(outer):
    tr=ids[tr0];te=ids[te0];inner=list(GroupKFold(min(3,len(set(group[tr])))).split(tr,groups=group[tr]));splitslog.append({'scope':scope,'task':task,'fold':fold,'train_groups':group[tr].tolist(),'test_groups':group[te].tolist(),'training_species_support':truth[tr0].sum(axis=0).tolist() if task=='identity' else None,'test_species_support':truth[te0].sum(axis=0).tolist() if task=='identity' else None,'inner_test_groups':[group[tr[v]].tolist() for a,v in inner]})
    baseline[te0]=truth[tr0].mean(axis=0) if task=='CO_coverage' else (truth[tr0].mean(axis=0)>=.5).astype(int)
    for rep in REPS:
     candidates=CLS if task=='identity' else REG;ranks=[];inners=[];thresholds=[]
     for recipe in candidates:
      X=bf[rep][recipe[0]];ip=np.zeros_like(truth[tr0],float)
      for a,v in inner:
       if task=='CO_coverage':ip[v]=reg_estimator(recipe).fit(X[tr[a]],truth[tr0[a]]).predict(X[tr[v]])
       else:ip[v]=cls_score(cls_fit(X[tr[a]],truth[tr0[a]],recipe),X[tr[v]])
      if task=='CO_coverage':quality=-mean_absolute_error(truth[tr0],ip);th=None
      else:quality,th,_=threshold_fit(truth[tr0],ip)
      ranks.append(float(quality));inners.append(ip);thresholds.append(th)
     win=int(np.argmax(ranks));recipe=candidates[win];X=bf[rep][recipe[0]]
     if task=='CO_coverage':model=reg_estimator(recipe).fit(X[tr],truth[tr0]);score=model.predict(X[te]);pred=score
     else:model=cls_fit(X[tr],truth[tr0],recipe);score=cls_score(model,X[te]);pred=(score>=thresholds[win]).astype(int)
     store[rep]['score'][te0]=score;store[rep]['prediction'][te0]=pred;name=f'{scope}_{task}_{rep}_fold{fold}';path=out/'models'/f'{name}.joblib';path.parent.mkdir(exist_ok=True);bundle={'model':model,'recipe':recipe,'thresholds':thresholds[win],'task':task,'representation':rep,'energy':z['energy'],'target':'CO/all_top_surface_metals' if task=='CO_coverage' else SPECIES,'reference_ids':sorted(set(paired[i]['reference_id'] for i in tr)),'reference_transformation':'equal-physical-Cu sources; common reference edge-step; signed linear features; training-only scaling','training_groups':group[tr].tolist(),'test_groups':group[te].tolist(),'test_pair_indices':te.tolist(),'scope':scope,'reference_required':rep in ['delta','clean_plus_delta'],'reference_scale':paired[int(tr[0])]['pair_scale'],'reference_hashes':sorted(set(paired[i]['reference_corrected_record_sha256'] for i in tr)),'units':{'energy':'eV','native_absorption':'Mbarn_per_Cu','model_intensity':'clean-edge-step units','cumulative_difference':'clean-edge-step eV','d1':'clean-edge-step/eV','d2':'clean-edge-step/eV^2'},'scale_windows_eV':{'pre':[8974.,8977.],'post':[9018.,9025.]},'physical_top_layers':[4,5],'surface_Cu_fraction':36/68,'surface_sites':68,'domain':'CuAu; no verified Cu100 differential training'};joblib.dump(bundle,path);np.savez_compressed(out/f'{name}_inner.npz',scores=np.array(inners),training_pair_indices=tr);ranklog.append({'scope':scope,'task':task,'rep':rep,'fold':fold,'candidates':candidates,'inner_quality':ranks,'selected':recipe,'thresholds':thresholds[win]});models.append({'path':str(path.relative_to(B)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
   for rep,values in store.items():
    pred=values['prediction'];np.savez_compressed(out/f'{scope}_{task}_{rep}_oof.npz',truth=truth,prediction=pred,score=values['score'],pair_indices=ids,configuration=np.array([paired[i]['configuration'] for i in ids]));row={'scope':scope,'task':task,'representation':rep,'n_configurations':len(ids)}
    if task=='CO_coverage':
     for unit,key in [('CO/all_top_surface_metals','coverage_metal'),('CO/surface_Cu_loading','coverage_Cu_loading'),('CO_per_nm2','CO_per_nm2')]:
      y=np.array([paired[i][key] for i in ids]);factor=y/truth;coverage.append({**row,'target':unit,**regression_metrics(y,pred*factor)})
     for i,y,pv in zip(ids,truth,pred):full_predictions.append({'scope':scope,'task':task,'representation':rep,'configuration':paired[i]['configuration'],'truth_metal':y,'prediction_metal':pv,'truth_Cu_loading':paired[i]['coverage_Cu_loading'],'truth_CO_per_nm2':paired[i]['CO_per_nm2']})
    else:
     pp,rr,ff,ss=precision_recall_fscore_support(truth,pred,zero_division=0);active=truth.sum(axis=0)>0;identities.append({**row,**f1_metrics(truth,pred),'macro_F1_positive_support_labels':float(ff[active].mean()),'positive_support_labels':','.join(np.array(SPECIES)[active])});write_json(out/f'{scope}_{rep}_per_species.json',[{'species':sp,'precision':float(p),'recall':float(r),'F1':float(f),'support':int(s)} for sp,p,r,f,s in zip(SPECIES,pp,rr,ff,ss)])
   if task=='CO_coverage':coverage.append({'scope':scope,'task':task,'representation':'training_mean','n_configurations':len(ids),'target':'CO/all_top_surface_metals',**regression_metrics(truth,baseline)})
   else:identities.append({'scope':scope,'task':task,'representation':'training_label_majority','n_configurations':len(ids),**f1_metrics(truth,baseline)})
 write_json(out/'splits.json',splitslog);write_json(out/'selection_ranks.json',ranklog);write_json(out/'saved_models.json',models);pd.DataFrame(coverage).to_csv(out/'coverage_metrics.csv',index=False);pd.DataFrame(identities).to_csv(out/'identity_metrics.csv',index=False);pd.DataFrame(full_predictions).to_csv(out/'coverage_predictions.csv',index=False)
 # A single fixed clean reference makes signed linear features an affine shift.
 # StandardScaler/Ridge should therefore give the same prediction on raw physical
 # and difference inputs for the SAME fixed recipe, independent of any selector.
 invariant=[]
 for scope,ids in [('matched',np.array([i for i,r in enumerate(paired) if r['eligible_primary'] and r['CO_only']],int)),('same_substrate_sensitivity',np.array([i for i,r in enumerate(paired) if r['CO_only']],int))]:
  for fold,(a,v) in enumerate(GroupKFold(min(5,len(ids))).split(ids,groups=group[ids])):
   rec=('DCT16','ridge',10.);p1=reg_estimator(rec).fit(bf['raw_physical']['DCT16'][ids[a]],np.array([paired[i]['coverage_metal'] for i in ids[a]])).predict(bf['raw_physical']['DCT16'][ids[v]]);p2=reg_estimator(rec).fit(bf['delta']['DCT16'][ids[a]],np.array([paired[i]['coverage_metal'] for i in ids[a]])).predict(bf['delta']['DCT16'][ids[v]]);invariant.append({'scope':scope,'fold':fold,'max_abs_prediction_difference':float(max(abs(p1-p2))),'fixed_recipe':rec})
 write_json(out/'affine_equivalence_control.json',invariant)
 return {'saved_models':len(models),'coverage_metrics':coverage,'identity_metrics':identities}


def predict_bundle(bundle,bank):
 X=bank[bundle['representation']][bundle['recipe'][0]]
 if bundle['task']=='CO_coverage':return bundle['model'].predict(X)
 s=cls_score(bundle['model'],X);return s>=bundle['thresholds'],s

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();print(json.dumps(train_study(a.study),indent=2))

