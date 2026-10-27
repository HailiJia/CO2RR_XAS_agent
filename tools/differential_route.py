"""Shared audited training/inference for paired records, with strict references."""
from pathlib import Path
import json,shutil,hashlib
import numpy as np,joblib
from .differential_xas import pair,features,ReferenceError,experimental_pair
from .differential_training import train_study,banks,predict_bundle,cls_score
from .progress_snapshots import write_json


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_prepared(study):
 B=Path(study);z=np.load(B/'paired_arrays.npz');rows=sorted([r for r in json.loads((B/'pair_manifest.json').read_text()) if 'pair_index'in r],key=lambda r:r['pair_index']);checked=[]
 for i,r in enumerate(rows):
  sample=B/'scale'/r['corrected_record'];ref=B/'scale'/r['reference_record']
  if sha(sample)!=r['sample_corrected_record_sha256'] or sha(ref)!=r['reference_corrected_record_sha256']:raise ReferenceError('Paired source record hash mismatch')
  a=json.loads(sample.read_text());b=json.loads(ref.read_text());pop={int(k):v['population'] for k,v in r['populations'].items()};p=pair(a,b,z['energy'],pop,{4:36,5:36},[4,5],r['domain'],'CuAu')
  if not np.allclose(p['delta'],z['delta'][i],rtol=0,atol=1e-12):raise ReferenceError('Prepared differential arrays do not reconstruct from paired records')
  checked.append({'configuration':r['configuration'],'sample_sha256':sha(sample),'reference_sha256':sha(ref),'pair_reconstructed':True})
 return checked

def train(data_root,output_dir):
 source=Path(data_root).resolve();target=Path(output_dir).resolve()
 if target.exists() and any(target.iterdir()):raise ValueError('New empty output directory required; archived studies cannot be overwritten')
 checked=validate_prepared(source);target.mkdir(parents=True,exist_ok=True)
 for name in ['pair_manifest.json','paired_arrays.npz']:shutil.copyfile(source/name,target/name)
 shutil.copytree(source/'scale/corrected_records',target/'scale/corrected_records')
 if (source/'geometry_targets.json').exists():shutil.copyfile(source/'geometry_targets.json',target/'geometry_targets.json')
 result=train_study(target)
 if (target/'geometry_targets.json').exists():
  from .differential_geometry import run as geometry_run
  geometry_run(target)
 # Actual saved-model reloading through the same feature API.
 z=np.load(target/'paired_arrays.npz');bf=banks(z);receipt=[]
 for p in sorted((target/'ml/models').glob('*.joblib')):
  m=joblib.load(p);i=np.array(m['test_pair_indices']);sub={rep:{f:x[i] for f,x in bank.items()} for rep,bank in bf.items()};value=predict_bundle(m,sub);path=target/'ml'/f"{m['scope']}_{m['task']}_{m['representation']}_oof.npz";expected=np.load(path);lookup={int(v):j for j,v in enumerate(expected['pair_indices'])};take=[lookup[int(v)] for v in i];actual=value[0] if m['task']=='identity' else value
  if not np.array_equal(actual,expected['prediction'][take]):raise RuntimeError('Saved paired model reload mismatch')
  receipt.append({'model':p.name,'sha256':sha(p),'reload_prediction_identical':True})
 geometry_replay=[]
 if (target/'geometry/models').exists():
  for p in (target/'geometry/models').glob('*.joblib'):
   m=joblib.load(p);idx=np.asarray(m['test_pair_indices']);expected=np.load(target/'geometry'/f"{m['scope']}_{m['representation']}_oof.npz");lookup={int(v):i for i,v in enumerate(expected['pair_indices'])};actual=m['model'].predict(bf[m['representation']][m['recipe'][0]][idx]);assert np.array_equal(actual,expected['prediction'][[lookup[int(i)] for i in idx]])
   geometry_replay.append({'model':p.name,'sha256':sha(p),'reload_prediction_identical':True})
 trace={'geometry_models_reloaded':len(geometry_replay),'geometry_models':geometry_replay,'action':'audited_training','recipe':'differential_v1','checked_source_pairs':len(checked),'source_pair_manifest_sha256':sha(source/'pair_manifest.json'),'source_arrays_sha256':sha(source/'paired_arrays.npz'),'recipe_sha256':sha(target/'ml/protocol.json'),'models_reloaded':len(receipt),'models':receipt,'HPC_submissions':False,'portal_writes':False,'not_independent_confirmation':'refit on identical declared cohorts, for real route/reload verification only'};write_json(target/'invocation_trace.json',trace);write_json(target/'pair_validation.json',checked)
 return {'output_dir':str(target),'metrics':{'coverage':str(target/'ml/coverage_metrics.csv'),'identity':str(target/'ml/identity_metrics.csv')},'trace':str(target/'invocation_trace.json'),'prediction_replay_identical':True,'models_reloaded':len(receipt)}

def infer(model_path,sample,reference,populations,reference_populations,domain,reference_domain,experimental_convention=None,allow_ood_evaluation=False):
 m=joblib.load(model_path);energy=np.asarray(m['energy']);ood=[]
 if reference is None:raise ReferenceError('A compatible explicit reference is required; no mean-reference borrowing or silent raw fallback')
 if domain!='CuAu':
  if not allow_ood_evaluation:raise ReferenceError('No validated differential model for this facet/domain')
  ood.append('unvalidated facet/domain')
 if experimental_convention is not None:
  # Paired experimental arrays must already have an independently established
  # per-Cu intensity and common absolute-energy convention. Not raw detector counts.
  if experimental_convention.get('energy_convention')!='photon_eV_common_calibration':raise ReferenceError('Declared common photon-energy calibration required')
  grids=[np.asarray(experimental_convention.get(k,[]),float) for k in ['sample_energy_eV','reference_energy_eV']]
  if any(len(e)!=len(v) or len(e)<2 or not np.all(np.diff(e)>0) or e[0]>energy[0]+1e-8 or e[-1]<energy[-1]-1e-8 for e,v in zip(grids,[sample,reference])):raise ReferenceError('Experimental pair lacks explicit full trained energy support; no extrapolation')
  sample=np.interp(energy,grids[0],sample);reference=np.interp(energy,grids[1],reference)
  ex=experimental_pair(sample,reference,energy,experimental_convention);d=np.atleast_2d(ex['delta']/ex['surface_fraction']);c=np.atleast_2d(ex['clean_common_scale']);a=c+d;ood+=['simulation-to-experiment transfer unvalidated','surface fraction independently supplied; stable-background approximation']
  if a.shape[1]!=len(energy):raise ReferenceError('Experimental arrays lack trained energy support')
 else:
  p=pair(sample,reference,energy,populations,reference_populations,m['physical_top_layers'],domain,reference_domain);d=p['delta'][None,:];c=p['clean_common_scale'][None,:];a=p['raw_common_scale'][None,:]
  if reference['record_id'] not in m['reference_ids']:ood.append('unseen clean-parent/reference family')
 bank={rep:features(x,energy) for rep,x in [('raw_physical',a),('delta',d)]};bank['clean_plus_delta']={k:np.hstack([features(c,energy)[k],features(d,energy)[k]]) for k in ['Combined','DCT8','DCT16','DCT24']}
 if m['representation']=='raw_normalized':raise ReferenceError('The archived-method raw control requires original native aggregate records; corrected paired records are not a silent substitute')
 if m['representation']=='corrected_normalized':
  from .spectral_multilabel import feature_bank
  from scipy.fft import dct
  z=feature_bank(a,energy);bank[m['representation']]={'Combined':z['shape_CDF_derivatives'],**{f'DCT{k}':dct(z['normalized'],type=2,norm='ortho',axis=1)[:,:k] for k in [8,16,24]}}
 value=m['model'].predict(bank[m['representation']][m['recipe'][0]]) if m['task']=='geometry' else predict_bundle(m,bank);result={'model_sha256':sha(model_path),'representation':m['representation'],'reference_required':True,'domain':domain,'review_flags':ood,'experimental_accuracy_validated':False,'transformation':m['reference_transformation'],'reference_ids_trained':m['reference_ids'],'units':m['units']}
 if m['task']=='CO_coverage':
  result['predicted_CO_per_all_surface_metals']=float(value[0]);fraction=experimental_convention.get('surface_Cu_fraction') if experimental_convention is not None else 1. if domain=='Cu100' else m['surface_Cu_fraction']
  if fraction is not None and 0<fraction<=1:result['predicted_CO_per_surface_Cu_loading']=float(value[0]/fraction)
  result['target_caveat']='Cu-normalized loading is not literal Cu-site occupancy; mixture use not validated; experimental surface composition must be independently supplied'
 elif m['task']=='geometry':
  result['predicted_configuration_geometry']=dict(zip(m['target'],value[0].tolist()));result['geometry_caveat']='configuration means/fractions, not independent molecule predictions; absent Au-rich labels unvalidated; out-of-range regression values require review'
 else:
  pred,scores=value;species=list(m['target']);result.update(predicted_identity_set=[sp for sp,yes in zip(species,pred[0]) if yes],scores=dict(zip(species,scores[0].tolist())),uncertainty='uncalibrated classifier scores; no reliable probability claim',missing_positive_training_labels=[species[j] for j,est in enumerate(m['model']['estimators']) if isinstance(est,int) and est==0])
 return result
