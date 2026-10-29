"""Frozen model review outputs on audited unsupported chemistry, never training examples."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd,joblib
from scipy.fft import dct
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.differential_xas import pair,features
from tools.spectral_multilabel import feature_bank,SPECIES
from tools.differential_training import cls_score
from tools.coverage_study import dump
B=ROOT/'study/dataset_20261029';m=json.load(open(B/'scale/configuration_manifest.json'));source=json.load(open(B/'configuration_manifest.json'));z=np.load(B/'spectra.npz');raw={r['configuration']:z[f'raw_{i}'] for i,r in enumerate(source)};E=np.load(B/'training_arrays.npz')['energy'];refs={dom:next(r for r in m if r['domain']==dom and not r['identity_labels']) for dom in ['CuAu','Cu100']};output=[]
for r in m:
 unknown=sorted(set(r['identity_labels'])-set(SPECIES))
 if not unknown:continue
 ref=refs[r['domain']];layers=[4,5] if r['domain']=='CuAu' else [3,4];pop={int(k):v['population'] for k,v in r['layer_evidence'].items()};rp={int(k):v['population'] for k,v in ref['layer_evidence'].items()};p=pair(json.load(open(B/'scale'/r['corrected_record'])),json.load(open(B/'scale'/ref['corrected_record'])),E,pop,rp,layers,r['domain'],r['domain']);n=feature_bank(raw[r['configuration']],E);banks={'normalized_raw':{'Combined':n['shape_CDF_derivatives'],'DCT16':dct(n['normalized'],type=2,norm='ortho',axis=1)[:,:16]},'physical_raw':features(p['raw_common_scale'],E),'delta':features(p['delta'],E)}
 for rep in banks:
  scores=[];pred=[]
  for model in sorted((B/'ml/models').glob(f'nested_refresh_joint_{rep}_fold*.joblib')):
   b=joblib.load(model);assert r['group'] not in b['train_groups'];s=cls_score(b['model'],banks[rep][b['recipe'][0]])[0];scores.append(s);pred.append(s>=b['thresholds'])
  vote=np.mean(pred,axis=0);output.append({'configuration':r['configuration'],'domain':r['domain'],'representation':rep,'unsupported_species':unknown,'review_required':True,'closed_label_prediction':[s for s,v in zip(SPECIES,vote) if v>=.5],'mean_uncalibrated_scores':np.mean(scores,axis=0).tolist(),'fold_votes':vote.tolist(),'actual_full_graph_labels':r['identity_labels'],'interpretation':'unsupported chemistry/interference query; no full-set accuracy or probability calibration claim'})
dump(B/'unsupported_queries.json',output);pd.DataFrame([{**r,'unsupported_species':';'.join(r['unsupported_species']),'closed_label_prediction':';'.join(r['closed_label_prediction']),'actual_full_graph_labels':';'.join(r['actual_full_graph_labels'])} for r in output]).drop(columns=['mean_uncalibrated_scores','fold_votes']).to_csv(B/'unsupported_queries.csv',index=False);print('review queries',len(output),'physical configurations',len({r['configuration'] for r in output}))
