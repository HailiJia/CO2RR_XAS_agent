"""CO-only trained models on unseen CO-containing mixtures; frozen ensemble diagnostic."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from sklearn.metrics import mean_absolute_error,r2_score
R=Path(__file__).resolve().parents[2];sys.path.insert(0,str(R))
from tools.differential_training import banks,predict_bundle
B=R/'study/differential_v1';z=np.load(B/'paired_arrays.npz');bf=banks(z);rows=sorted([r for r in json.load(open(B/'pair_manifest.json')) if 'pair_index'in r],key=lambda x:x['pair_index']);out=[];metrics=[]
for scope in ['matched','same_substrate_sensitivity']:
 ids=np.array([i for i,r in enumerate(rows) if 'CO'in r['labels'] and len(r['labels'])>1 and (scope!='matched' or r['eligible_primary'])]);truth=np.array([rows[i]['coverage_metal'] for i in ids])
 for rep in bf:
  predictions=[]
  for fold in range(5):
   m=joblib.load(B/'ml/models'/f'{scope}_CO_coverage_{rep}_fold{fold}.joblib');assert not set(m['training_groups'])&{rows[i]['group'] for i in ids};predictions.append(predict_bundle(m,{key:{f:a[ids] for f,a in bank.items()} for key,bank in bf.items()}))
  values=np.mean(predictions,axis=0);metrics.append({'scope':scope,'representation':rep,'n_mixture_configurations':len(ids),'MAE_CO_per_all_surface_metals':mean_absolute_error(truth,values),'R2':r2_score(truth,values) if np.ptp(truth)>0 else None,'protocol':'mean of five already saved CO-only fold models; mixtures absent from every fit/selection; interference diagnostic, not mixture-trained coverage validation'})
  for j,i in enumerate(ids):out.append({'scope':scope,'representation':rep,'configuration':rows[i]['configuration'],'labels':';'.join(rows[i]['labels']),'truth_CO_per_all_surface_metals':truth[j],'predicted_CO_per_all_surface_metals':values[j],'fold_model_predictions':json.dumps([float(p[j]) for p in predictions])})
pd.DataFrame(out).to_csv(B/'mixture_predictions.csv',index=False);pd.DataFrame(metrics).to_csv(B/'mixture_metrics.csv',index=False);print(pd.DataFrame(metrics).to_string(index=False))
