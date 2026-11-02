"""Decision-threshold distances and retained-query denominators; no calibration claim."""
from pathlib import Path
import numpy as np,pandas as pd,joblib
B=Path('study/relative_identity_guarded_v2');O=B/'results';rows=[];z=np.load(O/'inputs.npz')
for cohort in ['expanded118','joint130']:
 for rep in ['native_normalized','corrected_normalized','common_raw','relative_delta','signed_shape']:
  for family in ['logistic','trees']:
   name=f'nested_{cohort}_{rep}_{family}';q=np.load(O/(name+'.npz'));ids=q['indices'];th=np.zeros_like(q['score']);eligible=np.ones(len(ids),bool)
   for f in range(5):
    mask=q['fold']==f;m=joblib.load(O/'models'/f'{name}_{f}.joblib');th[mask]=m['thresholds'];eligible[mask]=np.sqrt(np.mean(z['delta'][ids[mask]]**2,axis=1))>m['floor']
   distance=np.min(abs(q['score']-th),axis=1)
   for margin in [0.,.05,.1,.2]:
    keep=eligible&(distance>=margin);truth=q['truth'];pred=q['prediction'];rows.append({'analysis':name,'minimum_distance_to_actual_training_threshold':margin,'n_queries':len(ids),'n_retained':int(keep.sum()),'retained_fraction':float(keep.mean()),'full_query_exact_error':float(np.any(truth!=pred,axis=1).mean()),'retained_exact_error':float(np.any(truth[keep]!=pred[keep],axis=1).mean()) if keep.any() else None,'false_positive_labels_all_queries':int(((truth==0)&(pred==1)).sum()),'false_positive_labels_retained':int(((truth[keep]==0)&(pred[keep]==1)).sum()),'calibration':'uncalibrated score-distance diagnostic; thresholds from inner training; margin grid not selected on outer queries','low_signal':'reference-pair signal floor uses training only; no measured detection-limit claim'})
pd.DataFrame(rows).to_csv(B/'review_policy_by_decision_threshold.csv',index=False);print('Saved80 fixed review policies with retained-query denominators')
