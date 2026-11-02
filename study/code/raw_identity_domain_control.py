"""Raw common-label augmentation control; Cu100 differential remains unavailable."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from scipy.fft import dct
from sklearn.model_selection import GroupKFold,LeaveOneOut
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spectral_multilabel import feature_bank,f1_metrics
from tools.differential_training import cls_fit,cls_score
from tools.updated_data_training import threshold_fit
from tools.progress_snapshots import write_json
B=ROOT/'study/differential_v1/raw_identity_augmentation';B.mkdir(exist_ok=True);old=ROOT/'study/coverage_20261026';rows=json.loads((old/'model_configuration_manifest.json').read_text());z=np.load(old/'model_inputs.npz');X=dct(feature_bank(z['raw'],z['energy'])['normalized'],type=2,norm='ortho',axis=1)[:,:16];labels=['CO','H','O','OH'];Y=np.array([[int(s in r['identity_labels']) for s in labels] for r in rows]);a=np.array([i for i,r in enumerate(rows) if r['domain']=='CuAu']);b=np.array([i for i,r in enumerate(rows) if r['domain']=='Cu100' and set(r['identity_labels'])<=set(['CHO','CO','CO2','COCHO','COH','COOH','H','O','OCCOH','OCHO','OH'])]);groups=np.array([r['group'] for r in rows]);predictions=[];splits=[];metrics=[]
for domain,ids,folds in [('CuAu',a,list(GroupKFold(5).split(a,groups=groups[a]))),('Cu100',b,list(LeaveOneOut().split(b)))]:
 for training in ['CuAu_only','CuAu_plus_Cu100']:
  scores=np.zeros_like(Y[ids],float);pred=np.zeros_like(Y[ids])
  for fold,(tr0,te0) in enumerate(folds):
   te=ids[te0];base=a[tr0] if domain=='CuAu' else a;tr=base if training=='CuAu_only' else np.r_[base,b if domain=='CuAu' else b[tr0]];inside=np.zeros_like(Y[tr],float)
   for u,v in GroupKFold(3).split(tr,groups=groups[tr]):inside[v]=cls_score(cls_fit(X[tr[u]],Y[tr[u]],('DCT16','logistic',1.)),X[tr[v]])
   _,thresholds,_=threshold_fit(Y[tr],inside);model=cls_fit(X[tr],Y[tr],('DCT16','logistic',1.));scores[te0]=cls_score(model,X[te]);pred[te0]=(scores[te0]>=thresholds).astype(int);name=f'{domain}_{training}_fold{fold}';joblib.dump({'model':model,'thresholds':thresholds,'train_indices':tr,'test_indices':te,'labels':labels,'feature':'normalized raw DCT16'},B/(name+'.joblib'));splits.append({'name':name,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'training_positive_support':Y[tr].sum(axis=0).tolist(),'test_positive_support':Y[te].sum(axis=0).tolist()});assert not set(groups[tr])&set(groups[te])
  result={'domain':domain,'training':training,'n':len(ids),'labels':';'.join(labels),**f1_metrics(Y[ids],pred)};metrics.append(result);np.savez_compressed(B/f'{domain}_{training}_oof.npz',truth=Y[ids],prediction=pred,scores=scores,configuration=np.array([rows[i]['configuration'] for i in ids]),labels=np.array(labels))
  for i,y,p,s in zip(ids,Y[ids],pred,scores):predictions.append({'domain':domain,'training':training,'configuration':rows[i]['configuration'],'truth':y.tolist(),'prediction':p.tolist(),'scores':s.tolist()})
pd.DataFrame(metrics).to_csv(B/'metrics.csv',index=False);write_json(B/'predictions.json',predictions);write_json(B/'protocol.json',{'status':'fixed raw-input common-label control, not differential or experimental validation','feature':'DCT16 of native minimum-subtracted unit-area raw XAS','estimator':'independent balanced Logistic C1, training-only constant-column removal and scaling','thresholds':'three inner grouped folds, micro/shrunk training-only decoder','labels':labels,'unsupported_H2O_CO3':'five configurations excluded from closed common-label fitting','CuAu_protocol':'configuration 5-fold, all Cu100 available for augmented outer training','Cu100_protocol':'leave one Cu100 configuration out; remaining 11 may augment CuAu','domain_balanced_aggregate':'mean of separately evaluated domain metrics, not an unseen-facet score','splits':splits});print(pd.DataFrame(metrics).to_string(index=False))
