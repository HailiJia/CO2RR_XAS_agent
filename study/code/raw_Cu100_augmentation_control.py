"""Fixed raw-input pilot only; not differential or independent facet validation."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
from scipy.fft import dct
from sklearn.model_selection import GroupKFold,LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spectral_multilabel import feature_bank
B=ROOT/'study/differential_v1/raw_augmentation';B.mkdir(exist_ok=True);old=ROOT/'study/coverage_20261026';r=json.loads((old/'model_configuration_manifest.json').read_text());z=np.load(old/'model_inputs.npz');X=dct(feature_bank(z['raw'],z['energy'])['normalized'],type=2,norm='ortho',axis=1)[:,:16];a=np.array([i for i,x in enumerate(r) if x['domain']=='CuAu' and x['CO_only']]);b=np.array([i for i,x in enumerate(r) if x['domain']=='Cu100' and x['CO_only']]);y=np.array([x['coverage_per_surface_Cu'] for x in r]);groups=np.array([x['group'] for x in r]);rows=[];splits=[]
def model():return make_pipeline(StandardScaler(),Ridge(alpha=10))
for domain,ids,folds in [('CuAu',a,list(GroupKFold(5).split(a,groups=groups[a]))),('Cu100',b,list(LeaveOneOut().split(b)))]:
 for fold,(train,test) in enumerate(folds):
  te=ids[test];base=a[train] if domain=='CuAu' else a
  for mode in ['CuAu_only','CuAu_plus_Cu100']:
   tr=base if mode=='CuAu_only' else np.r_[base,b if domain=='CuAu' else b[train]];m=model().fit(X[tr],y[tr]);p=m.predict(X[te]);assert not set(groups[tr])&set(groups[te]);name=f'{domain}_{mode}_{fold}';joblib.dump({'model':m,'train_indices':tr,'test_indices':te,'feature':'DCT16 of archived minimum-subtracted unit-area raw spectrum','target':'CO/surface_Cu_loading'},B/(name+'.joblib'));splits.append({'name':name,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist()})
   for i,v in zip(te,p):rows.append({'domain':domain,'training':mode,'fold':fold,'configuration':r[i]['configuration'],'truth_Cu_loading':y[i],'prediction_Cu_loading':float(v)})
pd.DataFrame(rows).to_csv(B/'heldout.csv',index=False);result=[]
for (domain,mode),d in pd.DataFrame(rows).groupby(['domain','training']):result.append({'domain':domain,'training':mode,'n':len(d),'MAE_Cu_loading':mean_absolute_error(d.truth_Cu_loading,d.prediction_Cu_loading)})
pd.DataFrame(result).to_csv(B/'metrics.csv',index=False);(B/'protocol.json').write_text(json.dumps({'status':'post-inspection fixed raw-input control; not differential or experimental validation','recipe':'DCT16; training StandardScaler; Ridge alpha10','target':'N(CO)/surface Cu: loading, not literal occupancy','splits':splits,'reference_prompt_values':{'Cu100_CuAu_only':.082124,'Cu100_augmented':.035186,'CuAu_only':.027756,'CuAu_augmented':.026766}},indent=2));print(pd.DataFrame(result).to_string(index=False))
