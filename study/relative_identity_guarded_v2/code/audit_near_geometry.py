from pathlib import Path
import json,itertools,collections
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
B=Path('study/dataset_20261029');rows=json.load(open(B/'training_manifest.json'));geoms={a['configuration']:a for a in json.load(open(B/'geometry_arrays.json'))};out=[]
for i,a in enumerate(rows):
 aa=geoms[a['configuration']];sa=np.array(aa['atoms']);pa=np.array(aa['positions']);cell=np.array(aa['cell']);inv=np.linalg.inv(cell)
 for b in rows[:i]:
  bb=geoms[b['configuration']];sb=np.array(bb['atoms'])
  if a['domain']!=b['domain'] or collections.Counter(sa)!=collections.Counter(sb) or a['observed_counts']!=b['observed_counts'] or not np.allclose(cell,bb['cell'],atol=2e-3,rtol=0):continue
  pb=np.array(bb['positions']);dist=[]
  for el in set(sa):
   xx=pa[sa==el];yy=pb[sb==el];df=(xx[:,None,:]-yy[None,:,:])@inv;df[:,:,:2]-=np.rint(df[:,:,:2]);ds=np.full(df.shape[:2],np.inf)
   for u,v in itertools.product([-1,0,1],repeat=2):
    shift=df.copy();shift[:,:,0]+=u;shift[:,:,1]+=v;ds=np.minimum(ds,np.linalg.norm(shift@cell,axis=2))
   m,n=linear_sum_assignment(ds**2);dist.extend(ds[m,n])
  rms=float(np.sqrt(np.mean(np.square(dist))));mx=float(max(dist));out.append({'configuration_a':a['configuration'],'configuration_b':b['configuration'],'RMS_A':rms,'max_A':mx,'cross_outer_fold':a['fold']!=b['fold'],'near_primary':rms<=.02 and mx<=.2,'near_strict':rms<=.02 and mx<=.05,'near_sensitivity':rms<=.05 and mx<=.1})
d=pd.DataFrame(out).sort_values('RMS_A');d.to_csv('study/relative_identity_guarded_v2/near_geometry_pairs.csv',index=False);print('Compared',len(d),'same-composition/connectivity pairs');print(d.head(8).to_string(index=False));print(d.groupby(['near_strict','near_sensitivity','cross_outer_fold']).size())
