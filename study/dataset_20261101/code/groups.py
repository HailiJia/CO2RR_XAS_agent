"""Conservative geometry-only grouping; inherit prior memberships and query folds."""
from pathlib import Path
import json,itertools,collections
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
from tools.coverage_study import dump,sha
B=Path(__file__).resolve().parents[1];R=B.parents[1];rows=json.load(open(B/'training_manifest.json'));geoms={a['configuration']:a for a in json.load(open(B/'geometry_arrays.json'))};old=json.load(open(R/'study/dataset_20261029/training_manifest.json'));mapping=json.load(open(R/'study/relative_identity_guarded_v2/near_duplicate_family_map.json'))['families'];parent=list(range(len(rows)));pairs=[]
def find(i):
 while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
 return i
def join(a,b):parent[find(a)]=find(b)
for i,a in enumerate(rows):
 aa=geoms[a['configuration']];sa=np.array(aa['atoms']);pa=np.array(aa['positions']);cell=np.array(aa['cell']);inv=np.linalg.inv(cell)
 for j,b in enumerate(rows[:i]):
  if a['old_index'] is not None and b['old_index'] is not None and mapping[old[a['old_index']]['configuration']]==mapping[old[b['old_index']]['configuration']]:join(i,j)
  bb=geoms[b['configuration']];sb=np.array(bb['atoms'])
  if a['domain']!=b['domain'] or collections.Counter(sa)!=collections.Counter(sb) or a['observed_counts']!=b['observed_counts'] or not np.allclose(cell,bb['cell'],atol=2e-3,rtol=0):continue
  pb=np.array(bb['positions']);dist=[]
  for el in set(sa):
   df=(pa[sa==el,None,:]-pb[sb==el][None,:,:])@inv;df[:,:,:2]-=np.rint(df[:,:,:2]);ds=np.full(df.shape[:2],np.inf)
   for u,v in itertools.product([-1,0,1],repeat=2):
    shift=df.copy();shift[:,:,0]+=u;shift[:,:,1]+=v;ds=np.minimum(ds,np.linalg.norm(shift@cell,axis=2))
   m,n=linear_sum_assignment(ds**2);dist.extend(ds[m,n])
  rms=float(np.sqrt(np.mean(np.square(dist))));mx=float(max(dist));near=rms<=.02 and mx<=.2
  if near:join(i,j)
  pairs.append({'a':a['configuration'],'b':b['configuration'],'RMS_A':rms,'max_A':mx,'near':near})
fam={}
for i,a in enumerate(rows):
 cluster=[j for j in range(len(rows)) if find(i)==find(j)];prior=[mapping[old[rows[j]['old_index']]['configuration']] for j in cluster if rows[j]['old_index'] is not None];fam[a['configuration']]=min(prior) if prior else min(rows[j]['configuration'] for j in cluster)
pd.DataFrame(pairs).to_csv(B/'near_geometry_pairs.csv',index=False);dump(B/'family_map.json',{'families':fam,'n_records':len(rows),'n_groups':len(set(fam.values())),'criteria':'same composition/connectivity, element-aware in-plane MIC; RMS<=.02A and max<=.2A; retain inherited prior clusters','independent_parent_families':False,'training_manifest_sha256':sha(B/'training_manifest.json')});print('GROUPS',len(rows),len(set(fam.values())))
