from pathlib import Path
import json
import numpy as np,pandas as pd,shutil
B=Path('study/relative_identity_guarded_v2');O=B/'group_learning';r=json.load(open(B/'results/source_manifest.json'));m=json.load(open(B/'near_duplicate_family_map.json'))['families'];fam=np.array([m[a['configuration']] for a in r]);rows=[]
for p in O.glob('fraction*.npz'):
 q=np.load(p);ids=q['indices'];y=q['truth'];pp=q['prediction'];w=np.array([1/sum(fam[ids]==f) for f in fam[ids]])[:,None];tp=np.sum(y*pp*w);f1=2*tp/np.sum((y+pp)*w);a=p.stem.split('_');rows.append({'fraction':float(a[0].removeprefix('fraction')),'seed':int(a[1]),'micro_f1':f1,'n_records':len(ids),'n_groups':len(set(fam[ids]))})
d=pd.DataFrame(rows).sort_values(['fraction','seed']);d.to_csv(O/'near_group_metrics.csv',index=False);print(d.groupby('fraction').micro_f1.agg(['mean','std']))
M=Path('study/manuscript_relative_identity_guarded_v2');D=M/'figure_data/controlled_identity';shutil.copy2(O/'near_group_metrics.csv',D/'group_learning_metrics.csv');shutil.copy2(B/'near_group_metrics.csv',D/'near_group_metrics.csv')
