from pathlib import Path
import numpy as np,pandas as pd
B=Path('study/relative_identity_v2');O=B/'results';rows=[]
def micro(y,p):
 tp=np.sum(y*p);return float(2*tp/max(1,np.sum(y)+np.sum(p)))
for stage in ['fixed0p5','transferred','common']:
 yy=[];pa=[];pb=[]
 for seed in [20261020,20261021,20261022]:
  a=np.load(O/f'size_old99_{seed}_{stage}.npz');b=np.load(O/f'size_expanded118_{seed}_{stage}.npz');lookup={int(i):j for j,i in enumerate(b['indices'])};ix=[lookup[int(i)] for i in a['indices']];yy.append(a['truth']);pa.append(a['prediction']);pb.append(b['prediction'][ix])
 differences=[micro(y,b)-micro(y,a) for y,a,b in zip(yy,pa,pb)];rng=np.random.default_rng(1330);bs=[]
 for _ in range(2000):
  ix=rng.integers(0,len(yy[0]),len(yy[0]));bs.append(np.mean([micro(y[ix],b[ix])-micro(y[ix],a[ix]) for y,a,b in zip(yy,pa,pb)]))
 rows.append(dict(threshold=stage,old99_mean=np.mean([micro(y,a) for y,a in zip(yy,pa)]),expanded_same99_mean=np.mean([micro(y,b) for y,b in zip(yy,pb)]),mean_difference=np.mean(differences),conditional_lower=np.quantile(bs,.025),conditional_upper=np.quantile(bs,.975),refit_seed_difference_SD=np.std(differences,ddof=1),uncertainty='paired configuration resampling shared across fixed seed predictions; refit/seed SD separate; not independent confirmation'))
pd.DataFrame(rows).to_csv(B/'paired_seed_mean_intervals.csv',index=False)
