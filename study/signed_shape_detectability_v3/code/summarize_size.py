"""Equal-near-group and record metrics, paired conditional bootstrap and seed SD."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tools.spectral_multilabel import SPECIES
B=Path('study/signed_shape_detectability_v3');O=B/'ml';r=json.load(open(O/'source_manifest.json'));mapping=json.load(open(O/'family_map.json'))['families'];fam=np.array([mapping[a['configuration']] for a in r]);rows=[];per=[];interval=[]
def contribution(ids,y,p):
 out=[];ex=[]
 for f in sorted(set(fam[ids])):
  mask=fam[ids]==f;yy=y[mask];pp=p[mask];out.append(np.stack([(yy*pp).mean(axis=0),yy.mean(axis=0),pp.mean(axis=0)]));ex.append(np.all(yy==pp,axis=1).mean())
 return np.array(out),np.array(ex)
def metrics(c,exact):
 tp=c[:,0].sum(axis=0);den=c[:,1:].sum(axis=(0,1));return np.array([2*tp.sum()/max(1e-20,den.sum()),np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0).mean(),exact.mean()])
for p in O.glob('*.npz'):
 q=np.load(p)
 if not {'indices','prediction','truth'}<=set(q.files):continue
 ids=q['indices'];y=q['truth'];pp=q['prediction'];parts=p.stem.split('_');meta={'analysis':p.stem,'kind':parts[1],'seed':int(parts[3]),'cohort':parts[2]} if parts[0]=='size' else {'analysis':p.stem,'kind':parts[1],'seed':int(parts[3]),'fraction':float(parts[2])}
 for weighting,w in [('record',np.ones(len(ids))),('near_group',np.array([1/sum(fam[ids]==f) for f in fam[ids]]))]:
  pr,re,ff,su=precision_recall_fscore_support(y,pp,sample_weight=w,average=None,zero_division=0);tp=np.sum(y*pp*w[:,None]);rows.append({**meta,'weighting':weighting,'n_records':len(ids),'n_groups':len(set(fam[ids])),'micro_f1':2*tp/max(1e-20,np.sum((y+pp)*w[:,None])),'macro_f1':ff.mean(),'exact_set_accuracy':np.average(np.all(y==pp,axis=1),weights=w)})
  for j,sp in enumerate(SPECIES):per.append({**meta,'weighting':weighting,'species':sp,'precision':pr[j],'recall':re[j],'f1':ff[j],'positive_support':su[j]})
for kind in ['trees','logistic']:
 for stage in ['fixed0p5','transferred','common']:
  ca=[];cb=[];ea=[];eb=[];differences=[]
  for seed in [20261020,20261021,20261022]:
   a=np.load(O/f'size_{kind}_old99_{seed}_{stage}.npz');b=np.load(O/f'size_{kind}_expanded118_{seed}_{stage}.npz');assert np.array_equal(a['indices'],b['indices']);x,xe=contribution(a['indices'],a['truth'],a['prediction']);xx,xxe=contribution(b['indices'],b['truth'],b['prediction']);ca.append(x);cb.append(xx);ea.append(xe);eb.append(xxe);delta=metrics(xx,xxe)-metrics(x,xe);differences.append(delta);rng=np.random.default_rng(1330);bs=[]
   for _ in range(2000):
    ix=rng.integers(0,len(x),len(x));bs.append(metrics(xx[ix],xxe[ix])-metrics(x[ix],xe[ix]))
   for j,name in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):interval.append({'kind':kind,'threshold':stage,'seed':seed,'metric':name,'old':metrics(x,xe)[j],'expanded_same_queries':metrics(xx,xxe)[j],'difference':delta[j],'lower':np.quantile(bs,.025,axis=0)[j],'upper':np.quantile(bs,.975,axis=0)[j],'scope':'fixed-prediction near-group bootstrap; no refit or new-parent uncertainty'})
  rng=np.random.default_rng(1730);bs=[]
  for _ in range(2000):
   ix=rng.integers(0,len(ca[0]),len(ca[0]));bs.append(np.mean([metrics(b[ix],be[ix])-metrics(a[ix],ae[ix]) for a,b,ae,be in zip(ca,cb,ea,eb)],axis=0))
  for j,name in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):interval.append({'kind':kind,'threshold':stage,'seed':'mean_all_three','metric':name,'old':np.mean([metrics(a,ae)[j] for a,ae in zip(ca,ea)]),'expanded_same_queries':np.mean([metrics(b,be)[j] for b,be in zip(cb,eb)]),'difference':np.mean(differences,axis=0)[j],'lower':np.quantile(bs,.025,axis=0)[j],'upper':np.quantile(bs,.975,axis=0)[j],'seed_refit_difference_SD':np.std(differences,axis=0,ddof=1)[j],'scope':'same sampled near-groups jointly across fixed seed predictions; seed/refit SD separate'})
pd.DataFrame(rows).to_csv(B/'size_and_learning_metrics.csv',index=False);pd.DataFrame(per).to_csv(B/'size_per_label.csv',index=False);pd.DataFrame(interval).to_csv(B/'paired_size_intervals.csv',index=False);d=pd.DataFrame(interval);print(d[(d.seed=='mean_all_three')&(d.metric=='micro_f1')].to_string(index=False))
