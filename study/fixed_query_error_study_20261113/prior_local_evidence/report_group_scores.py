from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,accuracy_score
O=Path(__file__).resolve().parent;B=O.parent/'CO2RR_Agent_MS/figure_data'; protocols=['baseline','ExtraTrees','RandomForest','SVM','Logistic','nested'];seeds=[20261020,20261021,20261022];saved={};results=[]
for name in protocols:
 for seed in seeds:
  z=np.load(O/f'{name}_{seed}.npz');y=z['truth'];p=z['prediction'];f=z['families'];w=np.array([1./np.sum(f==g) for g in f]);saved[name,seed]=p
  results.append(dict(protocol=name,seed=seed,micro_f1=float(f1_score(y,p,average='micro',sample_weight=w,zero_division=0)),macro_f1=float(f1_score(y,p,average='macro',sample_weight=w,zero_division=0)),exact_set_accuracy=float(accuracy_score(y,p,sample_weight=w))))
df=pd.DataFrame(results);df.to_csv(O/'group_weighted_seed_metrics.csv',index=False);means=df.groupby('protocol')[['micro_f1','macro_f1','exact_set_accuracy']].mean().reset_index();means.to_csv(O/'group_weighted_mean_metrics.csv',index=False)
votes={name:float(f1_score(y,np.sum([saved[name,s] for s in seeds],axis=0)>=2,average='micro',sample_weight=w)) for name in protocols}
# Verify the archived headline scores exactly, under the archived group weighting.
expected=pd.read_csv(B/'latest_141/recomputed_metrics.csv');checks=[]
for seed in seeds:
 old=float(expected[(expected.model=='control')&(expected.seed==str(seed))&(expected.subset=='all')].micro_f1.iloc[0]);new=float(df[(df.protocol=='baseline')&(df.seed==seed)].micro_f1.iloc[0]);assert abs(old-new)<1e-12;checks.append(dict(seed=seed,archived=old,retrained=new))
rng=np.random.default_rng(20261113);groups=np.unique(f); ids=[np.flatnonzero(f==g) for g in groups]
def micro(yy,pp,ww):
 tp=np.sum(ww[:,None]*yy*pp);return float(2*tp/np.sum(ww[:,None]*(yy+pp)))
diffs=[]
for _ in range(5000):
 ix=np.concatenate([ids[j] for j in rng.integers(len(groups),size=len(groups))]);diffs.append(np.mean([micro(y[ix],saved['nested',s][ix],w[ix])-micro(y[ix],saved['baseline',s][ix],w[ix]) for s in seeds]))
interval=dict(lower=float(np.quantile(diffs,.025)),upper=float(np.quantile(diffs,.975)),resamples=5000,scope='paired geometry-group bootstrap conditional on fixed OOF predictions')
summary=dict(weighting='Each of 135 geometry groups has total weight one across 141 records',best_individual_exploratory=max(results,key=lambda r:r['micro_f1']),mean_metrics=means.to_dict('records'),seed_results=results,voting_micro_f1=votes,baseline_exact_headline_reproduction=checks,paired_nested_minus_baseline=interval,conclusion='No improvement from this predeclared 16-candidate search',limitations='Two features and two hyperparameters per family; SVM scores are sigmoid decision margins, uncalibrated. No claim of exhaustive architecture optimization or experimental accuracy.')
(O/'group_weighted_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
lines=['# November 3 data: model comparison','', 'All 141 configurations / 135 geometry groups retained. Five saved grouped outer folds and three saved inner folds. Three seeds. Sixteen declared candidates: ExtraTrees and Random Forest with leaf sizes 1/2 (64 trees); RBF SVM and logistic regression with C=0.1/1; each tested with signed-shape Combined and DCT16 features. Selection and common threshold fitting use inner predictions only.','', 'Scores below weight each geometry group equally, matching the archived headline metric.','',means.to_markdown(index=False) if False else means.to_string(index=False),'',f'Highest individual micro F1: {summary["best_individual_exploratory"]["micro_f1"]:.7f}. All three fixed-baseline predictions and headline scores reproduce exactly. No improvement from this search.','', 'The earlier rerun used reconstructed inner folds and record-weighted reporting. Both differed from the archived experiment; the apparent drop was not a clean comparison.','',f'Nested selector minus baseline conditional paired 95% interval: [{interval["lower"]:.5f}, {interval["upper"]:.5f}].','',summary['limitations'],'','Manuscript unchanged. Models, selection evidence, held-out predictions, input hashes and model reload checks are saved in this directory.']
(O/'REPORT.md').write_text('\n'.join(lines)+'\n')
