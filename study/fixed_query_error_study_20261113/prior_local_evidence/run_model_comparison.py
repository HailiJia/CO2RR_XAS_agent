import ast,hashlib,json,platform,time
from pathlib import Path
import numpy as np,pandas as pd,scipy,sklearn,joblib
from scipy.fft import dct
from scipy.signal import savgol_filter
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score,accuracy_score
ROOT=Path(__file__).resolve().parent
O=ROOT/'model_comparison_20261113';O.mkdir(exist_ok=False);(O/'models').mkdir()
B=ROOT/'CO2RR_Agent_MS/figure_data'; C=ROOT/'CO2RR_Agent_MS/code/source_reference'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name,obj): (O/name).write_text(json.dumps(obj,indent=2)+'\n')
def metrics(y,p):return dict(micro_f1=float(f1_score(y,p,average='micro',zero_division=0)),macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),exact_set_accuracy=float(accuracy_score(y,p)))
for fn,names in [('differential_xas.py',['features']),('differential_identity.py',['shape_floor','signed_shape'])]:
 t=ast.parse((C/fn).read_text());exec(compile(ast.Module(body=[n for n in t.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(C/fn),'exec'))
rows=json.load(open(B/'latest_141/training_manifest_compact.json')); z=np.load(B/'differential_v2/training_arrays_141.npz'); fm=json.load(open(B/'latest_141/family_map.json'))['families']; fam=np.array([fm[r['configuration']] for r in rows]); species=('CHO','CO','CO2','COCHO','COH','COOH','H','O','OCCOH','OCHO','OH');y=np.array([[int(s in r['identity_labels']) for s in species] for r in rows]);splits=[s for s in json.load(open(B/'differential_v2/source_splits.json')) if s['kind']=='trees' and s['protocol']=='updated_full_cohort'];seeds=(20261020,20261021,20261022)
# Fixed in advance: 16 candidates, 4 families; same budget of 2 parameters x 2 spectral features each.
recipes=[dict(family=f,feature=v,param=p) for f,ps in [('ExtraTrees',[1,2]),('RandomForest',[1,2]),('SVM',[.1,1.]),('Logistic',[.1,1.])] for v in ['Combined','DCT16'] for p in ps]
protected={p:sha(p) for p in (ROOT/'CO2RR_Agent_MS').rglob('*') if p.is_file()}
sourcechecks=[]
for fn in ['source_manifest.json','previous_sources_manifest.json']:
 entries=json.load(open(ROOT/'refresh_20261103'/fn));entries=entries['files'] if isinstance(entries,dict) else entries;bad=[a['path'] for a in entries if not (ROOT/a['path']).exists() or sha(ROOT/a['path'])!=a['sha256']];sourcechecks.append(dict(manifest=fn,files=len(entries),mismatches=bad));assert not bad
assert sha(B/'differential_v2/training_arrays_141.npz')==json.load(open(B/'differential_v2/source_protocol.json'))['training_arrays_sha256']
dump('protocol.json',dict(seeds=seeds,candidates=recipes,primary='nested selection across all 16 candidates using only saved inner folds; 5 grouped outer folds, all 141 queries',thresholds=[.3,.4,.5,.6],selection='inner micro F1 minus .02(t-.5)^2; ties follow declared candidate order',groups=135,records=141,trees=64,svm='RBF,gamma=scale; sigmoid(decision_function) is an uncalibrated score',scaling='StandardScaler fitted on training only for SVM/logistic; signed RMS floor fitted on training only',baseline='ExtraTrees Combined leaf1 64 trees, saved exact inner/outer membership',interpretation='Best outer-scoring family is exploratory; nested selector is primary. No outer-based threshold or query selection',source_checks=sourcechecks,arrays_sha256=sha(B/'differential_v2/training_arrays_141.npz'),splits_sha256=sha(B/'differential_v2/source_splits.json'),runner_sha256=sha(__file__),versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,scipy=scipy.__version__),manuscript_changes=False))
def fit(X,Y,r,seed):
 models=[]
 for target in Y.T:
  if len(np.unique(target))<2:models.append(int(target[0]));continue
  f=r['family'];p=r['param']
  if f in ['ExtraTrees','RandomForest']:
   cls=ExtraTreesClassifier if f=='ExtraTrees' else RandomForestClassifier;m=cls(n_estimators=64,min_samples_leaf=int(p),max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=1)
  elif f=='SVM':m=make_pipeline(StandardScaler(),SVC(C=p,kernel='rbf',gamma='scale',class_weight='balanced',random_state=seed))
  else:m=make_pipeline(StandardScaler(),LogisticRegression(C=p,class_weight='balanced',solver='liblinear',max_iter=5000,random_state=seed))
  m.fit(X,target);models.append(m)
 return models
def score(models,X,r):return np.column_stack([np.full(len(X),m) if isinstance(m,int) else expit(m.decision_function(X)) if r['family']=='SVM' else m.predict_proba(X)[:,list(m.classes_).index(1)] for m in models])
results=[]; selection=[]; replay=[]; cached={}; comparisons=[]
for seed in seeds:
 preds={name:np.zeros_like(y) for name in ['baseline','nested','ExtraTrees','RandomForest','SVM','Logistic']};scores={name:np.zeros_like(y,float) for name in preds}
 for split in [s for s in splits if s['seed']==seed]:
  fold=split['fold'];tr=np.array(split['train']);te=np.array(split['test']);assert not set(fam[tr])&set(fam[te]);loc={int(i):j for j,i in enumerate(tr)};inside=np.zeros((len(recipes),len(tr),y.shape[1]))
  for inner in split['inner']:
   a=np.array(inner['train']);v=np.array(inner['validation']);assert set(a)<=set(tr) and set(v)<=set(tr);assert not set(fam[a])&set(fam[v]);floor=shape_floor(z['delta'][a]);bank=features(signed_shape(z['delta'],floor)[0],z['energy'])
   for k,r in enumerate(recipes):inside[k,[loc[int(i)] for i in v]]=score(fit(bank[r['feature']][a],y[a],r,seed),bank[r['feature']][v],r)
  ts=[.3,.4,.5,.6]; quality=np.array([[metrics(y[tr],s>=t)['micro_f1']-.02*(t-.5)**2 for t in ts] for s in inside]);thidx=np.argmax(quality,axis=1);q=quality[np.arange(len(recipes)),thidx];chosen={'baseline':0,'nested':int(np.argmax(q))}
  for f in ['ExtraTrees','RandomForest','SVM','Logistic']:
   ids=[k for k,r in enumerate(recipes) if r['family']==f];chosen[f]=max(ids,key=lambda k:q[k])
  np.savez_compressed(O/f'inner_{seed}_{fold}.npz',truth=y[tr],scores=inside,indices=tr,quality=quality)
  floor=shape_floor(z['delta'][tr]);bank=features(signed_shape(z['delta'],floor)[0],z['energy']);outer={}
  for k in sorted(set(chosen.values())):
   r=recipes[k];X=bank[r['feature']];m=fit(X[tr],y[tr],r,seed);s=score(m,X[te],r);path=O/'models'/f'seed{seed}_fold{fold}_candidate{k}.joblib';bundle=dict(models=m,recipe=r,floor=floor,energy=z['energy'],species=species,threshold=ts[int(thidx[k])],train=tr,test=te,seed=seed);joblib.dump(bundle,path,compress=3);assert np.array_equal(s,score(joblib.load(path)['models'],X[te],r));replay.append(dict(path=str(path.relative_to(O)),sha256=sha(path),reload_exact=True));outer[k]=s
  for name,k in chosen.items():
   scores[name][te]=outer[k];preds[name][te]=outer[k]>=ts[int(thidx[k])];selection.append(dict(seed=seed,fold=fold,protocol=name,candidate=k,recipe=recipes[k],threshold=ts[int(thidx[k])],inner_quality=float(q[k])))
  print('Completed seed',seed,'fold',fold,'nested',recipes[chosen['nested']],flush=True)
 for name in preds:
  np.savez_compressed(O/f'{name}_{seed}.npz',truth=y,prediction=preds[name],score=scores[name],families=fam);results.append(dict(protocol=name,seed=seed,**metrics(y,preds[name])));cached[name,seed]=preds[name]
 old=np.load(B/f'latest_141/canonical_{seed}.npz');comparisons.append(dict(seed=seed,exact_scores=bool(np.array_equal(scores['baseline'],old['score'])),exact_predictions=bool(np.array_equal(preds['baseline'],old['prediction'])),**metrics(y,preds['baseline'])));print('Baseline reconciliation',comparisons[-1],flush=True)
 dump('partial_results.json',results)
frame=pd.DataFrame(results);frame.to_csv(O/'seed_metrics.csv',index=False);means=frame.groupby('protocol')[['micro_f1','macro_f1','exact_set_accuracy']].agg(['mean','std']);means.to_csv(O/'mean_metrics.csv');voting={name:metrics(y,np.sum([cached[name,s] for s in seeds],axis=0)>=2) for name in preds}
# Conditional paired family bootstrap, preserving all views per sampled family.
rng=np.random.default_rng(20261113);groups=np.unique(fam);indices=[np.flatnonzero(fam==g) for g in groups];samples=[]
for _ in range(2000):
 ix=np.concatenate([indices[j] for j in rng.integers(len(groups),size=len(groups))]);diff=np.mean([metrics(y[ix],cached['nested',s][ix])['micro_f1']-metrics(y[ix],cached['baseline',s][ix])['micro_f1'] for s in seeds]);samples.append(diff)
interval=dict(mean_difference=float(np.mean([r['micro_f1'] for r in results if r['protocol']=='nested'])-np.mean([r['micro_f1'] for r in results if r['protocol']=='baseline'])),lower=float(np.quantile(samples,.025)),upper=float(np.quantile(samples,.975)),resamples=2000,scope='conditional on saved OOF predictions, group bootstrap; not independent experimental validation')
summary=dict(best_individual_exploratory=max(results,key=lambda r:r['micro_f1']),mean_metrics=frame.groupby('protocol')[['micro_f1','macro_f1','exact_set_accuracy']].mean().reset_index().to_dict('records'),seed_results=results,voting=voting,baseline_reconciliation=comparisons,threshold_discrepancy='First rerun reconstructed inner memberships differently; restored saved source_splits.json memberships for all comparisons',paired_nested_minus_baseline=interval,saved_models=len(replay),manuscript_unchanged=all(sha(p)==h for p,h in protected.items()),scope='Fresh fits on verified November 3 audited inputs; 141 configurations /135 groups; no expanded cohort')
dump('selection.json',selection);dump('model_replay.json',replay);dump('summary.json',summary);assert summary['manuscript_unchanged'];print(json.dumps(summary,indent=2),flush=True)
