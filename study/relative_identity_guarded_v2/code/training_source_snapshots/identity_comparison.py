"""Predeclared grouped relative-difference identity comparisons; no outer tuning."""
from pathlib import Path
import json,hashlib,platform
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
from .differential_xas import features
from .differential_identity import VERSION,shape_floor,signed_shape
from .spectral_multilabel import feature_bank,SPECIES,encode_labels,binary_scores,f1_metrics
from .fixed_top_two import fit as tree_fit
from .coverage_study import dump,sha
SEEDS=(20261020,20261021,20261022)
REPS=('native_normalized','corrected_normalized','common_raw','relative_delta','signed_shape')
CANDIDATES=tuple((f,k) for f in ['signed','DCT16','Combined'] for k in ['logistic','trees'])

def transform(x,e,rep,feature,floor):
 if rep in {'native_normalized','corrected_normalized'}:
  b=feature_bank(x,e);return {'signed':b['normalized'],'DCT16':__import__('scipy').fft.dct(b['normalized'],type=2,norm='ortho',axis=1)[:,:16],'Combined':b['shape_CDF_derivatives']}[feature]
 if rep=='signed_shape':x=signed_shape(x,floor)[0]
 return features(x,e)[feature]

def fit_model(X,y,kind,seed):
 if kind=='trees':return tree_fit(X,y,1,seed)
 scaler=StandardScaler().fit(X);xx=scaler.transform(X);estimators=[]
 for col in y.T:
  estimators.append(int(col[0]) if len(np.unique(col))<2 else LogisticRegression(C=1.,class_weight='balanced',solver='liblinear',random_state=seed,max_iter=3000).fit(xx,col))
 return {'kind':'logistic','scaler':scaler,'estimators':estimators}

def score_model(m,X):
 if m['kind']!='logistic':return binary_scores(m,X)
 xx=m['scaler'].transform(X)
 return np.column_stack([np.full(len(xx),a) if isinstance(a,int) else a.predict_proba(xx)[:,list(a.classes_).index(1)] for a in m['estimators']])

def common_threshold(y,s):
 ts=(.3,.4,.5,.6);q=[f1_metrics(y,(s>=t).astype(int))['micro_f1']-.02*(t-.5)**2 for t in ts]
 # One threshold for all labels, restrained toward .5; no per-label rare-class fitting.
 return float(ts[int(np.argmax(q))])

def train(data_root,output_dir,family_map=None):
 B=Path(data_root);O=Path(output_dir);O.mkdir(parents=True,exist_ok=True)
 if any(O.iterdir()):raise ValueError('Use a new empty output directory')
 M=O/'models';M.mkdir();r=json.load(open(B/'training_manifest.json'));z=np.load(B/'training_arrays.npz');e=z['energy'];y=encode_labels([a['identity_labels'] for a in r]);g=np.array([a['group'] for a in r]);fold=np.array([a['fold'] for a in r]);old=np.array([a['old99'] for a in r]);dom=np.array([a['domain'] for a in r]);cohorts={'old99':np.flatnonzero(old),'expanded118':np.flatnonzero(dom=='CuAu'),'joint130':np.arange(len(r))};arrays={k:z[v] for k,v in [('native_normalized','raw_native'),('corrected_normalized','raw_physical'),('common_raw','raw_physical'),('relative_delta','delta'),('signed_shape','delta')]};metrics=[];per=[];strata=[];splits=[];selections=[];replay=[];bootstrap=[];thresholds=[];saved={};curve=[];policies=[];stress=[]
 families=np.array([family_map['families'][a['configuration']] if family_map else a['group'] for a in r]);outer_context=[None]
 def guard(tr,te,outer=False):
  tr=np.asarray(tr,int);te=np.asarray(te,int)
  if outer:outer_context[0]=int(fold[te[0]]) if len(te) and len(set(fold[te]))==1 else None
  if not family_map:return tr
  blocked=set(families[te]);seen=set();keep=[]
  for i in sorted(tr,key=lambda i:(not old[i],r[i]['configuration'])):
   if families[i] not in blocked and families[i] not in seen:keep.append(int(i));seen.add(families[i])
  return np.array(sorted(keep),int)
 dump(O/'family_map.json',family_map or {'status':'legacy exact groups only'});dump(O/'source_manifest.json',r);np.savez_compressed(O/'inputs.npz',**{k:z[k] for k in z.files});np.savez_compressed(O/'targets_folds.npz',truth=y,fold=fold,groups=g,old99=old,domain=dom)
 def inner(tr):
  oo=np.flatnonzero(old & (fold!=outer_context[0])) if outer_context[0] is not None else np.array([i for i in tr if old[i]]);assignment={}
  if len(oo)>=3:
   for k,(_,v) in enumerate(GroupKFold(3).split(oo,groups=g[oo])):
    for i in oo[v]:assignment[int(i)]=k
  for i in tr:
   if int(i) not in assignment:assignment[int(i)]=int(hashlib.sha256(g[i].encode()).hexdigest()[:8],16)%3
  return [(np.array([i for i in tr if assignment[int(i)]!=k]),np.array([i for i in tr if assignment[int(i)]==k])) for k in range(3)]
 def fit_recipe(rep,feature,kind,tr,te,seed,augment=False):
  tr=guard(tr,te);floor=shape_floor(z['delta'][tr]);a=arrays[rep][tr];yy=y[tr]
  if augment:
   rng=np.random.default_rng(seed);copies=[]
   for x in a:
    shift=rng.choice([-.2,.2]);ok=(e+shift>=e[0])&(e+shift<=e[-1]);v=x.copy();v[ok]=np.interp(e[ok]+shift,e,x);v=gaussian_filter1d(v,.25/np.median(np.diff(e)),mode='nearest');v=v*rng.uniform(.95,1.05);v+=rng.normal(0,.01*max(np.max(abs(x)),1e-12),len(e));copies.append(v)
   a=np.vstack([a,copies]);yy=np.vstack([yy,yy])
  X=transform(a,e,rep,feature,floor);m=fit_model(X,yy,kind,seed);return m,score_model(m,transform(arrays[rep][te],e,rep,feature,floor)),floor
 def inside(rep,recipe,tr,seed,aug=False):
  out=np.zeros_like(y[tr],float);loc={int(v):j for j,v in enumerate(tr)}
  for a,v in inner(tr):
   if len(a) and len(v):out[[loc[int(i)] for i in v]]=fit_recipe(rep,*recipe,a,v,seed,aug)[1]
  return out
 def save(name,rep,recipe,tr,te,seed,model,sc,floor,th):
  tr=guard(tr,te);p=M/(name+'.joblib');bundle={'input_contract':VERSION,'representation':rep,'feature':recipe[0],'kind':recipe[1],'model':model,'floor':floor,'thresholds':np.full(len(SPECIES),th),'energy':e,'species':SPECIES,'domains':sorted(set(dom[tr])),'train_indices':tr,'test_indices':te,'train_groups':g[tr].tolist(),'test_groups':g[te].tolist(),'train_families':families[tr].tolist(),'test_families':families[te].tolist(),'reference_ids':sorted({r[i]['reference_id'] for i in tr}),'normalization':'signed-shape RMS after training floor' if rep=='signed_shape' else 'shared simulated reference near-edge divisor; not validated experimental full edge jump','seed':seed};joblib.dump(bundle,p,compress=3);mm=joblib.load(p);assert np.array_equal(score_model(mm['model'],transform(arrays[rep][te],e,rep,recipe[0],floor)),sc);replay.append({'path':str(p.relative_to(O)),'sha256':sha(p),'exact_reload':True});return bundle
 def record(name,ids,s,p,meta):
  np.savez_compressed(O/(name+'.npz'),indices=ids,truth=y[ids],score=s,prediction=p,fold=fold[ids]);saved[name]=(ids,s,p);metrics.append({**meta,'name':name,'subset':'all','n':len(ids),**f1_metrics(y[ids],p)})
  masks={'old99':old[ids],'new_only':~old[ids],'single':np.array([len(r[i]['identity_labels'])==1 for i in ids]),'mixed':np.array([len(r[i]['identity_labels'])>1 for i in ids]),'clean':y[ids].sum(axis=1)==0,'CuAu':dom[ids]=='CuAu','Cu100':dom[ids]=='Cu100'}
  cov=np.array([r[i]['coverage_ML'] for i in ids]);masks.update(coverage_le_0p1=cov<=.1,coverage_gt_0p1=cov>.1)
  for key in ['interface_distance_A','mean_interface_distance_A','coordination','location']:
   # Only actual declared audit fields can define strata; never infer from names.
   values=[r[i].get(key) for i in ids]
   for v in sorted({str(v) for v in values if v is not None}):masks[key+'='+v]=np.array([str(a)==v for a in values])
  for sub,mask in masks.items():
   if mask.any():
    yy=y[ids][mask];pp=p[mask];metrics.append({**meta,'name':name,'subset':sub,'n':int(mask.sum()),**f1_metrics(yy,pp),'false_positive_labels':int(((yy==0)&(pp==1)).sum())})
    if sub in ['CuAu','Cu100']:
     jj=[SPECIES.index(a) for a in ['CO','H','O','OH']];metrics.append({**meta,'name':name,'subset':sub+'_common4','n':int(mask.sum()),**f1_metrics(yy[:,jj],pp[:,jj])})
    pr,re,ff,su=precision_recall_fscore_support(yy,pp,average=None,zero_division=0)
    for j,sp in enumerate(SPECIES):per.append({'name':name,'subset':sub,'species':sp,'precision':pr[j],'recall':re[j],'f1':ff[j],'support':int(su[j]),'FP':int(((yy[:,j]==0)&(pp[:,j]==1)).sum()),'FN':int(((yy[:,j]==1)&(pp[:,j]==0)).sum())})
  for t in [0.,.05,.1,.2]:
   keep=np.min(abs(s-.5),axis=1)>=t;valid=keep.copy()
   # Margin is an uncalibrated review diagnostic, never optimized on test curves.
   policies.append({'name':name,'margin':t,'retained_fraction':float(keep.mean()),'n_retained':int(keep.sum()),'exact_error_retained':float(np.mean(np.any(y[ids][keep]!=p[keep],axis=1))) if keep.any() else None,'full_query_exact_error':float(np.mean(np.any(y[ids]!=p,axis=1)))})
 # Four source/normalization controls and signed-shape identity control. Equal compact budgets.
 for cohort in ['expanded118','joint130']:
  ids=cohorts[cohort]
  for rep in REPS:
   for family in ['logistic','trees']:
    recipes=[c for c in CANDIDATES if c[1]==family];scores=np.zeros_like(y[ids],float);pred=np.zeros_like(y[ids]);p5=np.zeros_like(pred)
    for f in range(5):
     tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);tr=guard(tr,te,outer=True);qs=[];ii=[];ths=[]
     for recipe in recipes:
      ss=inside(rep,recipe,tr,SEEDS[0]);th=common_threshold(y[tr],ss);qs.append(f1_metrics(y[tr],(ss>=th).astype(int))['micro_f1']);ii.append(ss);ths.append(th)
     win=int(np.argmax(qs));rec=recipes[win];m,s,floor=fit_recipe(rep,*rec,tr,te,SEEDS[0]);th=ths[win];scores[pos]=s;pred[pos]=s>=th;p5[pos]=s>=.5;save(f'nested_{cohort}_{rep}_{family}_{f}',rep,rec,tr,te,SEEDS[0],m,s,floor,th)
     selections.append({'cohort':cohort,'rep':rep,'family':family,'fold':f,'candidates':recipes,'inner_micro':qs,'selected':rec,'threshold':th});splits.append({'cohort':cohort,'fold':f,'train':tr.tolist(),'test':te.tolist(),'inner':[{'train':a.tolist(),'validation':v.tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[a].sum(axis=0)==0)]} for a,v in inner(tr)],'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]})
    meta={'protocol':'nested','cohort':cohort,'representation':rep,'estimator':family,'seed':SEEDS[0]};name=f'nested_{cohort}_{rep}_{family}';record(name,ids,scores,pred,meta);record(name+'_fixed0p5',ids,scores,p5,{**meta,'threshold':'fixed0p5'})
   print('nested complete',cohort,rep,flush=True)
 # Delta training-size anchor: same Combined/trees/leaf1 as previous raw anchor.
 transferred={}
 for seed in SEEDS:
  for cohort in ['old99','expanded118']:
   ids=cohorts[cohort];scores=np.zeros_like(y[ids],float);pp={k:np.zeros_like(y[ids]) for k in ['fixed0p5','transferred','common']}
   for f in range(5):
    tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);tr=guard(tr,te,outer=True);ip=inside('relative_delta',('Combined','trees'),tr,seed);th=common_threshold(y[tr],ip)
    if cohort=='old99':transferred[seed,f]=th
    m,s,floor=fit_recipe('relative_delta','Combined','trees',tr,te,seed);scores[pos]=s
    for stage,t in [('fixed0p5',.5),('transferred',transferred[seed,f]),('common',th)]:pp[stage][pos]=s>=t
    save(f'size_{cohort}_{seed}_{f}','relative_delta',('Combined','trees'),tr,te,seed,m,s,floor,th);thresholds.append({'cohort':cohort,'seed':seed,'fold':f,'common':th,'transferred':transferred[seed,f]})
   for stage,p in pp.items():record(f'size_{cohort}_{seed}_{stage}',ids,scores,p,{'protocol':'size','cohort':cohort,'representation':'relative_delta','seed':seed,'threshold':stage})
  # Grouped learning curve: fixed old99 outer queries; subset selection label-blind.
  ids=cohorts['expanded118'];query=cohorts['old99']
  for fraction in [.5,.75,1.]:
   scores=np.zeros_like(y[query],float)
   for f in range(5):
    tr=ids[fold[ids]!=f];rng=np.random.default_rng(seed+f);tr=rng.permutation(tr)[:max(3,int(len(tr)*fraction))];te=query[fold[query]==f];pos=np.flatnonzero(fold[query]==f);tr=guard(tr,te,outer=True);m,s,floor=fit_recipe('relative_delta','Combined','trees',tr,te,seed);scores[pos]=s;save(f'curve_{fraction}_{seed}_{f}','relative_delta',('Combined','trees'),tr,te,seed,m,s,floor,.5);curve.append({'fraction':fraction,'seed':seed,'fold':f,'n_training_groups':len(tr),'train_indices':tr.tolist(),'query_indices':te.tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]})
   record(f'curve_{fraction}_{seed}',query,scores,(scores>=.5).astype(int),{'protocol':'learning_curve','seed':seed,'fraction':fraction,'cohort':'old99_queries','representation':'relative_delta'})
  for stage in ['fixed0p5','transferred','common']:
   ia,sa,pa=saved[f'size_old99_{seed}_{stage}'];ib,sb,pb=saved[f'size_expanded118_{seed}_{stage}'];lookup={int(i):j for j,i in enumerate(ib)};pb=pb[[lookup[int(i)] for i in ia]];diff=f1_metrics(y[ia],pb)['micro_f1']-f1_metrics(y[ia],pa)['micro_f1'];rng=np.random.default_rng(930+seed);vals=[]
   for _ in range(1000):
    ii=rng.integers(0,len(ia),len(ia));vals.append(f1_metrics(y[ia][ii],pb[ii])['micro_f1']-f1_metrics(y[ia][ii],pa[ii])['micro_f1'])
   bootstrap.append({'seed':seed,'stage':stage,'difference_micro':diff,'lower':float(np.quantile(vals,.025)),'upper':float(np.quantile(vals,.975)),'definition':'conditional configuration bootstrap of fixed paired predictions; not refit uncertainty'})
  print('size/curve complete',seed,flush=True)
 # Entire-domain holdout, not a new-parent validation; common4 + full11 saved.
 for target in ['Cu100','CuAu']:
  tr=np.flatnonzero(dom!=target);te=np.flatnonzero(dom==target);tr=guard(tr,te,outer=True)
  for rep in ['corrected_normalized','relative_delta','signed_shape']:
   m,s,floor=fit_recipe(rep,'DCT16','logistic',tr,te,SEEDS[0]);th=common_threshold(y[tr],inside(rep,('DCT16','logistic'),tr,SEEDS[0]));save(f'domain_{target}_{rep}',rep,('DCT16','logistic'),tr,te,SEEDS[0],m,s,floor,th);record(f'domain_{target}_{rep}',te,s,(s>=th).astype(int),{'protocol':'domain_holdout','cohort':target,'representation':rep,'seed':SEEDS[0]});splits.append({'protocol':'domain_holdout','target':target,'train':tr.tolist(),'test':te.tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]})
 # Predeclared training-only augmentation, paired raw/delta stress on identical crops.
 ids=cohorts['expanded118'];step=np.median(np.diff(e));crop=(e>=e[0]+1)&(e<=e[-1]-1)
 for rep in ['corrected_normalized','relative_delta','signed_shape']:
  for aug in [False,True]:
   aggregate={name:np.zeros_like(y[ids],float) for name in ['clean','shift+0p2','relative_shift+0p2','broadening0p25','noise1pct','gain5pct']};low=np.zeros(len(ids),bool)
   for f in range(5):
    tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);tr=guard(tr,te,outer=True);rec=('DCT16','logistic');m,s,floor=fit_recipe(rep,*rec,tr,te,SEEDS[0],aug);save(f'augment_{rep}_{aug}_{f}',rep,rec,tr,te,SEEDS[0],m,s,floor,.5)
    # Perturb pair members before subtraction: shared shift/gain and residual sample-only shift.
    for name in aggregate:
     a=z['raw_physical'][te].copy();b=z['clean'][te].copy();rng=np.random.default_rng(746+f)
     if name in ['shift+0p2','relative_shift+0p2']:
      a[:,crop]=np.array([np.interp(e[crop]+.2,e,x) for x in a])
      if name=='shift+0p2':b[:,crop]=np.array([np.interp(e[crop]+.2,e,x) for x in b])
     if name=='broadening0p25':a=gaussian_filter1d(a,.25/step,axis=1);b=gaussian_filter1d(b,.25/step,axis=1)
     if name=='gain5pct':a*=1.05;b*=1.05
     if name=='noise1pct':
      shared=rng.normal(0,.01*b.max(),len(e));a+=shared+rng.normal(0,.01*a.max(),a.shape);b+=shared+rng.normal(0,.01*b.max(),b.shape)
     x=a if rep=='corrected_normalized' else a-b
     # Same supported interior; boundary values unused; replace margins by unperturbed inputs.
     x[:,~crop]=arrays[rep][te][:,~crop]
     aggregate[name][pos]=score_model(m,transform(x,e,rep,rec[0],floor))
    low[pos]=np.sqrt(np.mean(z['delta'][te]**2,axis=1))<=floor
   for name,s in aggregate.items():record(f'augment_{rep}_{aug}_{name}',ids,s,(s>=.5).astype(int),{'protocol':'synthetic_stress','representation':rep,'augmented_training':aug,'scenario':name,'cohort':'expanded118','seed':SEEDS[0]});stress.append({'rep':rep,'augmented':aug,'low_signal_count':int(low.sum()),'n_queries':len(ids),'perturbed_interior_eV':[float(e[crop][0]),float(e[crop][-1])],'boundary_convention':'outer1eV margins retain native signal; finite-window sensitivity, not full instrument shift','synthetic_not_measured_uncertainty':True})
 # Dilution reduces amplitude; never reinterpret it as known surface coverage.
 for rep in ['relative_delta','signed_shape']:
  ids=cohorts['expanded118']
  for fraction in [.1,.25,.5,1.]:
   scores=np.zeros_like(y[ids],float)
   for f in range(5):
    te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);m=joblib.load(M/f'augment_{rep}_False_{f}.joblib');scores[pos]=score_model(m['model'],transform(z['delta'][te]*fraction,e,rep,'DCT16',m['floor']))
   record(f'dilution_{rep}_{fraction}',ids,scores,(scores>=.5).astype(int),{'protocol':'synthetic_dilution','representation':rep,'surface_fraction':fraction,'cohort':'expanded118'})
 pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);pd.DataFrame(per).to_csv(O/'per_label.csv',index=False);pd.DataFrame(policies).to_csv(O/'review_policy.csv',index=False);pd.DataFrame(bootstrap).to_csv(O/'paired_bootstrap.csv',index=False);dump(O/'splits.json',splits);dump(O/'selection.json',selections);dump(O/'thresholds.json',thresholds);dump(O/'learning_membership.json',curve);dump(O/'stress_protocol.json',stress);dump(O/'model_replay.json',replay)
 dump(O/'protocol.json',{'near_family_guard':family_map,'version':VERSION,'seeds':SEEDS,'representations':REPS,'candidates':CANDIDATES,'nested':'paired old training inner groups; hash assignment additions; common threshold .3/.4/.5/.6 with .02(t-.5)^2 penalty; fixed.5 also shown','signed_shape_floor':'max(1e-10,1% of tenth-percentile nonzero training RMS); heuristic not measured detection limit; low inputs stay zero','augmentation':'one synthetic training copy; shifts±.2eV, Gaussian sigma.25eV, gain±5%, noise1% native peak; same configuration fold; illustrative ranges without measured noise metadata','scope':'conditional same-reference simulation; one reference family/domain; no measured experimental validation; no coverage recalibration','source_hashes':{str(p):sha(p) for p in [B/'training_arrays.npz',B/'training_manifest.json',Path(__file__),Path(__file__).with_name('differential_identity.py')]},'python':platform.python_version()})
 return {'status':'success','output_dir':str(O),'metrics':str(O/'metrics.csv'),'models':len(replay),'prediction_replay_identical':True,'trace':str(O/'protocol.json')}
