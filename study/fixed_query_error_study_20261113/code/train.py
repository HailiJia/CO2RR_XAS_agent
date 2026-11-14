"""Fixed-query data-addition experiment and no more than two targeted changes."""
import json,hashlib,sys,platform
from pathlib import Path
import numpy as np,pandas as pd,sklearn,scipy,joblib
from sklearn.metrics import f1_score,accuracy_score,roc_auc_score,average_precision_score
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO));O=Path(__file__).resolve().parents[1]
from tools.differential_identity import shape_floor,signed_shape
from tools.differential_xas import features
from tools.fixed_top_two import fit
from tools.spectral_multilabel import binary_scores,SPECIES
SEEDS=(20261020,20261021,20261022)
def dump(n,x): (O/n).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=json.load(open(O/'dataset_manifest.json'));z=np.load(O/'inputs.npz');delta=z['delta'];energy=z['energy'];n=len(rows);oldn=141;y=np.array([[int(s in r['identity_labels']) for s in SPECIES] for r in rows]);fam=np.array([r['near_group'] for r in rows]);dom=np.array([r['domain'] for r in rows]);splits=[s for s in json.load(open(REPO/'study/dataset_20261101/ml/splits.json')) if s['kind']=='trees' and s['protocol']=='updated_full_cohort'];prior=json.load(open(O/'prior_local_evidence/selection.json'));ths={(s['seed'],s['fold']):s['threshold'] for s in prior if s['protocol']=='baseline'};coonly=np.array([r['identity_labels']==['CO'] for r in rows]);h=SPECIES.index('H');oh=SPECIES.index('OH');co=SPECIES.index('CO');nids=np.arange(oldn,n)
 (O/'models').mkdir(exist_ok=True);(O/'predictions').mkdir(exist_ok=True)
 protocol=dict(estimator='canonical independent ExtraTrees 64 trees leaf1 sqrt balanced; one canonical view/group',seeds=SEEDS,primary='old-only versus old-plus-new on fixed original 141 spectra/labels/query folds; original per-seed/fold thresholds transferred to BOTH arms',features='population-weighted physical top-two clean-subtracted spectra; training-floor signed RMS; Combined',query_guard='purge every held-out near group from training; five inherited cross-fold query groups retained explicitly; no train/test family overlap',new_outer_fold='inherit related old fold, else sha256(near_group) first8 modulo5',inner='saved old membership exactly; new unrelated groups assigned by sha256(near_group) modulo3; all validation-related groups purged',recalibration='separate arm only; common .3/.4/.5/.6, inner micro F1-.02(t-.5)^2',target1='H-only cutoff from common + [-.10,-.05,0,.05,.10], clipped[.05,.95]; inner group-weighted H F1, ties prefer unchanged; require inner all-H AUC>=.75 and >=3 positive/negative groups; cap extra CO-only false-H mass at one geometry group',target2='If inner H AUC among CO-containing cases <.80, compare training-only CO-template signed residual for H/OH; pure-CO mean template by substrate before RMS, nonnegative least-squares scalar; training features leave their entire group out of templates; original cutoff retained; unsupported template domains keep original score',residual_selection='H/OH individually activated only if grouped inner label F1 improves >=.02; H extra CO-only false-call mass <=1; no outer-score gating',templates='mean one canonical pure-CO spectrum per training group, shared relative intensity scale, no true query loading or site metadata',all_queries_retained=True,source_sha256=sha(__file__),dataset_sha256=sha(O/'inputs.npz'),splits_sha256=sha(REPO/'study/dataset_20261101/ml/splits.json'),versions=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__),experimental_validation=False)
 dump('evaluation_protocol.json',protocol);dump('original_thresholds.json',[dict(seed=s,fold=f,threshold=t) for (s,f),t in sorted(ths.items())]);audit=[];supports=[];ranking=[];targetlog=[];reload=[];reconcile=[];methods=['old_only_locked','expanded_locked','old_only_recalibrated','expanded_recalibrated','expanded_H_threshold','expanded_residual']
 def weights(ids):return np.array([1/np.sum(fam[ids]==fam[i]) for i in ids])
 def score_metrics(yy,pp,w=None):return dict(micro_f1=float(f1_score(yy,pp,average='micro',zero_division=0,sample_weight=w)),macro_f1=float(f1_score(yy,pp,average='macro',zero_division=0,sample_weight=w)),exact_set_accuracy=float(accuracy_score(yy,pp,sample_weight=w)))
 def label_f1(yy,pp,w):return float(f1_score(yy,pp,zero_division=0,sample_weight=w))
 def extend(base,extra,blocked):
  blocked=set(blocked);seen=set();keep=[]
  for i in list(base)+sorted(extra,key=lambda i:rows[i]['configuration']):
   if fam[i] not in blocked and fam[i] not in seen:seen.add(fam[i]);keep.append(int(i))
  return np.array(sorted(keep),int)
 def bank(a):floor=shape_floor(delta[a]);return features(signed_shape(delta,floor)[0],energy)['Combined'],floor
 def residual(train,query,loo=False):
  result=[];valid=[];coeff=[];templates=[]
  for i in query:
   candidates=[j for j in train if coonly[j] and dom[j]==dom[i] and (not loo or fam[j]!=fam[i])];unique={fam[j]:j for j in candidates};ids=list(unique.values());ok=len(ids)>=2
   t=np.mean(delta[ids],axis=0) if ok else np.zeros(len(energy));den=float(t@t);alpha=max(0,float(delta[i]@t)/den) if den>1e-20 else 0.;result.append(delta[i]-alpha*t);valid.append(ok);coeff.append(alpha);templates.append(ids)
  return np.array(result),np.array(valid),coeff,templates
 for seed in SEEDS:
  score={m:np.zeros_like(y,float) for m in methods};pred={m:np.zeros_like(y) for m in methods}
  for split in [s for s in splits if s['seed']==seed]:
   fold=split['fold'];te0=np.array(split['test']);ten=nids[np.array([rows[i]['fold']==fold for i in nids])];te=np.r_[te0,ten];oldtr=np.array(split['train']);expanded=extend(oldtr,nids[np.array([rows[i]['fold']!=fold for i in nids])],fam[te]);assert np.array_equal(oldtr,extend(oldtr,[],fam[te]));assert not set(fam[expanded])&set(fam[te]);common=ths[seed,fold];inners={};outermodels={};allinner={}
   for arm,tr in [('old_only',oldtr),('expanded',expanded)]:
    loc={int(i):j for j,i in enumerate(tr)};inside=np.zeros_like(y[tr],float);assign={fam[i]:k for k,v in enumerate(split['inner']) for i in v['validation']};assignment={i:assign.get(fam[i],int(hashlib.sha256(fam[i].encode()).hexdigest()[:8],16)%3) for i in tr};saved=[]
    for k,inner in enumerate(split['inner']):
     vv=np.array(sorted(inner['validation']+[int(i) for i in tr if i>=oldn and assignment[i]==k]),int);aa=extend(np.array(inner['train']),[i for i in tr if i>=oldn and assignment[i]!=k],fam[vv]);assert not set(fam[aa])&set(fam[vv]);assert set(aa)<=set(tr) and set(vv)<=set(tr);X,floor=bank(aa);inside[[loc[int(i)] for i in vv]]=binary_scores(fit(X[aa],y[aa],1,seed),X[vv]);saved.append(dict(train=aa.tolist(),validation=vv.tolist()));
    assert sorted(i for v in saved for i in v['validation'])==tr.tolist();inners[arm]=saved;allinner[arm]=inside
    options=[.3,.4,.5,.6];q=[score_metrics(y[tr],inside>=t)['micro_f1']-.02*(t-.5)**2 for t in options];newth=options[int(np.argmax(q))];X,floor=bank(tr);model=fit(X[tr],y[tr],1,seed);s=binary_scores(model,X[te]);path=O/'models'/f'{arm}_{seed}_{fold}.joblib';joblib.dump(dict(model=model,floor=floor,energy=energy,species=SPECIES,common_threshold=common,recalibrated_threshold=newth,train=tr,test=te,input_contract='relative-signed-identity-v2'),path,compress=3);assert np.array_equal(s,binary_scores(joblib.load(path)['model'],X[te]));reload.append(dict(path=str(path.relative_to(O)),sha256=sha(path),exact_reload=True));outermodels[arm]=(s,model)
    for method,t in [(arm+'_locked',common),(arm+'_recalibrated',newth)]:score[method][te]=s;pred[method][te]=s>=t
    for j,sp in enumerate(SPECIES):supports.append(dict(seed=seed,fold=fold,arm=arm,species=sp,positive_records=int(y[tr,j].sum()),positive_groups=len(set(fam[tr[y[tr,j]==1]])),negative_groups=len(set(fam[tr[y[tr,j]==0]]))))
    np.savez_compressed(O/f'inner_{arm}_{seed}_{fold}.npz',indices=tr,truth=y[tr],score=inside,threshold=common,recalibrated_threshold=newth)
    audit.append(dict(seed=seed,fold=fold,arm=arm,train=tr.tolist(),test_original=te0.tolist(),test_new=ten.tolist(),inner=saved,original_threshold=common,recalibrated_threshold=newth,purged_groups=sorted(set(fam[te]))))
   # Diagnose expanded inner probabilities before examining targeted outer results.
   tr=expanded;ins=allinner['expanded'];w=weights(tr);pos=y[tr,h]==1;co_mask=y[tr,co]==1
   auc=float(roc_auc_score(y[tr,h],ins[:,h],sample_weight=w)) if len(np.unique(y[tr,h]))==2 else None;coauc=float(roc_auc_score(y[tr[co_mask],h],ins[co_mask,h],sample_weight=w[co_mask])) if len(np.unique(y[tr[co_mask],h]))==2 else None
   for j,sp in enumerate(SPECIES):
    if len(np.unique(y[tr,j]))==2:ranking.append(dict(seed=seed,fold=fold,species=sp,scope='inner_training',auc=float(roc_auc_score(y[tr,j],ins[:,j],sample_weight=w)),average_precision=float(average_precision_score(y[tr,j],ins[:,j],sample_weight=w))))
   hs=common;eligible=auc is not None and auc>=.75 and len(set(fam[tr[pos]]))>=3 and len(set(fam[tr[~pos]]))>=3;choices=[];basefp=float(np.sum(w*coonly[tr]*(ins[:,h]>=common)));base_hq=label_f1(y[tr,h],ins[:,h]>=common,w)
   if eligible:
    opts=sorted(set(np.clip(np.array([common-.1,common-.05,common,common+.05,common+.1]),.05,.95)),key=lambda t:abs(t-common));best=base_hq
    for t in opts:
     fp=float(np.sum(w*coonly[tr]*(ins[:,h]>=t)));q=label_f1(y[tr,h],ins[:,h]>=t,w)-.02*(t-common)**2;choices.append(dict(threshold=float(t),quality=q,false_H_COonly_weight=fp))
     if fp<=basefp+1+1e-10 and q>best+1e-12:hs=float(t);best=q
   s=outermodels['expanded'][0];hp=(s>=common).astype(int);hp[:,h]=s[:,h]>=hs;score['expanded_H_threshold'][te]=s;pred['expanded_H_threshold'][te]=hp
   # Second targeted test, only if inner CO-conditioned H scores overlap.
   rs=s.copy();selected={};residualgate=coauc is not None and coauc<.8;res_inner=np.zeros((len(tr),2));res_valid=np.zeros(len(tr),bool);loc={int(i):j for j,i in enumerate(tr)};rinfo=[]
   if residualgate:
    for inner in inners['expanded']:
     aa=np.array(inner['train']);vv=np.array(inner['validation']);ra,va,ca,ta=residual(aa,aa,True);rv,vv_ok,cv,tv=residual(aa,vv);floor=shape_floor(ra);A=features(signed_shape(ra,floor)[0],energy)['Combined'];V=features(signed_shape(rv,floor)[0],energy)['Combined'];mm=fit(A,y[aa][:,[h,oh]],1,seed);res_inner[[loc[int(i)] for i in vv]]=binary_scores(mm,V);res_valid[[loc[int(i)] for i in vv]]=vv_ok;rinfo.append(dict(train=aa.tolist(),validation=vv.tolist(),templates_for_training=ta,templates_for_validation=tv,training_coefficients=ca,validation_coefficients=cv))
    for k,j in enumerate([h,oh]):
     blended=ins[:,j].copy();blended[res_valid]=res_inner[res_valid,k];baseq=label_f1(y[tr,j],ins[:,j]>=common,w);rq=label_f1(y[tr,j],blended>=common,w);fp=float(np.sum(w*coonly[tr]*(blended>=common)));support=len(set(fam[tr[y[tr,j]==1]]));selected[SPECIES[j]]=dict(enabled=bool(support>=3 and rq>=baseq+.02 and (j!=h or fp<=basefp+1+1e-10)),base_f1=baseq,residual_f1=rq,positive_groups=support)
    # Save raw residual scores as diagnostic even if the training-only selector rejects them.
    ra,va,ca,ta=residual(tr,tr,True);rv,ok,cv,tv=residual(tr,te);floor=shape_floor(ra);A=features(signed_shape(ra,floor)[0],energy)['Combined'];V=features(signed_shape(rv,floor)[0],energy)['Combined'];mm=fit(A,y[tr][:,[h,oh]],1,seed);ss=binary_scores(mm,V)
    for k,j in enumerate([h,oh]):
     if selected[SPECIES[j]]['enabled']:rs[ok,j]=ss[ok,k]
    np.savez_compressed(O/f'residual_diagnostic_{seed}_{fold}.npz',query=te,truth=y[te][:,[h,oh]],score=ss,template_available=ok,coefficients=cv,training_residual=ra,query_residual=rv,inner_scores=res_inner,inner_template_available=res_valid)
    path=O/'models'/f'residual_{seed}_{fold}.joblib';joblib.dump(dict(model=mm,floor=floor,train=tr,test=te,templates_for_training=ta,templates_for_queries=tv,energy=energy,targets=['H','OH'],selection=selected),path,compress=3);assert np.array_equal(ss,binary_scores(joblib.load(path)['model'],V));reload.append(dict(path=str(path.relative_to(O)),sha256=sha(path),exact_reload=True));dump(f'residual_template_audit_{seed}_{fold}.json',rinfo)
   score['expanded_residual'][te]=rs;pred['expanded_residual'][te]=rs>=common;targetlog.append(dict(seed=seed,fold=fold,H_AUC=auc,H_AUC_CO_cases=coauc,H_threshold_eligible=eligible,common_threshold=common,H_threshold=hs,H_threshold_candidates=choices,residual_test_supported=residualgate,residual_selection=selected))
   print('Completed',seed,'fold',fold,'old/new training',len(oldtr),len(expanded),'H thresholds',common,hs,'residual',selected,flush=True)
  oldp=np.load(O/'prior_local_evidence'/f'baseline_{seed}.npz');assert np.array_equal(score['old_only_locked'][:oldn],oldp['score']);assert np.array_equal(pred['old_only_locked'][:oldn],oldp['prediction']);reconcile.append(dict(seed=seed,exact_scores=True,exact_predictions=True))
  for m in methods:np.savez_compressed(O/'predictions'/f'{m}_{seed}.npz',truth=y,prediction=pred[m],score=score[m],families=fam,fold=np.array([r['fold'] for r in rows]),original_query=np.arange(n)<oldn,configuration=np.array([r['configuration'] for r in rows]))
  print('Exact baseline reproduction',seed,flush=True)
 dump('fold_manifest.json',audit);dump('target_selection.json',targetlog);dump('model_replay.json',reload);dump('baseline_reconciliation.json',reconcile);pd.DataFrame(supports).to_csv(O/'training_label_support.csv',index=False);pd.DataFrame(ranking).to_csv(O/'inner_score_ranking.csv',index=False);print('Training complete',flush=True)
if __name__=='__main__':main()
