"""Configuration-grouped 1029 refresh; frozen old studies are read-only controls."""
from pathlib import Path
import json,hashlib,sys,platform
import numpy as np,pandas as pd,joblib,sklearn
from scipy.fft import dct
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,r2_score,precision_recall_fscore_support
from .coverage_study import dump,sha
from .spectral_multilabel import feature_bank,SPECIES,encode_labels,binary_scores,f1_metrics
from .fixed_top_two import fit
from .updated_data_training import threshold_fit
from .differential_training import cls_fit,cls_score,reg_estimator
from .differential_xas import pair,features
SEEDS=(20261020,20261021,20261022)

def prepare(B):
 B=Path(B);R=Path(__file__).resolve().parents[1];m=json.load(open(B/'scale/configuration_manifest.json'));z=np.load(B/'scale/per_Cu_spectra.npz');E=z['energy'];refs={d:next(i for i,r in enumerate(m) if r['domain']==d and not r['identity_labels']) for d in ['CuAu','Cu100']};old=json.load(open(R/'study/coverage_20261026/model_configuration_manifest.json'));oldsplit=json.load(open(R/'study/coverage_20261026/identity_metrics.json'))['splits'];foldmap={g:r['fold'] for r in oldsplit for g in r['test_groups']};rawz=np.load(B/'spectra.npz');rawm=json.load(open(B/'configuration_manifest.json'));native={r['configuration']:rawz[f'raw_{i}'] for i,r in enumerate(rawm)};records=[];a=[];b=[];d=[];nr=[];seen=set();excluded=[]
 for i,r in enumerate(m):
  if r['configuration'].endswith('Cu100_2x2x4_clean'):continue
  if set(r['identity_labels'])-set(SPECIES):excluded.append({'configuration':r['configuration'],'reason':'unsupported chemistry, not silently dropped labels','labels':r['identity_labels']});continue
  if r['group'] in seen:excluded.append({'configuration':r['configuration'],'reason':'same physical geometry already retained'});continue
  seen.add(r['group']);ref=m[refs[r['domain']]];sample=json.load(open(B/'scale'/r['corrected_record']));reference=json.load(open(B/'scale'/ref['corrected_record']));layers=[4,5] if r['domain']=='CuAu' else [3,4];sc=sample['sample']['composition']['atom_counts'];rc=reference['sample']['composition']['atom_counts'];assert {e:sc.get(e,0) for e in ['Cu','Au']}=={e:rc.get(e,0) for e in ['Cu','Au']},'Incompatible substrate composition';sa=np.array(sample['sample']['geometry']['lattice_vectors_A']);ra=np.array(reference['sample']['geometry']['lattice_vectors_A']);assert np.allclose(sa@sa.T,ra@ra.T,atol=2e-3,rtol=0),'Incompatible substrate cell';pop={int(l):v['population'] for l,v in r['layer_evidence'].items()};rp={int(l):v['population'] for l,v in ref['layer_evidence'].items()};p=pair(sample,reference,E,pop,rp,layers,r['domain'],r['domain']);record={**r,'old99':r['group'] in foldmap,'fold':foldmap.get(r['group'],int(hashlib.sha256(r['group'].encode()).hexdigest()[:8],16)%5),'reference_id':reference['record_id'],'reference_record_sha256':sha(B/'scale'/ref['corrected_record']),'sample_record_sha256':sha(B/'scale'/r['corrected_record']),'paired_reference_match':'composition/cell/facet/physical layers verified; initial ancestry and relaxation constraints require separate evidence, not automatically assumed','clean_scale':p['scale']};records.append(record);a.append(p['raw_common_scale']);b.append(p['clean_common_scale']);d.append(p['delta']);nr.append(native[r['configuration']])
 assert sum(r['old99'] for r in records)==99,'Old cohort membership changed: versioned correction/comparison required'
 dump(B/'training_manifest.json',records);dump(B/'training_exclusions.json',excluded);np.savez_compressed(B/'training_arrays.npz',energy=E,raw_native=np.array(nr),raw_physical=np.array(a),clean=np.array(b),delta=np.array(d));return records

def train(data_root,output_dir):
 B=Path(data_root).resolve();O=Path(output_dir).resolve();O.mkdir(parents=True,exist_ok=True)
 if any(O.iterdir()):raise ValueError('New empty training output required')
 rows=prepare(B);z=np.load(B/'training_arrays.npz');Y=encode_labels([r['identity_labels'] for r in rows]);fold=np.array([r['fold'] for r in rows]);groups=np.array([r['group'] for r in rows]);domain=np.array([r['domain'] for r in rows]);cohorts={'baseline99_CuAu':np.array([i for i,r in enumerate(rows) if r['old99']]),'refresh_CuAu':np.flatnonzero(domain=='CuAu'),'refresh_joint':np.arange(len(rows))};bn=feature_bank(z['raw_native'],z['energy']);norm={'Combined':bn['shape_CDF_derivatives'],'DCT16':dct(bn['normalized'],type=2,norm='ortho',axis=1)[:,:16]};banks={'normalized_raw':norm,'physical_raw':features(z['raw_physical'],z['energy']),'delta':features(z['delta'],z['energy'])};metrics=[];per=[];support=[];selections=[];reload=[];predictions=[];basethreshold={};M=O/'models';M.mkdir();old_groups=set(groups[cohorts['baseline99_CuAu']]);old_arrays=np.load(Path(__file__).resolve().parents[1]/'study/coverage_20261026/model_inputs.npz');old_manifest=json.load(open(Path(__file__).resolve().parents[1]/'study/coverage_20261026/model_configuration_manifest.json'));lookup={r['group']:j for j,r in enumerate(old_manifest)};input_changes=[]
 for i in cohorts['baseline99_CuAu']:
  j=lookup[rows[i]['group']];input_changes.append({'configuration':rows[i]['configuration'],'native_array_identical_to_prior':bool(np.array_equal(z['raw_native'][i],old_arrays['raw'][j])),'label_identical_to_prior':set(rows[i]['identity_labels'])==set(old_manifest[j]['identity_labels'])})
 dump(O/'baseline_source_comparison.json',input_changes)
 def record(task,cohort,rep,stage,seed,ids,p,score):
  result={'task':task,'cohort':cohort,'representation':rep,'stage':stage,'seed':seed,'n':len(ids)}
  if task=='identity':
   result.update(f1_metrics(Y[ids],p))
   for dom in ['all','CuAu','Cu100','old99','new_only']:
    mask=np.ones(len(ids),bool) if dom=='all' else np.array([rows[i]['old99'] for i in ids]) if dom=='old99' else np.array([not rows[i]['old99'] for i in ids]) if dom=='new_only' else domain[ids]==dom
    if not mask.any():continue
    yy=Y[ids][mask];pp=p[mask];pr,re,f1,s=precision_recall_fscore_support(yy,pp,average=None,zero_division=0)
    predictions.append({**result,'evaluation_subset':dom,**f1_metrics(yy,pp),'n':int(mask.sum())})
    for k,sp in enumerate(SPECIES):per.append({**result,'subset':dom,'species':sp,'support':int(s[k]),'precision':pr[k],'recall':re[k],'F1':f1[k]})
    for mixed in [False,True]:
     mm=np.array([len(rows[i]['identity_labels'])>1 for i in ids[mask]])==mixed
     for k,sp in enumerate(SPECIES):
      pos=yy[:,k].astype(bool)&mm;per.append({**result,'subset':dom+('_mixed' if mixed else '_single'),'species':sp,'support':int(pos.sum()),'recall':float(pp[pos,k].mean()) if pos.any() else None})
  else:result.update(MAE=float(mean_absolute_error(score,p)),R2=float(r2_score(score,p)))
  metrics.append(result);np.savez_compressed(O/f'{task}_{cohort}_{rep}_{stage}_{seed}.npz',pair_indices=ids,truth=Y[ids] if task=='identity' else score,prediction=p,score=score if task=='identity' else p,fold=fold[ids],configuration=np.array([rows[i]['configuration'] for i in ids]))
 # Fixed Combined anchor: same physical top two, same old outer folds, no depth/feature selection.
 for cohort,ids in cohorts.items():
  for seed in SEEDS:
   scores=np.zeros((len(ids),len(SPECIES)));pred={k:np.zeros_like(scores,int) for k in ['threshold0p5','transferred_baseline_threshold','inner_retuned_threshold']}
   for f in range(5):
    tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);inside=np.zeros_like(Y[tr],float)
    # Existing old-group inner assignment remains unchanged across cohort extension.
    oldtr=np.array([i for i in tr if rows[i]['old99']]);assignment={}
    for k,(_,v) in enumerate(GroupKFold(3).split(oldtr,groups=groups[oldtr])):
     for i in oldtr[v]:assignment[int(i)]=k
    for i in tr:
     if int(i) not in assignment:assignment[int(i)]=int(hashlib.sha256(groups[i].encode()).hexdigest()[:8],16)%3
    for k in range(3):
     aa=np.array([j for j,i in enumerate(tr) if assignment[int(i)]!=k]);vv=np.array([j for j,i in enumerate(tr) if assignment[int(i)]==k]);inside[vv]=binary_scores(fit(norm['Combined'][tr[aa]],Y[tr[aa]],1,seed),norm['Combined'][tr[vv]])
    _,th,_=threshold_fit(Y[tr],inside)
    if cohort=='baseline99_CuAu':basethreshold[seed,f]=th
    model=fit(norm['Combined'][tr],Y[tr],1,seed);s=binary_scores(model,norm['Combined'][te]);scores[pos]=s
    for stage,t in [('threshold0p5',np.full(len(SPECIES),.5)),('transferred_baseline_threshold',basethreshold[seed,f]),('inner_retuned_threshold',th)]:pred[stage][pos]=s>=t
    name=f'fixed_{cohort}_{seed}_fold{f}';path=M/(name+'.joblib');joblib.dump({'model':model,'kind':'fixed_Combined_independent_trees','feature':'Combined','representation':'normalized_raw','thresholds':th,'transferred_thresholds':basethreshold[seed,f],'train_indices':tr,'test_indices':te,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'seed':seed},path);assert np.array_equal(binary_scores(joblib.load(path)['model'],norm['Combined'][te]),s);reload.append({'path':str(path.relative_to(O)),'sha256':sha(path),'match':True});support.append({'cohort':cohort,'fold':f,'seed':seed,'train_positive_support':Y[tr].sum(axis=0).tolist(),'test_positive_support':Y[te].sum(axis=0).tolist()})
   for stage in pred:record('identity',cohort,'normalized_raw',stage,seed,ids,pred[stage],scores)
   print('fixed complete',cohort,seed,flush=True)
 # Compact predeclared physical/raw/difference ablation, one fixed seed; no outer-score winner selection.
 candidates=tuple((f,'logistic',c) for f in ['Combined','DCT16'] for c in [.1,1.])+tuple((f,'trees',2) for f in ['Combined','DCT16'])
 for cohort in ['refresh_CuAu','refresh_joint']:
  ids=cohorts[cohort]
  for rep in banks:
   pred=np.zeros_like(Y[ids]);scores=np.zeros_like(Y[ids],float)
   for f in range(5):
    tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f);inner=list(GroupKFold(3).split(tr,groups=groups[tr]));q=[];ths=[]
    for recipe in candidates:
     X=banks[rep][recipe[0]];inside=np.zeros_like(Y[tr],float)
     for a,v in inner:inside[v]=cls_score(cls_fit(X[tr[a]],Y[tr[a]],recipe),X[tr[v]])
     quality,th,_=threshold_fit(Y[tr],inside);q.append(quality);ths.append(th)
    win=int(np.argmax(q));recipe=candidates[win];X=banks[rep][recipe[0]];model=cls_fit(X[tr],Y[tr],recipe);s=cls_score(model,X[te]);scores[pos]=s;pred[pos]=s>=ths[win];path=M/f'nested_{cohort}_{rep}_fold{f}.joblib';joblib.dump({'model':model,'kind':'compact_nested_identity','recipe':recipe,'representation':rep,'thresholds':ths[win],'train_indices':tr,'test_indices':te,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'reference_ids':sorted({rows[i]['reference_id'] for i in tr}),'reference_hashes':sorted({rows[i]['reference_record_sha256'] for i in tr})},path);assert np.array_equal(cls_score(joblib.load(path)['model'],X[te]),s);reload.append({'path':str(path.relative_to(O)),'sha256':sha(path),'match':True});selections.append({'task':'identity','cohort':cohort,'rep':rep,'fold':f,'candidates':candidates,'inner_micro_F1':q,'selected':recipe,'thresholds':ths[win]})
   record('identity',cohort,rep,'compact_nested',20261027,ids,pred,scores);print('nested complete',cohort,rep,flush=True)
 # Same-denominator coverage controls, training-only compact Ridge feature/alpha selection.
 for cohort,base in cohorts.items():
  ids=np.array([i for i in base if rows[i]['CO_only']]);y=np.array([r['coverage_ML'] for r in rows]);reps=['normalized_raw'] if cohort=='baseline99_CuAu' else list(banks)
  for rep in reps:
   prediction=np.zeros(len(ids));mean=np.zeros(len(ids))
   for f in range(5):
    tr=ids[fold[ids]!=f];te=ids[fold[ids]==f];pos=np.flatnonzero(fold[ids]==f)
    if not len(te):continue
    inner=list(GroupKFold(min(3,len(tr))).split(tr,groups=groups[tr]));recipes=[(feature,'ridge',alpha) for feature in ['Combined','DCT16'] for alpha in [1.,10.,100.]];quality=[]
    for rec in recipes:
     X=banks[rep][rec[0]];pp=np.zeros(len(tr))
     for a,v in inner:pp[v]=reg_estimator(rec).fit(X[tr[a]],y[tr[a]]).predict(X[tr[v]])
     quality.append(mean_absolute_error(y[tr],pp))
    rec=recipes[int(np.argmin(quality))];X=banks[rep][rec[0]];model=reg_estimator(rec).fit(X[tr],y[tr]);prediction[pos]=model.predict(X[te]);mean[pos]=y[tr].mean();path=M/f'coverage_{cohort}_{rep}_fold{f}.joblib';joblib.dump({'model':model,'kind':'CO_only_coverage','representation':rep,'recipe':rec,'target':'CO/all_surface_metal_sites','train_indices':tr,'test_indices':te,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist()},path);assert np.array_equal(joblib.load(path)['model'].predict(X[te]),prediction[pos]);reload.append({'path':str(path.relative_to(O)),'sha256':sha(path),'match':True});selections.append({'task':'coverage','cohort':cohort,'rep':rep,'fold':f,'candidates':recipes,'inner_MAE':quality,'selected':rec})
   record('CO_coverage',cohort,rep,'compact_nested',20261027,ids,prediction,y[ids]);record('CO_coverage',cohort,rep,'training_mean',20261027,ids,mean,y[ids])
   for dom in ['CuAu','Cu100']:
    mask=domain[ids]==dom
    if mask.any():metrics.append({'task':'CO_coverage','cohort':cohort,'representation':rep,'stage':'compact_nested_'+dom,'seed':20261027,'n':int(mask.sum()),'MAE':mean_absolute_error(y[ids][mask],prediction[mask]),'R2':r2_score(y[ids][mask],prediction[mask])})
 pd.DataFrame(metrics).to_csv(O/'metrics.csv',index=False);pd.DataFrame(predictions).to_csv(O/'identity_subsets.csv',index=False);pd.DataFrame(per).to_csv(O/'per_species_single_mixed.csv',index=False);dump(O/'selection.json',selections);dump(O/'split_support.json',support);dump(O/'model_replay.json',reload);dump(O/'protocol.json',{'version':'refresh-20261029','fixed_anchor':'Combined spectrum/CDF/d1/d2, physical top2 only;64 independent balanced ExtraTrees, leaf1,sqrt; three fixed seeds, never choose best seed','old_outer':'exact old99 GroupKFold assignment; deterministic geometry-hash folds for additions; no depth selection','fixed_inner':'old training-group assignment preserved, additions hash-assigned; thresholds0.5, old-baseline transferred, and training-only retuned reported separately','compact_search':candidates,'compact_inner':'3 GroupKFold within each outer training fold; training-only feature, model, thresholds and StandardScaler','paired_reference':'CuAu->clean CuAu; Cu100->actual supplied clean Cu100 raw electronic spectra locally convolved; same-model conditional inference, no independently proved parent/experimental transfer','coverage':'CO-only; N(CO)/all surface metal sites, different from Cu-normalized loading; no counts/metadata in predictors','supported_labels':SPECIES,'unsupported_chemistry':'retained audited records but excluded from closed11-label fitting; see training_exclusions','one_clean_CuAu_negative':'kept to preserve historical99 comparison; new clean Cu100 is reference input, not an added adsorbate sample','dependencies':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},'source_hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('dataset_refresh.py'),Path(__file__).with_name('differential_training.py')]},'not_independent_confirmation':'reused configurations and folds, not a locked prospective experiment'});return {'output_dir':str(O),'metrics':str(O/'metrics.csv'),'saved_model_replays':len(reload),'configurations':len(rows)}

