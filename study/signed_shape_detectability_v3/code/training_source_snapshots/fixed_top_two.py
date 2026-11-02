"""Paired fixed-top-two evaluation; no depth selection or seed selection."""
from pathlib import Path
import json,hashlib,sys
import numpy as np,pandas as pd,joblib,sklearn,scipy
from scipy.fft import dct
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import precision_recall_fscore_support
from .spectral_multilabel import SPECIES,feature_bank,binary_scores,f1_metrics
from .updated_data_training import threshold_fit,inner_folds
from .scientific_extensions import single_mixed
SEEDS=(20261020,20261021,20261022)
BASE_FEATURES=('normalized','CDF','derivative','shape_CDF_derivatives','DCT24')
SEARCH_FEATURES=('derivative','shape_CDF_derivatives','DCT8','DCT16','DCT24','DCT32')
CANDIDATES=[(f,1) for f in BASE_FEATURES]+[(f,l) for f in SEARCH_FEATURES for l in [1,2,4] if (f,l) not in [(v,1) for v in BASE_FEATURES]]
STAGES=('fixed_d1_threshold0p5','fixed_d1_transferred87_threshold','fixed_d1_retuned_threshold','original_feature_selection','regularized_selection')


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def fit(x,y,leaf,seed):
    models=[]
    for j in range(y.shape[1]):
        if len(np.unique(y[:,j]))<2:models.append(int(y[0,j]));continue
        models.append(ExtraTreesClassifier(n_estimators=64,min_samples_leaf=leaf,max_features='sqrt',class_weight='balanced',random_state=seed,n_jobs=1).fit(x,y[:,j]))
    return {'models':models,'scaler':None,'kind':'independent_trees','n_outputs':y.shape[1]}


def load_data(repo):
    base=Path(repo)/'study/retraining_20261022/refit/dataset';z=np.load(base/'cohort.npz');cv=pd.read_csv(base/'cv_manifest.csv');old=pd.read_csv(Path(repo)/'study/manuscript_package/figure_data/cv_manifest.csv');mapping={v:i for i,v in enumerate(cv.prior_configuration) if isinstance(v,str)};old_ids=np.array([mapping[v] for v in old.configuration]);new_ids=np.flatnonzero(cv.prior_configuration.isna());order=np.r_[old_ids,new_ids];raw=z['stack'][order];pop=z['populations'][order];avg=np.array([np.average(a[[3,2]],axis=0,weights=p[[3,2]]) for a,p in zip(raw,pop)]);bank=feature_bank(avg,z['energy']);bank.update({f'DCT{n}':dct(bank['normalized'],type=2,norm='ortho',axis=1)[:,:n] for n in [8,16,24,32]});y=z['truth'][order];fold=z['fold'][order];groups=z['groups'][order]
    assert np.array_equal(fold[:87],old.fold) and len(set(groups))==90
    return {'base':base,'energy':z['energy'],'raw_top_two':avg,'bank':bank,'y':y,'fold':fold,'groups':groups,'source_indices':order,'configuration':z['configuration'][order],'old_manifest':old}


def paired_inner(data,outer,n):
    train=np.flatnonzero(data['fold'][:n]!=outer);old_train=np.flatnonzero(data['fold'][:87]!=outer);assignment={}
    for k,(it,iv) in enumerate(inner_folds(data['y'][old_train],outer)):
        for i in old_train[iv]:assignment[int(i)]=k
    # Only additional OUTER-TRAINING rows enter this deterministic extension.
    for i in train:
        if i not in assignment:assignment[int(i)]=int(hashlib.sha256(str(data['groups'][i]).encode()).hexdigest()[:8],16)%3
    pairs=[]
    for k in range(3):
        tr=np.array([i for i in train if assignment[int(i)]!=k]);va=np.array([i for i in train if assignment[int(i)]==k]);assert not set(data['groups'][tr])&set(data['groups'][va]);pairs.append((tr,va))
    return train,pairs


def write_protocol(repo,out,data):
    protocol={'version':'fixed-top-two-v4','depth':'population-weighted raw layer 5 + layer 4 ONLY, before minimum-subtracted unit-area normalization','cohorts':[87,90],'old_order':'original frozen cv_manifest order; three additions appended; exact geometry mapping retained','outer_folds':'identical old configuration assignments; same recorded assignments for the three additions','inner_folds':'old 87 inner partitions fixed across cohorts/seeds; deterministic hash assignment for added outer-training configurations; no held-out labels used','seed_replicates':list(SEEDS),'seed_role':'tree randomness only; outer/inner partitions fixed; no best seed chosen','fixed_anchor':'d1, leaf1. Fixed 0.5 threshold and old-87 training-only thresholds transferred unchanged to 90 isolate training-data effects; retuned thresholds reported separately','existing_selection':list(BASE_FEATURES),'regularized_search_features':list(SEARCH_FEATURES),'minimum_leaf_sizes':[1,2,4],'candidates':[list(v) for v in CANDIDATES],'estimator':'11 independent ExtraTrees binary targets,64 trees,balanced,sqrt; constant predictors for single-class training','thresholds':'existing micro global+shrunk training-only decoder; no calibration or outer-test threshold fitting','stages':list(STAGES),'new_only_reporting':'CO precision, recall, F1 and exact identity sets. Ten absent labels untested; eleven-label macro F1 is provenance only, not a useful headline','comparison':'fixed prediction paired intervals condition on fit; across-seed variation is refitting variability, not independent confirmation','source_sha256':digest(__file__),'scientific_source_hashes':{str(p.relative_to(repo)):digest(p) for p in [Path(__file__),Path(repo)/'tools/updated_data_training.py',Path(repo)/'tools/spectral_multilabel.py']},'snapshot_sha256':digest(data['base']/'cohort.npz'),'dependencies':{'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'scipy':scipy.__version__}}
    p=out/'protocol.json'
    if p.exists() and json.loads(p.read_text())!=protocol:raise ValueError('Do not alter a declared experiment directory')
    p.write_text(json.dumps(protocol,indent=2)+'\n')


def evaluate(repo,out,n,seed):
    repo=Path(repo);out=Path(out);out.mkdir(parents=True,exist_ok=True);data=load_data(repo);write_protocol(repo,out,data);d=out/f'cohort{n}_seed{seed}';d.mkdir(exist_ok=True)
    if (d/'metrics.csv').exists():raise ValueError('Completed run exists; do not overwrite')
    y=data['y'][:n];fold=data['fold'][:n];nc=len(CANDIDATES);all_score=np.zeros((nc,n,len(SPECIES)));all_pred=np.zeros((nc,n,len(SPECIES)),int);stages={s:np.zeros((n,len(SPECIES)),int) for s in STAGES};stage_scores={s:np.zeros((n,len(SPECIES))) for s in STAGES};rows=[];supports=[];rankings=[];saved=[]
    anchor=CANDIDATES.index(('derivative',1));orig=[i for i,(f,l) in enumerate(CANDIDATES) if f in BASE_FEATURES and l==1];reg=[i for i,(f,l) in enumerate(CANDIDATES) if f in SEARCH_FEATURES]
    for outer in range(5):
        tr,inner=paired_inner(data,outer,n);te=np.flatnonzero(fold==outer);assert not set(data['groups'][tr])&set(data['groups'][te]);fold_dir=d/f'fold{outer}';fold_dir.mkdir(exist_ok=True);inside=np.zeros((nc,len(tr),len(SPECIES)));outside=np.zeros((nc,len(te),len(SPECIES)));thresholds=[];qualities=[];models=[];position={int(v):i for i,v in enumerate(tr)}
        for j,(feature,leaf) in enumerate(CANDIDATES):
            x=data['bank'][feature]
            for it,iv in inner:inside[j,[position[int(i)] for i in iv]]=binary_scores(fit(x[it],data['y'][it],leaf,seed),x[iv])
            q,ts,decoder=threshold_fit(y[tr],inside[j]);thresholds.append(ts);qualities.append(q);model=fit(x[tr],y[tr],leaf,seed);models.append(model);outside[j]=binary_scores(model,x[te]);all_score[j,te]=outside[j];all_pred[j,te]=outside[j]>=ts;rankings.append({'cohort':n,'seed':seed,'fold':outer,'feature':feature,'leaf':leaf,'inner_micro_f1':q,'decoder':decoder,'thresholds':json.dumps(ts.tolist())})
        thresholds=np.array(thresholds);old_path=out/f'cohort87_seed{seed}/fold{outer}/inner.npz';oldthreshold=thresholds[anchor] if n==87 else np.load(old_path)['thresholds'][anchor]
        winners={'original_feature_selection':max(orig,key=lambda j:qualities[j]),'regularized_selection':max(reg,key=lambda j:qualities[j])}
        choices={s:anchor for s in STAGES};choices.update(winners);used=set(choices.values())
        for stage,j in choices.items():
            ts=np.full(len(SPECIES),.5) if stage=='fixed_d1_threshold0p5' else oldthreshold if stage=='fixed_d1_transferred87_threshold' else thresholds[j];stages[stage][te]=outside[j]>=ts;stage_scores[stage][te]=outside[j];rows.append({'cohort':n,'seed':seed,'fold':outer,'stage':stage,'feature':CANDIDATES[j][0],'leaf':CANDIDATES[j][1],'thresholds':json.dumps(ts.tolist()),'inner_micro_f1':qualities[j],'train_indices':json.dumps(tr.tolist()),'test_indices':json.dumps(te.tolist())})
        for j in used:
            feature,leaf=CANDIDATES[j];p=fold_dir/f'{feature}_leaf{leaf}.joblib';bundle={'format':'co2rr-multilabel-v1','fit':models[j],'energy':data['energy'],'species':list(SPECIES),'depth':'top_2','feature':feature,'thresholds':thresholds[j],'provenance':{'cohort':n,'seed':seed,'fold':outer,'training_indices':tr.tolist(),'test_indices':te.tolist(),'source_snapshot_indices':data['source_indices'].tolist(),'candidate':j,'experiment':'fixed-top-two-v4'},'normalization':'raw population-weighted top-two average; minimum-subtracted unit-area'};joblib.dump(bundle,p,compress=3);assert np.array_equal(binary_scores(joblib.load(p)['fit'],data['bank'][feature][te]),outside[j]);saved.append({'path':str(p.relative_to(out)),'sha256':digest(p),'replay_identical':True})
        np.savez_compressed(fold_dir/'inner.npz',scores=inside,truth=y[tr],training_indices=tr,thresholds=thresholds,quality=qualities);(fold_dir/'splits.json').write_text(json.dumps([{'training_indices':a.tolist(),'validation_indices':b.tolist(),'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[a].sum(axis=0)==0)]} for a,b in inner],indent=2)+'\n');supports.extend([{'cohort':n,'seed':seed,'fold':outer,'species':s,'training_positives':int(y[tr,j].sum()),'test_positives':int(y[te,j].sum())} for j,s in enumerate(SPECIES)]);print('completed',n,seed,'fold',outer,flush=True)
    results=[];per=[];recall=[]
    for stage,p in stages.items():
        results.append({'cohort':n,'seed':seed,'stage':stage,**f1_metrics(y,p)});np.savez_compressed(d/(stage+'.npz'),truth=y,prediction=p,score=stage_scores[stage],fold=fold,configuration=data['configuration'][:n],species=SPECIES)
        pr,re,ff,su=precision_recall_fscore_support(y,p,zero_division=0);per.extend([{'cohort':n,'seed':seed,'stage':stage,'species':s,'precision':pr[j],'recall':re[j],'f1':ff[j],'support':int(su[j]),'FP':int(((y[:,j]==0)&(p[:,j]==1)).sum()),'FN':int(((y[:,j]==1)&(p[:,j]==0)).sum())} for j,s in enumerate(SPECIES)]);recall.extend([{'cohort':n,'seed':seed,**r} for r in single_mixed(y,p,stage)])
    np.savez_compressed(d/'candidate_oof.npz',truth=y,prediction=all_pred,score=all_score,fold=fold,candidates=np.array(CANDIDATES,dtype=str));pd.DataFrame(results).to_csv(d/'metrics.csv',index=False);pd.DataFrame(per).to_csv(d/'per_species.csv',index=False);pd.DataFrame(recall).to_csv(d/'single_mixed_recall.csv',index=False);pd.DataFrame(rows).to_csv(d/'selections.csv',index=False);pd.DataFrame(rankings).to_csv(d/'inner_ranking.csv',index=False);pd.DataFrame(supports).to_csv(d/'support.csv',index=False);(d/'saved_models.json').write_text(json.dumps(saved,indent=2)+'\n');return results


def run(data_root,output_dir,split_manifest=None):
    repo=Path(__file__).resolve().parents[1];out=Path(output_dir)
    if data_root and Path(data_root).resolve()!=(repo/'study/retraining_20261022/refit/dataset').resolve():raise ValueError('This paired study is pinned to the audited 87/90 snapshot; do not silently substitute another cohort')
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty experiment directory')
    out.mkdir(parents=True,exist_ok=True)
    for n in [87,90]:
        for seed in SEEDS:evaluate(repo,out,n,seed)
    return {'status':'success','recipe':'fixed_top_two_v4','output_dir':str(out),'trace':str(out/'protocol.json'),'metrics':'per-cohort/seed metrics.csv; no winning seed selected','prediction_replay_identical':True}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',required=True);p.add_argument('--cohort',type=int,choices=[87,90]);p.add_argument('--seed',type=int,choices=SEEDS);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if a.cohort and a.seed:evaluate(repo,a.output_dir,a.cohort,a.seed)
    else:print(run(None,a.output_dir))
