"""Versioned, small-sample nested XAS experiments. Spectra are the only predictors.

Frozen primary predictions are comparisons, never replaced. All selection,
compression, calibration and thresholds occur inside the outer training set.
"""
from pathlib import Path
import hashlib,json,sys,time
import numpy as np
import pandas as pd
import joblib
from scipy.fft import dct
from scipy.integrate import trapezoid,cumulative_trapezoid
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score,accuracy_score,precision_recall_fscore_support
from .spectral_multilabel import SPECIES,spectrum_layers
from .study_cohort import load_cohort
from .chemical_lineage import read_poscar,parent_key

SEED=20261022
FEATURES=['normalized','CDF','derivative','descriptors','DCT8','DCT16','DCT24','DCT32','PCA8','PCA16']
FAMILIES=['nearest_neighbor','independent_trees','classifier_chain','xgboost']
THRESHOLDS=[.15,.25,.35,.45,.5,.55,.65,.75]
WINDOWS=[(8976,8985),(8985,8998),(8998,9012),(9012,9027)]


def metrics(y,p):
    return {'micro_f1':float(f1_score(y,p,average='micro',zero_division=0)),
            'macro_f1':float(f1_score(y,p,average='macro',zero_division=0)),
            'exact_set_accuracy':float(accuracy_score(y,p))}


def transform_spectra(raw,energy,scenario='clean',indices=None):
    """Crop first-principles support, never extrapolate. Positive Gaussian kernel."""
    raw=np.asarray(raw);ids=np.arange(len(raw)) if indices is None else np.asarray(indices)
    sigma=float(scenario.split(':')[1]) if scenario.startswith('gaussian:') else .5 if scenario=='augmentation' else 0.
    work=raw.copy()
    if sigma:
        work=gaussian_filter1d(work,sigma/(energy[1]-energy[0]),axis=1,mode='constant',cval=0.,truncate=5)/gaussian_filter1d(np.ones_like(work),sigma/(energy[1]-energy[0]),axis=1,mode='constant',cval=0.,truncate=5)
    target=energy[(energy>=8976-1e-7)&(energy<=9027+1e-7)]
    out=[]
    for i,(row,gid) in enumerate(zip(work,ids)):
        rng=np.random.default_rng(SEED+int(gid)*101+sum(map(ord,scenario)))
        shift=float(scenario.split(':')[1]) if scenario.startswith('shift:') else float(rng.choice([-.5,.5])) if scenario=='augmentation' else 0.
        query=target-shift
        if query.min()<energy.min()-1e-7 or query.max()>energy.max()+1e-7:raise ValueError('Perturbation would extrapolate')
        y=np.interp(query,energy,row)
        noise=float(scenario.split(':')[1]) if scenario.startswith('noise:') else .02 if scenario=='augmentation' else 0.
        if noise:y=y+rng.normal(0,noise*np.max(np.abs(row)),len(y))
        out.append(y)
    return np.asarray(out),target


def features(raw,e,name):
    z=raw-raw.min(axis=1,keepdims=True);area=trapezoid(z,e,axis=1)
    if np.any(area<=1e-15):raise ValueError('Flat spectral predictor')
    z=z/area[:,None];der=savgol_filter(z,11,3,deriv=1,delta=e[1]-e[0],axis=1)
    if name=='normalized' or name.startswith('PCA'):return z
    if name=='derivative':return der
    if name=='CDF':return cumulative_trapezoid(z,e,axis=1,initial=0)
    if name.startswith('DCT'):return dct(z,type=2,norm='ortho',axis=1)[:,:int(name[3:])]
    if name=='descriptors':
        blocks=[]
        for lo,hi in WINDOWS:
            mask=(e>=lo-1e-7)&(e<=hi+1e-7);x=e[mask];a=z[:,mask];d=der[:,mask];w=trapezoid(a,x,axis=1)
            blocks.extend([w,trapezoid(a*x,x,axis=1)/np.maximum(w,1e-15)-8979,
                           d.mean(axis=1),d.max(axis=1),d.min(axis=1),x[a.argmax(axis=1)]-8979,a.max(axis=1)])
        return np.column_stack(blocks)
    raise ValueError(name)


def candidates(family):
    if family=='nearest_neighbor':return [{'k':k,'distance':m} for k in [1,3,5] for m in ['cosine','euclidean']]
    return [{'leaf':2,'order':'canonical','depth':2,'rounds':48,'regularization':5},
            {'leaf':4,'order':'training_prevalence','depth':3,'rounds':80,'regularization':10}]


def fit_target(x,y,family,cfg):
    if len(np.unique(y))<2:return int(y[0])
    if family=='xgboost':
        from xgboost import XGBClassifier
        model=XGBClassifier(n_estimators=cfg['rounds'],max_depth=cfg['depth'],learning_rate=.05,min_child_weight=3,
            reg_lambda=cfg['regularization'],reg_alpha=.1,subsample=.8,colsample_bytree=.8,n_jobs=1,random_state=SEED,
            scale_pos_weight=min(10.,(len(y)-y.sum())/y.sum()),tree_method='hist',eval_metric='logloss')
    else:model=ExtraTreesClassifier(n_estimators=32,min_samples_leaf=cfg['leaf'],class_weight='balanced',max_features='sqrt',n_jobs=1,random_state=SEED)
    return model.fit(x,y)


def probability(model,x):
    return np.full(len(x),float(model)) if isinstance(model,int) else model.predict_proba(x)[:,list(model.classes_).index(1)]


def fit_estimator(x,y,groups,family,cfg):
    if family=='nearest_neighbor':return {'nn':NearestNeighbors(n_neighbors=min(cfg['k'],len(x)),metric=cfg['distance']).fit(x),'y':y}
    order=list(range(y.shape[1])) if cfg['order']=='canonical' else np.argsort(-y.sum(axis=0),kind='stable').tolist()
    predecessor=None
    if family=='classifier_chain':
        # Group-out-of-fold predecessor proxies: independent generators see no
        # labels from the query group. Final inference proceeds sequentially.
        # This explicit proxy-chain variant avoids in-sample or true-label features.
        predecessor=np.zeros(y.shape,float)
        for tr,va in GroupKFold(3).split(x,y,groups):
            assert not set(groups[tr])&set(groups[va])
            for j in range(y.shape[1]):predecessor[va,j]=probability(fit_target(x[tr],y[tr,j],'independent_trees',cfg),x[va])
    models=[]
    for k,j in enumerate(order):
        augmented=np.column_stack([x,predecessor[:,order[:k]]]) if predecessor is not None and k else x
        models.append(fit_target(augmented,y[:,j],family,cfg))
    return {'models':models,'order':order,'chain':predecessor is not None}


def estimator_scores(model,x):
    if 'nn' in model:
        _,idx=model['nn'].kneighbors(x);return model['y'][idx].mean(axis=1)
    score=np.zeros((len(x),len(model['order'])))
    for k,(j,m) in enumerate(zip(model['order'],model['models'])):
        augmented=np.column_stack([x,score[:,model['order'][:k]]]) if model['chain'] and k else x
        score[:,j]=probability(m,augmented)
    return score


def fit_recipe(raw,y,groups,e,feature,family,cfg,augmented=False,ids=None):
    clean,grid=transform_spectra(raw,e,indices=ids);x=features(clean,grid,feature)
    if augmented:
        extra,_=transform_spectra(raw,e,'augmentation',ids);x=np.row_stack([x,features(extra,grid,feature)]);y=np.row_stack([y,y]);groups=np.r_[groups,groups]
    scaler=StandardScaler().fit(x) if family=='nearest_neighbor' or feature.startswith('PCA') else None
    if scaler is not None:x=scaler.transform(x)
    pca=PCA(n_components=min(int(feature[3:]),len(x)-1,x.shape[1]),svd_solver='full').fit(x) if feature.startswith('PCA') else None
    if pca is not None:x=pca.transform(x)
    return {'model':fit_estimator(x,y,groups,family,cfg),'scaler':scaler,'pca':pca,'feature':feature,'family':family,'config':cfg,'grid':grid,'energy':e}


def scores(bundle,raw,scenario='clean',ids=None):
    a,e=transform_spectra(raw,bundle['energy'],scenario,ids);x=features(a,e,bundle['feature'])
    if bundle['scaler'] is not None:x=bundle['scaler'].transform(x)
    if bundle['pca'] is not None:x=bundle['pca'].transform(x)
    return estimator_scores(bundle['model'],x)


def thresholds(y,s):
    t=max(THRESHOLDS,key=lambda t:f1_score(y,s>=t,average='micro',zero_division=0));result=np.full(y.shape[1],t)
    for j in range(y.shape[1]):
        if min(y[:,j].sum(),len(y)-y[:,j].sum())>=5:
            result[j]=max(THRESHOLDS,key=lambda t:f1_score(y[:,j],s[:,j]>=t,zero_division=0))
    return result


def calibrate(y,s):
    models=[];out=s.copy();supported=[]
    for j in range(y.shape[1]):
        ok=min(y[:,j].sum(),len(y)-y[:,j].sum())>=5 and np.std(s[:,j])>1e-10;supported.append(bool(ok))
        if ok:
            m=LogisticRegression(C=1,random_state=SEED).fit(s[:,j,None],y[:,j]);out[:,j]=m.predict_proba(s[:,j,None])[:,1];models.append(m)
        else:models.append(None)
    return models,out,np.asarray(supported)


def calibrated(models,s):
    return np.column_stack([s[:,j] if m is None else m.predict_proba(s[:,j,None])[:,1] for j,m in enumerate(models)])


def single_mixed(y,p,name):
    rows=[]
    for j,s in enumerate(SPECIES):
        for kind,mask in [('single',y.sum(axis=1)==1),('mixed',y.sum(axis=1)>1)]:
            positive=mask&(y[:,j]==1);n=int(positive.sum());hit=int(p[positive,j].sum());rows.append({'model':name,'species':s,'kind':kind,'support':n,'detected':hit,'recall':hit/n if n else None})
    return rows


def run(data_root,output_dir,split_manifest=None):
    repo=Path(__file__).resolve().parents[1];out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('New empty output directory required')
    audit=json.loads((repo/'study/extensions_v2/chemical_audit/summary.json').read_text())
    if audit['demonstrated_retained_label_mismatches'] or audit['ambiguous_retained']:raise ValueError('Resolve labels before fitting')
    split_path=Path(split_manifest) if split_manifest else repo/'study/manuscript_package/figure_data/cv_manifest.csv'
    cohort=load_cohort(data_root,split_path);split=cohort['split'];y=cohort['Y'];groups=cohort['groups'];e=cohort['energy'];folds=split.fold.to_numpy()
    # Physical population weighting occurs before per-configuration normalization.
    raw=np.asarray([spectrum_layers(r)[5][:,1] for r in cohort['records']])
    parents=[]
    for row in split.itertuples():parents.append(parent_key(read_poscar(Path(data_root)/row.configuration/'POSCAR')))
    split=split.copy();split['initial_metal_geometry_group']=parents;split.to_csv(out/'domain_inventory.csv',index=False)
    np.savez_compressed(out/'cohort_spectra.npz',raw=raw,energy=e,truth=y,fold=folds,groups=groups,configuration=split.configuration.to_numpy(dtype=str),species=SPECIES)
    scenarios=['clean']+[f'shift:{s}' for s in [-2.,-1.,-.5,-.2,.2,.5,1.,2.]]+[f'gaussian:{s}' for s in [.25,.5,1.]]+[f'noise:{s}' for s in [.01,.02,.05]]
    protocol={'version':'scientific-extensions-v2','seed':SEED,'features':FEATURES,'descriptor_windows_eV':WINDOWS,'descriptor_window_source':'fixed edge-relative near/post-edge intervals; frozen before reading outer-test scores, not selected from Figure 3c',
        'families':{f:candidates(f) for f in FAMILIES},'chain_variant':'sequential probability chain trained on group-OOF independent predecessor proxies; not true-label/in-sample predecessor features',
        'outer_folds':'frozen 87-configuration cv_manifest.csv','inner_folds':'GroupKFold(3) within each outer-training partition','early_stopping':'none; compact fixed round counts selected inside training',
        'calibration':'Platt logistic on inner grouped OOF scores; unavailable with <5 positives or negatives or constant scores','thresholds':THRESHOLDS,
        'stress_scenarios':scenarios,'augmentation':'one additional combined ±0.5 eV shift, sigma0.5 eV Gaussian and 2% peak noise per training configuration; no new independent positives',
        'common_predictor_window_eV':[8976,9027],'native_window_eV':[float(e.min()),float(e.max())],'crop_reason':'all ±2 eV tests have complete support; no extrapolation',
        'structural_metadata_in_predictors':False,'prospective_confirmation':'not available; repeatedly examined folds; reserve new configurations before inspection',
        'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),
        'dependencies':{'python':sys.version,'numpy':np.__version__}}
    import sklearn,scipy,xgboost
    protocol['dependencies'].update(sklearn=sklearn.__version__,scipy=scipy.__version__,xgboost=xgboost.__version__)
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    frozen=np.load(repo/'study/manuscript_package/figure_data/oof_independent_selected_micro.npz');assert np.array_equal(frozen['truth'],y)
    recall=single_mixed(y,frozen['prediction'],'frozen_primary');pd.DataFrame(recall).to_csv(out/'primary_single_mixed_recall.csv',index=False)
    cross=pd.read_csv(repo/'study/manuscript_package/figure_data/primary_single_mixed_recall.csv')
    assert [(r['support'],r['detected']) for r in recall]==list(zip(cross.support,cross.detected))
    reports=[{'model':'frozen_primary',**metrics(y,frozen['prediction'])}];selection=[];ablation=[];missing=[];stress=[];per_species=[];neighbors=[];coverage=[];allpred={'frozen_primary':frozen['prediction']};allscore={}
    for family in FAMILIES:
        for augment in [False,True]:
            name=family+('_augmented' if augment else '');oof=np.zeros(y.shape,int);oofs=np.zeros(y.shape,float)
            feature_pred={f:np.zeros(y.shape,int) for f in FEATURES}
            for outer in sorted(set(folds)):
                tr=np.flatnonzero(folds!=outer);te=np.flatnonzero(folds==outer);rank=[];inner=list(GroupKFold(3).split(raw[tr],y[tr],groups[tr]))
                for i,(it,iv) in enumerate(inner):missing.append({'model':name,'outer_fold':int(outer),'partition':f'inner_{i}','positive_support':y[tr[it]].sum(axis=0).tolist(),'missing_positive':[SPECIES[j] for j in np.flatnonzero(y[tr[it]].sum(axis=0)==0)]})
                for feature in FEATURES:
                    best=None
                    for cfg in candidates(family):
                        ss=np.zeros((len(tr),y.shape[1]))
                        for it,iv in inner:
                            model=fit_recipe(raw[tr[it]],y[tr[it]],groups[tr[it]],e,feature,family,cfg,augment,tr[it]);ss[iv]=scores(model,raw[tr[iv]],ids=tr[iv])
                        cal,cs,supported=calibrate(y[tr],ss);ts=thresholds(y[tr],cs);val=metrics(y[tr],cs>=ts)['micro_f1']
                        row={'feature':feature,'family':family,'config':cfg,'inner_micro_f1':val,'thresholds':ts,'calibrators':cal,'supported':supported}
                        if best is None or val>best['inner_micro_f1']:best=row
                    rank.append(best)
                    # A fixed-feature outer result is an ablation, not an outer-score selector.
                    fm=fit_recipe(raw[tr],y[tr],groups[tr],e,feature,family,best['config'],augment,tr)
                    feature_pred[feature][te]=calibrated(best['calibrators'],scores(fm,raw[te],ids=te))>=best['thresholds']
                win=max(rank,key=lambda r:r['inner_micro_f1']);model=fit_recipe(raw[tr],y[tr],groups[tr],e,win['feature'],family,win['config'],augment,tr)
                clean=calibrated(win['calibrators'],scores(model,raw[te],ids=te));oofs[te]=clean;oof[te]=clean>=win['thresholds']
                model.update(calibrators=win['calibrators'],thresholds=win['thresholds'],calibration_supported=win['supported'],training_indices=tr,heldout_indices=te,training_parent_groups=sorted(set(np.asarray(parents)[tr])),species=SPECIES)
                filename=out/f'{name}_fold{outer}.joblib';joblib.dump(model,filename,compress=3);loaded=joblib.load(filename)
                assert np.array_equal(calibrated(loaded['calibrators'],scores(loaded,raw[te],ids=te)),clean)
                selection.append({'model':name,'outer_fold':int(outer),'feature':win['feature'],'config':json.dumps(win['config']),'thresholds':json.dumps(win['thresholds'].tolist()),'inner_micro_f1':win['inner_micro_f1'],'calibration_supported':json.dumps(win['supported'].tolist()),'missing_positive_training':json.dumps([SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]),'model_sha256':hashlib.sha256(filename.read_bytes()).hexdigest(),'save_reload_identical':True})
                for scenario in scenarios:
                    ss=calibrated(win['calibrators'],scores(loaded,raw[te],scenario,te));stress.append({'model':name,'outer_fold':int(outer),'scenario':scenario,'indices':te.tolist(),'prediction':(ss>=win['thresholds']).astype(int).tolist()})
                # Review policy: only supported probabilities contribute a margin; sparse labels remain explicitly unresolved.
                support=win['supported'];margin=np.min(abs(clean[:,support]-win['thresholds'][support]),axis=1) if support.any() else np.zeros(len(te))
                for cutoff in [0.,.05,.1,.15,.2,.3]:
                    accept=(margin>=cutoff)&~np.any((clean>=win['thresholds'])&~support,axis=1)
                    coverage.append({'model':name,'fold':int(outer),'margin':cutoff,'accepted':int(accept.sum()),'total':len(te),'exact_errors':int(np.any(oof[te][accept]!=y[te][accept],axis=1).sum()),'unsupported_labels':json.dumps([SPECIES[j] for j in np.flatnonzero(~support)]),'certified_full_set_risk':False})
                # Transparent nearest examples are reviewed in the selected spectral representation only.
                if family=='nearest_neighbor':
                    clean_raw,grid=transform_spectra(raw[te],e,indices=te);xx=features(clean_raw,grid,model['feature']);xx=model['scaler'].transform(xx)
                    if model['pca'] is not None:xx=model['pca'].transform(xx)
                    distances,ii=model['model']['nn'].kneighbors(xx,n_neighbors=5)
                    for query,dd,nn in zip(te,distances,ii):
                        for k,(dist,idx) in enumerate(zip(dd,nn)):neighbors.append({'model':name,'query':split.iloc[query].configuration,'fold':int(outer),'rank':k+1,'training_configuration':split.iloc[tr[idx%len(tr)]].configuration,'augmented_example':bool(idx>=len(tr)),'distance':float(dist),'distance_kind':model['config']['distance']})
                print(name,'fold',outer,'selected',win['feature'],flush=True)
            reports.append({'model':name,**metrics(y,oof)});allpred[name]=oof;allscore[name]=oofs;recall+=single_mixed(y,oof,name)
            pr,re,ff,su=precision_recall_fscore_support(y,oof,zero_division=0)
            per_species.extend([{'model':name,'species':s,'precision':pr[j],'recall':re[j],'f1':ff[j],'support':int(su[j])} for j,s in enumerate(SPECIES)])
            for feature,pred in feature_pred.items():
                ablation.append({'model':name,'feature':feature,**metrics(y,pred)});np.savez_compressed(out/f'ablation_{name}_{feature}.npz',truth=y,prediction=pred,fold=folds)
            np.savez_compressed(out/f'oof_{name}.npz',truth=y,prediction=oof,score=oofs,fold=folds,species=SPECIES,configuration=split.configuration.to_numpy(dtype=str))
            pd.DataFrame(reports).to_csv(out/'model_comparison.csv',index=False)
    pd.DataFrame(selection).to_csv(out/'selections.csv',index=False);pd.DataFrame(ablation).to_csv(out/'feature_ablation.csv',index=False);pd.DataFrame(per_species).to_csv(out/'per_species.csv',index=False);pd.DataFrame(recall).to_csv(out/'single_mixed_recall.csv',index=False);pd.DataFrame(neighbors).to_csv(out/'nearest_training_examples.csv',index=False);pd.DataFrame(coverage).to_csv(out/'review_policy.csv',index=False)
    (out/'missing_positive_training.json').write_text(json.dumps(missing,indent=2)+'\n');(out/'stress_heldout_predictions.json').write_text(json.dumps(stress)+'\n')
    sr=[]
    for name in allpred:
        if name=='frozen_primary':continue
        for scenario in scenarios:
            pred=np.zeros_like(y)
            for row in stress:
                if row['model']==name and row['scenario']==scenario:pred[row['indices']]=row['prediction']
            sr.append({'model':name,'scenario':scenario,**metrics(y,pred)})
    pd.DataFrame(sr).to_csv(out/'robustness.csv',index=False)
    paired=[];rng=np.random.default_rng(SEED)
    for name,pred in allpred.items():
        if name=='frozen_primary':continue
        ds=np.asarray([[metrics(y[ix],pred[ix])[key]-metrics(y[ix],allpred['frozen_primary'][ix])[key] for key in ['micro_f1','macro_f1','exact_set_accuracy']] for ix in [rng.integers(0,len(y),len(y)) for _ in range(500)]])
        for j,key in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):paired.append({'model':name,'metric':key,'difference':metrics(y,pred)[key]-metrics(y,allpred['frozen_primary'])[key],'conditional_95_low':np.quantile(ds[:,j],.025),'conditional_95_high':np.quantile(ds[:,j],.975),'uncertainty':'paired configuration bootstrap of fixed OOF predictions, excludes refitting/selection uncertainty'})
    pd.DataFrame(paired).to_csv(out/'paired_uncertainty.csv',index=False)
    # Harder protocol: observed initial metal geometries, not verified ancestral parents.
    hard=np.zeros_like(y);hard_rows=[];parent_array=np.asarray(parents)
    for k,(tr,te) in enumerate(GroupKFold(3).split(raw,y,parent_array)):
        assert not set(parent_array[tr])&set(parent_array[te]);ss=np.zeros((len(tr),y.shape[1]));cfg=candidates('independent_trees')[0]
        for it,iv in GroupKFold(3).split(raw[tr],y[tr],groups[tr]):
            m=fit_recipe(raw[tr[it]],y[tr[it]],groups[tr[it]],e,'DCT16','independent_trees',cfg);ss[iv]=scores(m,raw[tr[iv]])
        ts=thresholds(y[tr],ss);m=fit_recipe(raw[tr],y[tr],groups[tr],e,'DCT16','independent_trees',cfg);hard[te]=scores(m,raw[te])>=ts
        hard_rows.append({'fold':k,'train':tr.tolist(),'test':te.tolist(),'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)]})
    (out/'harder_protocol.json').write_text(json.dumps({'grouping':'exact initial substrate coordinates only; independent ancestral-parent transfer not established','groups':len(set(parents)),'metrics':metrics(y,hard),'splits':hard_rows},indent=2)+'\n');np.savez_compressed(out/'harder_oof.npz',truth=y,prediction=hard)
    learning=[]
    for outer in sorted(set(folds)):
        whole=np.flatnonzero(folds!=outer);te=np.flatnonzero(folds==outer)
        for seed in [11,23,47]:
            shuffled=np.random.default_rng(seed).permutation(whole)
            for fraction in [.25,.5,.75,1.]:
                tr=shuffled[:max(6,int(len(whole)*fraction))];m=fit_recipe(raw[tr],y[tr],groups[tr],e,'DCT16','independent_trees',candidates('independent_trees')[0]);ss=scores(m,raw[te]);pred=ss>=.5
                learning.append({'outer_fold':int(outer),'seed':seed,'training_configurations':len(tr),'fraction':fraction,'recipe':'predeclared DCT16, independent trees leaf2, threshold0.5; no full-training hyperparameters reused',**metrics(y[te],pred),'missing_positive_training':json.dumps([SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)])})
    pd.DataFrame(learning).to_csv(out/'learning_curves.csv',index=False)
    # Priorities do not imply completed or improved calculations.
    priorities=[{'rank':1,'type':'scientific_priority_control','configuration':'matched relaxed clean parents and two-CO/intact-OCCO controls','reason':'isolate actual chemical contrasts; missing spectra are not repaired by ML'},
                {'rank':2,'type':'scientific_priority_control','configuration':'matched CO/H mixtures at fixed parent/site/loading','reason':'frozen H mixture recall 0/8 versus 14/14 single H'},
                {'rank':3,'type':'scientific_priority_control','configuration':'new independent hydrogenated C1/C2 configurations; lock confirmation subset','reason':'rare supports and missing-positive training folds'}]
    disagreement=np.std(np.stack(list(allscore.values())),axis=0).mean(axis=1)
    for i in np.argsort(-disagreement)[:5]:priorities.append({'rank':len(priorities)+1,'type':'heldout_model_disagreement_review','configuration':split.iloc[i].configuration,'reason':f'OOF score disagreement {disagreement[i]:.4f}; existing case review, not a new independent positive'})
    (out/'next_actions.json').write_text(json.dumps(priorities,indent=2)+'\n')
    trace={'action':'audited_training','recipe':'scientific_extensions_v2','data':'available local export; sanitized paths','label_audit':audit,'configuration_count':len(y),'model_count':40,'all_saved_models_reloaded_identically':True,'spectral_predictors_only':True,'no_HPC_or_portal_calls':True,'source_sha256':protocol['source_hash'],'protocol_sha256':hashlib.sha256((out/'protocol.json').read_bytes()).hexdigest(),'results':reports}
    (out/'invocation_trace.json').write_text(json.dumps(trace,indent=2)+'\n');return {'status':'success','recipe':'scientific_extensions_v2','output_dir':str(out),'metrics':reports,'trace':str(out/'invocation_trace.json')}
