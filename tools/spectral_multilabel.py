"""Species-presence inference, separate from categorical classification.

Strict grids; fixed class order; only spectra enter features. The published
nested-selector OOF result remains distinct from a post-hoc fixed winner.
"""
from collections import Counter
import hashlib,json,platform
from pathlib import Path
import numpy as np
from scipy.fft import dct
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier,RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import sklearn,scipy,joblib
from .layer_population import aggregate_layers

SPECIES=('CHO','CO','CO2','COCHO','COH','COOH','H','O','OCCOH','OCHO','OH')
DEPTHS={'layer_5':[5],'layer_4':[4],'layer_3':[3],'layer_2':[2],'top_2':[5,4],'top_3':[5,4,3],'all_4':[5,4,3,2]}
SEED=20261020

def identity_set(value):
    if isinstance(value,dict):result={str(k) for k,v in value.items() if int(v)>0}
    elif isinstance(value,(list,tuple,set)):result=set(map(str,value))
    elif isinstance(value,str):
        # Only known atomic species tokens separated by explicit legacy delimiters.
        result={'H'} if value=='H_0p75' else {'CO'} if value=='2CO_adjacent' else set(value.replace('+','_').split('_')) if value else set()
    elif value is None:raise ValueError('Missing identity evidence')
    else:raise ValueError('Unsupported species-presence target')
    result-={'clean','plain','none'}
    unknown=result-set(SPECIES)
    if unknown:raise ValueError('Unseen/unsupported species: '+', '.join(sorted(unknown)))
    return tuple(sorted(result))

def record_identity(record):
    c=record.get('sample',{}).get('composition',{})
    for field in ['adsorbate_counts','adsorbate_identities','adsorbate_identity']:
        if field in c:return identity_set(c[field])
    sample=record.get('sample',{});value=sample.get('ml_labels',{}).get('adsorbate_identity')
    audit=sample.get('structure_descriptors',{}).get('component_audit',{})
    if audit.get('status')=='verified':return identity_set(audit['counts'])
    return identity_set(value)

def encode_labels(labels,species=SPECIES):
    if len(species)!=len(set(species)):raise ValueError('Duplicate species ordering')
    out=np.zeros((len(labels),len(species)),int)
    for i,value in enumerate(labels):
        keys=identity_set(value)
        if set(keys)-set(species):raise ValueError('Species missing from explicit output order')
        out[i]=[int(s in keys) for s in species]
    return out

def spectrum_layers(record):
    arrays={}
    for s in record['measurement']['series']:
        if not s['series_id'].startswith('Cu_K_layer_'):continue
        layer=int(s['series_id'].rsplit('_',1)[1])
        if layer in arrays:raise ValueError('Duplicate layer series')
        iv=s['independent_variables'][0];ch=s['channels'][0]
        if iv['unit']!='eV' or ch['name']!='absorption_cross_section':raise ValueError('Wrong energy unit/channel')
        a=np.column_stack([iv['values'],ch['values']]).astype(float)
        if len(a)<21 or not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0) or np.ptp(a[:,1])<=1e-8:raise ValueError('Invalid spectral arrays')
        arrays[layer]=a
    if set(arrays)!={2,3,4,5}:raise ValueError('Incomplete layers')
    if not all(np.array_equal(arrays[2][:,0],a[:,0]) for a in arrays.values()):raise ValueError('Incompatible layer grids')
    return arrays

def feature_bank(x,energy):
    x=np.atleast_2d(np.asarray(x,float));energy=np.asarray(energy,float)
    if x.shape[1]!=len(energy) or not np.isfinite(x).all() or not np.all(np.diff(energy)>0):raise ValueError('Invalid feature grid')
    step=np.diff(energy)
    if not np.allclose(step,step[0],atol=1e-8):raise ValueError('Uniform grid required for SG derivative recipe')
    z=x-x.min(axis=1,keepdims=True);area=np.trapezoid(z,energy,axis=1)
    if np.any(area<=0):raise ValueError('Nonpositive normalized area')
    z/=area[:,None];de=float(np.median(step))
    sm=savgol_filter(z,11,3,axis=1);d1=savgol_filter(z,11,3,deriv=1,delta=de,axis=1);d2=savgol_filter(z,11,3,deriv=2,delta=de,axis=1)
    cdf=cumulative_trapezoid(z,energy,axis=1,initial=0)
    return dict(normalized=z,CDF=cdf,derivative=d1,shape_CDF_derivatives=np.hstack([sm,cdf,d1,d2]),DCT24=dct(z,type=2,norm='ortho',axis=1)[:,:24])

def feature_for_records(records,populations,energy,depth,feature):
    layers=[spectrum_layers(r) for r in records]
    if any(not np.array_equal(a[2][:,0],energy) for a in layers):raise ValueError('Incompatible fitted energy grid; extrapolation disabled')
    if len(populations)!=len(records):raise ValueError('Missing represented populations')
    if depth!='depth_channels':
        raw=np.array([aggregate_layers(a,p,DEPTHS[depth])[1] for a,p in zip(layers,populations)])
        return feature_bank(raw,energy)[feature]
    stack=np.array([[a[l][:,1] for l in [2,3,4,5]] for a in layers]);pos=stack-stack.min(axis=2,keepdims=True)
    areas=np.trapezoid(pos[:,3,:],energy,axis=1)
    if np.any(areas<=0):raise ValueError('Invalid top layer area')
    scaled=pos/areas[:,None,None];de=float(np.median(np.diff(energy)))
    def block(z,w=11):return np.hstack([savgol_filter(z,w,3,axis=1),savgol_filter(z,w,3,deriv=1,delta=de,axis=1),savgol_filter(z,w,3,deriv=2,delta=de,axis=1)])
    if feature=='contrast':return np.hstack([block(scaled[:,3,:]),block(scaled[:,3,:]-scaled[:,2,:])])
    if feature=='multiscale':return np.hstack([block(scaled[:,3,:],w) for w in [7,11,21]])
    raise ValueError('Unknown depth feature')

def implementation_fingerprint():
    paths=[Path(__file__),Path(__file__).with_name('component_audit.py'),Path(__file__).with_name('layer_population.py'),Path(__file__).with_name('contact_checks.py'),Path(__file__).with_name('study_cohort.py'),Path(__file__).with_name('ml_splits.py')]
    import pandas as pd
    versions=dict(python=platform.python_version(),pandas=pd.__version__,numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__)
    return dict(code={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},dependencies=versions)

def fit_binary(X,Y,kind='independent_trees'):
    X=np.asarray(X,float);Y=np.asarray(Y,int)
    if Y.ndim!=2 or not np.isin(Y,[0,1]).all():raise ValueError('Binary multi-output targets required')
    scale=StandardScaler().fit(X) if kind in ['logistic','SVM'] else None
    a=scale.transform(X) if scale is not None else X
    models=[]
    if kind in ['extra_trees','random_forest']:
        cls=ExtraTreesClassifier if kind=='extra_trees' else RandomForestClassifier
        model=cls(n_estimators=96,min_samples_leaf=1 if kind=='extra_trees' else 2,class_weight='balanced' if kind=='extra_trees' else 'balanced_subsample',random_state=SEED,n_jobs=1).fit(a,Y)
        models=[model]
    else:
        for j in range(Y.shape[1]):
            if len(np.unique(Y[:,j]))<2:models.append(int(Y[0,j]));continue
            if kind=='independent_trees':m=ExtraTreesClassifier(n_estimators=64,max_features='sqrt',class_weight='balanced',min_samples_leaf=1,random_state=SEED,n_jobs=1)
            elif kind=='logistic':m=LogisticRegression(C=1,class_weight='balanced',solver='liblinear',max_iter=3000,random_state=SEED)
            elif kind=='SVM':m=SVC(C=3,class_weight='balanced',kernel='rbf',gamma='scale')
            else:raise ValueError('Unknown binary model')
            models.append(m.fit(a,Y[:,j]))
    return dict(models=models,scaler=scale,kind=kind,n_outputs=Y.shape[1],no_positive_training=np.flatnonzero(Y.sum(axis=0)==0).tolist())

def binary_scores(bundle,X):
    a=bundle['scaler'].transform(X) if bundle['scaler'] is not None else np.asarray(X)
    if bundle['kind'] in ['extra_trees','random_forest']:
        model=bundle['models'][0];probs=model.predict_proba(a)
        return np.column_stack([p[:,list(c).index(1)] if 1 in c else np.zeros(len(a)) for p,c in zip(probs,model.classes_)])
    return np.column_stack([np.full(len(a),m) if isinstance(m,int) else expit(m.decision_function(a)) if bundle['kind']=='SVM' else m.predict_proba(a)[:,1] for m in bundle['models']])

def f1_metrics(y,p):
    from sklearn.metrics import f1_score,accuracy_score
    return dict(micro_f1=float(f1_score(y,p,average='micro',zero_division=0)),macro_f1=float(f1_score(y,p,average='macro',zero_division=0)),exact_set_accuracy=float(accuracy_score(y,p)))

def choose_threshold(y,scores):
    options=sorted([.15,.25,.35,.45,.5,.55,.65,.75],key=lambda t:abs(t-.5))
    return max(options,key=lambda t:f1_metrics(y,scores>=t)['micro_f1'])

def select_training_recipe(Xs,Y,groups,kinds=('independent_trees',)):
    """Ranking/threshold selection consumes OUTER TRAINING data only."""
    groups=np.asarray(groups)
    if len(np.unique(groups))<3:raise ValueError('At least three independent training groups required')
    folds=list(GroupKFold(min(3,len(np.unique(groups)))).split(np.zeros(len(Y)),Y,groups));rank=[]
    for key,X in Xs.items():
        for kind in kinds:
            score=np.zeros(Y.shape,float)
            for tr,va in folds:
                if set(groups[tr])&set(groups[va]):raise ValueError('Group leakage')
                score[va]=binary_scores(fit_binary(X[tr],Y[tr],kind),X[va])
            threshold=choose_threshold(Y,score);rank.append(dict(recipe=key,kind=kind,threshold=threshold,inner_micro_f1=f1_metrics(Y,score>=threshold)['micro_f1']))
    return sorted(rank,key=lambda r:-r['inner_micro_f1'])

def save_model(path,fit,energy,species,depth,feature,thresholds,provenance):
    bundle=dict(format='co2rr-multilabel-v1',fit=fit,energy=np.asarray(energy),species=list(species),depth=depth,feature=feature,thresholds=np.asarray(thresholds),provenance=provenance,fingerprint=implementation_fingerprint(),normalization='raw population-weighted average; minimum-subtracted unit-area',prediction_function='tools.spectral_multilabel.predict_records')
    joblib.dump(bundle,path);return bundle

def predict_records(bundle,records,populations):
    if bundle.get('format')!='co2rr-multilabel-v1':raise ValueError('Unsupported model format')
    x=feature_for_records(records,populations,bundle['energy'],bundle['depth'],bundle['feature'])
    score=binary_scores(bundle['fit'],x)
    return score>=bundle['thresholds'],score

def rank_outer_training(Xs,Y,groups,train_indices,test_indices,kinds=('independent_trees',)):
    tr=np.asarray(train_indices,int);te=np.asarray(test_indices,int);groups=np.asarray(groups)
    if set(tr)&set(te) or set(groups[tr])&set(groups[te]):raise ValueError('Outer group leakage')
    if sorted(np.r_[tr,te].tolist())!=list(range(len(Y))):raise ValueError('Outer split must partition all rows')
    return select_training_recipe({k:v[tr] for k,v in Xs.items()},Y[tr],groups[tr],kinds)

def train_cohort_holdout(cohort,fold=0,depth='layer_5',features=('normalized','CDF','derivative'),kinds=('independent_trees',)):
    split=cohort['split'];tr=np.flatnonzero(split.fold.to_numpy()!=fold);te=np.flatnonzero(split.fold.to_numpy()==fold)
    Xs={f:feature_for_records(cohort['records'],cohort['populations'],cohort['energy'],depth,f) for f in features}
    ranking=rank_outer_training(Xs,cohort['Y'],cohort.get('groups',split.structure_hash.to_numpy()),tr,te,kinds);winner=ranking[0]
    fit=fit_binary(Xs[winner['recipe']][tr],cohort['Y'][tr],winner['kind']);score=binary_scores(fit,Xs[winner['recipe']][te]);pred=score>=winner['threshold']
    provenance=dict(selection='training-only grouped inner CV; new integration recipe, not archived nested benchmark',outer_fold=int(fold),training_geometry_hashes=split.iloc[tr].structure_hash.tolist(),test_geometry_hashes=split.iloc[te].structure_hash.tolist(),training_record_hashes=split.iloc[tr].record_sha256.tolist(),training_populations=[cohort['populations'][i] for i in tr])
    bundle=dict(format='co2rr-multilabel-v1',fit=fit,energy=cohort['energy'],species=list(SPECIES),depth=depth,feature=winner['recipe'],thresholds=np.full(len(SPECIES),winner['threshold']),provenance=provenance,fingerprint=implementation_fingerprint(),normalization='raw population-weighted average before unit-area normalization',prediction_function='tools.spectral_multilabel.predict_records')
    return dict(bundle=bundle,metrics=f1_metrics(cohort['Y'][te],pred),prediction=pred,score=score,train_indices=tr,test_indices=te,ranking=ranking,missing_positive_training=[SPECIES[j] for j in fit['no_positive_training']])

def validate_fit_cache(path,train_indices,test_indices,fingerprint):
    with np.load(path) as z:
        if 'fingerprint' not in z or z['fingerprint'].item()!=fingerprint:raise ValueError('Cached implementation/dependency fingerprint mismatch')
        if not np.array_equal(z['train_indices'],train_indices) or not np.array_equal(z['test_indices'],test_indices):raise ValueError('Cached train/test partition mismatch')
        if set(map(int,train_indices))&set(map(int,test_indices)):raise ValueError('Cached train/test leakage')
        if not np.isfinite(z['inner_score']).all() or not np.isfinite(z['test_score']).all():raise ValueError('Nonfinite cached predictions')
        return z['inner_score'].copy(),z['test_score'].copy()
