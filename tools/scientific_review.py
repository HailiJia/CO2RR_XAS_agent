"""Reviewable predictions from extension bundles; no structural predictors."""
import numpy as np
from .scientific_extensions import scores,calibrated

def predict_with_review(bundle,raw,energy,*,parent_group=None,depth_layer=5):
    energy=np.asarray(energy,float);expected=np.asarray(bundle['energy'])
    if energy.shape!=expected.shape or not np.allclose(energy,expected,atol=1e-7,rtol=0):
        return {'status':'review_required','reasons':['outside trained energy grid; do not extrapolate'],'predictions':[]}
    score=calibrated(bundle['calibrators'],scores(bundle,raw));supported=np.asarray(bundle['calibration_supported']);thresholds=bundle['thresholds'];species=bundle['species']
    domain=[]
    if depth_layer!=5:domain.append('outside trained surface-layer representation')
    if parent_group is None:domain.append('parent domain unknown; ancestral-parent transfer not established')
    elif parent_group not in bundle['training_parent_groups']:domain.append('unseen exact substrate geometry group; ancestral-parent transfer not established')
    rows=[]
    for s in score:
        identities=[str(n) for n,p,t in zip(species,s,thresholds) if p>=t]
        rows.append({'identity_set':identities,'scores':dict(zip(species,map(float,s))),'thresholds':dict(zip(species,map(float,thresholds))),
            'unsupported_calibration_labels':[str(n) for n,ok in zip(species,supported) if not ok],
            'supported_minimum_margin':float(np.min(abs(s[supported]-thresholds[supported]))) if supported.any() else None,
            'review_required':bool(domain or not supported.all()),'domain_flags':domain,'probability_caveat':'inner-training calibrated where supported; no certified full-set risk'})
    return {'status':'success','predictions':rows,'structural_metadata_used_as_predictors':False}
