"""Signed clean-referenced XAS on an explicit common physical Cu scale.

The physical difference and feature bank preserve amplitude. Optional signed
shape normalization is an identity-only control in differential_identity; never
use it for absolute coverage. Source weighting is audited before subtraction.
"""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy.fft import dct
from scipy.integrate import cumulative_trapezoid
from scipy.signal import savgol_filter

VERSION='signed-differential-v1'
PRE=(8974.,8977.);POST=(9018.,9025.)
class ReferenceError(ValueError):pass

def _series(record,populations,top_layers,energy):
    arrays={}
    correction=record.get('source_scale_correction',{})
    if sorted(correction.get('physical_top_layers',[]))!=sorted(top_layers):raise ReferenceError('Physical top-layer definition mismatch')
    if {int(k):int(v) for k,v in correction.get('populations',{}).items()}!={int(k):int(populations[k]) for k in top_layers if k in populations}:raise ReferenceError('Verified physical populations do not match record provenance')
    if correction.get('isotropic_cross_section_unit')!='Mbarn_per_Cu' or not correction.get('native_reconstruction_verified'):raise ReferenceError('Audited per-Cu source scale/reconstruction required')
    for s in record['measurement']['series']:
        l=int(s['series_id'].rsplit('_',1)[1]);iv=s['independent_variables'][0];ch=s['channels'][0]
        if iv['unit']!='eV' or ch['unit']!='Mbarn_per_Cu':raise ReferenceError('Incompatible energy/intensity units')
        arrays[l]=np.column_stack([iv['values'],ch['values']])
    if any(l not in arrays for l in top_layers):raise ReferenceError('Missing required physical top layer; no substitution')
    if any(l not in populations or populations[l]<=0 for l in top_layers):raise ReferenceError('Verified physical populations required')
    result=[]
    for l in top_layers:
        a=arrays[l]
        if not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0) or a[0,0]>energy[0]+1e-8 or a[-1,0]<energy[-1]-1e-8:raise ReferenceError('Invalid or insufficient energy support; no extrapolation')
        result.append(np.interp(energy,a[:,0],a[:,1]))
    return sum(v*populations[l] for v,l in zip(result,top_layers))/sum(populations[l] for l in top_layers)

def clean_scale(clean,energy,pre=PRE,post=POST):
    e=np.asarray(energy);pre_mask=(e>=pre[0])&(e<=pre[1]);post_mask=(e>=post[0])&(e<=post[1])
    if pre[0]<e.min()-1e-8 or post[1]>e.max()+1e-8 or min(pre_mask.sum(),post_mask.sum())<3:raise ReferenceError('Insufficient predeclared clean scaling windows')
    step=float(np.mean(clean[post_mask])-np.mean(clean[pre_mask]))
    if not np.isfinite(step) or step<=0:raise ReferenceError('Invalid clean reference edge-step scale')
    return step

def difference(adsorbed,clean,scale):
    a,b=np.asarray(adsorbed,float),np.asarray(clean,float)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(scale) or scale<=0:raise ReferenceError('Invalid paired arrays/scale')
    return (a-b)/scale

def pair(sample,reference,energy,populations,reference_populations,top_layers,domain,reference_domain):
    if reference is None:raise ReferenceError('A substrate-specific clean reference is required')
    if domain!=reference_domain:raise ReferenceError('Cross-facet/composition reference substitution is forbidden')
    if any(r.get('source_scale_correction',{}).get('physical_domain')!=domain for r in [sample,reference]):raise ReferenceError('Declared facet/domain disagrees with audited record provenance')
    c=reference.get('sample',{}).get('composition',{})
    if 'adsorbate_counts' not in c or any(c['adsorbate_counts'].values()):raise ReferenceError('Reference must be explicitly clean; covered/unknown reference rejected')
    e=np.asarray(energy,float);a=_series(sample,populations,top_layers,e);b=_series(reference,reference_populations,top_layers,e);s=clean_scale(b,e)
    return {'energy':e,'raw_per_Cu':a,'clean_per_Cu':b,'scale':s,'raw_common_scale':a/s,'clean_common_scale':b/s,'delta':difference(a,b,s),'version':VERSION,'unit':'shared clean near-edge scale units','reference_id':reference['record_id'],'sample_id':sample['record_id'],'physical_domain':domain,'top_layers':top_layers,'scaling_windows_eV':{'pre':PRE,'post':POST}}

def features(delta,energy):
    """S=SG11,3(delta); Dk=d^k SG/dE^k; C=integral delta dE.

    DCTk = first k orthonormal type-II coefficients including the DC term.
    Combined=[S,C,D1,D2]. Every operation is linear and retains amplitude.
    """
    x=np.atleast_2d(np.asarray(delta,float));e=np.asarray(energy,float)
    if x.shape[1]!=len(e) or len(e)<25 or not np.isfinite(x).all() or not np.all(np.diff(e)>0) or not np.allclose(np.diff(e),np.diff(e)[0],atol=1e-8):raise ValueError('Finite signed arrays on supported uniform energy grid required')
    step=float(np.median(np.diff(e)));s=savgol_filter(x,11,3,axis=1);d1=savgol_filter(x,11,3,deriv=1,delta=step,axis=1);d2=savgol_filter(x,11,3,deriv=2,delta=step,axis=1);c=cumulative_trapezoid(x,e,axis=1,initial=0);coeff=dct(x,type=2,norm='ortho',axis=1)
    return {'signed':x.copy(),'smoothed':s,'derivative':d1,'d2':d2,'cumulative_difference':c,'Combined':np.concatenate([s,c,d1,d2],axis=1),**{f'DCT{k}':coeff[:,:k] for k in [8,16,24]}}

def experimental_pair(sample,reference,energy,convention,task="identity"):
    if task=="identity":
        from .differential_identity import identity_input
        return identity_input(sample,reference,energy,convention)
    if task!="coverage":raise ReferenceError("Unknown experimental task")
    required={'Cu_basis','incident_intensity_correction','thickness_concentration','self_absorption','detection_geometry','reference_residual_coverage','edge_step_rule'}
    if reference is None or not required<=set(convention):raise ReferenceError('Declared compatible experimental reference and calibration/population convention required')
    if convention['Cu_basis']!='same independently calibrated per-Cu basis':raise ReferenceError('Detector gain equality or separate arbitrary normalization is not a physical Cu basis')
    if convention.get('surface_fraction') is None or not np.isfinite(float(convention['surface_fraction'])) or not 0<float(convention['surface_fraction'])<=1:raise ReferenceError('Experimental all-Cu dilution fraction required before surface-coverage prediction; otherwise only an adsorbate-related Cu-fraction proxy is supported')
    if convention['edge_step_rule']!='shared reference-derived scale with fixed documented windows':raise ReferenceError('This predictor requires the declared shared reference scale; separate edge steps need a separately evaluated amplitude assumption')
    scale=clean_scale(np.asarray(reference),np.asarray(energy))
    return {'delta':difference(sample,reference,scale),'clean_common_scale':np.asarray(reference)/scale,'reference_scale':scale,'energy':np.asarray(energy),'experimental_accuracy_validated':False,'surface_fraction':convention['surface_fraction'],'target_caveat':'reference residual coverage changes the target to a change relative to reference','convention':convention}
