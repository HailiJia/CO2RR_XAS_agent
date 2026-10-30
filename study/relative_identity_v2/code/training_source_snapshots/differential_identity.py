"""Identity input contract: relative signed observables, never absolute coverage."""
import hashlib,json
from pathlib import Path
import numpy as np
from .differential_xas import ReferenceError,features
VERSION='relative-signed-identity-v2'

def identity_input(sample,reference,energy,convention):
    """Positive common gain cancels in a shared reference-RMS divisor.

    Precomputed differences need no unavailable raw pair. Metadata is a declaration,
    not proof of compatibility or experimental validation. Never remove the sign.
    """
    required={'input_kind','sign','energy_registration','normalization','reference_chemistry','domain','noise','processing'}
    if not isinstance(convention,dict) or required-set(convention):raise ReferenceError('Identity requires sign, registration, normalization, reference chemistry, domain, noise and processing declarations')
    if convention['sign']!='reaction-minus-reference' or not convention['energy_registration'] or not convention['reference_chemistry'] or not convention['domain']:raise ReferenceError('Explicit reaction-minus-reference sign, registration and reference/domain required')
    e=np.asarray(energy,float);a=np.asarray(sample,float)
    if e.ndim!=1 or a.shape!=e.shape or len(e)<25 or not np.isfinite(e).all() or not np.isfinite(a).all() or not np.all(np.diff(e)>0):raise ReferenceError('Finite increasing energy grid and matching signed intensity required')
    kind=convention['input_kind'];normal=convention['normalization']
    if kind=='precomputed_difference':
        if reference is not None:raise ReferenceError('Precomputed difference must not be subtracted twice')
        if normal not in {'signed_rms','shared_reference_rms','shared_reference_near_edge_scale','documented_positive_divisor'}:raise ReferenceError('Unsupported signed normalization convention')
        if normal=='documented_positive_divisor' and not convention.get('normalization_details'):raise ReferenceError('Document the positive divisor')
        d=a.copy();scale=None
    elif kind=='reaction_reference_pair':
        b=np.asarray(reference,float)
        if reference is None or b.shape!=a.shape or not np.isfinite(b).all():raise ReferenceError('Compatible pair arrays required')
        if convention.get('relative_gain')!='common_or_independently_corrected' or convention.get('background')!='documented_compatible':raise ReferenceError('Pair gain/background compatibility must be documented; independent arbitrary normalization forbidden')
        if normal!='shared_reference_rms':raise ReferenceError('Relative pair contract uses one reference-RMS divisor on the declared full input interval')
        scale=float(np.sqrt(np.mean(b*b)))
        if scale<=0:raise ReferenceError('Reference scale is zero')
        d=(a-b)/scale
    else:raise ReferenceError('Choose precomputed_difference or reaction_reference_pair')
    return {'delta':d,'energy':e,'reference_scale':scale,'convention':convention,'input_contract':VERSION,'experimental_accuracy_validated':False,'absolute_coverage_available':False,'target_caveat':'Identity only; unknown gain/active fraction/reference coverage prevents absolute loading. Residual adsorbates define a change relative to reference.'}

def shape_floor(delta):
    rms=np.sqrt(np.mean(np.asarray(delta)**2,axis=1));positive=rms[rms>1e-12]
    return max(1e-10,.01*float(np.quantile(positive,.1))) if len(positive) else 1e-10

def signed_shape(delta,floor):
    x=np.atleast_2d(np.asarray(delta,float));r=np.sqrt(np.mean(x*x,axis=1));valid=r>floor
    # Invalid/near-null inputs remain exactly zero and must be reviewed/abstained.
    out=np.zeros_like(x);out[valid]=x[valid]/r[valid,None]
    return out,valid

def infer_identity(model_path,sample,reference,energy,convention):
    import joblib
    from .identity_comparison import score_model,transform
    m=joblib.load(model_path)
    if m.get('input_contract')!=VERSION:raise ReferenceError('Not a relative-identity model')
    q=identity_input(sample,reference,energy,convention);e=q['energy'];grid=np.asarray(m['energy'])
    if e[0]>grid[0]+1e-8 or e[-1]<grid[-1]-1e-8:raise ReferenceError('Missing trained energy support; no extrapolation')
    if m['representation'] not in {'relative_delta','signed_shape'}:raise ReferenceError('Precomputed differences cannot reconstruct raw absorption')
    if m['representation']=='relative_delta' and convention['normalization']!='shared_reference_near_edge_scale':raise ReferenceError('Amplitude-sensitive model requires the trained near-edge relative scale; use a signed-shape model for another documented positive divisor')
    d=np.interp(grid,e,q['delta'])[None,:];noise=convention.get('noise');noise_rms=noise.get('rms_in_input_units') if isinstance(noise,dict) else None
    if noise_rms is not None and (not np.isfinite(float(noise_rms)) or float(noise_rms)<0):raise ReferenceError('Noise RMS must be finite and nonnegative')
    if convention['normalization']=='signed_rms':
        snr=convention.get('pre_normalization_signal_to_noise')
        if snr is None or not np.isfinite(float(snr)) or float(snr)<3:
            return {'status':'abstain_unresolved_signal_floor','predicted_identity_set':None,'reason':'Unit-normalized input requires a documented pre-normalization SNR >=3; no absolute calibration or raw pair required','experimental_accuracy_validated':False}
    floor=max(m['floor'],3*float(noise_rms)) if noise_rms is not None else m['floor']
    if np.sqrt(np.mean(d*d))<=floor:return {**q,'delta':d[0].tolist(),'energy':grid.tolist(),'status':'abstain_low_signal','predicted_identity_set':None,'experimental_accuracy_validated':False}
    X=transform(d,grid,m['representation'],m['feature'],m['floor']);s=score_model(m['model'],X);p=s[0]>=m['thresholds']
    return {'status':'review_required' if convention['domain'] not in m['domains'] else 'conditional_prediction','predicted_identity_set':[sp for sp,v in zip(m['species'],p) if v],'scores':dict(zip(m['species'],s[0].tolist())),'uncertainty':'uncalibrated scores; simulation-to-experiment transfer unvalidated','experimental_accuracy_validated':False,'absolute_coverage_available':False,'reference_chemistry':convention['reference_chemistry'],'normalization':convention['normalization'],'input_contract':VERSION,'model_sha256':hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),'review_flags':['unmeasured experimental accuracy','reference chemistry and domain must be independently checked']}
