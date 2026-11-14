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
        scale=stable_rms(b)
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
    d=np.interp(grid,e,q['delta'])[None,:]
    # Detectability and the representation's positive divisor are different facts.
    rms=stable_rms(d[0]);noise=convention.get('noise');noise_rms=None
    if isinstance(noise,dict):
        if 'rms_difference_in_pair_units' in noise:
            if convention['input_kind']!='reaction_reference_pair':raise ReferenceError('Pair-unit noise requires a raw relative pair')
            noise_rms=float(noise['rms_difference_in_pair_units'])/q['reference_scale']
        elif 'rms_in_input_units' in noise:
            if convention['input_kind']=='reaction_reference_pair' and noise.get('unit_basis')!='normalized_difference':raise ReferenceError('Declare normalized_difference noise units or supply rms_difference_in_pair_units')
            noise_rms=float(noise['rms_in_input_units'])
        if noise_rms is not None and (not np.isfinite(noise_rms) or noise_rms<0):raise ReferenceError('Noise RMS must be finite and nonnegative')
    reported=convention.get('pre_normalization_signal_to_noise')
    if reported is not None:
        reported=float(reported)
        if not np.isfinite(reported) or reported<0:raise ReferenceError('Pre-normalization SNR must be finite and nonnegative')
    if convention['normalization']=='signed_rms' and rms>0 and not np.isclose(rms,1.,rtol=1e-4,atol=1e-8):raise ReferenceError('signed_rms declaration contradicts the supplied difference RMS')
    base={'experimental_accuracy_validated':False,'absolute_coverage_available':False,'input_contract':VERSION,'detectability_rule':'RMS signal/noise >=3, or documented pre-normalization SNR >=3; not a calibrated detection limit','model_sha256':hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),'reference_chemistry':convention['reference_chemistry'],'normalization':convention['normalization'],'target_caveat':q['target_caveat']}
    if rms==0:return {**base,'status':'abstain_low_signal','predicted_identity_set':None,'reason':'exactly zero signed difference'}
    snr=rms/noise_rms if noise_rms is not None and noise_rms>0 else None
    if snr is not None and reported is not None and (snr>=3)!=(reported>=3):raise ReferenceError('Noise-derived and reported SNR give contradictory detectability decisions')
    evidence=reported if reported is not None else snr
    simulation=convention.get('detectability_basis')=='audited_simulation_scale'
    if simulation and convention['normalization']!='shared_reference_near_edge_scale':raise ReferenceError('Simulation-amplitude floor requires the original audited near-edge convention')
    if evidence is not None and evidence<3:return {**base,'status':'abstain_low_signal','predicted_identity_set':None,'signal_to_noise':float(evidence)}
    if evidence is None and not simulation:return {**base,'status':'review_required_detectability','predicted_identity_set':None,'reason':'Detectability unknown; a normalization divisor is not noise evidence'}
    if simulation and evidence is None and rms<=m['floor']:return {**base,'status':'abstain_low_signal','predicted_identity_set':None,'reason':'declared audited simulation-scale heuristic, not an experimental detection limit'}
    if m['representation']=='signed_shape' and not (simulation and evidence is None):
        # Scale to stable RMS only after the unit-compatible gate, then bypass the
        # simulation-amplitude mask. This retains signs even for tiny divisors.
        x=d/np.max(abs(d));x=x/stable_rms(x[0]);X=transform(x,grid,'signed_shape',m['feature'],0.)
    else:X=transform(d,grid,m['representation'],m['feature'],m['floor'])
    s=score_model(m['model'],X);p=s[0]>=m['thresholds']
    return {**base,'status':'review_required' if convention['domain'] not in m['domains'] else 'conditional_prediction','predicted_identity_set':[sp for sp,v in zip(m['species'],p) if v],'scores':dict(zip(m['species'],s[0].tolist())),'signal_to_noise':float(evidence) if evidence is not None else None,'uncertainty':'uncalibrated classifier scores; simulation-to-experiment transfer unvalidated','review_flags':['unmeasured experimental accuracy','reference chemistry and domain must be independently checked','unknown residual reference coverage changes the target to a relative change']}


def stable_rms(values):
    x=np.asarray(values,float);scale=float(np.max(abs(x)))
    return scale*float(np.sqrt(np.mean((x/scale)**2))) if scale else 0.
