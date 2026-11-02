"""Numerical reference comparison under a fixed recorded protocol.

No plot digitization. Raw provenance and allowed alignment must be supplied.
"""
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.stats import pearsonr
REQUIRED=('source','license','backend','mode','version','structure','core_hole','convergence','broadening','energy_calibration','interval_eV','allowed_shift_eV','allowed_scale','allowed_offset')

def compare_reference(simulated,experimental,protocol):
    missing=[k for k in REQUIRED if k not in protocol]
    if missing:raise ValueError('Missing reference provenance/protocol: '+', '.join(missing))
    if any(protocol.get(k) in [None,'','unknown','unavailable'] for k in ['source','license','backend','version','structure','core_hole','convergence','broadening','energy_calibration']):raise ValueError('Reference provenance is unresolved')
    if protocol.get('data_kind')!='raw_numeric':raise ValueError('Exact reference metrics require raw numeric traces, not plot digitization')
    sim=np.asarray(simulated,float);exp=np.asarray(experimental,float);lo,hi=protocol['interval_eV'];mask=(exp[:,0]>=lo)&(exp[:,0]<=hi);x=exp[mask,0];y=exp[mask,1]
    shiftmax=float(protocol['allowed_shift_eV'])
    if len(x)<5 or not np.isfinite(sim).all() or not np.isfinite(exp).all() or not np.all(np.diff(sim[:,0])>0) or not np.all(np.diff(exp[:,0])>0):raise ValueError('Invalid raw reference grids')
    if x[0]-shiftmax<sim[0,0] or x[-1]+shiftmax>sim[-1,0]:raise ValueError('Shift protocol would extrapolate beyond simulated grid')
    span=np.ptp(y)
    if span<=0:raise ValueError('Constant experimental trace')
    def score(parameters):
        shift,scale,offset=parameters;return scale*np.interp(x-shift,sim[:,0],sim[:,1])+offset
    def metrics(parameters):
        predicted=score(parameters)
        return dict(normalized_RMSE=float(np.sqrt(np.mean((predicted-y)**2))/span),pearson_r=float(pearsonr(predicted,y).statistic),edge_displacement_eV=float(x[np.argmax(np.gradient(predicted,x))]-x[np.argmax(np.gradient(y,x))]),peak_displacement_eV=float(x[np.argmax(predicted)]-x[np.argmax(y)]),shift_eV=float(parameters[0]),scale=float(parameters[1]),offset=float(parameters[2]))
    bounds=[(-shiftmax,shiftmax),tuple(protocol['allowed_scale']),tuple(protocol['allowed_offset'])]
    fit=minimize(lambda p:np.mean((score(p)-y)**2),[0,1,0],bounds=bounds)
    return dict(unshifted=metrics([0,1,0]),aligned=metrics(fit.x),optimizer_success=bool(fit.success),protocol=protocol)
