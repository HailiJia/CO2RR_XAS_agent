"""Reference-v2: native-input, fixed-window min--max similarity comparison.

reference-v1 stays immutable. No inferred universal lifetime convolution,
unsupported local background, variable-window fitting, or digitized spectra.
"""
from pathlib import Path
import hashlib,importlib.util,json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import cosine
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import minimize_scalar
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
V1=ROOT.parent/'reference_benchmark'
GRID=np.linspace(8980.,9020.,201)
MATERIALS=['Cu','Cu2O','CuO']
METHODS=['fdmnes_green','fdmnes_fdm','feff','vasp_pbe','vasp_gw']
LABELS=dict(zip(METHODS,['FDMNES green','FDMNES fdm','FEFF','VASP PBE','VASP GW']))
COLORS=dict(zip(METHODS,['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']))
BOUNDS={'fdmnes_green':[-4.,4.],'fdmnes_fdm':[-4.,4.],'feff':[-15.,-6.],'vasp_pbe':[105.,135.],'vasp_gw':[110.,145.]}


def normalize(y):
    y=np.asarray(y,float)
    if not np.isfinite(y).all():raise ValueError('Nonfinite intensity')
    minimum=float(y.min());divisor=float(y.max()-minimum)
    if divisor<=1e-14*max(1.,abs(y).max()):raise ValueError('Flat intensity')
    z=(y-minimum)/divisor
    if z.min() < -1e-12 or z.max()>1+1e-12:raise ValueError('Normalization failed')
    return z,minimum,divisor


def similarity(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    if len(a)<20 or a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('Invalid comparison')
    # Undefined cases are blank/NaN, never zero agreement or fabricated p-values.
    return {'cosine_similarity':float(1-cosine(a,b)) if min(np.linalg.norm(a),np.linalg.norm(b))>0 else None,
            'pearson_r':float(pearsonr(a,b).statistic) if min(np.ptp(a),np.ptp(b))>0 else None,
            'spearman_rho':float(spearmanr(a,b).statistic) if min(np.ptp(a),np.ptp(b))>0 else None,
            'rmse':float(np.sqrt(np.mean((a-b)**2)))}


def gaussian(x,y,sigma):
    if not sigma:return y.copy()
    # Uniform auxiliary grid remains entirely inside native support. Boundary
    # kernel mass is renormalized; no spectral extrapolation or negative lobes.
    step=min(.1,float(np.median(np.diff(x))))
    grid=np.linspace(x.min(),x.max(),int(np.ceil((x.max()-x.min())/step))+1)
    a=np.interp(grid,x,y);width=sigma/(grid[1]-grid[0])
    z=gaussian_filter1d(a,width,mode='constant',cval=0.,truncate=5)/gaussian_filter1d(np.ones_like(a),width,mode='constant',cval=0.,truncate=5)
    return np.interp(x,grid,z)


def sample(curve,shift):
    x,y=curve
    if x.min()+shift>GRID.min()+1e-7 or x.max()+shift<GRID.max()-1e-7:raise ValueError('Full fixed window unavailable')
    return np.interp(GRID-shift,x,y)


def fit_offset(curve,reference,bounds):
    objective=[]
    def score(shift):
        try:a,_,_=normalize(sample(curve,shift));b,_,_=normalize(sample(reference,0.))
        except ValueError:return 1e9
        return float(np.sqrt(np.mean((a-b)**2)))
    xs=np.arange(bounds[0],bounds[1]+.025,.05)
    vals=np.asarray([score(x) for x in xs]);i=int(vals.argmin())
    if vals[i]>=1e9:return None,[],{'status':'full fixed window unsupported'}
    lo=max(bounds[0],xs[i]-.06);hi=min(bounds[1],xs[i]+.06)
    result=minimize_scalar(score,bounds=(lo,hi),method='bounded',options={'xatol':1e-5})
    shift=float(result.x) if result.fun<vals[i] else float(xs[i]);best=score(shift)
    interval=xs[vals<=best+.01]
    return shift,[{'shift_eV':float(x),'objective_RMSE':float(v) if v<1e9 else None} for x,v in zip(xs,vals)],{
        'bounds_eV':bounds,'boundary_hit':bool(abs(shift-bounds[0])<.051 or abs(shift-bounds[1])<.051),
        'RMSE_plus_0p01_shift_sensitivity_eV':[float(interval.min()),float(interval.max())],
        'sensitivity_not_confidence_interval':True,'objective_RMSE':best}


def run():
    ROOT.mkdir(exist_ok=True);(ROOT/'arrays').mkdir(exist_ok=True);(ROOT/'figures').mkdir(exist_ok=True)
    provenance=pd.read_csv(V1/'provenance.csv').fillna('');curves={};audit=[]
    for row in provenance.itertuples():
        if row.status!='available':continue
        path=V1/'processed'/f'{row.material}_{row.method}_raw.csv';d=pd.read_csv(path);x,y=d.values.T
        if row.method!='experiment' and y.min() < -1e-12:raise ValueError('Negative native calculation; investigate before normalization')
        curves[row.material,row.method]=x,y
        legacy=V1/'processed'/f'{row.material}_{row.method}_common.csv'
        oldmin=float(pd.read_csv(legacy).iloc[:,1].min()) if legacy.exists() else None
        audit.append({'material':row.material,'method':row.method,'native_min':float(y.min()),'native_max':float(y.max()),'native_nonnegative':bool(y.min()>=-1e-12),
                      'v1_processed_min':oldmin,'first_negative_stage':'raw experimental baseline/noise' if y.min()<0 else 'v1 local-background subtraction' if oldmin is not None and oldmin<0 else 'none observed',
                      'v2_background_correction':'none: v1 local window not independently justified; min-max is explicitly a relative-intensity transform, not a physical background correction',
                      'native_array_sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        pd.DataFrame({'energy_eV':x,'native_intensity':y}).to_csv(ROOT/'arrays'/f'{row.material}_{row.method}_native.csv',index=False)
        if row.method.startswith('vasp'):
            pd.DataFrame({'energy_eV':x,'native_CH_SIGMA_0p1_intensity':y,'additional_Gaussian_sigma1_intensity':gaussian(x,y,1.)}).to_csv(ROOT/'arrays'/f'{row.material}_{row.method}_broadening_stages.csv',index=False)
    offsets={};alignment=[];objective_rows=[]
    for method in METHODS:
        shift,curve,info=fit_offset(curves['Cu',method],curves['Cu','experiment'],BOUNDS[method]);offsets[method]=shift
        alignment.append({'material':'Cu','method':method,'comparison':'transferable_Cu_calibration','shift_eV':shift,**info})
        objective_rows += [{'material':'Cu','method':method,**r} for r in curve]
    rows=[];arrays=[];normalization=[];unsupported=[]
    for material in MATERIALS:
        reference=curves[material,'experiment'];b,bmin,bdiv=normalize(sample(reference,0.))
        normalization.append({'material':material,'method':'experiment','comparison':'all','shift_eV':0.,'minimum':bmin,'divisor':bdiv})
        for method in METHODS:
            if (material,method) not in curves:
                unsupported.append({'material':material,'method':method,'reason':'no verified numeric XAS series in source snapshot or repeat local search'});continue
            curve=curves[material,method];common=offsets[method]
            shape,_,info=fit_offset(curve,reference,[common-5,common+5]);alignment.append({'material':material,'method':method,'comparison':'separate_shape_alignment','shift_eV':shape,**info})
            extra=(curve[0],gaussian(*curve,1.))
            for comparison,shift,series,sigma in [('Cu_calibrated',common,curve,0.),('shape_aligned',shape,curve,0.),('native_axis',0.,curve,0.),('Gaussian_sigma1_diagnostic',common,extra,1.)]:
                try:raw=sample(series,shift);a,minimum,divisor=normalize(raw)
                except (ValueError,TypeError):unsupported.append({'material':material,'method':method,'comparison':comparison,'reason':'identical full window unavailable; no variable intersection'});continue
                if method!='experiment' and raw.min() < -1e-12:raise ValueError('Substantive negative pre-normalization absorption')
                rows.append({'material':material,'method':LABELS[method],'method_key':method,'processing_version':'reference-v2','comparison_type':comparison,
                    'normalization':'(I-min)/(max-min), extrema on identical 8980–9020 eV grid','energy_min_eV':8980.,'energy_max_eV':9020.,'n_points':201,
                    'shift_eV':shift,'Gaussian_sigma_added_eV':sigma,'Gaussian_FWHM_added_eV':2*np.sqrt(2*np.log(2))*sigma,
                    'Lorentzian_HWHM_added_eV':0.,'Lorentzian_FWHM_added_eV':0.,
                    'native_processing':'CH_SIGMA=0.1 eV, executable-specific width convention unverified' if method.startswith('vasp') else 'native FEFF lifetime/self-energy' if method=='feff' else 'raw single CuO site; final convolution absent' if material=='CuO' and method=='fdmnes_fdm' else 'native energy-dependent convolution retained',
                    **similarity(a,b)})
                normalization.append({'material':material,'method':method,'comparison':comparison,'shift_eV':shift,'minimum':minimum,'divisor':divisor})
                arrays.extend([{'material':material,'method':method,'comparison_type':comparison,'energy_eV':float(e),'calculation':float(v),'experiment':float(w)} for e,v,w in zip(GRID,a,b)])
    pd.DataFrame(audit).to_csv(ROOT/'before_after_processing_audit.csv',index=False);pd.DataFrame(normalization).to_csv(ROOT/'normalization.csv',index=False)
    pd.DataFrame(rows).to_csv(ROOT/'similarity_metrics.csv',index=False);data=pd.DataFrame(arrays);data.to_csv(ROOT/'plotted_arrays.csv',index=False)
    pd.DataFrame(objective_rows).to_csv(ROOT/'calibration_objectives.csv',index=False);(ROOT/'alignment.json').write_text(json.dumps(alignment,indent=2)+'\n');pd.DataFrame(unsupported).to_csv(ROOT/'unavailable_comparisons.csv',index=False)
    provenance.to_csv(ROOT/'provenance_v1_source_link.csv',index=False)
    protocol={'version':'reference-v2','archived_v1_unchanged':True,'native_arrays':'verified reference-v1 raw parsed arrays; source hashes/provenance retained','grid_eV':GRID.tolist(),
        'primary':'data-as-supplied native broadening, no additional lifetime or instrumental convolution; conditional relative-intensity similarity, not a calibrated physical cross-section accuracy comparison',
        'Gaussian_diagnostic':'sigma1 eV from original notebook; not a measured instrument resolution or validated equivalent processing for unrelated OUTCAR sources',
        'background':'no unsupported local pre-edge subtraction; raw experiment baseline/noise retained outside comparison; final min-max is relative scaling only',
        'shift_sign':'E_display = E_native + shift_eV','method_Cu_offsets_eV':offsets,'bounds_eV':BOUNDS,
        'bounds_basis':'FDMNES Energpho axis near Cu K; FEFF omega offset near notebook -8/-9; distinct PBE/GW PAW core energies put observed native onset about 110–130 eV below experiment; broad bounds frozen before oxide comparison',
        'objective':'RMSE after identical min-max normalization on FULL fixed 8980–9020 eV/201-point grid at every trial; unavailable support rejected',
        'degrees_of_freedom':'primary one Cu calibration shift per declared method, transferred unchanged to oxides; widths fixed at zero added; shape diagnostic one additional per-compound shift bounded ±5 eV',
        'fitting_uncertainty':'objective profiles and RMSE+0.01 shift-sensitivity intervals; not statistical confidence intervals',
        'negative_tolerance':1e-12,'no_clipping_or_absolute_values':True,'metrics':'cosine = normalized dot product without mean-centering (SciPy cosine); Pearson centered pearsonr; Spearman average-rank ties via spearmanr; no per-energy p-values',
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (ROOT/'settings.json').write_text(json.dumps(protocol,indent=2)+'\n')
    # Reproduce the recoverable notebook cell on its actual input, separately.
    orphan=np.loadtxt(V1/'raw/Cu_GW.dat');broadened=gaussian_filter1d(orphan[:,1],1/(orphan[1,0]-orphan[0,0]))
    pd.DataFrame({'native_energy_eV':orphan[:,0],'raw_intensity':orphan[:,1],'notebook_Gaussian_sigma1':broadened,'cell1_energy_plus120':orphan[:,0]+120,'cell5_energy_plus124':orphan[:,0]+124}).to_csv(ROOT/'arrays/notebook_Cu_GW_actual_input_unverified_lineage.csv',index=False)
    (ROOT/'notebook_reconciliation.json').write_text(json.dumps({'original_notebook_sha256':hashlib.sha256((V1/'raw/Cu_XAS.ipynb').read_bytes()).hexdigest(),
        'cell1_reproduced_on_actual_Cu_GW_dat':True,'Cu_GW_dat_lineage_to_parsed_OUTCAR':'unverified; not substituted or included in method metrics','Cu_PBE_dat_bytes':(V1/'raw/Cu_PBE.dat').stat().st_size,
        'cell2_status':'unavailable empty original input; separate verified OUTCAR PBE series is explicitly different input','cell3_plus125_and_cell5_reused_intensity':'cannot reproduce a coherent multi-method route after empty PBE input; exploratory offsets are not calibrations',
        'v1_universal_extra_Lorentzian':'removed in v2: no independently established remaining width for these VASP outputs; native CH_SIGMA alone does not resolve instrumental width',
        'sigma1':'exact notebook route saved for actual source; extra Gaussian diagnostic also supplied for parsed spectra without claiming equivalent lineage'},indent=2)+'\n')
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    for comparison in ['Cu_calibrated','shape_aligned','native_axis','Gaussian_sigma1_diagnostic']:
        fig,axes=plt.subplots(3,5,figsize=(14,8),sharex=True,sharey=True)
        for i,material in enumerate(MATERIALS):
            for j,method in enumerate(METHODS):
                ax=axes[i,j];q=data[(data.material==material)&(data.method==method)&(data.comparison_type==comparison)]
                if len(q):
                    ax.plot(q.energy_eV-8979,q.experiment,c='black',lw=1.5,label='Experiment')
                    provisional=material=='CuO' and method=='fdmnes_fdm'
                    ax.plot(q.energy_eV-8979,q.calculation,c=COLORS[method],lw=1.5,ls='--' if provisional else '-',label=LABELS[method]+(' (raw site 1)' if provisional else ''))
                    if provisional:ax.legend(fontsize=7,loc='upper right')
                else:ax.set_axis_off()
                if i==0:ax.set_title(LABELS[method])
                if j==0:ax.set_ylabel(material+'\nIntensity (a.u.)')
                if i==2:ax.set_xlabel('Photon energy − 8979 eV')
                ax.set_xlim(1,41);ax.set_ylim(-.03,1.05)
        fig.suptitle('reference-v2: '+comparison);fig.text(.5,.015,'Black: Experiment. Dashed CuO FDMNES fdm: raw single-site series; completed native convolution absent. VASP GW denotes the supplied PAW/input alias, not a verified GW/BSE treatment.',ha='center',fontsize=8)
        fig.tight_layout(rect=(0,.045,1,.95))
        for ext in ['pdf','svg','png']:fig.savefig(ROOT/'figures'/f'reference_grid_{comparison}.{ext}',dpi=450)
        plt.close(fig)
    print(pd.DataFrame(rows).query('comparison_type=="Cu_calibrated"')[['material','method','shift_eV','cosine_similarity','pearson_r','spearman_rho']].to_string(index=False))


if __name__=='__main__':run()
