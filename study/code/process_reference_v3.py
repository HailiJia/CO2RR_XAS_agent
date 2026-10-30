"""First-feature diagnostic and transferable Cu calibration; fixed widths, no warping."""
from pathlib import Path
import sys,json,hashlib,importlib.util
import numpy as np,pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks,savgol_filter
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import cosine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/reference_benchmark_v3';V1=ROOT/'study/reference_benchmark';B.mkdir(exist_ok=True);(B/'arrays').mkdir(exist_ok=True);(B/'figures').mkdir(exist_ok=True)
MATERIALS=['Cu','Cu2O','CuO'];METHODS=['fdmnes_green','fdmnes_fdm','feff','vasp_pbe','vasp_gw'];LABELS=['FDMNES green','FDMNES fdm','FEFF','VASP PBE','VASP GW'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00'];GRID=np.arange(8980.,9015.01,.2);NOMINAL={'fdmnes_green':0.,'fdmnes_fdm':0.,'feff':-8.,'vasp_pbe':120.,'vasp_gw':125.}

def norm(y):
 y=np.asarray(y,float);lo=float(y.min());div=float(np.ptp(y))
 if not np.isfinite(y).all() or div<=1e-15:raise ValueError('Flat or invalid intensity')
 return (y-lo)/div,lo,div

def gaussian(e,y,sigma):
 if not sigma:return y.copy()
 if not np.allclose(np.diff(e),np.diff(e)[0],atol=1e-5):
  xx=np.linspace(e[0],e[-1],len(e));yy=np.interp(xx,e,y)
 else:xx=e;yy=y
 width=sigma/(xx[1]-xx[0]);mass=gaussian_filter1d(np.ones_like(xx),width,mode='constant',cval=0,truncate=5);out=gaussian_filter1d(yy,width,mode='constant',cval=0,truncate=5)/mass;return np.interp(e,xx,out)

def feature(e,y,material,nominal):
 x=e+nominal;window={'Cu':(8985.,9000.),'Cu2O':(8974.,8992.),'CuO':(8977.,8989.)}[material];mask=(x>=window[0])&(x<=window[1]);xx=x[mask];yy=y[mask]
 if len(xx)<7:return None
 if material=='CuO':
  # The experimental early CuO feature is an inflection, not the later ~8997
  # white-line maximum. No fictional 8982-eV peak is imposed.
  grad=np.gradient(gaussian(xx,yy,.35),xx);i=int(np.argmax(grad))
  return float(xx[i]) if 0<i<len(xx)-1 else None
 yn,_,_=norm(yy);peaks,prop=find_peaks(yn,prominence=.02)
 return float(xx[peaks[0]]) if len(peaks) else None

def sample(curve,shift):
 x,y=curve
 if x[0]+shift>GRID[0]+1e-7 or x[-1]+shift<GRID[-1]-1e-7:raise ValueError('Full fixed comparison window not supported; no extrapolation')
 return np.interp(GRID-shift,x,y)

curves={};native={};provenance=[];vpro=json.loads((B/'current_VASP_provenance.json').read_text());current={(r['material'],r['folder_label']):r for r in vpro['sources']}
for material in MATERIALS:
 for method in ['experiment']+METHODS:
  path=V1/'processed'/f'{material}_{method}_raw.csv';source='reference-v1 supplied numeric source'
  new=current.get((material,'PBE' if method=='vasp_pbe' else 'GW' if method=='vasp_gw' else ''))
  if new is not None:
   if not new['available']:provenance.append({'material':material,'method':method,'status':'missing_numeric_current_output','reason':new['unavailable_reason']});continue
   d=pd.read_csv(B/f"{material}_{new['folder_label']}_native.csv");x=d.iloc[:,0].to_numpy();y=d.isotropic_eps2.to_numpy();path=B/f"{material}_{new['folder_label']}_native.csv";source='new current OUTCAR/HDF5 diagonal response, independently cross-checked'
  elif path.exists():x,y=pd.read_csv(path).values.T
  else:provenance.append({'material':material,'method':method,'status':'missing','reason':'No numeric series in supplied reference source and current source search'});continue
  if method!='experiment' and y.min()<-1e-12:raise ValueError('Negative native calculation requires investigation')
  sigma=1. if method.startswith('vasp') else 0.;broad=gaussian(x,y,sigma);native[material,method]=(x,y);curves[material,method]=(x,broad);pd.DataFrame({'energy_eV':x,'native':y,'native_plus_declared_Gaussian':broad}).to_csv(B/'arrays'/f'{material}_{method}_stages.csv',index=False);provenance.append({'material':material,'method':method,'status':'available','source':source,'source_path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'additional_Gaussian_sigma_eV':sigma,'additional_Gaussian_FWHM_eV':2*np.sqrt(2*np.log(2))*sigma,'additional_Lorentzian_HWHM_eV':0.,'additional_Lorentzian_FWHM_eV':0.,'native_broadening':'preserved as supplied; VASP CH_SIGMA 0.1 eV where logged','no_double_convolution':True})
anchors={m:feature(*curves[m,'experiment'],m,0.) for m in MATERIALS};cu_offsets={}
for method in METHODS:
 if ('Cu',method) not in curves:continue
 position=feature(*curves['Cu',method],'Cu',NOMINAL[method]);cu_offsets[method]=NOMINAL[method]+anchors['Cu']-position if position is not None else NOMINAL[method]
metrics=[];plotted=[];settings=[];missing=[];draw={}
for material in MATERIALS:
 experiment=sample(curves[material,'experiment'],0);exp,emin,ediv=norm(experiment)
 for alignment in ['Cu_calibrated_transfer','first_feature_shape_diagnostic']:
  for method in METHODS:
   if (material,method) not in curves:missing.append({'material':material,'method':method,'alignment':alignment,'reason':'numeric series unavailable; see current source provenance'});continue
   common=cu_offsets[method];observed=feature(*curves[material,method],material,NOMINAL[method]);shift=common;status='Cu-calibrated transferable offset'
   if alignment=='first_feature_shape_diagnostic':
    if observed is not None and anchors[material] is not None:shift=NOMINAL[method]+anchors[material]-observed;status='compound-specific early-feature diagnostic; chemical shifts removed'
    else:status='early feature not resolved; Cu-calibrated offset retained and explicitly marked'
   try:y0=sample(curves[material,method],shift);y,lo,div=norm(y0)
   except ValueError as error:missing.append({'material':material,'method':method,'alignment':alignment,'reason':str(error)});continue
   sigma=1. if method.startswith('vasp') else 0.;row={'material':material,'method':method,'processing_version':'reference-v3','comparison_type':alignment,'normalization':'(I-Imin)/(Imax-Imin), extrema over identical fixed 8980–9015 eV interval','energy_min_eV':GRID[0],'energy_max_eV':GRID[-1],'n_points':len(GRID),'shift_eV':shift,'shift_sign':'E_photon=E_native+shift','feature_native_plus_nominal_eV':observed,'experimental_early_feature_eV':anchors[material],'feature_kind':'early rising-edge derivative maximum' if material=='CuO' else 'first resolved local maximum','alignment_status':status,'Gaussian_sigma_eV':sigma,'Gaussian_FWHM_eV':2.354820045*sigma,'Lorentzian_additional_HWHM_eV':0.,'Lorentzian_additional_FWHM_eV':0.,'cosine_similarity':float(1-cosine(y,exp)),'Pearson_r':float(pearsonr(y,exp).statistic),'Spearman_rho':float(spearmanr(y,exp).statistic),'RMSE':float(np.sqrt(np.mean((y-exp)**2))),'offset_intensity':lo,'divisor_intensity':div};metrics.append(row);settings.append(row)
   for E,I,J,A in zip(GRID,y,exp,y0):plotted.append({'material':material,'method':method,'comparison_type':alignment,'energy_eV':E,'calculation_normalized':I,'experiment_normalized':J,'calculation_pre_normalization':A})
   if alignment=='first_feature_shape_diagnostic':draw[material,method]=(y,status)
# One reader-visible table/score panel sits immediately below each material.
plt.rcParams.update({'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7})
fig,axes=plt.subplots(2,3,figsize=(6.5,5.3),gridspec_kw={'height_ratios':[2.4,1.35]})
for j,mat in enumerate(MATERIALS):
 ax=axes[0,j];exp,_,_=norm(sample(curves[mat,'experiment'],0));ax.plot(GRID,exp,c='black',lw=1.7,label='Experiment')
 table=[]
 for k,method in enumerate(METHODS):
  if (mat,method) not in draw:table.append([['Green','FDM','FEFF','PBE','GW label'][k],'—','—','—']);continue
  y,status=draw[mat,method];fallback=status.startswith('early feature not resolved');label=LABELS[k]+(' (Cu offset)' if fallback else '');ax.plot(GRID,y,c=colors[k],lw=1.25,ls=':' if fallback else '-',label=label);r=next(x for x in metrics if x['material']==mat and x['method']==method and x['comparison_type']=='first_feature_shape_diagnostic');table.append([['Green','FDM','FEFF','PBE','GW label'][k],f"{r['cosine_similarity']:.3f}",f"{r['Pearson_r']:.3f}",f"{r['Spearman_rho']:.3f}"])
 ax.set(xlim=(8975,9015),ylim=(-.02,1.08),xlabel='Photon energy (eV)',ylabel='Intensity (a.u.)',title=mat);bottom=axes[1,j];bottom.axis('off');tab=bottom.table(cellText=table,colLabels=['Method','Cos','r','rho'],loc='upper center',colWidths=[.43,.19,.19,.19]);tab.auto_set_font_size(False);tab.set_fontsize(6.8);tab.scale(1,1.35)
handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.035),fontsize=7,frameon=False);axes[0,1].text(.03,.94,'Dotted FEFF: Cu offset',transform=axes[0,1].transAxes,fontsize=6.2);fig.tight_layout();fig.savefig(B/'figures/reference_comparison.png',dpi=400,bbox_inches='tight');plt.close(fig)
pd.DataFrame(metrics).to_csv(B/'similarity_metrics.csv',index=False);pd.DataFrame(plotted).to_csv(B/'plotted_arrays.csv',index=False);pd.DataFrame(missing).to_csv(B/'missing_series.csv',index=False);(B/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');(B/'settings.json').write_text(json.dumps({'version':'reference-v3','window_eV':[8980,9015],'display_axis_eV':[8975,9015],'unplotted_8975_8980':'outside full common support after calibration; no extrapolation or padded intensities','Gaussian_for_VASP_sigma_eV':1.,'source_of_Gaussian':'author Cu_XAS.ipynb, fixed prescription not fitted instrument resolution','native_CH_SIGMA':'0.1 eV where logged; no universal Lorentzian added','feature_rules':'Cu/Cu2O first local maximum; CuO first rising-edge derivative maximum, not late white line','experimental_anchors_eV':anchors,'Cu_calibration_offsets_eV':cu_offsets,'main_view':'separate first-feature shape diagnostic, with transferable offsets also scored and retained','CuO_fdm_CPU_time_s':129600,'FEFF_module_times':'CPU times, not wall time','normalization':'same fixed-interval minmax for every trace; no unsupported pre-edge subtraction','no_negative_absorption_masking':'raw calculation negativity is checked; raw experimental baseline/noise retained; minmax is a declared relative-intensity comparison','metrics':'all scores use exact plotted grid and arrays; no independent-energy-point p-values','timing_CPU_conventions':'CuO FDMNES fdm 129600 s retained; FEFF module values are CPU time; VASP SCF and XAS stage CPU times are separate, allocated-core-time estimates are not measured CPU', 'GW_label':'GW-PAW is a dataset/folder label; current Cu2O independent-particle PBE+U response is not established quasiparticle GW or BSE'},indent=2)+'\n');print('available metric rows',len(metrics),'missing rows',len(missing),'anchors',anchors)
