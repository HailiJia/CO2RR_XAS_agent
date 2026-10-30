"""Correct Cu PBE's early-peak assignment, then rebuild matching scores/SI grids.
The 8982-eV anchor is author specified. Broadening stays sigma=1 eV as supplied.
No score optimization, stretching, extrapolation or intensity clipping is performed.
"""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd
from scipy.signal import find_peaks
from scipy.stats import pearsonr,spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parents[1];D=P/'figure_data'
afile=D/'reference_plotted_arrays.csv';mfile=D/'similarity_metrics.csv'
# Idempotent: preserve the incoming arrays exactly once.
for file in [afile,mfile]:
 backup=file.with_name(file.stem+'_before_20261028.csv')
 if not backup.exists():backup.write_bytes(file.read_bytes())
a=pd.read_csv(D/'reference_plotted_arrays_before_20261028.csv');m=pd.read_csv(D/'similarity_metrics_before_20261028.csv')
cu=pd.read_csv(D/'reference_sources/Cu_vasp_pbe_stages.csv');e=cu.energy_eV.to_numpy();y=cu.native_plus_declared_Gaussian.to_numpy()
row=m.query('material=="Cu" and method=="vasp_pbe"').iloc[0];previous=float(row.shift_eV)
# The first resolved small Cu PBE feature was incorrectly assigned to 8993 eV.
# Locate that feature on the saved broadened native spectrum, then translate rigidly.
mask=(e+previous>=8990)&(e+previous<=8995);ii=np.flatnonzero(mask);peaks,_=find_peaks(y[ii]);assert len(peaks)==1
native_peak=float(e[ii[peaks[0]]]);target=8982.;shift=target-native_peak;change=shift-previous
records=[]
for material,view,new_shift in [('Cu','Cu_calibrated_transfer',shift),('Cu','first_feature_shape_diagnostic',shift),('Cu2O','Cu_calibrated_transfer',shift)]:
 stage=D/'reference_sources'/f'{material}_vasp_pbe_stages.csv';d=pd.read_csv(stage);x=d.energy_eV.to_numpy();yy=d.native_plus_declared_Gaussian.to_numpy();sel=(a.material==material)&(a.method=='vasp_pbe')&(a.comparison_type==view);ix=a.index[sel];grid=a.loc[ix,'energy_eV'].to_numpy();j=m.index[(m.material==material)&(m.method=='vasp_pbe')&(m.comparison_type==view)][0]
 # Establish exact correspondence to the incoming processing before changing it.
 np.testing.assert_allclose(np.interp(grid-float(m.at[j,'shift_eV']),x,yy),a.loc[ix,'calculation_pre_normalization'],atol=1e-15,rtol=1e-10)
 assert x.min()<=min(grid-new_shift) and x.max()>=max(grid-new_shift)
 raw=np.interp(grid-new_shift,x,yy);norm=(raw-raw.min())/np.ptp(raw);exp=a.loc[ix,'experiment_normalized'].to_numpy()
 a.loc[ix,'calculation_pre_normalization']=raw;a.loc[ix,'calculation_normalized']=norm
 values={'shift_eV':new_shift,'processing_version':'reference-v3-Cu-PBE-anchor-correction','alignment_status':'author-specified 8982 eV Cu early-peak calibration; transferred within PBE' if view=='Cu_calibrated_transfer' else 'Cu first small peak assigned to 8982 eV; fixed Gaussian retained','offset_intensity':raw.min(),'divisor_intensity':np.ptp(raw),'cosine_similarity':float(norm@exp/(np.linalg.norm(norm)*np.linalg.norm(exp))),'Pearson_r':float(pearsonr(norm,exp).statistic),'Spearman_rho':float(spearmanr(norm,exp).statistic),'RMSE':float(np.sqrt(np.mean((norm-exp)**2)))}
 if material=='Cu':values.update(experimental_early_feature_eV=target,feature_kind='author-specified small early-feature anchor; not a fitted white-line assignment')
 for k,v in values.items():m.at[j,k]=v
 records.append({'material':material,'view':view,**values,'source_sha256':hashlib.sha256(stage.read_bytes()).hexdigest()})
a.to_csv(afile,index=False);m.to_csv(mfile,index=False)
settings=json.load(open(D/'reference_settings.json'));settings['Cu_calibration_offsets_eV']['vasp_pbe']=shift;settings['Cu_PBE_correction_20261028']={'native_early_peak_eV':native_peak,'target_eV':target,'previous_shift_eV':previous,'corrected_shift_eV':shift,'rigid_change_eV':change,'width_change':False,'scope':'Cu both views; inherited Cu2O PBE transferable calibration; compound-specific Cu2O unchanged'};(D/'reference_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
(D/'Cu_PBE_alignment_correction.json').write_text(json.dumps({'request':'align first small Cu PBE peak at 8982 eV','native_peak_eV':native_peak,'target_eV':target,'previous_shift_eV':previous,'corrected_shift_eV':shift,'rigid_change_eV':change,'changed_rows':records},indent=2)+'\n')
# Write the complete SI score table from the same arrays used for all figure scores.
method_labels={'fdmnes_green':'Green','fdmnes_fdm':'FDM','feff':'FEFF','vasp_pbe':'PBE','vasp_gw':'GW-PAW'};lines=[]
for _,r in m.iterrows():
 mat=r.material.replace('Cu2O','Cu$_2$O');v='Cu-cal.' if r.comparison_type=='Cu_calibrated_transfer' else 'feature';lines.append(f"{mat} & {method_labels[r.method]} & {v} & {r.shift_eV:.4f} & {r.cosine_similarity:.4f} & {r.Pearson_r:.4f} & {r.Spearman_rho:.4f}\\\\")
si=P/'Supporting_Information.tex';s=si.read_text();start=s.index('Cu & Green & Cu-cal.');end=s.index('\\bottomrule\\end{longtable}',start);s=s[:start]+'\n'.join(lines)+'\n'+s[end:];si.write_text(s)
# Rebuild both SI comparison grids; no original repository checkout is needed.
methods=list(method_labels);labels=['FDMNES green','FDMNES fdm','FEFF','VASP PBE','VASP GW-PAW'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#C59000']
for kind,name in [('first_feature_shape_diagnostic','SI_reference_grid'),('Cu_calibrated_transfer','SI_reference_Cu_calibrated')]:
 f,axes=plt.subplots(3,5,figsize=(10,6.8),sharex=True,sharey=True)
 for i,mat in enumerate(['Cu','Cu2O','CuO']):
  for j,method in enumerate(methods):
   ax=axes[i,j];z=a[(a.material==mat)&(a.method==method)&(a.comparison_type==kind)]
   if len(z):
    r=m[(m.material==mat)&(m.method==method)&(m.comparison_type==kind)].iloc[0];fallback=str(r.alignment_status).startswith('early feature not resolved');ax.plot(z.energy_eV,z.experiment_normalized,c='black',lw=1.2,label='Experiment');ax.plot(z.energy_eV,z.calculation_normalized,c=colors[j],lw=1.2,ls=':' if fallback else '-',label=labels[j]+(' (Cu offset)' if fallback else ''));ax.text(.03,.03,f'cos {r.cosine_similarity:.3f}\nr {r.Pearson_r:.3f}; ρ {r.Spearman_rho:.3f}',transform=ax.transAxes,fontsize=7);ax.legend(fontsize=5.5)
   else:ax.text(.5,.5,'No completed\nnumeric response',ha='center',va='center',transform=ax.transAxes,fontsize=8)
   if i==0:ax.set_title(labels[j],fontsize=9)
   if j==0:ax.set_ylabel(mat.replace('Cu2O','Cu₂O')+'\nIntensity (a.u.)',fontsize=8)
   if i==2:ax.set_xlabel('Photon energy (eV)',fontsize=8)
   ax.set(xlim=(8975,9015),ylim=(0,1.08));ax.tick_params(labelsize=7)
 f.tight_layout();f.savefig(P/'figures'/(name+'.png'),dpi=350,bbox_inches='tight');plt.close(f)
print(json.dumps({'Cu_PBE_shift_change_eV':change,'Cu_PBE_main_metrics':records[1]},indent=2))
