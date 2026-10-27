"""Full SI grids, timing provenance and an actually executed portable notebook."""
from pathlib import Path
import sys,json,re,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/reference_benchmark_v3';d=pd.read_csv(B/'plotted_arrays.csv');metrics=pd.read_csv(B/'similarity_metrics.csv');methods=['fdmnes_green','fdmnes_fdm','feff','vasp_pbe','vasp_gw'];labels=['FDMNES green','FDMNES fdm','FEFF','VASP PBE','VASP GW'];colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00'];mats=['Cu','Cu2O','CuO']
for kind,name in [('first_feature_shape_diagnostic','SI_reference_grid'),('Cu_calibrated_transfer','SI_reference_Cu_calibrated')]:
 fig,axes=plt.subplots(3,5,figsize=(10,6.8),sharex=True,sharey=True)
 for i,mat in enumerate(mats):
  for j,method in enumerate(methods):
   ax=axes[i,j];z=d[(d.material==mat)&(d.method==method)&(d.comparison_type==kind)]
   if len(z):
    row=metrics[(metrics.material==mat)&(metrics.method==method)&(metrics.comparison_type==kind)].iloc[0];fallback=str(row.alignment_status).startswith('early feature not resolved');ax.plot(z.energy_eV,z.experiment_normalized,c='black',lw=1.2,label='Experiment');ax.plot(z.energy_eV,z.calculation_normalized,c=colors[j],lw=1.2,ls=':' if fallback else '-',label=labels[j]+(' (Cu offset)' if fallback else ''));ax.text(.03,.03,f'cos {row.cosine_similarity:.3f}\nr {row.Pearson_r:.3f}; ρ {row.Spearman_rho:.3f}',transform=ax.transAxes,fontsize=7);ax.legend(fontsize=6)
   else:ax.text(.5,.5,'No completed\nnumeric response',ha='center',va='center',transform=ax.transAxes,fontsize=8)
   if i==0:ax.set_title(labels[j],fontsize=9)
   if j==0:ax.set_ylabel(mat+'\nIntensity (a.u.)',fontsize=8)
   if i==2:ax.set_xlabel('Photon energy (eV)',fontsize=8)
   ax.set(xlim=(8975,9015),ylim=(-.02,1.08));ax.tick_params(labelsize=7)
 fig.tight_layout();fig.savefig(B/'figures'/f'{name}.png',dpi=350,bbox_inches='tight');plt.close(fig)
# Keep the author's CPU correction distinct from old log/run correspondence.
timing=[{'material':'CuO','method':'FDMNES fdm','stage':'XAS','reported_CPU_time_s':129600.,'elapsed_s':None,'allocated_core_time_estimate_s':None,'source_basis':'author-supplied corrected CPU time','legacy_archived_CPU_time_s':91018.9,'caveat':'legacy log/run correspondence not established; no speedup inferred'}]
for source in json.load(open(B/'current_VASP_provenance.json'))['sources']:
 for row in source['timing_by_stage']:timing.append({'material':source['material'],'method':'VASP '+source['folder_label'],'source_basis':'reported VASP CPU versus elapsed, retained separately','caveat':'reported CPU is not allocated core-time; hardware/resources/settings must accompany comparisons',**row})
# FEFF prints module CPU times; preserve them as CPU, not elapsed wall time.
for material in mats:
 folder=ROOT/'study/reference_benchmark/raw'/material/'feff'
 for p in folder.glob('*'):
  if not p.is_file() or p.suffix.lower() in ['.bin','.pdf','.png']:continue
  text=p.read_text(errors='replace')
  for line in text.splitlines():
   if 'cpu' in line.lower() and 'time' in line.lower():timing.append({'material':material,'method':'FEFF','stage':'module/log entry','source_basis':'CPU time; module aggregation not assumed comparable across codes','raw_logged_time_line':line.strip(),'source':str(p.relative_to(ROOT)),'source_sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
pd.DataFrame(timing).to_csv(B/'timing_summary.csv',index=False)
(B/'README.md').write_text('# Numeric reference benchmark v3\n\nCu2O PBE and GW-PAW inputs/OUTCAR/HDF5 responses were recovered from the current supplied folder, with independent diagonal-tensor checks. CuO PBE is retained as a missing numeric slot: the current HDF5 is unreadable and OUTCAR lacks a complete response table. Original sources, including incomplete files, are uploaded in the differential source package.\n\nAll curves use fixed-window minmax normalization on the same 8980–9015 eV grid for display and cosine/Pearson/Spearman scores. Axes preserve 8975–9015 eV, including the unextrapolated pre-window margin. Main early-feature diagnostic is separate from Cu-calibrated transferable offsets. Cu2O anchors its real first peak at 8981.2919 eV; CuO uses its first rising-edge inflection at 8983.2926 eV rather than a fictional peak at 8982 or the later white line. A dotted Cu2O FEFF curve explicitly retains Cu calibration because the early peak is unresolved.\n\nVASP retains native CH_SIGMA and adds the author-notebook Gaussian sigma1 eV (FWHM2.35482 eV), not a fitted lifetime width. Other already convolved backends are not convolved twice. Native/broadened and aligned arrays, normalization offsets/divisors, shifts and both comparison types are saved. No unsupported pre-edge subtraction or clipping of negative absorption is used. Raw experimental baseline/noise values remain unchanged.\n\nThe current Cu2O inputs use PBE+U (Ueff6 eV on Cu) and independent-particle core-hole response. Cu_GW/O_GW and IALGO labels do not establish quasiparticle GW or BSE. Method titles use GW-PAW with these settings disclosed.\n\nCuO FDMNES-fdm CPU time129600 s is retained as supplied; an older archived log value91018.9 s remains separately noted, without a claimed matched run. FEFF module entries are CPU times. VASP SCF and XAS stages remain separate from allocated core-time estimates. No measured speedup is claimed.\n')
