"""Render live-export checkpoints distinctly from generated seeds and training data."""
from pathlib import Path
import sys,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.contact_checks import minimum_image_vectors
B=ROOT/'study/differential_v1/checkpoints';F=B/'figures';F.mkdir(exist_ok=True)
r=json.loads((B/'case_audit.json').read_text());table=[]
for x in r:
 if x['kind']!='captured_relaxation_checkpoint':continue
 c=x['current'];i=x['initial'];table.append({'case':x['case'],'snapshot_UTC':x['capture_utc'],'basis':x['snapshot_basis'],'ionic_step_current_OUTCAR':x['ionic_step_in_captured_OUTCAR'],'OSZICAR_step':x['OSZICAR_last_completed_step'],'restart_POSCAR_CC_A':i['CC_A'],'current_CC_A':c['CC_A'],'CO_distances_A':json.dumps(c['CO_core_distances_A']),'proton_transfer_candidates':json.dumps(c['protonated_dimer_candidates']),'binding_motif':json.dumps(c['binding_motif']),'verdict':c['verdict'],'force_last_evaluation_eV_A':x['max_free_force_eV_A'],'run_status':x['run_status'],'fixed_C_flags':json.dumps(x['fixed_C_flags'])})
 s=x['current_structure'];p=np.asarray(s['positions']);a=np.array(s['atoms']);carbon=np.array(c['C_indices_1based'])-1;center=p[carbon].mean(axis=0);q=center+minimum_image_vectors(p-center,s['cell'],(True,True,False));ids=np.flatnonzero((np.linalg.norm((q-center)[:,:2],axis=1)<6)&((q[:,2]>center[2]-4)|(np.isin(a,['C','O','H','K']))));colors={'Cu':'#5189ac','Au':'#d7b54b','C':'#303030','O':'#c84b46','H':'#f7f7f7','K':'#9251a6'};fig,ax=plt.subplots(1,2,figsize=(11,4.2))
 for k,axes in enumerate([(0,1),(0,2)]):
  for j in ids:ax[k].scatter(q[j,axes[0]],q[j,axes[1]],s=90 if a[j]!='H' else 40,c=colors.get(a[j],'.5'),edgecolor='.4',lw=.6)
  for j in np.flatnonzero(np.isin(a,['C','O','H','K'])):ax[k].text(q[j,axes[0]]+.06,q[j,axes[1]]+.06,a[j]+str(j+1),fontsize=7)
  ax[k].plot(q[carbon,axes[0]],q[carbon,axes[1]],ls=':',c='black',lw=1,label=f'C–C distance {c["CC_A"]:.2f} Å (guide, not a bond)');ax[k].legend(fontsize=7);ax[k].set_aspect('equal');ax[k].set_xlabel('x (Å)');ax[k].set_ylabel(('y' if k==0 else 'z')+' (Å)')
 fig.suptitle(x['case'].split('/')[-1]+' — UNFINISHED checkpoint\n'+c['verdict'],fontsize=11);fig.tight_layout();fig.savefig(F/(x['case'].replace('/','__')+'_geometry.png'),dpi=300);plt.close(fig)
 t=x['trajectory'];fig,ax=plt.subplots(figsize=(6.2,3.7));ax.plot([v['ionic_step_in_captured_OUTCAR'] for v in t],[v['CC_A'] for v in t],marker='o',ms=3,label='C–C');ax.plot([v['ionic_step_in_captured_OUTCAR'] for v in t],[min((h['O_H_A'] for h in v['proton_distances']),default=float('nan')) for v in t],ls='--',label='Shortest water O–H');ax.plot([v['ionic_step_in_captured_OUTCAR'] for v in t],[min((h['min_C_H_A'] for h in v['proton_distances']),default=float('nan')) for v in t],ls=':',label='Shortest dimer C–H');ax.axhspan(1.65,2.2,color='.7',alpha=.2,label='C–C cutoff sensitivity');ax.set(xlabel='Ionic step in captured OUTCAR (restart-local)',ylabel='Minimum-image distance (Å)',title=x['case'].split('/')[-1]+' — available trajectory');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(F/(x['case'].replace('/','__')+'_trajectory.png'),dpi=300);plt.close(fig)
with (B/'current_summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
(B/'README.md').write_text('# Current OCCO–water checkpoints\n\n'+str(len(table))+' exported relaxation checkpoints were captured read-only. These are UNFINISHED snapshots, not converged intermediates or XAS training examples. Full source timestamps/hashes are in capture_manifest.json; source bytes are uploaded in ../sources/checkpoints.tar.gz.\n\nThe supplied restart POSCAR is not assumed to be the original intended dimer. Original seed candidates, including water/K, reversed atop–bridge and C,O-bound alternatives, are separately inventoried. Starting seeds without associated outputs are not called relaxed or running.\n\nFor the three current CO2RR/OCCO_H2O runs, C–C distances are approximately 3.10–3.48 Å; actual C–O/H connectivity, contacts, force/convergence evidence and restart-local trajectories are in case_audit.json and current_summary.csv. A long C–C distance is supported by independent C–O assignment; hydrogen bonding is distinguished from a covalent O–H/C–H change.\n')
