"""Review matched ASE snapshots and molecular graphs at visible atom scale."""
from pathlib import Path
import json,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import dump
from tools.contact_checks import minimum_image_vectors
B=ROOT/'study/coverage_20261026';O=B/'geometry_review';O.mkdir(exist_ok=True);rows=json.loads((B/'record_geometry_audit.json').read_text());colors={'Cu':'#5c91b3','Au':'#d7b13d','C':'#303030','O':'#c84c48','H':'#f5f5f5'}
cu=[r for r in rows if r['domain']=='Cu100'];summary=[]
for r in cu:
 a=r['ASE_lineage'];s=a['structure'];p=np.asarray(s['positions']);els=np.array(s['atoms']);graph=a['observed_graph'];original=p.copy()
 for m in graph['components']:
  todo=[m['atom_indices'][0]-1];seen={todo[0]};edges=[(i-1,j-1) for i,j in m['bonds']]
  while todo:
   i=todo.pop()
   for aa,bb in edges:
    j=bb if aa==i else aa if bb==i else None
    if j is not None and j not in seen:p[j]=p[i]+minimum_image_vectors(original[j]-original[i],s['cell'],(True,True,False));seen.add(j);todo.append(j)
 x=p[:,0]+.28*p[:,1];y=p[:,2]+.1*p[:,1];ids=np.flatnonzero((p[:,2]>p[els=='Cu',2].max()-2.5)|(els!='Cu'));fig,ax=plt.subplots(figsize=(6,4.7))
 for i in ids:ax.scatter(x[i],y[i],s=130 if els[i]!='H' else 85,color=colors[els[i]],edgecolor='.35',zorder=3)
 for m in graph['components']:
  for i,j in m['bonds']:ax.plot(x[np.array([i,j])-1],y[np.array([i,j])-1],color='.2',lw=1.6,zorder=2)
  for i in m['atom_indices']:ax.text(x[i-1]+.08,y[i-1]+.08,els[i-1]+str(i),fontsize=8)
 ax.set_aspect('equal');ax.set(xlabel='Oblique x + 0.28y (Å)',ylabel='z + 0.10y (Å)',title=r['configuration'].split('/')[-1].replace('Cu100_2x2x4_','')+'\nOriginal ASE snapshot; matched XAS input')
 fig.text(.5,.01,'Cu blue; C dark; O red; H white. Solid lines: measured molecular bonds.\nIndices follow original ASE order. This view does not establish VASP run termination.',ha='center',fontsize=8);fig.tight_layout(rect=[0,.06,1,1]);name=r['configuration'].split('/')[-1]
 for ext in ['pdf','svg','png']:fig.savefig(O/(name+'.'+ext),dpi=250,bbox_inches='tight')
 plt.close(fig)
 # An explicitly derived coordinate export, never an original relaxed CONTCAR.
 elements=list(dict.fromkeys(els));order=np.array([i for e in elements for i,v in enumerate(els) if v==e]);frac=original@np.linalg.inv(np.asarray(s['cell']));text=['DERIVED from original ASE snapshot; not an original VASP CONTCAR','1.0']+[' '.join(f'{v:.12f}' for v in row) for row in s['cell']]+[' '.join(elements),' '.join(str(sum(els==e)) for e in elements),'Direct']+[' '.join(f'{v:.12f}' for v in frac[i]) for i in order];(O/(name+'_derived.vasp')).write_text('\n'.join(text)+'\n');summary.append({'configuration':r['configuration'],'ASE_path':a['path'],'ASE_sha256':a['sha256'],'max_free_force_eV_per_A':a['max_free_force_eV_per_A'],'XAS_match_max_A':a['atom_mapping']['max_distance_A'],'graph_status':graph['status'],'identities':graph['counts'],'force_and_energy_snapshot_is_not_run_termination':True})
dump(O/'ASE_snapshot_audit.json',summary)
