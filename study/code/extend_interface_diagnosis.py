"""Contact-matched and actual failed-representation neighbor diagnostics."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spectral_multilabel import feature_for_records
from tools.component_audit import read_poscar
from tools.contact_checks import minimum_image_vectors
B=ROOT/'study/top_two_20261023/diagnosis'


def signature(m,cut='2.25'):
 return sorted([(sum(x['element']=='Cu' for x in v[cut]),sum(x['element']=='Au' for x in v[cut])) for v in m['C_metal_contacts_by_cutoff'].values()])


def main():
 data=ROOT/'study/retraining_20261022/refit/dataset';z=np.load(data/'cohort.npz');cv=pd.read_csv(data/'cv_manifest.csv');old=np.flatnonzero(cv.prior_configuration.notna());meta=json.load(open(B/'local_geometry_metadata.json'));query=meta['query'];qi=query['index'];rows=[]
 for m in meta['training_and_query_metadata']:
  if m['index'] not in old:continue
  if m['identities']==['CO'] and m['CO_molecules']==query['CO_molecules']:
   rows.append({'configuration':m['configuration'],'index':m['index'],'CO_molecules':m['CO_molecules'],'CO_coverage':m['CO_coverage_per_surface_Cu'],'interface_distance_A':m['interface_distance_A'],'query_contact_signature':json.dumps(signature(query)),'training_contact_signature':json.dumps(signature(m)),'same_contact_signature':signature(query)==signature(m),'same_initial_metal_coordinate_group':m['initial_parent_group']==query['initial_parent_group']})
 pd.DataFrame(rows).to_csv(B/'same_coverage_binding_contacts.csv',index=False)
 records=[];pop=[]
 for a,p in zip(z['stack'],z['populations']):
  records.append({'measurement':{'series':[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'unit':'eV','values':z['energy'].tolist()}],'channels':[{'name':'absorption_cross_section','values':a[l-2].tolist()}]} for l in [2,3,4,5]]}});pop.append({l:int(p[l-2]) for l in [2,3,4,5]})
 a=feature_for_records(records,pop,z['energy'],'depth_channels','multiscale');dist=cdist(a[[qi]],a[old],metric='cosine')[0];eu=cdist(a[[qi]],a[old],metric='euclidean')[0];loo=cdist(a[old],a[old],metric='cosine');np.fill_diagonal(loo,np.inf);pct=float((loo.min(axis=1)<=dist.min()).mean()*100);pd.DataFrame({'training_configuration':z['configuration'][old],'cosine_distance':dist,'euclidean_distance':eu,'query_nearest_distance_percentile_vs_training_LOO':pct}).sort_values('cosine_distance').to_csv(B/'failed_multiscale_model_neighbors.csv',index=False)
 ids=[qi]+[r['index'] for r in rows];fig,axs=plt.subplots(1,len(ids),figsize=(5*len(ids),4.8),squeeze=False)
 for ax,i in zip(axs.flat,ids):
  m=meta['training_and_query_metadata'][i];s=read_poscar(ROOT/m['geometry_file']);p=s['positions'];el=np.array(s['atoms']);cell=s['cell'];ads=np.flatnonzero(np.isin(el,['C','O','H']));center=p[ads].mean(axis=0);q=minimum_image_vectors(p-center,cell,(True,True,False));xy=np.column_stack([q[:,0]-.35*q[:,1],q[:,2]+.25*q[:,1]]);metal=np.flatnonzero(np.isin(el,['Cu','Au'])&(np.linalg.norm(q,axis=1)<5));selected=np.r_[metal,ads]
  for c,sites in m['C_metal_contacts_by_cutoff'].items():
   for contact in sites['2.25']:
    a,b=int(c)-1,int(contact['metal_index'])-1;ax.plot(*xy[[a,b]].T,ls='--',lw=1,c='#777777',zorder=1)
  for c in np.flatnonzero(el=='C'):
   oxy=np.flatnonzero(el=='O');j=oxy[np.argmin(np.linalg.norm(minimum_image_vectors(p[oxy]-p[c],cell,(True,True,False)),axis=1))];ax.plot(*xy[[c,j]].T,c='#333333',lw=2,zorder=2)
  colors={'Cu':'#4B92C3','Au':'#DDB644','C':'#333333','O':'#D84E47','H':'white'}
  for j in selected:
   ax.scatter(*xy[j],s=100 if el[j] in ['Cu','Au'] else 150,c=colors[el[j]],edgecolors='#333333',linewidths=.6,zorder=3)
   if j in ads:ax.annotate(f'{el[j]}{j+1}',xy[j],xytext=(5,7),textcoords='offset points',fontsize=7)
  ax.set_aspect('equal');ax.set(xlim=(-5,5),ylim=(-3.2,3.5),xlabel='Oblique local coordinate (Å)',ylabel='Height + oblique coordinate (Å)');ax.set_title(('Query' if i==qi else 'Same-loading old CO')+'\n'+m['configuration'].split('/')[-1],fontsize=9)
 fig.tight_layout(rect=(0,.08,1,1));fig.text(.5,.01,'Cu blue; Au gold; C dark; O red. Solid: molecular C–O. Dashed: C–metal distances ≤2.25 Å, a contact guide rather than electronic bond order.',ha='center',fontsize=8)
 for ext in ['pdf','svg','png']:fig.savefig(B/('local_contact_views.'+ext),dpi=300,bbox_inches='tight')
 plt.close(fig);summary=json.load(open(B/'summary.json'));summary.update(failed_multiscale_nearest_distance_percentile=pct,query_contact_signature_2p25A=signature(query),same_CO_coverage_training_cases=len(rows),exact_contact_signature_matches=sum(r['same_contact_signature'] for r in rows),contact_signature_caveat='count of nearby Cu/Au at explicit distance cutoff, not a binding-energy or electronic-bond assignment',initial_parent_caveat='exact initial coordinate keys are not verified independent ancestral parents');(B/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'failed_representation_percentile':pct,'same_loading_cases':rows},indent=2))
if __name__=='__main__':main()
