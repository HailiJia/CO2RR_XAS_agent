"""Spectral/structural neighbor audit; metadata never enters the classifier."""
from pathlib import Path
import json,gzip,tarfile,hashlib,sys
import numpy as np,pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.component_audit import read_poscar,audit_components
from tools.chemical_lineage import local_info,parent_key
from tools.spectral_multilabel import feature_bank,SPECIES
B=ROOT/'study/retraining_20261022';OUT=ROOT/'study/top_two_20261023/diagnosis'

def main():
 OUT.mkdir(parents=True,exist_ok=True);d=B/'refit/dataset';z=np.load(d/'cohort.npz');cv=pd.read_csv(d/'cv_manifest.csv');query='2CO_adjacent/2CO_adjacent_0p056_AuCu_interface';qi=list(z['configuration']).index(query);old=cv.prior_configuration.notna().to_numpy();ids=np.flatnonzero(old);x=z['energy'];raw=np.array([np.average(a[[3,2]],axis=0,weights=p[[3,2]]) for a,p in zip(z['stack'],z['populations'])]);bank=feature_bank(raw,x);manifest={r['path']:r for r in json.load(open(d/'source_manifest.json'))};metadata=[]
 with tarfile.open(d/'source_evidence.tar') as archive:
  def get(path):
   r=manifest[path];b=gzip.decompress(archive.extractfile('source_objects/'+r['sha256']+'.gz').read());assert hashlib.sha256(b).hexdigest()==r['sha256'];return b,r['sha256']
  for i,row in cv.iterrows():
   alias=row.configuration;p=OUT/'geometries';p.mkdir(exist_ok=True);b,h=get(alias+'/XAS/FDMNES/green/layer_5/CONTCAR');file=p/(str(i)+'.vasp');file.write_bytes(b);s=read_poscar(file);a=audit_components(s);initial,ih=get(alias+'/POSCAR');ip=p/(str(i)+'_initial.vasp');ip.write_bytes(initial);parent=parent_key(read_poscar(ip));ads=np.flatnonzero(np.isin(s['atoms'],['C','O','H']))+1;info=local_info(s,ads);normal=np.cross(s['cell'][0],s['cell'][1]);normal/=np.linalg.norm(normal);height=s['positions']@normal;el=np.array(s['atoms']);cu=np.flatnonzero((el=='Cu')&(height>height[el=='Cu'].max()-.6));carbons=np.flatnonzero(el=='C')+1;contacts=[r for r in info['metal_contacts_under_3A'] if r[0] in carbons];by={str(int(c)):{str(cut):[{'metal_index':r[1],'element':r[2],'distance_A':r[3]} for r in contacts if r[0]==c and r[3]<=cut] for cut in [2.,2.25,2.5,2.75]} for c in carbons}
   metadata.append({'index':i,'configuration':alias,'geometry_sha256':h,'initial_sha256':ih,'initial_parent_group':parent,'counts':a['counts'],'identities':a['identities'],'CO_molecules':a['counts'].get('CO',0),'surface_Cu_sites':len(cu),'CO_coverage_per_surface_Cu':a['counts'].get('CO',0)/len(cu),'interface_distance_A':info['interface_distance_A'],'C_metal_contacts_by_cutoff':by,'geometry_file':str(file.relative_to(ROOT))})
 m=metadata[qi];rows=[]
 for feature in ['normalized','derivative']:
  a=bank[feature];dist=cdist(a[[qi]],a[ids],metric='cosine')[0];eu=cdist(a[[qi]],a[ids],metric='euclidean')[0];loo=cdist(a[ids],a[ids],metric='cosine');np.fill_diagonal(loo,np.inf);mins=loo.min(axis=1)
  for j,k in enumerate(ids):
   t=metadata[k];rows.append({'representation':feature,'training_configuration':t['configuration'],'training_index':int(k),'cosine_distance':float(dist[j]),'euclidean_distance':float(eu[j]),'same_initial_parent_group':t['initial_parent_group']==m['initial_parent_group'],'same_CO_count':t['CO_molecules']==m['CO_molecules'],'CO_only':t['identities']==['CO'],'CO_molecules':t['CO_molecules'],'CO_coverage_per_surface_Cu':t['CO_coverage_per_surface_Cu'],'interface_distance_A':t['interface_distance_A'],'query_nearest_distance_percentile_vs_training_LOO':float((mins<=dist.min()).mean()*100),'pearson_r':float(pearsonr(a[qi],a[k]).statistic)})
 result=pd.DataFrame(rows).sort_values(['representation','cosine_distance']);result.to_csv(OUT/'all_training_neighbors.csv',index=False);(OUT/'local_geometry_metadata.json').write_text(json.dumps({'query':m,'training_and_query_metadata':metadata,'contacts_definition':'distance summaries at 2.0/2.25/2.5/2.75 A; audits only, not inferred bond order or classifier predictors','interface_definition':info['interface_method']},indent=2)+'\n')
 selected=result[(result.representation=='normalized')&result.CO_only&result.same_CO_count].head(4);general=result[result.representation=='normalized'].head(4);selected=pd.concat([selected,general]).drop_duplicates('training_index');selected.to_csv(OUT/'review_neighbors.csv',index=False)
 figure,axs=plt.subplots(2,1,figsize=(9,6.5),sharex=True);axs[0].plot(x,bank['normalized'][qi],c='black',lw=2,label='Missed interface 2CO');arrays=[{'configuration':query,'energy_eV':float(e),'normalized_intensity_per_eV':float(v),'raw_top_two_per_absorber':float(r)} for e,v,r in zip(x,bank['normalized'][qi],raw[qi])]
 for row in selected.itertuples():
  k=row.training_index;line=axs[0].plot(x,bank['normalized'][k],lw=1,label=metadata[k]['configuration'])[0];axs[1].plot(x,bank['normalized'][qi]-bank['normalized'][k],lw=1,c=line.get_color(),label='Query − '+metadata[k]['configuration']);arrays += [{'configuration':metadata[k]['configuration'],'energy_eV':float(e),'normalized_intensity_per_eV':float(v),'raw_top_two_per_absorber':float(r)} for e,v,r in zip(x,bank['normalized'][k],raw[k])]
 axs[0].set(ylabel='Unit-area intensity (eV⁻¹)',title='Top-two-layer spectra: query and old-training neighbors');axs[0].legend(fontsize=6);axs[1].axhline(0,c='grey',lw=.7,label='Zero difference');axs[1].set(xlabel='Photon energy (eV)',ylabel='Normalized query − neighbor');figure.tight_layout()
 for ext in ['pdf','svg','png']:figure.savefig(OUT/('neighbor_spectra.'+ext),dpi=350,bbox_inches='tight')
 plt.close(figure);pd.DataFrame(arrays).to_csv(OUT/'plotted_neighbor_arrays.csv',index=False)
 oldpred=np.load(B/'refit/models/new_configuration_holdout.npz');j=list(oldpred['configuration']).index(query);model=__import__('joblib').load(B/'refit/models/new_configuration_holdout.joblib');scores={s:{'score':float(oldpred['score'][j,k]),'threshold':float(model['thresholds'][k])} for k,s in enumerate(SPECIES)}
 summary={'query':query,'original_missed_model':{'depth':model['depth'],'feature':model['feature'],'scores':scores},'comparison':'all 87 old configurations, excluding all three additions/recovered cases','query_geometry':m,'closest_matched_CO_coverage':selected.to_dict('records'),'weak_Cu_perturbation':'not established; compatible clean-parent spectrum is absent, so adsorbate-induced amplitude cannot be isolated','unfamiliar_environment':'spectral distances and geometric descriptors are descriptive; they do not establish a cause of the missed label','difference_plot':'independently unit-area-normalized descriptive query-neighbor difference; not an adsorbate-minus-clean physical contrast'};(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
if __name__=='__main__':main()
