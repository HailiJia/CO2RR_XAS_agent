"""Paired errors from committed fixed joint controls; no model search/refitting."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from tools.spectral_multilabel import SPECIES,f1_metrics
from tools.coverage_study import dump,sha
B=Path('study/signed_shape_detectability_v3');D=Path('study/dataset_20261029');I=Path('study/relative_identity_guarded_v2/results');r=json.load(open(I/'source_manifest.json'));fammap=json.load(open(I/'family_map.json'))['families'];fam=np.array([fammap[a['configuration']] for a in r]);y=np.load(I/'targets_folds.npz')['truth'];mol=json.load(open(D/'molecule_audit.json'));geom={a['configuration']:a for a in json.load(open(D/'geometry_arrays.json'))};by={}
for m in mol:by.setdefault(m['configuration'],[]).append(m)
meta=[]
for i,a in enumerate(r):
 ms=by.get(a['configuration'],[]);dist=[m['interface_distance_A'] for m in ms if m.get('interface_distance_A') is not None];motifs=[]
 for m in ms:
  if m['species']=='CO':
   symbols=geom[a['configuration']]['atoms'];carbon=[j for j in m['atom_indices'] if symbols[j-1]=='C'];c=[t for t in m.get('metal_contacts_under_3A',[]) if t[0] in carbon and t[3]<=2.5];ncu=sum(t[2]=='Cu' for t in c);nau=sum(t[2]=='Au' for t in c);n=ncu+nau;motifs.append(f"CO_Ccontacts{n}_Cu{ncu}_Au{nau}")
  if m['species']=='OCHO':motifs.append('OCHO_Ocontacts'+str(sum(t[2] in ['Cu','Au'] and t[3]<=2.5 and geom[a['configuration']]['atoms'][t[0]-1]=='O' for t in m.get('metal_contacts_under_3A',[]))))
 meta.append({'index':i,'configuration':a['configuration'],'near_group':fam[i],'domain':a['domain'],'identity_set':';'.join(a['identity_labels']),'molecules':sum(a['observed_counts'].values()),'CO_loading_all_top_metals':a['coverage_ML'],'motif_contact_proxy':';'.join(sorted(set(motifs))) or 'not_CO_or_OCHO','minimum_interface_distance_A':min(dist) if dist and a['domain']=='CuAu' else None,'record_sha256':a['sample_record_sha256']})
pd.DataFrame(meta).to_csv(B/'configuration_descriptors.csv',index=False)
metrics=[];labels=[];changes=[];mixtures=[];support=[]
for j,sp in enumerate(SPECIES):
 ids=np.flatnonzero(y[:,j]);pure=np.array([i for i in ids if r[i]['identity_labels']==[sp]],int);support.append({'species':sp,'positive_records':len(ids),'positive_near_groups':len(set(fam[ids])),'pure_records':len(pure),'pure_near_groups':len(set(fam[pure])),'individual_molecules':sum(a['observed_counts'].get(sp,0) for a in r)})
pd.DataFrame(support).to_csv(B/'support_records_groups_molecules.csv',index=False)
reps=['native_normalized','corrected_normalized','common_raw','relative_delta','signed_shape'];arrays={}
for rep in reps:
 p=I/f'fixed_rep_joint130_{rep}_trees.npz';q=np.load(p);assert np.array_equal(q['indices'],np.arange(130));arrays[rep]=q
 d=pd.DataFrame(meta);masks={'all':np.ones(130,bool),'CuAu':d.domain=='CuAu','Cu100':d.domain=='Cu100','single':y.sum(axis=1)==1,'mixed':y.sum(axis=1)>1,'clean':y.sum(axis=1)==0,'CO_loading_le0p1':(y[:,SPECIES.index('CO')]>0)&(d.CO_loading_all_top_metals<=.1),'CO_loading_gt0p1':(y[:,SPECIES.index('CO')]>0)&(d.CO_loading_all_top_metals>.1),'interface_le1p5_A':(d.domain=='CuAu')&(d.minimum_interface_distance_A<=1.5),'interface_gt1p5_A':(d.domain=='CuAu')&(d.minimum_interface_distance_A>1.5)}
 for motif in sorted(set(d.motif_contact_proxy)):masks['motif='+motif]=d.motif_contact_proxy==motif
 for subset,mask in masks.items():
  ids=np.flatnonzero(mask)
  if not len(ids):continue
  yy=y[ids];pp=q['prediction'][ids];w=np.array([1/sum(fam[ids]==f) for f in fam[ids]])
  for weightname,weight in [('record',np.ones(len(ids))),('near_group',w)]:
   pr,re,ff,su=precision_recall_fscore_support(yy,pp,average=None,zero_division=0,sample_weight=weight);tp=np.sum(yy*pp*weight[:,None]);micro=2*tp/max(1e-20,np.sum((yy+pp)*weight[:,None]));metrics.append({'representation':rep,'subset':subset,'weighting':weightname,'n_records':len(ids),'n_near_groups':len(set(fam[ids])),'micro_f1':micro,'macro_f1':ff.mean(),'exact_set_accuracy':np.average(np.all(yy==pp,axis=1),weights=weight),'false_positive_labels':np.sum((yy==0)*(pp==1)*weight[:,None]),'omitted_labels':np.sum((yy==1)*(pp==0)*weight[:,None])})
   for k,sp in enumerate(SPECIES):labels.append({'representation':rep,'subset':subset,'weighting':weightname,'species':sp,'precision':pr[k],'recall':re[k],'f1':ff[k],'support':su[k]})
  if subset in ['CuAu','Cu100']:
   jj=[SPECIES.index(a) for a in ['CO','H','O','OH']];metrics.append({'representation':rep,'subset':subset+'_common4','weighting':'record','n_records':len(ids),'n_near_groups':len(set(fam[ids])),**f1_metrics(yy[:,jj],pp[:,jj])})
 for i in np.flatnonzero(y.sum(axis=1)>1):
  pred=q['prediction'][i];mixtures.append({**meta[i],'representation':rep,'predicted_set':';'.join(sp for sp,v in zip(SPECIES,pred) if v),'omitted':';'.join(sp for j,sp in enumerate(SPECIES) if y[i,j] and not pred[j]),'false_positive':';'.join(sp for j,sp in enumerate(SPECIES) if not y[i,j] and pred[j]),'exact':bool(np.array_equal(pred,y[i]))})
a=arrays['relative_delta']['prediction'];b=arrays['signed_shape']['prediction']
for i in range(130):
 changes.append({**meta[i],'delta_set':';'.join(sp for sp,v in zip(SPECIES,a[i]) if v),'shape_set':';'.join(sp for sp,v in zip(SPECIES,b[i]) if v),'delta_exact':bool(np.array_equal(a[i],y[i])),'shape_exact':bool(np.array_equal(b[i],y[i])),'recovered_labels':';'.join(sp for j,sp in enumerate(SPECIES) if y[i,j] and b[i,j] and not a[i,j]),'newly_omitted_labels':';'.join(sp for j,sp in enumerate(SPECIES) if y[i,j] and a[i,j] and not b[i,j]),'removed_false_positives':';'.join(sp for j,sp in enumerate(SPECIES) if not y[i,j] and a[i,j] and not b[i,j]),'new_false_positives':';'.join(sp for j,sp in enumerate(SPECIES) if not y[i,j] and b[i,j] and not a[i,j])})
pd.DataFrame(metrics).to_csv(B/'existing_representation_strata.csv',index=False);pd.DataFrame(labels).to_csv(B/'existing_per_label.csv',index=False);pd.DataFrame(mixtures).to_csv(B/'mixture_predictions.csv',index=False);pd.DataFrame(changes).to_csv(B/'paired_prediction_changes.csv',index=False);dump(B/'existing_prediction_hashes.json',{rep:sha(I/f'fixed_rep_joint130_{rep}_trees.npz') for rep in reps})
for rep in ['relative_delta','signed_shape']:
 p=arrays[rep]['prediction'];mask=np.array([a['domain']=='CuAu' for a in r])&(y.sum(axis=1)>1);print(rep,'CuAu mixture exact',int(np.all(p[mask]==y[mask],axis=1).sum()),'/',int(mask.sum()))
print(pd.DataFrame(metrics).query("subset in ['all','CuAu','Cu100','Cu100_common4']").to_string(index=False))
