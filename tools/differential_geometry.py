"""Configuration-level contact/location regression, never molecular pseudo-samples."""
from pathlib import Path
import json
from collections import Counter
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from .differential_training import banks,reg_estimator
from .coverage_inference import contacts
from .component_audit import audit_components
from .progress_snapshots import write_json

TARGETS=['mean_CO_Cu_contacts','mean_CO_Au_contacts','fraction_Cu3','fraction_Cu2Au','fraction_CuAu2','fraction_Au3','fraction_atop','fraction_bridge','mean_C_boundary_distance_A']

def boundary_distances(s):
 a=np.array(s['atoms']);p=np.asarray(s['positions']);cell=np.asarray(s['cell']);normal=np.cross(cell[0],cell[1]);normal/=np.linalg.norm(normal);height=p@normal;top=[]
 for element in ['Cu','Au']:
  ids=sorted(np.flatnonzero(a==element),key=lambda i:height[i]);groups=[]
  for i in ids:
   if not groups or height[i]-height[groups[-1][-1]]>.6:groups.append([])
   groups[-1].append(i)
  if groups:top.extend(groups[-1])
 frac=p@np.linalg.inv(cell);rows=[]
 for i in sorted(top,key=lambda i:frac[i,1]%1):
  y=frac[i,1]%1
  if not rows or (y-rows[-1][-1][1])*np.linalg.norm(cell[1])>.6:rows.append([])
  rows[-1].append((i,y))
 centers=[np.mean([y for i,y in row]) for row in rows];types=[Counter(a[i] for i,y in row).most_common(1)[0][0] for row in rows];edges=[(centers[i]+((centers[(i+1)%len(rows)]-centers[i])%1)/2)%1 for i in range(len(rows)) if types[i]!=types[(i+1)%len(rows)]]
 if not edges:return None
 perpendicular=np.linalg.norm(cell[1]-np.dot(cell[1],cell[0])/np.dot(cell[0],cell[0])*cell[0]);ci=np.flatnonzero(a=='C');return [min(min(abs((frac[i,1]-e)%1),1-abs((frac[i,1]-e)%1)) for e in edges)*perpendicular for i in ci]

def prepare_targets(study,source_geometry):
 B=Path(study);geometry={x['configuration']:x for x in json.loads(Path(source_geometry).read_text())};rows=[]
 for r in json.loads((B/'pair_manifest.json').read_text()):
  if 'pair_index' not in r or not r['CO_only']:continue
  s=geometry[r['configuration']];c=contacts(s,2.25);n=len(c);d=boundary_distances(s);v=[np.mean([x for x,y in c]),np.mean([y for x,y in c])]+[sum(v==key for v in c)/n for key in [(3,0),(2,1),(1,2),(0,3)]]+[sum(sum(v)==k for v in c)/n for k in [1,2]]+[np.mean(d) if d else None];rows.append({'configuration':r['configuration'],'pair_index':r['pair_index'],'group':r['group'],'eligible_primary':r['eligible_primary'],'targets':dict(zip(TARGETS,v)),'contacts_per_molecule_2p25A':c,'contacts_2p15A':contacts(s,2.15),'contacts_2p35A':contacts(s,2.35),'carbon_boundary_distances_A':d,'target_source':'actual saved XAS geometry; structural quantities never enter XAS predictors'})
 write_json(B/'geometry_targets.json',rows);return rows

def run(study):
 B=Path(study);O=B/'geometry';O.mkdir(exist_ok=True);targets=json.loads((B/'geometry_targets.json').read_text());z=np.load(B/'paired_arrays.npz');bf=banks(z);summary=[];predrows=[]
 for scope,rows in [('matched',[r for r in targets if r['eligible_primary']]),('same_substrate_sensitivity',targets)]:
  ids=np.array([r['pair_index'] for r in rows]);groups=np.array([r['group'] for r in rows]);Y=np.array([[r['targets'][k] for k in TARGETS] for r in rows],float);valid=np.isfinite(Y).all(axis=0);p0=np.zeros_like(Y[:,valid]);folds=list(GroupKFold(min(5,len(rows))).split(ids,groups=groups))
  for rep in ['corrected_normalized','raw_physical','delta','clean_plus_delta']:
   pred=np.zeros_like(Y[:,valid])
   for fold,(tr,te) in enumerate(folds):
    X=bf[rep]['DCT16'];model=reg_estimator(('DCT16','ridge',10.)).fit(X[ids[tr]],Y[tr][:,valid]);pred[te]=model.predict(X[ids[te]]);p0[te]=Y[tr][:,valid].mean(axis=0);path=O/'models'/f'{scope}_{rep}_fold{fold}.joblib';path.parent.mkdir(exist_ok=True);joblib.dump({'model':model,'task':'geometry','representation':rep,'recipe':('DCT16','ridge',10.),'energy':z['energy'],'target':np.array(TARGETS)[valid].tolist(),'training_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'test_pair_indices':ids[te].tolist(),'scope':scope,'reference_ids':sorted(set(r['reference_id'] for r in json.loads((B/'pair_manifest.json').read_text()) if r.get('pair_index') in ids[tr])),'reference_transformation':'equal physical Cu; common clean scale; signed DCT16; training-only scaling','units':'contacts, fractions, Angstrom respectively','physical_top_layers':[4,5],'surface_Cu_fraction':36/68},path)
   for j,name in enumerate(np.array(TARGETS)[valid]):summary.append({'scope':scope,'representation':rep,'target':name,'n_configurations':len(rows),'MAE':mean_absolute_error(Y[:,valid][:,j],pred[:,j]),'training_mean_MAE':mean_absolute_error(Y[:,valid][:,j],p0[:,j]),'R2':r2_score(Y[:,valid][:,j],pred[:,j]) if np.ptp(Y[:,valid][:,j]) else None,'positive_configuration_support':int(np.count_nonzero(Y[:,valid][:,j]))})
   np.savez_compressed(O/f'{scope}_{rep}_oof.npz',truth=Y[:,valid],prediction=pred,pair_indices=ids,target=np.array(TARGETS)[valid]);predrows.extend({'scope':scope,'representation':rep,'configuration':r['configuration'],'truth':y.tolist(),'prediction':p.tolist()} for r,y,p in zip(rows,Y[:,valid],pred))
 pd.DataFrame(summary).to_csv(O/'metrics.csv',index=False);write_json(O/'predictions.json',predrows);write_json(O/'protocol.json',{'recipe':'fixed DCT16/Ridge alpha10, training-only constant removal and scaling; no outer-score selection','cutoffs_A':[2.15,2.25,2.35],'location':'mean carbon distance to periodic Cu/Au top-row boundaries; each element top layer identified separately','pure_Cu_location':'undefined rather than zero; no Cu100 differential model without clean reference','independent_samples':'one physical configuration, not each CO molecule','Au_rich_support':'CuAu2/Au3 CO hollow environments absent; zero-target errors do not validate their recovery'})
 return len(summary)
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--prepare-from');a=p.parse_args()
 if a.prepare_from:prepare_targets(a.study,a.prepare_from)
 print(run(a.study))
