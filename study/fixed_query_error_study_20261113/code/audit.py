"""Read-only source audit; outputs are confined to the new study directory."""
import argparse,collections,hashlib,json,re,shutil,sys,subprocess,itertools
from pathlib import Path
import numpy as np,pandas as pd
from scipy.optimize import linear_sum_assignment
REPO=Path(__file__).resolve().parents[3];sys.path.insert(0,str(REPO))
from tools.coverage_study import deck_structure,native,layer_evidence
from tools.component_audit import read_poscar
from tools.cu100_lineage import snapshot_graph
from tools.chemical_lineage import correspondence
from tools.dataset_labels import geometry_hash
from tools.fdmnes_scale import headers
from tools.differential_xas import clean_scale
O=Path(__file__).resolve().parents[1];B=REPO/'study/dataset_20261101'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(n,x): (O/n).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
def serial(s):return {k:np.asarray(v).tolist() for k,v in s.items()}
def settings(p):
 t=p.read_text();return {'Green':bool(re.search(r'(?im)^Green\s*$',t)),'SCF':bool(re.search(r'(?im)^SCF\s*$',t)),'Perdew':bool(re.search(r'(?im)^Perdew\s*$',t)),'radius_A':float(re.search(r'(?im)^Radius\s*\n\s*([\d.]+)',t)[1])}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--old-raw',type=Path,required=True);ap.add_argument('--new-raw',type=Path,required=True);a=ap.parse_args();O.mkdir(exist_ok=True)
 old=json.load(open(B/'training_manifest.json'));oldby={r['configuration']:i for i,r in enumerate(old)};z=np.load(B/'training_arrays.npz');E=z['energy'];fm=json.load(open(B/'family_map.json'))['families'];geos={g['configuration']:{k:np.asarray(v) if k!='atoms' else v for k,v in g.items() if k!='configuration'} for g in json.load(open(B/'geometry_arrays.json'))};proof=[]
 # Preserve every old query exactly, including its originally audited lineage.
 dump('old_manifest.json',old);shutil.copyfile(B/'training_arrays.npz',O/'old_inputs.npz');dump('old_family_map.json',fm)
 local=a.old_raw/'model_comparison_20261113';(O/'prior_local_evidence').mkdir(exist_ok=True)
 for p in [local/'REPORT.md',local/'protocol.json',local/'summary.json',local/'group_weighted_summary.json',local/'selection.json',local/'report_group_scores.py',a.old_raw/'run_model_comparison.py',*[local/f'baseline_{s}.npz' for s in [20261020,20261021,20261022]]]:
  if p.exists():shutil.copyfile(p,O/'prior_local_evidence'/p.name);proof.append(dict(path=str(p),sha256=sha(p)))
 tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().split('\0');protected={p:sha(REPO/p) for p in tracked if p and (REPO/p).is_file() and not p.startswith(str(O.relative_to(REPO))+'/')};dump('protected_tracked_hashes.json',protected)
 provenance=dict(repository='HailiJia/CO2RR_XAS_agent',draft_pr=18,base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),previous_dataset='study/dataset_20261101: 141 configurations/135 near groups',previous_inputs_sha256=sha(B/'training_arrays.npz'),previous_manifest_sha256=sha(B/'training_manifest.json'),new_dataset=str(a.new_raw.resolve()),old_raw=str(a.old_raw.resolve()),new_dataset_label='local November 13 source export CO2RR_1113; not uploaded at base commit',prior_report='local November 13 report found outside checkout; prior scores are verified unchanged-cohort model comparisons, not added-data effects',prior_local_files=proof)
 dump('provenance.json',provenance)
 inv=[];newrows=[];newdelta=[];newphys=[];newclean=[];newgeos={};sources={};layeradd=[]
 def capture(p):
  if p.exists() and p.is_file():sources[str(p.relative_to(a.new_raw))]=dict(sha256=sha(p),bytes=p.stat().st_size)
 refs={}
 for dom,alias,ls in [('CuAu','plain',[4,5]),('Cu100','Cu100/Cu100_2x2x4_clean',[3,4])]:
  src=a.new_raw/alias/'XAS/FDMNES/green';s=deck_structure(src/f'layer_{ls[-1]}/Cu_layer{ls[-1]}_in.txt');assert not snapshot_graph(s)['counts'];ix=next(i for i,r in enumerate(old) if r['domain']==dom);clean=z['clean'][ix];assert all(np.array_equal(clean,z['clean'][j]) for j,r in enumerate(old) if r['domain']==dom);refs[dom]=dict(geometry=s,settings=settings(src/f'layer_{ls[-1]}/Cu_layer{ls[-1]}_in.txt'),scale=old[ix]['clean_scale'],clean=clean,reference_id=old[ix]['reference_id'],reference_record_sha256=old[ix]['reference_record_sha256']);assert abs(clean_scale(clean,E)-1)<1e-8
  # Verify pinned complete reference sources; an incomplete redundant Cu100 source is not used.
  receipts=json.load(open(B/'scale/source_weight_audit.json'));refproof=[q for q in receipts if q['configuration']==alias];checked=[]
  if dom=='Cu100':
   for q in refproof:
    for v in q.get('complete_sources',[]):p=a.new_raw/v['path'];assert sha(p)==v['sha256'];capture(p);checked.append(v)
  else:
   # Old selected manifest fixes the reference native spectra and decks.
   entries=json.load(open(a.old_raw/'refresh_20261103/source_manifest.json'))
   for v in entries:
    if v['path'].startswith('plain/XAS/FDMNES/green/') and (v['path'].endswith('_conv.txt') or v['path'].endswith('_in.txt') or v['path'].endswith('absorbers.tsv')):
     p=a.new_raw/v['path'];assert sha(p)==v['sha256'];capture(p);checked.append(v)
  assert checked;dump(f'reference_{dom}.json',dict(configuration=alias,reference_id=refs[dom]['reference_id'],pinned_clean_scale=refs[dom]['scale'],checked_sources=checked,clean_labels=[],note='Pinned audited clean physical spectrum, not current incomplete redundant native Cu100 aggregate'))
 folders=sorted(a.new_raw.glob('**/XAS/FDMNES/green'))
 for folder in folders:
  alias=str(folder.relative_to(a.new_raw)).split('/XAS/')[0];dom='Cu100' if alias.startswith('Cu100/') else 'CuAu';fs=native(folder);prior=native(a.old_raw/alias/'XAS/FDMNES/green');added=sorted(set(fs)-set(prior));layeradd.extend(dict(configuration=alias,layer=l,top_two=l in ([3,4] if dom=='Cu100' else [4,5]),already_old_query=alias in oldby) for l in added)
  row=dict(configuration=alias,domain=dom,old_query=alias in oldby,available_layers=sorted(fs),added_layers=added,eligible_new=False,classification='original_frozen_query' if alias in oldby else 'candidate');inv.append(row);ls=[3,4] if dom=='Cu100' else [4,5]
  try:
   decks={l:deck_structure(folder/f'layer_{l}/Cu_layer{l}_in.txt') for l in ls};s=decks[ls[-1]];g=snapshot_graph(s);row.update(observed_counts=g['counts'],identity_labels=g['identities'],graph_status=g['status'],connectivity=g['components'],graph_sensitivity=g['sensitivity_factors'],geometry_hash=geometry_hash(s));root=a.new_raw/alias/'CONTCAR';row['relaxed_geometry_match']=correspondence(read_poscar(root),s) if root.exists() and root.stat().st_size else dict(status='missing_root_CONTCAR')
   initial=a.new_raw/alias/'POSCAR'
   if initial.exists() and initial.stat().st_size:
    ig=snapshot_graph(read_poscar(initial));row['initial_counts']=ig['counts'];row['initial_to_XAS_connectivity_changed']=ig['counts']!=g['counts']
   supplied=folder/'isaac_record_layers.json'
   if supplied.exists():
    d=json.load(open(supplied));row['supplied_counts']=d.get('sample',{}).get('composition',{}).get('adsorbate_counts');row['supplied_labels_match']=row['supplied_counts']==g['counts']
   if alias in oldby:
    i=oldby[alias];row['frozen_labels_match_XAS']=set(g['identities'])==set(old[i]['identity_labels']);row['frozen_geometry_match_XAS']=correspondence(geos[alias],s)['status'];row['retention_basis']='original frozen query always retained; newest root geometry does not overwrite original query';continue
   if g['status']!='verified':raise ValueError('Unknown or cutoff-sensitive XAS connectivity')
   if set(g['identities'])-{'CHO','CO','CO2','COCHO','COH','COOH','H','O','OCCOH','OCHO','OH'}:raise ValueError('Unsupported identity outside frozen eleven-label task: '+','.join(g['identities']))
   if not set(ls)<=set(fs):raise ValueError('Missing physical top-two convolved spectra')
   if row['relaxed_geometry_match']['status']!='matched':raise ValueError('No exact relaxed root CONTCAR/XAS lineage for a new case')
   if correspondence(decks[ls[0]],s)['status']!='matched':raise ValueError('Top-two XAS geometries differ')
   ref=refs[dom];rs=ref['geometry'];assert collections.Counter(x for x in s['atoms'] if x in ['Cu','Au'])==collections.Counter(x for x in rs['atoms'] if x in ['Cu','Au']), 'Substrate composition mismatch'
   assert np.allclose(s['cell'],rs['cell'],atol=2e-3,rtol=0),'Clean-reference cell mismatch'
   row['reference_id']=ref['reference_id'];row['reference_match']='same cell, substrate composition, Green/SCF/Perdew/radius6, grid and pinned physical per-Cu basis; parent transfer not established'
   per={};pops={};evidence={}
   for l in ls:
    src=folder/f'layer_{l}';assert settings(src/f'Cu_layer{l}_in.txt')==ref['settings'],'Numerical settings mismatch';ev=layer_evidence(src,l,decks[l]);hs=headers(src,l);ids=set(pd.read_csv(src/'absorbers.tsv',sep='\t').atom_index);assert {h['atom_index'] for h in hs}==ids and len(hs)==len(ids),'Source/population mismatch';assert all(h['natomsym_f']==1 for h in hs),'Nonunit source symmetry needs separately verified reconvolution'
    for h in hs:
     p=src/h['file'];arr=np.loadtxt(p,skiprows=2);assert np.array_equal(arr[:,0],E) and np.isfinite(arr).all(),'Incomplete source grid';capture(p)
    assert np.array_equal(fs[l][1][:,0],E),'Native energy grid mismatch';pops[l]=ev['population'];per[l]=fs[l][1][:,1]/pops[l];evidence[l]=dict(population_evidence=ev,headers=hs,physical_scale='complete explicit sites; every source natomsym_f=1; native aggregate divided by verified physical N; no symmetry correction required',native_reconvolution='not rerun; complete saved final convolution log/source mapping verified')
    for p in [src/f'Cu_layer{l}_in.txt',src/'absorbers.tsv',src/f'Cu_layer{l}_conv.txt',src/f'Cu_layer{l}_convolution_in.txt',src/'CONTCAR',src/ev.get('convolution_log','fdmnes.out')]:capture(p)
   raw=sum(per[l]*pops[l] for l in ls)/sum(pops.values());physical=raw/ref['scale'];delta=physical-ref['clean'];row.update(eligible_new=True,classification='usable_new_configuration',layer_evidence=evidence,top_two_populations=pops,clean_scale=ref['scale']);capture(root);capture(initial);newrows.append(dict(configuration=alias,domain=dom,identity_labels=g['identities'],observed_counts=g['counts'],connectivity=g['components'],group=row['geometry_hash'],reference_id=ref['reference_id'],reference_record_sha256=ref['reference_record_sha256'],clean_scale=ref['scale'],old99=False,old_index=None));newdelta.append(delta);newphys.append(physical);newclean.append(ref['clean']);newgeos[alias]=s
  except (ValueError,KeyError,IndexError,AssertionError,FileNotFoundError,TypeError) as exc:
   row['exclusion']=str(exc);row['classification']='original_frozen_query' if alias in oldby else 'excluded';row['old_query_preserved']=alias in oldby
 # Exact same conservative near criterion as the frozen dataset, inheriting all old clusters.
 allrows=old+newrows;allgeos={**geos,**newgeos};parents=list(range(len(allrows)))
 def find(i):
  while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
  return i
 def join(i,j):parents[find(i)]=find(j)
 for i in range(len(old)):
  for j in range(i):
   if fm[old[i]['configuration']]==fm[old[j]['configuration']]:join(i,j)
 pairs=[]
 for i in range(len(old),len(allrows)):
  aa=allgeos[allrows[i]['configuration']];sa=np.array(aa['atoms']);pa=np.array(aa['positions']);cell=np.array(aa['cell']);invcell=np.linalg.inv(cell)
  for j in range(i):
   bb=allgeos[allrows[j]['configuration']];sb=np.array(bb['atoms']);pb=np.array(bb['positions'])
   if allrows[i]['domain']!=allrows[j]['domain'] or collections.Counter(sa)!=collections.Counter(sb) or allrows[i]['observed_counts']!=allrows[j]['observed_counts'] or not np.allclose(cell,bb['cell'],atol=2e-3,rtol=0):continue
   dist=[]
   for el in set(sa):
    df=(pa[sa==el,None]-pb[sb==el][None])@invcell;df[:,:,:2]-=np.rint(df[:,:,:2]);ds=np.full(df.shape[:2],np.inf)
    for u,v in itertools.product([-1,0,1],repeat=2):
     dd=df.copy();dd[:,:,0]+=u;dd[:,:,1]+=v;ds=np.minimum(ds,np.linalg.norm(dd@cell,axis=2))
    m,n=linear_sum_assignment(ds**2);dist.extend(ds[m,n])
   rms=float(np.sqrt(np.mean(np.square(dist))));mx=float(max(dist));near=rms<=.02 and mx<=.2;pairs.append(dict(a=allrows[i]['configuration'],b=allrows[j]['configuration'],RMS_A=rms,max_A=mx,near=near))
   if near:join(i,j)
 for i,r in enumerate(allrows):
  cluster=[j for j in range(len(allrows)) if find(i)==find(j)];oldmembers=[j for j in cluster if j<len(old)];oldf={old[j]['fold'] for j in oldmembers};pass # Original query folds span five inherited near groups; preserve queries and purge related groups in every fit
  r['near_group']=min(fm[old[j]['configuration']] for j in oldmembers) if oldmembers else min(allrows[j]['configuration'] for j in cluster);r['fold']=(r['fold'] if i<len(old) else min(oldf)) if oldmembers else int(hashlib.sha256(r['near_group'].encode()).hexdigest()[:8],16)%5;r['is_original_query']=i<len(old)
 dump('inherited_query_fold_conflicts.json',[dict(near_group=g,records=[dict(configuration=r['configuration'],fold=r['fold']) for r in allrows[:len(old)] if r['near_group']==g]) for g in sorted(set(r['near_group'] for r in allrows[:len(old)])) if len({r['fold'] for r in allrows[:len(old)] if r['near_group']==g})>1]);dump('dataset_manifest.json',allrows);dump('geometry_manifest.json',[dict(configuration=r['configuration'],**serial(allgeos[r['configuration']])) for r in allrows]);dump('audit_inventory.json',inv);dump('near_geometry_pairs.json',pairs);dump('source_manifest.json',sources);pd.DataFrame(layeradd).to_csv(O/'additional_layers.csv',index=False)
 np.savez_compressed(O/'inputs.npz',energy=E,delta=np.vstack([z['delta'],np.array(newdelta).reshape(-1,len(E))]),raw_physical=np.vstack([z['raw_physical'],np.array(newphys).reshape(-1,len(E))]),clean=np.vstack([z['clean'],np.array(newclean).reshape(-1,len(E))]))
 counts=[]
 for r in newrows:
  for typ,values in [('substrate',[r['domain']]),('species',r['identity_labels']),('exact_mixture',['+'.join(r['identity_labels']) or 'clean'])]:
   for v in values:counts.append(dict(configuration=r['configuration'],near_group=r['near_group'],category=typ,value=v,new_independent_near_group=r['near_group'] not in set(fm.values())))
 cf=pd.DataFrame(counts);cf.to_csv(O/'new_case_counts_long.csv',index=False)
 summary=dict(previous_records=len(old),previous_groups=len(set(fm.values())),usable_new_records=len(newrows),usable_new_groups=len(set(r['near_group'] for r in newrows)-set(fm.values())),new_views_existing_groups=sum(r['near_group'] in set(fm.values()) for r in newrows),total_records=len(allrows),total_groups=len(set(r['near_group'] for r in allrows)),additional_layer_spectra=len(layeradd),new_top_two_layer_spectra=sum(v['top_two'] for v in layeradd),excluded_nonold=sum(not r['old_query'] and not r['eligible_new'] for r in inv),counts=cf.groupby(['category','value']).agg(configurations=('configuration','nunique'),groups=('near_group','nunique')).reset_index().to_dict('records') if len(cf) else [],new_cases=[dict(configuration=r['configuration'],labels=r['identity_labels'],counts=r['observed_counts'],near_group=r['near_group'],fold=r['fold']) for r in newrows]);dump('audit_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
