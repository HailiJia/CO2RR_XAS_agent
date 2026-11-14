"""Read-only, time-stamped unfinished relaxation audit; never training labels."""
from pathlib import Path
import datetime,hashlib,json,re,os
from collections import Counter
import numpy as np
from .component_audit import read_poscar
from .contact_checks import minimum_image_vectors
from .chemical_lineage import correspondence


def write_json(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,default=lambda a:a.tolist() if isinstance(a,np.ndarray) else str(a))+'\n')
def capture(p,dst):
 for _ in range(3):
  a=p.stat();b=p.read_bytes();z=p.stat()
  if (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns) and len(b)==a.st_size:break
 else:raise ValueError('File changing during snapshot: '+str(p))
 dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(b)
 return {'source':str(p),'snapshot':str(dst),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'source_mtime_utc':datetime.datetime.fromtimestamp(a.st_mtime,datetime.timezone.utc).isoformat(),'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'stable_read':True}
def flags(path,n):
 lines=path.read_text().splitlines();i=7
 if lines[i].lower().startswith('s'):return [lines[i+2+j].split()[3:6] for j in range(n)]
 return [['T']*3 for _ in range(n)]
def outcar_frames(p,s):
 if not p.exists():return [],[],[]
 lines=p.read_text(errors='replace').splitlines();frames=[];forces=[];steps=[];n=len(s['atoms']);cell=s['cell'].copy();step=0
 for i,line in enumerate(lines):
  if 'direct lattice vectors' in line:
   try:cell=np.array([[float(x) for x in lines[i+j+1].split()[:3]] for j in range(3)])
   except (ValueError,IndexError):pass
  if 'POSITION' not in line or 'TOTAL-FORCE' not in line:continue
  try:a=np.array([[float(x) for x in lines[i+2+j].split()] for j in range(n)])
  except (ValueError,IndexError):continue
  if a.shape!=(n,6) or not np.isfinite(a).all():continue
  step+=1;frames.append({'atoms':s['atoms'],'positions':a[:,:3],'cell':cell.copy()});forces.append(a[:,3:]);steps.append(step)
 return frames,forces,steps

def distances(s,left,right):
 return np.linalg.norm(minimum_image_vectors(s['positions'][np.asarray(left)[:,None]]-s['positions'][np.asarray(right)[None,:]],s['cell'],(True,True,False)),axis=-1)
def inspect(s,cids,oid=None):
 a=np.array(s['atoms']);o=np.flatnonzero(a=='O');h=np.flatnonzero(a=='H');metal=np.flatnonzero(np.isin(a,['Cu','Au']));cc=float(distances(s,[cids[0]],[cids[1]])[0,0]);co=distances(s,cids,o);dimer_o=np.array(oid,int) if oid is not None else o[np.argsort(co.min(axis=0))[:2]];oh=distances(s,o,h);ch=distances(s,cids,h);contacts=[]
 for i in list(cids)+list(dimer_o):
  ds=distances(s,[i],metal)[0];contacts.append({'atom_index':int(i+1),'element':a[i],'contacts':[{'atom_index':int(metal[j]+1),'element':a[metal[j]],'distance_A':float(d)} for j,d in enumerate(ds) if d<2.6]})
 proton=[{'H_index':int(h[j]+1),'bonded_O_index':int(o[int(np.argmin(oh[:,j]))]+1),'O_H_A':float(oh[:,j].min()),'min_C_H_A':float(ch[:,j].min())} for j in range(len(h))]
 dimerproton=[p for p in proton if (p['bonded_O_index']-1 in dimer_o and p['O_H_A']<1.2) or p['min_C_H_A']<1.3]
 water=[{'O_index':int(i+1),'H_indices':[int(h[j]+1) for j,d in enumerate(oh[k]) if d<1.2],'OH_distances_A':oh[k].tolist()} for k,i in enumerate(o) if i not in dimer_o]
 hb=[]
 for j,hi in enumerate(h):
  order=np.argsort(oh[:,j]);donor=o[order[0]]
  if oh[order[0],j]>1.2:continue
  for k in order[1:]:
   if oh[k,j]>2.5:continue
   v=minimum_image_vectors(s['positions'][donor]-s['positions'][hi],s['cell'],(True,True,False));w=minimum_image_vectors(s['positions'][o[k]]-s['positions'][hi],s['cell'],(True,True,False));angle=float(np.rad2deg(np.arccos(np.clip(np.dot(v,w)/np.linalg.norm(v)/np.linalg.norm(w),-1,1))));hb.append({'donor_O':int(donor+1),'H':int(hi+1),'acceptor_O':int(o[k]+1),'H_acceptor_A':float(oh[k,j]),'angle_deg':angle,'hydrogen_bond_geometric_guide':bool(angle>=120 and oh[k,j]>=1.2)})
 ccontacts=[x for x in contacts if x['element']=='C'];ocontacts=[x for x in contacts if x['element']=='O'];nc=sum(any(t['distance_A']<2.4 for t in x['contacts']) for x in ccontacts);no=sum(any(t['distance_A']<2.4 for t in x['contacts']) for x in ocontacts)
 co_core=distances(s,cids,dimer_o);carbonyl=bool(np.all(co_core.min(axis=1)<1.7));verdict='inconclusive'
 if cc<1.85 and carbonyl:verdict='protonated/other coupled intermediate' if dimerproton else 'retained two-C-bound OCCO' if nc==2 and no==0 else 'retained but rearranged C,O-bound C2O2' if no>0 else 'inconclusive'
 elif cc>2.2 and carbonyl and not dimerproton:verdict='dissociated 2CO'
 elif cc>2.2:verdict='inconclusive'
 return {'C_indices_1based':(np.asarray(cids)+1).tolist(),'dimer_O_indices_1based':(dimer_o+1).tolist(),'CC_A':cc,'CO_core_distances_A':co_core.tolist(),'all_CO_distances_A':co.tolist(),'O_H_bond_assignments':proton,'protonated_dimer_candidates':dimerproton,'water_connectivity':water,'hydrogen_bond_candidates':hb,'metal_contacts':contacts,'binding_motif':{'C_atoms_metal_bound':nc,'O_atoms_metal_bound':no,'contact_cutoff_A':2.4},'CC_cutoff_sensitivity':{str(c):cc<c for c in [1.65,1.75,1.85,2.0,2.2]},'verdict':verdict,'borderline':bool(1.65<cc<2.2),'hydrogen_bond_not_proton_transfer':'H bonds to dimer O <1.2 A or C <1.3 A are separately flagged; longer directed O-H...O contacts are not covalent proton transfer'}

def run(downloads,out):
 downloads=Path(downloads);out=Path(out);out.mkdir(parents=True,exist_ok=True);seeds=[];runs=[];search=[]
 for folder,dirs,names in os.walk(downloads):
  dirs[:]=[d for d in dirs if not d.startswith(('.','PR18')) and d not in {'node_modules','__pycache__','CO2RR_ML','co2rr_digital_discovery','benchmark','openniw_haili'}]
  for name in names:
   if not (name in {'POSCAR','CONTCAR'} or name.startswith('POSCAR_')):continue
   p=Path(folder)/name
   try:
    if p.stat().st_size==0:continue
    lines=p.read_text().splitlines();counts=dict(zip(lines[5].split(),map(int,lines[6].split())))
   except (ValueError,IndexError,UnicodeError):continue
   if counts.get('C')!=2 or counts.get('H',0)<2 or counts.get('O',0)<3 or abs(counts.get('H',0)-2*(counts.get('O',0)-2))>1:continue
   search.append(str(p))
   if name=='POSCAR' and (p.parent/'OUTCAR').exists():runs.append(p.parent)
   elif name!='CONTCAR':seeds.append(p)
 rows=[];capture_manifest=[]
 for p in sorted(set(seeds)):
  try:s=read_poscar(p);c=np.flatnonzero(np.array(s['atoms'])=='C');x=inspect(s,c)
  except Exception as e:rows.append({'case':str(p.relative_to(downloads)),'kind':'seed_parse_error','error':str(e)});continue
  target=out/'sources'/p.relative_to(downloads);capture_manifest.append(capture(p,target));f=flags(p,len(s['atoms']));rows.append({'case':str(p.relative_to(downloads)),'kind':'starting_seed_only','composition':dict(Counter(s['atoms'])),'initial':x,'current':None,'run_status':'no associated relaxation output located at this seed path','fixed_C_flags':[f[i] for i in c]})
 for folder in sorted(set(runs)):
  case=str(folder.relative_to(downloads));dest=out/'sources'/case
  for name in ['POSCAR','CONTCAR','OUTCAR','OSZICAR','XDATCAR','vasprun.xml','vaspout.h5','INCAR','KPOINTS']:
   if (folder/name).exists():capture_manifest.append(capture(folder/name,dest/name))
  for p in folder.glob('slurm-*.out'):capture_manifest.append(capture(p,dest/p.name))
  initial=read_poscar(dest/'POSCAR');c=np.flatnonzero(np.array(initial['atoms'])=='C');oi=np.flatnonzero(np.array(initial['atoms'])=='O');oid=oi[np.argsort(distances(initial,c,oi).min(axis=0))[:2]];frames,forces,steps=outcar_frames(dest/'OUTCAR',initial);current=None;basis=None
  try:
   current=read_poscar(dest/'CONTCAR');basis='latest complete CONTCAR'
  except (ValueError,IndexError,FileNotFoundError):pass
  if frames:
   link=correspondence(current,frames[-1]) if current is not None else None
   if current is None or link['status']!='matched' and (folder/'OUTCAR').stat().st_mtime>(folder/'CONTCAR').stat().st_mtime:current=frames[-1];basis='latest complete OUTCAR POSITION block'
   step=len(frames) if correspondence(current,frames[-1])['status']=='matched' else None
  else:step=None
  f=flags(dest/'POSCAR',len(initial['atoms']));free=np.array([i for i,v in enumerate(f) if any(x.upper()=='T' for x in v)]);text=(dest/'OUTCAR').read_text(errors='replace');osz=(dest/'OSZICAR').read_text() if (dest/'OSZICAR').exists() else '';ions=re.findall(r'^\s*(\d+)\s+F=',osz,re.M);electronic=re.findall(r'(?:DAV|RMM):\s*(\d+)\s+.*',osz);relaxed='reached required accuracy' in text
  trajectory=[]
  for k,frame in enumerate(frames):trajectory.append({'ionic_step_in_captured_OUTCAR':k+1,'CC_A':inspect(frame,c,oid)['CC_A'],'proton_distances':inspect(frame,c,oid)['O_H_bond_assignments'],'max_free_force_eV_A':float(np.linalg.norm(forces[k][free],axis=1).max())})
  np.savez_compressed(out/(case.replace('/','__')+'_trajectory.npz'),positions=np.array([x['positions'] for x in frames]),cell=np.array([x['cell'] for x in frames]),forces=np.array(forces),atom_symbols=np.array(initial['atoms']))
  templates=[]
  for p in seeds:
   try:
    a=read_poscar(p)
    if Counter(a['atoms'])!=Counter(initial['atoms']):continue
    m=correspondence(a,initial);templates.append({'seed':str(p.relative_to(downloads)),'rms_displacement_A':m.get('rms_distance_A'),'mapping':m.get('one_based_mapping')})
   except (ValueError,IndexError):continue
  templates=sorted([t for t in templates if t['rms_displacement_A'] is not None],key=lambda t:t['rms_displacement_A'])[:3]
  row={'case':case,'kind':'captured_relaxation_checkpoint','capture_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'snapshot_basis':basis,'ionic_step_in_captured_OUTCAR':step,'OSZICAR_last_completed_step':int(ions[-1]) if ions else None,'last_electronic_iteration':int(electronic[-1]) if electronic else None,'composition':dict(Counter(initial['atoms'])),'initial':inspect(initial,c,oid),'current':inspect(current,c,oid) if current else None,'nearest_original_seed_candidates':templates,'initial_caveat':'POSCAR may be a restart geometry. Original seed candidates are reported by measured displacement, not assigned from names.','fixed_C_flags':[f[i] for i in c],'max_free_force_eV_A':float(np.linalg.norm(forces[-1][free],axis=1).max()) if forces else None,'force_basis':'last complete OUTCAR force evaluation; CONTCAR may be the subsequent optimizer update','run_status':'ionic convergence message present' if relaxed else 'UNFINISHED checkpoint; no ionic convergence message','termination_footer_present':'General timing and accounting' in text,'trajectory':trajectory,'current_structure':current,'electronic_caveat':'last SCF iteration count and source OSZICAR retained; a complete step is not proof of ionic convergence'}
  rows.append(row)
 write_json(out/'case_audit.json',rows);write_json(out/'capture_manifest.json',capture_manifest);write_json(out/'search_inventory.json',{'roots':[str(downloads)],'candidate_geometry_files':search,'note':'composition search includes intermediate and seed geometries, not completed-record filtering; seeds without output are not completed relaxations'})
 return rows
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--downloads',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();r=run(a.downloads,a.output_dir);print(Counter(x['kind'] for x in r));print([(x['case'],x.get('current',{}).get('verdict')) for x in r if x['kind']=='captured_relaxation_checkpoint'])
