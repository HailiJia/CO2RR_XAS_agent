"""Hash all selected 1029 scientific sources; upload changed bytes, reuse pinned source archives."""
from pathlib import Path
import sys,hashlib,json,tarfile,gzip,io,datetime,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import folders
from tools.progress_snapshots import write_json
RAW=Path(sys.argv[1]).resolve();OUT=ROOT/'study/dataset_20261029/sources';OUT.mkdir(exist_ok=True)
old=json.load(open(ROOT/'study/differential_v1/sources/path_hash_size_manifest.json'));byhash={r['sha256']:r for r in old};selected={};excluded={}
def select(p):
 if p.is_file():selected[str(p.relative_to(RAW))]=p
for f in folders(RAW):
 for sub in f.glob('layer_*'):
  if not sub.is_dir():continue
  for p in sub.iterdir():
   n=p.name
   if p.is_file() and (re.fullmatch(r'Cu_layer\d+_\d+\.txt',n) or n.endswith(('_in.txt','_conv.txt')) or n in ['CONTCAR','POSCAR','absorbers.tsv','fdmfile.txt','exit_code.txt'] or n.startswith('fdmnes') and n.endswith('.out') and ('convolution' in n or p.stat().st_size<200000) or n.startswith('slurm') and p.stat().st_size<200000):select(p)
 alias=f.parents[2]
 for n in ['POSCAR','CONTCAR','INCAR','KPOINTS','OSZICAR']:select(alias/n)
 for n in ['isaac_record_layers.json','layer_summary.tsv']:select(f/n)
# Current benchmarks and unfinished relaxation sources stay scientific data, not training examples.
for name in ['benchmark','OCCO_H2O']:
 for p in (RAW/name).rglob('*'):
  if p.is_file() and p.stat().st_size<50000000 and not any(q.startswith('.') for q in p.relative_to(RAW).parts) and (p.name in ['POSCAR','CONTCAR','INCAR','KPOINTS','OUTCAR','OSZICAR','XDATCAR','vasprun.xml','vaspout.h5','Cu_XAS.ipynb','xmu.dat','feff.inp'] or p.suffix.lower() in ['.csv','.ipynb'] or p.name.endswith(('_conv.txt','_in.txt'))):select(p)
manifest=[];new=[];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
for label,p in sorted(selected.items()):
 st=p.stat();data=p.read_bytes();en=p.stat();assert (st.st_size,st.st_mtime_ns)==(en.st_size,en.st_mtime_ns),'Changing source '+label
 h=hashlib.sha256(data).hexdigest();r={'path':label,'bytes':len(data),'sha256':h,'mtime_UTC':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'captured_UTC':now}
 if h in byhash:r['storage']={'previous_archive':'study/differential_v1/sources/'+byhash[h]['archive'],'member':byhash[h]['logical_path'],'pinned_commit':'db1efb575773f341e3fa726aa0d312848fb3019c'}
 else:new.append((label,p,r))
 manifest.append(r)
parts=[]
for start in range(0,len(new),250):
 group=new[start:start+250];name=f'new_sources_{start//250:03d}.tar.gz';dest=OUT/name
 with dest.open('wb') as stream,gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tf:
  for label,p,r in group:
   data=p.read_bytes();assert hashlib.sha256(data).hexdigest()==r['sha256'];info=tarfile.TarInfo(label);info.size=len(data);info.mtime=0;tf.addfile(info,io.BytesIO(data));r['storage']={'archive':name,'member':label}
 with tarfile.open(dest) as tf:
  for label,p,r in group:assert hashlib.sha256(tf.extractfile(label).read()).hexdigest()==r['sha256']
 parts.append({'archive':name,'bytes':dest.stat().st_size,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'members':len(group)});print(name,dest.stat().st_size,flush=True)
write_json(OUT/'manifest.json',manifest);write_json(OUT/'archives.json',parts)
write_json(OUT/'inventory_summary.json',{'root':'CO2RR_1029','source_bytes_selected':sum(r['bytes'] for r in manifest),'selected_files':len(manifest),'new_files':len(new),'reused_exact_source_bytes':len(manifest)-len(new),'new_compressed_bytes':sum(r['bytes'] for r in parts),'exclusions':'wavefunctions, charge densities, licensed POTCAR contents, executable binaries, hidden files, verbose BAV/raw-run/relaxation logs and trajectories redundant to retained exact CONTCAR/inputs/raw per-site electronic spectra/final convolution logs; benchmark files above50MB excluded from this ML refresh package (prior benchmark sources remain archived); no new calculation or portal write','source_packages_are_not_all_completed_training_examples':True})
