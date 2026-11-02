"""Capture actual selected source bytes, never potentials, executables or wavefunctions."""
from pathlib import Path
from tools.coverage_study import folders
import tarfile,json,hashlib,re,datetime
B=Path(__file__).resolve().parents[1];root=B/'local_export';rows=json.load(open(B/'training_manifest.json'));aliases={r['configuration'] for r in rows if r['old_index'] is None}|{'plain','Cu100/Cu100_2x2x4_clean'};files=set()
# Original author records preserve pre-correction labels in all supplied records.
for folder in folders(root):
 p=folder/'isaac_record_layers.json'
 if p.is_file():files.add(p)
for alias in aliases:
 p=root/alias
 for name in ['CONTCAR','POSCAR','INCAR','KPOINTS']:
  if (p/name).is_file():files.add(p/name)
 for layer in (p/'XAS/FDMNES/green').glob('layer_*'):
  if not layer.is_dir():continue
  for q in layer.iterdir():
   if q.is_file() and (q.name in ['CONTCAR','absorbers.tsv'] or re.fullmatch(r'Cu_layer\d+_(?:\d+|in|conv|convolution_in)\.txt',q.name)):files.add(q)
manifest=[]
with tarfile.open(B/'selected_sources.tar.gz','w:gz') as tar:
 for p in sorted(files):
  b=p.read_bytes();name=str(p.relative_to(root));h=hashlib.sha256(b).hexdigest();tar.add(p,arcname=name);assert hashlib.sha256(p.read_bytes()).hexdigest()==h;manifest.append({'path':name,'bytes':len(b),'sha256':h,'source_mtime_ns':p.stat().st_mtime_ns})
with tarfile.open(B/'selected_sources.tar.gz','r:gz') as tar:
 for x in manifest:assert hashlib.sha256(tar.extractfile(x['path']).read()).hexdigest()==x['sha256']
(B/'selected_sources_manifest.json').write_text(json.dumps({'capture_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source_root':'author CO2RR_11101; all paths relative','files':manifest,'archive_sha256':hashlib.sha256((B/'selected_sources.tar.gz').read_bytes()).hexdigest()},indent=2)+'\n');print('Selected sources',len(manifest),'archive bytes',(B/'selected_sources.tar.gz').stat().st_size)
