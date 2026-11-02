"""Upload-ready lossless scientific sources, never potentials/wavefunctions/binaries."""
from pathlib import Path
import sys,json,hashlib,tarfile,gzip,io,re,datetime
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.progress_snapshots import write_json
from tools.coverage_study import folders
B=ROOT/'study/differential_v1';O=B/'sources';O.mkdir(exist_ok=True);raw=Path('/Users/hjia/Downloads/CO2RR');downloads=raw.parent;manifest=[];parts=[]
def pack(name,entries):
 files={str(p): (p,label) for p,label in entries if p.is_file()};dest=O/(name+'.tar.gz');rows=[]
 with dest.open('wb') as output,gzip.GzipFile(filename='',fileobj=output,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as t:
  for p,label in sorted(files.values(),key=lambda a:a[1]):
   for attempt in range(3):
    st=p.stat();data=p.read_bytes();end=p.stat()
    if (st.st_size,st.st_mtime_ns)==(end.st_size,end.st_mtime_ns) and len(data)==st.st_size:break
   else:raise ValueError('Changing source '+str(p))
   sha=hashlib.sha256(data).hexdigest();info=tarfile.TarInfo(label);info.size=len(data);info.mtime=0;t.addfile(info,io.BytesIO(data));rows.append({'logical_path':label,'source_path':str(p),'sha256':sha,'bytes':len(data),'source_mtime_UTC':datetime.datetime.fromtimestamp(st.st_mtime,datetime.timezone.utc).isoformat(),'capture_UTC':datetime.datetime.now(datetime.timezone.utc).isoformat(),'archive':dest.name})
 with tarfile.open(dest) as t:
  members={m.name:hashlib.sha256(t.extractfile(m).read()).hexdigest() for m in t if m.isfile()}
 assert members=={r['logical_path']:r['sha256'] for r in rows}
 manifest.extend(rows);parts.append({'path':dest.name,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'bytes':dest.stat().st_size,'members':len(rows),'contents_hash_verified':True});print(dest.name,dest.stat().st_size,len(rows),flush=True)
# Already stable read-only captures, including unfinished trajectories/full outputs.
c=B/'checkpoints/sources';pack('checkpoints',[(p,'downloads/'+str(p.relative_to(c))) for p in c.rglob('*') if p.is_file()])
allfolders=list(folders(raw));nonunit=[];groups={'plain_clean':[],'Cu100':[],'CuAu':[]}
for f in allfolders:
 alias=str(f.relative_to(raw)).split('/XAS/')[0];kind='Cu100' if alias.startswith('Cu100/') else 'plain_clean' if alias.startswith('plain') else 'CuAu';entries=[];multiple=False
 for p in f.glob('layer_*/Cu_layer*_[0-9]*.txt'):
  lid=int(p.parent.name.split('_')[-1])
  if lid not in ([3,4] if kind=='Cu100' else [4,5]):continue
  if not re.fullmatch(r'Cu_layer\d+_\d+\.txt',p.name):continue
  try:m=float(p.read_text().splitlines()[0].split('=')[0].split()[13]);multiple|=abs(m-1)>1e-8
  except (ValueError,IndexError):pass
 for sub in f.glob('layer_*'):
  if not sub.is_dir():continue
  for p in sub.iterdir():
   if not p.is_file():continue
   n=p.name
   lid=int(sub.name.split('_')[-1]);physical_top=lid in ([3,4] if kind=='Cu100' else [4,5])
   include=bool(re.fullmatch(r'Cu_layer\d+_\d+\.txt',n) or n.endswith(('_conv.txt','_in.txt')) or n in {'CONTCAR','POSCAR','absorbers.tsv','fdmfile.txt','exit_code.txt'} or n.startswith('fdmnes_convolution') and n.endswith('.out'))
   if n.endswith('_bav.txt') and physical_top and (multiple or kind=='plain_clean'):include=True
   if include:entries.append((p,'workspace/'+str(p.relative_to(raw))))
 for p in [raw/alias/n for n in ['POSCAR','CONTCAR','INCAR','KPOINTS']]+[f/'isaac_record_layers.json',f/'layer_summary.tsv']:
  if p.is_file():entries.append((p,'workspace/'+str(p.relative_to(raw))))
 groups[kind].append(entries)
 if multiple:nonunit.append(alias)
for kind,group in groups.items():
 for start in range(0,len(group),15):pack(kind+'_'+str(start//15).zfill(2),[v for entries in group[start:start+15] for v in entries])
# Original Cu100 snapshots preserve original ordering/forces; original water/K seeds
# are also present in the checkpoint archive, separately from live checkpoints.
pack('Cu100_original_structures',[(p,'workspace/'+str(p.relative_to(raw))) for p in (raw/'Cu-Ox_updated_structures/100').rglob('*.json')])
# Real benchmark SCF/XAS evidence. POTCAR metadata are read only to extract
# provenance elsewhere; potential contents and large wavefunctions are excluded.
allow={'INCAR','KPOINTS','POSCAR','CONTCAR','OUTCAR','OSZICAR','vasprun.xml','vaspout.h5','CORE_DIELECTRIC_IMAG.dat','xmu.dat','feff.inp','Cu_XAS.ipynb'};entries=[]
for base in [raw/'benchmark',downloads/'VASP_inputs_Cu_PBE',ROOT/'study/reference_benchmark/raw']:
 if not base.exists():continue
 for p in base.rglob('*'):
  if not p.is_file() or any(v.startswith('.') for v in p.relative_to(base).parts):continue
  if p.name in allow or p.name.endswith(('_conv.txt','_in.txt')) or base==ROOT/'study/reference_benchmark/raw' and p.suffix.lower() in {'.dat','.txt','.csv','.ipynb'}:
   prefix='reference_v1_raw' if base==ROOT/'study/reference_benchmark/raw' else 'workspace/benchmark' if base==raw/'benchmark' else 'downloads/VASP_inputs_Cu_PBE';entries.append((p,prefix+'/'+str(p.relative_to(base))))
pack('reference_calculations',entries)
write_json(O/'path_hash_size_manifest.json',manifest);write_json(O/'archives.json',parts);write_json(O/'nonunit_source_cases.json',nonunit)
(O/'README.md').write_text('# Supplied scientific sources\n\nThese independent tar.gz parts preserve logical folders and exact source bytes. Every member and archive hash was verified after compression. No licensed potentials, executable binaries or wavefunction checkpoints are included. `checkpoints.tar.gz` contains time-stamped UNFINISHED OCCO–water snapshots and available trajectories, not converged intermediates or training spectra. The benchmark part includes actual current SCF/XAS OUTCAR/XML/HDF5 files even when the run is incomplete. CuAu/Cu100 parts contain actual per-absorber spectra, canonical convolved curves, absorber tables, input decks and final convolution logs; nonunit-multiplicity cases also preserve BAV evidence. Original record bytes are sources, not silently replaced corrected records.\n')
