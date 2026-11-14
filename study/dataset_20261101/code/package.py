"""Verified lossless parts for expanded model/record/convolution artifacts."""
from pathlib import Path
import json,hashlib,tarfile,io,tempfile,sys
B=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
if '--unpack' in sys.argv:
 m=json.load(open(B/'artifact_archive.json'));data=bytearray()
 for row in m['parts']:
  b=(B/row['path']).read_bytes();assert sha(b)==row['sha256'];data.extend(b)
 assert sha(data)==m['sha256']
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for x in tar.getmembers():
   assert x.isfile() and '..' not in Path(x.name).parts and not Path(x.name).is_absolute();b=tar.extractfile(x).read();assert sha(b)==m['files'][x.name]['sha256'];p=B/x.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 print('Restored',len(m['files']),'verified files')
else:
 files=[]
 for folder in ['records','scale/corrected_records','scale/reconvolution','ml/models']:files.extend(p for p in (B/folder).rglob('*') if p.is_file())
 files.append(B/'source_evidence.tar');files=sorted(files)
 with tempfile.TemporaryDirectory() as td:
  path=Path(td)/'archive.tar.gz'
  with tarfile.open(path,'w:gz') as tar:
   for p in files:tar.add(p,arcname=str(p.relative_to(B)))
  data=path.read_bytes();parts=[]
  for i,start in enumerate(range(0,len(data),12*1024**2)):
   p=B/f'artifacts.tar.gz.part{i:03d}';b=data[start:start+12*1024**2];p.write_bytes(b);parts.append({'path':p.name,'bytes':len(b),'sha256':sha(b)})
  with tarfile.open(path,'r:gz') as tar:
   for p in files:assert tar.extractfile(str(p.relative_to(B))).read()==p.read_bytes()
 (B/'artifact_archive.json').write_text(json.dumps({'sha256':sha(data),'parts':parts,'files':{str(p.relative_to(B)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in files}},indent=2)+'\n');print('Packaged',len(files),'files',len(data),'bytes')
