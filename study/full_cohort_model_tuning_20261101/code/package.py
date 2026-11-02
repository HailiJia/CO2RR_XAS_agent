"""Verified lossless model archive, split into practical parts."""
from pathlib import Path
import json,hashlib,tarfile,io,tempfile,sys
B=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
if '--unpack' in sys.argv:
 m=json.load(open(B/'model_archive.json'));data=bytearray()
 for row in m['parts']:
  b=(B/row['path']).read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256'];data.extend(b)
 assert sha(data)==m['sha256']
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for x in tar.getmembers():
   assert x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts
   b=tar.extractfile(x).read();assert sha(b)==m['files'][x.name]['sha256'];p=B/x.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 print('Restored',len(m['files']),'verified evaluation/final models')
else:
 files=sorted((B/'results/models').glob('*.joblib'))+sorted((B/'final_models').glob('*.joblib'));assert len(files)==39
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'models.tar.gz'
  with tarfile.open(p,'w:gz') as tar:
   for f in files:tar.add(f,arcname=str(f.relative_to(B)))
  data=p.read_bytes();parts=[]
  for i,start in enumerate(range(0,len(data),12*1024**2)):
   f=B/f'models.tar.gz.part{i:03d}';b=data[start:start+12*1024**2];f.write_bytes(b);parts.append(dict(path=f.name,bytes=len(b),sha256=sha(b)))
  with tarfile.open(p,'r:gz') as tar:
   for f in files:assert tar.extractfile(str(f.relative_to(B))).read()==f.read_bytes()
 (B/'model_archive.json').write_text(json.dumps(dict(sha256=sha(data),parts=parts,files={str(f.relative_to(B)):dict(bytes=f.stat().st_size,sha256=sha(f.read_bytes())) for f in files}),indent=2)+'\n');print('Archived39 models',len(data),'bytes')
