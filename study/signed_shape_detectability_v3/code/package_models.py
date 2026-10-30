"""Lossless multipart model archive with byte/hash and safe-unpack verification."""
from pathlib import Path
import hashlib,json,tarfile,tempfile,sys,io
B=Path(__file__).resolve().parents[1]
def sha(b):return hashlib.sha256(b).hexdigest()
def unpack():
 m=json.load(open(B/'model_archive_manifest.json'));data=bytearray()
 for row in m['parts']:
  b=(B/row['path']).read_bytes();assert len(b)==row['bytes'] and sha(b)==row['sha256'];data.extend(b)
 assert sha(data)==m['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  for member in tar.getmembers():
   assert member.isfile() and member.name.startswith('ml/models/') and '..' not in Path(member.name).parts
   b=tar.extractfile(member).read();expected=m['files'][member.name];assert len(b)==expected['bytes'] and sha(b)==expected['sha256'];p=B/member.name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
 print('Restored',len(m['files']),'verified models')
if '--unpack' in sys.argv:unpack()
else:
 files=sorted((B/'ml/models').glob('*.joblib'));assert len(files)==150
 with tempfile.TemporaryDirectory() as td:
  p=Path(td)/'models.tar.gz'
  with tarfile.open(p,'w:gz') as tar:
   for f in files:tar.add(f,arcname=str(f.relative_to(B)))
  data=p.read_bytes();parts=[]
  for i,start in enumerate(range(0,len(data),12*1024**2)):
   part=B/f'models.tar.gz.part{i:03d}';b=data[start:start+12*1024**2];part.write_bytes(b);parts.append({'path':part.name,'bytes':len(b),'sha256':sha(b)})
  with tarfile.open(p,'r:gz') as tar:
   assert len(tar.getmembers())==len(files)
   for f in files:assert tar.extractfile(str(f.relative_to(B))).read()==f.read_bytes()
 m={'archive_sha256':sha(data),'bytes':len(data),'parts':parts,'files':{str(f.relative_to(B)):{'bytes':f.stat().st_size,'sha256':sha(f.read_bytes())} for f in files},'excluded':'no source records or historical archives changed'};(B/'model_archive_manifest.json').write_text(json.dumps(m,indent=2)+'\n');print('Packaged',len(files),'models',len(data),'bytes',len(parts),'parts')
