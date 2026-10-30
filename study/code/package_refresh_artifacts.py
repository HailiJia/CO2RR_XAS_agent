"""Verified lossless model/reconstruction packaging; source records remain readable."""
from pathlib import Path
import sys,json,hashlib,tarfile
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/dataset_20261029';A=B/'artifacts'
def sha(data):return hashlib.sha256(data).hexdigest()
def pack():
 A.mkdir(exist_ok=True);manifest=[]
 for name,paths in [('models',list((B/'ml/models').glob('*.joblib'))),('reconstruction',list((B/'scale/reconvolution').rglob('*'))+[B/'source_evidence.tar'])]:
  paths=sorted(p for p in paths if p.is_file());dest=A/(name+'.tar.gz');members=[]
  with tarfile.open(dest,'w:gz',compresslevel=9) as tf:
   for p in paths:
    data=p.read_bytes();r={'path':str(p.relative_to(B)),'bytes':len(data),'sha256':sha(data)};members.append(r);tf.add(p,arcname=r['path'],recursive=False)
  with tarfile.open(dest) as tf:
   for r in members:assert sha(tf.extractfile(r['path']).read())==r['sha256']
  manifest.append({'archive':dest.name,'bytes':dest.stat().st_size,'sha256':sha(dest.read_bytes()),'members':members})
 (A/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print([(r['archive'],r['bytes']) for r in manifest])
def unpack():
 for r in json.load(open(A/'manifest.json')):
  p=A/r['archive'];assert sha(p.read_bytes())==r['sha256']
  with tarfile.open(p) as tf:
   for m in r['members']:
    dest=(B/m['path']).resolve();assert B.resolve() in dest.parents;data=tf.extractfile(m['path']).read();assert len(data)==m['bytes'] and sha(data)==m['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_bytes()==data,'Changed derived output must not be overwritten'
    else:dest.write_bytes(data)
 print('Verified derived artifacts unpacked')
if __name__=='__main__':
 unpack() if '--unpack'in sys.argv else pack()
