"""Losslessly package bulky derived outputs, with verified per-member hashes."""
from pathlib import Path
import json,hashlib,tarfile,argparse
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/differential_v1';A=B/'artifacts'
def digest(b):return hashlib.sha256(b).hexdigest()
def pack():
 A.mkdir(exist_ok=True);groups={'reconvolution':list((B/'scale/reconvolution').rglob('*')),'models_and_inner':list((B/'ml/models').rglob('*'))+list((B/'ml').glob('*_inner.npz'))+list((B/'geometry/models').rglob('*'))+list((B/'raw_augmentation').glob('*.joblib'))+list((B/'raw_identity_augmentation').glob('*.joblib')),'stress_models_predictions':list((B/'robustness/models').rglob('*'))+list((B/'robustness').glob('*.npz'))};manifest=[]
 for name,paths in groups.items():
  paths=sorted(p for p in paths if p.is_file());dest=A/(name+'.tar.gz');members=[]
  with tarfile.open(dest,'w:gz',compresslevel=9) as tf:
   for p in paths:
    key=str(p.relative_to(B));data=p.read_bytes();members.append({'path':key,'bytes':len(data),'sha256':digest(data)});tf.add(p,arcname=key,recursive=False)
  with tarfile.open(dest) as tf:
   assert len(tf.getmembers())==len(members)
   for r in members:assert digest(tf.extractfile(r['path']).read())==r['sha256']
  manifest.append({'archive':dest.name,'bytes':dest.stat().st_size,'sha256':digest(dest.read_bytes()),'members':members})
 (A/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');print([(r['archive'],r['bytes'],len(r['members'])) for r in manifest])
def unpack():
 for r in json.load(open(A/'manifest.json')):
  p=A/r['archive'];assert digest(p.read_bytes())==r['sha256']
  with tarfile.open(p) as tf:
   for m in r['members']:
    dest=(B/m['path']).resolve();assert B.resolve() in dest.parents
    data=tf.extractfile(m['path']).read();assert len(data)==m['bytes'] and digest(data)==m['sha256'];dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists():assert dest.read_bytes()==data,'Refuse to overwrite changed output'
    else:dest.write_bytes(data)
 print('Verified and unpacked derived artifacts')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--unpack',action='store_true');a=p.parse_args();unpack() if a.unpack else pack()
