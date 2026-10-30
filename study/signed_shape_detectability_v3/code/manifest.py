from pathlib import Path
import json,hashlib,sys
B=Path(__file__).resolve().parents[1];p=B/'artifact_sha256.json'
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
if '--verify' in sys.argv:
 m=json.load(open(p))
 for name,row in m.items():
  f=B/name;assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],name
 print('Verified',len(m),'versioned artifact hashes')
else:
 files=[f for f in sorted(B.rglob('*')) if f.is_file() and '__pycache__' not in f.parts and f!=p and not ('ml/models' in str(f.relative_to(B)))];p.write_text(json.dumps({str(f.relative_to(B)):{'bytes':f.stat().st_size,'sha256':sha(f)} for f in files},indent=2)+'\n');print('Recorded',len(files),'artifacts')
