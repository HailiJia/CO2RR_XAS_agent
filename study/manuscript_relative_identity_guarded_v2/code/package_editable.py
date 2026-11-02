"""Build the editable ZIP with a non-self-referential content manifest."""
from pathlib import Path
import argparse,hashlib,json,zipfile
B=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def build(output):
 output=Path(output).resolve();files=[p for p in sorted(B.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='package_manifest.json' and p.resolve()!=output]
 manifest={str(p.relative_to(B)):{'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in files};path=B/'package_manifest.json';path.write_text(json.dumps(manifest,indent=2)+'\n')
 # The ZIP includes the manifest; the manifest deliberately excludes itself.
 with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
  for p in files+[path]:z.write(p,'CO2RR_Reviewed_Manuscript/'+str(p.relative_to(B)))
 with zipfile.ZipFile(output) as z:
  assert z.testzip() is None
  for p in files+[path]:assert z.read('CO2RR_Reviewed_Manuscript/'+str(p.relative_to(B)))==p.read_bytes()
 for name,row in manifest.items():
  data=(B/name).read_bytes();assert len(data)==row['bytes'] and sha(data)==row['sha256']
 print(json.dumps({'output':str(output),'bytes':output.stat().st_size,'sha256':sha(output.read_bytes()),'hashed_files':len(manifest),'zip_members':len(files)+1}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
