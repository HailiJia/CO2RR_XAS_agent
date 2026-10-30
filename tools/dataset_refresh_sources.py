"""Restore exact selected source bytes, including pinned reused archives; no network or portal."""
from pathlib import Path
import argparse,json,hashlib,tarfile
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1]
def restore(output):
 output=Path(output).resolve();output.mkdir(parents=True,exist_ok=True);B=ROOT/'study/dataset_20261029';rows=json.load(open(B/'sources/manifest.json'));groups=defaultdict(list)
 for r in rows:
  st=r['storage'];archive=ROOT/st['previous_archive'] if 'previous_archive'in st else B/'sources'/st['archive'];groups[archive].append(r)
 count=0
 for archive,entries in groups.items():
  with tarfile.open(archive) as tf:
   for r in entries:
    p=(output/r['path']).resolve()
    if output not in p.parents:raise ValueError('Unsafe source path')
    data=tf.extractfile(r['storage']['member']).read();assert len(data)==r['bytes'] and hashlib.sha256(data).hexdigest()==r['sha256'];p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data,'Refuse to overwrite a changed source'
    else:p.write_bytes(data)
    count+=1
 # Exact original Cu100 snapshots already uploaded and independently matched in the audit.
 archive=ROOT/'study/differential_v1/sources/Cu100_original_structures.tar.gz';old={r['logical_path']:r for r in json.load(open(ROOT/'study/differential_v1/sources/path_hash_size_manifest.json')) if r['archive']==archive.name}
 with tarfile.open(archive) as tf:
  for member in tf.getmembers():
   if not member.isfile():continue
   data=tf.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==old[member.name]['sha256'];name=member.name.removeprefix('workspace/');p=(output/name).resolve();assert output in p.parents;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==data
   else:p.write_bytes(data)
 # Supplemental surviving partial-run lineage logs, stored as practical lossless parts.
 extra=B/'sources/partial_layer_archive.json'
 if extra.exists():
  import tempfile
  info=json.load(open(extra));manifest=json.load(open(B/'sources/partial_layer_lineage_manifest.json'))
  with tempfile.TemporaryFile() as joined:
   h=hashlib.sha256()
   for part in info['parts']:
    data=(B/'sources'/part['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==part['sha256'];h.update(data);joined.write(data)
   assert h.hexdigest()==info['sha256'];joined.seek(0)
   with tarfile.open(fileobj=joined,mode='r:gz') as tf:
    for r in manifest:
     data=tf.extractfile(r['path']).read();assert hashlib.sha256(data).hexdigest()==r['sha256'];p=(output/r['path']).resolve();assert output in p.parents;p.parent.mkdir(parents=True,exist_ok=True)
     if p.exists():assert p.read_bytes()==data
     else:p.write_bytes(data)
  count+=len(manifest)
 print(json.dumps({'source_files_verified_and_restored':count,'original_Cu100_snapshots_verified':len(old),'output':str(output)}))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();restore(a.output)

def partition_archive(path,limit=16000000):
 path=Path(path);parts=[];h=hashlib.sha256()
 with path.open('rb') as f:
  index=0
  while True:
   data=f.read(limit)
   if not data:break
   dest=path.with_name(path.name+f'.part{index:03d}');dest.write_bytes(data);h.update(data);parts.append({'path':dest.name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()});index+=1
 assert h.hexdigest()==hashlib.sha256(path.read_bytes()).hexdigest()
 path.unlink();return parts
