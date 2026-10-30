"""Lossless versioned model package; safe hash-verified portable restore."""
from pathlib import Path
import hashlib,json,tarfile,sys,io,gzip
B=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
if '--pack' in sys.argv:
 rows=[]
 with (B/'models.tar.gz').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='') as gz,tarfile.open(fileobj=gz,mode='w|') as tf:
  for folder in ['results/models','paired_robustness/models','strict_sensitivity/models','group_learning/models']:
   for p in sorted((B/folder).glob('*.joblib')):
    data=p.read_bytes();name=str(p.relative_to(B));info=tarfile.TarInfo(name);info.size=len(data);info.mtime=0;tf.addfile(info,io.BytesIO(data));rows.append({'path':name,'bytes':len(data),'sha256':sha(data)})
 data=(B/'models.tar.gz').read_bytes();parts=[]
 for i,offset in enumerate(range(0,len(data),16000000)):
  name=f'models.tar.gz.part{i:03d}';chunk=data[offset:offset+16000000];(B/name).write_bytes(chunk);parts.append({'path':name,'bytes':len(chunk),'sha256':sha(chunk)})
 (B/'model_archive_manifest.json').write_text(json.dumps({'archive':'models.tar.gz','bytes':len(data),'sha256':sha(data),'parts':parts,'members':rows},indent=2)+'\n');(B/'models.tar.gz').unlink();print('Packed',len(rows),'models in',len(parts),'parts')
else:
 m=json.load(open(B/'model_archive_manifest.json'));chunks=[]
 for part in m['parts']:
  data=(B/part['path']).read_bytes();assert len(data)==part['bytes'] and sha(data)==part['sha256'];chunks.append(data)
 data=b''.join(chunks);assert len(data)==m['bytes'] and sha(data)==m['sha256'];lookup={a['path']:a for a in m['members']}
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tf:
  for member in tf:
   assert member.isfile() and member.name in lookup;data=tf.extractfile(member).read();r=lookup.pop(member.name);assert len(data)==r['bytes'] and sha(data)==r['sha256'];p=(B/member.name).resolve();assert B.resolve() in p.parents
   if '--unpack' in sys.argv:
    p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():assert p.read_bytes()==data
    else:p.write_bytes(data)
 assert not lookup;print('Verified',len(m['members']),'models')
