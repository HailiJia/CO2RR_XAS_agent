"""Freeze conversion inputs and compact raw evidence, excluding potentials/binaries."""
from pathlib import Path
import sys,json,tarfile,io,hashlib,gzip,re
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import folders,sha,dump
B=ROOT/'study/coverage_20261026';raw=Path(sys.argv[1]);files={};manifest=[]
for f in folders(raw):
 alias=str(f.relative_to(raw)).split('/XAS/')[0]
 paths=[raw/alias/'CONTCAR',raw/alias/'POSCAR']
 for sub in f.glob('layer_*'):
  paths+=list(sub.glob('Cu_layer*_in.txt'))+list(sub.glob('Cu_layer*_conv.txt'))+list(sub.glob('fdmnes_convolution*.out'))+[sub/'absorbers.tsv',sub/'CONTCAR']
 for p in sorted(set(paths)):
  if not p.is_file():continue
  if p.stat().st_size>5*1024**2:continue # large logs remain author-local; converter assets preserve hashes
  h=sha(p);files[h]=p;manifest.append({'path':str(p.relative_to(raw)),'sha256':h,'bytes':p.stat().st_size})
for p in list((raw/'Cu-Ox_updated_structures/100').rglob('*.json'))+[(raw/'convert_layers_to_isaac.sh.before_20261026') if (raw/'convert_layers_to_isaac.sh.before_20261026').exists() else raw/'convert_layers_to_isaac.sh',raw/'tools/convert_simulation_xas_run_to_isaac.py',raw/'Cu100/manifest.csv',raw/'Cu100/README.md',raw/'Cu100/prepare_cu100_green.py']:
 h=sha(p);files[h]=p;manifest.append({'path':str(p.relative_to(raw)),'sha256':h,'bytes':p.stat().st_size})
with (B/'source_evidence.tar.gz').open('wb') as compressed, gzip.GzipFile(fileobj=compressed,mode='wb',mtime=0,filename='') as gz, tarfile.open(fileobj=gz,mode='w') as t:
 for h,p in sorted(files.items()):
  b=p.read_bytes();i=tarfile.TarInfo('objects/'+h);i.size=len(b);i.mtime=0;t.addfile(i,io.BytesIO(b))
dump(B/'source_manifest.json',manifest)
(B/'source_evidence.tar').unlink(missing_ok=True)
# Backend raw-site arrays are hashed but not duplicated as unrelated bulk exports.
if not (B/'native_absorber_file_manifest.json').exists():
 rawsites=[]
 for f in folders(raw):
  for p in f.glob('layer_*/Cu_layer*_*.txt'):
   if re.fullmatch(r'Cu_layer\d+_\d+\.txt',p.name):rawsites.append({'path':str(p.relative_to(raw)),'sha256':sha(p),'bytes':p.stat().st_size})
 dump(B/'native_absorber_file_manifest.json',rawsites)
if not (B/'native_runtime_log_manifest.json').exists():
 rawlogs=[]
 for f in folders(raw):
  for p in f.glob('layer_*/fdmnes*.out'):rawlogs.append({'path':str(p.relative_to(raw)),'sha256':sha(p),'bytes':p.stat().st_size})
 dump(B/'native_runtime_log_manifest.json',rawlogs)
# Original snapshots with no corresponding supplied calculation are explicit gaps.
calculated={Path(r['configuration']).name for r in json.loads((B/'conversion_inventory.json').read_text()) if r['configuration'].startswith('Cu100/')}
dump(B/'Cu100_snapshots_without_outputs.json',[{'path':str(p.relative_to(raw)),'sha256':sha(p),'status':'no corresponding Cu100 calculation folder located in supplied export; not a claim of never run'} for p in (raw/'Cu-Ox_updated_structures/100').rglob('*.json') if p.stem not in calculated])
# A source/geometry mismatch must be visible inside the record itself, not
# only in a companion table. Native measurement.series remain unchanged.
review=json.loads((B/'record_geometry_audit.json').read_text())
for r in review:
 if r.get('exclusion')=='Unresolved root CONTCAR / calculation-input mismatch':
  p=B/r['converted_record'];d=json.loads(p.read_text());d['geometry_audit']={'status':'review_required','reason':r['exclusion'],'training_eligible':False};d['measurement'].setdefault('qc',{}).update(status='flagged',notes=r['exclusion']);dump(p,d);r['audited_record_sha256']=sha(p)
dump(B/'record_geometry_audit.json',review)
import pandas as pd
pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in review]).to_csv(B/'record_geometry_audit.csv',index=False)
# Copy newly converted records to missing source locations, without replacing
# any author's existing record. All authoritative audit versions remain in B.
written=[];previous={r['path']:r for r in json.loads((B/'new_local_records_written.json').read_text())} if (B/'new_local_records_written.json').exists() else {}
for r in json.loads((B/'record_geometry_audit.json').read_text()):
 if r.get('conversion')!='converted':continue
 dst=raw/r['configuration']/'XAS/FDMNES/green/isaac_record_layers.json'
 if dst.exists() and (str(dst.relative_to(raw)) not in previous or sha(dst)!=previous[str(dst.relative_to(raw))]['sha256']):continue
 src=B/r['converted_record'];dst.write_bytes(src.read_bytes());written.append({'path':str(dst.relative_to(raw)),'sha256':sha(dst),'geometry_eligible':r['eligible'],'geometry_status':'input_geometry_only_CONTCAR_pending' if r['domain']=='Cu100' else r.get('exclusion','audited')})
dump(B/'new_local_records_written.json',written)
print('source entries',len(manifest),'unique objects',len(files),'new local records',len(written))
