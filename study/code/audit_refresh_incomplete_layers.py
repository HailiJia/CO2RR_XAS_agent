"""Inventory partial layers and retain an unvalidated numeric recovery without training on it."""
from pathlib import Path
import os,shutil
import sys,json,re,tarfile,hashlib,io,gzip
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fdmnes_scale import headers,convolve
from tools.coverage_study import dump,sha
B=ROOT/'study/dataset_20261029';raw=B/'local_export';binary=os.environ.get('FDMNES_BINARY',shutil.which('fdmnes') or '/tmp/fdmnes_mac_official/fdmnes_Mac/fdmnes_mac');audit=json.load(open(B/'record_geometry_audit.json'));out=[];lineage=[]
for r in audit:
 if r.get('exclusion')!='Missing top-two physical layers':continue
 for l in sorted(set([4,5])-set(r['layers'])):
  folder=raw/r['configuration']/f'XAS/FDMNES/green/layer_{l}';hs=headers(folder,l);table=pd.read_csv(folder/'absorbers.tsv',sep='\t');ids=set(table.atom_index);missing=sorted(ids-{h['atom_index'] for h in hs});bad=[]
  for h in hs:
   a=np.loadtxt(folder/h['file'],skiprows=2)
   if len(a)!=276 or a[0,0]!=8974 or a[-1,0]!=9029:bad.append({'atom':h['atom_index'],'points':len(a),'window':[float(a[0,0]),float(a[-1,0])]})
  bav=folder/f'Cu_layer{l}_bav.txt';started=list(map(int,re.findall(r'Absorbing atom\s+(\d+) beginning',bav.read_text(errors='replace')))) if bav.exists() else [];status='incomplete electronic site coverage; no full-layer reconstruction justified'
  if not missing and not bad:
   dest=B/'recovered_candidates'/r['configuration']/f'layer_{l}';a,_=convolve(binary,folder,l,dest,'unit_weight_control');pd.DataFrame({'energy_eV':a[:,0],'candidate_native_sum':a[:,1],'candidate_per_Cu':a[:,1]/len(ids)}).to_csv(dest/'candidate_arrays.csv',index=False);status='numeric convolution recoverable; surviving BAV records subset-only recalculation, full raw-site geometry lineage unverified; EXCLUDED from ML'
  out.append({'configuration':r['configuration'],'missing_layer':l,'physical_population':len(ids),'complete_raw_file_count':len(hs)-len(bad),'missing_raw_atom_indices':missing,'incomplete_raw_ranges':bad,'BAV_absorber_starts':started,'source_multiplicity_sum':sum(h['natomsym_f'] for h in hs),'status':status})
  if bav.exists():lineage.append((bav,str(bav.relative_to(raw))))
dump(B/'incomplete_layer_status.json',out);pd.DataFrame([{**r,'missing_raw_atom_indices':json.dumps(r['missing_raw_atom_indices']),'incomplete_raw_ranges':json.dumps(r['incomplete_raw_ranges']),'BAV_absorber_starts':json.dumps(r['BAV_absorber_starts'])} for r in out]).to_csv(B/'incomplete_layer_status.csv',index=False)
# Preserve the actual surviving logs needed to independently review these exclusions.
archive=B/'sources/partial_layer_lineage.tar.gz';manifest=[]
with archive.open('wb') as stream,gzip.GzipFile(filename='',fileobj=stream,mode='wb',mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tf:
 for p,name in lineage:
  data=p.read_bytes();m=tarfile.TarInfo(name);m.size=len(data);m.mtime=0;tf.addfile(m,io.BytesIO(data));manifest.append({'path':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'archive':archive.name})
with tarfile.open(archive) as tf:
 for r in manifest:assert hashlib.sha256(tf.extractfile(r['path']).read()).hexdigest()==r['sha256']
dump(B/'sources/partial_layer_lineage_manifest.json',manifest);info={'archive':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'note':'additional raw provenance logs needed to verify partial/mixed-run exclusions; not completed training spectra'}
from tools.dataset_refresh_sources import partition_archive
info['parts']=partition_archive(archive);dump(B/'sources/partial_layer_archive.json',info)
print('partial layers',len(out),'extra lineage archive bytes',info['bytes'])
