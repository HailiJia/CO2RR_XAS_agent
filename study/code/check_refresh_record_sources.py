"""Compare every versioned series with its exact source array and geometry copies."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import native,deck_structure,dump,sha
from tools.component_audit import read_poscar
from tools.chemical_lineage import correspondence
B=ROOT/'study/dataset_20261029';raw=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else B/'local_export';table=[]
for p in sorted((B/'records').rglob('isaac_record_layers.json')):
 alias=str(p.parent.relative_to(B/'records'));d=json.load(open(p));f=raw/alias/'XAS/FDMNES/green';fs=native(f)
 if alias.endswith('Cu100_2x2x4_clean'):continue
 for series in d['measurement']['series']:
  l=int(series['series_id'].rsplit('_',1)[1]);a=np.column_stack([series['independent_variables'][0]['values'],series['channels'][0]['values']]);assert np.array_equal(a,fs[l][1]);s=deck_structure(f/f'layer_{l}/Cu_layer{l}_in.txt');copy=f/f'layer_{l}/CONTCAR';root=raw/alias/'CONTCAR';cm=correspondence(read_poscar(copy),s) if copy.exists() else {'status':'not_supplied'};rm=correspondence(read_poscar(root),s) if root.exists() else {'status':'not_supplied'};table.append({'configuration':alias,'layer':l,'record_sha256':sha(p),'source_array_sha256':sha(fs[l][0]),'array_exact':True,'layer_CONTCAR_status':cm['status'],'root_CONTCAR_status':rm['status'],'layer_CONTCAR_mapping':cm,'root_CONTCAR_mapping':rm,'root_or_layer_mismatch_retained_for_review':cm['status'] not in ['matched','not_supplied'] or rm['status'] not in ['matched','not_supplied']})
dump(B/'record_series_validation.json',table);pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,dict)} for r in table]).to_csv(B/'record_series_validation.csv',index=False);print('series',len(table),'layer-copy mismatches',sum(r['layer_CONTCAR_status'] not in ['matched','not_supplied'] for r in table),'root mismatches',sum(r['root_CONTCAR_status'] not in ['matched','not_supplied'] for r in table))
