"""Separate verified composition/cell/population compatibility from parent/constraint evidence."""
from pathlib import Path
import json,sys
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_study import dump
from tools.component_audit import read_poscar
from tools.chemical_lineage import correspondence
from tools.progress_snapshots import flags
B=ROOT/'study/dataset_20261029';raw=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else B/'local_export';rows=json.load(open(B/'training_manifest.json'));result=[];clean=read_poscar(raw/'plain/CONTCAR');cf=np.array(flags(raw/'plain/CONTCAR',len(clean['atoms'])))
for r in rows:
 item={'configuration':r['configuration'],'domain':r['domain'],'composition_cell_physical_layers_checked_by_training_route':True,'initial_metal_model_match':False,'independent_ancestral_lineage':'not established by coordinate equality alone'}
 if r['domain']=='Cu100':
  fixed=r.get('ASE_lineage',{}).get('fixed_indices_0based');clean_flags=flags(raw/'Cu100/Cu100_2x2x4_clean/XAS/FDMNES/green/layer_4/CONTCAR',16);clean_fixed=[i for i,f in enumerate(clean_flags) if f==['F','F','F']];item.update(constraint_match=fixed==clean_fixed,reference_match='same Cu16 cell/facet and bottom-eight fixed convention; exact initial parent ancestry unproven',sample_fixed_indices=fixed,clean_fixed_indices=clean_fixed)
 else:
  p=raw/r['configuration']/'POSCAR'
  if p.exists():
   s=read_poscar(p);mask=np.isin(s['atoms'],['Cu','Au']);metal={'atoms':np.array(s['atoms'])[mask].tolist(),'positions':s['positions'][mask],'cell':s['cell']};link=correspondence(metal,clean);ff=np.array(flags(p,len(s['atoms'])))[mask];matched=link['status']=='matched';same=False
   if matched:same=all(np.array_equal(ff[a-1],cf[b-1]) for a,b in link['one_based_mapping'])
   item.update(initial_metal_model_match=bool(matched and same),constraint_match=bool(same) if matched else None,initial_metal_correspondence=link,reference_match='initial metal model + constraint match' if matched and same else 'weaker same-substrate/reference sensitivity; constraints or initial parent not matched')
  else:item.update(constraint_match=None,reference_match='weaker reference; no initial POSCAR supplied')
 result.append(item)
dump(B/'reference_compatibility.json',result);pd.DataFrame([{k:v for k,v in r.items() if not isinstance(v,(dict,list))} for r in result]).to_csv(B/'reference_compatibility.csv',index=False);print('initial-model matched AuCu',sum(r['domain']=='CuAu' and r['initial_metal_model_match'] for r in result),'Cu100 constraint matches',sum(r['domain']=='Cu100' and r['constraint_match'] for r in result))
