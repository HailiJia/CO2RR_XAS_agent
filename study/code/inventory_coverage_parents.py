"""Inventory exact initial metal geometries without inventing parent lineage."""
from pathlib import Path
import sys,json
from collections import Counter
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.component_audit import read_poscar
from tools.chemical_lineage import correspondence,parent_key
from tools.coverage_study import dump,sha
B=ROOT/'study/coverage_20261026';raw=Path(sys.argv[1]);rows=json.loads((B/'model_configuration_manifest.json').read_text());items=[]
def metal(s):
 ids=np.isin(s['atoms'],['Cu','Au']);return {'atoms':np.array(s['atoms'])[ids].tolist(),'positions':s['positions'][ids],'cell':s['cell']}
clean=raw/'plain/CONTCAR';ref=metal(read_poscar(clean)) if clean.exists() else None
for r in rows:
 p=raw/r['configuration']/'POSCAR'
 if not p.exists():continue
 s=read_poscar(p);m=metal(s);items.append({'configuration':r['configuration'],'initial_POSCAR':str(p.relative_to(raw)),'sha256':sha(p),'metal_coordinate_hash':parent_key(s),'metal_composition':dict(Counter(m['atoms'])),'matches_saved_relaxed_clean_coordinates':correspondence(m,ref) if ref else None,'caution':'exact coordinate families are not independent ancestral parents; trajectory/initial-parent lineage can be missing'})
dump(B/'parent_control_inventory.json',{'initial_geometries':items,'exact_initial_metal_geometry_families':len(set(x['metal_coordinate_hash'] for x in items)),'verified_independent_parent_count':None,'clean_reference_record_available':any(r['configuration']=='plain' for r in rows),'clean_reference_layers':[4,5],'note':'New clean top-two spectra are available. Do not call independently min-max/unit-area-scaled curves a physical adsorbate-minus-clean difference; source-scale and parent/control matching are separate required checks. No physical clean subtraction is used in this coverage study.'})
print('initial metal coordinate families',len(set(x['metal_coordinate_hash'] for x in items)),'exact clean matches',sum(x['matches_saved_relaxed_clean_coordinates']['status']=='matched' for x in items))
