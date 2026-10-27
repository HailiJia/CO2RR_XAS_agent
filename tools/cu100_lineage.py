"""Match original ASE snapshots to exact FDMNES geometry; no relaxation claim."""
from pathlib import Path
import json,hashlib
import numpy as np
from .chemical_lineage import correspondence
from .component_audit import audit_components
from collections import Counter

def array(d):
 shape,typ,values=d['__ndarray__'];return np.asarray(values).reshape(shape)
def snapshot_graph(s):
    """Extend explicit graph identities for water/carbonate; no charge inferred."""
    graph=audit_components(s)
    for m in graph['components']:
        if m['species'] is not None:continue
        f=m['formula'];edges=m['bonds'];types=[tuple(sorted([s['atoms'][i-1],s['atoms'][j-1]])) for i,j in edges]
        if f=={'O':1,'H':2} and types.count(('H','O'))==2 and len(edges)==2:m['species']='H2O'
        elif f=={'C':1,'O':3} and types.count(('C','O'))==3 and len(edges)==3:m['species']='CO3'
    graph['counts']=dict(Counter(m['species'] for m in graph['components'] if m['species']))
    graph['identities']=sorted(graph['counts'])
    graph['status']='review_required' if graph['sensitivity_factors'] or any(m['species'] is None for m in graph['components']) else 'verified'
    graph['extension']='water = two O-H bonds; CO3 = three C-O bonds; formal charge not inferred'
    return graph

def verify(root,configuration,structure):
    base=Path(root)/'Cu-Ox_updated_structures/100';name=Path(configuration).name+'.json';paths=sorted(base.rglob(name))
    if len(paths)!=1:return {'status':'missing_or_ambiguous_snapshot','candidate_paths':[str(p.relative_to(root)) for p in paths]}
    p=paths[0];d=json.loads(p.read_text());ids=d.get('ids',[])
    if len(ids)!=1:raise ValueError('ASE file has multiple frames; explicit frame selection required')
    row=d[str(ids[0])];numbers=array(row['numbers']);symbols={1:'H',6:'C',8:'O',29:'Cu'};s={'atoms':[symbols[int(z)] for z in numbers],'positions':array(row['positions']),'cell':array(row['cell']['array'])};match=correspondence(s,structure)
    if match['status']!='matched':raise ValueError('Original ASE snapshot differs from actual XAS-input geometry')
    fixed=set()
    for c in row.get('constraints',[]):
        if c.get('name')=='FixAtoms':fixed.update(c['kwargs']['indices'])
    force=array(row['forces']);free=[i for i in range(len(numbers)) if i not in fixed];maximum=float(np.linalg.norm(force[free],axis=1).max())
    return {'status':'matched_original_ASE_snapshot','path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'ASE_unique_id':row.get('unique_id'),'atom_mapping':match,'observed_graph':snapshot_graph(s),'max_free_force_eV_per_A':maximum,'fixed_indices_0based':sorted(fixed),'calculator':row.get('calculator'),'calculator_parameters':row.get('calculator_parameters'),'relaxation_status':'stored VASP energy/forces snapshot; stopping criterion and run termination not supplied','periodicity_for_graph':[True,True,False],'stored_pbc':array(row['pbc']).tolist(),'structure':{k:np.asarray(v).tolist() for k,v in s.items()}}
