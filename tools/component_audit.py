"""Read-only slab component audit. Atom indices in reports are one-based.

Fallback cutoffs reproduce the legacy benchmark, with exact in-plane minimum
images and a nonperiodic vacuum. ±10% sensitivity is a review flag, not a vote.
Mappings preserve requested identity and membership; observed bonds are always
measured from the actual calculation-input coordinates.
"""
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import numpy as np
from .contact_checks import minimum_image_vectors

CUTOFFS = {('C','C'):1.85, ('C','O'):1.70, ('C','H'):1.30,
           ('H','O'):1.20, ('H','H'):.90, ('O','O'):1.65}
FDMNES_CELL_LENGTH_TOLERANCE_A = 2e-5
FDMNES_CELL_ANGLE_TOLERANCE_DEG = 2e-5

def read_poscar(path):
    lines=Path(path).read_text().splitlines();scale=float(lines[1])
    cell=np.array([list(map(float,line.split())) for line in lines[2:5]])
    if scale < 0: scale=(-scale/abs(np.linalg.det(cell)))**(1/3)
    if scale == 0: raise ValueError('Zero POSCAR scale')
    cell*=scale;elements=lines[5].split();counts=list(map(int,lines[6].split()))
    atoms=np.repeat(elements,counts).tolist();i=7
    if lines[i].lower().startswith('s'): i+=1
    direct=lines[i].lower().startswith('d');i+=1
    coords=np.array([list(map(float,line.split()[:3])) for line in lines[i:i+len(atoms)]])
    positions=coords@cell if direct else coords*scale
    if positions.shape!=(len(atoms),3) or not np.isfinite(positions).all():raise ValueError('Invalid POSCAR coordinates')
    return dict(atoms=atoms,positions=positions,cell=cell)

def _components(structure,factor=1.):
    atoms=np.asarray(structure['atoms']);pos=np.asarray(structure['positions']);ids=np.flatnonzero(np.isin(atoms,['C','O','H']))
    delta=pos[ids,None,:]-pos[None,ids,:]
    distance=np.linalg.norm(minimum_image_vectors(delta,structure['cell'],(True,True,False)),axis=-1)
    adjacent=np.zeros(distance.shape,bool)
    for i in range(len(ids)):
        for j in range(i):adjacent[i,j]=adjacent[j,i]=distance[i,j] < factor*CUTOFFS.get(tuple(sorted([atoms[ids[i]],atoms[ids[j]]])),0)
    seen=set();result=[]
    for i in range(len(ids)):
        if i in seen:continue
        todo=[i];seen.add(i);group={i}
        while todo:
            a=todo.pop()
            for b in np.flatnonzero(adjacent[a]):
                if int(b) not in seen:seen.add(int(b));group.add(int(b));todo.append(int(b))
        formula=Counter(atoms[ids[list(group)]]);c,o,h=[formula[x] for x in ['C','O','H']]
        edges=[(a,b) for a in group for b in group if a<b and adjacent[a,b]]
        def n_edges(left,right):
            return sum({atoms[ids[a]],atoms[ids[b]]}=={left,right} for a,b in edges)
        def bonded_to(element, partner):
            return [a for a in group if atoms[ids[a]]==element and any(
                adjacent[a,b] and atoms[ids[b]]==partner for b in group)]
        def c2o2_core():
            cc=n_edges('C','C');co=n_edges('C','O');oo=n_edges('O','O')
            c_degrees=[sum(adjacent[a,b] for b in group if atoms[ids[b]] in ('C','O'))
                       for a in group if atoms[ids[a]]=='C']
            o_degrees=[sum(adjacent[a,b] for b in group if atoms[ids[b]] in ('C','O'))
                       for a in group if atoms[ids[a]]=='O']
            return cc==1 and co==2 and oo==0 and sorted(c_degrees)==[2,2] and sorted(o_degrees)==[1,1]
        species=None
        if (c,o,h)==(0,1,0) and not edges: species='O'
        elif (c,o,h)==(0,0,1) and not edges: species='H'
        elif (c,o,h)==(0,1,1) and len(edges)==1 and n_edges('O','H')==1: species='OH'
        elif (c,o,h)==(1,1,0) and len(edges)==1 and n_edges('C','O')==1: species='CO'
        elif (c,o,h)==(1,2,0) and len(edges)==2 and n_edges('C','O')==2 and n_edges('O','O')==0:
            if all(sum(adjacent[a,b] for b in group if atoms[ids[b]]=='C')==1
                   for a in group if atoms[ids[a]]=='O'): species='CO2'
        elif (c,o,h)==(1,1,1) and len(edges)==2 and n_edges('C','O')==1 and n_edges('C','H')+n_edges('O','H')==1:
            if all(any(adjacent[a,b] and atoms[ids[b]]=='O' for b in group)
                   for a in group if atoms[ids[a]]=='C'):
                species='COH' if n_edges('O','H')==1 else 'CHO'
        elif (c,o,h)==(1,2,1) and len(edges)==3 and n_edges('C','O')==2 and n_edges('O','O')==0:
            if all(sum(adjacent[a,b] for b in group if atoms[ids[b]]=='C')==1
                   for a in group if atoms[ids[a]]=='O'):
                species='COOH' if n_edges('O','H')==1 else 'OCHO' if n_edges('C','H')==1 else None
        elif (c,o,h)==(2,2,0) and len(edges)==3 and c2o2_core(): species='OCCO'
        elif (c,o,h)==(2,2,1) and len(edges)==4 and c2o2_core():
            if n_edges('O','H')==1: species='OCCOH'
            elif n_edges('C','H')==1: species='COCHO'
        result.append(dict(species=species,formula=dict(formula),atom_indices=[int(ids[a])+1 for a in sorted(group)],
                           bonds=[[int(ids[a])+1,int(ids[b])+1] for a in sorted(group) for b in sorted(group) if a<b and adjacent[a,b]]))
    return result

def audit_components(structure,molecules=()):
    components=_components(structure);counts=dict(Counter(x['species'] for x in components if x['species']))
    sensitivity=[]
    for factor in [.9,1.1]:
        variant=_components(structure,factor)
        if [(x['species'],x['atom_indices']) for x in variant]!=[(x['species'],x['atom_indices']) for x in components]:sensitivity.append(factor)
    mappings=[];owned=set()
    for molecule in molecules or ():
        ids=[int(i)+1 for i in molecule.get('atom_indices',[])]
        if not ids or len(ids)!=len(set(ids)) or owned.intersection(ids) or any(i<1 or i>len(structure['atoms']) for i in ids):raise ValueError('Invalid or overlapping molecule mapping')
        owned.update(ids)
        observed=[x for x in components if set(x['atom_indices'])&set(ids)]
        mappings.append(dict(requested=molecule.get('species'),requested_atom_indices=ids,
                             observed=[x['species'] for x in observed],exact_component=any(set(x['atom_indices'])==set(ids) for x in observed),
                             requested_formula=molecule.get('formula_counts'),observed_formula=dict(Counter(structure['atoms'][i-1] for i in ids))))
    for molecule,mapping in zip(molecules or (),mappings):
        expected=molecule.get('expected_edges')
        if expected is not None:
            expected_edges={tuple(sorted((int(a),int(b)))) for a,b in expected}
            mapped=set(mapping['requested_atom_indices'])
            observed_edges={tuple(sorted(edge)) for component in components
                            if set(component['atom_indices'])<=mapped for edge in component['bonds']}
            mapping['expected_edges_one_based']=sorted(map(list,expected_edges))
            mapping['edge_graph_matches']=expected_edges==observed_edges
        exact=[x for x in components if set(x['atom_indices'])==set(mapping['requested_atom_indices'])]
        mapping['observed_species']=[x['species'] for x in exact]
        mapping['identity_matches']=mapping.get('requested') is None or mapping['observed_species']==[mapping['requested']]
    mapping_review=any((m['requested_formula'] is not None and m['requested_formula']!=m['observed_formula'])
                       or m.get('edge_graph_matches') is False or m.get('identity_matches') is False
                       or not m['exact_component'] for m in mappings)
    ambiguous=bool(sensitivity or mapping_review or any(x['species'] is None for x in components))
    return dict(version='slab-components-v1',index_base=1,pbc=[True,True,False],cutoffs_A={':'.join(k):v for k,v in CUTOFFS.items()},
                components=components,counts=counts,identities=sorted(counts),status='review_required' if ambiguous else 'verified',
                sensitivity_factors=sensitivity,mapping_formula_review_required=mapping_review,mapping_evidence=mappings,method='mapped membership + measured connectivity' if molecules else 'legacy connectivity fallback')

def attach_component_audit(record,structure,molecules=()):
    """Attach observed audit to a NEW record; keep requested labels untouched."""
    result=deepcopy(record);audit=audit_components(structure,molecules)
    sample=result.setdefault('sample',{});sample.setdefault('structure_descriptors',{})['component_audit']=audit
    if audit['status']=='verified':
        sample.setdefault('composition',{}).update(adsorbate_counts=audit['counts'],adsorbate_identities=audit['identities'],adsorbate_count_total=sum(audit['counts'].values()))
    return result

def write_metadata_correction(path,corrected,output_dir,apply=False):
    """Opt-in correction, full byte backup and measurement-array hash contract."""
    path=Path(path);original=path.read_bytes();before=json.loads(original)
    digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest(before.get('measurement',{}).get('series'))!=digest(corrected.get('measurement',{}).get('series')):raise ValueError('Metadata correction changes measurement arrays')
    new=(json.dumps(corrected,indent=2)+'\n').encode();entry=dict(before_sha256=hashlib.sha256(original).hexdigest(),after_sha256=hashlib.sha256(new).hexdigest(),measurement_sha256=digest(before.get('measurement',{}).get('series')),applied=bool(apply))
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if apply:
        backup=out/(entry['before_sha256']+'.json');backup.write_bytes(original);path.write_bytes(new);entry['backup']=backup.name
    (out/(entry['before_sha256']+'.correction.json')).write_text(json.dumps(entry,indent=2)+'\n')
    return entry

def calculation_structure(folder):
    """Resolve full-cell coordinates from an actual deck; never guess a parent.

    FDMNES Molecule/cluster-only inputs are deliberately withheld here: cluster
    membership is not the complete decorated configuration's molecule count.
    """
    folder=Path(folder)
    candidates=sorted(folder.glob('*_in.txt'))
    for deck in candidates:
        parsed=read_fdmnes_crystal(deck)
        if parsed is not None:
            return dict(atoms=parsed['atoms'],positions=parsed['fractional']@parsed['cell'],cell=parsed['cell']),deck.name
    if (folder/'INCAR').is_file() and (folder/'POSCAR').is_file():return read_poscar(folder/'POSCAR'),'POSCAR'
    return None,'actual full-cell calculation input unavailable; parent mapping requires review'

def read_fdmnes_crystal(deck):
    """Read the six FDMNES lattice parameters and indexed fractional sites."""
    import re
    is_path=isinstance(deck,Path) or ('\n' not in str(deck) and len(str(deck))<240)
    path=Path(deck) if is_path else None
    text=path.read_text(errors='replace') if path is not None and path.is_file() else str(deck)
    match=re.search(r'(?ims)^Crystal\s*\n(.*?)\n\s*Absorber',text)
    if not match:return None
    lines=match[1].strip().splitlines()
    try:params=np.asarray(list(map(float,lines[0].split())),float)
    except (ValueError,IndexError):return None
    if params.shape!=(6,) or not np.isfinite(params).all():return None
    a,b,c,alpha,beta,gamma=params
    if min(a,b,c)<=0 or any(x<=0 or x>=180 for x in (alpha,beta,gamma)):return None
    alpha,beta,gamma=np.deg2rad([alpha,beta,gamma])
    sg=np.sin(gamma)
    if abs(sg)<1e-10:return None
    cx=c*np.cos(beta);cy=c*(np.cos(alpha)-np.cos(beta)*np.cos(gamma))/sg
    cz2=c*c-cx*cx-cy*cy
    if cz2<=0:return None
    cell=np.array([[a,0,0],[b*np.cos(gamma),b*sg,0],[cx,cy,np.sqrt(cz2)]])
    try:rows=np.array([list(map(float,line.split())) for line in lines[1:] if line.strip()])
    except ValueError:return None
    symbols={1:'H',6:'C',8:'O',29:'Cu',79:'Au'}
    if rows.ndim!=2 or rows.shape[1]!=4 or any(int(z) not in symbols or z!=int(z) for z in rows[:,0]):return None
    return dict(atoms=[symbols[int(z)] for z in rows[:,0]],fractional=rows[:,1:],cell=cell,
                lattice_parameters=[float(x) for x in params],deck=str(path) if path else None)

def legacy_benchmark_geometry_hash(structure):
    """Reproduce the frozen benchmark hash solely to verify its source linkage.

    This is not the current grouping algorithm: new grouping uses in-plane PBC
    and nonperiodic vacuum. Keeping this compatibility hash prevents replacing
    frozen calculation coordinates without an explicit new protocol/version.
    """
    cell=np.asarray(structure['cell'],float);frac=np.asarray(structure['positions'],float)@np.linalg.inv(cell)
    canonical=sorted((str(a),*map(float,v)) for a,v in zip(structure['atoms'],np.mod(np.round(frac%1,8),1)))
    return hashlib.sha256(json.dumps([np.round(cell,7).tolist(),canonical]).encode()).hexdigest()
