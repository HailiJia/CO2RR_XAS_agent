"""Utilities for using a relaxed POSCAR/CONTCAR as an interface parent."""
from pathlib import Path
from typing import Any, Dict, List
import numpy as np
from .utils import read_structure_file

def _read_vasp_any(path: str) -> Dict[str, Any]:
    """Read POSCAR text even when an uploaded file has a nonstandard name."""
    lines=Path(path).read_text().splitlines(); scale=float(lines[1]); cell=np.asarray([[float(x) for x in lines[i].split()[:3]] for i in range(2,5)],float)*scale
    elems=lines[5].split(); counts=[int(x) for x in lines[6].split()]; n=sum(counts); k=7
    if lines[k].strip().lower().startswith('s'): k+=1
    direct=lines[k].strip().lower().startswith('d'); k+=1
    q=np.asarray([[float(x) for x in lines[k+i].split()[:3]] for i in range(n)],float)
    if direct: q=q@cell
    return {'atoms':sum(([e]*c for e,c in zip(elems,counts)),[]),'positions':q,'cell':cell,'metadata':{}}

def _layers(z: np.ndarray, tol: float = 0.35) -> List[np.ndarray]:
    order=np.argsort(z); groups=[]
    for i in order:
        if not groups or z[i]-z[groups[-1][-1]]>tol: groups.append([int(i)])
        else: groups[-1].append(int(i))
    return [np.asarray(g,dtype=int) for g in groups]

def load_relaxed_interface(path: str, element1: str='Cu', element2: str='Au', cu_rows: int=4, au_rows: int=4) -> Dict[str,Any]:
    try: s=read_structure_file(path)
    except Exception: s=_read_vasp_any(path)
    atoms=list(s['atoms']); pos=np.asarray(s['positions'],float); cell=np.asarray(s['cell'],float)
    ls=_layers(pos[:,2]); top=ls[-1]
    # The uploaded 1:1 parent has two periodic boundaries.  Keep both explicit
    # and use the internal Cu/Au boundary near the middle of the y cell.
    split=float(cell[1,1])*0.5
    ys=pos[top,1]
    e1=[i for i in top if atoms[i]==element1]; e2=[i for i in top if atoms[i]==element2]
    if e1 and e2:
        # select the largest Cu-to-smallest Au gap after circular unwrapping
        cu=np.sort(pos[e1,1]); au=np.sort(pos[e2,1]);
        split=float(0.5*(np.max(cu[cu<cell[1,1]*0.5])+np.min(au))) if np.any(cu<cell[1,1]*0.5) else split
    m=dict(s.get('metadata',{})); m.update({
      'structure_source':'uploaded_relaxed_parent','parent_file':str(Path(path).resolve()),
      'relaxation_status':'relaxed','structure_type':'interface','interface_type':'lateral',
      'element1':element1,'element2':element2,'facet1':'111','facet2':'111','facet':'111',
      'cu_rows':int(cu_rows),'au_rows':int(au_rows),'stripe_ratio':'1:1','layers1':len(ls),'layers2':len(ls),
      'top_layer_atom_count':int(len(top)), 'parent_layer_counts':[int(len(x)) for x in ls],
      'interface':{'type':'lateral','interface_axis':'x','split_axis':'y','split_coordinate':split,
        'cu_rows':int(cu_rows),'au_rows':int(au_rows),'element1_rows':int(cu_rows),'element2_rows':int(au_rows),
        'stripe_ratio':'1:1','actual_repeats':{'Cu':9,'Au':8},
        'parent_relaxed':True}
    })
    s['metadata']=m
    return s
