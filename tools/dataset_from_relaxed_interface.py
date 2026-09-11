"""Create a labeled Cu K-edge showcase set from a relaxed Cu/Au(111) parent."""
from pathlib import Path
import argparse, csv, json, shutil
import numpy as np
from .relaxed_interface import load_relaxed_interface
from .utils import ADSORBATES
ADSORBATES.setdefault('O', {'atoms':['O'], 'positions':np.array([[0.,0.,0.]]), 'binding_atom':0})

def _write_poscar(path, atoms, pos, cell):
    elems=[]
    for a in atoms:
        if a not in elems: elems.append(a)
    lines=['Cu/Au relaxed-parent showcase','1.0']+[" %.12f %.12f %.12f"%tuple(v) for v in cell]
    lines += [' '.join(elems),' '.join(str(atoms.count(e)) for e in elems),'Selective dynamics','Cartesian']
    lines += [' %.12f %.12f %.12f T T T'%tuple(p) for p in pos]
    Path(path).write_text('\n'.join(lines)+'\n')

def _top_indices(s):
    p=np.asarray(s['positions']); z=p[:,2]; return np.where(z>=z.max()-0.6)[0]

def _site(s, element, rank=0, avoid=()):
    p=np.asarray(s['positions']); zmax=max(float(p[i,2]) for i,a in enumerate(s['atoms']) if a==element); top=np.where(p[:,2]>=zmax-0.6)[0]; ids=[int(i) for i in top if s['atoms'][i]==element and int(i) not in avoid]
    if not ids: raise ValueError('no top '+element)
    # relaxed rows are approximately separated by 2.1--2.5 A; rank from the
    # nearest internal boundary for Cu and the nearest external boundary for Au.
    targets={'Cu':[7.3,5.15,2.9,0.8],'Au':[9.7,12.25,14.75,17.3]}[element]
    t=targets[min(rank,len(targets)-1)]
    return min(ids,key=lambda i:abs(float(p[i,1])-t))

def _add(s, ads, site_ids, coverage, region):
    atoms=list(s['atoms']); p=np.asarray(s['positions'],float).copy(); cell=np.asarray(s['cell'],float)
    data=ADSORBATES[ads]; z0=float(p[:,2].max())
    for k,idx in enumerate(site_ids):
        ref=p[idx].copy(); xyz=np.asarray(data['positions'],float).copy(); b=int(data['binding_atom'])
        xyz += ref + np.array([0.,0.,2.0+0.18*k]) - xyz[b]
        atoms.extend(list(data['atoms'])); p=np.vstack([p,xyz])
    meta=dict(s['metadata']); meta.update({'adsorbate':ads,'adsorbate_count':len(site_ids),'coverage_fraction':len(site_ids)/68.0,
      'coverage_ml_top_sites':len(site_ids)/68.0,'adsorption_region':region,'binding_element':'Cu',
      'local_geometry_label':region,'xas_edge':'Cu K','absorber_element':'Cu','label_schema_version':'1.0'})
    return {'atoms':atoms,'positions':p,'cell':cell,'metadata':meta}

def build(parent, out):
    out=Path(out); out.mkdir(parents=True,exist_ok=True); shutil.copy2(parent,out/'parent_CONTCAR')
    base=load_relaxed_interface(parent); rows=[]; n=0
    specs=[('clean',None,0,1,'clean'),('H','Cu',0,1,'Cu_side_interface'),('OH','Cu',0,1,'Cu_side_interface'),('O','Cu',0,1,'Cu_side_interface'),('CO','Cu',0,1,'Cu_side_interface'),('CHO','Cu',0,1,'Cu_side_interface'),('CO2','Cu',0,1,'Cu_side_interface'),('CH','Cu',0,1,'Cu_side_interface'),('CH2','Cu',1,1,'Cu_near_interface_row_1'),('CH3','Cu',1,1,'Cu_near_interface_row_1'),('OCCO','Cu',0,1,'Cu_Au_boundary_bridge'),('COCO','Cu',0,1,'Cu_Au_boundary_bridge'),('CO','Cu',1,1,'Cu_near_interface_row_1'),('CO','Cu',2,1,'Cu_near_interface_row_2'),('OH','Cu',1,1,'Cu_near_interface_row_1'),('OH','Cu',2,1,'Cu_near_interface_row_2'),('H','Cu',1,1,'Cu_near_interface_row_1'),('H','Cu',2,1,'Cu_near_interface_row_2'),('CO','Cu',0,2,'Cu_side_interface'),('CO','Cu',0,4,'Cu_side_interface'),('OH','Cu',0,2,'Cu_side_interface'),('OH','Cu',0,4,'Cu_side_interface'),('H','Cu',0,2,'Cu_side_interface'),('CO','Au',0,1,'Au_side_interface'),('CO+OH','Cu',0,2,'coadsorption_Cu_interface')]
    for ads,el,rank,count,region in specs:
        if ads=='clean': s=base; actual='clean'; ids=[]
        elif ads=='CO+OH':
            ids=[_site(base,'Cu',0),_site(base,'Cu',1,avoid=(_site(base,'Cu',0),))]; s=_add(base,'CO',ids[:1],1,region); s=_add(s,'OH',[_site(s,'Cu',1,avoid=(ids[0],))],1,region); actual='CO+OH'; ids=[]
        else:
            actual=ads; ids=[_site(base,el,rank)]
            if count>1:
                pool=[i for i in _top_indices(base) if base['atoms'][i]==el and i!=ids[0]]; ids += sorted(pool,key=lambda i:abs(float(base['positions'][i,0]-base['positions'][ids[0],0])))[:count-1]
            s=_add(base,ads,ids,count,region)
        sid=f'{n:03d}_{actual}_{region}_{count}site'; d=out/sid; d.mkdir(); _write_poscar(d/'POSCAR',s['atoms'],s['positions'],s['cell']); (d/'structure_info.json').write_text(json.dumps(s['metadata'],indent=2,default=str));
        rows.append({'id':sid,'adsorbate':actual,'coverage_ml':s['metadata'].get('coverage_ml_top_sites',0.0),'region':region,'local_geometry_label':region,'absorber':'Cu','edge':'K','parent_relaxation':'relaxed','structure_status':'parent_relaxed_plus_adsorbate_unrelaxed'}); n+=1
    with (out/'dataset_manifest.csv').open('w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    (out/'dataset_manifest.json').write_text(json.dumps(rows,indent=2)); return rows
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--parent',required=True); ap.add_argument('--output',required=True); a=ap.parse_args(); print(f'generated {len(build(a.parent,a.output))} structures in {a.output}')
