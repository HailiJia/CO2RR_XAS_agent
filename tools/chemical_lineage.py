"""Read-only geometry/record lineage audit; names are search hints, never targets."""
from pathlib import Path
from collections import Counter
import argparse, hashlib, json, os
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .component_audit import read_poscar, read_fdmnes_crystal, audit_components
from .contact_checks import minimum_image_vectors
from .dataset_labels import geometry_hash


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def parent_key(s):
    ids=np.isin(s['atoms'],['Cu','Au'])
    return geometry_hash(dict(atoms=np.asarray(s['atoms'])[ids].tolist(),positions=s['positions'][ids],cell=s['cell']))


def correspondence(a,b):
    if Counter(a['atoms'])!=Counter(b['atoms']):return {'status':'different_atom_counts'}
    if not np.allclose(a['cell']@a['cell'].T,b['cell']@b['cell'].T,atol=2e-3,rtol=0):return {'status':'different_cell_metric'}
    bp=b['positions']@np.linalg.inv(b['cell'])@a['cell'];mapping=[];dist=[]
    for element in sorted(set(a['atoms'])):
        ia=np.flatnonzero(np.asarray(a['atoms'])==element);ib=np.flatnonzero(np.asarray(b['atoms'])==element)
        delta=minimum_image_vectors(a['positions'][ia,None]-bp[ib][None,:],a['cell'],(True,True,False))
        cost=np.linalg.norm(delta,axis=-1);ii,jj=linear_sum_assignment(cost)
        mapping.extend([[int(ia[i]+1),int(ib[j]+1)] for i,j in zip(ii,jj)]);dist.extend(cost[ii,jj])
    return {'status':'matched' if max(dist)<2e-3 else 'geometry_changed','max_distance_A':float(max(dist)),
            'rms_distance_A':float(np.sqrt(np.mean(np.square(dist)))),'one_based_mapping':sorted(mapping),
            'index_reordered':any(i!=j for i,j in mapping),
            'mapping_caveat':'Element-wise nearest-coordinate assignment verifies exact geometry copies. For changed relaxed geometries it is not a trajectory-based proof of atom exchange or reordering.',
            'same_index_max_displacement_A':float(np.linalg.norm(minimum_image_vectors(a['positions']-bp,a['cell'],(True,True,False)),axis=1).max()) if a['atoms']==b['atoms'] else None}


def local_info(s,ids):
    atoms=np.asarray(s['atoms']);p=s['positions'];ids=np.asarray(ids,int)-1
    metal=np.flatnonzero(np.isin(atoms,['Cu','Au']));h=np.flatnonzero(atoms=='H');c=np.flatnonzero(atoms=='C');o=np.flatnonzero(atoms=='O')
    def distances(left,right):
        return [[int(i+1),int(j+1),float(np.linalg.norm(minimum_image_vectors(p[i]-p[j],s['cell'],(True,True,False))))] for i in left for j in right]
    contacts=distances(ids,metal);contacts=[x for x in contacts if x[2]<3.0]
    # Boundary estimate uses the top-layer Cu/Au row sequence; disclosed geometric descriptor only.
    normal=np.cross(s['cell'][0],s['cell'][1]);normal/=np.linalg.norm(normal);height=p@normal
    top=metal[height[metal]>max(height[metal])-.6];frac=p@np.linalg.inv(s['cell']);rows=[]
    for i in sorted(top,key=lambda i:frac[i,1]%1):
        y=float(frac[i,1]%1)
        if not rows or (y-rows[-1][-1][1])*np.linalg.norm(s['cell'][1])>.6:rows.append([])
        rows[-1].append((i,y))
    centers=[np.mean([v for _,v in row]) for row in rows];types=[Counter(atoms[i] for i,_ in row).most_common(1)[0][0] for row in rows]
    edges=[(centers[i]+((centers[(i+1)%len(rows)]-centers[i])%1)/2)%1 for i in range(len(rows)) if types[i]!=types[(i+1)%len(rows)]]
    a=s['cell'][0];b=s['cell'][1];perp=np.linalg.norm(b-b.dot(a)/a.dot(a)*a)
    ds=[min(abs((frac[i,1]-e)%1),1-abs((frac[i,1]-e)%1))*perp for i in ids for e in edges]
    return {'CH_distances_A':distances(h,c),'OH_distances_A':distances(h,o),'metal_contacts_under_3A':[[i,j,atoms[j-1],d] for i,j,d in contacts],
            'interface_distance_A':float(min(ds)) if ds else None,'interface_method':'minimum adsorbate distance to periodic top-row Cu/Au boundaries; 0.6 A row/layer tolerance'}


def scan(root):
    root=Path(root);files=[]
    for top in sorted(root.iterdir()):
        if not top.is_dir() or top.name in {'benchmark','CO2RR_ML','co2rr_digital_discovery'} or top.name.startswith(('PR18','.' )):continue
        for folder,dirs,names in os.walk(top):
            dirs[:]=[d for d in dirs if not d.startswith('.')]
            for name in names:
                if name in {'POSCAR','CONTCAR','INCAR','KPOINTS'} or name.endswith('_in.txt') or name.endswith('_conv.txt') or name.startswith('isaac_record') and name.endswith('.json'):
                    files.append(Path(folder)/name)
    return files


def run(root,repo,out):
    root=Path(root);repo=Path(repo);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=scan(root);inv=pd.read_csv(repo/'study/manuscript_package/figure_data/all_unique_structures.csv');split=pd.read_csv(repo/'study/manuscript_package/figure_data/cv_manifest.csv')
    alias_info={a:r._asdict() for r in inv.itertuples(index=False) for a in json.loads(r.aliases)}
    geometries={};errors=[]
    for f in files:
        if f.name not in {'POSCAR','CONTCAR'} or not f.stat().st_size:continue
        try:
            s=read_poscar(f);audit=audit_components(s);rel=f.relative_to(root).as_posix();alias=rel.split('/XAS/')[0] if '/XAS/' in rel else str(f.parent.relative_to(root))
            geometries[rel]={'structure':s,'audit':audit,'alias':alias,'path':rel,'sha256':digest(f),'parent_key':parent_key(s)}
        except (ValueError,IndexError,KeyError) as e:errors.append({'path':str(f.relative_to(root)),'error':str(e)})
    candidates={g['alias'] for g in geometries.values() if set(g['audit']['identities'])&{'CHO','COH'} or any(t in g['alias'] for t in ['CHO','COH'])}
    molecules=[];configs=[];links=[];assets=[]
    for alias in sorted(candidates):
        gs=[g for g in geometries.values() if g['alias']==alias];initial=geometries.get(alias+'/POSCAR');final=geometries.get(alias+'/CONTCAR')
        recs=[f for f in files if f.name.startswith('isaac_record') and str(f.relative_to(root)).startswith(alias+'/XAS/')]
        records=[]
        for f in recs:
            d=json.loads(f.read_text());comp=d.get('sample',{}).get('composition',{});records.append({'path':str(f.relative_to(root)),'sha256':digest(f),'labels':comp.get('adsorbate_counts',comp.get('adsorbate_identities'))})
        for g in gs:
            a=g['audit'];row={'configuration':alias,'stage':'XAS_used_copy' if '/XAS/' in g['path'] else 'initial' if g['path'].endswith('/POSCAR') else 'relaxed',
                              'path':g['path'],'sha256':g['sha256'],'counts':json.dumps(a['counts'],sort_keys=True),'status':a['status'],'cutoff_sensitivity':json.dumps(a['sensitivity_factors']),
                              'atom_counts':json.dumps(dict(Counter(g['structure']['atoms'])),sort_keys=True),'parent_key':g['parent_key']}
            for i,m in enumerate(a['components']):
                molecules.append({**row,'molecule':i,'species':m['species'],'atom_indices':json.dumps(m['atom_indices']),'bonds':json.dumps(m['bonds']),**{k:json.dumps(v) for k,v in local_info(g['structure'],m['atom_indices']).items()}})
            if '/XAS/' in g['path'] and final:links.append({'configuration':alias,'from':final['path'],'to':g['path'],**correspondence(final['structure'],g['structure'])})
            if row['stage'] in {'initial','relaxed'}:assets.append({'configuration':alias,'stage':row['stage'],'path':g['path'],'structure':{k:np.asarray(v).tolist() for k,v in g['structure'].items()},'audit':a})
        if initial and final:links.append({'configuration':alias,'from':initial['path'],'to':final['path'],**correspondence(initial['structure'],final['structure'])})
        # Match actual full-cell deck coordinates to copies; all layer decks are inspected.
        deck_results=[]
        for f in files:
            rel=str(f.relative_to(root))
            if not rel.startswith(alias+'/XAS/') or not f.name.endswith('_in.txt'):continue
            parsed=read_fdmnes_crystal(f)
            if parsed:
                s={'atoms':parsed['atoms'],'positions':parsed['fractional']@parsed['cell'],'cell':parsed['cell']};a=audit_components(s)
                deck_results.append({'path':rel,'sha256':digest(f),'counts':a['counts'],'status':a['status'],'final_correspondence':correspondence(final['structure'],s) if final else None})
        used=[g for g in gs if '/XAS/' in g['path'] and g['path'].endswith('/layer_5/CONTCAR')]
        target=split[split.configuration==alias];meta=alias_info.get(alias,{})
        observed=used[0]['audit'] if used else final['audit'] if final else initial['audit'] if initial else None
        configs.append({'configuration':alias,'retained':not target.empty,'ml_status':meta.get('ml_status','not present in frozen unique inventory'),
                        'intended_initial_counts':initial['audit']['counts'] if initial else None,'relaxed_counts':final['audit']['counts'] if final else None,
                        'XAS_used_counts':used[0]['audit']['counts'] if used else None,'frozen_target':target.iloc[0].labels if len(target) else None,
                        'observed_status':observed['status'] if observed else None,'frozen_labels_match_observed':set(target.iloc[0].labels.split(';'))==set(observed['identities']) if len(target) and observed else None,
                        'initial_path':initial['path'] if initial else None,'initial_sha256':initial['sha256'] if initial else None,
                        'relaxed_path':final['path'] if final else None,'relaxed_sha256':final['sha256'] if final else None,'initial_parent_key':initial['parent_key'] if initial else None,
                        'records':records,'decks':deck_results,'geometry_copies':[{k:v for k,v in g.items() if k not in {'structure','audit'}} for g in gs]})
    (out/'configuration_audit.json').write_text(json.dumps(configs,indent=2)+'\n');pd.DataFrame([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in c.items() if k not in {'decks','geometry_copies'}} for c in configs]).to_csv(out/'configuration_audit.csv',index=False)
    pd.DataFrame(molecules).to_csv(out/'molecule_audit.csv',index=False);(out/'atom_correspondence.json').write_text(json.dumps(links,indent=2)+'\n');(out/'geometry_views_data.json').write_text(json.dumps(assets,indent=2)+'\n')
    inventory=[{'path':str(f.relative_to(root)),'bytes':f.stat().st_size,'role':f.name} for f in files]
    (out/'artifact_search_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n');(out/'parse_failures.json').write_text(json.dumps(errors,indent=2)+'\n')
    # Summary counts are configuration counts; duplicates are identified by frozen inventory aliases.
    summary={'candidate_aliases':len(configs),'geometry_files_parsed':len(geometries),'artifact_paths_searched':len(files),'retained_candidates':sum(c['retained'] for c in configs),
             'demonstrated_retained_label_mismatches':[c['configuration'] for c in configs if c['retained'] and c['frozen_labels_match_observed'] is False],
             'ambiguous_retained':[c['configuration'] for c in configs if c['retained'] and c['observed_status']!='verified'],
             'retained_CH0_vs_COH_note':'CHO-only bins count configurations, not molecules or all CHO-containing mixtures; basename CHO is not chemical evidence for COH folders'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
    return geometries,files


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--repo-root',default=str(Path(__file__).resolve().parents[1]));p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.data_root,a.repo_root,a.output_dir)
