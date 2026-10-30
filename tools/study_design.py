"""Reconcile planned controls and the matched input quartet, without new jobs."""
from collections import Counter
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .component_audit import read_poscar,audit_components
from .contact_checks import minimum_image_vectors
from .study_cohort import sha256
from .dataset_labels import geometry_hash

QUARTET=[('2CO','interface','2CO_adjacent/2CO_adjacent_0p056_interface'),('2CO','terrace','2CO_adjacent/2CO_adjacent_0p056_interior'),('OCCO','interface','OCCO/OCCO_0p056_interface'),('OCCO','terrace','OCCO/OCCO_0p056_interior')]

def _cyclic_motifs(row_elements):
    """Count overlapping three-row stripe windows on the periodic surface."""
    result=Counter();n=len(row_elements)
    if n<3:return result
    names={(3,0):'Cu3',(2,1):'Cu2Au',(1,2):'CuAu2',(0,3):'Au3'}
    for i in range(n):
        window=[row_elements[(i+j)%n] for j in range(3)]
        if any(x not in {'Cu','Au'} for x in window):continue
        result[names[(window.count('Cu'),window.count('Au'))]]+=1
    return result

def _circular_distance(a,b):
    d=abs((float(a)-float(b))%1.0)
    return min(d,1.0-d)

def reconcile_design(repo_root,data_root,analysis_dir,output_dir):
    repo=Path(repo_root);root=Path(data_root);base=Path(analysis_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    inventory=pd.read_csv(base/'all_unique_structures.csv');support=pd.read_csv(base/'species_support.csv')
    priorities=[]
    for target,reason,rank in [('clean','No eligible clean controls; needed to measure false positives against no adsorbate',1),('CO2','Three positives; reserve new independent groups for both training and confirmation',2),('COCHO','One positive; a held-out fold has zero training positives',2),('OCCOH','Four positives; sparse independent confirmation',2),('CO+H / CO / CO+OH','Matched parent/site/loading comparisons isolate coadsorbate omissions',3),('intact OCCO / two CO','No eligible intact OCCO positives; verify final connectivity and matched parents',3)]:
        n=int(support.set_index('species').loc[target,'support']) if target in support.species.values else 0 if target in ['clean','intact OCCO / two CO'] else None
        priorities.append(dict(priority=rank,target=target,current_independent_positive_geometries=n,requested_new_count=None,rationale=reason,selection_rule='Independent final geometry, not extra layers/sites; reserve confirmation before inspection'))
    pd.DataFrame(priorities).to_csv(out/'missing_calculation_priorities.csv',index=False)
    # Explicit minimal row counts define the ratios, not evidence that any
    # requested matrix cell has been generated or relaxed.
    planned=[]
    for parent,cu,au in [('pure Cu(111)',3,0),('pure Au(111)',0,3),('Cu:Au 1:1',1,1),('Cu:Au 2:1',2,1),('Cu:Au 1:2',1,2),('Cu:Au 3:1',3,1),('Cu:Au 1:3',1,3)]:
        for region in (['terrace'] if not cu or not au else ['interface','near-interface','terrace']):
            planned.append(dict(parent=parent,explicit_Cu_rows=cu,explicit_Au_rows=au,total_lateral_rows=cu+au,region=region,coverage='must specify numerator/denominator',distribution='planned stratification',species_scope='clean, C1, C2, coadsorbates',source_match='unverified; folder names alone cannot establish row ratio',completion='not asserted'))
    pd.DataFrame(planned).to_csv(out/'planned_structural_matrix.csv',index=False)
    actual=[];initial_lookup={}
    for row in inventory.itertuples():
        aliases=json.loads(row.aliases)
        for alias in aliases:
            initial=root/alias/'POSCAR'
            if initial.exists() and initial.stat().st_size:
                try:initial_lookup.setdefault(geometry_hash(read_poscar(initial)),[]).append(alias)
                except (ValueError,IndexError):pass
        initial=next((root/a/'POSCAR' for a in aliases if (root/a/'POSCAR').exists() and (root/a/'POSCAR').stat().st_size),None)
        if initial is None:continue
        try:
            st=read_poscar(initial);aa=np.array(st['atoms']);pp=st['positions'];normal=np.cross(st['cell'][0],st['cell'][1]);normal/=np.linalg.norm(normal);hh=pp@normal;mm=np.isin(aa,['Cu','Au']);gg=[]
            for i in sorted(np.flatnonzero(mm),key=lambda i:hh[i]):
                if not gg or hh[i]-hh[gg[-1][-1]]>.6:gg.append([])
                gg[-1].append(int(i))
            top_ids=gg[-1];frac=pp@np.linalg.inv(st['cell']);length=np.linalg.norm(st['cell'][1]);counts={}
            for element in ['Cu','Au']:
                values=sorted(float(frac[i,1]%1)*length for i in top_ids if aa[i]==element);rows=[]
                for v in values:
                    if not rows or v-rows[-1][-1]>.6:rows.append([])
                    rows[-1].append(v)
                counts[element]=len(rows)
            ordered=sorted(top_ids,key=lambda i:float(frac[i,1]%1.0));groups=[]
            for i in ordered:
                y=float(frac[i,1]%1.0)*length
                if not groups or y-groups[-1][-1][1]>.6:groups.append([])
                groups[-1].append((int(i),y))
            row_elements=[Counter(aa[i] for i,_ in g if aa[i] in {'Cu','Au'}).most_common(1)[0][0] for g in groups]
            row_centers=[float(np.mean([frac[i,1]%1.0 for i,_ in g]))%1.0 for g in groups]
            motifs=_cyclic_motifs(row_elements)
            nonmetal=np.flatnonzero(~mm);ads_y=float(np.mean(frac[nonmetal,1])%1.0) if len(nonmetal) else None
            local_motif='none';interface_distance=None
            if ads_y is not None and len(row_centers)>=3:
                near=int(np.argmin([_circular_distance(ads_y,y) for y in row_centers]))
                local=[row_elements[(near+j-1)%len(row_elements)] for j in range(3)]
                local_motif={(3,0):'Cu3',(2,1):'Cu2Au',(1,2):'CuAu2',(0,3):'Au3'}.get((local.count('Cu'),local.count('Au')),'mixed/unknown')
                a=np.asarray(st['cell'][0],float);b=np.asarray(st['cell'][1],float);bperp=np.linalg.norm(b-np.dot(b,a/np.linalg.norm(a))*a/np.linalg.norm(a))
                edges=[(row_centers[k]+((row_centers[(k+1)%len(row_centers)]-row_centers[k])%1.0)/2.0)%1.0 for k in range(len(row_centers)) if row_elements[k]!=row_elements[(k+1)%len(row_centers)]]
                if edges:interface_distance=min(_circular_distance(ads_y,e) for e in edges)*bperp
            actual.append(dict(configuration=row.configuration,Cu_rows=counts['Cu'],Au_rows=counts['Au'],top_row_sequence=';'.join(row_elements),Cu3=motifs['Cu3'],Cu2Au=motifs['Cu2Au'],CuAu2=motifs['CuAu2'],Au3=motifs['Au3'],adsorbate_local_motif=local_motif,nearest_interface_A=interface_distance,row_method='initial top-layer fractional-y clusters; 0.6 A row gap; cyclic 3-row windows; inferred from coordinates',ml_status=row.ml_status))
        except (ValueError,IndexError):continue
    pd.DataFrame(actual).to_csv(out/'actual_structural_matrix.csv',index=False)
    for plan in planned:
        cu,au=plan['explicit_Cu_rows'],plan['explicit_Au_rows']
        candidates=[a for a in actual if (not au and a['Au_rows']==0 and a['Cu_rows']>0) or (not cu and a['Cu_rows']==0 and a['Au_rows']>0) or (cu and au and a['Cu_rows']*au==a['Au_rows']*cu and a['Cu_rows']>0 and a['Au_rows']>0)]
        plan['matching_ratio_unique_sources']=len(candidates)
        motif=plan['parent'].split()[-1]
        plan['matching_local_motif_unique_sources']=sum(a['adsorbate_local_motif']==motif for a in actual) if motif in {'Cu3','Cu2Au','CuAu2','Au3'} else None
        plan['source_match']='ratio observed; requested region still unverified' if candidates else 'requested ratio absent from audited source inventory'
    pd.DataFrame(planned).to_csv(out/'planned_structural_matrix.csv',index=False)
    quartet=[];jobs=[];parents=[]
    for state,region,relative in QUARTET:
        p=repo/'data/CuAu_dataset'/relative/'POSCAR';structure=read_poscar(p);atoms=np.array(structure['atoms']);pos=structure['positions'];cell=structure['cell'];ids=np.flatnonzero(atoms=='C');delta=minimum_image_vectors(pos[ids[1]]-pos[ids[0]],cell)
        normal=np.cross(cell[0],cell[1]);normal/=np.linalg.norm(normal);height=pos@normal;metals=np.isin(atoms,['Cu','Au']);groups=[]
        for i in sorted(np.flatnonzero(metals),key=lambda i:height[i]):
            if not groups or height[i]-height[groups[-1][-1]]>.6:groups.append([])
            groups[-1].append(int(i))
        if len(groups)!=5:raise ValueError('Cannot resolve the five physical metal layers')
        top=np.zeros(len(atoms),bool);top[groups[-1]]=True;cu_top=int(np.sum(top&(atoms=='Cu')));all_top=int(top.sum())
        parent=sorted((str(a),*np.round(v,7)) for a,v in zip(atoms[metals],pos[metals]));parents.append((cell,parent))
        loading=dict(Counter(atoms[~metals]));assert loading=={'C':2,'O':2}
        matches=initial_lookup.get(geometry_hash(structure),[])
        observed_candidates=[]
        for match in matches:
            relaxed=root/match/'CONTCAR'
            if relaxed.exists() and relaxed.stat().st_size:observed_candidates.append(dict(configuration=match,final_sha256=sha256(relaxed),audit=audit_components(read_poscar(relaxed))))
        observed=observed_candidates[0]['audit'] if len(observed_candidates)==1 else None
        settings={name:sha256(p.parent/name) for name in ['INCAR','KPOINTS'] if (p.parent/name).exists()}
        row=dict(state_requested=state,region=region,input=str(p.relative_to(repo)),input_sha256=sha256(p),minimum_CC_A=float(np.linalg.norm(delta)),loading=loading,Cu_top_sites=cu_top,all_top_sites=all_top,C_equivalent_coverage_Cu=2/cu_top,molecular_coverage_Cu=(2 if state=='2CO' else 1)/cu_top,coverage_note='C-equivalent 2/Cu-top-sites; molecular coverage differs for one OCCO and two CO',settings_hashes=settings,initial_geometry_aliases=matches,relaxed_candidates=observed_candidates,relaxed_available=observed is not None,final_component_audit=observed)
        quartet.append(row)
        if observed is None or observed['counts'].get('OCCO',0)==0 and state=='OCCO':jobs.append(dict(input=str(p.relative_to(repo)),state_requested=state,region=region,action='prepare relaxation or review dissociation; do not force coupled labels',submission_authorized=False,requested_edges=['Cu K','C K','O K'],spectral_contrasts_status='unmeasured',backend_support='verify generated inputs and parsers before submitting'))
    samecell=all(np.allclose(parents[0][0],x[0]) for x in parents);sameparent=all(parents[0][1]==x[1] for x in parents)
    # Expanded evidence recovery supersedes the exact-initial-only lookup.
    from .controlled_comparisons import recover_quartet
    recovered=recover_quartet(root,repo)
    for row,evidence in zip(quartet,recovered['structures']):
        row.update(evidence)
    (out/'matched_quartet.json').write_text(json.dumps(dict(shared_cell=samecell,shared_metal_parent=sameparent,structures=quartet,contrast_status=recovered['contrast_status'],search_artifact_paths=recovered['search_artifact_paths']),indent=2)+'\n')
    (out/'missing_job_manifest.json').write_text(json.dumps(dict(jobs=jobs,submitted=False),indent=2)+'\n')
    # Export actual record geometry/count support without training new tasks.
    geometry=[]
    for row in pd.read_csv(base/'cv_manifest.csv').itertuples():
        d=json.loads((root/row.record).read_text());c=d['sample']['composition'];sites=d['sample'].get('geometry',{}).get('adsorption_site_counts',{})
        geometry.append(dict(configuration=row.configuration,fold=row.fold,counts=json.dumps(c.get('adsorbate_counts',{}),sort_keys=True),site_counts=json.dumps(sites,sort_keys=True),geometry_verified_for_prediction=False,performance='unmeasured'))
    pd.DataFrame(geometry).to_csv(out/'geometry_coverage_support_audit.csv',index=False)
    return dict(inventory_unique=len(inventory),matched_quartet=quartet,shared_cell=samecell,shared_parent=sameparent)
