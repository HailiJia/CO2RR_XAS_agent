"""Evidence-based recovery of quartet, hydrogenation, and clean controls."""
from pathlib import Path
from collections import Counter
import hashlib,json,re,shutil
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .chemical_lineage import scan,parent_key,correspondence,local_info,digest
from .component_audit import read_poscar,audit_components
from .dataset_labels import geometry_hash
from .contact_checks import minimum_image_vectors
from .spectral_multilabel import spectrum_layers
from .layer_population import verify_layer_population

QUARTET=[('2CO','interface','2CO_adjacent/2CO_adjacent_0p056_interface'),('2CO','terrace','2CO_adjacent/2CO_adjacent_0p056_interior'),('OCCO','interface','OCCO/OCCO_0p056_interface'),('OCCO','terrace','OCCO/OCCO_0p056_interior')]


def equivalent_candidates(candidates):
    """Duplicate aliases of identical observed geometry are one outcome."""
    unique={}
    for c in candidates:unique.setdefault(c['geometry_hash'],[]).append(c)
    return list(unique.values())


def inplane_rms(a,b):
    if Counter(a['atoms'])!=Counter(b['atoms']):return None
    ds=[];normal=np.cross(a['cell'][0],a['cell'][1]);normal/=np.linalg.norm(normal)
    for el in ['C','O']:
        ia=np.flatnonzero(np.asarray(a['atoms'])==el);ib=np.flatnonzero(np.asarray(b['atoms'])==el)
        delta=minimum_image_vectors(a['positions'][ia,None]-b['positions'][ib][None,:],a['cell'],(True,True,False));delta-=np.einsum('ijk,k->ij',delta,normal)[:,:,None]*normal
        cost=np.linalg.norm(delta,axis=-1);ii,jj=linear_sum_assignment(cost);ds.extend(cost[ii,jj])
    return float(np.sqrt(np.mean(np.square(ds)))) if ds else None


def deck_settings(folder):
    keys=[]
    for l in [2,3,4,5]:
        p=folder/f'layer_{l}/Cu_layer{l}_in.txt'
        if not p.exists():return None
        lines=p.read_text().splitlines();before=[];skip=False
        for line in lines:
            t=line.strip()
            if t.lower()=='crystal':break
            if skip:skip=False;continue
            if t.lower()=='filout':skip=True;continue
            if t and not t.startswith('!'):before.append(t.lower())
        keys.append(before)
    return keys


def recover_quartet(root,repo,files=None):
    root=Path(root);repo=Path(repo);files=files or scan(root);initials={};finals=[]
    for f in files:
        if f.name not in {'POSCAR','CONTCAR'} or not f.stat().st_size:continue
        try:s=read_poscar(f)
        except (ValueError,IndexError):continue
        rel=str(f.relative_to(root));alias=rel.split('/XAS/')[0] if '/XAS/' in rel else str(f.parent.relative_to(root))
        if f.name=='POSCAR' and '/XAS/' not in rel:initials[alias]=(s,rel)
        if f.name=='CONTCAR':finals.append({'path':rel,'alias':alias,'geometry_hash':geometry_hash(s),'sha256':digest(f),'observed_counts':audit_components(s)['counts']})
    rows=[]
    for state,region,alias in QUARTET:
        planned=repo/'data/CuAu_dataset'/alias/'POSCAR';p=read_poscar(planned);exact=[a for a,(s,path) in initials.items() if geometry_hash(s)==geometry_hash(p)]
        possible=set(exact)|{alias};found=[f for f in finals if f['alias'] in possible];outcomes=equivalent_candidates(found)
        recs=[f for f in files if f.name.startswith('isaac_record') and any(str(f.relative_to(root)).startswith(a+'/XAS/') for a in possible)]
        compatible_initial=alias in exact
        actual=initials.get(alias);comparison=correspondence(p,actual[0]) if actual else None
        selected=outcomes[0] if len(outcomes)==1 else None;counts=selected[0]['observed_counts'] if selected else None
        valid_identity=bool(counts=={'CO':2} if state=='2CO' else counts=={'OCCO':1}) if counts else False
        if len(outcomes)>1:status='ambiguous_lineage_multiple_non_equivalent_final_geometries'
        elif found and state=='OCCO' and not valid_identity:status='dissociated_or_changed_connectivity'
        elif recs and not compatible_initial:status='completed_but_incompatible_with_pinned_initial_version'
        elif recs and valid_identity:status='completed_candidate_requires_full_pair_settings_verification'
        elif found:status='relaxed_available_spectral_output_not_located'
        elif actual and (root/alias/'OUTCAR').exists():status='unfinished_or_failed_attempt; no valid relaxed output located'
        else:status='not_located_by_expanded_search; never_run_not_established'
        missing=[]
        if not found:missing.append('valid relaxed geometry linked to pinned initial input')
        if not recs:missing.append('verified Cu K-edge output and absorber population; AI-ready record optional')
        if recs and not compatible_initial:missing.append('lineage proving or regenerating the pinned initial version; existing completed record is recoverable but not the same input')
        rows.append({'requested_state':state,'region':region,'pinned_input':str(planned.relative_to(repo)),'pinned_input_sha256':digest(planned),'alias':alias,'exact_initial_aliases':exact,
            'local_initial':actual[1] if actual else None,'local_initial_sha256':digest(root/actual[1]) if actual else None,'pinned_vs_local_initial':comparison,
            'relaxed_available':bool(found),'relaxed_unique_geometries':len(outcomes),'relaxed_candidates':found,'observed_counts':counts,'intact_requested_identity':valid_identity,
            'records':[{'path':str(f.relative_to(root)),'sha256':digest(f)} for f in recs],'status':status,'missing_artifacts':missing,
            'additional_C_O_edges':'extensions, not prerequisite for a valid first Cu K-edge pair'})
    available={r['region']:[] for r in rows}
    for r in rows:
        if r['status']=='completed_candidate_requires_full_pair_settings_verification':available[r['region']].append(r['requested_state'])
    status='Cu coupling comparison requires verified matched intact OCCO and 2CO records; no matched pair recovered' if not any(set(v)=={'2CO','OCCO'} for v in available.values()) else 'region-level completed candidates recovered; validate pair settings before spectral subtraction'
    return {'structures':rows,'contrast_status':status,'search_artifact_paths':len(files),'no_new_jobs_submitted':True}


def run(root,repo,out):
    root=Path(root);repo=Path(repo);out=Path(out);out.mkdir(parents=True,exist_ok=True);files=scan(root)
    quartet=recover_quartet(root,repo,files);(out/'matched_quartet_v2.json').write_text(json.dumps(quartet,indent=2)+'\n')
    pd.DataFrame([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in row.items()} for row in quartet['structures']]).to_csv(out/'quartet_status.csv',index=False)
    inventory=[];eligible=[]
    for f in files:
        if f.name!='isaac_record_layers.json':continue
        alias=str(f.relative_to(root)).split('/XAS/')[0];folder=f.parent;entry={'configuration':alias,'record':str(f.relative_to(root)),'record_sha256':digest(f)}
        try:
            d=json.loads(f.read_text());layers=spectrum_layers(d);used=folder/'layer_5/CONTCAR';s=read_poscar(used);a=audit_components(s);initial=read_poscar(root/alias/'POSCAR')
            pops={l:verify_layer_population(folder/f'layer_{l}',l)['population'] for l in [2,3,4,5]}
            for l in [2,3,4,5]:
                saved=np.loadtxt(folder/f'layer_{l}/Cu_layer{l}_conv.txt',skiprows=1)
                if not np.array_equal(saved[:,0],layers[l][:,0]) or not np.allclose(saved[:,1],layers[l][:,1],rtol=1e-7,atol=1e-9):raise ValueError('record/output mismatch')
            entry.update(status='completed_validated',observed_counts=a['counts'],component_status=a['status'],XAS_geometry=str(used.relative_to(root)),XAS_geometry_sha256=digest(used),parent_key=parent_key(initial),
                         atom_counts=dict(Counter(s['atoms'])),populations=pops,numerical_settings=deck_settings(folder),local_environment=local_info(s,np.flatnonzero(np.isin(s['atoms'],['C','O','H']))+1),
                         initial_sha256=digest(root/alias/'POSCAR'),relaxed_sha256=digest(root/alias/'CONTCAR') if (root/alias/'CONTCAR').exists() else None)
            eligible.append({'entry':entry,'structure':s,'initial':initial,'layers':layers})
        except (ValueError,IndexError,KeyError,FileNotFoundError) as exc:entry.update(status='completed_but_incompatible_or_incomplete',reason=str(exc))
        inventory.append(entry)
    (out/'completed_record_recovery.json').write_text(json.dumps(inventory,indent=2)+'\n')
    pairs=[];valid=[]
    for i,a in enumerate(eligible):
        ca=a['entry']['observed_counts'];formula=a['entry']['atom_counts'];n=formula.get('C',0)
        if not (ca=={'CO':n,'H':n} and n>0):continue
        for b in eligible:
            cb=b['entry']['observed_counts']
            if cb not in [{'CHO':n},{'COH':n}] or formula!=b['entry']['atom_counts']:continue
            failures=[]
            if a['entry']['parent_key']!=b['entry']['parent_key']:failures.append('different initial substrate coordinate parent')
            if a['entry']['numerical_settings']!=b['entry']['numerical_settings']:failures.append('different recorded XAS controls')
            if a['entry']['populations']!=b['entry']['populations']:failures.append('different absorber populations')
            rms=inplane_rms(a['initial'],b['initial'])
            if rms is None or rms>.75:failures.append('initial carbon/oxygen adsorption pattern differs by >0.75 A in plane')
            ad=a['entry']['local_environment']['interface_distance_A'];bd=b['entry']['local_environment']['interface_distance_A']
            if ad is None or bd is None:failures.append('interface location not independently resolved')
            elif abs(ad-bd)>.5:failures.append('relaxed interface-distance difference >0.5 A')
            if a['entry']['component_status']!='verified' or b['entry']['component_status']!='verified':failures.append('ambiguous connectivity')
            pair={'uncoupled':a['entry']['configuration'],'hydrogenated':b['entry']['configuration'],'atom_counts':formula,'same_initial_parent':a['entry']['parent_key']==b['entry']['parent_key'],
                  'initial_CO_pattern_inplane_RMS_A':rms,'interface_distances_A':[ad,bd],'accepted':not failures,'rejection_reasons':failures}
            pairs.append(pair)
            if not failures:
                valid.append(pair);x=a['layers'][5][:,0]
                if not np.array_equal(x,b['layers'][5][:,0]):pair['accepted']=False;pair['rejection_reasons']=['different grids'];continue
                # Both outputs are raw per-Cu site averages from validated equal populations.
                pd.DataFrame({'energy_eV':x,'CO_plus_H':a['layers'][5][:,1],'hydrogenated':b['layers'][5][:,1],
                              'difference':b['layers'][5][:,1]-a['layers'][5][:,1]}).to_csv(out/f'controlled_hydrogenation_{len(valid)}.csv',index=False)
    pd.DataFrame([{k:json.dumps(v) if isinstance(v,(list,dict)) else v for k,v in p.items()} for p in pairs]).to_csv(out/'hydrogenation_candidate_pairs.csv',index=False)
    # Clean records and raw outputs are searched separately: lack of a record is not evidence of absent spectra.
    clean=[]
    for alias in ['plain','plain/2fixed']:
        p=root/alias/'CONTCAR';s=read_poscar(p);key=parent_key(s)
        outputs=[f for f in files if str(f.relative_to(root)).startswith(alias+'/') and f.name.endswith('_conv.txt')]
        clean.append({'configuration':alias,'relaxed_path':str(p.relative_to(root)),'sha256':digest(p),'observed_counts':audit_components(s)['counts'],'parent_key':key,
                      'raw_spectral_outputs':[str(f.relative_to(root)) for f in outputs],'matching_adsorbed_initial_parent_count':sum(e['entry']['parent_key']==key for e in eligible),
                      'status':'raw outputs need reconciliation' if outputs else 'relaxed clean structure exists; no numeric clean XAS output located in expanded search','missing_artifacts':['Cu K-edge layer/site spectra, actual XAS deck, absorber population and convolution lineage'] if not outputs else []})
    # Search ALL saved geometry copies for any unadsorbed output, not just plain aliases.
    clean_elsewhere=[]
    for f in files:
        if f.name=='CONTCAR' and '/XAS/' in str(f):
            try:
                s=read_poscar(f)
                if not set(s['atoms'])&{'C','O','H'}:clean_elsewhere.append(str(f.relative_to(root)))
            except (ValueError,IndexError):pass
    (out/'clean_parent_search.json').write_text(json.dumps({'clean_controls':clean,'other_clean_XAS_geometry_copies':clean_elsewhere,'adsorbate_minus_clean_pairs':0 if not clean_elsewhere and not any(c['raw_spectral_outputs'] for c in clean) else 'requires validation','species_absent_mean_is_not_clean':True},indent=2)+'\n')
    prepared=out/'prepared_controls';prepared.mkdir(exist_ok=True)
    for state,region,alias in QUARTET:
        src=repo/'data/CuAu_dataset'/alias;dest=prepared/(state+'_'+region);dest.mkdir(exist_ok=True)
        for name in ['POSCAR','INCAR','KPOINTS','submit_relax.sh']:
            if (src/name).exists():shutil.copyfile(src/name,dest/name)
    # Ready-to-run clean XAS inputs use actual separately relaxed clean structures.
    template=next((e for e in eligible if e['entry']['configuration'].startswith('CO/')),eligible[0])
    for cleanrow in clean:
        src=root/cleanrow['configuration']/'CONTCAR';s=read_poscar(src);dest=prepared/cleanrow['configuration'].replace('/','_');dest.mkdir(exist_ok=True);shutil.copyfile(src,dest/'relaxed_clean_POSCAR')
        cell=s['cell'];length=np.linalg.norm(cell,axis=1);angles=[np.rad2deg(np.arccos(np.clip(cell[a].dot(cell[b])/length[a]/length[b],-1,1))) for a,b in [(1,2),(0,2),(0,1)]]
        frac=s['positions']@np.linalg.inv(cell);height=s['positions']@np.cross(cell[0],cell[1])/np.linalg.norm(np.cross(cell[0],cell[1]));heights=[]
        for h in sorted(height):
            if not heights or h-heights[-1][-1]>.6:heights.append([])
            heights[-1].append(h)
        for layer in [2,3,4,5]:
            copper=np.flatnonzero((np.asarray(s['atoms'])=='Cu')&(abs(height-np.mean(heights[layer-1]))<.6))+1
            deck='Filout\n Cu_layer'+str(layer)+'\nRange\n -5. 0.2 50.\nRadius\n 6.0\nEnergpho\nSCF\nGreen\nPerdew\nCrystal\n '+' '.join(map(str,[*length,*angles]))+'\n'
            deck+='\n'.join(f'{29 if el=="Cu" else 79} '+' '.join(f'{v:.10f}' for v in f) for el,f in zip(s['atoms'],frac))+'\nAbsorber\n '+' '.join(map(str,copper))+'\nConvolution\nEnd\n'
            (dest/f'Cu_layer{layer}_in.txt').write_text(deck)
    (prepared/'README.md').write_text('Prepared inputs only. No submissions. Supply licensed potentials locally for relaxation. Clean decks use separately relaxed plain and plain/2fixed structures, not frozen-metal removals. Verify the matching adsorbed parent, fixed-layer convention, 6 Angstrom controls, absorber populations and executable/core-hole/convolution prescription before running; missing native convolution logs cannot be inferred. C/O edges are extensions, not prerequisites for a valid Cu pair.\n')
    summary={'validated_completed_records':len(eligible),'record_attempts':len(inventory),'hydrogenation_candidates':len(pairs),'valid_hydrogenation_pairs':sum(p['accepted'] for p in pairs),'matched_Cu_coupling_pairs':0,'clean_pairs':0 if not clean_elsewhere and not any(c['raw_spectral_outputs'] for c in clean) else None,'criteria':'same initial metal parent, elemental loading, numerical controls, absorber populations; <=0.75 A initial in-plane C/O pattern RMS and <=0.5 A interface-distance mismatch; no naming-based identity','no_new_HPC_jobs':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));return summary


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.data_root,Path(__file__).resolve().parents[1],a.output_dir)
