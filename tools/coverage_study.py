"""Versioned conversion, geometry audit and spectra-only CO coverage experiment.

No job submission or portal access. Source files and old records are preserved.
Cu100 input-deck evidence is a separate domain, not verified relaxed CONTCAR.
"""
from pathlib import Path
import os,sys,json,hashlib,re,subprocess,copy,tarfile,io,gzip
from collections import Counter
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from .component_audit import read_poscar,read_fdmnes_crystal,audit_components
from .chemical_lineage import correspondence,local_info
from .dataset_labels import geometry_hash
from .contact_checks import minimum_image_vectors
from .spectral_multilabel import spectrum_layers,record_identity,feature_bank,SPECIES

SEED=20261026

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
def folders(root):
    for top in sorted(Path(root).iterdir()):
        if not top.is_dir() or top.name.startswith(('PR18','.')) or top.name in {'CO2RR_ML','co2rr_digital_discovery','benchmark','tools'}:continue
        yield from sorted(top.glob('**/XAS/FDMNES/green'))

def native(folder):
    result={}
    for p in sorted(folder.glob('layer_*/Cu_layer*_conv.txt')):
        m=re.fullmatch(r'Cu_layer(\d+)_conv.txt',p.name)
        if not m:continue
        try:
            a=np.loadtxt(p,skiprows=1)
            if a.ndim!=2 or a.shape[1]!=2 or len(a)<50 or not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0):continue
            result[int(m[1])]=(p,a)
        except ValueError:continue
    return result

def convert(root,repo,out):
    root=Path(root).resolve();repo=Path(repo).resolve();out=Path(out).resolve();out.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ['PATH'],ISAAC_ABSORBER='Cu',ISAAC_EDGE='K');rows=[]
    for folder in folders(root):
        alias=str(folder.relative_to(root)).split('/XAS/')[0];fs=native(folder);dst=out/'records'/alias/'isaac_record_layers.json';dst.parent.mkdir(parents=True,exist_ok=True)
        old=folder/'isaac_record_layers.json';row={'configuration':alias,'layers':sorted(fs),'source_record':str(old.relative_to(root)) if old.exists() else None,'source_record_sha256':sha(old) if old.exists() else None}
        # Canonical four-layer route supplied by the author. Other complete layer
        # sets use generic conversion + explicit native-series merge, never fake layers.
        if not fs:row.update(conversion='no_usable_spectrum');rows.append(row);continue
        cmd=['bash',str(repo/'convert_layers_to_isaac.sh'),str(folder),str(dst)] if len(fs)==4 and not alias.startswith('Cu100/') else [sys.executable,str(repo/'tools/convert_simulation_xas_run_to_isaac.py'),'--input-dir',str(folder),'--output',str(dst)]
        run=subprocess.run(cmd,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        (dst.parent/'conversion.log').write_text(run.stdout)
        if run.returncode or not dst.exists():row.update(conversion='failed',error=run.stdout[-2000:]);rows.append(row);continue
        d=json.loads(dst.read_text());d['descriptors']={'outputs':[]} # generic single-series descriptors do not describe a merged record
        d['measurement']['series']=[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'name':'energy','unit':'eV','values':a[:,0].tolist()}],'channels':[{'name':'absorption_cross_section','unit':'arb','role':'simulated_observable','values':a[:,1].tolist()}]} for l,(p,a) in sorted(fs.items())]
        d.setdefault('links',[]).append({'rel':'source_export','target':alias+'/XAS/FDMNES/green'})
        d['conversion_provenance']={'version':'coverage-v1-20261026','route':'author_layer_shell' if cmd[0]=='bash' else 'generic_converter_explicit_available_layers','relaxed_structure_status':'not_reported' if alias.startswith('Cu100/') else 'requires_independent_audit','original_record_sha256':row['source_record_sha256'],'native_files':{str(l):{'path':str(p.relative_to(root)),'sha256':sha(p)} for l,(p,a) in fs.items()}}
        dump(dst,d);(dst.parent/'converter_original.json.gz').write_bytes(gzip.compress(dst.read_bytes(),mtime=0));row.update(conversion='converted',converted_record=str(dst.relative_to(out)),converted_record_sha256=sha(dst));rows.append(row)
    dump(out/'conversion_inventory.json',rows);return rows

def deck_structure(path):
    d=read_fdmnes_crystal(path)
    if d is None:raise ValueError('Missing complete Crystal geometry')
    return {'atoms':d['atoms'],'positions':d['fractional']@d['cell'],'cell':d['cell']}

def layer_evidence(folder,layer,s):
    """Check canonical aggregate covers its full physical Cu layer.

    Generalizes layer count and index ordering; does not change archived verifier.
    """
    from .layer_population import verify_layer_population
    if (folder/'CONTCAR').exists() and len(set(np.round(s['positions'][np.array(s['atoms'])=='Cu',2],0)))>=5:
        return verify_layer_population(folder,layer)
    table=pd.read_csv(folder/'absorbers.tsv',sep='\t');ids=table.atom_index.to_numpy(int)
    deck=folder/f'Cu_layer{layer}_in.txt';text=deck.read_text();selected=np.array(list(map(int,re.search(r'(?im)^Absorber\s*\n([^\n]+)',text)[1].split())))
    if len(set(ids))!=len(ids) or any(s['atoms'][i-1]!='Cu' for i in ids):raise ValueError('Invalid absorber table')
    normal=np.cross(s['cell'][0],s['cell'][1]);normal/=np.linalg.norm(normal);heights=s['positions']@normal;groups=[]
    for i in sorted([i for i,a in enumerate(s['atoms']) if a=='Cu'],key=lambda i:heights[i]):
        if not groups or heights[i]-heights[groups[-1][-1]]>.6:groups.append([])
        groups[-1].append(i)
    if layer>len(groups) or set(ids)!={i+1 for i in groups[layer-1]}:raise ValueError('Absorber population does not equal physical layer')
    if not set(selected)<=set(ids):raise ValueError('Deck absorber outside layer')
    # Verify final site sequence and source E_cut mapping (source convention copied
    # from existing audited verifier), or complete main run with explicit sites.
    logs=[]
    for p in sorted(folder.glob('fdmnes*.out')):
        pairs=re.findall(r'Gamma_hole\s*=.*?E_cut\s*=\s*([-\d.]+).*?site\s+(\d+)',p.read_text(errors='replace'))
        if [int(site) for cut,site in pairs]==list(range(1,len(ids)+1)):logs.append((p,pairs))
    if not logs:raise ValueError('No complete final convolution log')
    p,pairs=logs[-1]
    conv=folder/f'Cu_layer{layer}_convolution_in.txt'
    if not conv.exists():raise ValueError('Missing explicit final convolution source deck')
    text=conv.read_text();ordered=[int(v) for v in re.findall(r'(?m)^\s*Cu_layer'+str(layer)+r'_(\d+)\s*$',text)]
    weights=re.findall(r'(?m)^\s*1\.0\s+0\.0\s*$',text)
    if not ordered and re.search(r'(?im)^Calculation\s*\n\s*Cu_layer'+str(layer)+r'\s*\n',text) and set(selected)==set(ids):
        ordered=sorted(ids.tolist())
        population_basis='full aggregate; complete final site log and unique physical sites'
    elif len(ordered)==len(set(ordered)) and set(ordered)==set(ids) and len(weights)==len(ids):
        population_basis='explicit equal-weight site-source list in recorded convolution order'
    else:raise ValueError('Convolution does not document full-layer population')
    files=[folder/f'Cu_layer{layer}_{i}.txt' for i in ordered]
    if any(not f.exists() for f in files):raise ValueError('Missing raw absorber files')
    for f,(cut,site) in zip(files,pairs):
        header=list(map(float,f.read_text().splitlines()[0].split('=')[0].split()))
        if header[1]!=29 or abs(header[6]-float(cut))>.00051:raise ValueError('Wrong convolution source mapping')
    return {'layer':layer,'population':len(ids),'physical_layers':len(groups),'convolution_log':p.name,'convolution_log_sha256':sha(p),'population_basis':population_basis,'absorber_indices':ids.tolist(),'convolution_source_order':ordered,'convolution_deck_sha256':sha(conv)}

def audit(root,repo,out):
    root=Path(root).resolve();repo=Path(repo).resolve();out=Path(out).resolve();inventory=json.loads((out/'conversion_inventory.json').read_text());rows=[];molecules=[];samples=[];frozen=json.loads((repo/'study/retraining_20261022/historical_lineage.json').read_text())['records_by_legacy_geometry'];archive={};source=[]
    def capture(p):
        if not p.exists():return
        h=sha(p);archive[h]=p;source.append({'path':str(p.relative_to(root)),'sha256':h,'bytes':p.stat().st_size})
    for item in inventory:
        alias=item['configuration'];folder=root/alias/'XAS/FDMNES/green';fs=native(folder);row=dict(item);row.update(domain='Cu100' if alias.startswith('Cu100/') else 'CuAu',eligible=False);rows.append(row)
        if item['conversion']!='converted':continue
        record=out/item['converted_record'];d=json.loads(gzip.decompress((record.parent/'converter_original.json.gz').read_bytes()));row['converter_labels']=d.get('sample',{}).get('composition',{}).get('adsorbate_counts');old=folder/'isaac_record_layers.json'
        row['supplied_labels']=json.loads(old.read_text()).get('sample',{}).get('composition',{}).get('adsorbate_counts') if old.exists() else None
        try:
            structures={l:deck_structure(folder/f'layer_{l}/Cu_layer{l}_in.txt') for l in fs};s=structures[max(fs)]
            settings={}
            for l in fs:
                text=(folder/f'layer_{l}/Cu_layer{l}_in.txt').read_text()
                radius=re.search(r'(?im)^Radius\s*\n\s*([\d.]+)',text)
                settings[l]={'Green':bool(re.search(r'(?im)^Green\s*$',text)),'SCF':bool(re.search(r'(?im)^SCF\s*$',text)),'Perdew':bool(re.search(r'(?im)^Perdew\s*$',text)),'radius_A':float(radius[1]) if radius else None}
                if not all(settings[l][k] for k in ['Green','SCF','Perdew']) or settings[l]['radius_A']!=6.:raise ValueError('Numerical settings outside declared Green/SCF/Perdew/radius6 route')
            row['settings']=settings
            for l,b in structures.items():
                if correspondence(s,b)['status']!='matched':raise ValueError('Different layer calculation geometries')
            from .cu100_lineage import snapshot_graph
            graph=snapshot_graph(s) if row['domain']=='Cu100' else audit_components(s);final=root/alias/'CONTCAR';row.update(graph_status=graph['status'],observed_counts=graph['counts'],connectivity=graph['components'],cutoff_sensitivity=graph['sensitivity_factors'],CONTCAR_available=final.exists())
            if final.exists():
                link=correspondence(read_poscar(final),s);row['CONTCAR_mapping']=link
                if link['status']!='matched':
                    from .component_audit import legacy_benchmark_geometry_hash
                    copyfile=folder/f'layer_{max(fs)}/CONTCAR'
                    oldline=frozen.get(legacy_benchmark_geometry_hash(read_poscar(copyfile))) if copyfile.exists() and correspondence(read_poscar(copyfile),s)['status']=='matched' else None
                    if not oldline or sha(final)!=oldline['root_relaxed_sha256'] or not copyfile.exists() or sha(copyfile)!=oldline['XAS_used_geometry_sha256'] or any(sha(p)!=oldline['convolved_output_sha256'].get(str(l)) for l,(p,a) in fs.items()):raise ValueError('Unresolved root CONTCAR / calculation-input mismatch')
                    row['historical_geometry_discrepancy']='Retained exact archived geometry/output hashes; no new relaxed geometry claim'
                row['relaxed_counts']=audit_components(read_poscar(final))['counts']
            elif row['domain']!='Cu100':raise ValueError('Missing relaxed CONTCAR')
            else:
                from .cu100_lineage import verify
                row['ASE_lineage']=verify(root,alias,s)
                if row['ASE_lineage']['status']!='matched_original_ASE_snapshot':raise ValueError('Missing or ambiguous original Cu100 ASE snapshot')
                capture(root/row['ASE_lineage']['path'])
            row['label_matches_converter']=row['converter_labels']==graph['counts']
            row['label_matches_supplied']=row['supplied_labels']==graph['counts'] if old.exists() else None
            # Preserve conflicting source labels. Versioned metadata uses observed
            # graph only if stable; uncertain graphs do not enter model targets.
            d.setdefault('sample',{}).setdefault('structure_descriptors',{})['component_audit']=graph
            d['geometry_provenance']={'basis':'actual_FDMNES_Crystal','CONTCAR_verified':bool(final.exists() and row.get('CONTCAR_mapping',{}).get('status')=='matched'),'ASE_lineage':row.get('ASE_lineage'),'relaxation_status':'ASE snapshot matched; production termination not verified' if not final.exists() else 'saved_CONTCAR_with_explicit_lineage_audit','geometry_sha256':geometry_hash(s)}
            if graph['status']=='verified':d['sample'].setdefault('composition',{}).update(adsorbate_counts=graph['counts'],adsorbate_identities=graph['identities'],adsorbate_count_total=sum(graph['counts'].values()))
            dump(record,d);row['audited_record_sha256']=sha(record)
            for m in graph['components']:molecules.append({'configuration':alias,**m,**local_info(s,m['atom_indices'])})
            if graph['status']!='verified':raise ValueError('Unknown identity or cutoff-sensitive connectivity')
            required=[3,4] if row['domain']=='Cu100' else [4,5]
            if not set(required)<=set(fs):raise ValueError('Missing top-two physical layers')
            evid={l:layer_evidence(folder/f'layer_{l}',l,structures[l]) for l in required};row['layer_evidence']=evid
            for l in required:
                if not np.array_equal(fs[l][1][:,0],fs[required[0]][1][:,0]):raise ValueError('Different top-two grids')
            energy=fs[required[0]][1][:,0];raw=sum(fs[l][1][:,1]*evid[l]['population'] for l in required)/sum(evid[l]['population'] for l in required)
            atoms=np.array(s['atoms']);normal=np.cross(s['cell'][0],s['cell'][1]);normal/=np.linalg.norm(normal);h=s['positions']@normal;metal=np.flatnonzero(np.isin(atoms,['Cu','Au']));top=[]
            for element in ['Cu','Au']:
                ids=sorted(np.flatnonzero(atoms==element),key=lambda i:h[i]);groups=[]
                for i in ids:
                    if not groups or h[i]-h[groups[-1][-1]]>.6:groups.append([])
                    groups[-1].append(i)
                if groups:top.extend(groups[-1])
            top=np.array(top,int);area=float(np.linalg.norm(np.cross(s['cell'][0],s['cell'][1])));nco=graph['counts'].get('CO',0)
            row.update(eligible=True,group=geometry_hash(s),n_CO=nco,surface_sites=len(top),surface_Cu=int(sum(atoms[top]=='Cu')),surface_Au=int(sum(atoms[top]=='Au')),coverage_ML=nco/len(top),coverage_CO_per_nm2=nco/(area/100),cell_area_A2=area,CO_only=graph['identities']==['CO'],identity_labels=graph['identities'])
            # All top surface metal atoms define 1 ML across domains. Also expose
            # Cu-only denominator to distinguish legacy Cu36 loading labels.
            row['coverage_per_surface_Cu']=nco/row['surface_Cu'] if row['surface_Cu'] else None
            row['initial_parent_available']=(root/alias/'POSCAR').exists()
            d['geometry_provenance']['top_two_layers']=required
            d['geometry_provenance']['top_two_populations']={str(l):evid[l]['population'] for l in required}
            dump(record,d);row['audited_record_sha256']=sha(record)
            samples.append((row.copy(),raw,energy,s))
            for l in required:
                for p in [fs[l][0],folder/f'layer_{l}/Cu_layer{l}_in.txt',folder/f'layer_{l}/absorbers.tsv',folder/f'layer_{l}'/evid[l]['convolution_log'] if evid[l].get('convolution_log') else folder/f'layer_{l}/fdmnes.out']:capture(p)
            capture(final)
        except (ValueError,KeyError,IndexError,FileNotFoundError,TypeError) as e:row.update(exclusion=str(e))
    dump(out/'record_geometry_audit.json',rows);dump(out/'molecule_audit.json',molecules);dump(out/'source_manifest.json',source)
    with tarfile.open(out/'source_evidence.tar','w') as tar:
        for h,p in sorted(archive.items()):
            data=p.read_bytes();info=tarfile.TarInfo('objects/'+h);info.size=len(data);info.mtime=0;tar.addfile(info,io.BytesIO(data))
    # Keep exact geometry/arrays portable without duplicating every raw site file.
    dump(out/'configuration_manifest.json',[a for a,x,e,s in samples]);dump(out/'geometry_arrays.json',[{'configuration':a['configuration'],**{k:np.asarray(v).tolist() for k,v in s.items()}} for a,x,e,s in samples])
    np.savez_compressed(out/'spectra.npz',**{f'raw_{i}':x for i,(a,x,e,s) in enumerate(samples)},**{f'energy_{i}':e for i,(a,x,e,s) in enumerate(samples)})
    pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,(dict,list))} for x in rows]).to_csv(out/'record_geometry_audit.csv',index=False)
    return rows

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('action',choices=['convert','audit']);p.add_argument('--data-root',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    result=globals()[a.action](a.data_root,repo,a.output_dir);print(Counter(x.get('conversion') if a.action=='convert' else ('eligible_'+x['domain'] if x['eligible'] else x.get('exclusion',x.get('conversion'))) for x in result))
