"""Versioned current-export record audit and physical-source reconstruction.

Never modifies supplied records, electronic spectra or live inputs. No scheduler.
"""
from pathlib import Path
import json,hashlib,copy,re
import numpy as np,pandas as pd
from .coverage_study import sha,dump,deck_structure,native
from .component_audit import read_poscar
from .cu100_lineage import snapshot_graph
from .chemical_lineage import correspondence,local_info
from .dataset_labels import geometry_hash
from .fdmnes_scale import convolve,headers
from .spectral_multilabel import spectrum_layers

def fix_records(root,out):
 root=Path(root);out=Path(out);audit=json.load(open(out/'record_geometry_audit.json'));changes=[]
 for r in audit:
  if r['conversion']!='converted':continue
  alias=r['configuration'];f=root/alias/'XAS/FDMNES/green';record=out/r['converted_record'];d=json.load(open(record));fs=native(f)
  try:s=deck_structure(f/f'layer_{max(fs)}/Cu_layer{max(fs)}_in.txt')
  except (ValueError,FileNotFoundError):continue
  graph=snapshot_graph(s);final=root/alias/'CONTCAR';match=correspondence(read_poscar(final),s) if final.exists() else r.get('ASE_lineage',{}).get('atom_mapping',{});supplied=f/'isaac_record_layers.json';before=json.load(open(supplied)) if supplied.exists() else None
  comparisons=[]
  if before:
   try:
    curves={int(v['series_id'].rsplit('_',1)[1]):np.column_stack([v['independent_variables'][0]['values'],v['channels'][0]['values']]) for v in before['measurement']['series'] if v['series_id'].startswith('Cu_K_layer_')}
    for l,(p,a) in fs.items():comparisons.append({'layer':l,'exact_array_match':l in curves and np.array_equal(curves[l],a)})
   except Exception as e:comparisons.append({'error':str(e)})
  contacts=[{**m,**local_info(s,m['atom_indices'])} for m in graph['components']]
  d['sample']['material'].update(formula=''.join(e+(str(s['atoms'].count(e)) if s['atoms'].count(e)>1 else '') for e in ['Cu','Au','C','O','H'] if e in s['atoms']), notes='Actual XAS-used geometry; exact CONTCAR/ASE correspondence reported separately. A CONTCAR alone does not establish ionic convergence.');d['record_id']='co2rr-20261029-'+geometry_hash(s)[:24];d['sample']['geometry']={'lattice_vectors_A':np.asarray(s['cell']).tolist(),'structure_source':'actual FDMNES Crystal; CONTCAR/ASE correspondence reported separately','structure_sha256':geometry_hash(s),'component_audit':graph,'observed_components_and_contacts':contacts,'vacuum_periodic':False,'contact_definition':'minimum-image in-plane metal distances listed below3A; no pure-Cu atop/bridge relabeled Cu3 hollow'}
  d['sample']['composition'].update(atom_counts={e:s['atoms'].count(e) for e in sorted(set(s['atoms']))},total_atoms=len(s['atoms']),adsorbate_counts=graph['counts'],adsorbate_identities=graph['identities'],adsorbate_count_total=sum(graph['counts'].values()))
  d.setdefault('geometry_provenance',{}).update(current_export='CO2RR_1029',source_record_sha256=sha(supplied) if supplied.exists() else None,CONTCAR_or_ASE_mapping=match,label_status=graph['status'],eligible_for_model=r['eligible'],exclusion=r.get('exclusion'),relaxation_termination='not independently established by a CONTCAR file alone')
  d['conversion_provenance']['version']='dataset-refresh-20261029';dump(record,d)
  changes.append({'configuration':alias,'supplied_record_sha256':sha(supplied) if supplied.exists() else None,'versioned_record_sha256':sha(record),'labels_before':before.get('sample',{}).get('composition',{}).get('adsorbate_counts') if before else None,'labels_after':graph['counts'],'label_graph_status':graph['status'],'source_geometry_match':match.get('status'),'supplied_arrays':comparisons,'metadata_action':'replaced site/distribution-derived geometric claims with exact XAS-used graph/cell/contacts; preserved source bytes','training_eligible':r['eligible'],'exclusion':r.get('exclusion')})
 dump(out/'record_corrections.json',changes);pd.DataFrame([{**{k:v for k,v in x.items() if not isinstance(v,(list,dict))},'labels_before':json.dumps(x['labels_before'],sort_keys=True),'labels_after':json.dumps(x['labels_after'],sort_keys=True),'supplied_arrays':json.dumps(x['supplied_arrays'])} for x in changes]).to_csv(out/'record_corrections.csv',index=False);return changes

def symmetry_rule(s,ids,hs):
 """Inverse multiplicity for complete explicit sites; native weights for proven symmetry reps."""
 physical=set(map(int,ids));sources={h['atom_index'] for h in hs}
 if len(sources)!=len(hs) or any(not np.isfinite(h['natomsym_f']) or h['natomsym_f']<=0 for h in hs):raise ValueError('Duplicate source or invalid multiplicity')
 if physical==sources:return 'all_physical_sites', {h['atom_index']:1/h['natomsym_f'] for h in hs},None
 if not sources<physical or abs(sum(h['natomsym_f'] for h in hs)-len(physical))>1e-8:raise ValueError('Incomplete/unexplained physical source coverage')
 # Supported exact symmetry-reduced clean Cu(100) model: reflection x<->y.
 swap=np.eye(3)[[1,0,2]];metric=np.asarray(s['cell'])@np.asarray(s['cell']).T
 if not np.allclose(swap@metric@swap.T,metric,atol=2e-5,rtol=0):raise ValueError('Representative symmetry is not an isometry of the actual cell')
 f=np.asarray(s['positions'])@np.linalg.inv(s['cell']);t=dict(s);t['positions']=f[:,[1,0,2]]@s['cell'];link=correspondence(s,t)
 if link['status']!='matched' or link['max_distance_A']>2e-5:raise ValueError('Symmetry representatives lack exact geometry mapping')
 mapping=dict(link['one_based_mapping']);orbits={h['atom_index']:{h['atom_index'],mapping[h['atom_index']]} for h in hs}
 if set.union(*orbits.values())!=physical or sum(len(v) for v in orbits.values())!=len(physical) or any(len(orbits[h['atom_index']])!=h['natomsym_f'] for h in hs):raise ValueError('Symmetry orbit coverage/multiplicities disagree')
 return 'nonoverlapping_symmetry_representatives',{h['atom_index']:1. for h in hs},{'operation':'fractional x/y reflection; z unchanged','mapping':link,'orbits':{str(k):sorted(v) for k,v in orbits.items()}}

def scale(root,out,binary):
 root=Path(root);out=Path(out);rows=json.load(open(out/'configuration_manifest.json'));receipts=[];records=[];arrays=[];native_arrays=[];E=None
 # Recover missing final clean Cu100 convolution from actual saved electronic sources.
 alias='Cu100/Cu100_2x2x4_clean';f=root/alias/'XAS/FDMNES/green';clean={};cs=deck_structure(f/'layer_4/Cu_layer4_in.txt');clink=correspondence(read_poscar(f/'layer_4/CONTCAR'),cs)
 if clink['status']!='matched' or snapshot_graph(cs)['counts']:raise ValueError('Invalid clean Cu100 geometry')
 for l in [3,4]:
  ss=deck_structure(f/f'layer_{l}/Cu_layer{l}_in.txt');assert correspondence(cs,ss)['status']=='matched';tab=pd.read_csv(f/f'layer_{l}/absorbers.tsv',sep='\t');hs=headers(f/f'layer_{l}',l);rule,weights,evidence=symmetry_rule(ss,tab.atom_index,hs);a,_=convolve(binary,f/f'layer_{l}',l,out/f'scale/reconvolution/{alias}/layer_{l}','unit_weight_control');n=len(tab);clean[l]=a[:,1]/n;E=a[:,0];receipts.append({'configuration':alias,'layer':l,'source_rule':rule,'physical_Cu_sites':n,'source_multiplicity_sum':sum(h['natomsym_f'] for h in hs),'weights':weights,'symmetry_evidence':evidence,'status':'derived final convolution from complete saved electronic source arrays; no new electronic calculation','source_headers':hs,'native_reproduced':None})
 base=copy.deepcopy(json.load(open(out/'records/Cu100/Cu100_2x2x4_CO_1_4_top/isaac_record_layers.json')));base['record_id']='co2rr-20261029-clean-Cu100';base['sample']['material'].update(name='clean Cu100',formula='Cu16',identifiers=[{'scheme':'internal','value':alias}],notes='Actual clean Cu100 XAS input and matching layer CONTCAR; final convolution locally derived from supplied raw electronic spectra.');base['links']=[{'rel':'source_export','target':alias+'/XAS/FDMNES/green'}];base['assets']=[{'asset_id':'clean_'+str(j),'content_role':'input_structure' if p.name=='CONTCAR' else 'raw_data_pointer','uri':str(p.relative_to(root/alias/'XAS/FDMNES/green')),'sha256':sha(p)} for j,p in enumerate(sorted((root/alias/'XAS/FDMNES/green').glob('layer_*/*'))) if p.is_file() and (p.name in ['CONTCAR','absorbers.tsv'] or p.name.endswith('_in.txt') or re.fullmatch(r'Cu_layer\d+_\d+\.txt',p.name))];base['system']['configuration']={'absorber':'Cu','edge':'K','generator_name':'dataset_refresh_v1','source_folder':alias+'/XAS/FDMNES/green','core_hole_treatment':'not_reported'};base['sample']['composition'].update(atom_counts={'Cu':16},total_atoms=16,adsorbate_counts={},adsorbate_identities=[],adsorbate_count_total=0);base['sample']['geometry']={'lattice_vectors_A':np.asarray(cs['cell']).tolist(),'component_audit':snapshot_graph(cs),'structure_source':'matching layer CONTCAR and actual clean Cu100 XAS input'};base['geometry_provenance']={'CONTCAR_verified':True,'CONTCAR_mapping':clink,'geometry_sha256':geometry_hash(cs),'relaxation_status':'layer CONTCAR matches XAS geometry; separate relaxation termination not supplied','convolution_status':'locally derived from supplied complete raw site spectra','not_new_HPC':True};base['measurement']['series']=[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'name':'energy','unit':'eV','values':E.tolist()}],'channels':[{'name':'absorption_cross_section','unit':'Mbarn_per_Cu','values':clean[l].tolist()}]} for l in [3,4]]
 base['source_scale_correction']={'physical_domain':'Cu100','physical_top_layers':[3,4],'populations':{3:4,4:4},'isotropic_cross_section_unit':'Mbarn_per_Cu','native_reconstruction_verified':True,'source_rule':'nonoverlapping symmetry representatives cover all physical sites; do not divide each representative by multiplicity'};dump(out/'scale/corrected_records'/alias/'isaac_record_layers.json',base);dump(out/'records'/alias/'isaac_record_layers.json',base)
 cleanrow={'configuration':alias,'domain':'Cu100','group':geometry_hash(cs),'identity_labels':[],'CO_only':False,'n_CO':0,'surface_sites':4,'surface_Cu':4,'surface_Au':0,'coverage_ML':0.,'coverage_per_surface_Cu':0.,'coverage_CO_per_nm2':0.,'cell_area_A2':float(np.linalg.norm(np.cross(cs['cell'][0],cs['cell'][1]))),'layer_evidence':{str(l):{'population':4} for l in [3,4]},'derived_clean_record':True,'corrected_record':str(Path('corrected_records')/alias/'isaac_record_layers.json')};records.append(cleanrow);arrays.append((clean[3]+clean[4])/2);native_arrays.append(arrays[-1])
 for row in rows:
  alias=row['configuration'];required=[3,4] if row['domain']=='Cu100' else [4,5];f=root/alias/'XAS/FDMNES/green';d=json.load(open(out/'records'/alias/'isaac_record_layers.json'));physical={};native_per={}
  for l in required:
   src=f/f'layer_{l}';ss=deck_structure(src/f'Cu_layer{l}_in.txt');tab=pd.read_csv(src/'absorbers.tsv',sep='\t');hs=headers(src,l);rule,weights,ev=symmetry_rule(ss,tab.atom_index,hs);where=out/f'scale/reconvolution/{alias}/layer_{l}';a,_=convolve(binary,src,l,where,'native');saved=np.loadtxt(src/f'Cu_layer{l}_conv.txt',skiprows=1);rel=float(np.max(abs(saved[:,1]-a[:,1]))/np.max(abs(saved[:,1])));assert np.array_equal(saved[:,0],a[:,0]) and rel<2e-7,(alias,l,rel)
   unit,_=convolve(binary,src,l,where,'unit_weight_control');assert np.max(abs(unit[:,1]-a[:,1]))/np.max(abs(a[:,1]))<2e-7,'Native broadening differs from declared explicit prescription'
   corrected,_=convolve(binary,src,l,where,'physical') if rule=='all_physical_sites' else (unit,hs)
   n=len(tab);physical[l]=corrected[:,1]/n;native_per[l]=saved[:,1]/n;assert np.array_equal(E,a[:,0]);receipts.append({'configuration':alias,'layer':l,'native_reproduced':True,'native_relative_error':rel,'physical_Cu_sites':n,'source_rule':rule,'weights':weights,'source_multiplicity_sum':sum(h['natomsym_f'] for h in hs),'source_headers':hs,'symmetry_evidence':ev})
  pop={l:row['layer_evidence'][str(l)]['population'] for l in required};avg=lambda data:sum(data[l]*pop[l] for l in required)/sum(pop.values());arrays.append(avg(physical));native_arrays.append(avg(native_per));d['measurement']['series']=[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'name':'energy','unit':'eV','values':E.tolist()}],'channels':[{'name':'absorption_cross_section','unit':'Mbarn_per_Cu','values':physical[l].tolist()}]} for l in required];d['source_scale_correction']={'physical_domain':row['domain'],'physical_top_layers':required,'populations':pop,'isotropic_cross_section_unit':'Mbarn_per_Cu','native_reconstruction_verified':True,'source_rule':'per-source inverse convolution multiplicity only when every physical site already explicitly listed','binary_sha256':sha(binary),'core_hole':'not_reported; no electronic default inference'};dest=Path('corrected_records')/alias/'isaac_record_layers.json';dump(out/'scale'/dest,d);records.append({**row,'corrected_record':str(dest)})
 dump(out/'scale/configuration_manifest.json',records);dump(out/'scale/source_weight_audit.json',receipts);np.savez_compressed(out/'scale/per_Cu_spectra.npz',energy=E,physical=np.array(arrays),native_per_unique_Cu=np.array(native_arrays),configuration=np.array([r['configuration'] for r in records]));dump(out/'scale/software.json',{'binary_sha256':sha(binary),'actual_banner':(out/'clean_Cu100_reconvolution/layer_4/native_log.txt').read_text().splitlines()[0] if (out/'clean_Cu100_reconvolution/layer_4/native_log.txt').exists() else 'see generated convolution logs','convolution_only':True,'no_new_electronic_calculation':True,'note':'convolution multiplies by header natomsym_f; raw source file itself is not assumed symmetry-summed'});finalize_metadata(out,root)
 return len(records)

def finalize_metadata(out,root):
 """Complete formula/source metadata without changing measurement or graph arrays."""
 out=Path(out);root=Path(root);alias='Cu100/Cu100_2x2x4_clean';source=root/alias/'XAS/FDMNES/green'
 for collection in [out/'records',out/'scale/corrected_records']:
  for p in collection.rglob('isaac_record_layers.json'):
   d=json.load(open(p));counts=d['sample']['composition']['atom_counts'];d['sample']['material'].update(formula=''.join(e+(str(counts[e]) if counts[e]>1 else '') for e in ['Cu','Au','C','O','H'] if counts.get(e)),notes='Actual XAS-used geometry; exact CONTCAR/ASE correspondence and convergence limitations are reported separately.')
   if d['record_id']=='co2rr-20261029-clean-Cu100':
    d['sample']['material'].update(name='clean Cu100',formula='Cu16',identifiers=[{'scheme':'internal','value':alias}]);d['links']=[{'rel':'source_export','target':alias+'/XAS/FDMNES/green'}];d['assets']=[{'asset_id':'clean_'+str(j),'content_role':'input_structure' if q.name=='CONTCAR' else 'raw_data_pointer','uri':str(q.relative_to(source)),'sha256':sha(q)} for j,q in enumerate(sorted(source.glob('layer_*/*'))) if q.is_file() and (q.name in ['CONTCAR','absorbers.tsv'] or q.name.endswith('_in.txt') or re.fullmatch(r'Cu_layer\d+_\d+\.txt',q.name))];d['system']['configuration']={'absorber':'Cu','edge':'K','generator_name':'dataset_refresh_v1','source_folder':alias+'/XAS/FDMNES/green','core_hole_treatment':'not_reported'}
   dump(p,d)
 changes=json.load(open(out/'record_corrections.json'))
 for r in changes:r['versioned_record_sha256']=sha(out/'records'/r['configuration']/'isaac_record_layers.json')
 dump(out/'record_corrections.json',changes);pd.DataFrame([{**{k:v for k,v in x.items() if not isinstance(v,(list,dict))},'labels_before':json.dumps(x['labels_before'],sort_keys=True),'labels_after':json.dumps(x['labels_after'],sort_keys=True),'supplied_arrays':json.dumps(x['supplied_arrays'])} for x in changes]).to_csv(out/'record_corrections.csv',index=False)
 audit=json.load(open(out/'record_geometry_audit.json'));pd.DataFrame([{k:v for k,v in x.items() if not isinstance(v,(list,dict))} for x in audit]).to_csv(out/'record_geometry_audit.csv',index=False)
