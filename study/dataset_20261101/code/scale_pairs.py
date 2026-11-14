"""Reconstruct new sources; pin audited unchanged corrections/reference by exact bytes."""
from pathlib import Path
import json,copy,hashlib,shutil
import numpy as np,pandas as pd
from tools.coverage_study import dump,sha,native,deck_structure
from tools.dataset_refresh import symmetry_rule
from tools.fdmnes_scale import headers,convolve
from tools.chemical_lineage import correspondence
from tools.differential_xas import pair
from tools.spectral_multilabel import SPECIES
R=Path(__file__).resolve().parents[3];B=R/'study/dataset_20261101';P=R/'study/dataset_20261029';raw=B/'local_export';binary='/tmp/fdmnes_mac_official/fdmnes_Mac/fdmnes_mac';old=json.load(open(P/'training_manifest.json'));osm=json.load(open(P/'scale/configuration_manifest.json'));oldby={x['group']:x for x in osm};oz=np.load(P/'training_arrays.npz');E=oz['energy'];rows=json.load(open(B/'configuration_manifest.json'));out=[];proof=[];exclude=[];clean={}
# Frozen, independently audited clean Cu100 reference: exact geometry and complete representative source bytes.
for dom,alias in [('CuAu','plain'),('Cu100','Cu100/Cu100_2x2x4_clean')]:
 ref=next(x for x in osm if x['configuration']==alias);d=json.load(open(P/'scale'/ref['corrected_record']));layer=5 if dom=='CuAu' else 4;s=deck_structure(raw/alias/f'XAS/FDMNES/green/layer_{layer}/Cu_layer{layer}_in.txt');assert correspondence(s,deck_structure(raw/alias/f'XAS/FDMNES/green/layer_{layer-1}/Cu_layer{layer-1}_in.txt'))['status']=='matched';assert np.allclose(np.asarray(d['sample']['geometry']['lattice_vectors_A']),s['cell']);h=json.load(open(P/'scale/source_weight_audit.json'));evidence=[]
 for entry in [x for x in h if x['configuration']==alias]:
  for src in entry['source_headers']:
   p=raw/alias/f"XAS/FDMNES/green/layer_{entry['layer']}"/src['file'];assert sha(p)==src['sha256'],str(p);evidence.append({'path':str(p.relative_to(raw)),'sha256':sha(p)})
 if dom=='Cu100':
  clean[dom]=(ref,d);dump(B/'scale'/ref['corrected_record'],d);proof.append({'configuration':alias,'action':'pinned previous verified clean reference; complete representative source bytes match current export','source_record_sha256':sha(P/'scale'/ref['corrected_record']),'complete_sources':evidence,'current_incomplete_redundant_source':'layer3/Cu_layer3_11.txt ends9002.4eV; not extrapolated or included; current4-source native clean is not asserted usable'})
for row in rows:
 alias=row['configuration'];ls=[3,4] if row['domain']=='Cu100' else [4,5];f=raw/alias/'XAS/FDMNES/green';d=json.load(open(B/'records'/alias/'isaac_record_layers.json'));fs=native(f);previous=oldby.get(row['group']);reuse=False
 if previous and previous.get('corrected_record'):
  p=P/'records'/previous['configuration']/'isaac_record_layers.json'
  if p.exists():
   q=json.load(open(p));curves={int(v['series_id'].rsplit('_',1)[1]):np.column_stack([v['independent_variables'][0]['values'],v['channels'][0]['values']]) for v in q['measurement']['series']};reuse=all(l in curves and np.array_equal(fs[l][1],curves[l]) for l in ls)
 try:
  if reuse:
   q=json.load(open(P/'scale'/previous['corrected_record']));d['measurement']=copy.deepcopy(q['measurement']);d['source_scale_correction']=copy.deepcopy(q['source_scale_correction']);proof.append({'configuration':alias,'action':'reused frozen audited correction','previous_configuration':previous['configuration'],'geometry_hash_match':True,'all_required_native_arrays_exact':True,'previous_corrected_record_sha256':sha(P/'scale'/previous['corrected_record'])})
  else:
   physical={};receipts=[]
   for l in ls:
    src=f/f'layer_{l}';s=deck_structure(src/f'Cu_layer{l}_in.txt');tab=pd.read_csv(src/'absorbers.tsv',sep='\t');hs=headers(src,l);rule,w,ev=symmetry_rule(s,tab.atom_index,hs)
    for h in hs:
     a=np.loadtxt(src/h['file'],skiprows=2)
     if not np.array_equal(a[:,0],E):raise ValueError('Incomplete/incompatible raw absorber grid: '+h['file'])
    where=B/f'scale/reconvolution/{alias}/layer_{l}';a,_=convolve(binary,src,l,where,'native');assert np.array_equal(a[:,0],E);err=np.max(abs(a[:,1]-fs[l][1][:,1]))/max(abs(a[:,1]));assert err<2e-7
    unit,_=convolve(binary,src,l,where,'unit_weight_control');assert np.max(abs(unit[:,1]-a[:,1]))/max(abs(a[:,1]))<2e-7
    c,_=convolve(binary,src,l,where,'physical') if rule=='all_physical_sites' else (unit,hs);physical[l]=c[:,1]/len(tab);receipts.append({'layer':l,'rule':rule,'weights':w,'headers':hs,'physical_population':len(tab),'native_error':float(err),'symmetry_evidence':ev})
   d['measurement']['series']=[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'name':'energy','unit':'eV','values':E.tolist()}],'channels':[{'name':'absorption_cross_section','unit':'Mbarn_per_Cu','values':physical[l].tolist()}]} for l in ls];d['source_scale_correction']={'physical_domain':row['domain'],'physical_top_layers':ls,'populations':{int(l):row['layer_evidence'][str(l)]['population'] for l in ls},'isotropic_cross_section_unit':'Mbarn_per_Cu','native_reconstruction_verified':True,'binary_sha256':sha(binary)};proof.append({'configuration':alias,'action':'new local convolution of saved electronic spectra only','layers':receipts})
  d['record_id']=d['record_id'].replace('20261029','20261101');dest=Path('corrected_records')/alias/'isaac_record_layers.json';dump(B/'scale'/dest,d);out.append({**row,'corrected_record':str(dest)})
  if alias=='plain':clean['CuAu']=(out[-1],d)
 except (ValueError,AssertionError,FileNotFoundError) as e:exclude.append({'configuration':alias,'reason':str(e)});print('SCALE EXCLUDE',alias,e,flush=True)
print('SCALED',len(out),'exclude',exclude,flush=True);dump(B/'scale/configuration_manifest.json',out);dump(B/'scale/source_weight_audit.json',proof);dump(B/'scale/exclusions.json',exclude)
oldgrp={x['group']:(i,x) for i,x in enumerate(old)};oldalias={x['configuration']:(i,x) for i,x in enumerate(old)};tr=[];arr={k:[] for k in ['raw_physical','clean','delta','raw_native']};comparison=[];unsupported=[]
for row in out:
 if set(row['identity_labels'])-set(SPECIES):unsupported.append({'configuration':row['configuration'],'labels':row['identity_labels']});continue
 ref,rd=clean[row['domain']];d=json.load(open(B/'scale'/row['corrected_record']));ls=[3,4] if row['domain']=='Cu100' else [4,5];pop={l:row['layer_evidence'][str(l)]['population'] for l in ls};rp={l:ref['layer_evidence'][str(l)]['population'] for l in ls};p=pair(d,rd,E,pop,rp,ls,row['domain'],row['domain']);prior=oldgrp.get(row['group']) or oldalias.get(row['configuration']);entry={**row,'old_index':prior[0] if prior else None,'old99':prior[1]['old99'] if prior else False,'fold':prior[1]['fold'] if prior else int(hashlib.sha256(row['group'].encode()).hexdigest()[:8],16)%5,'reference_id':rd['record_id'],'reference_record_sha256':sha(B/'scale'/ref['corrected_record']),'sample_record_sha256':sha(B/'scale'/row['corrected_record']),'clean_scale':p['scale']};tr.append(entry)
 for k,v in [('raw_physical',p['raw_common_scale']),('clean',p['clean_common_scale']),('delta',p['delta'])]:arr[k].append(v)
 fs=native(raw/row['configuration']/'XAS/FDMNES/green');arr['raw_native'].append(sum(fs[l][1][:,1]*pop[l] for l in ls)/sum(pop.values()))
 if prior:
  j,pr=prior;comparison.append({'configuration':row['configuration'],'previous_configuration':pr['configuration'],'old_index':j,'geometry_identical':row['group']==pr['group'],'labels_identical':row['identity_labels']==pr['identity_labels'],'delta_bitwise_equal':np.array_equal(p['delta'],oz['delta'][j]),'delta_max_abs_difference':float(max(abs(p['delta']-oz['delta'][j])))})
dump(B/'training_manifest.json',tr);dump(B/'training_exclusions.json',unsupported);dump(B/'previous_input_comparison.json',comparison);np.savez_compressed(B/'training_arrays.npz',energy=E,**{k:np.asarray(v) for k,v in arr.items()});print('TRAINING',len(tr),'old',len(comparison),'new',len(tr)-len(comparison),'identical deltas',sum(x['delta_bitwise_equal'] for x in comparison),flush=True)
