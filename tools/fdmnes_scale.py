"""Verify native convolution and recover equal-physical-Cu source weights.

Only local convolution of SAVED electronic spectra is executed. No SCF, Green,
FDM, scheduler, portal, licensed potential or source geometry is changed.
"""
from pathlib import Path
import re,json,hashlib,shutil,subprocess,tempfile
import numpy as np
import pandas as pd
from .progress_snapshots import write_json

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def headers(folder,layer):
 files=sorted((p for p in folder.glob(f'Cu_layer{layer}_*.txt') if re.fullmatch(r'Cu_layer'+str(layer)+r'_\d+\.txt',p.name)),key=lambda p:int(p.stem.rsplit('_',1)[1]));result=[]
 for p in files:
  values=list(map(float,p.read_text().splitlines()[0].split('=')[0].split()));result.append({'file':p.name,'atom_index':int(p.stem.rsplit('_',1)[1]),'natomsym_f':values[13],'E_cut_eV':values[6],'Epsii_eV':values[9],'sha256':digest(p)})
 return result

def convolve(binary,source,layer,out,mode='native'):
 """Use an isolated copy; metadata and saved raw arrays remain unchanged."""
 source=Path(source);out=Path(out);out.mkdir(parents=True,exist_ok=True);hs=headers(source,layer);deck=source/f'Cu_layer{layer}_convolution_in.txt'
 if mode=='native':text=deck.read_text()
 elif mode in ['physical','unit_weight_control']:
  text='Calculation\n'
  for h in hs:
   weight=1/h['natomsym_f'] if mode=='physical' else 1.
   text+=f"  {Path(h['file']).stem}\n  {weight:.16g} 0.0\n"
  text+=f'\nConv_out\n  Cu_layer{layer}_conv\n\nConvolution\n\nGamma_hole\n  1.55\nGamma_max\n  10.0\nEcent\n  30.0\nElarg\n  30.0\n\nEnd\n'
 else:raise ValueError(mode)
 with tempfile.TemporaryDirectory(prefix='fdmnes_convolution_') as td:
  work=Path(td)
  for h in hs:shutil.copyfile(source/h['file'],work/h['file'])
  (work/'reconstruct_in.txt').write_text(text);(work/'fdmfile.txt').write_text('1\nreconstruct_in.txt\n')
  run=subprocess.run([str(Path(binary).resolve())],cwd=work,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=120)
  (out/f'{mode}_log.txt').write_text(run.stdout);(out/f'{mode}_in.txt').write_text(text)
  p=work/f'Cu_layer{layer}_conv.txt'
  if run.returncode or not p.exists():raise ValueError('Convolution failed: '+run.stdout[-1000:])
  a=np.loadtxt(p,skiprows=1)
  if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all():raise ValueError('Invalid reconstructed curve')
  shutil.copyfile(p,out/f'{mode}_aggregate.txt')
 return a,hs

def run(raw,repo,out,binary):
 raw=Path(raw);repo=Path(repo);out=Path(out);out.mkdir(parents=True,exist_ok=True);rows=json.loads((repo/'study/coverage_20261026/configuration_manifest.json').read_text());audits=[];records=[];native=[];physical=[];energy=None
 for row in rows:
  alias=row['configuration'];required=[4,5] if row['domain']=='CuAu' else [3,4];src=raw/alias/'XAS/FDMNES/green';record=json.loads((repo/'study/coverage_20261026/records'/alias/'isaac_record_layers.json').read_text());per={};old={};valid=True
  for layer in required:
   folder=src/f'layer_{layer}';where=out/'reconvolution'/alias/f'layer_{layer}';saved=np.loadtxt(folder/f'Cu_layer{layer}_conv.txt',skiprows=1);a,hs=convolve(binary,folder,layer,where,'native');N=row['layer_evidence'][str(layer)]['population']
   if len(hs)!=N or {x['atom_index'] for x in hs}!=set(pd.read_csv(folder/'absorbers.tsv',sep='\t').atom_index):raise ValueError('Raw sources do not cover unique physical sites: '+alias)
   err=float(np.max(abs(saved[:,1]-a[:,1])));rel=err/max(abs(saved[:,1]));match=bool(np.array_equal(saved[:,0],a[:,0]) and rel<2e-7);multiple=any(abs(x['natomsym_f']-1)>1e-8 for x in hs)
   unit,hs=convolve(binary,folder,layer,where,'unit_weight_control') if multiple else (a,hs)
   explicit_err=float(np.max(abs(unit[:,1]-a[:,1]))/max(abs(a[:,1])))
   corrected,hs=convolve(binary,folder,layer,where,'physical') if multiple else (a,hs)
   correct=corrected[:,1]/N
   meta={'configuration':alias,'domain':row['domain'],'layer':layer,'physical_Cu_sites':N,'source_multiplicity_sum':sum(h['natomsym_f'] for h in hs),'source_headers':hs,'nonunit_multiplicity':multiple,'native_reproduced':match,'native_max_abs_error':err,'native_relative_max_error':float(rel),'explicit_unit_weight_relative_error':explicit_err,'native_engine_header':(where/'native_log.txt').read_text().splitlines()[0] if (where/'native_log.txt').read_text().splitlines() else None,'source_original_convolution_deck_sha256':digest(folder/f'Cu_layer{layer}_convolution_in.txt'),'native_output_sha256':digest(folder/f'Cu_layer{layer}_conv.txt'),'per_Cu_basis':'sum over every physical site of its raw symmetry-summed spectrum weighted by 1/natomsym_f, then divide by physical N','correction':'individual source weights, no guessed aggregate amplitude divisor','intensity_unit':'Mbarn / physical Cu for isotropic <xanes>','all_sources_isotropic_two_column':all(np.loadtxt(folder/h['file'],skiprows=2).shape[1]==2 for h in hs)}
   if not match or explicit_err>2e-7:valid=False
   pd.DataFrame({'energy_eV':saved[:,0],'native_aggregate':saved[:,1],'native_aggregate_per_unique_Cu_without_symmetry_correction':saved[:,1]/N,'corrected_per_Cu':correct}).to_csv(where/'scale_comparison.csv',index=False)
   per[layer]=correct;old[layer]=saved[:,1]/N;audits.append(meta)
   for s in record['measurement']['series']:
    if s['series_id']==f'Cu_K_layer_{layer}':s['channels'][0]['values']=correct.tolist();s['channels'][0]['unit']='Mbarn_per_Cu'
  if energy is None:energy=saved[:,0]
  if not np.array_equal(energy,saved[:,0]):raise ValueError('Cohort grids differ; no extrapolation')
  pop={l:row['layer_evidence'][str(l)]['population'] for l in required};average=sum(per[l]*pop[l] for l in required)/sum(pop.values());oldaverage=sum(old[l]*pop[l] for l in required)/sum(pop.values());physical.append(average);native.append(oldaverage);record['measurement']['series']=[s for s in record['measurement']['series'] if int(s['series_id'].rsplit('_',1)[1]) in required];record['source_scale_correction']={'version':'differential-v1','physical_domain':row['domain'],'slab_Cu_layers':5 if row['domain']=='CuAu' else 4,'scope':'top-two only; original ISAAC record preserved in archived coverage study','isotropic_cross_section_unit':'Mbarn_per_Cu','native_reconstruction_verified':valid,'raw_source_scale':'symmetry-summed source; per-physical-site contribution = source/natomsym_f','physical_top_layers':required,'populations':pop,'reference_free_chemistry':'record contains absorption, not a fabricated clean-subtracted measurement'};write_json(out/'corrected_records'/alias/'isaac_record_layers.json',record);records.append({**row,'scale_reconstruction_verified':valid,'corrected_record':str(Path('corrected_records')/alias/'isaac_record_layers.json')})
 write_json(out/'source_weight_audit.json',audits);write_json(out/'configuration_manifest.json',records);np.savez_compressed(out/'per_Cu_spectra.npz',energy=energy,physical=np.array(physical),native_per_unique_Cu=np.array(native),configuration=np.array([r['configuration'] for r in records]));pd.DataFrame([{k:v for k,v in a.items() if not isinstance(v,(list,dict))} for a in audits]).to_csv(out/'source_weight_audit.csv',index=False)
 write_json(out/'software_provenance.json',{'download_page':'https://fdmnes.neel.cnrs.fr/download/','page_release_label':'2026-07-28','actual_package_and_executable':'2026-10-22','official_Mac_archive_url':'https://hub.neel.cnrs.fr/index.php/s/9F57Nao8iNwtFyR/download','binary_sha256':digest(binary),'binary_not_uploaded':True,'source_zip_url':'https://hub.neel.cnrs.fr/index.php/s/bwjwrY2fprcyjaX/download','source_zip_sha256':digest('/tmp/fdmnes_source_official.zip'),'convolution_source_sha256':digest('/tmp/fdmnes_source_official/fdmnes_fortran_prog/convolution.f90'),'manual_sha256':digest('/tmp/fdmnes_mac_official/fdmnes_Mac/Doc/Manual_Eng.pdf'),'native_outputs_version':'28 July 2026','version_caveat':'The current downloaded source/binary is October, not falsely identified as July. Native convolution is checked numerically for each included layer; defaults are not asserted version-specific without matching source. Recorded widths are explicitly stated in corrected decks.'})
 return {'configurations':len(records),'layers':len(audits),'nonunit_layers':sum(x['nonunit_multiplicity'] for x in audits),'native_reproductions':sum(x['native_reproduced'] for x in audits)}
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--binary',required=True);a=p.parse_args();print(run(a.raw,Path(__file__).resolve().parents[1],a.output_dir,a.binary))
