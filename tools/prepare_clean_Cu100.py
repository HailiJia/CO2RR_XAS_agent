"""Prepare missing clean-reference inputs without calculating or submitting jobs."""
from pathlib import Path
import json,hashlib
import numpy as np
from .progress_snapshots import write_json
from .component_audit import read_poscar


def poscar(s,title,fixed):
 p=np.asarray(s['positions']);cell=np.asarray(s['cell']);f=p@np.linalg.inv(cell);return '\n'.join([title,'1.0']+[' '.join(f'{x:.12f}' for x in row) for row in cell]+['Cu',str(len(p)),'Selective dynamics','Direct']+[' '.join(f'{v:.12f}' for v in q)+' '+('F F F' if i in fixed else 'T T T') for i,q in enumerate(f)])+'\n'
def prepare(study):
 B=Path(study);out=B/'ready_inputs/Cu100_clean_relax';out.mkdir(parents=True,exist_ok=True);r=next(x for x in json.loads((B/'scale/configuration_manifest.json').read_text()) if x['configuration'].endswith('CO_1_4_top'));source=r['ASE_lineage'];s=source['structure'];ids=np.flatnonzero(np.array(s['atoms'])=='Cu');clean={'positions':np.asarray(s['positions'])[ids],'cell':np.asarray(s['cell']),'atoms':['Cu']*len(ids)};fixed=set(range(8));(out/'POSCAR').write_text(poscar(clean,'INITIAL clean Cu100 guess, metals taken from covered ASE snapshot; RELAX BEFORE XAS',fixed));(out/'INCAR').write_text('ENCUT = 520\nPREC = Accurate\nEDIFF = 1E-6\nEDIFFG = -0.05\nIBRION = 2\nNSW = 200\nISIF = 2\nISMEAR = 1\nSIGMA = 0.20\nISPIN = 1\nLREAL = Auto\nLASPH = .TRUE.\nLDIPOL = .TRUE.\nIDIPOL = 3\nLWAVE = .FALSE.\nLCHARG = .FALSE.\n');(out/'KPOINTS').write_text('Proposed convergence-tested starting mesh, not a completed calculation\n0\nGamma\n6 6 1\n0 0 0\n')
 (out/'XAS_template.txt').write_text('Filout\n  Cu_layer{layer}\nRange\n  -5. 0.2 50.\nRadius\n  6.0\nEnergpho\nMemory_save\nSCF\nGreen\nPerdew\nCrystal\n  {validated_relaxed_cell_and_coordinates}\nAbsorber\n  {full_unique_layer_indices}\nEdge\n  K\nEnd\n')
 write_json(out/'provenance.json',{'status':'NOT CALCULATED; NOT SUBMITTED','source_ASE':source['path'],'source_sha256':source['sha256'],'initial_guess':'adsorbates removed from covered metal snapshot; subsequent clean relaxation required','not_a_completed_relaxed_clean_reference':True,'physical_populations':{'3':4,'4':4},'source_covered_snapshot_not_relaxed_clean':True,'planned_XAS':'same 4-layer cell, top layers3+4, Green/SCF/Perdew/radius6; current validated per-source equal-physical-Cu convention','potential':'Use licensed Cu PAW consistent with source relaxation and verify convergence; no potential supplied','constraints':'bottom 8 Cu fixed, top 8 free; constraints remain visible','k_mesh':'6x6x1 proposed; verify against available parent relaxation settings before treating controls as quantitatively matched'})
 (out/'README.md').write_text('# Missing clean Cu100 reference\n\nNo completed bare Cu100 geometry/XAS record was found in the current CO2RR export, original Cu100 ASE snapshot tree, CO2RR_XAS_agent/data export or Downloads/vasp_inputs. Existing Cu100 adsorbate configurations are not clean references. This directory is a proposed clean relaxation starting point, not a frozen-metal spectrum and not a completed relaxed clean reference. No jobs were submitted.\n\nAfter a verified successful relaxation, call `tools.prepare_clean_Cu100.prepare_xas(relax_dir, out_dir)`. It refuses to generate complete XAS decks without a successful relaxation record and the expected clean Cu16 cell. Do not replace this check with the starting POSCAR. Review PAW/mesh convergence against the original ASE calculation metadata, which lacks those detailed parameters.\n')
 write_json(B/'missing_references.json',[{'domain':'Cu100','missing':'completed clean Cu16 four-layer (100) slab CONTCAR/relaxation lineage + layer3/4 raw site spectra and convolution outputs','searched_roots':['CO2RR/Cu100','CO2RR/Cu-Ox_updated_structures/100','Downloads/CO2RR_XAS_agent/data','Downloads/vasp_inputs'],'ready_inputs':str(out.relative_to(B)),'joint_differential_training_status':'not available; no cross-facet substitute used','not_a_claim_of_never_run':True}])
 return out

def prepare_xas(relax_dir,out_dir):
    relax=Path(relax_dir);out=Path(out_dir);text=(relax/'OUTCAR').read_text(errors='replace')
    if 'reached required accuracy' not in text:raise ValueError('Successful relaxation evidence required before XAS')
    s=read_poscar(relax/'CONTCAR')
    if set(s['atoms'])!={'Cu'} or len(s['atoms'])!=16:raise ValueError('Clean Cu16 four-layer model required')
    heights=s['positions'][:,2];groups=[np.argsort(heights)[4*i:4*i+4] for i in range(4)]
    if min(np.diff([heights[g].mean() for g in groups]))<1.:raise ValueError('Four physical layers not established')
    lengths=np.linalg.norm(s['cell'],axis=1)
    if not np.allclose(s['cell'],np.diag(lengths),atol=1e-5):raise ValueError('Expected compatible orthogonal Cu100 cell')
    for layer in [3,4]:
        d=out/f'layer_{layer}';d.mkdir(parents=True,exist_ok=True);f=s['positions']@np.linalg.inv(s['cell']);deck=f'Filout\n Cu_layer{layer}\nRange\n -5. 0.2 50.\nRadius\n 6.0\nEnergpho\nSCF\nGreen\nPerdew\nCrystal\n '+' '.join(map(str,[*lengths,90,90,90]))+'\n'+''.join('29 '+' '.join(f'{v:.12f}' for v in row)+'\n' for row in f)+f'Absorber\n '+ ' '.join(str(i+1) for i in groups[layer-1])+'\nEdge\n K\nEnd\n';(d/f'Cu_layer{layer}_in.txt').write_text(deck)
    return out
if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--study',required=True);a=p.parse_args();print(prepare(a.study))
