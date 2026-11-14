"""Inventory local reference output availability; potentials are hashed, never copied."""
from pathlib import Path
import argparse,json,hashlib,re
import pandas as pd
import h5py
import xml.etree.ElementTree as ET

def run(root,out):
    root=Path(root);out=Path(out);rows=[]
    for material in ['Cu','Cu2O','CuO']:
        for method in ['PBE','GW']:
            base=root/material/method/'XAS';row={'material':material,'method':method,'directory':str(base),'artifacts':[],'PAW_lineage':[]}
            for p in sorted((root/material/method).rglob('*')):
                if not p.is_file():continue
                if p.name not in ['OUTCAR','vasprun.xml','vaspout.h5','INCAR','POTCAR']:continue
                entry={'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
                if p.name=='POTCAR':
                    entry['TITEL']=re.findall(r'^\s*TITEL\s*=\s*(.*)',p.read_text(errors='replace'),re.M);row['PAW_lineage'].append(entry);continue
                if p.name=='INCAR':entry['tags']={k:v.strip() for k,v in re.findall(r'^\s*(ALGO|ICORELEVEL|CH_LSPEC|CH_SIGMA|CLNT|CLN|CLL|CLZ|NBANDS|ENCUT|LHFCALC|LSPECTRAL|LOPTICS)\s*=\s*([^!#\n]+)',p.read_text(errors='replace'),re.M)}
                if p.name=='OUTCAR':entry['imaginary_dielectric_blocks']=p.read_text(errors='replace').count('frequency dependent IMAGINARY DIELECTRIC FUNCTION')
                if p.name=='vasprun.xml':
                    try:
                        tree=ET.parse(p);entry['dielectric_blocks']=len(tree.findall('.//dielectricfunction'))
                    except ET.ParseError as e:entry['parse_status']=str(e)
                if p.name=='vaspout.h5':
                    names=[]
                    try:
                        with h5py.File(p) as f:f.visit(lambda n:names.append(n) if any(t in n.lower() for t in ['dielectric','xas','spectrum']) else None)
                        entry['spectrum_named_datasets']=names
                    except OSError as e:entry['parse_status']=str(e)
                row['artifacts'].append(entry)
            rows.append(row)
    out.mkdir(exist_ok=True);(out/'local_output_search.json').write_text(json.dumps({'root_role':'author working-folder benchmark, searched again for newly available outputs','series':rows,'PAW_policy':'TITEL and SHA only; no proprietary potential contents uploaded','energy_and_quantity':'native core-excitation energy, trace/3 of final imaginary dielectric tensor block; no absolute photon calibration','method_assignment':'PBE and GW supplied labels; no verified GW/BSE stage; separate PAW/core-zero calibration','native_width':'CH_SIGMA 0.1 eV recorded. Official guide calls this Gaussian width; source-specific sigma/FWHM interpretation not independently pinned. Added sigma1 diagnostic is explicitly Gaussian sigma=1 eV (FWHM=2.35482 eV).','missing':'Cu2O PBE SCF output has no numeric XAS block; Cu2O GW no XAS OUTCAR; CuO PBE incomplete output. No usable replacement recovered from XML/HDF5.'},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
