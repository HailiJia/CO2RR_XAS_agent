"""Supplementary read-only run-log search; absence is not never-run evidence."""
from pathlib import Path
import argparse,json,hashlib,os

def run(root,out):
    root=Path(root);rows=[];archives=[]
    for top in root.iterdir():
        if not top.is_dir() or top.name in ['benchmark','CO2RR_ML','co2rr_digital_discovery'] or top.name.startswith(('.','PR18')):continue
        for folder,dirs,files in os.walk(top):
            dirs[:]=[d for d in dirs if not d.startswith('.')]
            for name in files:
                p=Path(folder)/name
                if name.endswith(('.zip','.tar','.tgz','.tar.gz')):archives.append(str(p.relative_to(root)))
                if name not in ['OUTCAR','OSZICAR','vasprun.xml','vasp.out','relax.log'] and not name.endswith(('.log','.out')):continue
                text=p.read_text(errors='replace');rows.append({'path':str(p.relative_to(root)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'vasp_ionic_converged_marker':'reached required accuracy' in text,'vasp_final_timing_marker':'General timing and accounting' in text,'scope':'markers alone do not prove matched lineage or a usable spectrum'})
    Path(out).write_text(json.dumps({'logs':rows,'simulation_archives_found':archives,'absence_status':'not located; never-run status not inferred','scope':'scientific export directories; excludes analysis/package copies and benchmark'},indent=2)+'\n')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.root,a.out)
