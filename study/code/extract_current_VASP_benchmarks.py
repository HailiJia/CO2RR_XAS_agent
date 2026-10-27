"""Read real current OUTCAR/XML/HDF5 evidence; no code-name-as-GW inference."""
from pathlib import Path
import sys,json,hashlib,re
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/reference_benchmark_v3';B.mkdir(exist_ok=True);raw=Path(sys.argv[1]);rows=[]
for material,method in [('Cu2O','PBE'),('Cu2O','GW'),('CuO','PBE')]:
    source=raw/'benchmark'/material/method/'XAS';out=source/'OUTCAR';text=out.read_text(errors='replace');incar=(source/'INCAR').read_text();tags={}
    for line in incar.splitlines():
        line=line.split('#')[0].split('!')[0]
        if '=' in line:k,v=line.split('=',1);tags[k.strip().upper()]=v.strip()
    entry={'material':material,'folder_label':method,'display_label':'VASP PBE' if method=='PBE' else 'VASP (GW-PAW)','source_folder':str(source.relative_to(raw)),'tags':tags,'OUTCAR_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'PAW_dataset_labels':sorted(set(re.findall(r'TITEL\s*=\s*([^\n]+)',text))),'electronic_termination_footer':'General timing and accounting' in text,'logged_independent_particle_response':'independent particle, no local field effects' in text,'GW_quasiparticle_or_BSE_established':False,'available':False,'stage':'XAS','electronic_EDIFF_reached':'aborting loop because EDIFF is reached' in text,'logged_IALGO':[int(v) for v in re.findall(r'IALGO\s*=\s*(\d+)',text)],'convergence_caveat':'complete response table and electronic EDIFF checked separately from run footer; current input file may be newer than the recorded numerical table'}
    if (source/'POTCAR').exists():entry['licensed_POTCAR_sha256_only']=hashlib.sha256((source/'POTCAR').read_bytes()).hexdigest()
    for name in ['OUTCAR','vasprun.xml','vaspout.h5','INCAR','KPOINTS','OSZICAR']:
        p=source/name
        if p.exists():entry.setdefault('files',[]).append({'path':str(p.relative_to(raw)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    # HDF5 two components are real/imaginary. Validate against the actual
    # OUTCAR diagonal table instead of inferring a plotted curve.
    try:
        with h5py.File(source/'vaspout.h5') as f:
            e=f['results/linear_response/XAS_energies_dielectric_function'][:];tensor=f['results/linear_response/XAS_dielectric_function'][:]
        if tensor.shape!=(3,3,len(e),2) or not np.isfinite(tensor).all():raise ValueError('Unsupported/nonfinite response tensor')
        start=text.rfind('frequency dependent IMAGINARY DIELECTRIC FUNCTION');parsed=[]
        for line in text[start:].splitlines()[3:]:
            try:v=[float(a) for a in line.split()]
            except ValueError:
                if parsed:break
                continue
            if len(v)!=7:
                if parsed:break
                continue
            parsed.append(v)
        t=np.array(parsed)
        if t.shape!=(len(e),7):raise ValueError('Complete independent OUTCAR table not available')
        diag=np.column_stack([tensor[i,i,:,1] for i in range(3)]);error=float(np.max(abs(t[:,1:4]-diag)));energy_error=float(np.max(abs(t[:,0]-e)))
        if energy_error>1e-6 or error>1e-9:raise ValueError('OUTCAR/HDF5 tensor mismatch')
        y=diag.mean(axis=1)
        if y.min()<-1e-12 or not np.all(np.diff(e)>0):raise ValueError('Invalid nonnegative absorption input')
        np.savetxt(B/f'{material}_{method}_native.csv',np.column_stack([e,diag,y]),delimiter=',',header='native_energy_eV,eps2_xx,eps2_yy,eps2_zz,isotropic_eps2',comments='');entry.update(available=True,numeric_quantity='imaginary independent-particle dielectric response; arithmetic mean xx/yy/zz; relative intensity, not absolute absorption cross section',tensor_shape=list(tensor.shape),native_energy_range_eV=[float(e[0]),float(e[-1])],OUTCAR_HDF5_max_diagonal_error=error,OUTCAR_HDF5_energy_error_eV=energy_error,numeric_complete=True)
    except (OSError,KeyError,ValueError) as exc:entry['unavailable_reason']=str(exc)
    # SCF preparatory stage is kept distinct from XAS runtime.
    timing=[]
    for stage,folder in [('SCF',source.parent),('XAS',source)]:
        p=folder/'OUTCAR'
        if not p.exists():continue
        tx=p.read_text(errors='replace');cpu=re.findall(r'Total CPU time used \(sec\):\s*([\d.]+)',tx);elapsed=re.findall(r'Elapsed time \(sec\):\s*([\d.]+)',tx);ranks=re.findall(r'running\s+(?:on\s+)?(\d+)\s+(?:total cores|mpi-ranks|total ranks)',tx)
        timing.append({'stage':stage,'reported_CPU_time_s':float(cpu[-1]) if cpu else None,'elapsed_s':float(elapsed[-1]) if elapsed else None,'logged_total_cores':int(ranks[-1]) if ranks else None,'allocated_core_time_estimate_s':float(elapsed[-1])*int(ranks[-1]) if elapsed and ranks else None,'estimate_not_measured_CPU':True,'source':str(p.relative_to(raw)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    entry['timing_by_stage']=timing;rows.append(entry)
(B/'current_VASP_provenance.json').write_text(json.dumps({'h5py_version':h5py.__version__,'sources':rows},indent=2)+'\n');print([(r['material'],r['folder_label'],r['available'],r.get('unavailable_reason')) for r in rows])
