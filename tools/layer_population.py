"""Final-layer population evidence. E_cut log rounding tolerance: 0.00051 eV."""
import gzip
import hashlib
import re
from pathlib import Path
import numpy as np
import pandas as pd
from .component_audit import (read_poscar,read_fdmnes_crystal,
    FDMNES_CELL_LENGTH_TOLERANCE_A,FDMNES_CELL_ANGLE_TOLERANCE_DEG)

ECUT_TOLERANCE_EV=.00051

def _cell_lengths_angles(cell):
    a,b,c=np.asarray(cell,float)
    lengths=np.linalg.norm([a,b,c],axis=1)
    def angle(u,v):return float(np.rad2deg(np.arccos(np.clip(np.dot(u,v)/(np.linalg.norm(u)*np.linalg.norm(v)),-1,1))))
    return np.r_[lengths,angle(b,c),angle(a,c),angle(a,b)]

def verify_layer_population(folder,layer):
    folder=Path(folder);stem=f'Cu_layer{layer}';structure=read_poscar(folder/'CONTCAR');atoms=np.array(structure['atoms'])
    table=pd.read_csv(folder/'absorbers.tsv',sep='\t');indices=table.atom_index.to_numpy(int)
    if len(indices)!=len(set(indices)) or np.any(indices<1) or np.any(indices>len(atoms)):raise ValueError('Duplicate or out-of-range layer sites')
    if not np.all(atoms[indices-1]=='Cu'):raise ValueError('Wrong absorber element')
    normal=np.cross(structure['cell'][0],structure['cell'][1]);normal/=np.linalg.norm(normal)
    heights=structure['positions']@normal
    if not np.allclose(heights[indices-1],table.height_A,atol=2e-5,rtol=0):raise ValueError('Wrong absorber heights')
    groups=[]
    for index in sorted(np.flatnonzero(atoms=='Cu'),key=lambda i:heights[i]):
        if not groups or heights[index]-heights[groups[-1][-1]]>.6:groups.append([])
        groups[-1].append(int(index))
    if len(groups)!=5 or set(indices)!={i+1 for i in groups[layer-1]}:raise ValueError('Layer table does not cover the full physical Cu layer')
    deck_path=folder/f'{stem}_in.txt';deck=deck_path.read_text();selected=list(map(int,re.search(r'(?im)^Absorber\s*\n([^\n]+)',deck)[1].split()))
    if len(selected)!=len(set(selected)) or not set(selected)<=set(indices):raise ValueError('Invalid main-deck absorber indices')
    parsed=read_fdmnes_crystal(deck)
    if parsed is None:raise ValueError('Invalid or missing six-parameter FDMNES Crystal cell')
    values=np.column_stack([np.array([{'Cu':29,'Au':79,'C':6,'O':8,'H':1}[a] for a in parsed['atoms']]),parsed['fractional']])
    numbers={'Cu':29,'Au':79,'C':6,'O':8,'H':1}
    if len(values)!=len(atoms) or not np.array_equal(values[:,0],[numbers[a] for a in atoms]):raise ValueError('Calculation input elements disagree')
    ref_metric=_cell_lengths_angles(structure['cell']);deck_metric=_cell_lengths_angles(parsed['cell'])
    if not np.allclose(deck_metric[:3],ref_metric[:3],atol=FDMNES_CELL_LENGTH_TOLERANCE_A,rtol=0):raise ValueError('FDMNES Crystal cell lengths disagree with relaxed structure')
    if not np.allclose(deck_metric[3:],ref_metric[3:],atol=FDMNES_CELL_ANGLE_TOLERANCE_DEG,rtol=0):raise ValueError('FDMNES Crystal cell angles disagree with relaxed structure')
    ref_fractional=structure['positions']@np.linalg.inv(structure['cell'])
    delta=(values[:,1:]-ref_fractional)@structure['cell']
    from .contact_checks import minimum_image_vectors
    # Fractional coordinates compare the same lattice frame, are invariant to
    # a rigid Cartesian rotation of the complete cell, wrap only in-plane, and
    # leave vacuum-direction displacement nonperiodic.
    if not np.allclose(minimum_image_vectors(delta,structure['cell'],(True,True,False)),0,atol=3e-6):raise ValueError('Calculation input coordinates disagree')
    evidence=[];site_paths=[];grid=None
    # A separate final convolution log supersedes a subset recalculation deck.
    logs=([folder/'fdmnes.out'] if (folder/'fdmnes.out').exists() else [])+sorted(folder.glob('fdmnes_convolution*.out'))
    candidates=[]
    for log in logs:
        entries_found=re.findall(r'Gamma_hole\s*=.*?E_cut\s*=\s*([-\d.]+).*?site\s+(\d+)',log.read_text())
        if entries_found:candidates.append((log,entries_found))
    # Prefer a complete convolution among surviving attempts, not a subset log.
    complete=[pair for pair in candidates if [int(site) for cut,site in pair[1]]==list(range(1,len(indices)+1))]
    if complete:
        log,entries=complete[-1]
    elif candidates:
        log,entries=candidates[-1]
    else:
        log=None;entries=[]
    if entries:
        ordered=sorted(indices.tolist())
        if [int(site) for cut,site in entries]!=list(range(1,len(ordered)+1)):raise ValueError('Partial or duplicate final convolution sites')
        files=sorted(folder.glob(stem+'_[0-9]*.txt'),key=lambda p:int(p.stem.rsplit('_',1)[1]))
        if [int(p.stem.rsplit('_',1)[1]) for p in files]!=ordered:raise ValueError('Convolution source file coverage mismatch')
        conv=folder/f'{stem}_convolution_in.txt'
        if conv.exists():
            text=conv.read_text()
            if not re.search(r'(?im)^Calculation\s*\n\s*'+stem+r'\s*$',text) or not re.search(r'(?im)^Conv_out\s*\n\s*'+stem+r'_conv\s*$',text):raise ValueError('Wrong final convolution source/destination')
        elif set(selected)!=set(indices):raise ValueError('Missing final convolution deck for subset recalculation')
        for atom,path,(cut,site) in zip(ordered,files,entries):
            header=list(map(float,path.read_text().splitlines()[0].split('=')[0].split()))
            if header[1]!=29 or not np.isfinite(header[13]) or header[13]<=0:raise ValueError('Wrong site element or invalid source multiplicity')
            error=abs(header[6]-float(cut))
            if error>ECUT_TOLERANCE_EV:raise ValueError('Unmatched E_cut')
            a=np.loadtxt(path,skiprows=2)
            if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0):raise ValueError('Invalid site spectrum')
            if grid is not None and not np.array_equal(grid,a[:,0]):raise ValueError('Incompatible site grids')
            grid=a[:,0];site_paths.append(path)
            evidence.append(dict(atom_index=int(atom),convolution_site=int(site),E_cut_difference_eV=error,source_symmetry_multiplicity=header[13],source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        provenance='final_convolution'
    else:
        if set(selected)!=set(indices):raise ValueError('Subset main deck without verified full final convolution')
        bav=folder/f'{stem}_bav.txt'
        logpath=bav if bav.exists() else bav.with_name(bav.name+'.gz')
        logged=[]
        if logpath.exists():
            opener=gzip.open if logpath.suffix=='.gz' else open
            with opener(logpath,'rt') as handle:
                for line in handle:
                    if 'Absorbing atom' in line:
                        m=re.search(r'Absorbing atom\s+(\d+) beginning',line)
                        if m:logged.append(int(m[1]))
        if sorted(logged)!=sorted(indices.tolist()):raise ValueError('Full main-run log does not cover layer')
        site_paths=[folder/f'{stem}_{i}.txt' for i in sorted(indices)]
        for atom,path in zip(sorted(indices),site_paths):
            a=np.loadtxt(path,skiprows=2)
            if not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0):raise ValueError('Invalid site spectrum')
            if grid is not None and not np.array_equal(grid,a[:,0]):raise ValueError('Incompatible site grids')
            grid=a[:,0];evidence.append(dict(atom_index=int(atom),source=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        provenance='full_main_run_with_site_files'
    output=folder/f'{stem}_conv.txt';a=np.loadtxt(output,skiprows=1)
    if a.ndim!=2 or a.shape[1]!=2 or not np.isfinite(a).all() or not np.all(np.diff(a[:,0])>0):raise ValueError('Invalid convolved grid')
    return dict(layer=int(layer),population=len(indices),main_deck_population=len(selected),provenance=provenance,
                convolution_log=log.name if log else None,convolution_log_sha256=hashlib.sha256(log.read_bytes()).hexdigest() if log else None,population_basis='unique physical Cu atom indices, not sum of source symmetry multiplicities',E_cut_tolerance_eV=ECUT_TOLERANCE_EV,mean_height_A=float(table.height_A.mean()),output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),sites=evidence,
                fdmnes_cell_parameters=parsed['lattice_parameters'],cell_length_tolerance_A=FDMNES_CELL_LENGTH_TOLERANCE_A,
                cell_angle_tolerance_deg=FDMNES_CELL_ANGLE_TOLERANCE_DEG,coordinate_frame='fractional coordinates in matching six-parameter cell; in-plane PBC only')

def aggregate_layers(arrays,populations,layers):
    """Average compatible raw arrays before any normalization."""
    chosen=[np.asarray(arrays[int(layer)],float) for layer in layers]
    if not all(np.array_equal(chosen[0][:,0],a[:,0]) for a in chosen):raise ValueError('Incompatible layer grids; extrapolation disabled')
    weights=np.array([populations[int(layer)] for layer in layers],float)
    if not np.isfinite(weights).all() or np.any(weights<=0):raise ValueError('Invalid represented populations')
    return chosen[0][:,0].copy(),np.average(np.stack([a[:,1] for a in chosen]),axis=0,weights=weights)
