"""Immutable, hash-logged cohort export and relocated scientific audit."""
import gzip,hashlib,json,re,shutil
from pathlib import Path
import numpy as np
import pandas as pd
from .component_audit import read_poscar,audit_components,legacy_benchmark_geometry_hash
from .dataset_labels import geometry_hash
from .layer_population import verify_layer_population
from .spectral_multilabel import record_identity,spectrum_layers,encode_labels,implementation_fingerprint

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def load_cohort(data_root,split_manifest,verify=True):
    root=Path(data_root).resolve();split=pd.read_csv(split_manifest);records=[];populations=[];evidence=[];grid=None;current_groups=[]
    if split.configuration.duplicated().any() or split.structure_hash.duplicated().any():raise ValueError('Duplicate benchmark configuration/geometry')
    for row in split.itertuples():
        path=(root/row.record).resolve()
        if not path.is_relative_to(root):raise ValueError('Record path escapes data root')
        if sha256(path)!=row.record_sha256:raise ValueError('Frozen record hash mismatch: '+row.record)
        d=json.loads(path.read_text());layers=spectrum_layers(d)
        if grid is not None and not np.array_equal(grid,layers[2][:,0]):raise ValueError('Incompatible cohort grids')
        grid=layers[2][:,0]
        structure=read_poscar(path.parent/'layer_5/CONTCAR')
        if legacy_benchmark_geometry_hash(structure)!=row.structure_hash:raise ValueError('Frozen calculation geometry changed; create a new version/comparison')
        audit=audit_components(structure);current_groups.append(geometry_hash(structure))
        try:observed_identity=record_identity(d)
        except ValueError as exc:
            c=d.get('sample',{}).get('composition',{})
            if any(k in c for k in ['adsorbate_counts','adsorbate_identities','adsorbate_identity']) or 'Missing identity' not in str(exc):raise
            observed_identity=tuple(audit['identities'])
        if set(observed_identity)!=set(row.labels.split(';'))-{'clean'}:raise ValueError('Frozen identity mismatch')
        pop={};report={}
        if verify:
            if audit['status']!='verified' or set(audit['identities'])!=set(observed_identity):raise ValueError('Component-label review required: '+row.configuration)
            for l in [2,3,4,5]:
                info=verify_layer_population(path.parent/f'layer_{l}',l);pop[l]=info['population'];report[l]=info
                local=np.loadtxt(path.parent/f'layer_{l}/Cu_layer{l}_conv.txt',skiprows=1)
                if not np.array_equal(local[:,0],layers[l][:,0]) or not np.allclose(local[:,1],layers[l][:,1],rtol=1e-7,atol=1e-9):raise ValueError('Record/output mismatch')
            means=[report[l]['mean_height_A'] for l in [2,3,4,5]]
            if not np.all(np.diff(means)>0):raise ValueError('Invalid layer order')
        else:
            raise ValueError('Supply audited populations; unverified cohort loading is disabled')
        records.append(d);populations.append(pop);evidence.append(dict(configuration=row.configuration,populations=pop,layers=report,component_audit=audit))
    for group in set(current_groups):
        if split.loc[np.array(current_groups)==group,'fold'].nunique()!=1:raise ValueError('Current-geometry group leakage in frozen splits')
    return dict(groups=np.array(current_groups),records=records,populations=populations,energy=grid,split=split,Y=encode_labels([row.split(';') for row in split.labels]),evidence=evidence)

def export_cohort(data_root,analysis_dir,output_dir,split_manifest=None,full_logs=False):
    root=Path(data_root).resolve();analysis=Path(analysis_dir).resolve();out=Path(output_dir).resolve()
    if out.exists() and any(out.iterdir()):raise ValueError('Export destination must be empty; never overwrite an export')
    out.mkdir(parents=True,exist_ok=True);split=pd.read_csv(split_manifest or analysis/'cv_manifest.csv');split.to_csv(out/'splits.csv',index=False)
    files=[];configs=[];potentials=[];seen=set()
    def copy(path,role):
        path=Path(path)
        if not path.exists() or not path.is_file():return
        rel=path.relative_to(root)
        if str(rel) in seen:return
        seen.add(str(rel));dest=out/'data'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        compressed=(path.name.endswith('_bav.txt') or path.name.startswith('fdmnes_raw')) and not full_logs
        before=hashlib.sha256();stored=dest.with_name(dest.name+'.gz') if compressed else dest
        with path.open('rb') as src:
            writer=gzip.open(stored,'wb',compresslevel=1) if compressed else stored.open('wb')
            with writer as dst:
                for chunk in iter(lambda:src.read(1024*1024),b''):before.update(chunk);dst.write(chunk)
        files.append(dict(path=str(stored.relative_to(out)),source_relative=str(rel),role=role,bytes=path.stat().st_size,
                          source_sha256=before.hexdigest(),sha256=sha256(stored),encoding='gzip-lossless' if compressed else 'identity'))
    aliases=pd.read_csv(analysis/'all_unique_structures.csv')
    for n,row in enumerate(split.itertuples()):
        record=root/row.record
        if sha256(record)!=row.record_sha256:raise ValueError('Source record changed: '+row.record)
        copy(record,'audited_isaac_record');folder=record.parent;source=root/row.configuration
        for name in ['POSCAR','CONTCAR','INCAR','KPOINTS','structure_info.json']:copy(source/name,'source_initial_final_or_numerical_settings')
        for pot in source.glob('POTCAR*'):
            if pot.is_file():potentials.append(dict(configuration=row.configuration,identifier=pot.name,potential_identifiers=[line.split('=',1)[1].strip() for line in pot.read_text(errors='replace').splitlines() if 'TITEL' in line and '=' in line],sha256=sha256(pot),bytes=pot.stat().st_size,exported=False))
        for layer in [2,3,4,5]:
            p=folder/f'layer_{layer}';stem=f'Cu_layer{layer}'
            for name in ['CONTCAR','POSCAR','absorbers.tsv',stem+'_in.txt',stem+'_convolution_in.txt',stem+'_conv.txt',stem+'_bav.txt','fdmnes.out','exit_code.txt','jobid.txt']:
                copy(p/name,'layer_input_output_or_log')
            for f in sorted(p.glob(stem+'_[0-9]*.txt')):copy(f,'site_spectrum')
            for pattern in ['*_resume_in.txt','fdmfile.txt','fdmnes_*.out','slurm-*.out','slurm-*.err','submit.sh']:
                for f in sorted(p.glob(pattern)):copy(f,'attempt_deck_or_execution_log')
        matches=aliases[aliases.aliases.map(lambda x:row.configuration in json.loads(x))]
        aliaslist=json.loads(matches.iloc[0].aliases) if len(matches) else [row.configuration]
        # Preserve input/final bytes for every known alias, not only its name.
        alias_hashes={}
        for alias in aliaslist:
            alias_hashes[alias]={}
            for name in ['POSCAR','CONTCAR']:
                path=root/alias/name
                if path.is_file() and path.stat().st_size:copy(path,'alias_coordinates');alias_hashes[alias][name]=sha256(path)
        configs.append(dict(configuration=row.configuration,record=row.record,fold=int(row.fold),legacy_geometry_hash=row.structure_hash,
                            aliases=aliaslist,source_coordinate_hashes=alias_hashes,record_sha256=row.record_sha256,
                            actual_xas_coordinates_sha256=sha256(folder/'layer_5/CONTCAR'),labels=row.labels.split(';')))
        if (n+1)%10==0:print('Exported',n+1,'/',len(split),flush=True)
    # Freeze the published protocol, targets and reversible metadata provenance.
    protocol=out/'protocol';protocol.mkdir()
    names=['all_unique_structures.csv','all_structure_folders.csv','record_audit.csv','layer_population_audit.csv','prior_fold_assignments.csv','run_config.json','run_summary.json','environment.json','record_corrections.json','two_CO_population_audit.json']
    names += [p.name for p in analysis.glob('records_before_*.tar.gz')]
    for name in names:
        p=analysis/name
        if p.exists():shutil.copy2(p,protocol/name)
    manifest=dict(version='co2rr-cohort-export-v1',scientific_benchmark='71bb0a20b369faba842b413a09544dd66c127a78:87-configurations',
                  source_mutated=False,configuration_count=len(configs),configurations=configs,files=files,potentials=potentials,
                  fingerprint=implementation_fingerprint(),logs='lossless gzip; verify_export/materialize before legacy audit' if not full_logs else 'original bytes')
    # Hash ALL exported files, including splits and provenance, excluding this self-referential manifest.
    manifest['exported_bytes']=[dict(path=str(p.relative_to(out)),sha256=sha256(p),bytes=p.stat().st_size) for p in sorted(out.rglob('*')) if p.is_file()]
    (out/'dataset_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def verify_export(root,materialize=False):
    root=Path(root);d=json.loads((root/'dataset_manifest.json').read_text())
    for f in d['exported_bytes']:
        p=(root/f['path']).resolve()
        if not p.is_relative_to(root.resolve()) or sha256(p)!=f['sha256']:raise ValueError('Export byte hash mismatch: '+f['path'])
    for f in d['files']:
        source=root/f['path'];h=hashlib.sha256()
        reader=gzip.open(source,'rb') if f['encoding']=='gzip-lossless' else source.open('rb')
        target=root/'data'/f['source_relative'];writer=target.open('wb') if materialize and f['encoding']=='gzip-lossless' and not target.exists() else None
        try:
            with reader:
                for chunk in iter(lambda:reader.read(1024*1024),b''):
                    h.update(chunk)
                    if writer:writer.write(chunk)
        finally:
            if writer:writer.close()
        if h.hexdigest()!=f['source_sha256']:raise ValueError('Uncompressed source hash mismatch')
    return d
