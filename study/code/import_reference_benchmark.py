"""Copy only reviewable reference inputs; leave the supplied directory immutable."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--destination', type=Path, required=True)
    a = p.parse_args()
    out = a.destination
    out.mkdir(parents=True, exist_ok=True)
    inventory = []
    for f in sorted(a.source.rglob('*')):
        if not f.is_file():
            continue
        rel = f.relative_to(a.source)
        # Explicit allowlist excludes wavefunctions, charge exports, potentials,
        # binaries, checkpoints, and unrelated simulation intermediates.
        top = len(rel.parts) == 1 and (f.suffix in {'.dat', '.cif', '.json', '.ipynb'} or f.name.endswith('_POSCAR'))
        fdm = any(x in rel.parts for x in ('fdm_green', 'fdm_fdm')) and f.suffix in {'.txt', '.pbs', '.out', '.err'}
        feff = 'feff' in rel.parts and (f.name in {'feff.inp', 'xmu.dat', 'feff.pbs', 'job.out', 'job.err', 'log.dat', 'HEADER', 'sfconv.inp'} or f.name.endswith('_feff_xmu.dat'))
        vasp = any(x in rel.parts for x in ('PBE', 'GW')) and f.name in {'INCAR', 'KPOINTS', 'POSCAR', 'OUTCAR', 'vasp.pbs', 'job.out', 'job.err'}
        include = (top or fdm or feff or vasp) and not any(x.startswith('.') for x in rel.parts)
        row = {'source_file': rel.as_posix(), 'bytes': f.stat().st_size, 'included': include}
        if include:
            b = f.read_bytes()
            row['sha256'] = hashlib.sha256(b).hexdigest()
            target = out / 'raw' / rel
            if f.name == 'OUTCAR' or f.name.endswith('_bav.txt'):
                target = target.with_name(target.name + '.gz')
                target.parent.mkdir(parents=True, exist_ok=True)
                # Lossless compression, reproducible header; uncompressed SHA above.
                target.write_bytes(gzip.compress(b, mtime=0))
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, target)
            row['repository_file'] = target.relative_to(out).as_posix()
            row['repository_bytes'] = target.stat().st_size
            row['repository_sha256'] = hashlib.sha256(target.read_bytes()).hexdigest()
        else:
            row['reason'] = 'outside explicit spectroscopy/input/log allowlist; original retained locally'
        inventory.append(row)
    (out / 'source_inventory.json').write_text(json.dumps(inventory, indent=2) + '\n')
    notebook = json.loads((out/'raw/Cu_XAS.ipynb').read_text())
    audit = [{'index': i, 'type': c['cell_type'], 'source': ''.join(c['source']),
              'execution_count': c.get('execution_count'), 'output_types': [x['output_type'] for x in c.get('outputs', [])]}
             for i, c in enumerate(notebook['cells'])]
    (out/'original_notebook_cells.json').write_text(json.dumps(audit, indent=2)+'\n')
    print(f'Inventoried {len(inventory)} files; copied {sum(x["included"] for x in inventory)} allowed inputs/logs.')


if __name__ == '__main__':
    main()
