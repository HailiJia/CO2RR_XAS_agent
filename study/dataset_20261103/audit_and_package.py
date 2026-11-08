"""Read-only source capture; this does not fit models or assign chemical labels."""
from pathlib import Path
import csv, hashlib, json, math, tarfile, datetime

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2) + '\n')

previous = json.loads((OUT/'previous_sources_manifest.json').read_text())['files']
comparison = []
for item in previous:
    p = ROOT/item['path']
    actual = sha(p) if p.is_file() else None
    comparison.append(dict(path=item['path'], previous_sha256=item['sha256'], current_sha256=actual,
                           status='missing' if actual is None else 'unchanged' if actual == item['sha256'] else 'changed'))
dump('previous_source_comparison.json', comparison)

selected = set()
spectra = []
roots = []
for folder in sorted(ROOT.iterdir()):
    if not folder.is_dir() or folder.name.startswith('.') or folder.name in {'CO2RR_Agent_MS','refresh_20261103'}:
        continue
    for p in folder.rglob('*'):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if 'XAS' in p.parts and (p.name.endswith('_conv.txt') or p.name.endswith('_in.txt') or p.name in {'CONTCAR','POSCAR','absorbers.tsv','layer_summary.tsv','method.txt','isaac_record_layers.json','exit_code.txt','fdmfile.txt'}):
            selected.add(p)
        elif p.name in {'CONTCAR','POSCAR','sites.csv','site_reference.csv'} and 'XAS' not in p.parts:
            selected.add(p)
        if p.name.endswith('_conv.txt') and 'XAS' in p.parts:
            rows=[]; error=None
            try:
                for line in p.read_text().splitlines():
                    cells=line.split()
                    if not cells: continue
                    try: values=[float(c.replace('D','E')) for c in cells]
                    except ValueError: continue
                    rows.append(values)
                valid=bool(rows) and all(len(r)>=2 and all(math.isfinite(v) for v in r) for r in rows)
                increasing=valid and all(b[0]>a[0] for a,b in zip(rows,rows[1:]))
            except Exception as exc:
                error=str(exc); valid=False; increasing=False
            spectra.append(dict(path=str(rel), n_points=len(rows), finite_numeric=valid, strictly_increasing_energy=increasing,
                                energy_min=rows[0][0] if rows else None, energy_max=rows[-1][0] if rows else None,error=error))

manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(selected)]
dump('source_manifest.json',manifest)
dump('spectrum_audit.json',spectra)
protected=[]
for p in sorted((ROOT/'CO2RR_Agent_MS').rglob('*')):
    if p.is_file(): protected.append(dict(path=str(p.relative_to(ROOT)),sha256=sha(p)))
dump('manuscript_unchanged_manifest.json',protected)
with tarfile.open(OUT/'latest_sources.tar.xz','w:xz',preset=9) as archive:
    for p in sorted(selected): archive.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
with tarfile.open(OUT/'latest_sources.tar.xz','r:xz') as archive:
    actual={m.name:hashlib.sha256(archive.extractfile(m).read()).hexdigest() for m in archive.getmembers()}
assert actual == {m['path']:m['sha256'] for m in manifest}
archive=OUT/'latest_sources.tar.xz'
parts=[]
with archive.open('rb') as source:
    i=0
    while chunk:=source.read(96000):
        p=OUT/f'latest_sources.tar.xz.part{i:03d}'; p.write_bytes(chunk)
        parts.append(dict(path=p.name,bytes=len(chunk),sha256=sha(p)));i+=1
dump('archive_manifest.json',dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),parts=parts,verified_member_hashes=len(actual)))
counts={s:sum(x['status']==s for x in comparison) for s in ['unchanged','changed','missing']}
summary=dict(capture_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(ROOT),selected_files=len(manifest),selected_bytes=sum(x['bytes'] for x in manifest),convolved_spectra=len(spectra),invalid_spectra=[x['path'] for x in spectra if not x['finite_numeric'] or not x['strictly_increasing_energy']],previous_source_comparison=counts,archive_bytes=archive.stat().st_size,archive_parts=len(parts),manuscript_files_protected=len(protected),retraining_status='BLOCKED: NumPy/SciPy/pandas/scikit-learn unavailable; no new model metrics',scope='Source capture, not an eligibility or chemical-lineage approval; excludes raw per-absorber unconvolved spectra, OUTCAR, wavefunctions and executables.')
dump('audit_summary.json',summary)
print(json.dumps(summary,indent=2))
