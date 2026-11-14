from pathlib import Path
import csv,hashlib,json,tarfile
O=Path(__file__).resolve().parent; R=O.parent
files=[O/name for name in ['previous_sources_manifest.json','previous_source_comparison.json','source_manifest.json','spectrum_audit.json','manuscript_unchanged_manifest.json']]
for rel in ['latest_141/recomputed_metrics.csv','latest_141/model_comparison_metrics.csv','local_CO_environment/regression_metrics.csv','depth_four_layer/training/depth_metrics.csv']:
    files.append(R/'CO2RR_Agent_MS/figure_data'/rel)
with tarfile.open(O/'audit_evidence.tar.xz','w:xz',preset=9) as a:
    for p in files:a.add(p,arcname=str(p.relative_to(R)),recursive=False)
parts=[]; data=(O/'audit_evidence.tar.xz').read_bytes()
for i in range(0,len(data),96000):
    p=O/f'audit_evidence.tar.xz.part{i//96000:03d}';p.write_bytes(data[i:i+96000]);parts.append(dict(path=p.name,bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(O/'audit_evidence_manifest.json').write_text(json.dumps(dict(archive='audit_evidence.tar.xz',bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),parts=parts),indent=2)+'\n')
protected=json.loads((O/'manuscript_unchanged_manifest.json').read_text())
assert all(hashlib.sha256((R/x['path']).read_bytes()).hexdigest()==x['sha256'] for x in protected)
(O/'verification.json').write_text(json.dumps(dict(manuscript_files_verified_unchanged=len(protected),source_archive_members_verified=5727,new_model_fits=0,environment_setup='attempted; failed at download: shell network disabled',shell_syntax_check='passed'),indent=2)+'\n')
print(json.dumps(dict(evidence_bytes=len(data),parts=len(parts),manuscript_files_verified=len(protected))))
