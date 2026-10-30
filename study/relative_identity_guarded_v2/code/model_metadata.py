from pathlib import Path
import json,hashlib,joblib
B=Path('study/relative_identity_guarded_v2');O=B/'results';rows=json.load(open(O/'source_manifest.json'));protocol=json.load(open(O/'protocol.json'));hashes=protocol['source_hashes'];before=[]
for p in (O/'models').glob('*.joblib'):
 m=joblib.load(p);m['training_input_hashes']=hashes;m['reference_record_hashes']=sorted({rows[i]['reference_record_sha256'] for i in m['train_indices']});m['source_scale_provenance']='study/dataset_20261029/scale/source_weight_audit.json';m['task']='identity';m['experimental_accuracy_validated']=False;m['absolute_coverage_available']=False;joblib.dump(m,p,compress=3)
for name in ['model_replay.json','fixed_representation_models.json']:
 p=O/name;records=json.load(open(p))
 for r in records:r['sha256']=hashlib.sha256((O/r['path']).read_bytes()).hexdigest()
 p.write_text(json.dumps(records,indent=2)+'\n')
