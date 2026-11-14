"""Replay all new models, source objects, record arrays and group disjointness."""
from pathlib import Path
import sys,json,hashlib,tarfile,gzip
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.coverage_inference import predict,bank
from tools.spectral_multilabel import binary_scores
B=ROOT/'study/coverage_20261026';rows=json.loads((B/'model_configuration_manifest.json').read_text());inputs=np.load(B/'model_inputs.npz');features=bank(inputs['raw'],inputs['energy']);bygroup={r['group']:i for i,r in enumerate(rows)};checks=[];coverage=pd.read_csv(B/'coverage_heldout_predictions.csv');contact={x['configuration']:x for x in json.loads((B/'contact_predictions.json').read_text())};ioof=np.load(B/'identity_oof.npz');imap={v:i for i,v in enumerate(ioof['configuration'])}
for path in sorted((B/'models').glob('*.joblib')):
 m=joblib.load(path);assert not set(m['train_groups'])&set(m['test_groups']),path;ids=np.array([bygroup[g] for g in m['test_groups']]);records=[json.loads((B/'records'/rows[i]['configuration']/'isaac_record_layers.json').read_text()) for i in ids]
 if 'estimators'in m:
  score=binary_scores(m['estimators'],features['Combined'][ids]);pred=(score>=m['thresholds']).astype(int)
  expected=ioof if path.stem.startswith('identity_fold') else np.load(B/'identity_Cu100_external.npz');mapping={v:i for i,v in enumerate(expected['configuration'])};take=[mapping[rows[i]['configuration']] for i in ids];assert np.array_equal(score,expected['score'][take]) and np.array_equal(pred,expected['prediction'][take])
 else:
  score=m['estimator'].predict(features[m['feature']][ids]);reload=[]
  for i,d in zip(ids,records):
   pops={int(k):v['population'] for k,v in rows[i]['layer_evidence'].items()};reload.append(predict(d,m,pops)[0])
  assert np.allclose(score,np.array(reload),rtol=0,atol=1e-11),(path,float(np.max(abs(score-np.array(reload)))))
  if path.stem.startswith('coverage_fold'):
   frame=coverage[(coverage.protocol=='coverage_configuration_folds')&(coverage.model=='nested')].set_index('configuration');assert np.allclose(score,[frame.loc[rows[i]['configuration'],'prediction_ML'] for i in ids],rtol=0,atol=1e-12)
  elif path.stem.startswith('contacts_fold'):
   expected=[[contact[rows[i]['configuration']]['prediction_mean_Cu'],contact[rows[i]['configuration']]['prediction_mean_Au']] for i in ids];assert np.allclose(score,expected,rtol=0,atol=1e-12)
 checks.append({'file':str(path.relative_to(B)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'model_replay':True,'groups_disjoint':True})
# Native arrays in every converted record reproduce exact preserved canonical raw curves.
source=json.loads((B/'source_manifest.json').read_text());expected={x['sha256'] for x in source};objects={};seen=set();needed={x['sha256'] for x in source if x['path'].endswith('_conv.txt')}
with tarfile.open(B/'source_evidence.tar.gz') as t:
 for m in t:
  b=t.extractfile(m).read();h=hashlib.sha256(b).hexdigest();assert m.name=='objects/'+h and h in expected;seen.add(h)
  if h in needed:objects[h]=b
assert seen==expected
for r in json.loads((B/'conversion_inventory.json').read_text()):
 if r['conversion']!='converted':continue
 d=json.loads((B/r['converted_record']).read_text());original=json.loads(gzip.decompress((B/r['converted_record']).parent.joinpath('converter_original.json.gz').read_bytes()));assert d['measurement']['series']==original['measurement']['series']
 for s in d['measurement']['series']:
  l=s['series_id'].rsplit('_',1)[1];h=d['conversion_provenance']['native_files'][l]['sha256'];a=np.loadtxt(__import__('io').StringIO(objects[h].decode()),skiprows=1);assert np.array_equal(a[:,0],s['independent_variables'][0]['values']) and np.array_equal(a[:,1],s['channels'][0]['values'])
for name,h in json.loads((B/'protocol.json').read_text())['source_hashes'].items():assert hashlib.sha256((ROOT/'tools'/name).read_bytes()).hexdigest()==h,name
print(json.dumps({'model_files_replayed':len(checks),'source_objects_verified':len(seen),'source_manifest_entries':len(source),'records_native_arrays_unchanged':126,'scientific_source_hashes_match':True,'models':checks},indent=2))
