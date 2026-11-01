from pathlib import Path
import json
import numpy as np
R=Path(__file__).resolve().parents[1];B=R/'study/dataset_20261101'

def test_previous_inputs_labels_and_folds_are_unchanged():
 r=json.load(open(B/'training_manifest.json'));old=json.load(open(R/'study/dataset_20261029/training_manifest.json'));z=np.load(B/'training_arrays.npz');oz=np.load(R/'study/dataset_20261029/training_arrays.npz');assert len(r)==141
 for i,a in enumerate(r):
  if a['old_index'] is not None:
   j=a['old_index'];np.testing.assert_array_equal(z['delta'][i],oz['delta'][j]);assert a['identity_labels']==old[j]['identity_labels'] and a['fold']==old[j]['fold']

def test_all_fit_partitions_exclude_near_query_groups():
 r=json.load(open(B/'training_manifest.json'));f=json.load(open(B/'family_map.json'))['families'];g=np.array([f[a['configuration']] for a in r]);assert len(set(g))==135
 for s in json.load(open(B/'ml/splits.json')):
  assert not set(g[s['train']])&set(g[s['test']]);assert len(s['train'])==len(set(g[s['train']]))
  for i in s['inner']:assert not set(g[i['train']])&set(g[i['validation']])

def test_record_fix_initializes_missing_composition_from_audited_geometry(tmp_path,monkeypatch):
 import tools.dataset_refresh as module
 out=tmp_path/'out';out.mkdir();p=out/'record.json';p.write_text(json.dumps({'sample':{'material':{}},'conversion_provenance':{}}));(out/'record_geometry_audit.json').write_text(json.dumps([{'conversion':'converted','configuration':'case','converted_record':'record.json','eligible':False}]))
 monkeypatch.setattr(module,'native',lambda f:{5:(tmp_path/'unused',np.ones((2,2)))})
 monkeypatch.setattr(module,'deck_structure',lambda p:{'atoms':['Cu'],'positions':np.zeros((1,3)),'cell':np.eye(3)})
 monkeypatch.setattr(module,'snapshot_graph',lambda s:{'components':[],'counts':{},'identities':[],'status':'verified'})
 module.fix_records(tmp_path/'raw',out);assert json.load(open(p))['sample']['composition']['atom_counts']=={'Cu':1}
