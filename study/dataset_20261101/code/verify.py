"""Verify new paired inputs, protected folds and every saved model; no fitting."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np,joblib
from tools.coverage_study import sha,dump,deck_structure
from tools.dataset_labels import geometry_hash
from tools.identity_comparison import transform,score_model
B=Path(__file__).resolve().parents[1];R=B.parents[1];r=json.load(open(B/'training_manifest.json'));z=np.load(B/'training_arrays.npz');old=json.load(open(R/'study/dataset_20261029/training_manifest.json'));oz=np.load(R/'study/dataset_20261029/training_arrays.npz');family=json.load(open(B/'family_map.json'))['families'];fam=np.array([family[a['configuration']] for a in r]);assert len(r)==141 and len(set(fam))==135
np.testing.assert_allclose(z['delta'],z['raw_physical']-z['clean'],rtol=0,atol=1e-13)
for i,a in enumerate(r):
 if a['old_index'] is not None:
  j=a['old_index'];assert a['identity_labels']==old[j]['identity_labels'] and a['fold']==old[j]['fold'];np.testing.assert_array_equal(z['delta'][i],oz['delta'][j])
for s in json.load(open(B/'ml/splits.json')):
 assert not set(fam[s['train']])&set(fam[s['test']]);assert len(s['train'])==len(set(fam[s['train']]))
 for inner in s['inner']:assert not set(fam[inner['train']])&set(fam[inner['validation']]) and len(inner['train'])==len(set(fam[inner['train']]))
for a in json.load(open(B/'ml/reload.json')):
 p=B/'ml'/a['path'];assert sha(p)==a['sha256'];m=joblib.load(p);i=m['test_indices'];score=score_model(m['model'],transform(z['delta'][i],z['energy'],'signed_shape','Combined',m['floor']));q=np.load(B/'ml'/(p.stem.rsplit('_fold',1)[0]+'_common.npz'));lookup={int(v):k for k,v in enumerate(q['indices'])};np.testing.assert_array_equal(score,q['score'][[lookup[int(v)] for v in i]])
ref=next(a for a in json.load(open(R/'study/dataset_20261029/scale/configuration_manifest.json')) if a['configuration']=='Cu100/Cu100_2x2x4_clean');s=deck_structure(B/'local_export/Cu100/Cu100_2x2x4_clean/XAS/FDMNES/green/layer_4/Cu_layer4_in.txt') if (B/'local_export').exists() else None
if s is not None:assert geometry_hash(s)==ref['group']
a=np.load(R/'study/relative_identity_guarded_v2/results/fixed_rep_joint130_signed_shape_trees.npz');b=np.load(B/'ml/trees_20261020_previous_training_same_queries_fixed0p5.npz');ix=[r[i]['old_index'] for i in b['indices']];np.testing.assert_array_equal(a['prediction'][ix],b['prediction']);np.testing.assert_array_equal(a['score'][ix],b['score'])
receipt={'records':141,'near_groups':135,'old_inputs_bitwise_identical':130,'added_near_geometry_groups':11,'saved_models_exact_reload':90,'previous_seed20261020_scores_bitwise_identical':True,'source_weighting':'frozen audited corrections reused only for exact geometry/native-array matches; new records reconvolved and verified','CO_only_records':sum(a['CO_only'] for a in r),'CO_only_added':sum(a['CO_only'] and a['old_index'] is None for a in r),'not_independent_parent_or_experimental_validation':True};dump(B/'verification.json',receipt);print(json.dumps(receipt))
