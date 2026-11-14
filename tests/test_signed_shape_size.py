"""Integrity checks on the declared training-size experiment and bounded decoder."""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'study/signed_shape_detectability_v3'

def test_size_queries_folds_and_effective_inner_groups_are_preserved():
 t=np.load(B/'ml/targets_folds.npz');fam=t['families'];query=np.flatnonzero(t['old99']);assert len(query)==99 and len(set(fam[query]))==96 and len(set(fam))==124
 for seed in [20261020,20261021,20261022]:
  for kind in ['trees','logistic']:
   for stage in ['fixed0p5','transferred','common']:
    old=np.load(B/f'ml/size_{kind}_old99_{seed}_{stage}.npz');new=np.load(B/f'ml/size_{kind}_expanded118_{seed}_{stage}.npz')
    np.testing.assert_array_equal(old['indices'],query);np.testing.assert_array_equal(new['indices'],query);np.testing.assert_array_equal(old['fold'],new['fold']);np.testing.assert_array_equal(old['truth'],new['truth'])
 for s in json.load(open(B/'verification/effective_inner_splits.json')):
  assert not set(fam[s['train']])&set(fam[s['validation']]);assert len(s['train'])==len(set(fam[s['train']]))

def test_learning_curve_is_group_nested_and_query_exclusive():
 t=np.load(B/'ml/targets_folds.npz');fam=t['families'];m=json.load(open(B/'ml/learning_membership.json'))
 for kind in ['trees','logistic']:
  for seed in [20261020,20261021,20261022]:
   for fold in range(5):
    rows=sorted([r for r in m if r['kind']==kind and r['seed']==seed and r['fold']==fold],key=lambda r:r['fraction']);assert set(rows[0]['train'])<=set(rows[1]['train'])<=set(rows[2]['train'])
    for r in rows:assert r['training_groups']==len(set(fam[r['train']])) and not set(fam[r['train']])&set(fam[r['test']])

def test_label_decoder_independent_limit_and_missing_class_remain_explicit():
 from tools.spectral_label_decoder import decode,pair_lift
 y=np.array([[1,0,0],[0,1,0],[1,1,0]]);s=np.array([[.5,.49,0],[.2,.8,0]]);J=pair_lift(y);np.testing.assert_array_equal(decode(s,J,0),(s>=.5).astype(int));assert np.isfinite(J).all();assert not decode(s,J,.25)[:,2].any()

def test_shared_size_recipe_checks_audited_sources_before_dispatch(monkeypatch,tmp_path):
 from tools.audited_training import run_audited_training
 import tools.signed_shape_size as module
 seen=[]
 monkeypatch.setattr(module,'train',lambda a,b:seen.append((a,b)) or {'status':'contract_dispatch_verified'})
 result=run_audited_training(ROOT/'study/dataset_20261029',tmp_path/'out',recipe='signed_shape_size_v3');assert result['status']=='contract_dispatch_verified' and seen
