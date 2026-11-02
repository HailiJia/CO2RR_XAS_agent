from pathlib import Path
import json
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'study/relative_identity_guarded_v2'

def test_close_relaxations_and_nominal_coverage_aliases_share_near_groups():
 m=json.load(open(B/'near_duplicate_family_map.json'));f=m['families'];assert m['n_families']==124;assert f['H/H_0p75/interface_biased2']==f['H/H_0p5/interface_biased2'];assert f['CHO/CHO_0p056_uniform']==f['CHO/CHO_0p056_interior_Cu']
 p=pd.read_csv(B/'near_geometry_pairs.csv');assert ((p.RMS_A<=.02)&(p.max_A<=.05)).sum()==3;assert ((p.RMS_A<=.02)&(p.max_A<=.2)).sum()==6

def test_saved_outer_and_inner_fit_partitions_exclude_related_query_geometries():
 f=json.load(open(B/'near_duplicate_family_map.json'))['families'];rows=json.load(open(B/'results/source_manifest.json'));families=np.array([f[r['configuration']] for r in rows]);splits=json.load(open(B/'results/splits.json'))
 for s in splits:
  tr=s['train'];te=s['test'];assert not set(families[tr])&set(families[te]);assert len(set(families[tr]))==len(tr)
  for inner in s.get('inner',[]):assert not set(families[inner['train']])&set(families[inner['validation']])

def test_guard_does_not_change_labels_grids_groups_or_original_query_folds():
 a=np.load(ROOT/'study/dataset_20261029/training_arrays.npz');b=np.load(B/'results/inputs.npz')
 for k in a.files:np.testing.assert_array_equal(a[k],b[k])
 old=json.load(open(ROOT/'study/dataset_20261029/training_manifest.json'));new=json.load(open(B/'results/source_manifest.json'));assert old==new
 t=np.load(B/'results/targets_folds.npz');assert t['fold'].tolist()==[r['fold'] for r in old];assert sum(t['old99'])==99

def test_primary_learning_subsets_sample_groups_after_query_exclusion():
 from tools.grouped_identity_learning import canonical_training
 rows=[{'old99':True,'configuration':'old'},{'old99':False,'configuration':'near_view'},{'old99':False,'configuration':'independent'}];families=np.array(['a','a','b']);assert canonical_training(rows,families,np.arange(3),np.array([1])).tolist()==[2];assert canonical_training(rows,families,np.arange(3),np.array([],int)).tolist()==[0,2]
 f=json.load(open(B/'near_duplicate_family_map.json'))['families'];rr=json.load(open(B/'results/source_manifest.json'));family=np.array([f[r['configuration']] for r in rr])
 for q in json.load(open(B/'group_learning/membership.json')):
  assert q['sampled_groups']==len(set(family[q['train']]))
  assert not set(family[q['train']])&set(family[q['test']])
