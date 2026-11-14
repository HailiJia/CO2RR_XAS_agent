"""Independent invariants and saved-model replay for this new study only."""
import ast,json,hashlib,sys
from pathlib import Path
import numpy as np,pandas as pd,joblib
O=Path(__file__).resolve().parents[1];REPO=O.parents[1];sys.path.insert(0,str(REPO))
from tools.differential_identity import signed_shape
from tools.differential_xas import features
from tools.spectral_multilabel import binary_scores,SPECIES
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 rows=json.load(open(O/'dataset_manifest.json'));old=json.load(open(O/'old_manifest.json'));z=np.load(O/'inputs.npz');oz=np.load(O/'old_inputs.npz');fam=np.array([r['near_group'] for r in rows]);checks={}
 for k in ['delta','raw_physical','clean']:np.testing.assert_array_equal(z[k][:141],oz[k])
 for i,r in enumerate(old):assert rows[i]['configuration']==r['configuration'] and rows[i]['identity_labels']==r['identity_labels'] and rows[i]['fold']==r['fold']
 checks['original141_inputs_labels_query_folds_exact']=True;folds=json.load(open(O/'fold_manifest.json'))
 for s in folds:
  tr=s['train'];te=s['test_original']+s['test_new'];assert not set(fam[tr])&set(fam[te]);assert len(set(fam[tr]))==len(tr)
  for inner in s['inner']:assert not set(fam[inner['train']])&set(fam[inner['validation']]);assert set(inner['train'])<=set(tr) and set(inner['validation'])<=set(tr)
 checks['all_outer_inner_geometry_guards']=True
 for p in O.glob('residual_template_audit_*.json'):
  for inner in json.load(open(p)):
   a=inner['train'];v=inner['validation']
   for i,template in zip(a,inner['templates_for_training']):assert set(map(int,template))<=set(a) and fam[i] not in set(fam[list(map(int,template))])
   for i,template in zip(v,inner['templates_for_validation']):assert set(map(int,template))<=set(a) and fam[i] not in set(fam[list(map(int,template))])
 checks['residual_templates_out_of_group_and_training_only']=True;count=0
 for p in (O/'models').glob('*.joblib'):
  b=joblib.load(p)
  if p.stem.startswith('residual_'):
   d=np.load(O/f'residual_diagnostic_{p.stem[len("residual_"):]}.npz');X=features(signed_shape(d['query_residual'],b['floor'])[0],z['energy'])['Combined'];np.testing.assert_array_equal(binary_scores(b['model'],X),d['score'])
  else:
   arm='old_only' if p.stem.startswith('old_only_') else 'expanded';seed=int(p.stem.rsplit('_',2)[1]);X=features(signed_shape(z['delta'][b['test']],b['floor'])[0],z['energy'])['Combined'];pred=np.load(O/'predictions'/f'{arm}_locked_{seed}.npz');s=binary_scores(b['model'],X);np.testing.assert_array_equal(s,pred['score'][b['test']]);np.testing.assert_array_equal(s>=b['common_threshold'],pred['prediction'][b['test']])
  count+=1
 checks['replayed_models']=count
 for seed in [20261020,20261021,20261022]:
  p=np.load(O/'predictions'/f'old_only_locked_{seed}.npz');prior=np.load(O/'prior_local_evidence'/f'baseline_{seed}.npz');np.testing.assert_array_equal(p['prediction'][:141],prior['prediction']);np.testing.assert_array_equal(p['score'][:141],prior['score'])
 checks['all_three_archived_baselines_exact']=True
 for p in (O/'code').glob('*.py'):ast.parse(p.read_text())
 protected=json.load(open(O/'protected_tracked_hashes.json'));changed=[p for p,h in protected.items() if not (REPO/p).is_file() or sha(REPO/p)!=h];assert not changed,changed;checks['protected_original_tracked_files']=len(protected);checks['manuscript_SI_figures_archived_benchmarks_unchanged']=True
 prov=json.load(open(O/'provenance.json'));source=Path(prov['new_dataset']);manifest=json.load(open(O/'source_manifest.json'));checks['source_files_rechecked']=0
 if source.exists():
  for p,v in manifest.items():assert sha(source/p)==v['sha256'];checks['source_files_rechecked']+=1
 checks['status']='passed';(O/'verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':main()
