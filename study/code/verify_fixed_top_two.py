"""Replay every saved fixed-top-two evaluation model on committed snapshot arrays."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fixed_top_two import load_data,CANDIDATES,SEEDS,paired_inner
from tools.spectral_multilabel import predict_records
B=ROOT/'study/top_two_20261023';E=B/'experiments'


def main():
 data=load_data(ROOT);z=np.load(data['base']/'cohort.npz');order=data['source_indices'];records=[];pops=[];results=[];protocol=json.load(open(E/'protocol.json'))
 for p,h in protocol['scientific_source_hashes'].items():assert hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h,p
 for i in order:
  stack=z['stack'][i];pop=z['populations'][i];records.append({'measurement':{'series':[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'unit':'eV','values':z['energy'].tolist()}],'channels':[{'name':'absorption_cross_section','values':stack[l-2].tolist()}]} for l in [2,3,4,5]]}});pops.append({l:int(pop[l-2]) for l in [2,3,4,5]})
 for path in sorted(E.rglob('*.joblib')):
  m=joblib.load(path);pr=m['provenance'];assert m['depth']=='top_2';n=pr['cohort'];seed=pr['seed'];fold=pr['fold'];te=np.array(pr['test_indices']);tr=np.array(pr['training_indices']);assert not set(data['groups'][tr])&set(data['groups'][te]);p,s=predict_records(m,[records[i] for i in te],[pops[i] for i in te]);expected=np.load(E/f'cohort{n}_seed{seed}/candidate_oof.npz')['score'][pr['candidate'],te];assert np.array_equal(s,expected),path;assert np.array_equal(p,s>=m['thresholds']);results.append({'path':str(path.relative_to(B)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'score_replay_identical':True,'depth':'top_2'})
 for path in sorted((B/'diagnosis/fixed_top_two_old_only').glob('*.joblib')):
  m=joblib.load(path);seed=m['provenance']['seed'];te=np.array(m['provenance']['test_indices']);p,s=predict_records(m,[records[i] for i in te],[pops[i] for i in te]);expected=np.load(path.parent/f'heldout_seed{seed}.npz');assert np.array_equal(s,expected['score']) and np.array_equal(p,expected['prediction']);results.append({'path':str(path.relative_to(B)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'score_replay_identical':True,'depth':'top_2','scope':'old-only three-case diagnostic'})
 for k in range(5):
  _,a=paired_inner(data,k,87);_,b=paired_inner(data,k,90)
  for (ta,va),(tb,vb) in zip(a,b):assert set(va)==set(vb[vb<87]) and set(ta)==set(tb[tb<87])
 a=np.load(E/'cohort87_seed20261020/original_feature_selection.npz');old=np.load(ROOT/'CO2RR_ML/updated_dataset_20261020/oof_depth_top_2_micro.npz');assert np.array_equal(a['prediction'],old['prediction']) and np.array_equal(a['truth'],old['truth'])
 print(json.dumps({'saved_model_files_verified':len(results),'all_score_replays_identical':True,'every_model_depth':'top_2','scientific_source_hashes_match':True,'old_inner_memberships_preserved_across_cohorts':True,'original_top_two_seed20261020_archived_predictions_match':True,'seed_repeats_are_not_independent_configurations':True,'models':results},indent=2))
if __name__=='__main__':main()
