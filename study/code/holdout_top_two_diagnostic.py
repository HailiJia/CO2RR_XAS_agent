"""Fixed Combined/top-two model trained only on old 87, three new cases held out."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fixed_top_two import load_data,fit,SEEDS,digest
from tools.updated_data_training import inner_folds,threshold_fit
from tools.spectral_multilabel import binary_scores,SPECIES
B=ROOT/'study/top_two_20261023/diagnosis/fixed_top_two_old_only'


def main():
 B.mkdir(parents=True,exist_ok=True);data=load_data(ROOT);x=data['bank']['shape_CDF_derivatives'];y=data['y'];tr=np.arange(87);te=np.arange(87,90);rows=[]
 protocol={'depth':'top_2 only','feature':'Combined fixed by author','leaf':1,'trees':64,'training':'old 87 only; all new/recovered configurations excluded from all fitting/threshold selection','seeds':list(SEEDS),'calibration':'none; tree scores are not calibrated probabilities','evaluation':'diagnostic only, n=3 CO-only, not prospectively registered','source_sha256':digest(__file__)};(B/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 for seed in SEEDS:
  inside=np.zeros((87,len(SPECIES)))
  for it,iv in inner_folds(y[:87],5):inside[iv]=binary_scores(fit(x[it],y[it],1,seed),x[iv])
  q,ts,decoder=threshold_fit(y[:87],inside);m=fit(x[tr],y[tr],1,seed);score=binary_scores(m,x[te]);bundle={'format':'co2rr-multilabel-v1','fit':m,'energy':data['energy'],'species':list(SPECIES),'depth':'top_2','feature':'shape_CDF_derivatives','thresholds':ts,'provenance':{'experiment':'fixed Combined top-two old-only diagnosis','training_indices':tr.tolist(),'test_indices':te.tolist(),'source_snapshot_indices':data['source_indices'].tolist(),'seed':seed},'normalization':'raw population-weighted top-two mean before unit-area normalization'};p=B/f'Combined_old87_seed{seed}.joblib';joblib.dump(bundle,p,compress=3);assert np.array_equal(binary_scores(joblib.load(p)['fit'],x[te]),score)
  np.savez_compressed(B/f'heldout_seed{seed}.npz',truth=y[te],score=score,thresholds=ts,prediction=(score>=ts).astype(int),configuration=data['configuration'][te],species=SPECIES,source_indices=data['source_indices'][te])
  for i,idx in enumerate(te):
   for j,s in enumerate(SPECIES):rows.append({'seed':seed,'configuration':data['configuration'][idx],'species':s,'truth':int(y[idx,j]),'score':float(score[i,j]),'training_only_threshold':float(ts[j]),'predicted':bool(score[i,j]>=ts[j]),'fixed0p5_prediction':bool(score[i,j]>=.5),'inner_micro_f1':q,'decoder':decoder,'model_sha256':digest(p)})
 pd.DataFrame(rows).to_csv(B/'all_scores.csv',index=False);print(pd.DataFrame(rows).query("species=='CO'").to_string(index=False))
if __name__=='__main__':main()
