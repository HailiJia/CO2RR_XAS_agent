"""Residual reference noise shared across queries, distinct from common-mode gain/noise."""
from pathlib import Path
import json
import numpy as np,pandas as pd,joblib
from tools.identity_comparison import score_model,transform
from tools.spectral_multilabel import f1_metrics
from tools.coverage_study import dump,sha

def evaluate(B=Path('study/relative_identity_guarded_v2'),verify_only=False):
 B=Path(B);P=B/'paired_robustness';z=np.load(B/'results/inputs.npz');t=np.load(B/'results/targets_folds.npz');e=z['energy'];mask=(e>=e[0]+1)&(e<=e[-1]-1);ec=e[mask];ids=np.flatnonzero(t['domain']=='CuAu');rows=[];policy=[]
 for rep in ['corrected_normalized','relative_delta','signed_shape']:
  for aug in [False,True]:
   score=np.zeros_like(t['truth'][ids],float);keep=np.zeros(len(ids),bool)
   for fold in range(5):
    te=ids[t['fold'][ids]==fold];pos=np.flatnonzero(t['fold'][ids]==fold);m=joblib.load(P/'models'/f'{rep}_aug{aug}_fold{fold}.joblib');a=z['raw_physical'][te].copy();b=z['clean'][te].copy();rng=np.random.default_rng(921+fold);sa=.01*np.max(abs(a),axis=1);sb=.01*float(np.max(abs(b)));common=rng.normal(0,sb,len(e));reference_error=rng.normal(0,sb,len(e));aa=a+common+rng.normal(size=a.shape)*sa[:,None];bb=b+common+reference_error;delta=(aa-bb)[:,mask];noise=np.sqrt(sa**2+sb**2);keep[pos]=np.sqrt(np.mean(delta**2,axis=1))>np.maximum(m['floor'],3*noise);x=aa[:,mask] if rep=='corrected_normalized' else delta;score[pos]=score_model(m['model'],transform(x,ec,rep,'DCT16',m['floor']))
   pred=(score>=.5).astype(int);name=P/f'{rep}_aug{aug}_shared_reference_noise1pct.npz'
   if verify_only:
    q=np.load(name);np.testing.assert_array_equal(q['score'],score);np.testing.assert_array_equal(q['retained'],keep)
   else:np.savez_compressed(name,indices=ids,truth=t['truth'][ids],prediction=pred,score=score,fold=t['fold'][ids],retained=keep)
   rows.append({'representation':rep,'training_pair_augmentation':aug,'scenario':'shared_reference_noise1pct','n_queries':len(ids),**f1_metrics(t['truth'][ids],pred)});policy.append({'representation':rep,'augmented_training':aug,'retained_fraction':float(keep.mean()),'n_retained':int(keep.sum()),'full_query_exact_error':float(np.any(t['truth'][ids]!=pred,axis=1).mean()),'retained_exact_error':float(np.any(t['truth'][ids][keep]!=pred[keep],axis=1).mean()) if keep.any() else None,'policy':'RMS paired response >3*known injected independent-difference noise; identical eligibility for raw/delta controls; synthetic not measured'})
 if not verify_only:
  pd.DataFrame(rows).to_csv(P/'shared_reference_noise_metrics.csv',index=False);pd.DataFrame(policy).to_csv(P/'shared_reference_noise_review.csv',index=False);dump(P/'shared_reference_noise_protocol.json',{'reference_error':'one independently drawn reference vector shared by every query in an outer fold','common_incident_component':'same vector in both pair members, cancels separately','reaction_noise':'independent per query','sigma':'1% of each native per-common-scale peak; reference1% clean peak','window':[float(ec[0]),float(ec[-1])],'scope':'same fixed models, no tuning/refit; illustrates shared-reference covariance and low-SNR abstention, not experimental uncertainty','seed':921,'source_sha256':sha(Path(__file__))})
 print('Shared-reference scenarios verified' if verify_only else 'Saved six shared-reference residual-noise scenarios')
if __name__=='__main__':evaluate()
