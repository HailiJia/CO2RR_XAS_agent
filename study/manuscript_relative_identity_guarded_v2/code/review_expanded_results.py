"""Recompute saved predictions and normalization diagnostics; never train a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'figure_data';N=D/'dataset_20261029'
rows=json.loads((N/'training_manifest_compact.json').read_text())
species=json.loads((N/'ml/protocol.json').read_text())['supported_labels']
lookup={r['configuration']:r for r in rows}
def metrics(z,mask=None,cols=None):
    t=z['truth'];p=z['prediction']
    if mask is not None:t=t[mask];p=p[mask]
    if cols is not None:t=t[:,cols];p=p[:,cols]
    if t.ndim==1:return dict(n=len(t),MAE=mean_absolute_error(t,p),R2=r2_score(t,p))
    return dict(n=len(t),micro_F1=f1_score(t,p,average='micro',zero_division=0),macro_F1=f1_score(t,p,average='macro',zero_division=0),exact_set=accuracy_score(t,p))
def run():
    checks={};saved=[];subsets=[]
    for f in sorted((N/'ml').glob('*.npz')):
        z=np.load(f);ids=z['pair_indices'];names=z['configuration']
        assert all(rows[i]['configuration']==n for i,n in zip(ids,names))
        assert all(rows[i]['fold']==v for i,v in zip(ids,z['fold']))
        if z['truth'].ndim==2:
            expected=np.array([[s in rows[i]['identity_labels'] for s in species] for i in ids],int)
            np.testing.assert_array_equal(z['truth'],expected)
        else:np.testing.assert_allclose(z['truth'],[rows[i]['coverage_ML'] for i in ids],atol=1e-14)
        saved.append(dict(file=f.name,**metrics(z)))
        if 'identity_refresh_' in f.name and 'compact_nested' in f.name:
            for dom in ['CuAu','Cu100']:
                for group in ['all','single','mixed','clean']:
                    mask=np.array([rows[i]['domain']==dom and (group=='all' or (len(rows[i]['identity_labels'])==1 if group=='single' else len(rows[i]['identity_labels'])>1 if group=='mixed' else len(rows[i]['identity_labels'])==0)) for i in ids])
                    if mask.any():subsets.append(dict(file=f.name,domain=dom,group=group,**metrics(z,mask)))
    pd.DataFrame(saved).to_csv(N/'review_recomputed_metrics.csv',index=False)
    pd.DataFrame(subsets).to_csv(N/'review_domain_mixture_scores.csv',index=False)
    checks['saved_prediction_files_checked']=len(saved)
    z=np.load(N/'training_arrays.npz');np.testing.assert_allclose(z['delta'],z['raw_physical']-z['clean'],atol=1e-13)
    checks['signed_difference_verified']=True
    # A common unknown multiplicative scale cancels in reference normalization.
    for factor in [.2,7.3,30.]:np.testing.assert_allclose((z['raw_physical']*factor-z['clean']*factor)/factor,z['delta'],atol=1e-13)
    checks['common_multiplicative_scale_cancels']=True
    for task in ['identity','CO_coverage']:
        a=np.load(N/f'ml/{task}_refresh_CuAu_physical_raw_compact_nested_20261027.npz');b=np.load(N/f'ml/{task}_refresh_CuAu_delta_compact_nested_20261027.npz')
        np.testing.assert_array_equal(a['truth'],b['truth'])
        if task=='identity':np.testing.assert_array_equal(a['prediction'],b['prediction'])
        else:np.testing.assert_allclose(a['prediction'],b['prediction'],atol=1e-12,rtol=0)
    checks['CuAu_raw_delta_identity_identical_coverage_equal_within_1e12']=True
    paired=[]
    for seed in [20261020,20261021,20261022]:
        b=np.load(N/f'ml/identity_baseline99_CuAu_normalized_raw_inner_retuned_threshold_{seed}.npz');paired.append(dict(seed=seed,stage='baseline99',**metrics(b)))
        for stage in ['inner_retuned_threshold','transferred_baseline_threshold']:
            q=np.load(N/f'ml/identity_refresh_CuAu_normalized_raw_{stage}_{seed}.npz');idx={n:i for i,n in enumerate(q['configuration'])};mask=np.array([idx[n] for n in b['configuration']]);np.testing.assert_array_equal(q['truth'][mask],b['truth']);paired.append(dict(seed=seed,stage=stage,**metrics(q,mask)))
    pd.DataFrame(paired).to_csv(N/'review_paired_identity.csv',index=False)
    q=np.load(N/'ml/identity_refresh_CuAu_delta_compact_nested_20261027.npz');ps=[]
    for j,s in enumerate(species):ps.append(dict(species=s,support=int(q['truth'][:,j].sum()),F1=f1_score(q['truth'][:,j],q['prediction'][:,j],zero_division=0)))
    pd.DataFrame(ps).to_csv(D/'Figure4b_current_per_species.csv',index=False)
    shape=[]
    def norm(a):
        v=a-a.min(axis=1,keepdims=True);return v/np.trapezoid(v,z['energy'],axis=1)[:,None]
    v=norm(z['raw_native'])-norm(z['raw_physical'])
    for dom in ['CuAu','Cu100']:
        ids=[i for i,r in enumerate(rows) if r['domain']==dom];err=np.trapezoid(abs(v[ids]),z['energy'],axis=1)
        shape.append(dict(domain=dom,n=len(ids),detectably_different=int((err>1e-8).sum()),max_integrated_absolute_difference=float(err.max())))
    pd.DataFrame(shape).to_csv(N/'review_native_vs_corrected_shape.csv',index=False)
    # Recompute the reference scores from the actual plotted arrays.
    a=pd.read_csv(D/'reference_plotted_arrays.csv');m=pd.read_csv(D/'similarity_metrics.csv');err=[]
    for (mat,method,view),q in a.groupby(['material','method','comparison_type']):
        x=q.calculation_normalized.to_numpy();y=q.experiment_normalized.to_numpy();r=m[(m.material==mat)&(m.method==method)&(m.comparison_type==view)].iloc[0]
        got=[x@y/(np.linalg.norm(x)*np.linalg.norm(y)),pearsonr(x,y).statistic,spearmanr(x,y).statistic];ex=r[['cosine_similarity','Pearson_r','Spearman_rho']].to_numpy(float);np.testing.assert_allclose(got,ex,atol=1e-12,rtol=0);err.append(float(np.max(abs(got-ex))))
    checks['reference_metric_triplets']=len(err);checks['max_reference_metric_error']=max(err)
    checks['cohorts']={'CuAu_closed_label':sum(r['domain']=='CuAu' for r in rows),'Cu100_closed_label':sum(r['domain']=='Cu100' for r in rows),'source_scaled':137,'versioned_records_source_report':146,'native_series_source_report':553}
    (N/'review_verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
if __name__=='__main__':run()
