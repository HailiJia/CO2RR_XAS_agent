"""Verify displayed benchmark metrics, cohort counts and saved OOF predictions; no refit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from sklearn.metrics import f1_score,accuracy_score
P=Path(__file__).resolve().parents[1];D=P/'figure_data';checks={}
a=pd.read_csv(D/'reference_plotted_arrays.csv');m=pd.read_csv(D/'similarity_metrics.csv');errors=[]
for (material,method,view),q in a.groupby(['material','method','comparison_type']):
 x=q.calculation_normalized.to_numpy();y=q.experiment_normalized.to_numpy();row=m[(m.material==material)&(m.method==method)&(m.comparison_type==view)].iloc[0]
 actual=np.array([x@y/(np.linalg.norm(x)*np.linalg.norm(y)),pearsonr(x,y).statistic,spearmanr(x,y).statistic]);expected=row[['cosine_similarity','Pearson_r','Spearman_rho']].to_numpy(float)
 np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=0);errors.append(float(abs(actual-expected).max()))
checks['benchmark']={'comparisons':len(errors),'max_metric_error':max(errors)}
names=['matched_identity_raw_physical','matched_identity_delta','same_substrate_sensitivity_identity_raw_physical'];loaded={}
for name in names:
 z=np.load(D/(name+'_oof.npz'));loaded[name]=z;t=z['truth'];p=z['prediction'];checks[name]={'n':len(t),'micro_F1':f1_score(t,p,average='micro',zero_division=0),'macro_F1_11':f1_score(t,p,average='macro',zero_division=0),'exact_set_accuracy':accuracy_score(t,p)}
for key in ['truth','prediction','configuration']:
 np.testing.assert_array_equal(loaded[names[0]][key],loaded[names[1]][key])
checks['matched_raw_delta_predictions_identical']=True
z=np.load(D/'paired_arrays.npz');np.testing.assert_allclose(z['delta'],z['raw_physical']-z['clean'],atol=1e-13,rtol=0)
checks['signed_difference_verified']=True
c=pd.read_csv(D/'Figure2b_current_counts.csv');checks['cohort_counts']=c.groupby('cohort').n.sum().astype(int).to_dict();assert sorted(checks['cohort_counts'].values())==[17,34,98]
manifest=json.loads((D/'pair_manifest.json').read_text());cuau=[r for r in manifest if r['domain']=='CuAu'];species=pd.read_csv(D/'Figure4b_current_per_species.csv')
for _,r in species.iterrows():assert sum(r['species'] in x['labels'] for x in cuau)==int(r.support)
checks['Figure4b_support_matches_Figure2b_manifest']=True
(D/'review_verification.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))
