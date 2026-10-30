"""Describe cohort/support changes without attributing score changes to added data."""
from pathlib import Path
import json
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,accuracy_score
P=Path(__file__).resolve().parents[1];D=P/'figure_data'
a=np.load(D/'preferred_style/oof_depth_top_2_micro.npz');b=np.load(D/'same_substrate_sensitivity_identity_raw_physical_oof.npz')
rows=[dict(species=str(s),old_87_support=int(a['truth'][:,i].sum()),current_98_support=int(b['truth'][:,i].sum())) for i,s in enumerate(a['species'])]
pd.DataFrame(rows).to_csv(D/'cohort_support_comparison_20261028.csv',index=False)
metrics=[]
for name,z in [('old_87_fixed_top_two',a),('current_98_physical_top_two',b)]:
 metrics.append(dict(analysis=name,n=len(z['truth']),micro_F1=f1_score(z['truth'],z['prediction'],average='micro',zero_division=0),macro_F1=f1_score(z['truth'],z['prediction'],average='macro',zero_division=0),exact_set_accuracy=accuracy_score(z['truth'],z['prediction'])))
(D/'cohort_comparison_20261028.json').write_text(json.dumps({'analyses':metrics,'supports':rows,'interpretation':'Not a controlled training-size experiment: cohort, intensity reconstruction, candidate models and validation protocol differ. No causal attribution to adding data.','identity_mapping_caution':'Only 68 paths match literally; other paths include renamed H and COH cases. Group hashes use different conventions. Do not interpret unmatched paths as newly added physical structures without coordinate-based reconciliation.'},indent=2)+'\n')
print(json.dumps(metrics,indent=2))
