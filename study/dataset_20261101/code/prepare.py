"""Read-only conversion/audit of the November export, keeping old studies untouched."""
from pathlib import Path
import json,collections
from tools.coverage_study import convert,audit,sha,dump
from tools.dataset_refresh import fix_records
R=Path(__file__).resolve().parents[3];B=R/'study/dataset_20261101';raw=B/'local_export'
if __name__=='__main__':
 import sys
 if '--audit-only' not in sys.argv:
  inv=convert(raw,R,B);print('CONVERSION',collections.Counter(x['conversion'] for x in inv),flush=True)
 rows=audit(raw,R,B,extended_graph=True);print('AUDIT',collections.Counter('eligible_'+x['domain'] if x.get('eligible') else x.get('exclusion',x.get('conversion')) for x in rows),flush=True)
 fix_records(raw,B)
 for p in (B/'records').rglob('isaac_record_layers.json'):
  d=json.load(open(p));d['record_id']=d['record_id'].replace('20261029','20261101');d.setdefault('geometry_provenance',{})['current_export']='CO2RR_11101';d['conversion_provenance']['version']='dataset-refresh-20261101';dump(p,d)
 dump(B/'export_snapshot.json',{'source':str(raw),'parent_commit':'071ac428245b53ff015e28ec5172444afe9fd907','records_preserve_source':True,'converged_status':'not inferred from CONTCAR alone','manuscript_changes':False});print('PREPARED',flush=True)
