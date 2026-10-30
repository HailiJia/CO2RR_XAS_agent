"""Batch local validation, with explicitly authorized portal operations only."""
import argparse,json,zipfile
from pathlib import Path
from tools.isaac_roundtrip import semantic_hashes,roundtrip_records
from tools.isaac_portal_client import ISAACPortalClient

def load_records(path):
    path=Path(path)
    if path.suffix=='.zip':
        with zipfile.ZipFile(path) as z:return [json.loads(z.read(n)) for n in sorted(z.namelist()) if n.endswith('.json') and not n.startswith('__MACOSX/')]
    if path.is_dir():return [json.loads(p.read_text()) for p in sorted(path.rglob('*.json'))]
    return [json.loads(path.read_text())]

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',required=True);p.add_argument('--output',required=True);p.add_argument('--validate',action='store_true',help='Remote POST /validate; does not deposit');p.add_argument('--upload',action='store_true');p.add_argument('--allow-external-writes',action='store_true');p.add_argument('--schema');a=p.parse_args(argv)
    if a.upload and not a.allow_external_writes:p.error('--upload requires separately authorized --allow-external-writes')
    records=load_records(a.input);rows=[]
    client=ISAACPortalClient() if a.validate or a.upload else None
    for record in records:
        hashes=semantic_hashes(record);result=dict(record_id=record.get('record_id'),**hashes,validation='local_measurement_and_identity_contract',deposited=False)
        if a.schema:
            import jsonschema
            jsonschema.validate(record,json.loads(Path(a.schema).read_text()));result['validation']='pinned_local_schema'
        if a.validate:result['portal_validation']=client.validate_record(record)
        rows.append(result)
    if a.upload:
        receipts=roundtrip_records(records,client,allow_upload=True)
        for row,receipt in zip(rows,receipts):row.update(receipt,deposited=True)
    Path(a.output).write_text(json.dumps(rows,indent=2)+'\n');return rows
if __name__=='__main__':main()
