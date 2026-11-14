"""Opt-in portal reuse verification. Fixture receipts are never deposition evidence."""
import hashlib,json
from copy import deepcopy
from .spectral_multilabel import record_identity

def semantic_hashes(record):
    digest=lambda v:hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    return dict(measurement_sha256=digest(record['measurement']['series']),identity_sha256=digest(record_identity(record)))

def roundtrip_records(records,client,allow_upload=False,predict=None):
    if not allow_upload:raise PermissionError('Portal writes require explicit opt-in authorization')
    reports=[]
    for record in records:
        validation=client.validate_record(record)
        if validation.get('valid') is not True and validation.get('status') not in ['valid','success']:raise ValueError('Portal validation did not explicitly pass')
        created=client.create_record(record);stable_id=created.get('record_id') or created.get('id')
        if not stable_id:raise ValueError('No stable portal ID returned')
        fetched=client.fetch_record(stable_id)
        if semantic_hashes(record)!=semantic_hashes(fetched):raise ValueError('Round-trip measurement/identity hashes changed')
        predictions_equal=None
        if predict is not None:
            import numpy as np
            predictions_equal=bool(np.array_equal(predict(record),predict(fetched)))
            if not predictions_equal:raise ValueError('Round-trip predictions changed')
        reports.append(dict(record_id=stable_id,**semantic_hashes(record),prediction_identical=predictions_equal,transport=getattr(client,'transport','remote_portal')))
    return reports

class FixturePortal:
    transport='local_fixture_only'
    def __init__(self):self.records={}
    def validate_record(self,record):return dict(valid=bool(record.get('measurement',{}).get('series')))
    def create_record(self,record):
        stable='fixture-'+semantic_hashes(record)['measurement_sha256'][:16];self.records[stable]=deepcopy(record);return dict(record_id=stable)
    def fetch_record(self,stable):return deepcopy(self.records[stable])
