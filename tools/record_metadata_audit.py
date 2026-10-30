"""Schema-grounded proposed metadata repairs; no external writes by default."""
from copy import deepcopy
import hashlib,json
from pathlib import Path
from .component_audit import write_metadata_correction

def propose_fdmnes_metadata(record,schema):
    result=deepcopy(record);method=result.setdefault('computation',{}).setdefault('method',{})
    family_schema=schema['properties']['computation']['properties']['method']['properties']['family']
    if 'multiple_scattering_xas' in family_schema.get('enum',[]):method['family']='multiple_scattering_xas'
    else:
        # family is optional in the pinned schema. Do not misclassify FDMNES as
        # semi-empirical or invent an unsupported enum to satisfy a UI field.
        required=schema['properties']['computation']['properties']['method'].get('required',[])
        if 'family' in required:raise ValueError('Schema requires unsupported method family; review upstream vocabulary')
        method.pop('family',None)
    if method.get('core_hole_treatment') in ['not_reported','not_specified',None]:method.pop('core_hole_treatment',None)
    notes=method.get('notes','')
    method['notes']='FDMNES multiple-scattering XAS (Green-function mode where verified in input). Method family omitted: pinned ISAAC schema has no multiple-scattering enum and family is optional. Core-hole treatment remains scientifically unresolved until actual input/default-version evidence establishes it; no value is invented. Prior notes: '+notes
    facility=result.get('system',{}).get('facility',{})
    if facility.get('facility_name')=='NERSC':facility['organization']='Lawrence Berkeley National Laboratory'
    # These legacy fields are outside the pinned schema; original content is
    # retained in the sidecar and byte backup rather than mapped speculatively.
    for field in ['attribution','tags']:result.pop(field,None)
    context=result.get('context',{})
    if context.get('temperature_K') is None:context.pop('temperature_K',None)
    context.pop('temperature_basis',None)
    if 'temperature_K' not in context:result.pop('context',None)
    return result

def propose_from_paths(record_path,schema_path,output_dir,apply=False):
    import jsonschema
    record=json.loads(Path(record_path).read_text());schema=json.loads(Path(schema_path).read_text())
    corrected=propose_fdmnes_metadata(record,schema)
    # Report entire-record schema issues, not merely whether this patch is valid.
    errors=[dict(path='/'.join(map(str,e.path)),message=e.message) for e in jsonschema.Draft202012Validator(schema).iter_errors(corrected)]
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'proposed_record.json').write_text(json.dumps(corrected,indent=2)+'\n')
    result=write_metadata_correction(record_path,corrected,out,apply=apply)
    result.update(preserved_unmapped_metadata={k:record[k] for k in ['attribution','tags'] if k in record},original_context=record.get('context',{}),schema_sha256=hashlib.sha256(Path(schema_path).read_bytes()).hexdigest(),schema_errors=errors,portal_validated=False,published=False)
    (out/'schema_review.json').write_text(json.dumps(result,indent=2)+'\n');return result
