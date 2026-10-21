"""Build manuscript tables/evidence index from measured or explicitly absent data."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    pd.DataFrame([('source locations',172),('unique structures',155),('record files',102),('duplicates',14),('grid incompatible',1),('eligible configurations',87),('layer spectra',348),('single-species configurations',78),('mixed configurations',9),('overlapping positives',99),('eligible clean',0),('eligible intact OCCO',0)],columns=['quantity','value']).to_csv(tables/'inventory_labels.csv',index=False)
    capabilities=json.loads((ROOT/'results/backend_capabilities/capability_evidence.json').read_text());pd.DataFrame([dict(backend=r['backend'],absorber=r['absorber'],edge='K',generated_input=r['generated_input'],numerical_validation=False,experimental_RMSE=None,experimental_r=None,core_hole_evidence='unresolved until actual output/version available') for r in capabilities]).to_csv(tables/'backend_edges_reference_metrics.csv',index=False)
    execution=json.loads((ROOT/'results/execution/execution_summary.json').read_text());pd.DataFrame([dict(quantity=k,value=v,status='observed' if v is not None and not k.endswith('rate') else 'unmeasured') for k,v in execution.items()]).to_csv(tables/'workflow_outputs_retries_interventions.csv',index=False)
    items=[
        ('Archived identity/depth OOF metrics and CSV reconciliation','completed','results/replay/','All supplied primary/fixed-depth/exploratory values retained; 87 configurations, 348 spectra'),
        ('Exact-cohort local export and byte verification','completed','results/export_verification.json','87 configurations; source files unchanged; lossless logs; no publication'),
        ('Relocated primary saved-model replay','completed','results/model_replay/saved_model_replay.json','All five held-out fold predictions identical after save/load'),
        ('Complete cold-cache 179-recipe reproduction','completed' if (ROOT/'results/full_reproduction.json').exists() else 'in progress','results/full_reproduction.json','895 fresh fit caches; all 169 archived OOF arrays match element for element'),
        ('Shared component audit and seven corrected examples','implementation-tested','results/seven_label_regressions.json','Skew in-plane minimum images; vacuum nonperiodic; cutoff/mapping review gates'),
        ('Final population utility and recovered CO eligibility','implementation-tested','../tools/layer_population.py','36 final sites retained; partial/duplicate/wrong-element/height/E_cut/grid cases tested'),
        ('Actual legacy layer-record producer repair','blocked','provenance/repository_review.json','Source/version absent; root converter is not assumed to own it'),
        ('Shared multilabel application route and leakage guards','implementation-tested','../tests/test_study_reconciliation.py','Separate from categorical advisor; outer-test ranking invariance and missing positives tested; not independent confirmation'),
        ('Representative schema/facility metadata proposal','implementation-tested','provenance/schema_source.json','Zero local pinned-schema errors; arrays unchanged; no original edit or portal request'),
        ('Actual FDMNES core-hole treatment','unavailable','provenance/schema_source.json','Actual input/default-version evidence still needed; no invented value'),
        ('Planned matrix, source reconciliation and priorities','completed','results/design/','Source ratio audit and missing cells are separate from completion'),
        ('Matched input quartet and prepared relaxation inputs','completed','results/design/matched_quartet.json','Shared cell/parent, loading/denominator verified; no jobs submitted'),
        ('Matched relaxed Cu/C/O coupling contrasts/discrimination','blocked','prepared_jobs/job_manifest.json','No verified final matched quartet spectra; dissociation must not become an intact label'),
        ('Numerical experimental Cu/Cu2O/CuO comparisons','unavailable','results/reference_trace_inventory.csv','Raw experimental traces/provenance and named PDF unavailable; no image-derived precision'),
        ('Geometry/coverage inference performance','planned','results/design/geometry_coverage_support_audit.csv','Support exported; targets not sufficiently verified for a new performance claim'),
        ('Confirmation robustness, calibration and OOD','planned','provenance/perturbation_plan.json','No frozen genuinely new confirmation set or experiment-grounded ranges'),
        ('Execution event backfill and prospective event code','implementation-tested','results/execution/','735 stdout paths -> 651 distinct scheduler job IDs; denominator incomplete'),
        ('Execution success/cost/intervention rates','blocked','results/execution/execution_summary.json','Accounting/restart/manual-action history missing; rates null'),
        ('ISAAC round-trip local fixtures','implementation-tested','../tools/isaac_roundtrip.py','Stable IDs, measurement/identity hashes and predictions tested locally only'),
        ('ISAAC remote deposition/retrieval/independent reuse','unavailable','../batch_validate_upload_isaac_records.py','No external writes/validation performed; no receipts claimed'),
        ('Five outline-role PDF/SVG/PNG figures and exact data','completed','figures/','Rebuilt and visually inspected; unmeasured panels explicitly pending'),
        ('Original supplied manuscript/caption reconciliation','blocked','README.md','Named source package absent; separate working draft follows accessible outline'),
        ('Compile paper twice and inspect all rendered pages','blocked','results/manuscript_build/compile_status.json','TeX engine unavailable; zero completed passes; no page-review claim'),
    ]
    df=pd.DataFrame(items,columns=['item','status','evidence','limit']);df.to_csv(ROOT/'evidence_table.csv',index=False)
    lines=['# Evidence status','', '| Item | Status | Evidence / limit |','|---|---|---|']
    for row in df.itertuples():lines.append(f'| {row.item} | {row.status} | [{row.evidence}]({row.evidence}) — {row.limit} |')
    (ROOT/'EVIDENCE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
