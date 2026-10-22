"""Build manuscript tables/evidence index from measured or explicitly absent data."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]

def main():
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    pd.DataFrame([('source locations',172),('unique structures',155),('record files',102),('duplicates',14),('grid incompatible',1),('eligible configurations',87),('layer spectra',348),('single-species configurations',78),('mixed configurations',9),('overlapping positives',99),('eligible clean',0),('eligible intact OCCO',0)],columns=['quantity','value']).to_csv(tables/'inventory_labels.csv',index=False)
    capabilities=json.loads((ROOT/'results/backend_capabilities/capability_evidence.json').read_text());pd.DataFrame([dict(backend=r['backend'],absorber=r['absorber'],edge='K',generated_input=r['generated_input'],numerical_validation=False,experimental_RMSE=None,experimental_r=None,core_hole_evidence='see versioned input/log audit; ISAAC field remains not_reported') for r in capabilities]).to_csv(tables/'backend_edges_reference_metrics.csv',index=False)
    execution=json.loads((ROOT/'results/execution/execution_summary.json').read_text());pd.DataFrame([dict(quantity=k,value=v,status='observed' if v is not None and not k.endswith('rate') else 'unmeasured') for k,v in execution.items()]).to_csv(tables/'workflow_outputs_retries_interventions.csv',index=False)
    items=[
        ('Archived identity/depth OOF metrics and CSV reconciliation','measured','results/replay/','87 configurations, 348 spectra; primary 0.7791/0.6383 and exploratory winner retained separately'),
        ('Exact-cohort local export and byte verification','completed','results/export_verification.json','87 configurations; source files immutable; full export remains local and unpublished'),
        ('Relocated agent-invoked archived-selector training','measured / replayed','results/agent_training_trace.json','Fresh 895-cache run; primary score reproduced; all five saved-model held-out folds replay exactly'),
        ('Current cold-cache OOF comparison','completed; differences disclosed','results/full_reproduction.json','140/169 OOF files exactly match; primary and exploratory fixed-winner prediction arrays exact; other model variants differ under current dependencies'),
        ('Shared component graph audit','implementation-tested','results/seven_label_regressions.json','Mapping/topology and seven corrected examples; skew in-plane minimum images; vacuum nonperiodic; ambiguity remains review-required'),
        ('Final FDMNES cell and layer population utility','implementation-tested','../tools/layer_population.py','All six cell metrics checked; final 36-site CO populations retained; incompatible grid excluded'),
        ('Actual legacy layer-record producer','unavailable','provenance/repository_review.json','Producer absent from accessible source; no attribution to unrelated root converter'),
        ('Shared audited multilabel application and agent route','executed / tested','results/agent_training_trace.json','Natural-language planner dispatch verified; CO_H is multilabel; categorical task remains separate; outer-test perturbation does not alter ranking'),
        ('Representative schema/facility metadata audit','implementation-tested','results/schema_review/schema_review.json','No original record edited; measurement arrays preserved; no portal request'),
        ('FDMNES default core-hole interpretation','version-supported inference','results/schema_review/core_hole_provenance.json','348 input decks and 332 convolution logs; 16 logs absent; schema treatment remains not_reported'),
        ('Planned matrix and source reconciliation','completed','results/design/','Cu:Au 1:1,2:1,1:2,3:1,1:3 with explicit rows and local motifs; requested cells distinct from completed inventory'),
        ('Matched 2CO/OCCO input quartet','inputs verified; new jobs not submitted','results/design/matched_quartet.json','Shared parent, cell, 2C+2O, coverage denominator and initial C-C distances verified'),
        ('Matched relaxed Cu/C/O coupling contrasts','planned / unavailable','prepared_jobs/job_manifest.json','No matched relaxed outputs; possible dissociation is not labeled intact OCCO; no spectral contrast invented'),
        ('Experimental reference metrics','unavailable','results/reference_trace_inventory.csv','Supplied CuO comparison PDF is visual context only; trace-level numeric data/provenance missing'),
        ('Geometry/coverage inference performance','unmeasured','results/design/geometry_coverage_support_audit.csv','Support and missing-label audit exported; targets not adequate for performance claims'),
        ('Confirmation robustness, calibration and OOD','planned','provenance/perturbation_plan.json','No frozen independent confirmation set or measured perturbation protocol'),
        ('Existing execution event evidence','partial; rates unavailable','results/execution/','Surviving log paths linked where supported; attempted-run denominator incomplete'),
        ('End-to-end local workflow/inference trace','implementation-tested; scheduler simulated','results/local_fixture_trace.json','Four simulated jobs link actual structure/final geometry to existing FDMNES record and fold-held-out prediction'),
        ('Production retry/cost/intervention rates','unavailable','results/execution/execution_summary.json','Scheduler, output-validity, record-creation and ML denominators are not reconciled'),
        ('ISAAC round-trip local fixtures','implementation-tested','../tools/isaac_roundtrip.py','Stable IDs, measurement/identity hashes and prediction replay tested locally only'),
        ('Remote ISAAC deposition/retrieval','not performed','../batch_validate_upload_isaac_records.py','No external writes, validation or receipts'),
        ('Five revised PDF/SVG/PNG figures and exact data','rebuilt and verified','manuscript_package/figures/','Figure 2 diagrams illustrative; Figure 5 uses starting inputs'),
        ('Revised manuscript and five captions','source available; authoritative','manuscript_package/manuscript.tex','Unconfirmed declarations remain placeholders; supplied PDF is pre-edit context only'),
        ('Compile paper twice and inspect final pages','blocked','results/manuscript_build/compile_status.json','No TeX engine; no SI source; no final page inspection possible'),
        ('Current focused scientific/workflow suite','passed','results/test_status.json','101 passed, 0 failed, 4 dependency/fallback warnings'),
    ]
    df=pd.DataFrame(items,columns=['item','status','evidence','limit']);df.to_csv(ROOT/'evidence_table.csv',index=False)
    lines=['# Evidence status','', '| Item | Status | Evidence / limit |','|---|---|---|']
    for row in df.itertuples():lines.append(f'| {row.item} | {row.status} | [{row.evidence}]({row.evidence}) — {row.limit} |')
    (ROOT/'EVIDENCE.md').write_text('\n'.join(lines)+'\n')
if __name__=='__main__':main()
