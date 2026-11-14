"""Portable entry points for archived replay and full scientific reproduction."""
import argparse,json
from pathlib import Path

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('command',choices=['export','verify-export','audit','replay','model-replay','design','events']);p.add_argument('--data-root');p.add_argument('--export-root');p.add_argument('--output-dir',required=True);p.add_argument('--analysis-dir',default=str(Path(__file__).resolve().parents[1]/'CO2RR_ML/updated_dataset_20261020'));p.add_argument('--split-manifest');p.add_argument('--audit-only',action='store_true');p.add_argument('--full-logs',action='store_true');p.add_argument('--materialize',action='store_true');a=p.parse_args(argv)
    if a.command in ['export','audit','model-replay','design','events'] and not a.data_root:p.error('--data-root is required for this scientific operation')
    from .study_cohort import export_cohort,verify_export,load_cohort
    from .study_replay import replay_oof,replay_models
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True)
    if a.command=='export':
        # export_cohort expects an empty output directory, not an implicit overwrite.
        result=export_cohort(a.data_root,a.analysis_dir,out,a.split_manifest,a.full_logs);print('Exported configurations:',result['configuration_count'])
    elif a.command=='verify-export':print('Verified configurations:',verify_export(a.export_root or a.data_root,a.materialize)['configuration_count'])
    elif a.command=='audit':
        c=load_cohort(a.data_root,a.split_manifest or Path(a.analysis_dir)/'cv_manifest.csv');(out/'cohort_audit.json').write_text(json.dumps(c['evidence'],indent=2)+'\n');print('Audited configurations:',len(c['records']))
    elif a.command=='replay':print(replay_oof(a.analysis_dir,out,a.split_manifest).to_string(index=False))
    elif a.command=='model-replay':print(json.dumps(replay_models(a.data_root,a.analysis_dir,out,a.split_manifest),indent=2))
    elif a.command=='design':
        from .study_design import reconcile_design
        reconcile_design(Path(__file__).resolve().parents[1],a.data_root,a.analysis_dir,out)
    elif a.command=='events':
        from .execution_evidence import backfill_events
        print(backfill_events(a.data_root,out))
if __name__=='__main__':main()
