"""Portable archived-result validation; optionally audit a relocated export."""
from pathlib import Path
import argparse,sys
BASE=Path(__file__).resolve().parent
sys.path.insert(0,str(BASE.parents[1]))
from tools.study_replay import replay_oof
from tools.study_cohort import verify_export,load_cohort
p=argparse.ArgumentParser();p.add_argument('--data-root');p.add_argument('--output-dir',required=True);p.add_argument('--split-manifest',default=str(BASE/'cv_manifest.csv'));p.add_argument('--audit-only',action='store_true');p.add_argument('--export-root')
a=p.parse_args()
if a.export_root:verify_export(a.export_root)
if a.data_root:
    c=load_cohort(a.data_root,a.split_manifest);print('Audited configurations:',len(c['records']))
if not a.audit_only:print(replay_oof(BASE,a.output_dir,a.split_manifest).to_string(index=False))
