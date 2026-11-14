"""Build one local, explicitly simulated orchestration-to-inference trace.

The workflow uses a fake SLURM runner and copies the selected record's actual
final geometry into the temporary relaxation output. It does not run HPC or
create an ISAAC record. Existing record spectra and held-out model inference
are then linked by the verified configuration and geometry hashes.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import pandas as pd

from agent.co2rr_xas_agent import process_request
from tools.study_cohort import load_cohort, sha256, verify_export
from tools.spectral_multilabel import SPECIES, predict_records
from workflow.nersc_workflow import NERSCJobManager


class _FixtureRunner:
    def __init__(self, final_contcar):
        self.final_contcar=Path(final_contcar);self.last_script=None;self.job_id=0
    def __call__(self,cmd,**kwargs):
        exe=Path(cmd[0]).name
        if exe=='sbatch':
            self.last_script=cmd[-1];self.job_id+=1
            return subprocess.CompletedProcess(cmd,0,stdout=f'{self.job_id}\n',stderr='')
        if exe=='squeue':
            return subprocess.CompletedProcess(cmd,0,stdout='',stderr='')
        if exe=='sacct':
            if self.last_script and '/relax/' in self.last_script:
                relax=Path(self.last_script).parent
                relax.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(self.final_contcar,relax/'CONTCAR')
            return subprocess.CompletedProcess(cmd,0,stdout='COMPLETED\n',stderr='')
        return subprocess.CompletedProcess(cmd,1,stdout='',stderr='unexpected fixture command')


def create_local_fixture_trace(export_root,training_dir,output_path):
    export=Path(export_root).resolve();training=Path(training_dir).resolve();out=Path(output_path)
    verified=verify_export(export)
    if verified.get('configuration_count')!=87:raise ValueError('Expected the verified 87-configuration export')
    split=pd.read_csv(export/'splits.csv');row=split.iloc[0]
    source=export/'data'/row.configuration/'POSCAR'
    record_path=export/'data'/row.record
    final_contcar=record_path.parent/'layer_5'/'CONTCAR'
    if not source.is_file() or not final_contcar.is_file() or not record_path.is_file():raise FileNotFoundError('Fixture source, final structure, or audited record is absent')
    cohort=load_cohort(export/'data',export/'splits.csv')
    idx=split.index[split.configuration.eq(row.configuration)][0];fold=int(row.fold)
    model=training/f'primary_fold_{fold}.joblib'
    if not model.is_file():raise FileNotFoundError(f'Saved held-out fold model missing: primary_fold_{fold}.joblib')
    # Avoid a "parse results" phrase: the regex planner treats it as the
    # standalone parser action. The explicit argument below disables record
    # parsing while exercising the full relaxation-to-XAS orchestration.
    request=f"Submit and monitor a local XAS workflow from {source} using the test fixture scheduler."
    runner=_FixtureRunner(final_contcar);manager=NERSCJobManager(runner=runner)
    with tempfile.TemporaryDirectory(prefix='co2rr_linked_fixture_') as temp:
        result=process_request(request,output_dir=temp,use_llm=False,
                               job_manager=manager,software='FEFF',
                               parse_when_complete=False,poll_interval=1,
                               nersc_account='fixture',nersc_queue='regular',
                               nersc_nodes=1,nersc_walltime='00:05:00')
        if result.get('status')!='success':raise RuntimeError(f'Fixture workflow failed: {result.get("message")}')
        sig_path=Path(result['inputs']['xas_structure_signature']);sig_text=sig_path.read_text()
        signature_hash=sha256(sig_path)
        relaxed=Path(result['relaxed_structure_file']);relaxed_hash=sha256(relaxed)
        if relaxed_hash!=sha256(final_contcar) or f'source_sha256={relaxed_hash}' not in sig_text:
            raise RuntimeError('Relaxed geometry was not propagated into generated XAS inputs')
        job_events=[dict(job_id=str(j.get('job_id')),stage=j.get('stage'),software=j.get('software'),state=j.get('state'),simulated=True) for j in result.get('submitted_jobs',[])]
    loaded=__import__('joblib').load(model)
    pred,scores=predict_records(loaded,[cohort['records'][idx]],[cohort['populations'][idx]])
    oof=np.load(training/'oof_independent_selected_micro.npz');saved=oof['prediction'][idx]
    if not np.array_equal(pred[0],saved):raise RuntimeError('Inference does not reproduce archived held-out prediction')
    record=json.loads(record_path.read_text());series=record['measurement']['series']
    trace={
      'trace_kind':'local fixture joined to one existing verified calculation; scheduler events are simulated',
      'natural_language_request':'Submit and monitor local workflow from a verified structure; fixture scheduler; no record upload.',
      'planner_action':'nersc_workflow','tool_arguments':{'structure_file':'verified-export/data/<configuration>/POSCAR','software':'FEFF','submit':True,'monitor':True,'parse_when_complete':False,'runner':'fake local scheduler'},
      'configuration':row.configuration,'fold':fold,'structure_hash':row.structure_hash,
      'initial_poscar_sha256':sha256(source),'final_contcar_sha256':sha256(final_contcar),
      'workflow_relaxed_geometry_sha256':relaxed_hash,'generated_xas_structure_signature_sha256':signature_hash,
      'scheduler_events':job_events,'scheduler_events_are_simulated':True,
      'spectra_source':'pre-existing audited FDMNES record for this configuration; no spectrum was generated by fake FEFF jobs',
      'record_path':'verified-export/data/'+row.record,'record_sha256':sha256(record_path),
      'record_series_count':len(series),'model_path':model.name,'model_sha256':sha256(model),
      'species_order':list(map(str,oof['species'])),'heldout_truth':oof['truth'][idx].astype(int).tolist(),
      'heldout_prediction':pred[0].astype(int).tolist(),'archived_oof_prediction':saved.astype(int).tolist(),
      'prediction_replay_identical':True,'record_created_by_fixture':False,'portal_upload_performed':False,
      'credentials_included':False,
    }
    out=out.resolve();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(trace,indent=2)+'\n');return trace


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--export-root',required=True);p.add_argument('--training-dir',required=True);p.add_argument('--output',required=True);a=p.parse_args(argv)
    trace=create_local_fixture_trace(a.export_root,a.training_dir,a.output)
    print(json.dumps({'configuration':trace['configuration'],'simulated_jobs':len(trace['scheduler_events']),'prediction_replay_identical':trace['prediction_replay_identical'],'output':str(Path(a.output).resolve())},indent=2))
if __name__=='__main__':main()
