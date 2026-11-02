"""Evidence-only execution backfill; missing scheduler accounting is not success."""
import json,re,datetime
from pathlib import Path
from .study_cohort import sha256

def backfill_events(data_root,output_dir):
    root=Path(data_root);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True);attempts={}
    # Scheduler stdout paths establish observed attempts, not the full attempted
    # denominator. BAV and giant numerical output are not execution receipts.
    for path in sorted(root.rglob('slurm-*.out')):
        if 'CO2RR_ML' in path.relative_to(root).parts:continue
        job=re.search(r'slurm-(\d+)\.out$',path.name)
        if not job:continue
        relative=path.relative_to(root);parts=relative.parts
        stage=str(relative.parent);config=stage.split('/XAS/')[0]
        key=job[1];entry=attempts.setdefault(key,dict(workflow_id=config,configuration_id=config,stage=stage,attempt_id=job[1],submission_time=None,start_time=None,end_time=None,resources={},scheduler_terminal_state=None,process_exit_code=None,numerical_output_validity='unmeasured_in_execution_log',record_creation='separate_record_audit',ml_eligibility='separate_record_audit',restart_reason=None,restart_action=None,manual_intervention=None,evidence=[],configuration_aliases=[],stage_aliases=[]))
        if config not in entry['configuration_aliases']:entry['configuration_aliases'].append(config)
        if stage not in entry['stage_aliases']:entry['stage_aliases'].append(stage)
        raw=path.read_bytes();text=raw.decode(errors='replace') if len(raw)<1024*1024 else (raw[:65536]+raw[-65536:]).decode(errors='replace')
        entry['evidence'].append(dict(path=str(relative),sha256=sha256(path),parsed_complete=len(raw)<1024*1024))
        for key,pattern in [('start_time',r'(?im)^Start:\s*(\S+)'),('end_time',r'(?im)^End:\s*(\S+)')]:
            m=re.search(pattern,text)
            if m:entry[key]=m[1]
        m=re.search(r'(?im)^Exit code:\s*(-?\d+)',text)
        if m:entry['process_exit_code']=int(m[1])
        m=re.search(r'(?im)^State[=: ]+(COMPLETED|FAILED|CANCELLED|TIMEOUT|OUT_OF_MEMORY)',text)
        if m:entry['scheduler_terminal_state']=m[1]
        for label,pattern in [('mpi_ranks',r'MPI ranks:\s*(\d+)'),('cpus_per_rank',r'CPUs reserved per rank:\s*(\d+)')]:
            m=re.search(pattern,text)
            if m:entry['resources'][label]=int(m[1])
    rows=sorted(attempts.values(),key=lambda r:(r['configuration_id'],r['stage'],r['attempt_id']))
    with (out/'execution_events.jsonl').open('w') as f:
        for row in rows:f.write(json.dumps(row,sort_keys=True)+'\n')
    summary=dict(deduplication='scheduler job ID across copied folder aliases; all source paths retained',observed_scheduler_stdout_files=sum(len(r['evidence']) for r in rows),observed_attempts_with_surviving_scheduler_stdout=len(rows),observed_workflows=len({r['workflow_id'] for r in rows}),attempted_run_denominator_reconciled=False,first_pass_success_rate=None,recovery_success_rate=None,retry_cost=None,intervention_rate=None,reason='Surviving stdout does not establish complete submissions/accounting, restart intent, resource-hours or manual actions. Obtain sacct/exported workflow-state history before computing rates.')
    (out/'execution_summary.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

def append_event(path,event):
    """Append a hash-chained observed event; unknown fields remain explicit nulls.

    A local per-workflow log is intended for a single writer. Distributed jobs
    should use separate attempt logs and reconcile them, not share one NFS file.
    """
    import hashlib
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for field in ['workflow_id','configuration_id','attempt_id','event_type']:
        if not event.get(field):raise ValueError('Missing execution-event '+field)
    prior=None
    if path.exists():
        rows=read_events(path);prior=rows[-1]['event_sha256'] if rows else None
    row={key:None for key in ['submission_time','start_time','end_time','scheduler_terminal_state','process_exit_code','output_validator','record_created','ml_eligible','restart_reason','restart_action','manual_intervention']}
    row.update(resources={},event_observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    row.update(event)
    row['previous_sha256']=prior
    row['event_sha256']=hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with path.open('a') as f:f.write(json.dumps(row,sort_keys=True)+'\n')
    return row

def read_events(path):
    import hashlib
    rows=[];prior=None
    for line in Path(path).read_text().splitlines():
        row=json.loads(line);claimed=row.pop('event_sha256')
        if row.get('previous_sha256')!=prior or hashlib.sha256(json.dumps(row,sort_keys=True,separators=(',',':')).encode()).hexdigest()!=claimed:raise ValueError('Execution log chain/hash mismatch')
        row['event_sha256']=claimed;rows.append(row);prior=claimed
    return rows

def record_event_from_env(event):
    import os
    path=os.environ.get('CO2RR_EXECUTION_EVENT_LOG')
    if not path:return None
    observed=dict(event)
    observed['workflow_id']=os.environ.get('CO2RR_WORKFLOW_ID',observed.get('workflow_id'))
    observed['configuration_id']=os.environ.get('CO2RR_CONFIGURATION_ID',observed.get('configuration_id'))
    return append_event(path,observed)
