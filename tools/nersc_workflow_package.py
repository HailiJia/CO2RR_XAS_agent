#!/usr/bin/env python3
"""Generate a restartable NERSC workflow layer for relax -> XAS.

The generated workflow is intentionally NERSC-side and Slurm-driven. Streamlit can
upload a package, start `workflow_submit.sh`, and later inspect live status from
`workflow_state.json` plus Slurm, but the workflow itself does not require the web
app to stay open.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

PYTHON_BOOTSTRAP = r'''
module load python/3.11 2>/dev/null || module load python 2>/dev/null || true
PYTHON_CMD="$(printenv CO2RR_NERSC_PYTHON || true)"
if [ -z "$PYTHON_CMD" ]; then
  for py in python3.11 python3.10 python3.9 python3 python; do
    if command -v "$py" >/dev/null 2>&1; then
      if "$py" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 8) else 1)' >/dev/null 2>&1; then
        PYTHON_CMD="$py"
        break
      fi
    fi
  done
fi
if [ -z "$PYTHON_CMD" ]; then
  echo "ERROR: Python >= 3.8 is required. Load python/3.11 or set CO2RR_NERSC_PYTHON." >&2
  exit 90
fi
echo "Using Python: $PYTHON_CMD ($($PYTHON_CMD --version 2>&1))"
'''.strip()

STATE_PY = r'''#!/usr/bin/env python3
import argparse, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path

STATE = Path('workflow_state.json')


def now():
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def load():
    if STATE.exists():
        try:
            return json.loads(STATE.read_text())
        except Exception:
            pass
    return {'stage': 'new', 'jobs': {}, 'events': []}


def save(data):
    data['updated_utc'] = now()
    STATE.write_text(json.dumps(data, indent=2))


def event(data, message):
    data.setdefault('events', []).append({'time_utc': now(), 'message': message})
    data['events'] = data['events'][-200:]


def slurm_state(job_id):
    if not job_id:
        return 'UNKNOWN'
    cmds = [
        ['squeue', '-j', str(job_id), '-h', '-o', '%T'],
        ['sacct', '-j', str(job_id), '--format=State', '-P', '-n'],
    ]
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, universal_newlines=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            out = ''
        if out:
            return out.splitlines()[0].split('|')[0].strip().split()[0]
    return 'UNKNOWN'


def refresh(data):
    for job in data.get('jobs', {}).values():
        jid = job.get('job_id')
        if jid:
            job['state'] = slurm_state(jid)
            job['last_checked_utc'] = now()
    states = {name: job.get('state') for name, job in data.get('jobs', {}).items()}
    xas_names = [k for k in states if k.startswith('xas_')]
    if states.get('relax') == 'COMPLETED' and xas_names and all(states.get(k) == 'COMPLETED' for k in xas_names):
        data['stage'] = 'completed'
    elif any(str(v).startswith(('FAILED', 'CANCELLED', 'CANCELED', 'TIMEOUT', 'OUT_OF_MEMORY')) for v in states.values()):
        data['stage'] = 'needs_restart_or_review'
    elif xas_names:
        data['stage'] = 'xas_submitted'
    elif states.get('relax') == 'COMPLETED':
        data['stage'] = 'relax_completed'
    elif 'relax' in states:
        data['stage'] = 'relax_submitted'
    return data


def should_submit(data, name, resubmit_failed=False):
    job = data.get('jobs', {}).get(name)
    if not job:
        return 0
    state = slurm_state(job.get('job_id'))
    job['state'] = state
    if state in {'PENDING', 'RUNNING', 'CONFIGURING', 'COMPLETING'}:
        return 1
    if state == 'COMPLETED':
        return 1
    failed = state.startswith(('FAILED', 'CANCELLED', 'CANCELED', 'TIMEOUT', 'OUT_OF_MEMORY')) or state == 'UNKNOWN'
    return 0 if (failed and resubmit_failed) else 1


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='cmd')
    sub.add_parser('init')
    j = sub.add_parser('set-job')
    j.add_argument('--name', required=True)
    j.add_argument('--job-id', required=True)
    j.add_argument('--script', required=True)
    j.add_argument('--stage', default='')
    j.add_argument('--dependency', default='')
    sub.add_parser('refresh')
    ss = sub.add_parser('should-submit')
    ss.add_argument('--name', required=True)
    ss.add_argument('--resubmit-failed', action='store_true')
    sub.add_parser('cancel')
    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        raise SystemExit(2)

    data = load()
    if args.cmd == 'init':
        data.setdefault('created_utc', now())
        data.setdefault('jobs', {})
        data.setdefault('events', [])
        data['stage'] = data.get('stage') if data.get('stage') != 'new' else 'prepared'
        event(data, 'initialized workflow state')
        save(data)
    elif args.cmd == 'set-job':
        data.setdefault('jobs', {})[args.name] = {
            'job_id': str(args.job_id),
            'script': args.script,
            'dependency': args.dependency,
            'state': slurm_state(args.job_id),
            'submitted_utc': now(),
        }
        if args.stage:
            data['stage'] = args.stage
        event(data, 'registered {0} job {1}'.format(args.name, args.job_id))
        save(data)
    elif args.cmd == 'refresh':
        data = refresh(data)
        save(data)
        print(json.dumps(data, indent=2))
    elif args.cmd == 'should-submit':
        code = should_submit(data, args.name, args.resubmit_failed)
        save(data)
        raise SystemExit(code)
    elif args.cmd == 'cancel':
        data = refresh(data)
        for name, job in data.get('jobs', {}).items():
            jid = job.get('job_id')
            state = job.get('state', '')
            if jid and state not in {'COMPLETED', 'FAILED', 'CANCELLED', 'CANCELED'}:
                subprocess.call(['scancel', str(jid)])
                job['state'] = 'CANCEL_REQUESTED'
                event(data, 'cancel requested for {0} job {1}'.format(name, jid))
        data['stage'] = 'cancel_requested'
        save(data)
        print(json.dumps(data, indent=2))

if __name__ == '__main__':
    main()
'''

WORKFLOW_SUBMIT = '''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
{python_bootstrap}
"$PYTHON_CMD" workflow_state.py init

if [ ! -f 01_structure/submit_relax.sh ]; then
  echo "ERROR: missing 01_structure/submit_relax.sh" >&2
  exit 2
fi

if "$PYTHON_CMD" workflow_state.py should-submit --name relax; then
  RELAX_JOB_ID=$(sbatch --parsable 01_structure/submit_relax.sh)
  "$PYTHON_CMD" workflow_state.py set-job --name relax --job-id "${RELAX_JOB_ID}" --script 01_structure/submit_relax.sh --stage relax_submitted
else
  RELAX_JOB_ID=$("$PYTHON_CMD" -c "import json; print(json.load(open('workflow_state.json'))['jobs']['relax']['job_id'])")
  echo "Relaxation already submitted or completed: ${RELAX_JOB_ID}"
fi

if "$PYTHON_CMD" workflow_state.py should-submit --name workflow_xas; then
  XAS_DRIVER_JOB_ID=$(sbatch --parsable --dependency=afterok:${RELAX_JOB_ID} workflow_xas.sh)
  "$PYTHON_CMD" workflow_state.py set-job --name workflow_xas --job-id "${XAS_DRIVER_JOB_ID}" --script workflow_xas.sh --dependency afterok:${RELAX_JOB_ID} --stage relax_submitted
else
  echo "workflow_xas job already submitted or completed."
fi

"$PYTHON_CMD" workflow_state.py refresh
'''

WORKFLOW_XAS = '''#!/bin/bash
#SBATCH -J workflow_xas
#SBATCH -q regular
#SBATCH -A {account}
#SBATCH -C cpu
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -o workflow_xas-%j.out
#SBATCH -e workflow_xas-%j.err
set -euo pipefail
cd "$(dirname "$0")"
{python_bootstrap}
"$PYTHON_CMD" workflow_state.py refresh || true

if [ ! -f 01_structure/CONTCAR ]; then
  echo "ERROR: relaxation dependency succeeded but 01_structure/CONTCAR is missing" >&2
  exit 3
fi

if [ ! -f 02_XAS/.post_relax_generated ]; then
  "$PYTHON_CMD" {repo_root}/tools/remote_xas_from_contcar.py \
    --package-root . \
    --contcar 01_structure/CONTCAR \
    --absorber "{absorber}" \
    --edge "{edge}" \
    --vasp-method "{vasp_method}" \
    --cluster-radius {cluster_radius} \
    --fdmnes-method "{fdmnes_method}" \
    {fdmnes_scf_flag}\
    --potcar-dir "{potcar_dir}" \
    --account "{account}" \
    --queue "{queue}" \
    --nodes {nodes} \
    --walltime "{walltime}" \
    --email "{email}"
fi

for name in VASP FDMNES FEFF; do
  script="02_XAS/${{name}}/submit.sh"
  key="xas_${{name}}"
  if [ -f "${{script}}" ]; then
    if "$PYTHON_CMD" workflow_state.py should-submit --name "${{key}}" --resubmit-failed; then
      jid=$(sbatch --parsable "${{script}}")
      "$PYTHON_CMD" workflow_state.py set-job --name "${{key}}" --job-id "${{jid}}" --script "${{script}}" --stage xas_submitted
    else
      echo "${{key}} already submitted/completed; skipping"
    fi
  fi
done
"$PYTHON_CMD" workflow_state.py refresh
'''

WORKFLOW_RESTART = '''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
{python_bootstrap}
"$PYTHON_CMD" workflow_state.py init
"$PYTHON_CMD" workflow_state.py refresh || true

relax_state=$("$PYTHON_CMD" -c "import json; d=json.load(open('workflow_state.json')); print(d.get('jobs',{}).get('relax',{}).get('state',''))")
if [ -z "${relax_state}" ] || [[ "${relax_state}" =~ ^(FAILED|CANCELLED|CANCELED|TIMEOUT|OUT_OF_MEMORY|UNKNOWN)$ ]]; then
  echo "Restarting/submitting relaxation"
  jid=$(sbatch --parsable 01_structure/submit_relax.sh)
  "$PYTHON_CMD" workflow_state.py set-job --name relax --job-id "${jid}" --script 01_structure/submit_relax.sh --stage relax_submitted
  xjid=$(sbatch --parsable --dependency=afterok:${jid} workflow_xas.sh)
  "$PYTHON_CMD" workflow_state.py set-job --name workflow_xas --job-id "${xjid}" --script workflow_xas.sh --dependency afterok:${jid} --stage relax_submitted
elif [ "${relax_state}" = "COMPLETED" ]; then
  echo "Relaxation completed; continuing or restarting XAS stage"
  jid=$(sbatch --parsable workflow_xas.sh)
  "$PYTHON_CMD" workflow_state.py set-job --name workflow_xas --job-id "${jid}" --script workflow_xas.sh --stage relax_completed
else
  echo "Relaxation is ${relax_state}; ensure workflow_xas dependency exists"
  relax_jid=$("$PYTHON_CMD" -c "import json; print(json.load(open('workflow_state.json')).get('jobs',{}).get('relax',{}).get('job_id',''))")
  if [ -n "${relax_jid}" ] && "$PYTHON_CMD" workflow_state.py should-submit --name workflow_xas; then
    xjid=$(sbatch --parsable --dependency=afterok:${relax_jid} workflow_xas.sh)
    "$PYTHON_CMD" workflow_state.py set-job --name workflow_xas --job-id "${xjid}" --script workflow_xas.sh --dependency afterok:${relax_jid} --stage relax_submitted
  fi
fi
"$PYTHON_CMD" workflow_state.py refresh
'''

WORKFLOW_STATUS = '''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
{python_bootstrap}
"$PYTHON_CMD" workflow_state.py refresh
'''

WORKFLOW_CANCEL = '''#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
{python_bootstrap}
"$PYTHON_CMD" workflow_state.py cancel
'''


def render_script(template: str) -> str:
    return template.replace("{python_bootstrap}", PYTHON_BOOTSTRAP)


def write_executable(path: Path, content: str) -> None:
    path.write_text(content.rstrip() + "\n")
    path.chmod(0o755)


def ensure_manifest(package_root: Path, args: argparse.Namespace) -> None:
    manifest = {
        "workflow_engine": "nersc_slurm_dependency",
        "workflow_scripts": {
            "submit": "workflow_submit.sh",
            "xas": "workflow_xas.sh",
            "restart": "workflow_restart.sh",
            "status": "workflow_status.sh",
            "cancel": "workflow_cancel.sh",
            "state_helper": "workflow_state.py",
        },
        "python_selection": "CO2RR_NERSC_PYTHON or python/3.11 module or python >= 3.8",
        "xas_input_source": "post_relaxation_CONTCAR",
        "relax_submit": "01_structure/submit_relax.sh",
        "relaxed_structure": "01_structure/CONTCAR",
        "settings": vars(args),
    }
    (package_root / "workflow_manifest.json").write_text(json.dumps(manifest, indent=2))
    (package_root / "workflow_state.json").write_text(json.dumps({
        "stage": "prepared",
        "jobs": {},
        "events": [],
        "manifest": "workflow_manifest.json",
    }, indent=2))


def create_workflow(package_root: Path, args: argparse.Namespace) -> Dict[str, str]:
    package_root.mkdir(parents=True, exist_ok=True)
    write_executable(package_root / "workflow_state.py", STATE_PY)
    write_executable(package_root / "workflow_submit.sh", render_script(WORKFLOW_SUBMIT))
    write_executable(package_root / "workflow_xas.sh", render_script(WORKFLOW_XAS).format(
        repo_root=args.repo_root,
        account=args.account,
        queue=args.queue,
        nodes=args.nodes,
        walltime=args.walltime,
        email=args.email or "",
        absorber=args.absorber,
        edge=args.edge,
        vasp_method=args.vasp_method,
        cluster_radius=args.cluster_radius,
        fdmnes_method=args.fdmnes_method,
        fdmnes_scf_flag="--fdmnes-scf \\\n    " if args.fdmnes_scf else "",
        potcar_dir=args.potcar_dir,
    ))
    write_executable(package_root / "workflow_restart.sh", render_script(WORKFLOW_RESTART))
    write_executable(package_root / "workflow_status.sh", render_script(WORKFLOW_STATUS))
    write_executable(package_root / "workflow_cancel.sh", render_script(WORKFLOW_CANCEL))
    ensure_manifest(package_root, args)
    return {"status": "success", "package_root": str(package_root), "submit": str(package_root / "workflow_submit.sh")}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--package-root", default="generated_outputs/web_xas_agent")
    p.add_argument("--repo-root", default=os.environ.get("CO2RR_AGENT_REPO", str(Path.cwd())))
    p.add_argument("--absorber", default="Cu")
    p.add_argument("--edge", default="K")
    p.add_argument("--vasp-method", default="PBE")
    p.add_argument("--cluster-radius", type=float, default=6.0)
    p.add_argument("--fdmnes-method", default="Green")
    p.add_argument("--fdmnes-scf", action="store_true")
    p.add_argument("--potcar-dir", default="/global/common/software/nersc9/vasp/dependencies/pseudopotentials/PBE/potpaw_PBE")
    p.add_argument("--account", default=os.environ.get("CO2RR_NERSC_ACCOUNT", os.environ.get("NERSC_ACCOUNT", "m5268")))
    p.add_argument("--queue", default="regular")
    p.add_argument("--nodes", type=int, default=1)
    p.add_argument("--walltime", default="02:00:00")
    p.add_argument("--email", default="")
    args = p.parse_args()
    print(json.dumps(create_workflow(Path(args.package_root).resolve(), args), indent=2))


if __name__ == "__main__":
    main()
