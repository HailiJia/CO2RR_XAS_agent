#!/usr/bin/env python3
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "tools" / "nersc_workflow_package.py"
s = p.read_text()

if 'name="{name}"' in s and 'script="{script}"' in s and 'key="{key}"' in s and 'jid="{jid}"' in s:
    print("Already patched")
    raise SystemExit(0)

anchor = "        potcar_dir=args.potcar_dir,\n"
idx = s.find(anchor)
if idx < 0:
    raise SystemExit("Could not find potcar_dir anchor")

start = s.rfind("WORKFLOW_XAS.format(", 0, idx)
end = s.find("    ))", idx)
if start < 0 or end < 0:
    raise SystemExit("Could not locate target format block")

insert = (
    '        name="{name}",\n'
    '        script="{script}",\n'
    '        key="{key}",\n'
    '        jid="{jid}",\n'
)
p.write_text(s[:idx] + insert + s[idx:])
print("Patched", p)
