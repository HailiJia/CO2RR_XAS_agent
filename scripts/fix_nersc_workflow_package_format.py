#!/usr/bin/env python3
"""Fix Bash variable interpolation in tools/nersc_workflow_package.py.

WORKFLOW_XAS is a Python .format(...) template, but it also contains Bash
variables such as ${name}.  Python tries to interpret {name} as a format field
unless those Bash braces are escaped or supplied as literal replacement values.
This patch adds literal replacements for Bash loop variables so workflow_xas.sh
is generated correctly.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "tools" / "nersc_workflow_package.py"

OLD = '''        fdmnes_scf_flag="--fdmnes-scf \\\n    " if args.fdmnes_scf else "",
        potcar_dir=args.potcar_dir,
    ))
'''

NEW = '''        fdmnes_scf_flag="--fdmnes-scf \\\n    " if args.fdmnes_scf else "",
        potcar_dir=args.potcar_dir,
        # Preserve Bash loop variables in WORKFLOW_XAS.  The template is expanded
        # with Python .format(...), so Bash ${name}/${script}/${key}/${jid}
        # must be supplied as literal replacement strings.
        name="{name}",
        script="{script}",
        key="{key}",
        jid="{jid}",
    ))
'''


def main() -> None:
    text = PATH.read_text()
    if NEW.strip() in text:
        print("Already patched:", PATH)
        return
    if OLD not in text:
        raise RuntimeError("Could not find WORKFLOW_XAS.format(...) anchor to patch")
    PATH.write_text(text.replace(OLD, NEW, 1))
    print("Patched:", PATH)


if __name__ == "__main__":
    main()
