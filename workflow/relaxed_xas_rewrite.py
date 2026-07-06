"""Post-relaxation XAS input regeneration helpers.

This module keeps the production NERSC workflow honest: relaxation must finish
before XAS inputs are regenerated from ``relax/CONTCAR``.  It also provides a
small runtime guard that generated XAS submit scripts can call before launching
FEFF/FDMNES/VASP so stale inputs are detected instead of silently running from
an unrelaxed POSCAR.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Optional


def sha256_file(path: os.PathLike[str] | str) -> str:
    """Return a stable SHA256 digest for a file."""
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require_relaxed_contcar(output_dir: os.PathLike[str] | str) -> Path:
    """Return ``output_dir/relax/CONTCAR`` or raise a clear error."""
    contcar = Path(output_dir).resolve() / "relax" / "CONTCAR"
    if not contcar.is_file() or contcar.stat().st_size == 0:
        raise FileNotFoundError(
            f"Relaxed structure not found: {contcar}. Run/finish relaxation before generating or submitting XAS inputs."
        )
    return contcar


def structure_signature(path: os.PathLike[str] | str) -> Dict[str, object]:
    """Compact provenance signature for an input structure."""
    p = Path(path).resolve()
    return {
        "path": str(p),
        "sha256": sha256_file(p),
        "size_bytes": p.stat().st_size,
    }


def write_xas_structure_signature(output_dir: os.PathLike[str] | str, structure_file: os.PathLike[str] | str) -> Path:
    """Write a plain-text signature used by runtime submit-script guards."""
    out = Path(output_dir).resolve()
    sig = structure_signature(structure_file)
    path = out / ".xas_structure_source.sha256"
    path.write_text(
        "source_path={path}\nsource_sha256={sha256}\nsource_size_bytes={size_bytes}\n".format(**sig),
        encoding="utf-8",
    )
    return path


def _copy_signature_into_xas_dirs(output_dir: os.PathLike[str] | str, signature_file: Path) -> None:
    xas_root = Path(output_dir).resolve() / "xas"
    if not xas_root.is_dir():
        return
    for submit in xas_root.rglob("submit*.sh"):
        shutil.copy2(signature_file, submit.parent / signature_file.name)


def patch_submit_script_for_python3_and_structure_guard(script: os.PathLike[str] | str) -> bool:
    """Patch one generated submit script in place.

    Changes:
    - define ``ISAAC_PYTHON=${ISAAC_PYTHON:-python3}``;
    - replace here-doc calls using bare ``python`` with ``${ISAAC_PYTHON}``;
    - for XAS submit scripts, check ``.xas_structure_source.sha256`` before running.
    """
    path = Path(script)
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    original = text

    if "export ISAAC_PYTHON=" not in text:
        marker = "set -uo pipefail\n"
        if marker in text:
            text = text.replace(marker, marker + "export ISAAC_PYTHON=\"${ISAAC_PYTHON:-python3}\"\n", 1)
        else:
            text = "export ISAAC_PYTHON=\"${ISAAC_PYTHON:-python3}\"\n" + text

    text = text.replace("\npython - <<'PY'", "\n${ISAAC_PYTHON} - <<'PY'")
    text = text.replace("\npython - <<'PYEOF'", "\n${ISAAC_PYTHON} - <<'PYEOF'")
    text = text.replace("\npython - <<PY", "\n${ISAAC_PYTHON} - <<PY")
    text = text.replace("\npython - <<PYEOF", "\n${ISAAC_PYTHON} - <<PYEOF")

    is_xas = "/xas/" in str(path.resolve()).replace("\\", "/") or path.parent.parent.name.endswith("_edge")
    guard = "# XAS relaxed-structure consistency guard"
    if is_xas and guard not in text:
        guard_block = f"""
{guard}
if [ -f .xas_structure_source.sha256 ]; then
  expected_sha=$(awk -F= '/^source_sha256=/ {{print $2}}' .xas_structure_source.sha256 | tail -n 1)
  source_path=$(awk -F= '/^source_path=/ {{print $2}}' .xas_structure_source.sha256 | tail -n 1)
  if [ -z "${{expected_sha}}" ]; then
    echo 'ERROR: .xas_structure_source.sha256 exists but does not contain source_sha256.' >&2
    exit 2
  fi
  if [ ! -f "${{source_path}}" ]; then
    echo "ERROR: Relaxed source structure is missing: ${{source_path}}" >&2
    exit 2
  fi
  actual_sha=$(${{ISAAC_PYTHON}} - <<'PY'
import hashlib, os
p = os.environ.get('source_path')
h = hashlib.sha256()
with open(p, 'rb') as f:
    for chunk in iter(lambda: f.read(1024 * 1024), b''):
        h.update(chunk)
print(h.hexdigest())
PY
)
  if [ "${{actual_sha}}" != "${{expected_sha}}" ]; then
    echo 'ERROR: relax/CONTCAR changed after XAS inputs were generated; regenerate xas/* inputs before running.' >&2
    echo "Expected: ${{expected_sha}}" >&2
    echo "Actual:   ${{actual_sha}}" >&2
    exit 2
  fi
else
  echo 'WARNING: no .xas_structure_source.sha256 found; cannot verify XAS input structure provenance.' >&2
fi
"""
        # Insert after module/env setup and before the first calculation command.
        insert_before = "# =============================================================================\n# ISAAC run metadata: initialized before the calculation starts"
        if insert_before in text:
            text = text.replace(insert_before, guard_block + "\n" + insert_before, 1)
        else:
            text = text.replace("set -uo pipefail\n", "set -uo pipefail\n" + guard_block, 1)

    if text != original:
        path.write_text(text, encoding="utf-8")
        return True
    return False


def patch_submit_scripts(output_dir: os.PathLike[str] | str) -> int:
    """Patch all generated submit scripts below ``output_dir``."""
    root = Path(output_dir).resolve()
    count = 0
    for script in root.rglob("submit*.sh"):
        if patch_submit_script_for_python3_and_structure_guard(script):
            count += 1
    return count


def regenerate_xas_from_relaxed(
    *,
    output_dir: os.PathLike[str] | str,
    nersc_account: Optional[str] = None,
    nersc_queue: Optional[str] = None,
    nersc_nodes: Optional[int] = None,
    nersc_walltime: Optional[str] = None,
    email: Optional[str] = None,
    cluster_radius: float = 6.0,
    edge_override: Optional[str] = None,
    vasp_method: str = "PBE",
    potcar_dir: Optional[str] = None,
) -> Dict[str, object]:
    """Regenerate ``xas/*`` inputs from ``relax/CONTCAR`` after relaxation.

    This deliberately removes stale XAS folders first so no unrelaxed input deck
    can survive from a pre-relaxation generation pass.
    """
    from tools.xas_input_generator import execute_xas_input_generation

    out = Path(output_dir).resolve()
    contcar = require_relaxed_contcar(out)
    xas_root = out / "xas"
    if xas_root.exists():
        shutil.rmtree(xas_root)

    result = execute_xas_input_generation(
        structure_file=str(contcar),
        output_dir=str(out),
        nersc_account=nersc_account,
        nersc_queue=nersc_queue,
        nersc_nodes=nersc_nodes,
        nersc_walltime=nersc_walltime,
        email=email,
        cluster_radius=cluster_radius,
        edge_override=edge_override,
        vasp_method=vasp_method,
        potcar_dir=potcar_dir,
        structure_metadata={"source_file": str(contcar), "source_stage": "post_relaxation_CONTCAR"},
    )
    sig = write_xas_structure_signature(out, contcar)
    _copy_signature_into_xas_dirs(out, sig)
    patched = patch_submit_scripts(out)
    result["xas_structure_source"] = str(contcar)
    result["xas_structure_signature"] = str(sig)
    result["patched_submit_scripts"] = patched
    return result


def submit_scripts(scripts: Iterable[os.PathLike[str] | str], dependency_job_ids: Optional[Iterable[str]] = None) -> Dict[str, str]:
    """Submit scripts with optional afterok dependencies and return script -> job id."""
    deps = [str(j) for j in (dependency_job_ids or []) if j]
    submitted: Dict[str, str] = {}
    for script in scripts:
        script_path = Path(script).resolve()
        cmd = ["sbatch", "--parsable"]
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(deps))
        cmd.append(str(script_path))
        result = subprocess.run(cmd, cwd=str(script_path.parent), text=True, capture_output=True, check=False)
        if result.returncode != 0:
            raise RuntimeError("sbatch failed for {}: {}".format(script_path, result.stderr.strip() or result.stdout.strip()))
        submitted[str(script_path)] = result.stdout.strip().splitlines()[0].split(";")[0]
    return submitted


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate XAS inputs from relax/CONTCAR and patch submit scripts.")
    parser.add_argument("--output-dir", default=".")
    parser.add_argument("--nersc-account")
    parser.add_argument("--nersc-queue")
    parser.add_argument("--nersc-nodes", type=int)
    parser.add_argument("--nersc-walltime")
    parser.add_argument("--email")
    parser.add_argument("--cluster-radius", type=float, default=6.0)
    parser.add_argument("--edge-override")
    parser.add_argument("--vasp-method", default="PBE")
    parser.add_argument("--potcar-dir")
    args = parser.parse_args()
    result = regenerate_xas_from_relaxed(
        output_dir=args.output_dir,
        nersc_account=args.nersc_account,
        nersc_queue=args.nersc_queue,
        nersc_nodes=args.nersc_nodes,
        nersc_walltime=args.nersc_walltime,
        email=args.email,
        cluster_radius=args.cluster_radius,
        edge_override=args.edge_override,
        vasp_method=args.vasp_method,
        potcar_dir=args.potcar_dir,
    )
    print("Regenerated XAS inputs from", result["xas_structure_source"])
    print("Patched submit scripts:", result["patched_submit_scripts"])


if __name__ == "__main__":
    main()
