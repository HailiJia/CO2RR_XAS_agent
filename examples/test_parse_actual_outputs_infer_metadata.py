import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

"""Parse actual FEFF/FDMNES/VASP outputs and infer absorber/edge from input files.

This test intentionally does not pass absorber="Cu" or edge="K".
The parser infers them from feff.inp, FDMNES *_in.txt/fdmfile.txt, VASP INCAR,
or path fragments such as Cu_K_edge when available.
"""

import json
import shutil

from tools.result_parser import execute_result_parsing

FEFF_DIR = Path("/Users/hjia/Downloads/Cu/Cu/FEFF")
FDMNES_DIR = Path("/Users/hjia/Downloads/Cu/Cu/FDMNES_fdm")
VASP_DIR = Path("/Users/hjia/Downloads/Cu/Cu/VASP_GW/XAS")
BASE = Path("/Users/hjia/Downloads/Cu/Cu")

candidate_structure_files = [
    BASE / "POSCAR",
    BASE / "CONTCAR",
    BASE / "structure" / "POSCAR",
    BASE / "structure" / "CONTCAR",
    BASE / "relax" / "POSCAR",
    BASE / "relax" / "CONTCAR",
    BASE / "VASP_GW" / "POSCAR",
    VASP_DIR / "POSCAR",
    VASP_DIR / "CONTCAR",
]

structure_file = next((p for p in candidate_structure_files if p.exists()), None)
if structure_file is None:
    raise FileNotFoundError("Could not find POSCAR/CONTCAR automatically.")

print("Using structure file:", structure_file)

test_root = Path("generated_outputs/parse_actual_outputs_infer_metadata_test")
if test_root.exists():
    shutil.rmtree(test_root)

# Copy full input/output directories so the inference logic can read feff.inp,
# fdmfile.txt, *_in.txt, INCAR, etc., without touching the original folders.
copy_map = {
    "FEFF": (FEFF_DIR, test_root / "FEFF"),
    "FDMNES": (FDMNES_DIR, test_root / "FDMNES_fdm"),
    "VASP": (VASP_DIR, test_root / "VASP_GW" / "XAS"),
}

for label, (src, dst) in copy_map.items():
    if not src.exists():
        raise FileNotFoundError(src)
    shutil.copytree(src, dst)

results = {}

results["FEFF"] = execute_result_parsing(
    output_dir=str(test_root / "FEFF"),
    software="FEFF",
    structure_file=str(structure_file),
    parameters={"source_output_dir": str(FEFF_DIR)},
    description="Actual FEFF output parsed into ISAAC-ready record with inferred absorber/edge.",
    tags=["actual-output", "FEFF", "metadata-inference"],
)

results["FDMNES"] = execute_result_parsing(
    output_dir=str(test_root / "FDMNES_fdm"),
    software="FDMNES",
    structure_file=str(structure_file),
    parameters={"source_output_dir": str(FDMNES_DIR)},
    description="Actual FDMNES output parsed into ISAAC-ready record with inferred absorber/edge.",
    tags=["actual-output", "FDMNES", "metadata-inference"],
)

results["VASP"] = execute_result_parsing(
    output_dir=str(test_root / "VASP_GW" / "XAS"),
    software="VASP",
    structure_file=str(structure_file),
    parameters={"source_output_dir": str(VASP_DIR)},
    vasp_mode="trace",
    description="Actual VASP XAS OUTCAR parsed into ISAAC-ready record with inferred absorber/edge.",
    tags=["actual-output", "VASP", "metadata-inference"],
)

print("\n" + "=" * 80)
print("ISAAC record summary")
print("=" * 80)

for label, result in results.items():
    print(f"\n{label}")
    print(json.dumps(result, indent=2)[:2000])

    if result.get("status") != "success":
        continue

    record_path = Path(result["record_path"])
    record = json.loads(record_path.read_text())
    spectrum = record["results"]["spectra"][0]["data"]

    print("record_path:", record_path)
    print("absorber:", record["parameters"].get("absorber"))
    print("edge:", record["parameters"].get("edge"))
    print("metadata_inference:", record["parameters"].get("metadata_inference"))
    print("n_energy_points:", len(spectrum.get("energy", [])))

print("\nGenerated ISAAC records:")
for p in sorted(test_root.rglob("isaac_record.json")):
    print(p)
