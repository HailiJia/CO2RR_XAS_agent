"""Regression checks for ISAAC example records.

Run with: python tests/test_isaac_example_records.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.result_parser import execute_result_parsing

EXPECTED_TOP_LEVEL_KEYS = [
    "isaac_record_version",
    "record_id",
    "record_type",
    "record_domain",
    "source_type",
    "timestamps",
    "sample",
    "context",
    "system",
    "computation",
    "measurement",
    "assets",
    "links",
    "descriptors",
]


def _assert_portable_inferred_record(record: dict, software: str) -> None:
    assert list(record.keys()) == EXPECTED_TOP_LEVEL_KEYS
    assert record["isaac_record_version"] == "1.05"

    material = record["sample"]["material"]
    assert material["name"] == "Cu reference model for Cu K-edge XAS"
    assert material["formula"] == "Cu"
    assert f"{software} input files" in material["notes"]
    assert "Sample metadata inferred from POSCAR" in material["notes"]

    serialized = json.dumps(record)
    assert "CHO" not in serialized
    assert "Cu36" not in serialized
    assert "Pure Cu" not in serialized

    run = record["computation"]["run"]
    assert run["scheduler"] == "PBS"
    assert run["command"]
    assert run["cluster"] == "gen8"

    uris = [asset["uri"] for asset in record["assets"]]
    assert uris
    assert all(uri.startswith("https://github.com/HailiJia/CO2RR_XAS_agent/blob/main/") for uri in uris)
    assert not any("/Users/" in uri or "FDMNES_fdm" in uri for uri in uris)

    series = record["measurement"]["series"][0]
    energy = series["independent_variables"][0]["values"]
    signal = series["channels"][0]["values"]
    assert len(energy) == len(signal)
    assert len(energy) > 0


def test_checked_in_example_records_are_portable_and_input_inferred() -> None:
    examples = {
        "FEFF": REPO_ROOT / "example_ouput" / "FEFF" / "isaac_record.json",
        "FDMNES": REPO_ROOT / "example_ouput" / "FDMNES" / "isaac_record.json",
        "VASP": REPO_ROOT / "example_ouput" / "VASP" / "XAS" / "isaac_record.json",
    }
    for software, path in examples.items():
        _assert_portable_inferred_record(json.loads(path.read_text()), software)


def test_generator_infers_example_sample_without_sample_overrides() -> None:
    structure_file = REPO_ROOT / "example_ouput" / "VASP" / "XAS" / "POSCAR"
    sources = {
        "FEFF": REPO_ROOT / "example_ouput" / "FEFF",
        "FDMNES": REPO_ROOT / "example_ouput" / "FDMNES",
        "VASP": REPO_ROOT / "example_ouput" / "VASP" / "XAS",
    }

    with tempfile.TemporaryDirectory() as tmp:
        tmp_root = Path(tmp)
        for software, source_dir in sources.items():
            output_dir = tmp_root / software
            shutil.copytree(source_dir, output_dir)
            result = execute_result_parsing(
                output_dir=str(output_dir),
                software=software,
                structure_file=str(structure_file),
                absorber="Cu",
                edge="K",
                vasp_mode="trace",
                parameters={
                    "asset_base_uri": "https://github.com/HailiJia/CO2RR_XAS_agent/blob/main",
                    "repo_root": str(REPO_ROOT),
                    "prefer_generated_assets": True,
                },
            )
            assert result["status"] == "success"
            _assert_portable_inferred_record(result["isaac_record"], software)




def test_fdmnes_output_validation_checks_energy_window() -> None:
    structure_file = REPO_ROOT / "example_ouput" / "VASP" / "XAS" / "POSCAR"

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "FDMNES"
        shutil.copytree(REPO_ROOT / "example_ouput" / "FDMNES", output_dir)
        result = execute_result_parsing(
            output_dir=str(output_dir),
            software="FDMNES",
            structure_file=str(structure_file),
            absorber="Cu",
            edge="K",
            parameters={"strict_output_validation": True},
        )

    assert result["status"] == "success"
    validation = result["output_validation"]
    assert validation["status"] in {"valid", "warning"}
    energy_check = next(check for check in validation["checks"] if check["name"] == "energy_window_covers_input_range")
    assert energy_check["passed"] is True
    assert result["isaac_record"]["measurement"]["qc"]["output_consistency"]["expected_energy_window"]


def test_strict_output_validation_rejects_truncated_fdmnes_output() -> None:
    structure_file = REPO_ROOT / "example_ouput" / "VASP" / "XAS" / "POSCAR"

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "FDMNES"
        shutil.copytree(REPO_ROOT / "example_ouput" / "FDMNES", output_dir)
        conv = next(output_dir.glob("*_conv.txt"))
        lines = conv.read_text().splitlines()
        # Keep only the low-energy part so the parsed range no longer reaches
        # the FDMNES input Range upper bound.
        conv.write_text("\n".join(lines[:25]) + "\n")
        result = execute_result_parsing(
            output_dir=str(output_dir),
            software="FDMNES",
            structure_file=str(structure_file),
            absorber="Cu",
            edge="K",
            parameters={"strict_output_validation": True},
        )

    assert result["status"] == "error"
    assert result["output_validation"]["status"] == "error"
    assert any("energy range" in message for message in result["output_validation"]["errors"])

if __name__ == "__main__":
    test_checked_in_example_records_are_portable_and_input_inferred()
    test_generator_infers_example_sample_without_sample_overrides()
    test_fdmnes_output_validation_checks_energy_window()
    test_strict_output_validation_rejects_truncated_fdmnes_output()
    print("ISAAC example record regression checks passed")
