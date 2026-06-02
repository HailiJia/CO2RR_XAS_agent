"""Tests for CO2RR ISAAC XAS ML skill helpers."""

import json
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skills.skill_registry import SkillRegistry
from tools.ml_xas_workflow import execute_ml_dataset_assembly, load_isaac_records, build_spectrum_table


def test_ml_skill_is_registered_and_validates_parameters():
    registry = SkillRegistry(REPO_ROOT / "skills")
    assert "co2rr_isaac_xas_ml" in registry.list_skills()
    errors = registry.validate_parameters(
        "co2rr_isaac_xas_ml",
        {
            "isaac_record_paths": [str(REPO_ROOT / "example_ouput" / "FEFF" / "isaac_record.json")],
            "output_dir": "generated_outputs/ml_test",
        },
    )
    assert errors == []


def test_ml_ingestion_builds_spectrum_rows_from_isaac_records():
    records = load_isaac_records([str(REPO_ROOT / "example_ouput" / "FEFF" / "isaac_record.json")])
    rows = build_spectrum_table(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["record_id"]
    assert row["absorbing_element"] == "Cu"
    assert row["edge"] == "K"
    assert len(row["energy"]) == len(row["intensity"])


def test_ml_dataset_assembly_writes_tables_and_training_record():
    inputs = [
        str(REPO_ROOT / "example_ouput" / "FEFF" / "isaac_record.json"),
        str(REPO_ROOT / "example_ouput" / "FDMNES" / "isaac_record.json"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        result = execute_ml_dataset_assembly(inputs, output_dir=tmp)
        assert result["status"] == "success"
        spectrum_path = Path(result["files"]["spectrum_table"])
        condition_path = Path(result["files"]["condition_table"])
        training_path = Path(result["files"]["ml_training_record"])
        assert spectrum_path.exists()
        assert condition_path.exists()
        assert training_path.exists()
        spectra = json.loads(spectrum_path.read_text())
        training = json.loads(training_path.read_text())

    assert len(spectra) == 2
    assert all(row["valid_for_ml"] for row in spectra)
    assert "features" in spectra[0]
    assert training["record_type"] == "ml_training"
    assert training["record_domain"] == "machine_learning"
    assert training["dataset"]["spectrum_count"] == 2
    assert training["evaluation"]["split_strategy"] == "grouped_cv"


if __name__ == "__main__":
    test_ml_skill_is_registered_and_validates_parameters()
    test_ml_ingestion_builds_spectrum_rows_from_isaac_records()
    test_ml_dataset_assembly_writes_tables_and_training_record()
    print("CO2RR ISAAC XAS ML workflow tests passed")
