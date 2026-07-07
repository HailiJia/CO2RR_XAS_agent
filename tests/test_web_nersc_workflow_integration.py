from pathlib import Path

from tools.structure_generator import StructureGenerator
from tools.web_nersc_workflow_integration import prepare_single_structure_workflow_package


def test_prepare_single_structure_workflow_package_generates_relax_and_workflow_scripts(tmp_path):
    structure = StructureGenerator().generate_surface(
        element="Cu",
        facet="111",
        supercell=(1, 1),
        layers=2,
        vacuum=10.0,
    )
    structure.setdefault("metadata", {}).update({"element": "Cu", "facet": "111", "adsorbate": "clean"})

    result = prepare_single_structure_workflow_package(
        structure=structure,
        package_root=tmp_path / "web_xas_agent",
        repo_root=Path.cwd(),
        absorber="Cu",
        edge="K",
        relaxation_settings={"encut": 520, "nsw": 5},
        account="m5268",
        queue="regular",
        nodes=1,
        walltime="00:30:00",
    )

    root = Path(result["package_root"])
    assert (root / "01_structure" / "submit_relax.sh").is_file()
    assert (root / "01_structure" / "make_potcar.sh").is_file()
    assert (root / "workflow_submit.sh").is_file()
    assert (root / "workflow_xas.sh").is_file()
    assert (root / "workflow_status.sh").is_file()
    assert (root / "workflow_restart.sh").is_file()
    assert (root / "workflow_cancel.sh").is_file()
    assert (root / "workflow_state.py").is_file()
    assert (root / "workflow_manifest.json").is_file()
    assert (root / "workflow_state.json").is_file()

    submit_text = (root / "workflow_submit.sh").read_text()
    xas_text = (root / "workflow_xas.sh").read_text()
    relax_text = (root / "01_structure" / "submit_relax.sh").read_text()
    assert "sbatch --parsable 01_structure/submit_relax.sh" in submit_text
    assert "--dependency=afterok" in submit_text
    assert "remote_xas_from_contcar.py" in xas_text
    assert "01_structure/CONTCAR" in xas_text
    assert "./make_potcar.sh" in relax_text
