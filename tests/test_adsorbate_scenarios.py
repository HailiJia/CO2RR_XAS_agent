import numpy as np

from agent.co2rr_xas_agent import LocalIntentParser
from tools.adsorbate_scenarios import generate_adsorbate_scenarios
from tools.structure_generator import StructureGenerator, execute_structure_generation


def _cuau_5layer():
    gen = StructureGenerator()
    # Auto Cu/Au matching gives 9 Cu and 8 Au repeats along x.
    # rows_per_side=4 therefore gives 36 top-layer Cu sites.
    return gen.generate_interface(
        "Cu",
        "Au",
        facet1="111",
        facet2="111",
        supercell=(3, 4),
        layers1=5,
        layers2=5,
        vacuum=15.0,
    )


def _margin(position, cell):
    frac = np.asarray(position) @ np.linalg.inv(np.asarray(cell))
    x, y = frac[0] % 1.0, frac[1] % 1.0
    return min(x, 1.0 - x, y, 1.0 - y)


def test_local_parser_extracts_coverage_and_distribution_controls():
    parser = LocalIntentParser()
    intent = parser.parse(
        "Generate CO on Cu Au (111) at 0.25 ML, interface-biased, "
        "coverage based on Cu, boundary margin 0.15"
    )
    assert intent.parameters["coverage"] == 0.25
    assert intent.parameters["distribution"] == "interface_biased"
    assert intent.parameters["coverage_basis"] == "Cu"
    assert intent.parameters["boundary_margin"] == 0.15


def test_local_parser_recognizes_all_reasonable_scenarios():
    parser = LocalIntentParser()
    intent = parser.parse(
        "Generate all reasonable scenarios for OCCO on Cu Au (111) at 0.056 ML"
    )
    assert intent.parameters["all_scenarios"] is True
    assert intent.parameters["coverage"] == 0.056


def test_cu_referenced_coverage_count():
    structure = _cuau_5layer()
    generated = generate_adsorbate_scenarios(
        structure,
        "CO",
        coverage=0.25,
        distribution="uniform",
        coverage_basis="Cu",
        boundary_margin=0.12,
    )
    assert len(generated) == 1
    result = generated[0]
    assert result["metadata"]["coverage_denominator"] == 36
    assert result["metadata"]["n_adsorbates"] == 9
    assert result["metadata"]["coverage_actual"] == 0.25


def test_interface_and_interior_bias_are_distinct():
    structure = _cuau_5layer()
    interface = generate_adsorbate_scenarios(
        structure, "CO", 0.111, "interface_biased", coverage_basis="Cu"
    )[0]
    interior = generate_adsorbate_scenarios(
        structure, "CO", 0.111, "interior_biased", coverage_basis="Cu"
    )[0]

    interface_mean = np.mean(
        [site["interface_distance"] for site in interface["metadata"]["selected_sites"]]
    )
    interior_mean = np.mean(
        [site["interface_distance"] for site in interior["metadata"]["selected_sites"]]
    )
    assert interface_mean < interior_mean


def test_all_occ_o_scenarios_include_internal_cross_interface_pair():
    structure = _cuau_5layer()
    generated = generate_adsorbate_scenarios(
        structure,
        "OCCO",
        coverage=0.056,
        all_scenarios=True,
        coverage_basis="Cu",
        boundary_margin=0.12,
    )
    by_name = {item["metadata"]["scenario_name"]: item for item in generated}
    assert {"CuCu_uniform", "CuCu_interface", "AuCu_interface"} <= set(by_name)

    cross = by_name["AuCu_interface"]
    selected = cross["metadata"]["selected_sites"]
    assert len(selected) == 1
    assert selected[0]["composition"] == "AuCu"

    n_base = len(structure["atoms"])
    added = cross["positions"][n_base:]
    # OCCO order is C O C O.
    assert np.linalg.norm(added[0] - added[2]) < 1.6
    for position in added:
        assert _margin(position, cross["cell"]) >= 0.12 - 1e-10


def test_cho_and_coh_have_different_h_connectivity():
    structure = _cuau_5layer()
    cho = generate_adsorbate_scenarios(
        structure, "CHO", coverage=1.0 / 36.0, distribution="uniform", coverage_basis="Cu"
    )[0]
    coh = generate_adsorbate_scenarios(
        structure, "COH", coverage=1.0 / 36.0, distribution="uniform", coverage_basis="Cu"
    )[0]

    n = len(structure["atoms"])
    cho_added = cho["positions"][n:]
    coh_added = coh["positions"][n:]

    # C O H ordering in both; CHO has H on C, COH has H on O.
    assert abs(np.linalg.norm(cho_added[2] - cho_added[0]) - 1.10) < 1e-6
    assert abs(np.linalg.norm(coh_added[2] - coh_added[1]) - 0.98) < 1e-6


def test_cooh_all_scenarios_cover_cu_au_and_auc_u_bindings():
    structure = _cuau_5layer()
    generated = generate_adsorbate_scenarios(
        structure,
        "COOH",
        coverage=0.056,
        all_scenarios=True,
        coverage_basis="Cu",
    )
    names = {item["metadata"]["scenario_name"] for item in generated}
    assert {"Cu_uniform", "Cu_interface", "Au_uniform", "AuCu_interface"} <= names


def _pbc_distance(a, b, cell):
    frac = (np.asarray(b) - np.asarray(a)) @ np.linalg.inv(np.asarray(cell))
    frac[:2] -= np.round(frac[:2])
    return np.linalg.norm(frac @ np.asarray(cell))


def test_local_parser_extracts_multiple_coverages():
    parser = LocalIntentParser()
    intent = parser.parse(
        "Generate CO on Cu Au (111) at coverages 0.056, 0.111 and 0.25 ML, uniform"
    )
    assert intent.parameters["coverages"] == [0.056, 0.111, 0.25]
    assert intent.parameters["distribution"] == "uniform"


def test_high_coverage_occo_is_one_core_plus_spectator_co():
    structure = _cuau_5layer()
    result = generate_adsorbate_scenarios(
        structure,
        "OCCO",
        coverage=0.222,
        distribution="uniform",
        coverage_basis="Cu",
    )[0]

    assert result["metadata"]["n_occo"] == 1
    assert result["metadata"]["n_spectator_co"] == 6
    assert result["metadata"]["coverage_denominator"] == 36
    assert abs(result["metadata"]["coverage_actual"] - 8.0 / 36.0) < 1e-12

    n_base = len(structure["atoms"])
    added_atoms = result["atoms"][n_base:]
    assert added_atoms.count("C") == 8
    assert added_atoms.count("O") == 8
    assert sum(site["role"] == "OCCO_core" for site in result["metadata"]["selected_sites"]) == 1
    assert sum(site["role"] == "spectator_CO" for site in result["metadata"]["selected_sites"]) == 6


def test_cross_interface_cooh_hits_intended_au_c_and_cu_o_contacts():
    structure = _cuau_5layer()
    generated = generate_adsorbate_scenarios(
        structure,
        "COOH",
        coverage=1.0 / 36.0,
        distribution="cross_interface",
        coverage_basis="Cu",
        boundary_margin=0.12,
    )
    result = generated[0]
    site = result["metadata"]["selected_sites"][0]
    i, j = site["indices"]
    pair = [(structure["atoms"][i], structure["positions"][i]),
            (structure["atoms"][j], structure["positions"][j])]
    au = next(pos for element, pos in pair if element == "Au")
    cu = next(pos for element, pos in pair if element == "Cu")

    n = len(structure["atoms"])
    C, Ocar, Oh, H = result["positions"][n:n + 4]
    assert abs(_pbc_distance(C, au, result["cell"]) - 2.10) < 1e-6
    assert abs(_pbc_distance(Ocar, cu, result["cell"]) - 2.10) < 1e-6
    assert abs(np.linalg.norm(Ocar - C) - 1.23) < 1e-6
    assert abs(np.linalg.norm(Oh - C) - 1.33) < 1e-6
    assert abs(np.linalg.norm(H - Oh) - 0.99) < 1e-6


def test_cu_cu_cooh_is_asymmetric_bidentate():
    structure = _cuau_5layer()
    result = generate_adsorbate_scenarios(
        structure,
        "COOH",
        coverage=1.0 / 36.0,
        distribution="uniform",
        coverage_basis="Cu",
    )[0]
    site = result["metadata"]["selected_sites"][0]
    i, j = site["indices"]
    assert structure["atoms"][i] == "Cu"
    assert structure["atoms"][j] == "Cu"

    cu1 = structure["positions"][i]
    cu2 = structure["positions"][j]
    n = len(structure["atoms"])
    C, Ocar, _, _ = result["positions"][n:n + 4]

    c_contacts = sorted([
        _pbc_distance(C, cu1, result["cell"]),
        _pbc_distance(C, cu2, result["cell"]),
    ])
    o_contacts = sorted([
        _pbc_distance(Ocar, cu1, result["cell"]),
        _pbc_distance(Ocar, cu2, result["cell"]),
    ])

    assert abs(c_contacts[0] - 1.95) < 1e-6
    assert abs(o_contacts[0] - 2.00) < 1e-6
    assert abs(o_contacts[1] - 2.28) < 1e-6
    assert abs(np.linalg.norm(Ocar - C) - 1.26) < 1e-6


def test_oh_uses_fcc_hollows():
    structure = _cuau_5layer()
    result = generate_adsorbate_scenarios(
        structure,
        "OH",
        coverage=0.111,
        distribution="uniform",
        coverage_basis="Cu",
        boundary_margin=0.12,
    )[0]
    assert result["metadata"]["n_adsorbates"] == 4
    assert all(site["kind"] == "fcc" for site in result["metadata"]["selected_sites"])


def test_dense_co_075_is_supported_under_pbc():
    structure = _cuau_5layer()
    result = generate_adsorbate_scenarios(
        structure,
        "CO",
        coverage=0.75,
        distribution="uniform",
        coverage_basis="Cu",
        boundary_margin=0.12,
    )[0]
    assert result["metadata"]["n_adsorbates"] == 27
    assert abs(result["metadata"]["coverage_actual"] - 0.75) < 1e-12
    n = len(structure["atoms"])
    assert result["atoms"][n:].count("C") == 27
    assert result["atoms"][n:].count("O") == 27



def test_written_scenario_variants_have_unique_structure_ids(tmp_path):
    result = execute_structure_generation(
        mode="generate",
        metal1="Cu",
        metal2="Au",
        facet1="111",
        facet2="111",
        adsorbate="CO",
        supercell=(3, 4),
        layers=5,
        coverages=[0.056, 0.111],
        distribution="uniform",
        coverage_basis="Cu",
        output_dir=str(tmp_path),
    )
    assert result["status"] == "success"
    assert len(result["structures"]) == 2

    ids = [metadata["structure_id"] for metadata in result["structures"]]
    assert len(set(ids)) == 2
    assert any("cov0p056" in value for value in ids)
    assert any("cov0p111" in value for value in ids)

    for metadata in result["structures"]:
        nested = metadata["adsorbate_metadata"]
        assert nested["distribution"] == "uniform"
        assert nested["coverage_actual"] == metadata["coverage_actual"]

    paths = [entry["poscar"] for entry in result["files"]]
    assert any("cov_0p056_uniform" in path for path in paths)
    assert any("cov_0p111_uniform" in path for path in paths)
