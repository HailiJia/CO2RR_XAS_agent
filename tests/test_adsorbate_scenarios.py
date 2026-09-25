import numpy as np

from agent.co2rr_xas_agent import LocalIntentParser
from tools.adsorbate_scenarios import generate_adsorbate_scenarios
from tools.structure_generator import StructureGenerator


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


def test_cu_referenced_coverage_and_boundary_avoidance():
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

    n_base = len(structure["atoms"])
    for position in result["positions"][n_base:]:
        assert _margin(position, result["cell"]) >= 0.12 - 1e-10


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
