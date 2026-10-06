import json
from copy import deepcopy
from itertools import product

import numpy as np
import pytest

from tools.contact_checks import minimum_image_vectors, minimum_pair, validate_geometry
from tools.dataset_labels import annotate_structure, relaxed_metadata, write_dataset_manifest
from tools.structure_generator import StructureGenerator
from tools.stripe_interface_patch import install
from tools.utils import read_poscar


def test_short_covalent_bond_does_not_hide_a_metal_clash():
    atoms = ['C', 'H', 'Cu', 'Au']
    pos = [[0, 0, 3], [0, 0, 4.09], [5, 5, 0], [6.4, 5, 0]]
    report = minimum_pair(atoms, pos)
    assert report['severity'] == 'error'
    assert any(issue['pair'] == [2, 3] for issue in report['issues'])
    assert report['minimum_distance'] == pytest.approx(1.09)


def test_periodic_edge_clashes_and_vacuum_are_separate():
    report = validate_geometry(['Cu', 'Cu'], [[.1, 0, 0], [9.9, 0, 0]], np.diag([10., 10., 20.]))
    assert report['severity'] == 'error'
    assert report['distance'] == pytest.approx(.2)
    report = validate_geometry(['Cu', 'Cu'], [[0, 0, .1], [0, 0, 19.9]], np.diag([10., 10., 20.]))
    assert report['severity'] == 'info'


def test_minimum_image_for_skew_cell_matches_image_enumeration():
    cell = np.array([[5., 0, 0], [4.4, 1.2, 0], [0, 0, 20.]])
    vector = np.array([3.6, .8, .3])
    actual = np.linalg.norm(minimum_image_vectors(vector, cell))
    expected = min(np.linalg.norm(vector - np.array([i, j, 0]) @ cell) for i, j in product(range(-8, 9), repeat=2))
    assert actual == pytest.approx(expected)


@pytest.mark.parametrize('metals', [('Cu','Au'), ('Cu','Ni'), ('Cu','Pt'), ('Ni','Pt')])
@pytest.mark.parametrize('ads,region', [('CO','M1_side_interface'), ('CO','M2_terrace'), ('OCCO','M1_M2_boundary_bridge'), ('OCCO','M1_side_interface_dimer_along_x')])
def test_generic_regions_have_auditable_chemistry(metals, ads, region):
    install()
    gen = StructureGenerator()
    base = gen.generate_interface(*metals, supercell=(3,4), element1_rows=4, element2_rows=4)
    result = gen.add_adsorbate(base, ads, adsorption_region=region)
    assert result['metadata']['label_status'] == 'verified'
    assert result['metadata']['species_observed'] == ads
    assert all(b['intact'] for b in result['metadata']['geometry_qc']['bonds'])
    assert result['metadata']['geometry_requested'] == region
    assert result['metadata']['adsorption_binding_element'] in {None, *metals}


@pytest.mark.parametrize('ads', ['CO', 'OH', 'CH3', 'COCO', 'OCCO'])
def test_molecule_mapping_survives_poscar_order_and_relaxation(tmp_path, ads):
    install()
    gen = StructureGenerator()
    base = gen.generate_interface('Cu', 'Au', supercell=(3,4), element1_rows=4, element2_rows=4)
    region = 'M1_M2_boundary_bridge' if ads in {'COCO', 'OCCO'} else 'M1_side_interface'
    result = gen.add_adsorbate(base, ads, adsorption_region=region)
    original_id = result['metadata']['configuration_id']
    files = gen.save_structure(result, str(tmp_path))
    saved = read_poscar(files['poscar'])
    info = json.loads((tmp_path / 'structure_info.json').read_text())
    assert info['configuration_id'] == original_id
    actual = annotate_structure({**saved, 'metadata': info})
    assert all(b['intact'] for b in actual['metadata']['geometry_qc']['bonds'])
    assert actual['metadata']['species_observed'] == ('OCCO' if ads == 'COCO' else ads)
    relaxed = deepcopy(saved)
    relaxed['positions'] = np.asarray(relaxed['positions'], float)
    last = info['molecules'][0]['atom_indices'][-1]
    relaxed['positions'][last] += [0, 0, 4]
    updated = relaxed_metadata(relaxed, info, saved)
    assert updated['configuration_id'] == original_id
    assert updated['species_requested'] == info['species_requested']
    assert updated['label_status'] == 'review_required'
    assert updated['ml_targets']['adsorbate_identity'] is None
    manifest = write_dataset_manifest(tmp_path)
    assert manifest[0]['configuration_id'] == original_id


def test_mixed_co_h_keeps_molecule_counts_and_identity():
    gen = StructureGenerator()
    base = gen.generate_surface('Cu', supercell=(4,4))
    co = gen.add_adsorbate(base, 'CO', site='top', site_index=0)
    # Coadsorbates must use substrate sites rather than the existing CO oxygen.
    mixed = gen.add_adsorbate(co, 'H', site='top', site_index=5, height=1.7)
    assert mixed['metadata']['coverage_labels']['molecule_counts'] == {'CO':1, 'H':1}
    assert mixed['metadata']['configuration_id'] != co['metadata']['configuration_id']
    assert mixed['metadata']['species_requested'] == 'CO+H'
    assert mixed['metadata']['label_status'] == 'verified'


def test_unsupported_region_and_facet_fail_explicitly():
    install()
    gen = StructureGenerator()
    with pytest.raises(ValueError, match='fcc'):
        gen.generate_interface('Cu','Au', facet1='100', facet2='100')
    base = gen.generate_interface('Ni','Pt')
    with pytest.raises(ValueError, match='Unknown adsorption region'):
        gen.add_adsorbate(base, 'CO', adsorption_region='unknown')
    with pytest.raises(ValueError, match='named metal'):
        gen.add_adsorbate(base, 'CO', adsorption_region='Cu_side_interface')


def test_unknown_sites_and_uncalibrated_cooh_sweeps_fail_explicitly():
    from tools.adsorbate_scenarios import generate_adsorbate_scenarios
    gen = StructureGenerator()
    base = gen.generate_surface('Ni', supercell=(4,4))
    with pytest.raises(ValueError, match='Unknown adsorption site'):
        gen.add_adsorbate(base, 'CO', site='misspelled')
    with pytest.raises(ValueError, match='Cu/Au only'):
        generate_adsorbate_scenarios(base, 'COOH', .1)
    assert gen._resolve_adsorbate_site('CO', 'atop') == 'top'


def test_along_interface_dimer_uses_one_row_and_row_classes_include_both_boundaries():
    install()
    gen = StructureGenerator()
    base = gen.generate_interface('Ni','Pt', element1_rows=6, element2_rows=6)
    result = gen.add_adsorbate(base, 'OCCO', adsorption_region='M1_side_interface_dimer_along_x')
    i, j = result['metadata']['selected_sites'][0]['indices']
    assert base['positions'][i, 1] == pytest.approx(base['positions'][j, 1])
    from tools.stripe_interface_patch import _rank_rows_by_nearest_interface
    assert list(_rank_rows_by_nearest_interface([.7,1.7,2.7,3.7,4.7,5.7], 0, 6).values()) == [0,1,2,2,1,0]


def test_an_intended_bond_does_not_hide_an_extra_bond_to_its_periodic_image():
    from tools.dataset_labels import molecule_record
    atoms = ['C','O']
    positions = np.array([[0.,0.,3.],[1.16,0.,3.]])
    molecule = molecule_record('CO', [0,1], atoms, positions)
    report = validate_geometry(atoms, positions, np.diag([2.32,10.,20.]), molecules=[molecule])
    assert report['severity'] == 'error'
    assert any(issue['kind'] == 'unexpected_periodic_bond' for issue in report['issues'])
