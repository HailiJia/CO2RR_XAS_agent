import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from tools.dataset_labels import annotate_structure
from tools.result_parser import ISAACRecordGenerator
from tools.structure_generator import StructureGenerator
from tools.stripe_interface_patch import install
from tools.utils import read_poscar, write_poscar
from tools.xas_input_generator import _isaac_submit_metadata_finalize
from workflow.relaxed_xas_rewrite import regenerate_xas_from_relaxed


def test_relaxed_labels_reach_all_three_xas_backends(tmp_path):
    install()
    gen = StructureGenerator()
    initial = gen.add_adsorbate(gen.generate_interface('Cu','Au', element1_rows=4, element2_rows=4), 'CO')
    gen.save_structure(initial, str(tmp_path / 'relax'))
    initial_text = (tmp_path / 'relax' / 'POSCAR').read_text()
    initial_saved = read_poscar(str(tmp_path / 'relax' / 'POSCAR'))
    # A small rigid relaxation keeps connectivity while changing the geometry hash.
    relaxed = deepcopy(initial_saved)
    relaxed['positions'] = np.asarray(relaxed['positions'], float)
    relaxed['positions'][-2:] += [0.05, 0.03, 0.02]
    write_poscar(relaxed['atoms'], relaxed['positions'], relaxed['cell'], str(tmp_path / 'relax' / 'CONTCAR'))
    result = regenerate_xas_from_relaxed(output_dir=tmp_path, nersc_account='m1234', edge_override='K')
    assert (tmp_path / 'relax' / 'POSCAR').read_text() == initial_text
    metadata = json.loads((tmp_path / 'relaxed_structure_info.json').read_text())
    assert metadata['configuration_id'] == initial['metadata']['configuration_id']
    assert metadata['label_stage'] == 'post_relaxation'
    assert metadata['ml_targets']['adsorbate_identity'] == 'CO'
    assert metadata['geometry_hash'] != metadata['geometry_hash_initial']
    for backend in ('FEFF', 'FDMNES', 'VASP'):
        path = tmp_path / 'xas' / 'Cu_K_edge' / backend / 'structure_info.json'
        info = json.loads(path.read_text())
        assert info['configuration_id'] == metadata['configuration_id']
        assert info['atom_ids'] == metadata['atom_ids']
        assert info['ml_targets'] == metadata['ml_targets']
    assert Path(result['xas_structure_signature']).is_file()


def test_job_metadata_footer_preserves_labels_without_repo_imports(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    info = {'configuration_id':'cfg_test', 'ml_targets':{'adsorbate_identity':'CO'}, 'label_stage':'post_relaxation'}
    (tmp_path / 'structure_info.json').write_text(json.dumps(info))
    lines = _isaac_submit_metadata_finalize()
    start = lines.index("python3 - <<'PY'") + 1
    end = lines.index('PY', start)
    source = '\n'.join(lines[start:end])
    exec(compile(source, '<generated metadata footer>', 'exec'), {})
    metadata = json.loads((tmp_path / 'isaac_run_metadata.json').read_text())
    assert metadata['sample']['configuration_id'] == 'cfg_test'
    assert metadata['sample']['ml_labels']['adsorbate_identity'] == 'CO'


def test_isaac_records_use_measured_targets_and_configuration_linkage(tmp_path):
    gen = StructureGenerator()
    structure = gen.add_adsorbate(gen.generate_surface('Cu'), 'CO')
    files = gen.save_structure(structure, str(tmp_path))
    spectrum = {'energy':[0.,1.,2.,3.,4.], 'intensity':[0.,1.,2.,1.,0.], 'energy_units':'eV', 'intensity_units':'arb'}
    record = ISAACRecordGenerator().create_record(structure, 'FEFF', spectrum, {}, 'Cu', 'K', str(tmp_path), files['poscar'])
    assert record['sample']['configuration_id'] == structure['metadata']['configuration_id']
    assert record['sample']['ml_labels']['adsorbate_identity'] == 'CO'
    assert record['sample']['structure_descriptors']['atom_ids']


def test_native_slab_route_writes_molecule_and_atom_mappings(tmp_path):
    pytest.importorskip('pymatgen')
    pytest.importorskip('ase')
    from generators.slab_generator import SlabGenerator
    generator = SlabGenerator()
    outputs = generator.generate({'element':'Cu', 'miller_index':[1,1,1], 'layers':3,
                                  'supercell':[3,3], 'adsorbate':{'molecule':'CH3'}})
    generator.write_files(outputs, tmp_path)
    metadata = json.loads((tmp_path / 'structure_info.json').read_text())
    assert metadata['ml_targets']['adsorbate_identity'] == 'CH3'
    saved = read_poscar(str(tmp_path / 'POSCAR'))
    assert len(metadata['atom_ids']) == len(saved['atoms'])
    checked = annotate_structure({**saved, 'metadata':metadata})['metadata']
    assert all(bond['intact'] for bond in checked['geometry_qc']['bonds'])
