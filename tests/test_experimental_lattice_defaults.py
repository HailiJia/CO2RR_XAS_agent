from generators.slab_generator import LATTICE_CONSTANTS
from tools.utils import METAL_DATA


def test_experimental_lattice_defaults_are_consistent():
    expected = {
        "Al": 4.050,
        "Fe": 2.867,
        "Co": 2.507,
        "Ni": 3.524,
        "Cu": 3.615,
        "Zn": 2.665,
        "Mo": 3.147,
        "Pd": 3.891,
        "Ag": 4.085,
        "Pt": 3.924,
        "Au": 4.078,
    }

    for element, value in expected.items():
        assert METAL_DATA[element]["a"] == value
        assert LATTICE_CONSTANTS[element] == value
