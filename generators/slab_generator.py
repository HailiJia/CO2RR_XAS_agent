"""
Slab Generator - Generate surface slab structures with adsorbates
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np
import re

from pymatgen.core import Structure, Lattice, Element
from pymatgen.core.surface import SlabGenerator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp import Poscar
from pymatgen.io.ase import AseAtomsAdaptor

from ase import Atoms
from ase.build import molecule, add_adsorbate, fcc111, fcc100, fcc110
from ase.build import bcc111, bcc100, bcc110, hcp0001
from ase.constraints import FixAtoms

from generators.base_generator import BaseGenerator


# Lattice constants for common metals (in Angstroms)
LATTICE_CONSTANTS = {
    'Cu': 3.615,
    'Au': 4.078,
    'Ag': 4.085,
    'Pt': 3.924,
    'Pd': 3.891,
    'Ni': 3.524,
    'Al': 4.050,
    'Fe': 2.867,  # BCC
    'W': 3.165,   # BCC
    'Mo': 3.147,  # BCC
    'Ti': 2.951,  # HCP (a parameter)
    'Zn': 2.665,  # HCP (a parameter)
    'Co': 2.507,  # HCP (a parameter)
}

# Crystal structures for common metals
CRYSTAL_STRUCTURES = {
    'Cu': 'fcc', 'Au': 'fcc', 'Ag': 'fcc', 'Pt': 'fcc', 'Pd': 'fcc',
    'Ni': 'fcc', 'Al': 'fcc', 'Pb': 'fcc', 'Rh': 'fcc', 'Ir': 'fcc',
    'Fe': 'bcc', 'W': 'bcc', 'Mo': 'bcc', 'Cr': 'bcc', 'V': 'bcc',
    'Ti': 'hcp', 'Zn': 'hcp', 'Co': 'hcp', 'Ru': 'hcp', 'Os': 'hcp',
}

# CO2RR adsorbate definitions following the ISAAC VASP skill reference.
# Positions are built explicitly in _build_adsorbate_atoms so carbon anchors sit
# closest to the surface and all O/H atoms point away from the slab.
CO2RR_ADSORBATES = {
    'CO', 'CH', 'CH2', 'CHO', 'CHOH', 'CH3', 'CH4', 'COCO', 'OCCO'
}

ADSORBATE_DEFAULTS = {
    'CO': {'site': 'hollow', 'height': 1.30},
    'CH': {'site': 'hollow', 'height': 1.30},
    'CH2': {'site': 'hollow', 'height': 1.30},
    'CHO': {'site': 'bridge', 'height': 1.50},
    'CHOH': {'site': 'ontop', 'height': 1.85},
    'CH3': {'site': 'ontop', 'height': 1.85},
    'CH4': {'site': 'ontop', 'height': 1.85},
    'COCO': {'site': 'interface', 'height': 1.30},
    'OCCO': {'site': 'interface', 'height': 1.50},
}

# Legacy/free-form adsorbate molecule definitions retained for backwards compatibility.
ADSORBATE_MOLECULES = {
    'CO': {'atoms': ['C', 'O'], 'positions': [[0, 0, 0], [0, 0, 1.16]]},
    'CO2': {'atoms': ['C', 'O', 'O'], 'positions': [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]},
    'H': {'atoms': ['H'], 'positions': [[0, 0, 0]]},
    'OH': {'atoms': ['O', 'H'], 'positions': [[0, 0, 0], [0, 0, 0.97]]},
    'H2O': {'atoms': ['O', 'H', 'H'], 'positions': [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]},
    'COOH': {'atoms': ['C', 'O', 'O', 'H'], 'positions': [[0, 0, 0], [1.21, 0, 0], [-0.36, 1.22, 0], [-0.36, 1.82, 0.76]]},
    'CHO': {'atoms': ['C', 'O', 'H'], 'positions': [[0, 0, 0], [1.094, 0, 0.387], [-0.363, 0.943, 0.363]]},
    'HCOO': {'atoms': ['H', 'C', 'O', 'O'], 'positions': [[0, 0, 1.09], [0, 0, 0], [1.13, 0, -0.36], [-1.13, 0, -0.36]]},
}


class SlabGenerator(BaseGenerator):
    """Generate surface slab structures with optional adsorbates."""
    
    def __init__(self):
        super().__init__(skill_id='vasp_slab_generation')
    
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a slab structure based on parameters.
        
        Args:
            params: Validated parameters including element, miller_index, etc.
        
        Returns:
            Dictionary with 'structure' (pymatgen) and 'atoms' (ASE) objects
        """
        element_spec = params['element']
        elements = self._parse_element_spec(element_spec)
        miller_index = tuple(params['miller_index'])
        layers = params.get('layers', 4)
        vacuum = params.get('vacuum', 15.0)
        fix_layers = params.get('fix_layers', 2)
        supercell = params.get('supercell', [2, 2])
        lattice_constant = params.get('lattice_constant')
        if lattice_constant is None:
            lattice_constant = self._default_lattice_constant(elements)
        
        if lattice_constant is None:
            element_label = '/'.join(elements)
            raise ValueError(f"Unknown lattice constant for {element_label}. Please provide 'lattice_constant' parameter.")

        interface_match = params.get('interface_match', {})
        max_interface_strain = interface_match.get('max_strain', params.get('max_interface_strain', 0.05))
        max_interface_repeats = interface_match.get('max_repeats', params.get('max_interface_repeats', 12))
        interface_match_side = interface_match.get('match_side', params.get('interface_match_side', 'balanced'))
        
        # Generate slab using ASE or an explicitly matched bimetal interface builder.
        slab = self._build_slab_ase(
            elements=elements,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            lattice_constant=lattice_constant,
            supercell=supercell,
            max_interface_strain=max_interface_strain,
            max_interface_repeats=max_interface_repeats,
            interface_match_side=interface_match_side,
        )
        
        # Add adsorbate if specified
        adsorbate_params = params.get('adsorbate')
        if adsorbate_params and adsorbate_params.get('molecule'):
            slab = self._add_adsorbate(slab, adsorbate_params)
        
        # Apply constraints (fix bottom layers)
        slab = self._apply_constraints(slab, fix_layers)
        
        # Convert to pymatgen Structure
        adaptor = AseAtomsAdaptor()
        structure = adaptor.get_structure(slab)
        
        # Generate POSCAR content
        poscar = Poscar(structure)
        poscar_content = poscar.get_str()
        
        return {
            'atoms': slab,
            'structure': structure,
            'poscar': poscar_content,
            'params': params,
            'cell_parameters': self._cell_parameters(slab),
            'interface': slab.info.get('interface'),
            'adsorbate_placement': slab.info.get('adsorbate_placement'),
        }
    
    def _cell_parameters(self, atoms: Atoms) -> Dict[str, Any]:
        """Return cell lengths, angles, and vectors for diagnostics."""
        cell = atoms.get_cell()
        return {
            'lengths': [float(value) for value in cell.lengths()],
            'angles': [float(value) for value in cell.angles()],
            'vectors': [[float(component) for component in vector] for vector in cell.array],
        }

    def _parse_element_spec(self, element_spec: Any) -> List[str]:
        """Normalize single-metal and interface element specifications."""
        if isinstance(element_spec, (list, tuple)):
            elements = [str(element).strip() for element in element_spec]
        else:
            element_text = str(element_spec).strip()
            if '/' in element_text:
                elements = [part.strip() for part in element_text.split('/')]
            elif '-' in element_text:
                elements = [part.strip() for part in element_text.split('-')]
            else:
                elements = [element_text]

        elements = [self._normalize_element_symbol(element) for element in elements if element]
        if not elements:
            raise ValueError("At least one slab element must be specified.")
        if len(elements) > 2:
            raise ValueError("Only single-metal slabs and two-metal interfaces are supported.")
        return elements

    def _normalize_element_symbol(self, element_text: str) -> str:
        """Extract an element symbol from forms such as Cu, Cu(111), or Cu 111."""
        match = re.match(r"^\s*([A-Z][a-z]?)", element_text)
        if not match:
            raise ValueError(f"Could not parse element symbol from {element_text!r}.")
        return match.group(1)

    def _default_lattice_constant(self, elements: List[str]) -> Optional[float]:
        """Return a default lattice constant, averaging for two-metal interfaces."""
        lattice_constants = [LATTICE_CONSTANTS.get(element) for element in elements]
        if any(value is None for value in lattice_constants):
            return None
        return float(np.mean(lattice_constants))

    def _build_slab_ase(
        self,
        elements: List[str],
        miller_index: Tuple[int, int, int],
        layers: int,
        vacuum: float,
        lattice_constant: float,
        supercell: List[int],
        max_interface_strain: float = 0.05,
        max_interface_repeats: int = 12,
        interface_match_side: str = 'balanced',
    ) -> Atoms:
        """Build slab using ASE builders."""
        base_element = elements[0]
        crystal_structure = CRYSTAL_STRUCTURES.get(base_element, 'fcc')
        if len(elements) == 2:
            second_structure = CRYSTAL_STRUCTURES.get(elements[1], 'fcc')
            if second_structure != crystal_structure:
                raise ValueError(
                    "Interface slabs require elements with matching crystal structures; "
                    f"got {base_element}={crystal_structure} and {elements[1]}={second_structure}."
                )

        if len(elements) == 2 and crystal_structure == 'fcc' and miller_index == (1, 1, 1):
            return self._build_matched_fcc111_interface(
                elements=elements,
                layers=layers,
                vacuum=vacuum,
                supercell=supercell,
                max_strain=max_interface_strain,
                max_repeats=max_interface_repeats,
                match_side=interface_match_side,
            )

        # Map miller index to ASE builder function.  ASE's vacuum argument adds
        # the requested total vacuum around the slab when periodic z is disabled.
        builder_map = {
            ('fcc', (1, 1, 1)): lambda: fcc111(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
            ('fcc', (1, 0, 0)): lambda: fcc100(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
            ('fcc', (1, 1, 0)): lambda: fcc110(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
            ('bcc', (1, 1, 1)): lambda: bcc111(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
            ('bcc', (1, 0, 0)): lambda: bcc100(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
            ('bcc', (1, 1, 0)): lambda: bcc110(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
            ('hcp', (0, 0, 0, 1)): lambda: hcp0001(base_element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum),
        }

        key = (crystal_structure, miller_index)

        if key in builder_map:
            slab = builder_map[key]()
        else:
            # Fall back to pymatgen for arbitrary miller indices
            slab = self._build_slab_pymatgen(
                base_element, miller_index, layers, vacuum, lattice_constant, supercell
            )

        if len(elements) == 2:
            slab = self._assign_lateral_interface_domains(
                slab, left_element=elements[0], right_element=elements[1]
            )

        return slab

    def _build_matched_fcc111_interface(
        self,
        elements: List[str],
        layers: int,
        vacuum: float,
        supercell: List[int],
        max_strain: float,
        max_repeats: int,
        match_side: str,
    ) -> Atoms:
        """Build a minimal fcc(111)/fcc(111) lateral interface within a strain limit.

        A one-to-one Cu/Au(111) match strains one side by more than 10%.  This
        builder searches for the smallest integer repeat pair along the interface
        direction whose matched length keeps both domains within ``max_strain``.
        For Cu/Au with a 5% limit the minimal pair is 7 Cu repeats against 6 Au
        repeats.  The common length is balanced by default, but callers may pin
        the cell length to either side with ``match_side``.
        """
        left_element, right_element = elements
        left_lattice = LATTICE_CONSTANTS.get(left_element)
        right_lattice = LATTICE_CONSTANTS.get(right_element)
        if left_lattice is None or right_lattice is None:
            raise ValueError(
                f"Known lattice constants are required for matched interfaces: {left_element}/{right_element}."
            )

        match = self._find_minimal_interface_match(
            left_element=left_element,
            right_element=right_element,
            max_strain=max_strain,
            max_repeats=max_repeats,
            match_side=match_side,
        )
        left_repeats = match['left_repeats']
        right_repeats = match['right_repeats']
        # The integer match already defines the minimal commensurate repeat.
        # Do not multiply it by the ordinary slab supercell defaults, otherwise
        # Cu/Au(111) grows from the minimal 7+6 atoms/layer to unnecessarily
        # large cells such as 28+24 atoms/layer for a default 2x2 supercell.
        x_columns = 1
        y_repeats = 1
        left_rows = left_repeats * y_repeats
        right_rows = right_repeats * y_repeats
        common_y = match['common_length'] * y_repeats
        left_spacing_y = common_y / left_rows
        right_spacing_y = common_y / right_rows

        left_spacing_x = np.sqrt(3.0) * left_spacing_y / 2.0
        right_spacing_x = np.sqrt(3.0) * right_spacing_y / 2.0
        left_width = x_columns * left_spacing_x
        right_width = x_columns * right_spacing_x
        cell_x = left_width + right_width
        interface_x = left_width

        z_spacing = float(np.mean([left_lattice, right_lattice])) / np.sqrt(3.0)
        slab_thickness = (layers - 1) * z_spacing
        cell_z = slab_thickness + vacuum
        z_origin = vacuum / 2.0

        symbols = []
        positions = []
        for layer in range(layers):
            z = z_origin + layer * z_spacing
            left_layer_shift = (layer % 3) * left_spacing_y / 3.0
            right_layer_shift = (layer % 3) * right_spacing_y / 3.0
            x_layer_shift = (layer % 3) * min(left_spacing_x, right_spacing_x) / 3.0

            for x_column in range(x_columns):
                x = (x_column + 0.5) * left_spacing_x + x_layer_shift
                if x >= interface_x:
                    x -= left_spacing_x / 2.0
                for row in range(left_rows):
                    y = (row * left_spacing_y + left_layer_shift) % common_y
                    symbols.append(left_element)
                    positions.append([x, y, z])

                x = interface_x + (x_column + 0.5) * right_spacing_x + x_layer_shift
                if x >= cell_x:
                    x -= right_spacing_x / 2.0
                for row in range(right_rows):
                    y = (row * right_spacing_y + right_layer_shift) % common_y
                    symbols.append(right_element)
                    positions.append([x, y, z])

        atoms = Atoms(
            symbols=symbols,
            positions=positions,
            cell=[cell_x, common_y, cell_z],
            pbc=[True, True, False],
        )
        atoms.info['interface'] = {
            'type': 'lateral',
            'left_element': left_element,
            'right_element': right_element,
            'split_axis': 'x',
            'split_coordinate': interface_x,
            'interface_direction': 'y',
            'match': match,
            'left_atoms_per_layer': left_rows * x_columns,
            'right_atoms_per_layer': right_rows * x_columns,
            'requested_supercell': list(supercell),
            'effective_supercell': [x_columns, y_repeats],
            'description': (
                f"{left_element}/{right_element} fcc(111) domains share a lateral interface; "
                f"{left_repeats}:{right_repeats} repeats keep strain <= {max_strain:.1%}."
            ),
        }
        return atoms

    def _find_minimal_interface_match(
        self,
        left_element: str,
        right_element: str,
        max_strain: float,
        max_repeats: int,
        match_side: str,
    ) -> Dict[str, Any]:
        """Find the smallest integer surface repeat match within the strain limit."""
        left_period = LATTICE_CONSTANTS[left_element] / np.sqrt(2.0)
        right_period = LATTICE_CONSTANTS[right_element] / np.sqrt(2.0)
        match_side = str(match_side or 'balanced').lower()
        if match_side not in {'balanced', left_element.lower(), right_element.lower(), 'left', 'right'}:
            raise ValueError(
                "interface_match_side must be 'balanced', 'left', 'right', "
                f"'{left_element}', or '{right_element}'."
            )

        candidates = []
        for left_repeats in range(1, int(max_repeats) + 1):
            left_length = left_repeats * left_period
            for right_repeats in range(1, int(max_repeats) + 1):
                right_length = right_repeats * right_period
                if match_side in {'left', left_element.lower()}:
                    common_length = left_length
                    matched_to = left_element
                elif match_side in {'right', right_element.lower()}:
                    common_length = right_length
                    matched_to = right_element
                else:
                    common_length = 0.5 * (left_length + right_length)
                    matched_to = 'balanced'

                left_strain = common_length / left_length - 1.0
                right_strain = common_length / right_length - 1.0
                max_abs_strain = max(abs(left_strain), abs(right_strain))
                if max_abs_strain <= max_strain:
                    candidates.append({
                        'left_repeats': left_repeats,
                        'right_repeats': right_repeats,
                        'left_period': left_period,
                        'right_period': right_period,
                        'left_unstrained_length': left_length,
                        'right_unstrained_length': right_length,
                        'common_length': common_length,
                        'left_strain': left_strain,
                        'right_strain': right_strain,
                        'max_abs_strain': max_abs_strain,
                        'matched_to': matched_to,
                    })

        if not candidates:
            raise ValueError(
                f"No {left_element}/{right_element} interface match found within {max_strain:.1%} strain "
                f"using repeats <= {max_repeats}. Increase max_interface_repeats or max_interface_strain."
            )

        candidates.sort(key=lambda item: (
            item['left_repeats'] + item['right_repeats'],
            item['max_abs_strain'],
            abs(item['left_repeats'] - item['right_repeats']),
        ))
        return candidates[0]

    def _assign_lateral_interface_domains(self, slab: Atoms, left_element: str, right_element: str) -> Atoms:
        """Create a side-by-side bimetal surface interface in every slab layer.

        The first element occupies the lower-x half of the surface cell and the
        second element occupies the upper-x half, producing a periodic line
        interface parallel to the second in-plane cell vector. This represents
        interfaces such as Cu(111)/Au(111) as a strained, commensurate lateral
        Cu/Au domain boundary rather than a vertical overlayer.
        """
        positions = slab.get_positions()
        x_midpoint = 0.5 * (positions[:, 0].min() + positions[:, 0].max())
        symbols = [left_element if position[0] < x_midpoint else right_element for position in positions]
        slab.set_chemical_symbols(symbols)
        slab.info['interface'] = {
            'type': 'lateral',
            'left_element': left_element,
            'right_element': right_element,
            'split_axis': 'x',
            'split_coordinate': x_midpoint,
            'description': f'{left_element}/{right_element} domains share a line interface parallel to the y direction',
        }
        return slab

    def _build_slab_pymatgen(
        self,
        element: str,
        miller_index: Tuple[int, int, int],
        layers: int,
        vacuum: float,
        lattice_constant: float,
        supercell: List[int]
    ) -> Atoms:
        """Build slab using pymatgen for arbitrary miller indices."""
        from pymatgen.core import Structure, Lattice
        from pymatgen.core.surface import SlabGenerator as PMGSlabGenerator

        crystal_structure = CRYSTAL_STRUCTURES.get(element, 'fcc')

        # Create bulk structure
        if crystal_structure == 'fcc':
            lattice = Lattice.cubic(lattice_constant)
            bulk = Structure(
                lattice,
                [element] * 4,
                [[0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]
            )
        elif crystal_structure == 'bcc':
            lattice = Lattice.cubic(lattice_constant)
            bulk = Structure(
                lattice,
                [element] * 2,
                [[0, 0, 0], [0.5, 0.5, 0.5]]
            )
        else:
            raise ValueError(f"Unsupported crystal structure: {crystal_structure}")

        # Generate slab
        slab_gen = PMGSlabGenerator(
            bulk,
            miller_index,
            min_slab_size=layers * 2,  # Approximate
            min_vacuum_size=vacuum,
            center_slab=True
        )

        slabs = slab_gen.get_slabs()
        if not slabs:
            raise ValueError(f"Could not generate slab for {element}{miller_index}")

        slab_structure = slabs[0]

        # Make supercell
        slab_structure.make_supercell([supercell[0], supercell[1], 1])

        # Convert to ASE
        adaptor = AseAtomsAdaptor()
        return adaptor.get_atoms(slab_structure)

    def _add_adsorbate(self, slab: Atoms, adsorbate_params: Dict[str, Any]) -> Atoms:
        """Add adsorbate molecule to the slab."""
        mol_name = adsorbate_params['molecule']
        default = ADSORBATE_DEFAULTS.get(mol_name, {})
        binding_site = adsorbate_params.get('binding_site', default.get('site', 'ontop'))
        distance = adsorbate_params.get('distance', default.get('height', 2.0))
        orientation = adsorbate_params.get('orientation', [0, 0, 1])

        adsorbate = self._build_adsorbate_atoms(mol_name)

        if self._should_place_across_interface(slab, mol_name, binding_site):
            return self._add_bidentate_adsorbate_across_interface(slab, adsorbate, mol_name, distance)

        # Rotate adsorbate to match orientation
        adsorbate = self._orient_adsorbate(adsorbate, orientation)

        # Find binding site position
        site_position = self._find_binding_site(slab, binding_site)

        # Add adsorbate. The first atom in every explicit CO2RR adsorbate is the
        # surface anchor atom, so ASE places that atom at the requested height.
        add_adsorbate(slab, adsorbate, distance, position=site_position[:2])

        return slab

    def _should_place_across_interface(self, slab: Atoms, mol_name: str, binding_site: str) -> bool:
        """Return True when a bidentate adsorbate should straddle a bimetal interface."""
        return (
            slab.info.get('interface', {}).get('type') == 'lateral'
            and binding_site == 'interface'
            and mol_name in {'COCO', 'OCCO'}
        )

    def _add_bidentate_adsorbate_across_interface(
        self,
        slab: Atoms,
        adsorbate: Atoms,
        mol_name: str,
        distance: float,
    ) -> Atoms:
        """Place OCCO/COCO with one carbon over each side of a lateral interface."""
        bridge = self._find_interface_bridge(slab)
        carbon_indices = [index for index, symbol in enumerate(adsorbate.get_chemical_symbols()) if symbol == 'C']
        if len(carbon_indices) < 2:
            raise ValueError(f"{mol_name} requires at least two carbon atoms for interface placement.")

        adsorbate = self._align_adsorbate_axis(
            adsorbate,
            from_index=carbon_indices[0],
            to_index=carbon_indices[1],
            target_vector=bridge['vector'],
        )

        positions = adsorbate.get_positions()
        carbon_positions = positions[carbon_indices[:2]]
        carbon_center = carbon_positions.mean(axis=0)
        target_center = np.array([
            bridge['midpoint'][0],
            bridge['midpoint'][1],
            bridge['surface_z'] + distance,
        ])
        adsorbate.translate(target_center - carbon_center)
        slab.extend(adsorbate)
        slab.info['adsorbate_placement'] = {
            'molecule': mol_name,
            'site': 'interface',
            'mode': 'bidentate_across_lateral_interface',
            'left_element': bridge['left_element'],
            'right_element': bridge['right_element'],
            'left_surface_index': bridge['left_index'],
            'right_surface_index': bridge['right_index'],
            'carbon_indices': [len(slab) - len(adsorbate) + index for index in carbon_indices[:2]],
        }
        return slab

    def _align_adsorbate_axis(
        self,
        adsorbate: Atoms,
        from_index: int,
        to_index: int,
        target_vector: np.ndarray,
    ) -> Atoms:
        """Rotate an adsorbate so one internal bond aligns with a target vector."""
        source_vector = adsorbate.positions[to_index] - adsorbate.positions[from_index]
        source_norm = np.linalg.norm(source_vector)
        target_norm = np.linalg.norm(target_vector)
        if source_norm < 1e-12 or target_norm < 1e-12:
            return adsorbate

        source = source_vector / source_norm
        target = target_vector / target_norm
        axis = np.cross(source, target)
        axis_norm = np.linalg.norm(axis)
        if axis_norm > 1e-8:
            axis = axis / axis_norm
            angle_deg = np.degrees(np.arccos(np.clip(np.dot(source, target), -1, 1)))
            adsorbate.rotate(angle_deg, axis, center=adsorbate.positions[from_index])
        elif np.dot(source, target) < 0:
            adsorbate.rotate(180, 'z', center=adsorbate.positions[from_index])
        return adsorbate

    def _find_interface_bridge(self, slab: Atoms) -> Dict[str, Any]:
        """Find a nearest-neighbor mixed-metal bridge on the top lateral interface."""
        interface = slab.info.get('interface', {})
        left_element = interface.get('left_element')
        right_element = interface.get('right_element')
        if not left_element or not right_element:
            raise ValueError("Interface bridge requested for a slab without interface metadata.")

        positions = slab.get_positions()
        symbols = slab.get_chemical_symbols()
        z_coords = positions[:, 2]
        surface_z = z_coords.max()
        surface_indices = [index for index, z_coord in enumerate(z_coords) if z_coord > surface_z - 0.5]
        left_indices = [index for index in surface_indices if symbols[index] == left_element]
        right_indices = [index for index in surface_indices if symbols[index] == right_element]
        if not left_indices or not right_indices:
            raise ValueError("Could not find both metals in the top surface interface layer.")

        best_pair = None
        best_distance = None
        for left_index in left_indices:
            for right_index in right_indices:
                delta = positions[right_index, :2] - positions[left_index, :2]
                distance = np.linalg.norm(delta)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    best_pair = (left_index, right_index)

        left_index, right_index = best_pair
        left_position = positions[left_index]
        right_position = positions[right_index]
        return {
            'left_element': left_element,
            'right_element': right_element,
            'left_index': int(left_index),
            'right_index': int(right_index),
            'left_position': left_position,
            'right_position': right_position,
            'midpoint': 0.5 * (left_position + right_position),
            'vector': right_position - left_position,
            'surface_z': surface_z,
        }

    def _build_adsorbate_atoms(self, mol_name: str) -> Atoms:
        """Build an adsorbate with deterministic CO2RR geometries."""
        if mol_name in CO2RR_ADSORBATES:
            return self._build_co2rr_adsorbate(mol_name)

        # Retain support for previous examples and ASE's molecule database.
        if mol_name in ADSORBATE_MOLECULES:
            mol_def = ADSORBATE_MOLECULES[mol_name]
            return Atoms(symbols=mol_def['atoms'], positions=mol_def['positions'])

        try:
            return molecule(mol_name)
        except Exception:
            raise ValueError(f"Unknown adsorbate molecule: {mol_name}")

    def _build_co2rr_adsorbate(self, mol_name: str) -> Atoms:
        """Return CO2RR adsorbates anchored at atom 0 and pointing upward."""
        if mol_name == 'CO':
            return Atoms(['C', 'O'], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.16]])
        if mol_name == 'CH':
            return Atoms(['C', 'H'], positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.10]])
        if mol_name == 'CH2':
            return Atoms(
                ['C', 'H', 'H'],
                positions=[[0.0, 0.0, 0.0], [0.565, 0.0, 0.943], [-0.565, 0.0, 0.943]],
            )
        if mol_name == 'CHO':
            return Atoms(
                ['C', 'O', 'H'],
                positions=[[0.0, 0.0, 0.0], [1.094, 0.0, 0.387], [-0.363, 0.943, 0.363]],
            )
        if mol_name == 'CHOH':
            return Atoms(
                ['C', 'O', 'H', 'H'],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.30, 0.0, 0.25],
                    [-0.42, 0.90, 0.38],
                    [1.72, 0.0, 1.12],
                ],
            )
        if mol_name == 'CH3':
            return Atoms(
                ['C', 'H', 'H', 'H'],
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.09],
                    [1.028, 0.0, 0.363],
                    [-0.514, 0.890, 0.363],
                ],
            )
        if mol_name == 'CH4':
            return Atoms(
                ['C', 'H', 'H', 'H', 'H'],
                positions=[
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.09],
                    [1.028, 0.0, 0.363],
                    [-0.514, 0.890, 0.363],
                    [-0.514, -0.890, 0.363],
                ],
            )
        if mol_name == 'COCO':
            return Atoms(
                ['C', 'O', 'C', 'O'],
                positions=[
                    [0.0, 0.0, 0.0],
                    [-1.16, 0.0, 0.10],
                    [1.35, 0.0, 0.25],
                    [2.51, 0.0, 0.35],
                ],
            )
        if mol_name == 'OCCO':
            return Atoms(
                ['C', 'C', 'O', 'O'],
                positions=[
                    [0.0, 0.0, 0.0],
                    [1.37, 0.0, 0.0],
                    [-0.58, 0.0, 1.05],
                    [1.95, 0.0, 1.05],
                ],
            )
        raise ValueError(f"Unsupported CO2RR adsorbate: {mol_name}")

    def _orient_adsorbate(self, adsorbate: Atoms, orientation: List[float]) -> Atoms:
        """Rotate adsorbate to align with given orientation vector."""
        # Default orientation is along z-axis
        current = np.array([0, 0, 1], dtype=float)
        target = np.array(orientation, dtype=float)
        norm = np.linalg.norm(target)
        if norm < 1e-12:
            raise ValueError("Adsorbate orientation vector must be non-zero.")
        target = target / norm

        if not np.allclose(current, target):
            # Calculate rotation axis and angle
            axis = np.cross(current, target)
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(current, target), -1, 1))
                angle_deg = np.degrees(angle)
                adsorbate.rotate(angle_deg, axis, center=adsorbate.positions[0])
            elif np.dot(current, target) < 0:
                adsorbate.rotate(180, 'x', center=adsorbate.positions[0])

        return adsorbate

    def _find_binding_site(self, slab: Atoms, site_type: str) -> np.ndarray:
        """Find the position of a binding site on the surface."""
        if site_type == 'interface' and slab.info.get('interface', {}).get('type') == 'lateral':
            return self._find_interface_bridge(slab)['midpoint']

        # Get surface atoms (top layer)
        positions = slab.get_positions()
        z_coords = positions[:, 2]
        z_max = z_coords.max()
        z_threshold = z_max - 0.5  # Within 0.5 Å of top

        surface_mask = z_coords > z_threshold
        surface_positions = positions[surface_mask]

        if len(surface_positions) == 0:
            # Fallback to center of cell
            cell = slab.get_cell()
            return np.array([cell[0, 0] / 2, cell[1, 1] / 2, z_max])

        center_xy = np.mean(surface_positions[:, :2], axis=0)
        ordered = sorted(surface_positions, key=lambda pos: np.linalg.norm(pos[:2] - center_xy))
        surface_positions = np.array(ordered)

        if site_type == 'ontop':
            return surface_positions[0]

        if site_type == 'bridge':
            if len(surface_positions) >= 2:
                distances = np.linalg.norm(surface_positions[1:, :2] - surface_positions[0, :2], axis=1)
                nearest = surface_positions[1:][int(np.argmin(distances))]
                return (surface_positions[0] + nearest) / 2
            return surface_positions[0]

        if site_type in ['fcc', 'hcp', 'hollow']:
            if len(surface_positions) >= 3:
                return np.mean(surface_positions[:3], axis=0)
            return np.mean(surface_positions, axis=0)

        return surface_positions[0]

    def _apply_constraints(self, slab: Atoms, fix_layers: int) -> Atoms:
        """Fix bottom layers of the slab."""
        if fix_layers <= 0:
            return slab
        
        positions = slab.get_positions()
        z_coords = positions[:, 2]
        
        # Determine z-threshold for fixed atoms
        z_min = z_coords.min()
        z_max = z_coords.max()
        z_range = z_max - z_min
        
        # Estimate layer thickness
        unique_z = np.unique(np.round(z_coords, decimals=1))
        if len(unique_z) > 1:
            layer_thickness = np.min(np.diff(np.sort(unique_z)))
        else:
            layer_thickness = z_range / 4  # Fallback estimate
        
        z_threshold = z_min + fix_layers * layer_thickness + 0.1
        
        # Find atoms to fix
        fixed_indices = np.where(z_coords < z_threshold)[0].tolist()
        
        if fixed_indices:
            constraint = FixAtoms(indices=fixed_indices)
            slab.set_constraint(constraint)
        
        return slab
    
    def write_files(self, outputs: Dict[str, Any], output_dir: Path) -> None:
        """Write POSCAR file."""
        output_dir = Path(output_dir)
        
        poscar_path = output_dir / 'POSCAR'
        with open(poscar_path, 'w') as f:
            f.write(outputs['poscar'])