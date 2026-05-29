"""
Slab Generator - Generate surface slab structures with adsorbates
"""

from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import numpy as np

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

# Adsorbate molecule definitions
ADSORBATE_MOLECULES = {
    'CO': {'atoms': ['C', 'O'], 'positions': [[0, 0, 0], [0, 0, 1.13]]},
    'CO2': {'atoms': ['C', 'O', 'O'], 'positions': [[0, 0, 0], [0, 0, 1.16], [0, 0, -1.16]]},
    'H': {'atoms': ['H'], 'positions': [[0, 0, 0]]},
    'OH': {'atoms': ['O', 'H'], 'positions': [[0, 0, 0], [0, 0, 0.97]]},
    'H2O': {'atoms': ['O', 'H', 'H'], 'positions': [[0, 0, 0], [0.76, 0.59, 0], [-0.76, 0.59, 0]]},
    'COOH': {'atoms': ['C', 'O', 'O', 'H'], 'positions': [[0, 0, 0], [1.21, 0, 0], [-0.36, 1.22, 0], [-0.36, 1.82, 0.76]]},
    'CHO': {'atoms': ['C', 'H', 'O'], 'positions': [[0, 0, 0], [0.54, 0.93, 0], [0, 0, 1.21]]},
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
        element = params['element']
        miller_index = tuple(params['miller_index'])
        layers = params.get('layers', 4)
        vacuum = params.get('vacuum', 15.0)
        fix_layers = params.get('fix_layers', 2)
        supercell = params.get('supercell', [2, 2])
        lattice_constant = params.get('lattice_constant', LATTICE_CONSTANTS.get(element))
        
        if lattice_constant is None:
            raise ValueError(f"Unknown lattice constant for {element}. Please provide 'lattice_constant' parameter.")
        
        # Generate slab using ASE
        slab = self._build_slab_ase(
            element=element,
            miller_index=miller_index,
            layers=layers,
            vacuum=vacuum,
            lattice_constant=lattice_constant,
            supercell=supercell
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
            'params': params
        }
    
    def _build_slab_ase(
        self,
        element: str,
        miller_index: Tuple[int, int, int],
        layers: int,
        vacuum: float,
        lattice_constant: float,
        supercell: List[int]
    ) -> Atoms:
        """Build slab using ASE builders."""
        crystal_structure = CRYSTAL_STRUCTURES.get(element, 'fcc')
        
        # Map miller index to ASE builder function
        builder_map = {
            ('fcc', (1, 1, 1)): lambda: fcc111(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
            ('fcc', (1, 0, 0)): lambda: fcc100(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
            ('fcc', (1, 1, 0)): lambda: fcc110(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
            ('bcc', (1, 1, 1)): lambda: bcc111(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
            ('bcc', (1, 0, 0)): lambda: bcc100(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
            ('bcc', (1, 1, 0)): lambda: bcc110(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
            ('hcp', (0, 0, 0, 1)): lambda: hcp0001(element, size=(supercell[0], supercell[1], layers), a=lattice_constant, vacuum=vacuum/2),
        }
        
        key = (crystal_structure, miller_index)
        
        if key in builder_map:
            slab = builder_map[key]()
        else:
            # Fall back to pymatgen for arbitrary miller indices
            slab = self._build_slab_pymatgen(
                element, miller_index, layers, vacuum, lattice_constant, supercell
            )
        
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
        binding_site = adsorbate_params.get('binding_site', 'ontop')
        distance = adsorbate_params.get('distance', 2.0)
        orientation = adsorbate_params.get('orientation', [0, 0, 1])
        
        # Get or create adsorbate molecule
        if mol_name in ADSORBATE_MOLECULES:
            mol_def = ADSORBATE_MOLECULES[mol_name]
            adsorbate = Atoms(
                symbols=mol_def['atoms'],
                positions=mol_def['positions']
            )
        else:
            # Try ASE's molecule database
            try:
                adsorbate = molecule(mol_name)
            except Exception:
                raise ValueError(f"Unknown adsorbate molecule: {mol_name}")
        
        # Rotate adsorbate to match orientation
        adsorbate = self._orient_adsorbate(adsorbate, orientation)
        
        # Find binding site position
        site_position = self._find_binding_site(slab, binding_site)
        
        # Add adsorbate
        add_adsorbate(slab, adsorbate, distance, position=site_position[:2])
        
        return slab
    
    def _orient_adsorbate(self, adsorbate: Atoms, orientation: List[float]) -> Atoms:
        """Rotate adsorbate to align with given orientation vector."""
        from ase.build import rotate
        
        # Default orientation is along z-axis
        current = np.array([0, 0, 1])
        target = np.array(orientation)
        target = target / np.linalg.norm(target)
        
        if not np.allclose(current, target):
            # Calculate rotation axis and angle
            axis = np.cross(current, target)
            if np.linalg.norm(axis) > 1e-6:
                axis = axis / np.linalg.norm(axis)
                angle = np.arccos(np.clip(np.dot(current, target), -1, 1))
                angle_deg = np.degrees(angle)
                adsorbate.rotate(angle_deg, axis, center='COM')
        
        return adsorbate
    
    def _find_binding_site(self, slab: Atoms, site_type: str) -> np.ndarray:
        """Find the position of a binding site on the surface."""
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
        
        if site_type == 'ontop':
            # On top of first surface atom
            return surface_positions[0]
        
        elif site_type == 'bridge':
            # Between two nearest surface atoms
            if len(surface_positions) >= 2:
                return (surface_positions[0] + surface_positions[1]) / 2
            return surface_positions[0]
        
        elif site_type in ['fcc', 'hcp', 'hollow']:
            # Center of three nearest surface atoms
            if len(surface_positions) >= 3:
                return np.mean(surface_positions[:3], axis=0)
            return np.mean(surface_positions, axis=0)
        
        else:
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