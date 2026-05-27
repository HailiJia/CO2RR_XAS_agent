"""
Utility functions for CO2RR XAS Agent
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
import uuid
import re


# =============================================================================
# Constants
# =============================================================================

# Atomic numbers
ATOMIC_NUMBERS = {
    'H': 1, 'C': 6, 'N': 7, 'O': 8,
    'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25,
    'Fe': 26, 'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30,
    'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42, 'Ru': 44,
    'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48,
    'Hf': 72, 'Ta': 73, 'W': 74, 'Re': 75, 'Os': 76,
    'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80,
    'Al': 13
}

# Crystal structures and lattice constants (Angstrom)
METAL_DATA = {
    # FCC metals
    'Cu': {'structure': 'fcc', 'a': 3.615, 'facets': ['111', '100', '110']},
    'Ag': {'structure': 'fcc', 'a': 4.085, 'facets': ['111', '100', '110']},
    'Au': {'structure': 'fcc', 'a': 4.078, 'facets': ['111', '100', '110']},
    'Pt': {'structure': 'fcc', 'a': 3.924, 'facets': ['111', '100', '110']},
    'Pd': {'structure': 'fcc', 'a': 3.891, 'facets': ['111', '100', '110']},
    'Ni': {'structure': 'fcc', 'a': 3.524, 'facets': ['111', '100', '110']},
    'Al': {'structure': 'fcc', 'a': 4.050, 'facets': ['111', '100', '110']},
    'Rh': {'structure': 'fcc', 'a': 3.803, 'facets': ['111', '100', '110']},
    'Ir': {'structure': 'fcc', 'a': 3.839, 'facets': ['111', '100', '110']},
    # BCC metals
    'Fe': {'structure': 'bcc', 'a': 2.867, 'facets': ['110', '100', '111']},
    'Cr': {'structure': 'bcc', 'a': 2.910, 'facets': ['110', '100', '111']},
    'Mo': {'structure': 'bcc', 'a': 3.147, 'facets': ['110', '100', '111']},
    'W':  {'structure': 'bcc', 'a': 3.165, 'facets': ['110', '100', '111']},
    'V':  {'structure': 'bcc', 'a': 3.030, 'facets': ['110', '100', '111']},
    'Nb': {'structure': 'bcc', 'a': 3.301, 'facets': ['110', '100', '111']},
    'Ta': {'structure': 'bcc', 'a': 3.303, 'facets': ['110', '100', '111']},
    # HCP metals
    'Co': {'structure': 'hcp', 'a': 2.507, 'c': 4.069, 'facets': ['0001', '10-10']},
    'Zn': {'structure': 'hcp', 'a': 2.665, 'c': 4.947, 'facets': ['0001', '10-10']},
    'Ti': {'structure': 'hcp', 'a': 2.951, 'c': 4.686, 'facets': ['0001', '10-10']},
    'Zr': {'structure': 'hcp', 'a': 3.232, 'c': 5.147, 'facets': ['0001', '10-10']},
    'Ru': {'structure': 'hcp', 'a': 2.706, 'c': 4.282, 'facets': ['0001', '10-10']},
    'Os': {'structure': 'hcp', 'a': 2.735, 'c': 4.319, 'facets': ['0001', '10-10']},
}

# Edge energies (eV)
EDGE_ENERGIES = {
    # K-edges
    'C_K': 284, 'N_K': 410, 'O_K': 543,
    'Sc_K': 4492, 'Ti_K': 4966, 'V_K': 5465, 'Cr_K': 5989,
    'Mn_K': 6539, 'Fe_K': 7112, 'Co_K': 7709, 'Ni_K': 8333,
    'Cu_K': 8979, 'Zn_K': 9659,
    # L3-edges (for 4d and 5d)
    'Pd_L3': 3173, 'Ag_L3': 3351,
    'Pt_L3': 11564, 'Au_L3': 11919,
}

# Metal classification for edge selection
METALS_3D = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
METALS_4D = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
METALS_5D = ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

# CO2RR adsorbates
ADSORBATES = {
    'CO': {
        'atoms': ['C', 'O'],
        'positions': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.13]]),
        'binding_atom': 0
    },
    'CH': {
        'atoms': ['C', 'H'],
        'positions': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.09]]),
        'binding_atom': 0
    },
    'CH2': {
        'atoms': ['C', 'H', 'H'],
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.94, 0.54],
            [0.0, -0.94, 0.54]
        ]),
        'binding_atom': 0
    },
    'CHO': {
        'atoms': ['C', 'H', 'O'],
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.6, 0.0, 1.1]
        ]),
        'binding_atom': 0
    },
    'CHOH': {
        'atoms': ['C', 'H', 'O', 'H'],
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [1.09, 0.0, 0.0],
            [-0.6, 0.0, 1.2],
            [-0.6, 0.8, 1.6]
        ]),
        'binding_atom': 0
    },
    'CH3': {
        'atoms': ['C', 'H', 'H', 'H'],
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.0, 1.02, 0.36],
            [0.88, -0.51, 0.36],
            [-0.88, -0.51, 0.36]
        ]),
        'binding_atom': 0
    },
    'CH4': {
        'atoms': ['C', 'H', 'H', 'H', 'H'],
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.63, 0.63, 0.63],
            [-0.63, -0.63, 0.63],
            [-0.63, 0.63, -0.63],
            [0.63, -0.63, -0.63]
        ]),
        'binding_atom': 0
    },
    'COCO': {
        'atoms': ['C', 'O', 'C', 'O'],
        'positions': np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.2],
            [1.5, 0.0, 0.0],
            [1.5, 0.0, 1.2]
        ]),
        'binding_atom': 0
    },
    'OCCO': {
        # Crude *OCCO initial guess.
        # C-C backbone is roughly parallel to the surface.
        # binding_atom = first C. The second C is displaced along x.
        'atoms': ['O', 'C', 'C', 'O'],
        'positions': np.array([
            [-1.15, 0.0, 0.55],   # O attached to first C
            [0.00, 0.0, 0.00],    # first C, binding atom
            [1.45, 0.0, 0.00],    # second C
            [2.60, 0.0, 0.55]     # O attached to second C
        ]),
        'binding_atom': 1
    },

}

# Full CO2RR pathway
CO2RR_PATHWAY = ['CO', 'CH', 'CH2', 'CHO', 'CHOH', 'CH3', 'CH4', 'COCO', 'OCCO']

# Adsorption site offsets (fractional coordinates for FCC 111)
ADSORPTION_SITES = {
    'top': np.array([0.0, 0.0]),
    'bridge': np.array([0.5, 0.0]),
    'fcc': np.array([1/3, 1/3]),
    'hcp': np.array([2/3, 2/3]),
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_edge_for_element(element: str) -> Tuple[str, bool]:
    """
    Determine the appropriate XAS edge for an element.
    
    Returns:
        Tuple of (edge_name, vasp_allowed)
        - edge_name: 'K' or 'L23'
        - vasp_allowed: True if VASP can be used for this edge
    """
    if element in METALS_3D or element in ['C', 'N', 'O']:
        return 'K', True
    elif element in METALS_4D:
        return 'L23', False
    elif element in METALS_5D:
        return 'L23', False
    else:
        # Default to K-edge
        return 'K', True


def get_edge_energy(element: str, edge: str) -> float:
    """Get edge energy in eV for element and edge type."""
    if edge == 'K':
        key = f"{element}_K"
    else:
        key = f"{element}_L3"
    return EDGE_ENERGIES.get(key, 0.0)


def contains_carbon(adsorbate: str) -> bool:
    """Check if adsorbate contains carbon."""
    if adsorbate is None:
        return False
    return 'C' in ADSORBATES.get(adsorbate, {}).get('atoms', [])


def generate_uuid() -> str:
    """Generate a UUID string."""
    return str(uuid.uuid4())


def get_timestamp() -> str:
    """Get current timestamp in ISO8601 format."""
    return datetime.utcnow().isoformat() + "Z"


def ensure_dir(path: str) -> str:
    """Ensure directory exists, create if not."""
    os.makedirs(path, exist_ok=True)
    return path


def read_poscar(filepath: str) -> Dict:
    """
    Read VASP POSCAR/CONTCAR file.
    
    Returns:
        Dictionary with atoms, positions, cell, etc.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    comment = lines[0].strip()
    scale = float(lines[1].strip())
    
    # Lattice vectors
    cell = np.zeros((3, 3))
    for i in range(3):
        cell[i] = [float(x) for x in lines[2 + i].split()]
    cell *= scale
    
    # Element symbols (VASP 5 format)
    elements_line = lines[5].split()
    try:
        counts = [int(x) for x in elements_line]
        # VASP 4 format - no element symbols
        elements = None
        counts_line_idx = 5
    except ValueError:
        # VASP 5 format - has element symbols
        elements = elements_line
        counts = [int(x) for x in lines[6].split()]
        counts_line_idx = 6
    
    # Check for Selective dynamics
    next_line = lines[counts_line_idx + 1].strip()
    if next_line[0].lower() == 's':
        selective_dynamics = True
        coord_type = lines[counts_line_idx + 2].strip()
        pos_start = counts_line_idx + 3
    else:
        selective_dynamics = False
        coord_type = next_line
        pos_start = counts_line_idx + 2
    
    # Read positions
    n_atoms = sum(counts)
    positions = np.zeros((n_atoms, 3))
    
    for i in range(n_atoms):
        parts = lines[pos_start + i].split()
        positions[i] = [float(parts[j]) for j in range(3)]
    
    # Convert to Cartesian if Direct
    if coord_type[0].lower() == 'd':
        positions = np.dot(positions, cell)
    
    # Build atoms list
    atoms = []
    if elements:
        for elem, count in zip(elements, counts):
            atoms.extend([elem] * count)
    
    return {
        'comment': comment,
        'atoms': atoms,
        'positions': positions,
        'cell': cell,
        'counts': counts,
        'elements': elements,
        'selective_dynamics': selective_dynamics
    }


def write_poscar(
    atoms: List[str],
    positions: np.ndarray,
    cell: np.ndarray,
    filename: str,
    comment: str = "Generated by CO2RR-XAS Agent",
    selective_dynamics: bool = True,
    fix_bottom: bool = True
) -> str:
    """
    Write VASP POSCAR file.
    
    Args:
        atoms: List of element symbols
        positions: Cartesian coordinates (n_atoms, 3)
        cell: Cell matrix (3, 3)
        filename: Output filename
        comment: Comment line
        selective_dynamics: Include selective dynamics
        fix_bottom: Fix bottom half of atoms
        
    Returns:
        Path to written file
    """
    # Get unique elements and counts (preserve order)
    unique_elements = []
    counts = []
    for atom in atoms:
        if atom not in unique_elements:
            unique_elements.append(atom)
            counts.append(1)
        else:
            counts[unique_elements.index(atom)] += 1
    
    # Sort atoms by element
    sorted_indices = []
    for elem in unique_elements:
        for i, atom in enumerate(atoms):
            if atom == elem:
                sorted_indices.append(i)
    
    sorted_positions = positions[sorted_indices]
    
    # Convert to fractional coordinates
    # Lattice vectors are stored as rows throughout this code.
    # read_poscar uses: cart = frac @ cell
    # Therefore the inverse conversion is: frac = cart @ inv(cell)
    frac_positions = np.dot(sorted_positions, np.linalg.inv(cell))

    # Keep slab atoms inside the periodic surface cell in x/y.
    # Do not wrap z, because vacuum/slab-height issues should remain visible.
    frac_positions[:, :2] = frac_positions[:, :2] % 1.0
    
    # Determine which atoms to fix
    z_coords = sorted_positions[:, 2]
    z_mid = (z_coords.max() + z_coords.min()) / 2
    
    with open(filename, 'w') as f:
        f.write(f"{comment}\n")
        f.write("1.0\n")
        
        for vec in cell:
            f.write(f"  {vec[0]:16.10f}  {vec[1]:16.10f}  {vec[2]:16.10f}\n")
        
        f.write("  " + "  ".join(unique_elements) + "\n")
        f.write("  " + "  ".join(map(str, counts)) + "\n")
        
        if selective_dynamics:
            f.write("Selective dynamics\n")
        f.write("Direct\n")
        
        for i, pos in enumerate(frac_positions):
            line = f"  {pos[0]:16.10f}  {pos[1]:16.10f}  {pos[2]:16.10f}"
            if selective_dynamics:
                if fix_bottom and z_coords[i] < z_mid:
                    line += "  F F F"
                else:
                    line += "  T T T"
            f.write(line + "\n")
    
    return filename


def read_cif(filepath: str) -> Dict:
    """Read CIF using read_structure_file. Requires pymatgen or ase."""
    return read_structure_file(filepath)

def read_structure_file(filepath: str) -> Dict:
    """
    Read a structure file and return the internal structure dictionary.

    Supported directly:
    - POSCAR / CONTCAR / VASP-format files via read_poscar
    - CIF via pymatgen first, then ASE fallback if installed

    The returned positions are Cartesian coordinates and cell vectors are rows,
    consistent with the rest of this codebase.
    """
    path = os.path.expanduser(os.path.expandvars(filepath))
    lower = path.lower()
    name = os.path.basename(path).lower()

    if lower.endswith('.cif'):
        # Prefer pymatgen because it robustly handles symmetry-expanded CIFs.
        try:
            from pymatgen.core import Structure as PMGStructure

            pmg = PMGStructure.from_file(path)
            atoms = [str(site.specie.symbol) for site in pmg.sites]
            positions = np.array([site.coords for site in pmg.sites], dtype=float)
            cell = np.array(pmg.lattice.matrix, dtype=float)
            return {
                'comment': os.path.basename(path),
                'atoms': atoms,
                'positions': positions,
                'cell': cell,
                'metadata': {'source_file': filepath, 'source_format': 'cif'},
            }
        except ImportError:
            pass

        try:
            from ase.io import read as ase_read

            atoms_obj = ase_read(path)
            atoms = atoms_obj.get_chemical_symbols()
            positions = np.array(atoms_obj.get_positions(), dtype=float)
            cell = np.array(atoms_obj.get_cell().array, dtype=float)
            return {
                'comment': os.path.basename(path),
                'atoms': atoms,
                'positions': positions,
                'cell': cell,
                'metadata': {'source_file': filepath, 'source_format': 'cif'},
            }
        except ImportError as exc:
            raise ImportError(
                'CIF input requires pymatgen or ase. Install one of them, e.g. '
                '`pip install pymatgen` or `pip install ase`.'
            ) from exc

    # CONTCAR has the same format as POSCAR. Other VASP-like filenames are also accepted.
    if name in {'poscar', 'contcar'} or 'poscar' in name or 'contcar' in name or not lower.endswith('.cif'):
        structure = read_poscar(path)
        structure.setdefault('metadata', {})
        structure['metadata'].update({'source_file': filepath, 'source_format': 'vasp'})
        return structure

    raise ValueError(f'Unsupported structure file format: {filepath}')
