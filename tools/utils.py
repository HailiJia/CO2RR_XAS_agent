"""
Utility functions for CO2RR XAS Agent tools.
"""

import os
import uuid
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

import numpy as np


# =============================================================================
# Atomic data
# =============================================================================

ATOMIC_NUMBERS = {
    'H': 1, 'He': 2, 'Li': 3, 'Be': 4, 'B': 5, 'C': 6, 'N': 7, 'O': 8, 'F': 9, 'Ne': 10,
    'Na': 11, 'Mg': 12, 'Al': 13, 'Si': 14, 'P': 15, 'S': 16, 'Cl': 17, 'Ar': 18,
    'K': 19, 'Ca': 20, 'Sc': 21, 'Ti': 22, 'V': 23, 'Cr': 24, 'Mn': 25, 'Fe': 26,
    'Co': 27, 'Ni': 28, 'Cu': 29, 'Zn': 30, 'Ga': 31, 'Ge': 32, 'As': 33, 'Se': 34,
    'Br': 35, 'Kr': 36, 'Rb': 37, 'Sr': 38, 'Y': 39, 'Zr': 40, 'Nb': 41, 'Mo': 42,
    'Tc': 43, 'Ru': 44, 'Rh': 45, 'Pd': 46, 'Ag': 47, 'Cd': 48, 'In': 49, 'Sn': 50,
    'Sb': 51, 'Te': 52, 'I': 53, 'Xe': 54, 'Cs': 55, 'Ba': 56, 'La': 57, 'Ce': 58,
    'Pr': 59, 'Nd': 60, 'Pm': 61, 'Sm': 62, 'Eu': 63, 'Gd': 64, 'Tb': 65, 'Dy': 66,
    'Ho': 67, 'Er': 68, 'Tm': 69, 'Yb': 70, 'Lu': 71, 'Hf': 72, 'Ta': 73, 'W': 74,
    'Re': 75, 'Os': 76, 'Ir': 77, 'Pt': 78, 'Au': 79, 'Hg': 80, 'Tl': 81, 'Pb': 82,
    'Bi': 83, 'Po': 84, 'At': 85, 'Rn': 86, 'Fr': 87, 'Ra': 88, 'Ac': 89, 'Th': 90,
    'Pa': 91, 'U': 92, 'Np': 93, 'Pu': 94, 'Am': 95, 'Cm': 96,
}

# Approximate K-edge energies (eV) for common elements
K_EDGE_ENERGIES = {
    'C': 284, 'N': 410, 'O': 543, 'F': 697,
    'Na': 1071, 'Mg': 1303, 'Al': 1560, 'Si': 1839, 'P': 2146, 'S': 2472, 'Cl': 2822,
    'K': 3608, 'Ca': 4039, 'Sc': 4493, 'Ti': 4966, 'V': 5465, 'Cr': 5989, 'Mn': 6539,
    'Fe': 7112, 'Co': 7709, 'Ni': 8333, 'Cu': 8979, 'Zn': 9659,
    'Ga': 10367, 'Ge': 11103, 'As': 11867, 'Se': 12658, 'Br': 13474,
}

# Approximate L3-edge energies (eV) for common elements
L3_EDGE_ENERGIES = {
    'Fe': 707, 'Co': 778, 'Ni': 853, 'Cu': 933, 'Zn': 1022,
    'Mo': 2520, 'Ru': 2838, 'Rh': 3004, 'Pd': 3173, 'Ag': 3351,
    'W': 10207, 'Re': 10535, 'Os': 10871, 'Ir': 11215, 'Pt': 11564, 'Au': 11919,
}



# Common transition metals used by the agent.
METALS_3D = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn']
METALS_4D = ['Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd']
METALS_5D = ['Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg']

METAL_DATA = {
    'Al': {'structure': 'fcc', 'a': 4.05},
    'Fe': {'structure': 'bcc', 'a': 2.87},
    'Co': {'structure': 'hcp', 'a': 2.51, 'c': 4.07},
    'Ni': {'structure': 'fcc', 'a': 3.52},
    'Cu': {'structure': 'fcc', 'a': 3.615},
    'Zn': {'structure': 'hcp', 'a': 2.66, 'c': 4.95},
    'Mo': {'structure': 'bcc', 'a': 3.15},
    'Ru': {'structure': 'hcp', 'a': 2.71, 'c': 4.28},
    'Rh': {'structure': 'fcc', 'a': 3.80},
    'Pd': {'structure': 'fcc', 'a': 3.89},
    'Ag': {'structure': 'fcc', 'a': 4.09},
    'Ir': {'structure': 'fcc', 'a': 3.84},
    'Pt': {'structure': 'fcc', 'a': 3.92},
    'Au': {'structure': 'fcc', 'a': 4.08},
}

EDGE_ENERGIES = {'K': K_EDGE_ENERGIES, 'L3': L3_EDGE_ENERGIES}

ADSORBATES = {
    'CO': {'atoms': ['C', 'O'], 'positions': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.16]]), 'binding_atom': 0},
    'CO2': {'atoms': ['C', 'O', 'O'], 'positions': np.array([[0.0, 0.0, 0.0], [1.14,  0.0, 0.65], [-1.14, 0.0, 0.65]]),'binding_atom': 0},
    'H': {'atoms': ['H'], 'positions': np.array([[0.0, 0.0, 0.0]]), 'binding_atom': 0},
    'OH': {'atoms': ['O', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.97]]), 'binding_atom': 0},
    'H2O': {'atoms': ['O', 'H', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [0.76, 0.59, 0.0], [-0.76, 0.59, 0.0]]), 'binding_atom': 0},
    'CHO': {'atoms': ['C', 'O', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [1.094, 0.0, 0.387], [-0.363, 0.943, 0.363]]), 'binding_atom': 0},
    'CHOH': {'atoms': ['C', 'H', 'O', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [-0.6, 0.8, 0.2], [1.2, 0.0, 0.3], [1.7, 0.0, 1.1]]), 'binding_atom': 0},
    'CH': {'atoms': ['C', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.09]]), 'binding_atom': 0},
    'CH2': {'atoms': ['C', 'H', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.6], [-0.9, 0.0, 0.6]]), 'binding_atom': 0},
    'CH3': {'atoms': ['C', 'H', 'H', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.6], [-0.45, 0.78, 0.6], [-0.45, -0.78, 0.6]]), 'binding_atom': 0},
    'CH4': {'atoms': ['C', 'H', 'H', 'H', 'H'], 'positions': np.array([[0.0, 0.0, 0.0], [0.63, 0.63, 0.63], [-0.63, -0.63, 0.63], [-0.63, 0.63, -0.63], [0.63, -0.63, -0.63]]), 'binding_atom': 0},
    'COCO': {'atoms': ['C', 'O', 'C', 'O'], 'positions': np.array([[0.0, 0.0, 0.0], [1.15, 0.0, 0.2], [-1.35, 0.0, 0.0], [-2.5, 0.0, 0.2]]), 'binding_atom': 0},
    'OCCO': {'atoms': ['O', 'C', 'C', 'O'],'positions': np.array([[0.0, 0.0, 1.22],[0.0, 0.0, 0.0],[1.42, 0.0, 0.0],[1.42, 0.0, 1.22]]),'binding_atom': 1}
}

ADSORPTION_SITES = {
    'top': np.array([0.0, 0.0]),
    'bridge': np.array([0.5, 0.0]),
    'fcc': np.array([1.0 / 3.0, 1.0 / 3.0]),
    'hcp': np.array([2.0 / 3.0, 2.0 / 3.0]),
}

CO2RR_PATHWAY = ['CO', 'CHO', 'CHOH', 'CH3', 'CH4', 'COCO', 'OCCO']

# =============================================================================
# File and directory utilities
# =============================================================================

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def generate_uuid() -> str:
    """
    Generate a unique ID for ISAAC records.
    Format: 01JFH8SIMXAS + 14 random hex characters
    """
    prefix = "01JFH8SIMXAS"
    suffix = uuid.uuid4().hex[:14].upper()
    return f"{prefix}{suffix}"


def get_timestamp() -> str:
    """Get current UTC timestamp in ISO format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_file(filepath: str) -> str:
    """Calculate SHA256 hash of a file."""
    p = Path(filepath)
    if not p.exists() or not p.is_file():
        return "not_available"
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()




# =============================================================================
# Workflow defaults
# =============================================================================

def get_env_default(name: str, default: Any) -> Any:
    """Return a CO2RR workflow default from the environment when present."""
    return os.environ.get(name, default)

def get_nersc_defaults() -> Dict[str, Any]:
    """Centralized NERSC defaults; callers can override with CO2RR_NERSC_* env vars."""
    return {
        "account": get_env_default("CO2RR_NERSC_ACCOUNT", "mXXXX"),
        "queue": get_env_default("CO2RR_NERSC_QUEUE", "regular"),
        "nodes": int(get_env_default("CO2RR_NERSC_NODES", "2")),
        "walltime": get_env_default("CO2RR_NERSC_WALLTIME", "05:00:00"),
        "cluster": get_env_default("CO2RR_NERSC_CLUSTER", "Perlmutter"),
        "facility_name": get_env_default("CO2RR_NERSC_FACILITY", "NERSC"),
        "organization": get_env_default("CO2RR_NERSC_ORGANIZATION", "LBNL"),
    }

# =============================================================================
# Element and edge utilities
# =============================================================================

def get_edge_for_element(element: str) -> Tuple[str, float]:
    """
    Get default edge and approximate edge energy for an element.
    
    Args:
        element: Element symbol (e.g., 'Cu', 'Fe')
    
    Returns:
        Tuple of (edge_name, edge_energy_eV)
    """
    z = ATOMIC_NUMBERS.get(element, 0)
    
    # K-edge for lighter elements (Z <= 30, up to Zn)
    if z <= 30:
        energy = K_EDGE_ENERGIES.get(element, _approximate_k_edge_energy(z))
        return ("K", energy)
    
    # L3-edge for heavier elements
    energy = L3_EDGE_ENERGIES.get(element, _approximate_l3_edge_energy(z))
    return ("L3", energy)


def _approximate_k_edge_energy(z: int) -> float:
    """Approximate K-edge energy in eV using Moseley's law."""
    if z <= 0:
        return 0.0
    return 13.6 * (z - 2) ** 2


def _approximate_l3_edge_energy(z: int) -> float:
    """Approximate L3-edge energy in eV."""
    if z <= 0:
        return 0.0
    # Rough approximation for L3 edge
    return 13.6 * ((z - 7.4) / 2) ** 2


def get_element_from_z(z: int) -> Optional[str]:
    """Get element symbol from atomic number."""
    inverse = {v: k for k, v in ATOMIC_NUMBERS.items()}
    return inverse.get(z)


def get_edge_energy(element: str, edge: Optional[str] = None) -> float:
    """Return an approximate edge energy for an element/edge."""
    default_edge, default_energy = get_edge_for_element(element)
    edge_name = (edge or default_edge).upper()
    if edge_name == 'K':
        return K_EDGE_ENERGIES.get(element, _approximate_k_edge_energy(ATOMIC_NUMBERS.get(element, 0)))
    if edge_name in {'L3', 'L2', 'L23'}:
        return L3_EDGE_ENERGIES.get(element, _approximate_l3_edge_energy(ATOMIC_NUMBERS.get(element, 0)))
    return default_energy


def contains_carbon(structure: Dict[str, Any]) -> bool:
    """Return True if a structure dictionary contains carbon."""
    return 'C' in structure.get('atoms', [])


# =============================================================================
# Structure reading utilities
# =============================================================================

def read_structure_file(filepath: str) -> Dict[str, Any]:
    """
    Read structure from various file formats.
    
    Supports: POSCAR, CONTCAR, CIF, XYZ, and other formats via pymatgen/ASE.
    
    Args:
        filepath: Path to structure file
    
    Returns:
        Dictionary with:
            - atoms: List of element symbols
            - positions: List of [x, y, z] coordinates (Cartesian, Angstroms)
            - cell: 3x3 lattice matrix
            - metadata: Source file info
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Structure file not found: {filepath}")
    
    name = filepath.name.upper()
    suffix = filepath.suffix.lower()

    if name in ('POSCAR', 'CONTCAR') or suffix in ('.vasp', '.poscar'):
        try:
            structure = read_poscar(str(filepath))
            structure.setdefault('metadata', {})
            structure['metadata'].setdefault('source_file', str(filepath))
            structure['metadata'].setdefault('format', 'poscar_builtin')
            return structure
        except Exception:
            pass
    
    # Try pymatgen first
    try:
        from pymatgen.core import Structure, Lattice
        from pymatgen.io.vasp import Poscar
        
        if name in ('POSCAR', 'CONTCAR') or suffix in ('.vasp', '.poscar'):
            poscar = Poscar.from_file(str(filepath))
            structure = poscar.structure
        elif suffix == '.cif':
            structure = Structure.from_file(str(filepath))
        elif suffix == '.xyz':
            # XYZ files are molecules, need special handling
            from pymatgen.io.xyz import XYZ
            mol = XYZ.from_file(str(filepath)).molecule
            # Create a large cubic box for the molecule
            lattice = Lattice.cubic(50.0)
            structure = Structure(
                lattice, 
                mol.species, 
                mol.cart_coords, 
                coords_are_cartesian=True
            )
        else:
            # Try generic pymatgen reader
            structure = Structure.from_file(str(filepath))
        
        return {
            'atoms': [str(site.specie) for site in structure],
            'positions': structure.cart_coords.tolist(),
            'cell': structure.lattice.matrix.tolist(),
            'metadata': {
                'source_file': str(filepath),
                'format': 'pymatgen',
                'num_atoms': len(structure),
            }
        }
    except ImportError:
        pass
    except Exception as e:
        # Try ASE as fallback
        pass
    
    # Try ASE as fallback
    try:
        from ase.io import read as ase_read
        atoms = ase_read(str(filepath))
        
        return {
            'atoms': atoms.get_chemical_symbols(),
            'positions': atoms.get_positions().tolist(),
            'cell': atoms.get_cell().tolist(),
            'metadata': {
                'source_file': str(filepath),
                'format': 'ase',
                'num_atoms': len(atoms),
            }
        }
    except ImportError:
        pass
    except Exception:
        pass
    
    raise ValueError(
        f"Could not read structure from {filepath}. "
        "Install pymatgen or ase: pip install pymatgen ase"
    )


def read_poscar(filepath: str) -> Dict[str, Any]:
    """
    Read VASP POSCAR/CONTCAR file.
    
    This is a specialized reader that doesn't require pymatgen/ASE.
    
    Args:
        filepath: Path to POSCAR file
    
    Returns:
        Dictionary with atoms, positions, cell, metadata
    """
    filepath = Path(filepath)
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    # Line 0: Comment
    comment = lines[0].strip()
    
    # Line 1: Scale factor
    scale = float(lines[1].strip())
    
    # Lines 2-4: Lattice vectors
    cell = []
    for i in range(2, 5):
        vec = [float(x) * scale for x in lines[i].split()]
        cell.append(vec)
    
    # Line 5: Element symbols (VASP 5+) or counts (VASP 4)
    line5 = lines[5].split()
    
    # Check if line 5 contains element symbols or numbers
    try:
        int(line5[0])
        # VASP 4 format: no element symbols
        elements = None
        counts = [int(x) for x in line5]
        coord_start = 7
    except ValueError:
        # VASP 5+ format: element symbols on line 5
        elements = line5
        counts = [int(x) for x in lines[6].split()]
        coord_start = 8
    
    # Check for Selective dynamics
    if lines[coord_start - 1].strip().lower().startswith('s'):
        coord_start += 1
    
    # Check coordinate type
    coord_line = lines[coord_start - 1].strip().lower()
    is_cartesian = coord_line.startswith('c') or coord_line.startswith('k')
    
    # Read positions
    positions = []
    total_atoms = sum(counts)
    
    for i in range(coord_start, coord_start + total_atoms):
        parts = lines[i].split()
        pos = [float(parts[j]) for j in range(3)]
        
        if not is_cartesian:
            # Convert fractional to Cartesian
            cart = [0.0, 0.0, 0.0]
            for j in range(3):
                for k in range(3):
                    cart[k] += pos[j] * cell[j][k]
            pos = cart
        
        positions.append(pos)
    
    # Build atom list
    atoms = []
    if elements:
        for elem, count in zip(elements, counts):
            atoms.extend([elem] * count)
    else:
        # No element info, use placeholders
        for i, count in enumerate(counts):
            atoms.extend([f"X{i+1}"] * count)
    
    return {
        'atoms': atoms,
        'positions': positions,
        'cell': cell,
        'metadata': {
            'source_file': str(filepath),
            'format': 'poscar',
            'comment': comment,
            'num_atoms': total_atoms,
        }
    }


# =============================================================================
# Structure writing utilities
# =============================================================================

def write_poscar(
    atoms: List[str],
    positions: List[List[float]],
    cell: List[List[float]],
    filepath: str,
    comment: str = "Generated by CO2RR XAS Agent",
    direct: bool = False,
    selective_dynamics: Optional[List[List[bool]]] = None
) -> None:
    """
    Write VASP POSCAR file.
    
    Args:
        atoms: List of element symbols
        positions: List of [x, y, z] coordinates
        cell: 3x3 lattice matrix
        filepath: Output file path
        comment: Comment line
        direct: If True, write fractional coordinates
        selective_dynamics: Optional list of [T/F, T/F, T/F] for each atom
    """
    filepath = Path(filepath)
    ensure_dir(str(filepath.parent))
    
    # Count elements
    element_counts = {}
    element_order = []
    for atom in atoms:
        if atom not in element_counts:
            element_counts[atom] = 0
            element_order.append(atom)
        element_counts[atom] += 1
    
    # Convert positions if needed
    if direct:
        # Convert Cartesian to fractional
        cell_inv = np.linalg.inv(np.array(cell))
        frac_positions = []
        for pos in positions:
            frac = np.dot(cell_inv, pos)
            frac_positions.append(frac.tolist())
        write_positions = frac_positions
    else:
        write_positions = positions
    
    with open(filepath, 'w') as f:
        # Comment
        f.write(f"{comment}\n")
        
        # Scale
        f.write("1.0\n")
        
        # Lattice vectors
        for vec in cell:
            f.write(f"  {vec[0]:20.14f}  {vec[1]:20.14f}  {vec[2]:20.14f}\n")
        
        # Element symbols
        f.write("  " + "  ".join(element_order) + "\n")
        
        # Element counts
        f.write("  " + "  ".join(str(element_counts[e]) for e in element_order) + "\n")
        
        # Selective dynamics
        if selective_dynamics:
            f.write("Selective dynamics\n")
        
        # Coordinate type
        if direct:
            f.write("Direct\n")
        else:
            f.write("Cartesian\n")
        
        # Positions (sorted by element)
        atom_indices = []
        for elem in element_order:
            for i, atom in enumerate(atoms):
                if atom == elem:
                    atom_indices.append(i)
        
        for idx in atom_indices:
            pos = write_positions[idx]
            line = f"  {pos[0]:20.14f}  {pos[1]:20.14f}  {pos[2]:20.14f}"
            
            if selective_dynamics:
                sd = selective_dynamics[idx]
                flags = ["T" if x else "F" for x in sd]
                line += f"  {flags[0]}  {flags[1]}  {flags[2]}"
            
            f.write(line + "\n")