"""
Structure Generator Tool for CO2RR XAS Agent
Skill 1 Implementation
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from .utils import (
    METAL_DATA, ADSORBATES, ADSORPTION_SITES, CO2RR_PATHWAY,
    write_poscar, read_poscar, read_structure_file, ensure_dir, generate_uuid, get_timestamp
)


class StructureGenerator:
    """
    Generate atomic structures for CO2RR simulations.
    
    Supports:
    - FCC, BCC, HCP metal surfaces
    - Various facets (111, 100, 110, 0001)
    - Bimetallic interfaces
    - CO2RR adsorbate placement
    """
    
    def __init__(self):
        self.metal_data = METAL_DATA
        self.adsorbates = ADSORBATES
        self.sites = ADSORPTION_SITES
    
    def generate_fcc_surface(
        self,
        element: str,
        facet: str = "111",
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        vacuum: float = 15.0,
        lattice_constant: Optional[float] = None
    ) -> Dict:
        """
        Generate FCC metal surface slab.
        
        Args:
            element: Metal element symbol
            facet: Miller index ('111', '100', '110')
            supercell: (nx, ny) supercell dimensions
            layers: Number of atomic layers
            vacuum: Vacuum thickness in Angstrom
            lattice_constant: Override default lattice constant
            
        Returns:
            Structure dictionary
        """
        a = lattice_constant or self.metal_data[element]['a']
        
        if facet == "111":
            return self._generate_fcc_111(element, a, supercell, layers, vacuum)
        elif facet == "100":
            return self._generate_fcc_100(element, a, supercell, layers, vacuum)
        elif facet == "110":
            return self._generate_fcc_110(element, a, supercell, layers, vacuum)
        else:
            raise ValueError(f"Unsupported facet: {facet}")
    
    def _generate_fcc_111(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float
    ) -> Dict:
        """Generate FCC(111) surface."""
        # Surface vectors
        a1 = a / np.sqrt(2) * np.array([1.0, 0.0, 0.0])
        a2 = a / np.sqrt(2) * np.array([0.5, np.sqrt(3)/2, 0.0])
        
        # Layer spacing
        layer_spacing = a / np.sqrt(3)
        
        # ABC stacking
        stacking = [
            np.array([0.0, 0.0]),
            np.array([1/3, 1/3]),
            np.array([2/3, 2/3]),
        ]
        
        atoms = []
        positions = []
        
        for layer in range(layers):
            shift = stacking[layer % 3]
            z = layer * layer_spacing
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)
        
        positions = np.array(positions)
        
        # Center and add vacuum
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum
        # One-sided vacuum: slab starts at z = 0; vacuum is above the slab
        
        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height]
        ])
        
        return {
            'atoms': atoms,
            'positions': positions,
            'cell': cell,
            'metadata': {
                'element': element,
                'structure': 'fcc',
                'facet': '111',
                'supercell': list(supercell),
                'layers': layers,
                'vacuum': vacuum,
                'lattice_constant': a
            }
        }
    
    def _generate_fcc_100(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float
    ) -> Dict:
        """Generate FCC(100) surface."""
        # Surface vectors
        a1 = a / np.sqrt(2) * np.array([1.0, 0.0, 0.0])
        a2 = a / np.sqrt(2) * np.array([0.0, 1.0, 0.0])
        
        # Layer spacing
        layer_spacing = a / 2
        
        # AB stacking
        stacking = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
        ]
        
        atoms = []
        positions = []
        
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)
        
        positions = np.array(positions)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum
        # One-sided vacuum: slab starts at z = 0; vacuum is above the slab
        
        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height]
        ])
        
        return {
            'atoms': atoms,
            'positions': positions,
            'cell': cell,
            'metadata': {
                'element': element,
                'structure': 'fcc',
                'facet': '100',
                'supercell': list(supercell),
                'layers': layers,
                'vacuum': vacuum,
                'lattice_constant': a
            }
        }
    
    def _generate_fcc_110(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float
    ) -> Dict:
        """Generate FCC(110) surface."""
        # Surface vectors
        a1 = a / np.sqrt(2) * np.array([1.0, 0.0, 0.0])
        a2 = a * np.array([0.0, 1.0, 0.0])
        
        # Layer spacing
        layer_spacing = a / (2 * np.sqrt(2))
        
        # AB stacking
        stacking = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
        ]
        
        atoms = []
        positions = []
        
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)
        
        positions = np.array(positions)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum
        # One-sided vacuum: slab starts at z = 0; vacuum is above the slab
        
        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height]
        ])
        
        return {
            'atoms': atoms,
            'positions': positions,
            'cell': cell,
            'metadata': {
                'element': element,
                'structure': 'fcc',
                'facet': '110',
                'supercell': list(supercell),
                'layers': layers,
                'vacuum': vacuum,
                'lattice_constant': a
            }
        }
    
    def generate_bcc_surface(
        self,
        element: str,
        facet: str = "110",
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        vacuum: float = 15.0,
        lattice_constant: Optional[float] = None
    ) -> Dict:
        """Generate BCC metal surface slab."""
        a = lattice_constant or self.metal_data[element]['a']
        
        if facet == "110":
            return self._generate_bcc_110(element, a, supercell, layers, vacuum)
        elif facet == "100":
            return self._generate_bcc_100(element, a, supercell, layers, vacuum)
        else:
            raise ValueError(f"Unsupported BCC facet: {facet}")
    
    def _generate_bcc_110(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float
    ) -> Dict:
        """Generate BCC(110) surface."""
        # Surface vectors
        a1 = a * np.array([1.0, 0.0, 0.0])
        a2 = a / np.sqrt(2) * np.array([0.0, 1.0, 0.0])
        
        # Layer spacing
        layer_spacing = a / np.sqrt(2)
        
        # AB stacking
        stacking = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
        ]
        
        atoms = []
        positions = []
        
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)
        
        positions = np.array(positions)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum
        # One-sided vacuum: slab starts at z = 0; vacuum is above the slab
        
        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height]
        ])
        
        return {
            'atoms': atoms,
            'positions': positions,
            'cell': cell,
            'metadata': {
                'element': element,
                'structure': 'bcc',
                'facet': '110',
                'supercell': list(supercell),
                'layers': layers,
                'vacuum': vacuum,
                'lattice_constant': a
            }
        }
    
    def _generate_bcc_100(
        self,
        element: str,
        a: float,
        supercell: Tuple[int, int],
        layers: int,
        vacuum: float
    ) -> Dict:
        """Generate BCC(100) surface."""
        a1 = a * np.array([1.0, 0.0, 0.0])
        a2 = a * np.array([0.0, 1.0, 0.0])
        layer_spacing = a / 2
        
        stacking = [
            np.array([0.0, 0.0]),
            np.array([0.5, 0.5]),
        ]
        
        atoms = []
        positions = []
        
        for layer in range(layers):
            shift = stacking[layer % 2]
            z = layer * layer_spacing
            
            for i in range(supercell[0]):
                for j in range(supercell[1]):
                    pos = (i + shift[0]) * a1 + (j + shift[1]) * a2
                    pos[2] = z
                    atoms.append(element)
                    positions.append(pos)
        
        positions = np.array(positions)
        positions[:, 2] -= positions[:, 2].min()
        slab_height = positions[:, 2].max()
        cell_height = slab_height + vacuum
        # One-sided vacuum: slab starts at z = 0; vacuum is above the slab
        
        cell = np.array([
            supercell[0] * a1,
            supercell[1] * a2,
            [0.0, 0.0, cell_height]
        ])
        
        return {
            'atoms': atoms,
            'positions': positions,
            'cell': cell,
            'metadata': {
                'element': element,
                'structure': 'bcc',
                'facet': '100',
                'supercell': list(supercell),
                'layers': layers,
                'vacuum': vacuum,
                'lattice_constant': a
            }
        }
    
    def generate_surface(
        self,
        element: str,
        facet: str = "111",
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        vacuum: float = 15.0
    ) -> Dict:
        """
        Generate metal surface based on crystal structure.
        Automatically detects FCC/BCC/HCP.
        """
        if element not in self.metal_data:
            raise ValueError(f"Unknown element: {element}. Supported: {list(self.metal_data.keys())}")
        
        structure = self.metal_data[element]['structure']
        
        if structure == 'fcc':
            return self.generate_fcc_surface(element, facet, supercell, layers, vacuum)
        elif structure == 'bcc':
            return self.generate_bcc_surface(element, facet, supercell, layers, vacuum)
        elif structure == 'hcp':
            # HCP implementation would go here
            raise NotImplementedError("HCP surface generation not yet implemented")
        else:
            raise ValueError(f"Unknown structure type: {structure}")
    
    def generate_interface(
        self,
        element1: str,
        element2: str,
        facet1: str = "111",
        facet2: str = "111",
        supercell: Tuple[int, int] = (3, 3),
        layers1: int = 3,
        layers2: int = 3,
        vacuum: float = 15.0,
        interface_distance: float = 2.5
    ) -> Dict:
        """
        Generate bimetallic interface structure.
        
        Args:
            element1: Bottom metal element
            element2: Top metal element
            facet1: Facet for bottom metal
            facet2: Facet for top metal
            supercell: Supercell dimensions
            layers1: Layers of bottom metal
            layers2: Layers of top metal
            vacuum: Vacuum thickness
            interface_distance: Distance between metals at interface
            
        Returns:
            Interface structure dictionary
        """
        # Generate both surfaces without vacuum
        surf1 = self.generate_surface(element1, facet1, supercell, layers1, vacuum=0)
        surf2 = self.generate_surface(element2, facet2, supercell, layers2, vacuum=0)
        
        pos1 = np.array(surf1['positions'])
        pos2 = np.array(surf2['positions'])
        
        # Stack: place surf2 above surf1
        z_offset = pos1[:, 2].max() + interface_distance
        pos2[:, 2] += z_offset
        
        # Combine
        all_atoms = surf1['atoms'] + surf2['atoms']
        all_positions = np.vstack([pos1, pos2])
        
        # Add vacuum
        all_positions[:, 2] -= all_positions[:, 2].min()
        slab_height = all_positions[:, 2].max()
        cell_height = slab_height + vacuum
        # One-sided vacuum: slab starts at z = 0; vacuum is above the slab
        
        # Use cell from surf1 (assuming compatible)
        cell = surf1['cell'].copy()
        cell[2, 2] = cell_height
        
        return {
            'atoms': all_atoms,
            'positions': all_positions,
            'cell': cell,
            'metadata': {
                'element1': element1,
                'element2': element2,
                'structure_type': 'interface',
                'facet1': facet1,
                'facet2': facet2,
                'supercell': list(supercell),
                'layers1': layers1,
                'layers2': layers2,
                'vacuum': vacuum,
                'interface_distance': interface_distance
            }
        }
    
    def add_adsorbate(
        self,
        structure: Dict,
        adsorbate_name: str,
        site: str = "top",
        height: float = 2.0,
        site_index: int = 0
    ) -> Dict:
        """
        Add adsorbate to surface structure.
        
        Args:
            structure: Surface structure dictionary
            adsorbate_name: Name of adsorbate (CO, CHO, etc.)
            site: Adsorption site (top, bridge, fcc, hcp)
            height: Height above surface in Angstrom
            site_index: Index of surface atom for site reference
            
        Returns:
            New structure with adsorbate
        """
        if adsorbate_name not in self.adsorbates:
            raise ValueError(f"Unknown adsorbate: {adsorbate_name}. Available: {list(self.adsorbates.keys())}")
        
        ads_data = self.adsorbates[adsorbate_name]
        ads_atoms = ads_data['atoms']
        ads_positions = ads_data['positions'].copy()
        binding_idx = ads_data['binding_atom']
        
        positions = np.array(structure['positions'])
        cell = structure['cell']
        
        # Find top layer atoms
        z_max = positions[:, 2].max()
        top_mask = np.abs(positions[:, 2] - z_max) < 0.5
        top_indices = np.where(top_mask)[0]
        
        if site_index >= len(top_indices):
            site_index = 0
        
        # Reference position
        ref_pos = positions[top_indices[site_index]].copy()
        
        # Apply site offset
        site_offset = self.sites.get(site, np.array([0.0, 0.0]))
        
        # Get surface vectors for offset calculation
        supercell = structure['metadata'].get('supercell', [3, 3])
        a1 = cell[0] / supercell[0]
        a2 = cell[1] / supercell[1]
        
        base_pos = ref_pos.copy()
        base_pos[:2] += site_offset[0] * a1[:2] + site_offset[1] * a2[:2]
        base_pos[2] = z_max + height
        
        # Position adsorbate
        ads_positions -= ads_positions[binding_idx]  # Center on binding atom
        ads_positions += base_pos
        
        # Combine
        new_atoms = structure['atoms'].copy() + ads_atoms
        new_positions = np.vstack([positions, ads_positions])
        
        # Update metadata
        new_metadata = structure['metadata'].copy()
        new_metadata['adsorbate'] = adsorbate_name
        new_metadata['adsorption_site'] = site
        new_metadata['adsorption_height'] = height
        
        return {
            'atoms': new_atoms,
            'positions': new_positions,
            'cell': structure['cell'].copy(),
            'metadata': new_metadata
        }
    
    def generate_co2rr_pathway(
        self,
        element: str,
        facet: str = "111",
        adsorbates: Optional[List[str]] = None,
        supercell: Tuple[int, int] = (3, 3),
        layers: int = 4,
        site: str = "top",
        element2: Optional[str] = None,
        facet2: Optional[str] = None
    ) -> List[Dict]:
        """
        Generate complete CO2RR reaction pathway structures.
        
        Args:
            element: Primary catalyst metal
            facet: Surface facet
            adsorbates: List of adsorbates (default: full pathway)
            supercell: Supercell dimensions
            layers: Number of layers
            site: Adsorption site
            element2: Secondary metal for interface
            facet2: Facet for secondary metal
            
        Returns:
            List of structure dictionaries
        """
        if adsorbates is None:
            adsorbates = CO2RR_PATHWAY
        
        structures = []
        
        # Generate base surface or interface
        if element2:
            base_surface = self.generate_interface(
                element, element2, facet, facet2 or facet,
                supercell, layers, layers
            )
        else:
            base_surface = self.generate_surface(
                element, facet, supercell, layers
            )
        
        # Clean surface
        clean = base_surface.copy()
        clean['metadata'] = base_surface['metadata'].copy()
        clean['metadata']['step'] = 'clean'
        structures.append(clean)
        
        # Each adsorbate
        for ads in adsorbates:
            if element2:
                surface = self.generate_interface(
                    element, element2, facet, facet2 or facet,
                    supercell, layers, layers
                )
            else:
                surface = self.generate_surface(
                    element, facet, supercell, layers
                )
            
            structure = self.add_adsorbate(surface, ads, site=site)
            structure['metadata']['step'] = ads
            structures.append(structure)
        
        return structures
    
    def save_structure(
        self,
        structure: Dict,
        output_dir: str,
        name: Optional[str] = None
    ) -> Dict[str, str]:
        """
        Save structure to files.
        
        Returns:
            Dictionary with paths to saved files
        """
        ensure_dir(output_dir)
        
        # Generate name if not provided
        if name is None:
            meta = structure['metadata']
            parts = []
            if 'element' in meta:
                parts.append(meta['element'])
            if 'element1' in meta:
                parts.append(f"{meta['element1']}-{meta['element2']}")
            if 'facet' in meta:
                parts.append(meta['facet'])
            if 'adsorbate' in meta:
                parts.append(meta['adsorbate'])
            name = "_".join(parts) if parts else "structure"
        
        # Write POSCAR
        poscar_path = os.path.join(output_dir, "POSCAR")
        write_poscar(
            structure['atoms'],
            structure['positions'],
            structure['cell'],
            poscar_path,
            comment=f"CO2RR: {name}"
        )
        
        # Write metadata JSON
        metadata_path = os.path.join(output_dir, "structure_info.json")
        metadata = structure['metadata'].copy()
        metadata['n_atoms'] = len(structure['atoms'])
        metadata['composition'] = {}
        for atom in structure['atoms']:
            metadata['composition'][atom] = metadata['composition'].get(atom, 0) + 1
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'poscar': poscar_path,
            'metadata': metadata_path
        }


# =============================================================================
# Skill Interface Function
# =============================================================================

def execute_structure_generation(
    mode: str = "generate",
    structure_file: Optional[str] = None,
    metal1: Optional[str] = None,
    metal2: Optional[str] = None,
    facet1: str = "111",
    facet2: Optional[str] = None,
    adsorbate: Optional[Union[str, List[str]]] = None,
    site: str = "top",
    supercell: Tuple[int, int] = (3, 3),
    layers: int = 4,
    vacuum: float = 15.0,
    output_dir: str = "output",
    full_pathway: bool = False
) -> Dict:
    """
    Execute Skill 1: Structure Generation
    
    This is the main entry point for the structure generation skill.
    
    Args:
        mode: 'read_file' or 'generate'
        structure_file: Path to existing structure file (for read_file mode)
        metal1: Primary metal element
        metal2: Secondary metal for interface
        facet1: Facet for primary metal
        facet2: Facet for secondary metal
        adsorbate: Adsorbate name or list of names
        site: Adsorption site
        supercell: Supercell dimensions
        layers: Number of layers
        vacuum: Vacuum thickness
        output_dir: Output directory
        full_pathway: Generate full CO2RR pathway
        
    Returns:
        Dictionary with results and file paths
    """
    generator = StructureGenerator()
    results = {
        'status': 'success',
        'structures': [],
        'files': []
    }
    
    if mode == "read_file":
        # Read existing structure
        if not structure_file or not os.path.exists(structure_file):
            return {'status': 'error', 'message': f'Structure file not found: {structure_file}'}
        
        structure = read_structure_file(structure_file)
        structure['metadata'] = {'source_file': structure_file}
        
        struct_dir = os.path.join(output_dir, "structure")
        files = generator.save_structure(structure, struct_dir)
        results['structures'].append(structure['metadata'])
        results['files'].append(files)
        
    elif mode == "generate":
        if not metal1:
            return {'status': 'error', 'message': 'metal1 is required for generate mode'}
        
        if full_pathway:
            # Generate full CO2RR pathway
            structures = generator.generate_co2rr_pathway(
                element=metal1,
                facet=facet1,
                supercell=supercell,
                layers=layers,
                site=site,
                element2=metal2,
                facet2=facet2
            )
            
            for struct in structures:
                step = struct['metadata'].get('step', 'unknown')
                struct_dir = os.path.join(output_dir, step, "structure")
                files = generator.save_structure(struct, struct_dir, name=step)
                results['structures'].append(struct['metadata'])
                results['files'].append(files)
                
        else:
            # Generate single structure
            if metal2:
                structure = generator.generate_interface(
                    metal1, metal2, facet1, facet2 or facet1,
                    supercell, layers, layers, vacuum
                )
            else:
                structure = generator.generate_surface(
                    metal1, facet1, supercell, layers, vacuum
                )
            
            # Add adsorbate(s)
            if adsorbate:
                if isinstance(adsorbate, str):
                    adsorbate = [adsorbate]
                
                for ads in adsorbate:
                    struct = generator.add_adsorbate(structure.copy(), ads, site=site)
                    struct_dir = os.path.join(output_dir, ads, "structure")
                    files = generator.save_structure(struct, struct_dir)
                    results['structures'].append(struct['metadata'])
                    results['files'].append(files)
            else:
                # Clean surface
                struct_dir = os.path.join(output_dir, "clean", "structure")
                files = generator.save_structure(structure, struct_dir)
                results['structures'].append(structure['metadata'])
                results['files'].append(files)
    
    return results