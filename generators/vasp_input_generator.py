"""
VASP Input Generator - Generate VASP input files for XAS calculations using Lightshow
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import warnings

from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar, Kpoints, Incar

# Import from your Lightshow fork
try:
    from lightshow.parameters.vasp import VASPParameters
    LIGHTSHOW_AVAILABLE = True
except ImportError:
    LIGHTSHOW_AVAILABLE = False
    warnings.warn("Lightshow not available. Using fallback VASP parameter generation.")

from generators.base_generator import BaseGenerator


class VASPXASGenerator(BaseGenerator):
    """Generate VASP input files for XAS calculations."""
    
    def __init__(self):
        super().__init__(skill_id='vasp_xas_calculation')
        self.potcar_dir = None  # Set this to your POTCAR directory
    
    def set_potcar_directory(self, potcar_dir: str) -> None:
        """Set the directory containing POTCAR files."""
        self.potcar_dir = Path(potcar_dir)
    
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate VASP input files for XAS calculation.
        
        Args:
            params: Validated parameters
        
        Returns:
            Dictionary with INCAR, POSCAR, KPOINTS content
        """
        structure = params['structure']
        if isinstance(structure, str):
            structure = Structure.from_file(structure)
        elif isinstance(structure, dict):
            structure = Structure.from_dict(structure)
        
        absorbing_atom = params['absorbing_atom']
        edge = params['edge']
        
        # Build context for conditional defaults
        context = {
            'edge': edge,
            'spectroscopy_type': params.get('spectroscopy_type', 'XANES')
        }
        
        # Re-merge with context to get edge-specific defaults
        full_params = self.registry.merge_with_defaults(
            self.skill_id, params, context=context
        )
        
        if LIGHTSHOW_AVAILABLE:
            outputs = self._generate_with_lightshow(structure, full_params)
        else:
            outputs = self._generate_fallback(structure, full_params)
        
        outputs['params'] = full_params
        return outputs
    
    def _generate_with_lightshow(
        self, 
        structure: Structure, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate using Lightshow library."""
        absorbing_atom = params['absorbing_atom']
        edge = params['edge']
        
        # Determine absorbing atom index
        if 'index' in absorbing_atom:
            absorbing_index = absorbing_atom['index']
        else:
            # Find first atom of the specified element
            element = absorbing_atom['element']
            for i, site in enumerate(structure):
                if str(site.specie) == element:
                    absorbing_index = i
                    break
            else:
                raise ValueError(f"No {element} atom found in structure")
        
        # Create Lightshow VASP parameters
        vasp_params = VASPParameters(
            structure=structure,
            absorbing_atom=absorbing_index,
            edge=edge,
            xanes=params.get('spectroscopy_type') == 'XANES',
            exafs=params.get('spectroscopy_type') == 'EXAFS',
        )
        
        # Get generated content
        incar_dict = vasp_params.get_incar()
        
        # Override with user-specified parameters
        user_overrides = {
            'ENCUT': params.get('encut'),
            'NBANDS': params.get('nbands'),
            'ISMEAR': params.get('ismear'),
            'SIGMA': params.get('sigma'),
            'CH_NEDOS': params.get('ch_nedos'),
        }
        
        for key, value in user_overrides.items():
            if value is not None:
                incar_dict[key] = value
        
        # Generate KPOINTS
        kpoints_params = params.get('kpoints', {})
        kpoints = Kpoints.gamma_automatic(
            kpts=kpoints_params.get('grid', [2, 2, 1])
        )
        
        # Generate POSCAR
        poscar = Poscar(structure)
        
        return {
            'incar': Incar(incar_dict).get_str(),
            'incar_dict': incar_dict,
            'poscar': poscar.get_str(),
            'kpoints': kpoints.get_str(),
            'structure': structure,
            'absorbing_index': absorbing_index,
        }
    
    def _generate_fallback(
        self, 
        structure: Structure, 
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fallback generation without Lightshow."""
        absorbing_atom = params['absorbing_atom']
        edge = params['edge']
        
        # Determine absorbing atom index
        if 'index' in absorbing_atom:
            absorbing_index = absorbing_atom['index']
        else:
            element = absorbing_atom['element']
            for i, site in enumerate(structure):
                if str(site.specie) == element:
                    absorbing_index = i
                    break
            else:
                raise ValueError(f"No {element} atom found in structure")
        
        # Build INCAR for XAS
        incar_dict = self._build_xas_incar(params, edge, absorbing_index, structure)
        
        # Generate KPOINTS
        kpoints_params = params.get('kpoints', {})
        kpoints = Kpoints.gamma_automatic(
            kpts=kpoints_params.get('grid', [2, 2, 1])
        )
        
        # Generate POSCAR
        poscar = Poscar(structure)
        
        return {
            'incar': Incar(incar_dict).get_str(),
            'incar_dict': incar_dict,
            'poscar': poscar.get_str(),
            'kpoints': kpoints.get_str(),
            'structure': structure,
            'absorbing_index': absorbing_index,
        }
    
    def _build_xas_incar(
        self, 
        params: Dict[str, Any], 
        edge: str,
        absorbing_index: int,
        structure: Structure
    ) -> Dict[str, Any]:
        """Build INCAR dictionary for XAS calculation."""
        # Base parameters
        incar = {
            'SYSTEM': 'XAS calculation',
            'PREC': 'Accurate',
            'ENCUT': params.get('encut', 520),
            'ISMEAR': params.get('ismear', 0),
            'SIGMA': params.get('sigma', 0.05),
            'EDIFF': 1e-6,
            'NELM': 200,
            'LREAL': 'Auto',
            'ALGO': 'Normal',
            'LWAVE': False,
            'LCHARG': False,
        }
        
        # XAS-specific parameters
        core_hole = params.get('core_hole', 'FCH')
        
        if core_hole != 'none':
            incar['ICORELEVEL'] = 2  # Core-level shift calculation
            incar['CLNT'] = absorbing_index + 1  # 1-indexed in VASP
            
            # Edge-specific quantum numbers
            edge_qn = {
                'K': {'CLN': 1, 'CLL': 0, 'CLZ': 1.0},
                'L1': {'CLN': 2, 'CLL': 0, 'CLZ': 1.0},
                'L2': {'CLN': 2, 'CLL': 1, 'CLZ': 0.5},
                'L3': {'CLN': 2, 'CLL': 1, 'CLZ': 0.5},
                'M1': {'CLN': 3, 'CLL': 0, 'CLZ': 1.0},
                'M2': {'CLN': 3, 'CLL': 1, 'CLZ': 0.5},
                'M3': {'CLN': 3, 'CLL': 1, 'CLZ': 0.5},
            }
            
            if edge in edge_qn:
                incar.update(edge_qn[edge])
        
        # Spectroscopy parameters
        if params.get('spectroscopy_type') == 'XANES':
            incar['CH_LSPEC'] = True
            incar['CH_NEDOS'] = params.get('ch_nedos', 2000)
            incar['NBANDS'] = params.get('nbands', 200)
        
        # Functional
        functional = params.get('functional', 'PBE')
        if functional == 'PBE+U':
            incar['LDAU'] = True
            incar['LDAUTYPE'] = 2
        elif functional == 'HSE06':
            incar['LHFCALC'] = True
            incar['HFSCREEN'] = 0.2
        elif functional == 'SCAN':
            incar['METAGGA'] = 'SCAN'
        
        return incar
    
    def write_files(self, outputs: Dict[str, Any], output_dir: Path) -> None:
        """Write VASP input files."""
        output_dir = Path(output_dir)
        
        # Write INCAR
        with open(output_dir / 'INCAR', 'w') as f:
            f.write(outputs['incar'])
        
        # Write POSCAR
        with open(output_dir / 'POSCAR', 'w') as f:
            f.write(outputs['poscar'])
        
        # Write KPOINTS
        with open(output_dir / 'KPOINTS', 'w') as f:
            f.write(outputs['kpoints'])
        
        # Note: POTCAR generation requires access to POTCAR files
        # which are licensed and cannot be distributed
        potcar_note = output_dir / 'POTCAR_NOTE.txt'
        with open(potcar_note, 'w') as f:
            f.write("POTCAR file must be generated separately.\n")
            f.write("Use pymatgen or VASP's potcar generation tools.\n")
            elements = [str(site.specie) for site in outputs['structure']]
            unique_elements = list(dict.fromkeys(elements))
            f.write(f"Required elements: {', '.join(unique_elements)}\n")