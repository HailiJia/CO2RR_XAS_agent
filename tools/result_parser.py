"""
Result Parser Tool for CO2RR XAS Agent
Skill 3 Implementation
"""

import os
import json
import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid
from .utils import ensure_dir, read_poscar, get_timestamp, generate_uuid


class FEFFParser:
    """Parse FEFF output files."""
    
    def parse(self, output_dir: str) -> Dict:
        """
        Parse FEFF xmu.dat file.
        
        Returns:
            Dictionary with energy and mu arrays
        """
        xmu_file = os.path.join(output_dir, "xmu.dat")
        
        if not os.path.exists(xmu_file):
            return {'status': 'error', 'message': f'xmu.dat not found in {output_dir}'}
        
        energy = []
        mu = []
        
        with open(xmu_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('#') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        energy.append(float(parts[0]))
                        mu.append(float(parts[2]))  # mu is typically column 3
                    except ValueError:
                        continue
        
        return {
            'status': 'success',
            'software': 'FEFF',
            'spectrum': {
                'energy': energy,
                'intensity': mu,
                'energy_unit': 'eV',
                'intensity_unit': 'arb. units',
                'spectrum_type': 'XANES'
            }
        }


class FDMNESParser:
    """Parse FDMNES output files."""
    
    def parse(self, output_dir: str) -> Dict:
        """
        Parse FDMNES *_conv.txt file.
        
        Returns:
            Dictionary with energy and intensity arrays
        """
        # Find conv file
        conv_file = None
        for filename in os.listdir(output_dir):
            if filename.endswith('_conv.txt'):
                conv_file = os.path.join(output_dir, filename)
                break
        
        if not conv_file:
            return {'status': 'error', 'message': f'No *_conv.txt file found in {output_dir}'}
        
        energy = []
        intensity = []
        
        with open(conv_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line.startswith('!') or not line:
                    continue
                
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        energy.append(float(parts[0]))
                        intensity.append(float(parts[1]))
                    except ValueError:
                        continue
        
        return {
            'status': 'success',
            'software': 'FDMNES',
            'spectrum': {
                'energy': energy,
                'intensity': intensity,
                'energy_unit': 'eV',
                'intensity_unit': 'arb. units',
                'spectrum_type': 'XANES'
            }
        }


class VASPXASParser:
    """Parse VASP XAS output files."""
    
    def parse(
        self,
        output_dir: str,
        mode: str = "trace"
    ) -> Dict:
        """
        Parse VASP OUTCAR for XAS spectrum.
        
        Args:
            output_dir: Directory containing OUTCAR
            mode: Extraction mode ('trace', 'parallel', 'normal', 'tauc')
            
        Returns:
            Dictionary with energy and intensity arrays
        """
        outcar_file = os.path.join(output_dir, "OUTCAR")
        
        if not os.path.exists(outcar_file):
            return {'status': 'error', 'message': f'OUTCAR not found in {output_dir}'}
        
        # Parse OUTCAR for imaginary dielectric function
        energy, intensity = self._extract_dielectric(outcar_file, mode)
        
        if not energy:
            return {'status': 'error', 'message': 'Could not extract dielectric function from OUTCAR'}
        
        return {
            'status': 'success',
            'software': 'VASP',
            'spectrum': {
                'energy': energy,
                'intensity': intensity,
                'energy_unit': 'eV',
                'intensity_unit': 'arb. units',
                'spectrum_type': 'XANES',
                'extraction_mode': mode
            }
        }
    
    def _extract_dielectric(
        self,
        outcar_file: str,
        mode: str
    ) -> Tuple[List[float], List[float]]:
        """
        Extract imaginary dielectric function from OUTCAR.
        
        Modes:
        - trace: xx + yy + zz
        - parallel: xx + yy
        - normal: zz only
        - tauc: sqrt(E^2 * trace)
        """
        energy = []
        intensity = []
        
        with open(outcar_file, 'r') as f:
            content = f.read()
        
        # Find the imaginary dielectric function section
        pattern = r"frequency dependent IMAGINARY DIELECTRIC FUNCTION"
        match = re.search(pattern, content)
        
        if not match:
            return [], []
        
        # Get lines after the header
        start_idx = match.end()
        lines = content[start_idx:].split('\n')
        
        # Skip header lines
        data_started = False
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # Check if this is a data line (starts with a number)
            try:
                e = float(parts[0])
                data_started = True
            except (ValueError, IndexError):
                if data_started:
                    break  # End of data section
                continue
            
            if len(parts) < 7:
                continue
            
            try:
                e = float(parts[0])
                xx = float(parts[1])
                yy = float(parts[2])
                zz = float(parts[3])
                xy = float(parts[4])
                yz = float(parts[5])
                zx = float(parts[6])
                
                # Check for NaN
                if any(np.isnan([xx, yy, zz, xy, yz, zx])):
                    continue
                
                # Calculate intensity based on mode
                if mode == "trace":
                    val = xx + yy + zz
                elif mode == "parallel":
                    val = xx + yy
                elif mode == "normal":
                    val = zz
                elif mode == "tauc":
                    val = np.sqrt(e * e * (xx + yy + zz))
                else:
                    val = xx + yy + zz
                
                energy.append(e)
                intensity.append(val)
                
            except (ValueError, IndexError):
                continue
        
        return energy, intensity
    
    def write_spectrum_file(
        self,
        output_dir: str,
        mode: str = "trace"
    ) -> Optional[str]:
        """
        Extract spectrum and write to CORE_DIELECTRIC_IMAG.dat
        """
        result = self.parse(output_dir, mode)
        
        if result['status'] != 'success':
            return None
        
        spectrum = result['spectrum']
        output_file = os.path.join(output_dir, "CORE_DIELECTRIC_IMAG.dat")
        
        with open(output_file, 'w') as f:
            f.write(f"# VASP XAS spectrum (mode: {mode})\n")
            f.write("# Energy(eV)  Intensity\n")
            for e, i in zip(spectrum['energy'], spectrum['intensity']):
                f.write(f"{e:12.6f}  {i:12.6f}\n")
        
        return output_file


class ISAACRecordGenerator:
    """Generate ISAAC AI-ready records."""
    
    def create_record(
        self,
        structure: Dict,
        software: str,
        spectrum: Dict,
        parameters: Dict,
        absorber: str,
        edge: str,
        description: str = "",
        tags: List[str] = None
    ) -> Dict:
        """
        Create ISAAC AI-ready record.
        
        Args:
            structure: Structure dictionary
            software: Software used (FEFF, FDMNES, VASP_GW, VASP_PBE)
            spectrum: Parsed spectrum data
            parameters: Calculation parameters
            absorber: Absorbing element
            edge: Absorption edge
            description: Record description
            tags: List of tags
            
        Returns:
            ISAAC record dictionary
        """
        record = {
            "record_id": generate_uuid(),
            "record_type": "simulation",
            "created_at": get_timestamp(),
            "simulation_type": "xas",
            "software": software,
            
            "structure": {
                "format": "POSCAR",
                "atoms": structure.get('atoms', []),
                "positions": structure.get('positions', []).tolist() if isinstance(structure.get('positions'), np.ndarray) else structure.get('positions', []),
                "cell": structure.get('cell', []).tolist() if isinstance(structure.get('cell'), np.ndarray) else structure.get('cell', []),
                "pbc": [True, True, False],
                "metadata": structure.get('metadata', {})
            },
            
            "parameters": {
                "edge": edge,
                "absorber": absorber,
                **parameters
            },
            
            "results": {
                "spectra": [
                    {
                        "type": spectrum.get('spectrum_type', 'XANES'),
                        "edge": edge,
                        "absorber": absorber,
                        "data": {
                            "energy": spectrum.get('energy', []),
                            "intensity": spectrum.get('intensity', []),
                            "energy_unit": spectrum.get('energy_unit', 'eV'),
                            "intensity_unit": spectrum.get('intensity_unit', 'arb. units')
                        }
                    }
                ],
                "metadata": {
                    "software": software,
                    "extraction_mode": spectrum.get('extraction_mode')
                }
            },
            
            "provenance": {
                "generator": "CO2RR-XAS Agent",
                "generator_version": "1.0.0",
                "generation_date": get_timestamp(),
                "workflow": "co2rr_xas_simulation"
            },
            
            "tags": tags or ["CO2RR", "XAS", software, absorber, edge],
            "description": description
        }
        
        return record
    
    def save_record(self, record: Dict, output_path: str) -> str:
        """Save record to JSON file."""
        ensure_dir(os.path.dirname(output_path))
        
        with open(output_path, 'w') as f:
            json.dump(record, f, indent=2)
        
        return output_path


class ResultParser:
    """
    Main result parser combining all software parsers.
    """
    
    def __init__(self):
        self.feff_parser = FEFFParser()
        self.fdmnes_parser = FDMNESParser()
        self.vasp_parser = VASPXASParser()
        self.record_gen = ISAACRecordGenerator()
    
    def parse_results(
        self,
        output_dir: str,
        software: str,
        vasp_mode: str = "trace"
    ) -> Dict:
        """
        Parse results from specified software.
        
        Args:
            output_dir: Directory containing output files
            software: Software name (FEFF, FDMNES, VASP, VASP_GW, VASP_PBE)
            vasp_mode: Mode for VASP extraction
            
        Returns:
            Parsed results dictionary
        """
        software = software.upper()
        
        if software == "FEFF":
            return self.feff_parser.parse(output_dir)
        elif software == "FDMNES" or software.startswith("FDMNES"):
            return self.fdmnes_parser.parse(output_dir)
        elif software.startswith("VASP"):
            return self.vasp_parser.parse(output_dir, mode=vasp_mode)
        else:
            return {'status': 'error', 'message': f'Unknown software: {software}'}
    
    def parse_and_create_record(
        self,
        output_dir: str,
        software: str,
        structure_file: str,
        absorber: str,
        edge: str,
        parameters: Dict = None,
        vasp_mode: str = "trace",
        description: str = "",
        tags: List[str] = None
    ) -> Dict:
        """
        Parse results and create ISAAC record.
        
        Returns:
            Dictionary with parsed results and ISAAC record
        """
        # Parse results
        results = self.parse_results(output_dir, software, vasp_mode)
        
        if results['status'] != 'success':
            return results
        
        # Read structure
        structure = read_poscar(structure_file)
        
        # Create ISAAC record
        record = self.record_gen.create_record(
            structure=structure,
            software=software,
            spectrum=results['spectrum'],
            parameters=parameters or {},
            absorber=absorber,
            edge=edge,
            description=description,
            tags=tags
        )
        
        # Save record
        record_path = os.path.join(output_dir, "isaac_record.json")
        self.record_gen.save_record(record, record_path)
        
        return {
            'status': 'success',
            'spectrum': results['spectrum'],
            'isaac_record': record,
            'record_path': record_path
        }


# =============================================================================
# Skill Interface Function
# =============================================================================

def execute_result_parsing(
    output_dir: str,
    software: str,
    structure_file: str,
    absorber: str,
    edge: str,
    parameters: Dict = None,
    vasp_mode: str = "trace",
    description: str = "",
    tags: List[str] = None
) -> Dict:
    """
    Execute Skill 3: Result Parsing and ISAAC Record Generation
    
    Args:
        output_dir: Directory containing simulation output
        software: Software used (FEFF, FDMNES, VASP_GW, VASP_PBE)
        structure_file: Path to structure file
        absorber: Absorbing element
        edge: Absorption edge
        parameters: Calculation parameters
        vasp_mode: VASP extraction mode
        description: Record description
        tags: Record tags
        
    Returns:
        Dictionary with results and ISAAC record
    """
    parser = ResultParser()
    
    return parser.parse_and_create_record(
        output_dir=output_dir,
        software=software,
        structure_file=structure_file,
        absorber=absorber,
        edge=edge,
        parameters=parameters,
        vasp_mode=vasp_mode,
        description=description,
        tags=tags
    )