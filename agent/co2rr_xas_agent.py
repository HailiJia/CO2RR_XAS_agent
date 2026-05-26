"""
CO2RR XAS Agent
Main agent orchestration logic
"""

import os
import json
import re
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass

# Import tools
from tools.structure_generator import (
    StructureGenerator,
    execute_structure_generation
)
from tools.xas_input_generator import (
    XASInputGenerator,
    execute_xas_input_generation
)
from tools.result_parser import (
    ResultParser,
    execute_result_parsing
)
from tools.utils import (
    METAL_DATA, ADSORBATES, CO2RR_PATHWAY,
    read_poscar, ensure_dir
)


@dataclass
class ParsedIntent:
    """Parsed user intent from natural language."""
    action: str  # 'generate_structure', 'generate_xas', 'parse_results', 'full_workflow'
    metals: List[str]
    facets: List[str]
    adsorbates: List[str]
    full_pathway: bool
    structure_file: Optional[str]
    output_dir: str
    parameters: Dict[str, Any]


class IntentParser:
    """Parse natural language requests into structured intents."""
    
    def __init__(self):
    # Use case-sensitive chemical matching:
    # CO adsorbate should not be confused with Co metal.
        metal_keys = sorted(METAL_DATA.keys(), key=len, reverse=True)
        adsorbate_keys = sorted(ADSORBATES.keys(), key=len, reverse=True)

        self.metal_pattern = r'(?<![A-Za-z])(' + '|'.join(map(re.escape, metal_keys)) + r')(?![A-Za-z])'
        self.adsorbate_pattern = r'(?<![A-Za-z])(' + '|'.join(map(re.escape, adsorbate_keys)) + r')(?![A-Za-z])'
        self.facet_pattern = r'\b(111|100|110|0001)\b'
    
    def parse(self, request: str, output_dir: str = "output") -> ParsedIntent:
        """
        Parse natural language request.
        
        Args:
            request: Natural language request string
            output_dir: Default output directory
            
        Returns:
            ParsedIntent object
        """
        request_lower = request.lower()
        
        # Determine action
        if any(word in request_lower for word in ['parse', 'read output', 'isaac record', 'extract']):
            action = 'parse_results'
        elif any(word in request_lower for word in ['full workflow', 'complete', 'end-to-end']):
            action = 'full_workflow'
        elif any(word in request_lower for word in ['xas', 'x-ray', 'absorption', 'feff', 'fdmnes', 'edge']):
            action = 'generate_xas'
        else:
            action = 'generate_structure'

        # Extract metals
        metals = re.findall(self.metal_pattern, request)
        metals = list(dict.fromkeys(metals))
        
        # Extract facets
        facets = re.findall(self.facet_pattern, request)
        if not facets:
            facets = ['111']  # Default

        # Extract adsorbates
        adsorbates = re.findall(self.adsorbate_pattern, request)
        adsorbates = list(dict.fromkeys(adsorbates))
        
        # Check for full pathway
        full_pathway = any(phrase in request_lower for phrase in [
            'full pathway', 'complete pathway', 'all adsorbates',
            'reaction pathway', 'co2rr pathway'
        ])
        
        # Check for structure file
        structure_file = None
        file_patterns = [
            r'(?:from|read|use|file)\s+["\']?([^\s"\']+(?:POSCAR|CONTCAR|\.cif))["\']?',
            r'([^\s]+(?:POSCAR|CONTCAR|\.cif))'
        ]
        for pattern in file_patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                structure_file = match.group(1)
                break
        
        # Extract parameters
        parameters = {}

        # Default FDMNES options
        fdmnes_options = {
            "Quadrupole": False,
            "Relativism": False,
            "Spinorbit": False,
            "SCF": True,
            "SCFexc": False,
            "Screening": False,
            "Full_atom": False,
            "TDDFT": False,
            "Green": False,
        }

        def has_any(patterns):
            return any(re.search(p, request_lower) for p in patterns)

        def negated(term):
            return re.search(rf"\b(no|without|disable|disabled|exclude|excluding)\s+{term}\b", request_lower) is not None

        # FDMNES method
        if has_any([r"\bgreen\b", r"multiple\s+scattering", r"multiple-scattering"]):
            fdmnes_options["Green"] = True

        if has_any([r"\bfdm\b", r"finite\s+difference", r"finite-difference"]):
            fdmnes_options["Green"] = False

        # Boolean FDMNES cards
        if has_any([r"\bquadrupole\b"]):
            fdmnes_options["Quadrupole"] = True

        if has_any([r"\brelativism\b", r"\brelativistic\b"]):
            fdmnes_options["Relativism"] = True

        if has_any([r"\bspinorbit\b", r"spin\s*orbit", r"spin-orbit"]):
            fdmnes_options["Spinorbit"] = True

        if negated("scf"):
            fdmnes_options["SCF"] = False
        elif re.search(r"\bscf\b", request_lower):
            fdmnes_options["SCF"] = True

        if has_any([r"\bscfexc\b", r"scf\s*exc"]):
            fdmnes_options["SCFexc"] = True

        if has_any([r"\bscreening\b"]):
            fdmnes_options["Screening"] = True

        if has_any([r"\bfull_atom\b", r"full\s+atom", r"full-atom"]):
            fdmnes_options["Full_atom"] = True

        if has_any([r"\btddft\b", r"\btd-dft\b"]):
            fdmnes_options["TDDFT"] = True

        parameters["fdmnes_options"] = fdmnes_options

        # FDMNES / XAS radius in Angstrom
        radius_match = re.search(
            r"(?:radius|cluster\s+radius)\s*(?:=|is|to)?\s*([0-9]+(?:\.[0-9]+)?)",
            request_lower,
        )
        if radius_match:
            parameters["cluster_radius"] = float(radius_match.group(1))

        # FDMNES energy range: min step max
        range_match = re.search(
            r"(?:e_range|energy\s*range|range)\s*(?:=|is|:)?\s*(-?[0-9]+(?:\.[0-9]*)?)\s+([0-9]+(?:\.[0-9]*)?)\s+(-?[0-9]+(?:\.[0-9]*)?)",
            request_lower,
        )
        if range_match:
            parameters["fdmnes_energy_range"] = (
                float(range_match.group(1)),
                float(range_match.group(2)),
                float(range_match.group(3)),
            )

        # Optional edge override
        edge_match = re.search(r"\b(k|l2|l3|l23)\s*[- ]?edge\b", request_lower)
        if not edge_match:
            edge_match = re.search(r"\bedge\s*(?:=|is|:)?\s*(k|l2|l3|l23)\b", request_lower)

        if edge_match:
            parameters["edge_override"] = edge_match.group(1).upper()
                
            # Supercell
            supercell_match = re.search(r'(\d+)\s*[xX×]\s*(\d+)\s*supercell', request)
            if supercell_match:
                parameters['supercell'] = (int(supercell_match.group(1)), int(supercell_match.group(2)))
            
            # Layers
            layers_match = re.search(r'(\d+)\s*layer', request)
            if layers_match:
                parameters['layers'] = int(layers_match.group(1))
            
            # Site
            for site in ['top', 'bridge', 'fcc', 'hcp']:
                if site in request_lower:
                    parameters['site'] = site
                    break
            
            # NERSC account
            account_match = re.search(r'account\s+(\w+)', request, re.IGNORECASE)
            if account_match:
                parameters['nersc_account'] = account_match.group(1)
            
            # Email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', request)
            if email_match:
                parameters['email'] = email_match.group(0)
            
            return ParsedIntent(
                action=action,
                metals=metals,
                facets=facets,
                adsorbates=adsorbates,
                full_pathway=full_pathway,
                structure_file=structure_file,
                output_dir=output_dir,
                parameters=parameters
            )


class CO2RRXASAgent:
    """
    Main CO2RR XAS Agent.
    
    Orchestrates the three skills:
    1. Structure Generation
    2. XAS Input Generation
    3. Result Parsing & ISAAC Record
    """
    
    def __init__(self, default_output_dir: str = "co2rr_xas_output"):
        self.intent_parser = IntentParser()
        self.structure_gen = StructureGenerator()
        self.xas_gen = XASInputGenerator()
        self.result_parser = ResultParser()
        self.default_output_dir = default_output_dir
    
    def process_request(
        self,
        request: str,
        output_dir: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Process a natural language request.
        
        Args:
            request: Natural language request
            output_dir: Output directory (optional)
            **kwargs: Additional parameters to override
            
        Returns:
            Dictionary with results
        """
        output_dir = output_dir or self.default_output_dir
        
        # Parse intent
        intent = self.intent_parser.parse(request, output_dir)
        
        # Override with kwargs
        for key, value in kwargs.items():
            if hasattr(intent, key):
                setattr(intent, key, value)
            else:
                intent.parameters[key] = value
        
        # Execute based on action
        if intent.action == 'generate_structure':
            return self._execute_structure_generation(intent)
        elif intent.action == 'generate_xas':
            return self._execute_xas_generation(intent)
        elif intent.action == 'parse_results':
            return self._execute_result_parsing(intent)
        elif intent.action == 'full_workflow':
            return self._execute_full_workflow(intent)
        else:
            return {'status': 'error', 'message': f'Unknown action: {intent.action}'}
    
    def _execute_structure_generation(self, intent: ParsedIntent) -> Dict:
        """Execute structure generation skill."""
        if not intent.metals:
            return {'status': 'error', 'message': 'No metal element specified'}
        
        metal1 = intent.metals[0]
        metal2 = intent.metals[1] if len(intent.metals) > 1 else None
        facet1 = intent.facets[0]
        facet2 = intent.facets[1] if len(intent.facets) > 1 else None
        
        return execute_structure_generation(
            mode='read_file' if intent.structure_file else 'generate',
            structure_file=intent.structure_file,
            metal1=metal1,
            metal2=metal2,
            facet1=facet1,
            facet2=facet2,
            adsorbate=intent.adsorbates if intent.adsorbates else None,
            site=intent.parameters.get('site', 'top'),
            supercell=intent.parameters.get('supercell', (3, 3)),
            layers=intent.parameters.get('layers', 4),
            output_dir=intent.output_dir,
            full_pathway=intent.full_pathway
        )
    
    def _execute_xas_generation(self, intent: ParsedIntent) -> Dict:
        """Execute XAS input generation skill."""
        results = {'status': 'success', 'structures': []}
        
        # First generate structures if needed
        if not intent.structure_file:
            struct_result = self._execute_structure_generation(intent)
            if struct_result['status'] != 'success':
                return struct_result
            
            # Process each generated structure
            for files in struct_result['files']:
                poscar_path = files['poscar']
                struct_dir = os.path.dirname(poscar_path)
                base_dir = os.path.dirname(struct_dir)
                
                # Load metadata
                metadata_path = files['metadata']
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Generate XAS inputs
                xas_result = execute_xas_input_generation(
                    structure_file=poscar_path,
                    output_dir=base_dir,
                    nersc_account=intent.parameters.get('nersc_account', 'm5268'),
                    nersc_queue=intent.parameters.get('nersc_queue', 'regular'),
                    nersc_nodes=intent.parameters.get('nersc_nodes', 2),
                    nersc_walltime=intent.parameters.get('nersc_walltime', '8:00:00'),
                    email=intent.parameters.get('email'),
                    cluster_radius=intent.parameters.get('cluster_radius', 6.0),
                    structure_metadata=metadata,
                    fdmnes_options=intent.parameters.get("fdmnes_options"),
                    fdmnes_energy_range=intent.parameters.get("fdmnes_energy_range", (-5.0, 0.2, 50.0)),
                    edge_override=intent.parameters.get("edge_override"),
                )
                
                results['structures'].append({
                    'structure': metadata,
                    'xas_inputs': xas_result
                })
        else:
            # Use provided structure file
            xas_result = execute_xas_input_generation(
                structure_file=intent.structure_file,
                output_dir=intent.output_dir,
                nersc_account=intent.parameters.get('nersc_account', 'm5268'),
                nersc_queue=intent.parameters.get('nersc_queue', 'regular'),
                nersc_nodes=intent.parameters.get('nersc_nodes', 2),
                nersc_walltime=intent.parameters.get('nersc_walltime', '8:00:00'),
                email=intent.parameters.get('email'),
                cluster_radius=intent.parameters.get('cluster_radius', 6.0)
            )
            results['structures'].append({
                'structure_file': intent.structure_file,
                'xas_inputs': xas_result
            })
        
        return results
    
    def _execute_result_parsing(self, intent: ParsedIntent) -> Dict:
        """Execute result parsing skill."""
        # This requires output directory and software specification
        output_dir = intent.parameters.get('results_dir', intent.output_dir)
        software = intent.parameters.get('software', 'FEFF')
        structure_file = intent.structure_file or intent.parameters.get('structure_file')
        
        if not structure_file:
            return {'status': 'error', 'message': 'Structure file required for parsing'}
        
        absorber = intent.parameters.get('absorber', intent.metals[0] if intent.metals else 'Cu')
        edge = intent.parameters.get('edge', 'K')
        
        return execute_result_parsing(
            output_dir=output_dir,
            software=software,
            structure_file=structure_file,
            absorber=absorber,
            edge=edge,
            parameters=intent.parameters,
            vasp_mode=intent.parameters.get('vasp_mode', 'trace'),
            description=intent.parameters.get('description', ''),
            tags=intent.parameters.get('tags')
        )
    
    def _execute_full_workflow(self, intent: ParsedIntent) -> Dict:
        """Execute full workflow: structure -> XAS inputs."""
        # Force full pathway if not specified
        if not intent.adsorbates and not intent.full_pathway:
            intent.full_pathway = True
        
        if intent.full_pathway:
            intent.adsorbates = CO2RR_PATHWAY
        
        # Execute XAS generation (which includes structure generation)
        return self._execute_xas_generation(intent)
    
    def generate_structure(
        self,
        metal1: str,
        metal2: Optional[str] = None,
        facet1: str = "111",
        facet2: Optional[str] = None,
        adsorbate: Optional[Union[str, List[str]]] = None,
        site: str = "top",
        supercell: tuple = (3, 3),
        layers: int = 4,
        output_dir: Optional[str] = None,
        full_pathway: bool = False
    ) -> Dict:
        """
        Direct API for structure generation.
        
        Skill 1: Generate CO2RR structures
        """
        return execute_structure_generation(
            mode='generate',
            metal1=metal1,
            metal2=metal2,
            facet1=facet1,
            facet2=facet2,
            adsorbate=adsorbate,
            site=site,
            supercell=supercell,
            layers=layers,
            output_dir=output_dir or self.default_output_dir,
            full_pathway=full_pathway
        )
    
    def generate_xas_inputs(
        self,
        structure_file: str,
        output_dir: Optional[str] = None,
        nersc_account: str = "m5268",
        nersc_queue: str = "regular",
        email: Optional[str] = None,
        cluster_radius: float = 6.0
    ) -> Dict:
        """
        Direct API for XAS input generation.
        
        Skill 2: Generate XAS inputs for all methods
        """
        return execute_xas_input_generation(
            structure_file=structure_file,
            output_dir=output_dir or self.default_output_dir,
            nersc_account=nersc_account,
            nersc_queue=nersc_queue,
            email=email,
            cluster_radius=cluster_radius
        )
    
    def parse_results(
        self,
        output_dir: str,
        software: str,
        structure_file: str,
        absorber: str,
        edge: str,
        vasp_mode: str = "trace"
    ) -> Dict:
        """
        Direct API for result parsing.
        
        Skill 3: Parse results and create ISAAC record
        """
        return execute_result_parsing(
            output_dir=output_dir,
            software=software,
            structure_file=structure_file,
            absorber=absorber,
            edge=edge,
            vasp_mode=vasp_mode
        )


# =============================================================================
# Main Entry Point
# =============================================================================

def create_agent(output_dir: str = "co2rr_xas_output") -> CO2RRXASAgent:
    """Create and return a CO2RR XAS Agent instance."""
    return CO2RRXASAgent(default_output_dir=output_dir)


def process_request(request: str, output_dir: str = "co2rr_xas_output", **kwargs) -> Dict:
    """
    Process a natural language request.
    
    This is the main entry point for the Ask Sage agent.
    
    Examples:
        >>> process_request("Generate Cu(111) surface with CO adsorbate")
        >>> process_request("Full CO2RR pathway on Cu-Au interface")
        >>> process_request("XAS inputs for CHO on Pt(100)")
        >>> process_request("Parse FEFF results from ./output/FEFF/")
    
    Args:
        request: Natural language request
        output_dir: Output directory
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with results
    """
    agent = create_agent(output_dir)
    return agent.process_request(request, **kwargs)