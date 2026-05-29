"""
Agent Core - Main agent logic for CO2RR XAS calculations
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
from enum import Enum
import json

from skills.skill_registry import get_registry, SkillRegistry
from generators.slab_generator import SlabGenerator
from generators.vasp_input_generator import VASPXASGenerator
from generators.feff_input_generator import FEFFXASGenerator
from workflow.job_script_generator import JobScriptGenerator
from agent.parameter_extractor import ParameterExtractor


class WorkflowState(Enum):
    """States in the semi-autonomous workflow."""
    IDLE = "idle"
    PARAMETERS_EXTRACTED = "parameters_extracted"
    INPUTS_GENERATED = "inputs_generated"
    JOB_SCRIPT_READY = "job_script_ready"
    AWAITING_RESULTS = "awaiting_results"
    RESULTS_PARSED = "results_parsed"


class CO2RRXASAgent:
    """
    Main agent for CO2RR XAS calculations.
    
    Supports semi-autonomous workflow:
    1. User requests calculation
    2. Agent extracts parameters and generates inputs
    3. Agent generates job script
    4. User submits job manually
    5. User notifies agent when complete
    6. Agent parses results and suggests next steps
    """
    
    def __init__(
        self,
        skills_dir: Optional[str] = None,
        hpc_system: str = 'polaris',
        llm_client: Optional[Any] = None,
    ):
        """
        Initialize the agent.
        
        Args:
            skills_dir: Directory containing skill JSON files
            hpc_system: Target HPC system for job scripts
            llm_client: Optional LLM client for natural language processing
        """
        self.registry = get_registry(skills_dir)
        self.extractor = ParameterExtractor(llm_client)
        self.job_generator = JobScriptGenerator(system=hpc_system)
        
        # Generators
        self.slab_generator = SlabGenerator()
        self.vasp_generator = VASPXASGenerator()
        self.feff_generator = FEFFXASGenerator()
        
        # Workflow state
        self.state = WorkflowState.IDLE
        self.current_params: Dict[str, Any] = {}
        self.current_outputs: Dict[str, Any] = {}
        self.history: List[Dict[str, Any]] = []
    
    def list_available_skills(self) -> List[Dict[str, str]]:
        """List all available skills with descriptions."""
        skills = []
        for skill_id in self.registry.list_skills():
            skill = self.registry.get_skill(skill_id)
            skills.append({
                'id': skill_id,
                'name': skill.get('name', skill_id),
                'description': skill.get('description', ''),
            })
        return skills
    
    def process_request(
        self,
        request: str,
        output_dir: Optional[str] = None,
        project: str = "YOUR_PROJECT",
    ) -> Dict[str, Any]:
        """
        Process a natural language request.
        
        Args:
            request: Natural language request from user
            output_dir: Directory to write output files
            project: HPC project/allocation name
        
        Returns:
            Response dictionary with status and outputs
        """
        response = {
            'status': 'success',
            'message': '',
            'outputs': {},
            'next_steps': [],
        }
        
        try:
            # Determine request type
            request_lower = request.lower()
            
            if any(word in request_lower for word in ['slab', 'surface', 'interface']):
                response = self._handle_slab_request(request, output_dir)
            
            elif any(word in request_lower for word in ['xas', 'xanes', 'exafs', 'absorption']):
                if 'feff' in request_lower:
                    response = self._handle_feff_request(request, output_dir)
                else:
                    response = self._handle_vasp_xas_request(request, output_dir)
            
            elif 'job' in request_lower or 'submit' in request_lower:
                response = self._handle_job_script_request(request, output_dir, project)
            
            else:
                response['status'] = 'clarification_needed'
                response['message'] = (
                    "I can help with:\n"
                    "1. Generate surface slabs (e.g., 'Create Cu(111) slab with CO2')\n"
                    "2. Generate VASP XAS inputs (e.g., 'Setup XANES calculation for Cu K-edge')\n"
                    "3. Generate FEFF inputs (e.g., 'Create FEFF input for Cu K-edge XANES')\n"
                    "4. Generate job scripts (e.g., 'Create job script for VASP')\n"
                    "\nPlease specify what you'd like to do."
                )
            
            # Update history
            self.history.append({
                'request': request,
                'response': response,
                'state': self.state.value,
            })
            
        except Exception as e:
            response['status'] = 'error'
            response['message'] = f"Error processing request: {str(e)}"
        
        return response
    
    def _handle_slab_request(
        self, 
        request: str, 
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Handle slab generation request."""
        # Extract parameters
        params = self.extractor.extract_slab_parameters(request)
        
        if not params.get('element') or not params.get('miller_index'):
            return {
                'status': 'clarification_needed',
                'message': 'Please specify the element and surface orientation (e.g., Cu(111))',
                'extracted_params': params,
            }
        
        self.current_params = params
        self.state = WorkflowState.PARAMETERS_EXTRACTED
        
        # Generate slab
        outputs = self.slab_generator.run(
            user_params=params,
            output_dir=Path(output_dir) if output_dir else None,
        )
        
        self.current_outputs = outputs
        self.state = WorkflowState.INPUTS_GENERATED
        
        return {
            'status': 'success',
            'message': f"Generated {params['element']}{tuple(params['miller_index'])} slab",
            'outputs': {
                'poscar': outputs['poscar'],
                'params': outputs['params'],
            },
            'next_steps': [
                "Review the generated POSCAR",
                "Request VASP or FEFF XAS input generation",
                "Modify parameters if needed (e.g., 'change vacuum to 20 Å')",
            ],
        }
    
    def _handle_vasp_xas_request(
        self, 
        request: str, 
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Handle VASP XAS input generation request."""
        # Extract parameters
        params = self.extractor.extract_xas_parameters(request)
        
        # Check if we have a structure from previous step
        if 'structure' not in params:
            if 'structure' in self.current_outputs:
                params['structure'] = self.current_outputs['structure']
            else:
                return {
                    'status': 'clarification_needed',
                    'message': 'Please provide a structure first (generate a slab or provide a file path)',
                    'extracted_params': params,
                }
        
        if not params.get('edge'):
            return {
                'status': 'clarification_needed',
                'message': 'Please specify the absorption edge (e.g., K-edge, L3-edge)',
                'extracted_params': params,
            }
        
        if not params.get('absorbing_atom'):
            return {
                'status': 'clarification_needed',
                'message': 'Please specify the absorbing atom element',
                'extracted_params': params,
            }
        
        self.current_params.update(params)
        
        # Build context for edge-specific defaults
        context = {
            'edge': params['edge'],
            'spectroscopy_type': params.get('spectroscopy_type', 'XANES'),
        }
        
        # Generate VASP inputs
        outputs = self.vasp_generator.run(
            user_params=params,
            output_dir=Path(output_dir) if output_dir else None,
            context=context,
        )
        
        self.current_outputs.update(outputs)
        self.state = WorkflowState.INPUTS_GENERATED
        
        return {
            'status': 'success',
            'message': f"Generated VASP XAS inputs for {params['absorbing_atom']['element']} {params['edge']}-edge",
            'outputs': {
                'incar': outputs['incar'],
                'kpoints': outputs['kpoints'],
                'poscar': outputs['poscar'],
            },
            'next_steps': [
                "Review the generated input files",
                "Generate POTCAR using pymatgen or VASP tools",
                "Request job script generation",
            ],
        }
    
    def _handle_feff_request(
        self, 
        request: str, 
        output_dir: Optional[str]
    ) -> Dict[str, Any]:
        """Handle FEFF input generation request."""
        # Extract parameters
        params = self.extractor.extract_xas_parameters(request)
        
        # Check if we have a structure
        if 'structure' not in params:
            if 'structure' in self.current_outputs:
                params['structure'] = self.current_outputs['structure']
            else:
                return {
                    'status': 'clarification_needed',
                    'message': 'Please provide a structure first',
                    'extracted_params': params,
                }
        
        if not params.get('edge') or not params.get('absorbing_atom'):
            return {
                'status': 'clarification_needed',
                'message': 'Please specify the absorption edge and absorbing atom',
                'extracted_params': params,
            }
        
        self.current_params.update(params)
        
        # Build context
        context = {
            'edge': params['edge'],
            'spectroscopy_type': params.get('spectroscopy_type', 'XANES'),
        }
        
        # Generate FEFF inputs
        outputs = self.feff_generator.run(
            user_params=params,
            output_dir=Path(output_dir) if output_dir else None,
            context=context,
        )
        
        self.current_outputs.update(outputs)
        self.state = WorkflowState.INPUTS_GENERATED
        
        return {
            'status': 'success',
            'message': f"Generated FEFF input for {params['absorbing_atom']['element']} {params['edge']}-edge",
            'outputs': {
                'feff_inp': outputs['feff_inp'],
            },
            'next_steps': [
                "Review the generated feff.inp",
                "Request job script generation",
            ],
        }
    
    def _handle_job_script_request(
        self,
        request: str,
        output_dir: Optional[str],
        project: str,
    ) -> Dict[str, Any]:
        """Handle job script generation request."""
        request_lower = request.lower()
        
        if output_dir is None:
            return {
                'status': 'error',
                'message': 'Please specify an output directory for the job script',
            }
        
        output_path = Path(output_dir)
        
        if 'feff' in request_lower:
            script = self.job_generator.generate_feff_script(
                job_name='feff_xas',
                work_dir=str(output_path.absolute()),
                project=project,
            )
            script_name = 'submit_feff.sh'
        else:
            script = self.job_generator.generate_vasp_script(
                job_name='vasp_xas',
                work_dir=str(output_path.absolute()),
                project=project,
            )
            script_name = 'submit_vasp.sh'
        
        # Write script
        script_path = output_path / script_name
        self.job_generator.write_script(script, script_path)
        
        self.state = WorkflowState.JOB_SCRIPT_READY
        
        return {
            'status': 'success',
            'message': f"Generated job script: {script_name}",
            'outputs': {
                'job_script': script,
                'script_path': str(script_path),
            },
            'next_steps': [
                f"Review and edit {script_name} (update project name, walltime, etc.)",
                f"Submit with: qsub {script_name}",
                "Notify me when the job completes for result analysis",
            ],
        }
    
    def modify_parameters(self, modifications: Dict[str, Any]) -> Dict[str, Any]:
        """
        Modify current parameters with user-specified changes.
        
        Args:
            modifications: Dictionary of parameter modifications
        
        Returns:
            Updated parameters
        """
        def deep_update(base: dict, updates: dict) -> dict:
            for key, value in updates.items():
                if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                    deep_update(base[key], value)
                else:
                    base[key] = value
            return base
        
        self.current_params = deep_update(self.current_params, modifications)
        return self.current_params
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current workflow state and parameters."""
        return {
            'state': self.state.value,
            'params': self.current_params,
            'outputs_available': list(self.current_outputs.keys()),
        }
    
    def reset(self) -> None:
        """Reset agent state."""
        self.state = WorkflowState.IDLE
        self.current_params = {}
        self.current_outputs = {}