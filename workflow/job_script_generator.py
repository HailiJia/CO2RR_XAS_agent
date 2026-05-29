"""
Job Script Generator - Generate PBS/Cobalt job scripts for ALCF
"""

from typing import Dict, Any, Optional
from pathlib import Path
from string import Template


# Job script templates
PBS_TEMPLATE = Template("""#!/bin/bash
#PBS -N ${job_name}
#PBS -l select=${nodes}:ncpus=${cpus_per_node}:ngpus=${gpus_per_node}
#PBS -l walltime=${walltime}
#PBS -q ${queue}
#PBS -A ${project}
#PBS -l filesystems=${filesystems}
#PBS -o ${output_file}
#PBS -e ${error_file}

# Change to working directory
cd ${work_dir}

# Load modules
${module_loads}

# Set environment variables
${env_vars}

# Run command
${run_command}
""")

COBALT_TEMPLATE = Template("""#!/bin/bash
#COBALT -n ${nodes}
#COBALT -t ${walltime_minutes}
#COBALT -q ${queue}
#COBALT -A ${project}
#COBALT --jobname ${job_name}
#COBALT -o ${output_file}
#COBALT -e ${error_file}

# Change to working directory
cd ${work_dir}

# Load modules
${module_loads}

# Set environment variables
${env_vars}

# Run command
${run_command}
""")


class JobScriptGenerator:
    """Generate job submission scripts for HPC systems."""
    
    # Default configurations for different ALCF systems
    SYSTEM_DEFAULTS = {
        'polaris': {
            'scheduler': 'pbs',
            'queue': 'prod',
            'cpus_per_node': 32,
            'gpus_per_node': 4,
            'filesystems': 'home:eagle:grand',
            'modules': ['PrgEnv-nvhpc', 'vasp/6.4.1'],
            'vasp_command': 'mpiexec -n ${ntasks} vasp_std',
        },
        'aurora': {
            'scheduler': 'pbs',
            'queue': 'workq',
            'cpus_per_node': 104,
            'gpus_per_node': 6,
            'filesystems': 'home:flare',
            'modules': ['oneapi', 'vasp/6.4.1'],
            'vasp_command': 'mpiexec -n ${ntasks} vasp_std',
        },
        'theta': {
            'scheduler': 'cobalt',
            'queue': 'default',
            'cpus_per_node': 64,
            'gpus_per_node': 0,
            'modules': ['vasp/5.4.4'],
            'vasp_command': 'aprun -n ${ntasks} vasp_std',
        },
        'generic': {
            'scheduler': 'pbs',
            'queue': 'batch',
            'cpus_per_node': 32,
            'gpus_per_node': 0,
            'filesystems': 'home',
            'modules': ['vasp'],
            'vasp_command': 'mpirun -np ${ntasks} vasp_std',
        }
    }
    
    def __init__(self, system: str = 'polaris'):
        """
        Initialize job script generator.
        
        Args:
            system: Target HPC system ('polaris', 'aurora', 'theta', 'generic')
        """
        self.system = system
        self.defaults = self.SYSTEM_DEFAULTS.get(system, self.SYSTEM_DEFAULTS['generic'])
    
    def generate_vasp_script(
        self,
        job_name: str,
        work_dir: str,
        nodes: int = 1,
        walltime: str = "01:00:00",
        project: str = "YOUR_PROJECT",
        queue: Optional[str] = None,
        extra_modules: Optional[list] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Generate a VASP job submission script.
        
        Args:
            job_name: Name of the job
            work_dir: Working directory containing VASP input files
            nodes: Number of nodes
            walltime: Wall time in HH:MM:SS format
            project: Project/allocation name
            queue: Queue name (uses system default if not specified)
            extra_modules: Additional modules to load
            extra_env: Additional environment variables
        
        Returns:
            Job script content as string
        """
        # Calculate tasks
        cpus_per_node = self.defaults['cpus_per_node']
        gpus_per_node = self.defaults['gpus_per_node']
        ntasks = nodes * cpus_per_node
        
        # Build module load commands
        modules = self.defaults.get('modules', [])
        if extra_modules:
            modules = modules + extra_modules
        module_loads = '\n'.join([f'module load {m}' for m in modules])
        
        # Build environment variables
        env_lines = []
        if extra_env:
            for key, value in extra_env.items():
                env_lines.append(f'export {key}="{value}"')
        env_vars = '\n'.join(env_lines)
        
        # Build run command
        vasp_cmd_template = Template(self.defaults['vasp_command'])
        run_command = vasp_cmd_template.substitute(ntasks=ntasks)
        
        # Select template based on scheduler
        if self.defaults['scheduler'] == 'pbs':
            template = PBS_TEMPLATE
            script_vars = {
                'job_name': job_name,
                'nodes': nodes,
                'cpus_per_node': cpus_per_node,
                'gpus_per_node': gpus_per_node,
                'walltime': walltime,
                'queue': queue or self.defaults['queue'],
                'project': project,
                'filesystems': self.defaults.get('filesystems', 'home'),
                'output_file': f'{job_name}.out',
                'error_file': f'{job_name}.err',
                'work_dir': work_dir,
                'module_loads': module_loads,
                'env_vars': env_vars,
                'run_command': run_command,
            }
        else:  # cobalt
            # Convert walltime to minutes
            parts = walltime.split(':')
            walltime_minutes = int(parts[0]) * 60 + int(parts[1])
            
            template = COBALT_TEMPLATE
            script_vars = {
                'job_name': job_name,
                'nodes': nodes,
                'walltime_minutes': walltime_minutes,
                'queue': queue or self.defaults['queue'],
                'project': project,
                'output_file': f'{job_name}.out',
                'error_file': f'{job_name}.err',
                'work_dir': work_dir,
                'module_loads': module_loads,
                'env_vars': env_vars,
                'run_command': run_command,
            }
        
        return template.substitute(script_vars)
    
    def generate_feff_script(
        self,
        job_name: str,
        work_dir: str,
        nodes: int = 1,
        walltime: str = "00:30:00",
        project: str = "YOUR_PROJECT",
        feff_executable: str = "feff9",
        queue: Optional[str] = None,
    ) -> str:
        """
        Generate a FEFF job submission script.
        
        Args:
            job_name: Name of the job
            work_dir: Working directory containing feff.inp
            nodes: Number of nodes (usually 1 for FEFF)
            walltime: Wall time in HH:MM:SS format
            project: Project/allocation name
            feff_executable: FEFF executable name
            queue: Queue name
        
        Returns:
            Job script content as string
        """
        module_loads = 'module load feff'
        run_command = feff_executable
        
        if self.defaults['scheduler'] == 'pbs':
            template = PBS_TEMPLATE
            script_vars = {
                'job_name': job_name,
                'nodes': nodes,
                'cpus_per_node': 1,
                'gpus_per_node': 0,
                'walltime': walltime,
                'queue': queue or self.defaults['queue'],
                'project': project,
                'filesystems': self.defaults.get('filesystems', 'home'),
                'output_file': f'{job_name}.out',
                'error_file': f'{job_name}.err',
                'work_dir': work_dir,
                'module_loads': module_loads,
                'env_vars': '',
                'run_command': run_command,
            }
        else:
            parts = walltime.split(':')
            walltime_minutes = int(parts[0]) * 60 + int(parts[1])
            
            template = COBALT_TEMPLATE
            script_vars = {
                'job_name': job_name,
                'nodes': nodes,
                'walltime_minutes': walltime_minutes,
                'queue': queue or self.defaults['queue'],
                'project': project,
                'output_file': f'{job_name}.out',
                'error_file': f'{job_name}.err',
                'work_dir': work_dir,
                'module_loads': module_loads,
                'env_vars': '',
                'run_command': run_command,
            }
        
        return template.substitute(script_vars)
    
    def write_script(self, content: str, output_path: Path) -> None:
        """Write job script to file."""
        output_path = Path(output_path)
        with open(output_path, 'w') as f:
            f.write(content)
        # Make executable
        output_path.chmod(0o755)