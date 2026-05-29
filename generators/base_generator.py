"""
Base Generator - Abstract base class for all input generators
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pathlib import Path

from skills.skill_registry import get_registry, SkillRegistry


class BaseGenerator(ABC):
    """Abstract base class for input file generators."""
    
    def __init__(self, skill_id: str, registry: Optional[SkillRegistry] = None):
        """
        Initialize the generator with a skill.
        
        Args:
            skill_id: The skill identifier to use
            registry: Optional skill registry (uses global if not provided)
        """
        self.registry = registry or get_registry()
        self.skill_id = skill_id
        self.skill = self.registry.get_skill(skill_id)
        
        if self.skill is None:
            raise ValueError(f"Skill not found: {skill_id}")
    
    def prepare_parameters(
        self, 
        user_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Prepare final parameters by merging user input with defaults.
        
        Args:
            user_params: User-provided parameters
            context: Optional context for conditional defaults
        
        Returns:
            Final merged and validated parameters
        """
        # Merge with defaults
        params = self.registry.merge_with_defaults(
            self.skill_id, 
            user_params, 
            context=context
        )
        
        # Validate
        errors = self.registry.validate_parameters(self.skill_id, params)
        if errors:
            raise ValueError(f"Parameter validation failed: {'; '.join(errors)}")
        
        return params
    
    @abstractmethod
    def generate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate output based on parameters.
        
        Args:
            params: Validated parameters
        
        Returns:
            Dictionary containing generated outputs
        """
        pass
    
    @abstractmethod
    def write_files(self, outputs: Dict[str, Any], output_dir: Path) -> None:
        """
        Write generated outputs to files.
        
        Args:
            outputs: Generated outputs from generate()
            output_dir: Directory to write files to
        """
        pass
    
    def run(
        self, 
        user_params: Dict[str, Any], 
        output_dir: Optional[Path] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Full pipeline: prepare params, generate, and optionally write files.
        
        Args:
            user_params: User-provided parameters
            output_dir: Optional directory to write files (if None, don't write)
            context: Optional context for conditional defaults
        
        Returns:
            Generated outputs
        """
        params = self.prepare_parameters(user_params, context=context)
        outputs = self.generate(params)
        
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            self.write_files(outputs, output_dir)
        
        return outputs