"""
Skill Registry - Load, validate, and manage skills dynamically
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from copy import deepcopy
import jsonschema


class SkillRegistry:
    """Registry for loading and managing skills from JSON files."""
    
    def __init__(self, skills_dir: Optional[str] = None):
        """
        Initialize the skill registry.
        
        Args:
            skills_dir: Directory containing skill JSON files.
                       Defaults to the directory containing this file.
        """
        if skills_dir is None:
            skills_dir = Path(__file__).parent
        self.skills_dir = Path(skills_dir)
        self._skills: Dict[str, Dict[str, Any]] = {}
        self._load_all_skills()
    
    def _load_all_skills(self) -> None:
        """Load all skill JSON files from the skills directory."""
        for skill_file in self.skills_dir.glob("*_skill.json"):
            try:
                with open(skill_file, 'r') as f:
                    skill_data = json.load(f)
                skill_id = skill_data.get('skill_id', skill_file.stem)
                self._skills[skill_id] = skill_data
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Failed to load skill from {skill_file}: {e}")
    
    def get_skill(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get a skill by its ID."""
        return deepcopy(self._skills.get(skill_id))
    
    def list_skills(self) -> List[str]:
        """List all available skill IDs."""
        return list(self._skills.keys())
    
    def get_skill_description(self, skill_id: str) -> Optional[str]:
        """Get the description of a skill."""
        skill = self._skills.get(skill_id)
        return skill.get('description') if skill else None
    
    def get_defaults(self, skill_id: str) -> Dict[str, Any]:
        """Get default parameters for a skill."""
        skill = self._skills.get(skill_id)
        if skill is None:
            return {}
        return deepcopy(skill.get('defaults', {}))
    
    def get_parameters_schema(self, skill_id: str) -> Optional[Dict[str, Any]]:
        """Get the JSON schema for skill parameters."""
        skill = self._skills.get(skill_id)
        return deepcopy(skill.get('parameters')) if skill else None
    
    def merge_with_defaults(
        self, 
        skill_id: str, 
        user_params: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge user parameters with skill defaults.
        
        Args:
            skill_id: The skill identifier
            user_params: User-provided parameters (override defaults)
            context: Optional context for conditional defaults (e.g., edge type)
        
        Returns:
            Merged parameters dictionary
        """
        skill = self._skills.get(skill_id)
        if skill is None:
            raise ValueError(f"Unknown skill: {skill_id}")
        
        # Start with base defaults
        merged = deepcopy(skill.get('defaults', {}))
        
        # Apply context-specific defaults if available
        if context:
            # Handle edge-specific defaults for XAS
            if 'edge' in context and 'edge_specific_defaults' in skill:
                edge = context['edge']
                edge_defaults = skill['edge_specific_defaults'].get(edge, {})
                merged = self._deep_merge(merged, edge_defaults)
            
            # Handle spectroscopy-type-specific defaults
            if 'spectroscopy_type' in context and 'spectroscopy_specific_defaults' in skill:
                spec_type = context['spectroscopy_type']
                spec_defaults = skill['spectroscopy_specific_defaults'].get(spec_type, {})
                merged = self._deep_merge(merged, spec_defaults)
        
        # Apply user overrides (highest priority)
        merged = self._deep_merge(merged, user_params)
        
        return merged
    
    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """Deep merge two dictionaries, with override taking precedence."""
        result = deepcopy(base)
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        return result
    
    def validate_parameters(self, skill_id: str, params: Dict[str, Any]) -> List[str]:
        """
        Validate parameters against skill schema and rules.
        
        Returns:
            List of validation error messages (empty if valid)
        """
        errors = []
        skill = self._skills.get(skill_id)
        
        if skill is None:
            return [f"Unknown skill: {skill_id}"]
        
        # Validate against JSON schema
        schema = skill.get('parameters')
        if schema:
            try:
                jsonschema.validate(params, schema)
            except jsonschema.ValidationError as e:
                errors.append(f"Schema validation error: {e.message}")
        
        # Validate custom rules
        for rule in skill.get('validation_rules', []):
            rule_expr = rule.get('rule', '')
            rule_msg = rule.get('message', 'Validation failed')
            
            try:
                # Create a safe evaluation context with only the parameters
                if not self._evaluate_rule(rule_expr, params):
                    errors.append(rule_msg)
            except Exception as e:
                errors.append(f"Rule evaluation error for '{rule_expr}': {e}")
        
        return errors
    
    def _evaluate_rule(self, rule: str, params: Dict[str, Any]) -> bool:
        """Safely evaluate a validation rule."""
        # Simple rule parser for common patterns
        # Supports: param < value, param > value, param >= value, param <= value
        import re
        
        # Pattern: variable operator value
        match = re.match(r'(\w+)\s*([<>=!]+)\s*(\w+)', rule)
        if not match:
            return True  # Can't parse, assume valid
        
        left_name, operator, right_name = match.groups()
        
        # Get values (could be param names or literals)
        left_val = params.get(left_name, left_name)
        right_val = params.get(right_name, right_name)
        
        # Try to convert to numbers if they're numeric strings
        try:
            left_val = float(left_val) if isinstance(left_val, str) else left_val
            right_val = float(right_val) if isinstance(right_val, str) else right_val
        except ValueError:
            pass
        
        # Evaluate
        ops = {
            '<': lambda a, b: a < b,
            '>': lambda a, b: a > b,
            '<=': lambda a, b: a <= b,
            '>=': lambda a, b: a >= b,
            '==': lambda a, b: a == b,
            '!=': lambda a, b: a != b,
        }
        
        if operator in ops:
            return ops[operator](left_val, right_val)
        
        return True


# Global registry instance
_registry: Optional[SkillRegistry] = None


def get_registry(skills_dir: Optional[str] = None) -> SkillRegistry:
    """Get or create the global skill registry."""
    global _registry
    if _registry is None:
        _registry = SkillRegistry(skills_dir)
    return _registry