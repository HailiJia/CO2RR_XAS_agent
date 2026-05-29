"""
Parameter Extractor - Extract parameters from natural language using LLM
"""

from typing import Dict, Any, Optional, List
import json
import re


class ParameterExtractor:
    """Extract structured parameters from natural language requests."""
    
    # Patterns for common parameter extractions
    PATTERNS = {
        'element': r'\b(Cu|Au|Ag|Pt|Pd|Ni|Fe|Co|Ti|Zn|Al|W|Mo)\b',
        'miller_index': r'\((\d),?\s*(\d),?\s*(\d)\)',
        'layers': r'(\d+)\s*layers?',
        'vacuum': r'(\d+(?:\.\d+)?)\s*[AÅ]?\s*vacuum',
        'adsorbate': r'\b(CO2?|H2?O?|OH|COOH|CHO|HCOO)\b',
        'binding_site': r'\b(ontop|bridge|fcc|hcp|hollow)\b',
        'edge': r'\b([KLM][1-5]?)\s*[-]?\s*edge',
        'distance': r'(\d+(?:\.\d+)?)\s*[AÅ]?\s*(?:distance|height|above)',
    }
    
    def __init__(self, llm_client: Optional[Any] = None):
        """
        Initialize the parameter extractor.
        
        Args:
            llm_client: Optional LLM client for complex extractions
        """
        self.llm_client = llm_client
    
    def extract_slab_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract slab generation parameters from natural language.
        
        Args:
            text: Natural language request
        
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        text_lower = text.lower()
        
        # Extract element
        element_match = re.search(self.PATTERNS['element'], text, re.IGNORECASE)
        if element_match:
            params['element'] = element_match.group(1)
        
        # Extract miller index
        miller_match = re.search(self.PATTERNS['miller_index'], text)
        if miller_match:
            params['miller_index'] = [int(miller_match.group(i)) for i in range(1, 4)]
        
        # Extract layers
        layers_match = re.search(self.PATTERNS['layers'], text_lower)
        if layers_match:
            params['layers'] = int(layers_match.group(1))
        
        # Extract vacuum
        vacuum_match = re.search(self.PATTERNS['vacuum'], text_lower)
        if vacuum_match:
            params['vacuum'] = float(vacuum_match.group(1))
        
        # Extract adsorbate
        adsorbate_match = re.search(self.PATTERNS['adsorbate'], text, re.IGNORECASE)
        if adsorbate_match:
            params['adsorbate'] = {'molecule': adsorbate_match.group(1).upper()}
            
            # Extract binding site
            site_match = re.search(self.PATTERNS['binding_site'], text_lower)
            if site_match:
                params['adsorbate']['binding_site'] = site_match.group(1)
            
            # Extract distance
            dist_match = re.search(self.PATTERNS['distance'], text_lower)
            if dist_match:
                params['adsorbate']['distance'] = float(dist_match.group(1))
        
        return params
    
    def extract_xas_parameters(self, text: str) -> Dict[str, Any]:
        """
        Extract XAS calculation parameters from natural language.
        
        Args:
            text: Natural language request
        
        Returns:
            Dictionary of extracted parameters
        """
        params = {}
        text_lower = text.lower()
        
        # Extract edge
        edge_match = re.search(self.PATTERNS['edge'], text, re.IGNORECASE)
        if edge_match:
            params['edge'] = edge_match.group(1).upper()
        
        # Extract absorbing element
        element_match = re.search(self.PATTERNS['element'], text, re.IGNORECASE)
        if element_match:
            params['absorbing_atom'] = {'element': element_match.group(1)}
        
        # Extract spectroscopy type
        if 'xanes' in text_lower:
            params['spectroscopy_type'] = 'XANES'
        elif 'exafs' in text_lower:
            params['spectroscopy_type'] = 'EXAFS'
        
        # Extract core hole method
        if 'fch' in text_lower or 'full core hole' in text_lower:
            params['core_hole'] = 'FCH'
        elif 'hch' in text_lower or 'half core hole' in text_lower:
            params['core_hole'] = 'HCH'
        
        return params
    
    def extract_with_llm(
        self, 
        text: str, 
        skill_schema: Dict[str, Any],
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to extract parameters based on skill schema.
        
        Args:
            text: Natural language request
            skill_schema: JSON schema for the skill parameters
            system_prompt: Optional custom system prompt
        
        Returns:
            Dictionary of extracted parameters
        """
        if self.llm_client is None:
            raise ValueError("LLM client not configured")
        
        if system_prompt is None:
            system_prompt = """You are a parameter extraction assistant. 
Given a natural language request and a JSON schema, extract the relevant parameters.
Return ONLY a valid JSON object with the extracted parameters.
If a parameter is not mentioned, do not include it in the output.
Be precise with numerical values and units."""
        
        user_prompt = f"""Extract parameters from this request:

REQUEST: {text}

SCHEMA:
{json.dumps(skill_schema, indent=2)}

Return only the JSON object with extracted parameters."""
        
        # Call LLM (implementation depends on your LLM client)
        response = self.llm_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        
        return {}
    
    def merge_extractions(
        self, 
        regex_params: Dict[str, Any], 
        llm_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Merge regex and LLM extractions, preferring LLM for conflicts.
        
        Args:
            regex_params: Parameters extracted via regex
            llm_params: Parameters extracted via LLM
        
        Returns:
            Merged parameters
        """
        merged = regex_params.copy()
        
        for key, value in llm_params.items():
            if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value
        
        return merged