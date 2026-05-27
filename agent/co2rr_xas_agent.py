"""
CO2RR XAS Agent
Main agent orchestration logic.

This file keeps deterministic Python execution, but allows an optional LLM planner
for flexible natural-language parsing. If the LLM planner is unavailable or
fails, the agent falls back to LocalIntentParser.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Union

# Import tools
from tools.structure_generator import StructureGenerator, execute_structure_generation
from tools.xas_input_generator import XASInputGenerator, execute_xas_input_generation
from tools.result_parser import ResultParser, execute_result_parsing
from tools.utils import ADSORBATES, CO2RR_PATHWAY, METAL_DATA

try:
    from agent.schemas import EnergyRange, FDMNESOptions, ParsedIntent
except ImportError:  # pragma: no cover - useful when running from inside agent/
    from .schemas import EnergyRange, FDMNESOptions, ParsedIntent


class LocalIntentParser:
    """
    Lightweight regex parser for local/offline use.

    This parser is intentionally conservative. It can process natural language in
    common patterns, but it is not a substitute for the LLM planner.
    """

    def __init__(self):
        # Use case-sensitive chemical matching: CO adsorbate should not be
        # confused with Co metal.
        metal_keys = sorted(METAL_DATA.keys(), key=len, reverse=True)
        adsorbate_keys = sorted(ADSORBATES.keys(), key=len, reverse=True)

        self.metal_pattern = r"(?<![A-Za-z])(" + "|".join(map(re.escape, metal_keys)) + r")(?![A-Za-z])"
        self.adsorbate_pattern = r"(?<![A-Za-z])(" + "|".join(map(re.escape, adsorbate_keys)) + r")(?![A-Za-z])"
        self.facet_pattern = r"\b(111|100|110|0001)\b"

    def parse(self, request: str, output_dir: str = "output") -> ParsedIntent:
        """Parse a natural-language request into ParsedIntent."""
        request_lower = request.lower()

        # Determine action
        if any(word in request_lower for word in ["parse", "read output", "isaac record", "extract", "convert results"]):
            action = "parse_results"
        elif any(word in request_lower for word in ["full workflow", "complete", "end-to-end"]):
            action = "full_workflow"
        elif any(word in request_lower for word in ["xas", "x-ray", "absorption", "feff", "fdmnes", "edge"]):
            action = "generate_xas"
        else:
            action = "generate_structure"

        # Extract metals, facets, adsorbates
        metals = list(dict.fromkeys(re.findall(self.metal_pattern, request)))

        facets = re.findall(self.facet_pattern, request)
        if not facets:
            facets = ["111"]

        adsorbates = list(dict.fromkeys(re.findall(self.adsorbate_pattern, request)))

        full_pathway = any(
            phrase in request_lower
            for phrase in [
                "full pathway",
                "complete pathway",
                "all adsorbates",
                "reaction pathway",
                "co2rr pathway",
            ]
        )

        structure_file = self._extract_structure_file(request)
        parameters: Dict[str, Any] = {}

        parameters["fdmnes_options"] = self._extract_fdmnes_options(request_lower)

        # VASP method: one selected folder, not both PBE and GW.
        if re.search(r"\b(gw|gw0)\b", request_lower):
            parameters["vasp_method"] = "GW"
        elif re.search(r"\bpbe\b", request_lower):
            parameters["vasp_method"] = "PBE"

        potcar_match = re.search(
            r"(?:potcar\s*(?:dir|directory|path)|potpaw\s*(?:dir|directory|path))\s*(?:=|is|:)?\s*([^\s,;]+)",
            request,
            re.IGNORECASE,
        )
        if potcar_match:
            parameters["potcar_dir"] = potcar_match.group(1)

        cluster_radius = self._extract_float(
            request_lower,
            r"(?:radius|cluster\s+radius)\s*(?:=|is|to|of)?\s*([0-9]+(?:\.[0-9]+)?)",
        )
        if cluster_radius is not None:
            parameters["cluster_radius"] = cluster_radius

        energy_range = self._extract_energy_range(request_lower)
        if energy_range is not None:
            parameters["fdmnes_energy_range"] = energy_range.as_fdmnes_range()

        edge_match = re.search(r"\b(k|l2|l3|l23)\s*[- ]?edge\b", request_lower)
        if not edge_match:
            edge_match = re.search(r"\bedge\s*(?:=|is|:)?\s*(k|l2|l3|l23)\b", request_lower)
        if edge_match:
            edge = edge_match.group(1).upper()
            parameters["edge_override"] = edge
            parameters["edge"] = edge

        supercell_match = re.search(r"(\d+)\s*[xX×]\s*(\d+)\s*supercell", request)
        if supercell_match:
            parameters["supercell"] = (int(supercell_match.group(1)), int(supercell_match.group(2)))

        layers_match = re.search(r"(\d+)\s*layer", request_lower)
        if layers_match:
            parameters["layers"] = int(layers_match.group(1))

        for site in ["top", "bridge", "fcc", "hcp"]:
            if re.search(rf"\b{site}\b", request_lower):
                parameters["site"] = site
                break

        account_match = re.search(r"(?:nersc\s+)?account\s+([A-Za-z0-9_\-]+)", request, re.IGNORECASE)
        if account_match:
            parameters["nersc_account"] = account_match.group(1)

        queue_match = re.search(r"(?:queue|partition)\s+([A-Za-z0-9_\-]+)", request, re.IGNORECASE)
        if queue_match:
            parameters["nersc_queue"] = queue_match.group(1)

        email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", request)
        if email_match:
            parameters["email"] = email_match.group(0)

        return ParsedIntent(
            action=action,
            metals=metals,
            facets=facets,
            adsorbates=adsorbates,
            full_pathway=full_pathway,
            structure_file=structure_file,
            output_dir=output_dir,
            parameters=parameters,
        )

    def _extract_structure_file(self, request: str) -> Optional[str]:
        file_patterns = [
            r"(?:from|read|use|file)\s+[\"']?([^\s\"']+(?:POSCAR|CONTCAR|\.cif))[\"']?",
            r"([^\s]+(?:POSCAR|CONTCAR|\.cif))",
        ]
        for pattern in file_patterns:
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                return match.group(1)
        return None

    def _extract_fdmnes_options(self, request_lower: str) -> Dict[str, bool]:
        options = FDMNESOptions().to_fdmnes_dict()

        def has_any(patterns: List[str]) -> bool:
            return any(re.search(p, request_lower) for p in patterns)

        def negated(term: str) -> bool:
            return re.search(rf"\b(no|without|disable|disabled|exclude|excluding)\s+{term}\b", request_lower) is not None

        if has_any([r"\bgreen\b", r"multiple\s+scattering", r"multiple-scattering"]):
            options["Green"] = True
        if has_any([r"\bfdm\b", r"finite\s+difference", r"finite-difference"]):
            options["Green"] = False

        if has_any([r"\bquadrupole\b"]):
            options["Quadrupole"] = True
        if has_any([r"\brelativism\b", r"\brelativistic\b"]):
            options["Relativism"] = True
        if has_any([r"\bspinorbit\b", r"spin\s*orbit", r"spin-orbit"]):
            options["Spinorbit"] = True
        if negated("scf"):
            options["SCF"] = False
        elif re.search(r"\bscf\b", request_lower):
            options["SCF"] = True
        if has_any([r"\bscfexc\b", r"scf\s*exc"]):
            options["SCFexc"] = True
        if has_any([r"\bscreening\b"]):
            options["Screening"] = True
        if has_any([r"\bfull_atom\b", r"full\s+atom", r"full-atom"]):
            options["Full_atom"] = True
        if has_any([r"\btddft\b", r"\btd-dft\b"]):
            options["TDDFT"] = True

        return options

    def _extract_float(self, text: str, pattern: str) -> Optional[float]:
        match = re.search(pattern, text)
        if not match:
            return None
        return float(match.group(1))

    def _extract_energy_range(self, text: str) -> Optional[EnergyRange]:
        number = r"(-?[0-9]+(?:\.[0-9]*)?)"

        # Explicit FDMNES legacy order: energy range -5 0.2 50
        legacy = re.search(
            rf"(?:e_range|energy\s*range|range)\s*(?:=|is|:)?\s*{number}\s+{number}\s+{number}",
            text,
        )
        if legacy:
            return EnergyRange.from_any((legacy.group(1), legacy.group(2), legacy.group(3)))

        # Natural language: from -5 eV to 50 eV with step size 0.2 eV
        natural = re.search(
            rf"(?:energy\s*range|range)?\s*(?:start(?:ing)?\s+)?(?:from\s+)?{number}\s*(?:ev)?\s*(?:to|through|until|-)\s*{number}\s*(?:ev)?(?:[^0-9-]+)(?:step\s*size|step|interval|spacing|increment)\s*(?:of|=|:)?\s*{number}",
            text,
        )
        if natural:
            return EnergyRange(emin=float(natural.group(1)), emax=float(natural.group(2)), step=float(natural.group(3))).normalized()

        # Natural language with step first: step 0.2 from -5 to 50
        step_first = re.search(
            rf"(?:step\s*size|step|interval|spacing|increment)\s*(?:of|=|:)?\s*{number}.*?(?:from\s+)?{number}\s*(?:ev)?\s*(?:to|through|until|-)\s*{number}",
            text,
        )
        if step_first:
            return EnergyRange(emin=float(step_first.group(2)), emax=float(step_first.group(3)), step=float(step_first.group(1))).normalized()

        return None


class CO2RRXASAgent:
    """
    Main CO2RR XAS Agent.

    Execution remains deterministic Python. Natural-language interpretation is
    handled by an optional LLM planner, with LocalIntentParser as fallback.
    """

    def __init__(
        self,
        default_output_dir: str = "co2rr_xas_output",
        planner=None,
        use_llm: bool = False,
    ):
        self.local_intent_parser = LocalIntentParser()
        self.planner = planner
        self.use_llm = use_llm

        self.structure_gen = StructureGenerator()
        self.xas_gen = XASInputGenerator()
        self.result_parser = ResultParser()
        self.default_output_dir = default_output_dir

    def _parse_request(self, request: str, output_dir: str) -> ParsedIntent:
        """Use LLM planner if available; otherwise use local regex parser."""
        if self.use_llm and self.planner is not None:
            try:
                return self.planner.parse(request, output_dir)
            except Exception as exc:
                print(f"[warning] LLM planner failed; using local parser. Error: {exc}")

        return self.local_intent_parser.parse(request, output_dir)

    def process_request(self, request: str, output_dir: Optional[str] = None, **kwargs) -> Dict:
        """Process a natural-language request."""
        output_dir = output_dir or self.default_output_dir
        intent = self._parse_request(request, output_dir)

        for key, value in kwargs.items():
            if hasattr(intent, key):
                setattr(intent, key, value)
            else:
                intent.parameters[key] = value

        if intent.action == "generate_structure":
            return self._execute_structure_generation(intent)
        if intent.action == "generate_xas":
            return self._execute_xas_generation(intent)
        if intent.action == "parse_results":
            return self._execute_result_parsing(intent)
        if intent.action == "full_workflow":
            return self._execute_full_workflow(intent)
        return {"status": "error", "message": f"Unknown action: {intent.action}"}

    def _execute_structure_generation(self, intent: ParsedIntent) -> Dict:
        """Execute structure generation."""
        if intent.structure_file:
            return execute_structure_generation(
                mode="read_file",
                structure_file=intent.structure_file,
                output_dir=intent.output_dir,
            )

        if not intent.metals:
            return {"status": "error", "message": "No metal element specified"}

        metal1 = intent.metals[0]
        metal2 = intent.metals[1] if len(intent.metals) > 1 else None
        facet1 = intent.facets[0] if intent.facets else "111"
        facet2 = intent.facets[1] if len(intent.facets) > 1 else None

        return execute_structure_generation(
            mode="generate",
            metal1=metal1,
            metal2=metal2,
            facet1=facet1,
            facet2=facet2,
            adsorbate=intent.adsorbates if intent.adsorbates else None,
            site=intent.parameters.get("site", "top"),
            supercell=intent.parameters.get("supercell", (3, 3)),
            layers=intent.parameters.get("layers", 4),
            output_dir=intent.output_dir,
            full_pathway=intent.full_pathway,
        )

    def _common_xas_kwargs(self, intent: ParsedIntent, structure_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Build common XAS-generation kwargs used by both branches."""
        fdmnes_energy_range = EnergyRange.from_any(
            intent.parameters.get("fdmnes_energy_range", (-5.0, 0.2, 50.0))
        ).as_fdmnes_range()

        kwargs = {
            "nersc_account": intent.parameters.get("nersc_account", "m5268"),
            "nersc_queue": intent.parameters.get("nersc_queue", "regular"),
            "nersc_nodes": intent.parameters.get("nersc_nodes", 2),
            "nersc_walltime": intent.parameters.get("nersc_walltime", "8:00:00"),
            "email": intent.parameters.get("email"),
            "cluster_radius": intent.parameters.get("cluster_radius", 6.0),
            "fdmnes_options": intent.parameters.get("fdmnes_options"),
            "fdmnes_energy_range": fdmnes_energy_range,
            "edge_override": intent.parameters.get("edge_override"),
            "vasp_method": intent.parameters.get("vasp_method", "PBE"),
            "potcar_dir": intent.parameters.get("potcar_dir"),
        }
        if structure_metadata is not None:
            kwargs["structure_metadata"] = structure_metadata
        return kwargs

    def _execute_xas_generation(self, intent: ParsedIntent) -> Dict:
        """Execute XAS input generation."""
        results = {"status": "success", "structures": []}

        if not intent.structure_file:
            struct_result = self._execute_structure_generation(intent)
            if struct_result["status"] != "success":
                return struct_result

            for files in struct_result["files"]:
                poscar_path = files["poscar"]
                struct_dir = os.path.dirname(poscar_path)
                base_dir = os.path.dirname(struct_dir)

                metadata_path = files["metadata"]
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)

                xas_result = execute_xas_input_generation(
                    structure_file=poscar_path,
                    output_dir=base_dir,
                    **self._common_xas_kwargs(intent, structure_metadata=metadata),
                )

                results["structures"].append({
                    "structure": metadata,
                    "xas_inputs": xas_result,
                })
        else:
            # Existing-structure branch now receives the same FDMNES options,
            # energy range, and edge override as the generated-structure branch.
            xas_result = execute_xas_input_generation(
                structure_file=intent.structure_file,
                output_dir=intent.output_dir,
                **self._common_xas_kwargs(intent),
            )
            results["structures"].append({
                "structure_file": intent.structure_file,
                "xas_inputs": xas_result,
            })

        return results

    def _execute_result_parsing(self, intent: ParsedIntent) -> Dict:
        """Execute result parsing and ISAAC record generation."""
        output_dir = intent.parameters.get("results_dir", intent.output_dir)
        software = intent.parameters.get("software", "FEFF")
        structure_file = intent.structure_file or intent.parameters.get("structure_file")

        if not structure_file:
            return {"status": "error", "message": "Structure file required for parsing"}

        absorber = intent.parameters.get("absorber", intent.metals[0] if intent.metals else "Cu")
        edge = intent.parameters.get("edge", intent.parameters.get("edge_override", "K"))

        return execute_result_parsing(
            output_dir=output_dir,
            software=software,
            structure_file=structure_file,
            absorber=absorber,
            edge=edge,
            parameters=intent.parameters,
            vasp_mode=intent.parameters.get("vasp_mode", "trace"),
            description=intent.parameters.get("description", ""),
            tags=intent.parameters.get("tags"),
        )

    def _execute_full_workflow(self, intent: ParsedIntent) -> Dict:
        """Execute full workflow: structure generation -> XAS input generation."""
        if not intent.adsorbates and not intent.full_pathway:
            intent.full_pathway = True
        if intent.full_pathway:
            intent.adsorbates = CO2RR_PATHWAY
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
        full_pathway: bool = False,
    ) -> Dict:
        """Direct API for structure generation."""
        return execute_structure_generation(
            mode="generate",
            metal1=metal1,
            metal2=metal2,
            facet1=facet1,
            facet2=facet2,
            adsorbate=adsorbate,
            site=site,
            supercell=supercell,
            layers=layers,
            output_dir=output_dir or self.default_output_dir,
            full_pathway=full_pathway,
        )

    def generate_xas_inputs(
        self,
        structure_file: str,
        output_dir: Optional[str] = None,
        nersc_account: str = "m5268",
        nersc_queue: str = "regular",
        nersc_nodes: int = 2,
        nersc_walltime: str = "8:00:00",
        email: Optional[str] = None,
        cluster_radius: float = 6.0,
        fdmnes_options: Optional[Dict[str, bool]] = None,
        fdmnes_energy_range=(-5.0, 0.2, 50.0),
        edge_override: Optional[str] = None,
        vasp_method: str = "PBE",
        potcar_dir: Optional[str] = None,
    ) -> Dict:
        """Direct API for XAS input generation."""
        return execute_xas_input_generation(
            structure_file=structure_file,
            output_dir=output_dir or self.default_output_dir,
            nersc_account=nersc_account,
            nersc_queue=nersc_queue,
            nersc_nodes=nersc_nodes,
            nersc_walltime=nersc_walltime,
            email=email,
            cluster_radius=cluster_radius,
            fdmnes_options=fdmnes_options,
            fdmnes_energy_range=EnergyRange.from_any(fdmnes_energy_range).as_fdmnes_range(),
            edge_override=edge_override,
            vasp_method=vasp_method,
            potcar_dir=potcar_dir,
        )

    def parse_results(
        self,
        output_dir: str,
        software: str,
        structure_file: str,
        absorber: str,
        edge: str,
        vasp_mode: str = "trace",
    ) -> Dict:
        """Direct API for result parsing."""
        return execute_result_parsing(
            output_dir=output_dir,
            software=software,
            structure_file=structure_file,
            absorber=absorber,
            edge=edge,
            vasp_mode=vasp_mode,
        )


# =============================================================================
# Main Entry Points
# =============================================================================


def create_agent(
    output_dir: str = "co2rr_xas_output",
    planner=None,
    use_llm: bool = False,
) -> CO2RRXASAgent:
    """Create an agent. If use_llm=True and planner is None, build one from env."""
    if use_llm and planner is None:
        try:
            try:
                from agent.planner import create_llm_planner_from_env
            except ImportError:  # pragma: no cover
                from .planner import create_llm_planner_from_env
            planner = create_llm_planner_from_env()
        except Exception as exc:
            print(f"[warning] Could not initialize LLM planner; using local parser only. Error: {exc}")
            planner = None

    return CO2RRXASAgent(
        default_output_dir=output_dir,
        planner=planner,
        use_llm=use_llm,
    )


def process_request(
    request: str,
    output_dir: str = "co2rr_xas_output",
    planner=None,
    use_llm: bool = False,
    **kwargs,
) -> Dict:
    agent = create_agent(output_dir, planner=planner, use_llm=use_llm)
    return agent.process_request(request, **kwargs)
