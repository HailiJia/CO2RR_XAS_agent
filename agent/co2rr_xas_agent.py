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
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Import tools
from tools.structure_generator import StructureGenerator, execute_structure_generation
from tools.xas_input_generator import XASInputGenerator, execute_xas_input_generation
from tools.result_parser import ResultParser, execute_result_parsing
from workflow.nersc_workflow import execute_nersc_full_workflow
from tools.utils import ADSORBATES, CO2RR_PATHWAY, METAL_DATA, get_nersc_defaults

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
        elif any(phrase in request_lower for phrase in ["submit", "monitor", "nersc workflow", "run on nersc"]):
            action = "nersc_workflow"
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
        metadata_overrides = self._extract_ml_metadata(request)
        if metadata_overrides:
            parameters["metadata_overrides"] = metadata_overrides
            if metadata_overrides.get("catalyst", {}).get("elements"):
                metals = metadata_overrides["catalyst"]["elements"]
            if metadata_overrides.get("catalyst", {}).get("surface_facet"):
                facets = [str(metadata_overrides["catalyst"]["surface_facet"])]
            if metadata_overrides.get("adsorbate", {}).get("identity"):
                adsorbates = [metadata_overrides["adsorbate"]["identity"]]

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

        if "dry run" in request_lower or "do not submit" in request_lower or "don't submit" in request_lower:
            parameters["submit_jobs"] = False
        elif "submit" in request_lower or "run on nersc" in request_lower:
            parameters["submit_jobs"] = True

        if "monitor" in request_lower or "when finish" in request_lower or "when finished" in request_lower:
            parameters["monitor_jobs"] = True
        if "write record" in request_lower or "isaac" in request_lower or "when finish" in request_lower or "when finished" in request_lower:
            parameters["parse_when_complete"] = True

        poll_match = re.search(r"poll(?:ing)?(?: interval)?\s*(?:=|is|:)?\s*(\d+)", request_lower)
        if poll_match:
            parameters["poll_interval"] = int(poll_match.group(1))

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

    def _extract_ml_metadata(self, request: str) -> Dict[str, Any]:
        """Extract catalyst/adsorbate ML metadata from compact YAML-like prompts."""
        metadata: Dict[str, Any] = {}

        elements_match = re.search(r"elements\s*:\s*\[([^\]]+)\]", request, re.IGNORECASE)
        catalyst: Dict[str, Any] = {}
        if elements_match:
            catalyst["elements"] = [item.strip().strip("'\"") for item in elements_match.group(1).split(",") if item.strip()]
        for key, pattern in {
            "composition": r"composition\s*:\s*([A-Za-z0-9_\-]+)",
            "surface_facet": r"surface_facet\s*:\s*[\"']?([^\"'\s]+)[\"']?",
            "site_type": r"site_type\s*:\s*[\"']?([^\"'\s]+)[\"']?",
            "structure_id": r"structure_id\s*:\s*[\"']?([^\"'\s]+)[\"']?",
        }.items():
            match = re.search(pattern, request, re.IGNORECASE)
            if match:
                catalyst[key] = match.group(1)
        if catalyst:
            metadata["catalyst"] = {k: v for k, v in catalyst.items() if k != "structure_id"}
            if catalyst.get("structure_id"):
                metadata["structure_id"] = catalyst["structure_id"]

        adsorbate: Dict[str, Any] = {}
        ads_block = request[request.lower().find("adsorbate:"):] if "adsorbate:" in request.lower() else request
        for key, pattern in {
            "identity": r"identity\s*:\s*[\"']?([A-Za-z0-9_\-]+)[\"']?",
            "formula": r"formula\s*:\s*[\"']?([A-Za-z0-9_\-]+)[\"']?",
            "intermediate_class": r"intermediate_class\s*:\s*[\"']?([A-Za-z0-9_\-]+)[\"']?",
            "binding_mode": r"binding_mode\s*:\s*[\"']?([A-Za-z0-9_\-]+)[\"']?",
            "adsorption_site": r"adsorption_site\s*:\s*[\"']?([^\"'\n]+?)(?=\s+binding_atom\s*:|$)",
            "binding_atom": r"binding_atom\s*:\s*[\"']?([A-Za-z]+)[\"']?",
        }.items():
            match = re.search(pattern, ads_block, re.IGNORECASE)
            if match:
                adsorbate[key] = match.group(1).strip()
        if adsorbate:
            metadata["adsorbate"] = adsorbate
        return metadata

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


    def _needs_input(
        self,
        message: str,
        missing: List[Dict[str, Any]],
        intent: ParsedIntent,
        suggestions: Optional[List[str]] = None,
        recoveries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Return a structured clarification request instead of a bare error."""
        return {
            "status": "needs_input",
            "message": message,
            "missing_information": missing,
            "suggestions": suggestions or [],
            "recovery_actions": recoveries or [],
            "parsed_intent": {
                "action": intent.action,
                "metals": intent.metals,
                "facets": intent.facets,
                "adsorbates": intent.adsorbates,
                "structure_file": intent.structure_file,
                "output_dir": intent.output_dir,
                "parameters": intent.parameters,
            },
        }

    def _attach_agent_context(
        self,
        result: Dict[str, Any],
        warnings: Optional[List[str]] = None,
        recoveries: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Attach warnings/recovery metadata to normal tool results."""
        if not isinstance(result, dict):
            return result
        agent_context = result.setdefault("agent", {})
        if warnings:
            agent_context.setdefault("warnings", []).extend(warnings)
        if recoveries:
            agent_context.setdefault("recovery_actions", []).extend(recoveries)
        return result

    def _discover_structure_file(self, output_dir: str) -> Optional[str]:
        """Find a nearby POSCAR/CONTCAR when the user omitted structure_file."""
        root = Path(output_dir)
        candidates: List[Path] = []
        if root.exists():
            for name in ("POSCAR", "CONTCAR"):
                candidate = root / name
                if candidate.is_file():
                    candidates.append(candidate)
            candidates.extend(root.rglob("POSCAR"))
            candidates.extend(root.rglob("CONTCAR"))
        for base in root.parents[:2]:
            for name in ("POSCAR", "CONTCAR"):
                candidate = base / name
                if candidate.is_file():
                    candidates.append(candidate)
        if not candidates:
            return None
        candidates = sorted(set(candidates), key=lambda p: (len(p.parts), p.as_posix()))
        return str(candidates[0])

    def _validate_generation_inputs(self, intent: ParsedIntent) -> Optional[Dict[str, Any]]:
        if intent.structure_file:
            if not Path(intent.structure_file).exists():
                return self._needs_input(
                    f"Structure file not found: {intent.structure_file}",
                    [{"field": "structure_file", "reason": "Path does not exist."}],
                    intent,
                    suggestions=["Provide a valid POSCAR, CONTCAR, or CIF path.", "Or provide catalyst metals so the agent can generate a structure."],
                )
            return None
        if not intent.metals:
            return self._needs_input(
                "I need either a structure file or catalyst metal(s) before I can generate structures/XAS inputs.",
                [{"field": "metals", "reason": "No catalyst metal was parsed."}],
                intent,
                suggestions=[
                    "Add a catalyst such as `Cu(111)` or `elements: [Cu, Au]`.",
                    "Or add `from /path/to/POSCAR` to use an existing structure.",
                ],
            )
        return None

    def _validate_parse_inputs(self, intent: ParsedIntent) -> Optional[Dict[str, Any]]:
        structure_file = intent.structure_file or intent.parameters.get("structure_file")
        if structure_file and Path(structure_file).exists():
            return None
        recovered = self._discover_structure_file(intent.parameters.get("results_dir", intent.output_dir))
        if recovered:
            intent.structure_file = recovered
            intent.parameters["structure_file"] = recovered
            intent.parameters.setdefault("agent_recoveries", []).append(f"Recovered missing structure_file from {recovered}")
            return None
        missing_reason = "No structure_file was provided." if not structure_file else f"Structure file not found: {structure_file}"
        return self._needs_input(
            "I can parse spectra only after I know which structure produced them.",
            [{"field": "structure_file", "reason": missing_reason}],
            intent,
            suggestions=[
                "Pass `structure_file=/path/to/POSCAR` or include `from /path/to/POSCAR` in the request.",
                "Put POSCAR or CONTCAR in the result directory so the agent can auto-recover it.",
            ],
        )

    def process_request(self, request: str, output_dir: Optional[str] = None, **kwargs) -> Dict:
        """Process a natural-language request."""
        output_dir = output_dir or self.default_output_dir
        intent = self._parse_request(request, output_dir)

        for key, value in kwargs.items():
            if hasattr(intent, key):
                setattr(intent, key, value)
            else:
                intent.parameters[key] = value

        if intent.action in {"generate_structure", "generate_xas", "full_workflow", "nersc_workflow"}:
            validation = self._validate_generation_inputs(intent)
            if validation is not None:
                return validation
        if intent.action == "parse_results":
            validation = self._validate_parse_inputs(intent)
            if validation is not None:
                return validation

        if intent.action == "generate_structure":
            return self._execute_structure_generation(intent)
        if intent.action == "generate_xas":
            return self._execute_xas_generation(intent)
        if intent.action == "parse_results":
            return self._execute_result_parsing(intent)
        if intent.action == "full_workflow":
            return self._execute_full_workflow(intent)
        if intent.action == "nersc_workflow":
            return self._execute_nersc_workflow(intent)
        return self._needs_input(
            f"Unknown action: {intent.action}",
            [{"field": "action", "reason": "The request did not map to a supported workflow action."}],
            intent,
            suggestions=["Ask to generate a structure, generate XAS inputs, run a NERSC workflow, or parse results."],
        )

    def _execute_structure_generation(self, intent: ParsedIntent) -> Dict:
        """Execute structure generation."""
        if intent.structure_file:
            return execute_structure_generation(
                mode="read_file",
                structure_file=intent.structure_file,
                output_dir=intent.output_dir,
            )

        if not intent.metals:
            return self._needs_input(
                "No metal element specified.",
                [{"field": "metals", "reason": "No catalyst metal was parsed."}],
                intent,
                suggestions=["Add a metal such as Cu, Ag, Au, Pt, Pd, Ni, Fe, or Co."],
            )

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
            metadata_overrides=intent.parameters.get("metadata_overrides"),
        )

    def _common_xas_kwargs(self, intent: ParsedIntent, structure_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Build common XAS-generation kwargs used by both branches."""
        fdmnes_energy_range = EnergyRange.from_any(
            intent.parameters.get("fdmnes_energy_range", (-5.0, 0.2, 50.0))
        ).as_fdmnes_range()

        defaults = get_nersc_defaults()
        kwargs = {
            "nersc_account": intent.parameters.get("nersc_account", defaults["account"]),
            "nersc_queue": intent.parameters.get("nersc_queue", defaults["queue"]),
            "nersc_nodes": intent.parameters.get("nersc_nodes", defaults["nodes"]),
            "nersc_walltime": intent.parameters.get("nersc_walltime", defaults["walltime"]),
            "email": intent.parameters.get("email"),
            "cluster_radius": intent.parameters.get("cluster_radius", 6.0),
            "fdmnes_options": intent.parameters.get("fdmnes_options"),
            "fdmnes_energy_range": fdmnes_energy_range,
            "edge_override": intent.parameters.get("edge_override"),
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
        structure_metadata = intent.parameters.get("structure_metadata")

        if not structure_file:
            return self._needs_input(
                "Structure file required for parsing.",
                [{"field": "structure_file", "reason": "No POSCAR/CONTCAR/CIF was provided or discovered."}],
                intent,
                suggestions=["Pass structure_file=... or place POSCAR/CONTCAR in the results directory."],
            )

        # Absorber/edge can be inferred by tools.result_parser from FEFF/FDMNES/VASP input files.
        # Only pass explicit values when the user/request actually provided them.
        absorber = intent.parameters.get("absorber", intent.metals[0] if intent.metals else None)
        edge = intent.parameters.get("edge", intent.parameters.get("edge_override"))

        result = execute_result_parsing(
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
        return self._attach_agent_context(result, recoveries=intent.parameters.get("agent_recoveries"))

    def _execute_full_workflow(self, intent: ParsedIntent) -> Dict:
        """Execute full workflow: structure generation -> XAS input generation."""
        if not intent.adsorbates and not intent.full_pathway:
            intent.full_pathway = True
        if intent.full_pathway:
            intent.adsorbates = CO2RR_PATHWAY
        return self._execute_xas_generation(intent)


    def _execute_nersc_workflow(self, intent: ParsedIntent) -> Dict:
        """Generate NERSC inputs, submit jobs, monitor them, and write ISAAC records."""
        structure_file = intent.structure_file or intent.parameters.get("structure_file")
        structure_metadata = intent.parameters.get("structure_metadata")
        if not structure_file:
            if not intent.metals:
                return self._needs_input(
                    "Structure file or metal is required for NERSC workflow.",
                    [{"field": "structure_file_or_metals", "reason": "No structure file and no catalyst metal were parsed."}],
                    intent,
                    suggestions=["Provide `from /path/to/POSCAR` or a catalyst such as `Cu(111)` / `elements: [Cu, Au]`."],
                )
            struct_result = self._execute_structure_generation(intent)
            if struct_result.get("status") != "success":
                return struct_result
            # Use the first generated structure for the NERSC end-to-end workflow.
            structure_file = struct_result["files"][0]["poscar"]
            metadata_path = struct_result["files"][0].get("metadata")
            if metadata_path:
                with open(metadata_path, "r") as f:
                    structure_metadata = json.load(f)

        warnings: List[str] = []
        recoveries: List[str] = list(intent.parameters.get("agent_recoveries", []))
        submit = bool(intent.parameters.get("submit_jobs", True))
        monitor = bool(intent.parameters.get("monitor_jobs", True))
        parse_when_complete = bool(intent.parameters.get("parse_when_complete", True))
        if parse_when_complete and not monitor:
            warnings.append("parse_when_complete=True requires monitor=True; records will not be parsed automatically.")

        defaults = get_nersc_defaults()
        result = execute_nersc_full_workflow(
            structure_file=structure_file,
            output_dir=intent.output_dir,
            software=intent.parameters.get("software", "all"),
            submit=submit,
            monitor=monitor,
            parse_when_complete=parse_when_complete,
            poll_interval=int(intent.parameters.get("poll_interval", 60)),
            timeout=int(intent.parameters["monitor_timeout"]) if intent.parameters.get("monitor_timeout") is not None else None,
            nersc_account=intent.parameters.get("nersc_account", defaults["account"]),
            nersc_queue=intent.parameters.get("nersc_queue", defaults["queue"]),
            nersc_nodes=intent.parameters.get("nersc_nodes", defaults["nodes"]),
            nersc_walltime=intent.parameters.get("nersc_walltime", defaults["walltime"]),
            email=intent.parameters.get("email"),
            cluster_radius=intent.parameters.get("cluster_radius", 6.0),
            fdmnes_options=intent.parameters.get("fdmnes_options"),
            fdmnes_energy_range=EnergyRange.from_any(
                intent.parameters.get("fdmnes_energy_range", (-5.0, 0.2, 50.0))
            ).as_fdmnes_range(),
            edge_override=intent.parameters.get("edge_override"),
            vasp_method=intent.parameters.get("vasp_method", "PBE"),
            potcar_dir=intent.parameters.get("potcar_dir"),
            vasp_mode=intent.parameters.get("vasp_mode", "trace"),
            record_parameters=intent.parameters.get("record_parameters", {}),
            structure_metadata=structure_metadata,
        )
        job_manager_meta = result.get("job_manager", {}) if isinstance(result, dict) else {}
        warnings.extend(job_manager_meta.get("warnings", []))
        recoveries.extend(job_manager_meta.get("recovery_actions", []))
        return self._attach_agent_context(result, warnings=warnings, recoveries=recoveries)

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
        metadata_overrides: Optional[Dict[str, Any]] = None,
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
            metadata_overrides=metadata_overrides,
        )

    def generate_xas_inputs(
        self,
        structure_file: str,
        output_dir: Optional[str] = None,
        nersc_account: Optional[str] = None,
        nersc_queue: Optional[str] = None,
        nersc_nodes: Optional[int] = None,
        nersc_walltime: Optional[str] = None,
        email: Optional[str] = None,
        cluster_radius: float = 6.0,
        fdmnes_options: Optional[Dict[str, bool]] = None,
        fdmnes_energy_range=(-5.0, 0.2, 50.0),
        edge_override: Optional[str] = None,
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
        )

    def run_nersc_workflow(
        self,
        structure_file: str,
        output_dir: Optional[str] = None,
        software: str = "all",
        submit: bool = True,
        monitor: bool = True,
        parse_when_complete: bool = True,
        **kwargs,
    ) -> Dict:
        """Direct API for NERSC generate -> submit -> monitor -> parse workflow."""
        return execute_nersc_full_workflow(
            structure_file=structure_file,
            output_dir=output_dir or self.default_output_dir,
            software=software,
            submit=submit,
            monitor=monitor,
            parse_when_complete=parse_when_complete,
            **kwargs,
        )

    def parse_results(
        self,
        output_dir: str,
        software: str,
        structure_file: str,
        absorber: Optional[str] = None,
        edge: Optional[str] = None,
        vasp_mode: str = "trace",
    ) -> Dict:
        """Direct API for result parsing."""
        result = execute_result_parsing(
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
