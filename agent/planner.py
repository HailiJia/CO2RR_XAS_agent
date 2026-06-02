"""
LLM planner for the CO2RR XAS Agent.

This planner converts flexible natural language into an XASPlan JSON object, then
converts that plan into ParsedIntent for the existing deterministic Python tools.

It supports OpenAI-compatible chat/completions endpoints, including many hosted
or institutional gateways, by using OPENAI_BASE_URL / CO2RR_LLM_BASE_URL.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

try:
    from agent.schemas import ParsedIntent, XASPlan, model_json_schema_compat
except ImportError:  # pragma: no cover - useful when running from inside agent/
    from .schemas import ParsedIntent, XASPlan, model_json_schema_compat


try:
    from tools.utils import ADSORBATES, METAL_DATA
except Exception:  # pragma: no cover - planner can still run without tool imports
    ADSORBATES = {}
    METAL_DATA = {}


class LLMIntentPlanner:
    """
    Convert natural language into ParsedIntent using an OpenAI-compatible LLM.

    Parameters
    ----------
    model:
        Model name used by the endpoint.
    api_key:
        API key or token. If omitted, environment variables are used.
    base_url:
        Optional OpenAI-compatible base URL. Use this for ALCF/AskSage/internal
        gateways if they expose an OpenAI-compatible API.
    strict_schema:
        If True, first try JSON Schema response_format. If the endpoint rejects
        it, the planner automatically falls back to JSON-object/plain JSON mode.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        strict_schema: bool = True,
    ):
        self.model = model or os.getenv("CO2RR_LLM_MODEL") or os.getenv("OPENAI_MODEL") or os.getenv("ALCF_MODEL")
        self.api_key = api_key or os.getenv("CO2RR_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("ALCF_API_KEY") or os.getenv("ALCF_TOKEN")
        self.base_url = base_url or os.getenv("CO2RR_LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL") or os.getenv("ALCF_BASE_URL")
        self.temperature = temperature
        self.strict_schema = strict_schema

        if not self.model:
            raise ValueError(
                "No LLM model specified. Set CO2RR_LLM_MODEL or pass model=..."
            )
        if not self.api_key:
            raise ValueError(
                "No LLM API key/token specified. Set CO2RR_LLM_API_KEY, OPENAI_API_KEY, "
                "ALCF_API_KEY, or ALCF_TOKEN."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The LLM planner requires the openai package. Install it with: pip install openai"
            ) from exc

        kwargs = {"api_key": self.api_key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self.client = OpenAI(**kwargs)

    def parse(self, request: str, output_dir: str = "output") -> ParsedIntent:
        """Return ParsedIntent expected by CO2RRXASAgent."""
        plan = self.plan(request)
        return plan.to_parsed_intent(output_dir=output_dir)

    def plan(self, request: str) -> XASPlan:
        """Return the raw structured XASPlan from the LLM."""
        messages = [
            {"role": "system", "content": self._system_prompt()},
            {"role": "user", "content": request},
        ]

        schema = model_json_schema_compat(XASPlan)

        # First try strict JSON schema. Many OpenAI-compatible endpoints support
        # JSON mode but not full JSON Schema, so fall back gracefully.
        if self.strict_schema:
            try:
                content = self._chat(
                    messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "xas_plan",
                            "schema": schema,
                            "strict": False,
                        },
                    },
                )
                return self._parse_plan_content(content)
            except Exception:
                pass

        # Second try JSON object mode.
        try:
            content = self._chat(messages, response_format={"type": "json_object"})
            return self._parse_plan_content(content)
        except Exception:
            # Last fallback: plain text instruction, then extract JSON substring.
            fallback_messages = messages + [
                {
                    "role": "system",
                    "content": "Return only one valid JSON object. Do not include markdown fences.",
                }
            ]
            content = self._chat(fallback_messages, response_format=None)
            return self._parse_plan_content(content)

    def _chat(self, messages, response_format: Optional[Dict[str, Any]] = None) -> str:
        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
        }
        if response_format is not None:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)
        message = response.choices[0].message
        content = getattr(message, "content", None)
        if not content:
            raise ValueError("LLM returned an empty response.")
        return content

    def _parse_plan_content(self, content: str) -> XASPlan:
        data = self._extract_json(content)
        try:
            if hasattr(XASPlan, "model_validate"):
                return XASPlan.model_validate(data)
            return XASPlan.parse_obj(data)
        except Exception as exc:
            raise ValueError(f"LLM returned JSON, but it did not match XASPlan: {data}") from exc

    def _extract_json(self, content: str) -> Dict[str, Any]:
        text = content.strip()

        # Remove ```json fences if a non-strict endpoint adds them.
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
        if fence_match:
            text = fence_match.group(1)

        # If there is extra prose, extract the first JSON object.
        if not text.startswith("{"):
            obj_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
            if not obj_match:
                raise ValueError(f"Could not find JSON object in LLM response: {content}")
            text = obj_match.group(0)

        return json.loads(text)

    def _system_prompt(self) -> str:
        metals = ", ".join(sorted(METAL_DATA.keys())) if METAL_DATA else "Cu, Ag, Au, Pt, Pd, Ni, Fe, Co, Al"
        adsorbates = ", ".join(sorted(ADSORBATES.keys(), key=len, reverse=True)) if ADSORBATES else "CO, CHO, CHOH, COCO, OCCO"

        return f"""
You are the planning layer for a CO2RR XAS input-generation agent.
Your only job is to convert the user's natural-language request into one JSON object matching XASPlan.
The Python tools will generate files; do not generate file contents yourself.

Supported actions:
- generate_structure: only build POSCAR/metadata structures.
- generate_xas: generate relaxation and XAS input folders for a structure.
- full_workflow: generate structures and XAS inputs.
- nersc_workflow: generate NERSC inputs, submit jobs with sbatch, monitor with squeue/sacct, and parse completed outputs into ISAAC records.
- parse_results: parse FEFF/FDMNES/VASP outputs and create an ISAAC record.

Supported metals include: {metals}
Supported CO2RR adsorbates include: {adsorbates}
Supported facets include: 111, 100, 110, 0001.
Supported site values: top, bridge, fcc, hcp.
Supported software values: fdmnes, feff, vasp, all.
Supported edges: K, L2, L3, L23.
Supported vasp_method values: PBE, GW. Generate one VASP folder using the requested method, not both.

Rules:
1. Preserve chemical capitalization: CO is an adsorbate; Co is cobalt.
2. If the user mentions FDMNES Green, Green function, or multiple scattering, set fdmnes.method = "green".
3. If the user mentions finite difference or FDM, set fdmnes.method = "fdm".
4. If the user says without/no/disable SCF, set fdmnes.scf = false.
5. If the user says Quadrupole, Spinorbit/spin-orbit, Relativism, Screening, Full_atom/full atom, SCFexc, or TDDFT, set the corresponding boolean.
6. Energy range must be semantic JSON: {{"emin": -5.0, "emax": 50.0, "step": 0.2}}.
   Examples:
   - "energy range -5 0.2 50" -> emin=-5, emax=50, step=0.2
   - "from -5 eV to 50 eV with step size 0.2 eV" -> emin=-5, emax=50, step=0.2
   - "50 eV to -5 eV with interval 0.2 eV" -> emin=-5, emax=50, step=0.2
7. Use nersc_account for phrases like "NERSC account m5268" or "account m5268".
8. If the user asks to collect outputs, parse outputs, convert to ISAAC, or create ISAAC record, use action=parse_results.
9. If a structure file such as POSCAR, CONTCAR, or *.cif is supplied, put it in structure_file.
10. If no facet is specified, use ["111"].
11. If XAS is requested and no software is specified, use software="all".
12. If the user requests VASP GW/GW0, set vasp_method="GW". If the user requests VASP PBE, set vasp_method="PBE".
13. If no VASP method is specified, leave vasp_method unset; the execution layer defaults to PBE.
14. If the user gives a POTCAR/potpaw directory path, set potcar_dir to that exact path.
15. If the user asks to submit jobs, monitor jobs, run on NERSC, wait for completion, or write records after jobs finish, use action=nersc_workflow.
16. For nersc_workflow, set submit_jobs=true unless the user says dry run or do not submit; set monitor_jobs=true when the user asks to monitor/wait; set parse_when_complete=true when the user asks to write ISAAC records after completion.
17. Return only JSON.
""".strip()


def create_llm_planner_from_env(**kwargs) -> LLMIntentPlanner:
    """Convenience factory used by co2rr_xas_agent.create_agent(use_llm=True)."""
    return LLMIntentPlanner(**kwargs)
