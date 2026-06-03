"""Custodian jobs and handlers for XAS output validation.

These classes provide code-specific validation for completed FEFF/FDMNES/VASP
work directories before the normal ISAAC parser consumes output files.  The goal
is to catch stale, truncated, or mismatched outputs at the Custodian execution
layer, while still allowing the parser to embed the same validation details in
ISAAC QC metadata.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from custodian.custodian import Custodian, ErrorHandler, Job

from tools.result_parser import ResultParser, infer_xas_metadata, validate_output_consistency


class XASOutputValidationJob(Job):
    """Custodian job that validates a completed XAS output directory."""

    def __init__(
        self,
        software: str,
        output_dir: str,
        context: Dict[str, Any],
        structure_file: Optional[str] = None,
        absorber: Optional[str] = None,
        edge: Optional[str] = None,
        vasp_mode: str = "trace",
        parameters: Optional[Dict[str, Any]] = None,
    ):
        self.software = software
        self.output_dir = output_dir
        self.context = context
        self.structure_file = structure_file
        self.absorber = absorber
        self.edge = edge
        self.vasp_mode = vasp_mode
        self.parameters = dict(parameters or {})

    @property
    def name(self) -> str:
        return f"validate_{self.software.lower()}_output"

    def setup(self, directory: str = "./") -> None:
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def run(self, directory: str = "./") -> None:
        if self.context.get("validation_handled"):
            return None
        parser = ResultParser()
        parsed = parser.parse_results(self.output_dir, self.software, self.vasp_mode)
        if parsed.get("status") != "success":
            validation = {
                "status": "error",
                "checks": [],
                "warnings": [],
                "errors": [parsed.get("message", "Could not parse output for validation.")],
                "source_file": parsed.get("source_file"),
            }
        else:
            inferred = infer_xas_metadata(self.output_dir, self.software, structure_file=self.structure_file)
            absorber = self.absorber or inferred.get("absorber")
            edge = self.edge or inferred.get("edge")
            validation = validate_output_consistency(
                output_dir=self.output_dir,
                software=self.software,
                results=parsed,
                absorber=absorber,
                edge=edge,
                parameters=self.parameters,
            )
            validation["metadata_inference"] = inferred

        self.context["output_validation"] = validation
        issue = validation.get("status")
        if issue in {"warning", "error"}:
            self.context["xas_validation_issue"] = issue
        if issue == "error" and self.parameters.get("strict_output_validation"):
            self.context["skip_parse"] = True

    def postprocess(self, directory: str = "./") -> None:
        return None

    def as_dict(self) -> Dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "software": self.software,
            "output_dir": self.output_dir,
        }


class XASOutputValidationHandler(ErrorHandler):
    """Custodian handler that records XAS validation warnings/errors."""

    raises_runtime_error = False

    def __init__(self, context: Dict[str, Any]):
        self.context = context

    def check(self, directory: str = "./") -> bool:
        return bool(self.context.get("xas_validation_issue"))

    def correct(self, directory: str = "./") -> Dict[str, Any]:
        issue = self.context.pop("xas_validation_issue", None)
        validation = self.context.get("output_validation", {})
        actions = []
        errors = []
        if issue == "warning":
            msg = "Custodian recorded XAS output-validation warnings; ISAAC QC will include details."
            self.context.setdefault("warnings", []).append(msg)
            actions.append(msg)
        elif issue == "error":
            msg = "Custodian detected XAS output/input mismatch before ISAAC parsing."
            errors.extend(validation.get("errors") or [msg])
            if self.context.get("strict_output_validation"):
                action = "Strict output validation enabled; skipping ISAAC record creation for this job."
                self.context.setdefault("warnings", []).append(action)
                actions.append(action)
            else:
                action = "Non-strict validation; continuing so ISAAC QC records the mismatch."
                self.context.setdefault("warnings", []).append(action)
                actions.append(action)
        self.context["validation_handled"] = True
        return {"errors": errors, "actions": actions}

    def as_dict(self) -> Dict[str, Any]:
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
        }


def run_xas_output_validation(
    software: str,
    output_dir: str,
    structure_file: Optional[str] = None,
    absorber: Optional[str] = None,
    edge: Optional[str] = None,
    vasp_mode: str = "trace",
    parameters: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run Custodian-backed validation for a completed XAS output directory."""
    parameters = dict(parameters or {})
    context: Dict[str, Any] = {
        "software": software,
        "output_dir": output_dir,
        "strict_output_validation": bool(parameters.get("strict_output_validation")),
        "warnings": [],
    }
    job = XASOutputValidationJob(
        software=software,
        output_dir=output_dir,
        context=context,
        structure_file=structure_file,
        absorber=absorber,
        edge=edge,
        vasp_mode=vasp_mode,
        parameters=parameters,
    )
    custodian = Custodian(
        handlers=[XASOutputValidationHandler(context)],
        jobs=[job],
        max_errors=int(os.environ.get("CO2RR_XAS_VALIDATION_MAX_ERRORS", 3)),
        directory=output_dir,
        terminate_on_nonzero_returncode=False,
    )
    run_log = custodian.run()
    return {
        "backend": "custodian",
        "run_log": run_log,
        "output_validation": context.get("output_validation", {}),
        "warnings": context.get("warnings", []),
        "skip_parse": bool(context.get("skip_parse")),
    }
