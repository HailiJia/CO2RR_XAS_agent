"""
Shared schemas for the CO2RR XAS Agent.

This module intentionally contains no LLM client code. It is safe to import from
both the local regex parser and the execution layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field


@dataclass
class ParsedIntent:
    """Parsed user intent consumed by CO2RRXASAgent execution methods."""

    action: str  # generate_structure, generate_xas, parse_results, full_workflow, nersc_workflow
    metals: List[str]
    facets: List[str]
    adsorbates: List[str]
    full_pathway: bool
    structure_file: Optional[str]
    output_dir: str
    parameters: Dict[str, Any]


class EnergyRange(BaseModel):
    """
    Canonical semantic representation of an XAS energy grid.

    Internally the semantic order is (emin, emax, step). For FDMNES output,
    use as_fdmnes_range(), which returns (emin, step, emax).
    """

    emin: float = Field(description="Lower energy relative to edge, in eV")
    emax: float = Field(description="Upper energy relative to edge, in eV")
    step: float = Field(description="Positive energy step size, in eV")
    original_text: Optional[str] = Field(default=None, description="Original user phrase, if available")

    def normalized(self) -> "EnergyRange":
        emin = float(self.emin)
        emax = float(self.emax)
        step = abs(float(self.step))

        if emax < emin:
            emin, emax = emax, emin

        if step <= 0:
            raise ValueError("Energy step must be positive.")
        if emax <= emin:
            raise ValueError("Energy range must have finite width.")

        return EnergyRange(emin=emin, emax=emax, step=step, original_text=self.original_text)

    def as_fdmnes_range(self) -> Tuple[float, float, float]:
        """Return FDMNES Range card order: (emin, step, emax)."""
        er = self.normalized()
        return (er.emin, er.step, er.emax)

    @classmethod
    def default(cls) -> "EnergyRange":
        return cls(emin=-5.0, emax=50.0, step=0.2)

    @classmethod
    def from_any(cls, value: Optional[Union["EnergyRange", Dict[str, Any], Sequence[float]]]) -> "EnergyRange":
        """
        Build EnergyRange from LLM output, direct API input, or legacy tuples.

        Supported tuple/list conventions:
            (-5, 0.2, 50)  -> FDMNES order: emin, step, emax
            (-5, 50, 0.2)  -> semantic order: emin, emax, step
        """
        if value is None:
            return cls.default()

        if isinstance(value, EnergyRange):
            return value.normalized()

        if isinstance(value, dict):
            data = dict(value)
            # Accept common aliases from less strict LLMs.
            if "start" in data and "emin" not in data:
                data["emin"] = data.pop("start")
            if "end" in data and "emax" not in data:
                data["emax"] = data.pop("end")
            if "interval" in data and "step" not in data:
                data["step"] = data.pop("interval")
            if "spacing" in data and "step" not in data:
                data["step"] = data.pop("spacing")
            return cls(**data).normalized()

        values = list(value)
        if len(values) != 3:
            raise ValueError(f"Energy range must contain exactly 3 values, got {value!r}")

        a, b, c = [float(v) for v in values]

        # If middle value is small compared with total span, treat as FDMNES order.
        # Example: (-5, 0.2, 50) -> emin=-5, step=0.2, emax=50.
        if abs(b) < abs(c - a):
            return cls(emin=a, step=abs(b), emax=c).normalized()

        # Otherwise treat as semantic order.
        # Example: (-5, 50, 0.2) -> emin=-5, emax=50, step=0.2.
        return cls(emin=a, emax=b, step=abs(c)).normalized()


class FDMNESOptions(BaseModel):
    """LLM-facing FDMNES options."""

    method: Literal["fdm", "green"] = Field(default="fdm", description="fdm = finite difference; green = Green multiple scattering")
    quadrupole: bool = False
    relativism: bool = False
    spinorbit: bool = False
    scf: bool = True
    scfexc: bool = False
    screening: bool = False
    full_atom: bool = False
    tddft: bool = False

    def to_fdmnes_dict(self) -> Dict[str, bool]:
        """Return the exact option keys expected by FDMNESInputGenerator."""
        return {
            "Quadrupole": bool(self.quadrupole),
            "Relativism": bool(self.relativism),
            "Spinorbit": bool(self.spinorbit),
            "SCF": bool(self.scf),
            "SCFexc": bool(self.scfexc),
            "Screening": bool(self.screening),
            "Full_atom": bool(self.full_atom),
            "TDDFT": bool(self.tddft),
            "Green": self.method == "green",
        }

    @classmethod
    def from_fdmnes_dict(cls, value: Optional[Dict[str, Any]]) -> "FDMNESOptions":
        if not value:
            return cls()
        return cls(
            method="green" if value.get("Green", False) else "fdm",
            quadrupole=bool(value.get("Quadrupole", False)),
            relativism=bool(value.get("Relativism", False)),
            spinorbit=bool(value.get("Spinorbit", False)),
            scf=bool(value.get("SCF", True)),
            scfexc=bool(value.get("SCFexc", False)),
            screening=bool(value.get("Screening", False)),
            full_atom=bool(value.get("Full_atom", False)),
            tddft=bool(value.get("TDDFT", False)),
        )


class XASPlan(BaseModel):
    """Structured plan generated by the LLM planner."""

    action: Literal["generate_structure", "generate_xas", "parse_results", "full_workflow", "nersc_workflow"]
    software: Literal["fdmnes", "feff", "vasp", "all"] = "all"

    metals: List[str] = Field(default_factory=list)
    facets: List[str] = Field(default_factory=lambda: ["111"])
    adsorbates: List[str] = Field(default_factory=list)
    full_pathway: bool = False
    structure_file: Optional[str] = None

    absorber_elements: List[str] = Field(default_factory=list)
    edge: Optional[Literal["K", "L2", "L3", "L23"]] = None
    cluster_radius: Optional[float] = None
    energy_range: Optional[EnergyRange] = None
    fdmnes: Optional[FDMNESOptions] = None
    vasp_method: Optional[Literal["PBE", "GW", "pbe", "gw"]] = Field(
        default=None,
        description="Selected VASP XAS input flavor. Generate one VASP folder using this method, not both.",
    )
    potcar_dir: Optional[str] = Field(
        default=None,
        description="Optional root directory containing POTCAR subfolders, e.g. Cu_pv/POTCAR.",
    )

    site: Optional[Literal["top", "bridge", "fcc", "hcp"]] = None
    supercell: Optional[List[int]] = Field(default=None, description="Two integers, e.g. [3, 3]")
    layers: Optional[int] = None

    nersc_account: Optional[str] = None
    nersc_queue: Optional[str] = None
    nersc_nodes: Optional[int] = None
    nersc_walltime: Optional[str] = None
    email: Optional[str] = None

    results_dir: Optional[str] = None
    vasp_mode: Optional[Literal["trace", "parallel", "normal", "tauc"]] = None
    description: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    submit_jobs: Optional[bool] = Field(default=None, description="Submit generated jobs to NERSC/SLURM with sbatch.")
    monitor_jobs: Optional[bool] = Field(default=None, description="Monitor submitted jobs with squeue/sacct until they finish.")
    parse_when_complete: Optional[bool] = Field(default=None, description="Parse completed job outputs into ISAAC records.")
    poll_interval: Optional[int] = Field(default=None, description="Polling interval in seconds for NERSC job monitoring.")
    monitor_timeout: Optional[int] = Field(default=None, description="Optional monitor timeout in seconds.")

    def to_parsed_intent(self, output_dir: str) -> ParsedIntent:
        """Convert LLM-facing XASPlan to execution-facing ParsedIntent."""
        parameters: Dict[str, Any] = {
            "software": self.software,
        }

        if self.cluster_radius is not None:
            parameters["cluster_radius"] = float(self.cluster_radius)
        if self.energy_range is not None:
            parameters["fdmnes_energy_range"] = self.energy_range.as_fdmnes_range()
        if self.fdmnes is not None:
            parameters["fdmnes_options"] = self.fdmnes.to_fdmnes_dict()
        if self.edge is not None:
            parameters["edge_override"] = self.edge
            parameters["edge"] = self.edge
        if self.vasp_method is not None:
            parameters["vasp_method"] = self.vasp_method.upper()
        if self.potcar_dir is not None:
            parameters["potcar_dir"] = self.potcar_dir
        if self.absorber_elements:
            parameters["absorber_elements"] = list(self.absorber_elements)
            parameters["absorber"] = self.absorber_elements[0]
        if self.site is not None:
            parameters["site"] = self.site
        if self.supercell is not None and len(self.supercell) >= 2:
            parameters["supercell"] = (int(self.supercell[0]), int(self.supercell[1]))
        if self.layers is not None:
            parameters["layers"] = int(self.layers)
        if self.nersc_account is not None:
            parameters["nersc_account"] = self.nersc_account
        if self.nersc_queue is not None:
            parameters["nersc_queue"] = self.nersc_queue
        if self.nersc_nodes is not None:
            parameters["nersc_nodes"] = int(self.nersc_nodes)
        if self.nersc_walltime is not None:
            parameters["nersc_walltime"] = self.nersc_walltime
        if self.email is not None:
            parameters["email"] = self.email
        if self.results_dir is not None:
            parameters["results_dir"] = self.results_dir
        if self.vasp_mode is not None:
            parameters["vasp_mode"] = self.vasp_mode
        if self.description is not None:
            parameters["description"] = self.description
        if self.tags:
            parameters["tags"] = list(self.tags)
        if self.submit_jobs is not None:
            parameters["submit_jobs"] = bool(self.submit_jobs)
        if self.monitor_jobs is not None:
            parameters["monitor_jobs"] = bool(self.monitor_jobs)
        if self.parse_when_complete is not None:
            parameters["parse_when_complete"] = bool(self.parse_when_complete)
        if self.poll_interval is not None:
            parameters["poll_interval"] = int(self.poll_interval)
        if self.monitor_timeout is not None:
            parameters["monitor_timeout"] = int(self.monitor_timeout)

        return ParsedIntent(
            action=self.action,
            metals=list(self.metals),
            facets=list(self.facets or ["111"]),
            adsorbates=list(self.adsorbates),
            full_pathway=bool(self.full_pathway),
            structure_file=self.structure_file,
            output_dir=output_dir,
            parameters=parameters,
        )


def model_json_schema_compat(model: type[BaseModel]) -> Dict[str, Any]:
    """Pydantic v1/v2 compatible schema export."""
    if hasattr(model, "model_json_schema"):
        return model.model_json_schema()
    return model.schema()
