"""
Example usage of the CO2RR XAS Agent.

The local parser works offline but is regex based. For flexible natural language,
use the LLM planner by setting environment variables and passing use_llm=True.
"""

import os
import sys

# Add project root to path when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.co2rr_xas_agent import create_agent, process_request
from agent.planner import create_llm_planner_from_env


def example_local_parser():
    """Examples using the offline regex parser."""
    print("=" * 70)
    print("CO2RR XAS Agent - Local parser examples")
    print("=" * 70)

    result = process_request(
        "Generate Cu(111) surface with CO adsorbate",
        output_dir="examples/output/local_cu_co",
    )
    print("Local structure status:", result["status"])

    result = process_request(
        "Generate FDMNES Green XAS inputs without SCF, with Quadrupole and Spinorbit "
        "for CO on Cu(111), radius 7 Angstrom, energy range -5 0.2 50, K edge, "
        "NERSC account m5268",
        output_dir="examples/output/local_fdmnes_green",
    )
    print("Local XAS status:", result["status"])


def example_llm_planner():
    """Examples using the LLM planner for flexible natural language."""
    print("=" * 70)
    print("CO2RR XAS Agent - LLM planner example")
    print("=" * 70)

    # Required environment variables:
    #   CO2RR_LLM_MODEL or OPENAI_MODEL or ALCF_MODEL
    #   CO2RR_LLM_API_KEY or OPENAI_API_KEY or ALCF_API_KEY or ALCF_TOKEN
    # Optional for ALCF/AskSage/OpenAI-compatible gateways:
    #   CO2RR_LLM_BASE_URL or OPENAI_BASE_URL or ALCF_BASE_URL
    planner = create_llm_planner_from_env()

    request = (
        "Generate FDMNES Green multiple scattering XAS inputs without SCF, "
        "with Quadrupole and Spinorbit for CO on Cu(111), radius 7 Angstrom, "
        "energy range start from -5 eV to 50 eV with step size 0.2 eV, "
        "K edge, NERSC account m5268"
    )

    result = process_request(
        request,
        output_dir="examples/output/llm_fdmnes_green",
        planner=planner,
        use_llm=True,
    )
    print("LLM XAS status:", result["status"])


def example_direct_api():
    """Examples using direct Python API, with no natural-language parsing."""
    print("=" * 70)
    print("CO2RR XAS Agent - Direct API examples")
    print("=" * 70)

    agent = create_agent("examples/output/api_examples")

    result = agent.generate_structure(
        metal1="Ni",
        facet1="110",
        adsorbate="CH2",
        site="bridge",
        supercell=(4, 4),
        layers=5,
    )
    print("Direct structure status:", result["status"])

    result = agent.generate_xas_inputs(
        structure_file="examples/output/api_examples/CH2/structure/POSCAR",
        output_dir="examples/output/api_examples/CH2",
        nersc_account="m5268",
        cluster_radius=7.0,
        fdmnes_options={
            "Green": True,
            "SCF": False,
            "Quadrupole": True,
            "Spinorbit": True,
            "Relativism": False,
            "SCFexc": False,
            "Screening": False,
            "Full_atom": False,
            "TDDFT": False,
        },
        fdmnes_energy_range=(-5.0, 50.0, 0.2),  # semantic order also accepted
        edge_override="K",
    )
    print("Direct XAS status:", result["status"])


if __name__ == "__main__":
    example_local_parser()

    # Uncomment after setting LLM environment variables.
    # example_llm_planner()

    # Uncomment when the structure path exists.
    # example_direct_api()
