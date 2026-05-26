"""
Example usage of CO2RR XAS Agent
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agent.co2rr_xas_agent import create_agent, process_request


def example_natural_language():
    """Examples using natural language interface."""
    print("=" * 60)
    print("CO2RR XAS Agent - Natural Language Examples")
    print("=" * 60)
    
    # Example 1: Simple surface with adsorbate
    print("\n1. Generate Cu(111) surface with CO adsorbate")
    result = process_request(
        "Generate Cu(111) surface with CO adsorbate",
        output_dir="examples/output/example1"
    )
    print(f"   Status: {result['status']}")
    print(f"   Structures: {len(result.get('structures', []))}")
    
    # Example 2: Interface structure
    print("\n2. Create Cu-Au interface with CHO")
    result = process_request(
        "Create Cu-Au interface with CHO adsorbate",
        output_dir="examples/output/example2"
    )
    print(f"   Status: {result['status']}")
    
    # Example 3: Full CO2RR pathway
    print("\n3. Full CO2RR pathway on Ag(111)")
    result = process_request(
        "Full CO2RR pathway on Ag catalyst",
        output_dir="examples/output/example3"
    )
    print(f"   Status: {result['status']}")
    print(f"   Structures: {len(result.get('structures', []))}")
    
    # Example 4: XAS inputs for specific structure
    print("\n4. XAS inputs for OCCO on Pt(100)")
    result = process_request(
        "Generate XAS inputs for OCCO on Pt(100) surface",
        output_dir="examples/output/example4"
    )
    print(f"   Status: {result['status']}")
    
    # Example 5: Full workflow with interface
    print("\n5. Full XAS workflow for Cu-Au interface")
    result = process_request(
        "Complete XAS workflow for CO2RR on Cu-Au interface",
        output_dir="examples/output/example5",
        nersc_account="m1234",
        email="user@example.com"
    )
    print(f"   Status: {result['status']}")


def example_direct_api():
    """Examples using direct API."""
    print("\n" + "=" * 60)
    print("CO2RR XAS Agent - Direct API Examples")
    print("=" * 60)
    
    agent = create_agent("examples/output/api_examples")
    
    # Example 1: Generate single structure
    print("\n1. Generate Ni(110) surface with CH2")
    result = agent.generate_structure(
        metal1="Ni",
        facet1="110",
        adsorbate="CH2",
        site="bridge",
        supercell=(4, 4),
        layers=5
    )
    print(f"   Status: {result['status']}")
    
    # Example 2: Generate interface
    print("\n2. Generate Pd-Ag interface with COCO")
    result = agent.generate_structure(
        metal1="Pd",
        metal2="Ag",
        facet1="111",
        adsorbate="COCO",
        output_dir="examples/output/api_examples/pd_ag_coco"
    )
    print(f"   Status: {result['status']}")
    
    # Example 3: Full pathway
    print("\n3. Full CO2RR pathway on Au(111)")
    result = agent.generate_structure(
        metal1="Au",
        facet1="111",
        full_pathway=True,
        output_dir="examples/output/api_examples/au_pathway"
    )
    print(f"   Status: {result['status']}")
    print(f"   Structures: {len(result.get('structures', []))}")


def show_folder_structure():
    """Show expected folder structure."""
    print("\n" + "=" * 60)
    print("Expected Folder Structure")
    print("=" * 60)
    
    structure = """
    output/
    ├── CO/
    │   ├── structure/
    │   │   ├── POSCAR
    │   │   └── structure_info.json
    │   ├── relax/
    │   │   ├── POSCAR
    │   │   ├── INCAR
    │   │   ├── KPOINTS
    │   │   ├── POTCAR.spec
    │   │   └── submit_relax.sh
    │   └── xas/
    │       ├── Cu_K_edge/
    │       │   ├── FDMNES_fdm/
    │       │   │   ├── fdmfile.txt
    │       │   │   └── submit.sh
    │       │   ├── FDMNES_green/
    │       │   │   ├── fdmfile.txt
    │       │   │   └── submit.sh
    │       │   ├── FEFF/
    │       │   │   ├── feff.inp
    │       │   │   └── submit.sh
    │       │   ├── VASP_GW/
    │       │   │   ├── POSCAR
    │       │   │   ├── INCAR
    │       │   │   ├── KPOINTS
    │       │   │   ├── POTCAR.spec
    │       │   │   └── submit.sh
    │       │   └── VASP_PBE/
    │       │       └── ...
    │       └── C_K_edge/
    │           ├── FDMNES_fdm/
    │           ├── FDMNES_green/
    │           ├── FEFF/
    │           ├── VASP_GW/
    │           └── VASP_PBE/
    ├── CHO/
    │   └── ...
    └── ...
    """
    print(structure)


if __name__ == "__main__":
    example_natural_language()
    example_direct_api()
    show_folder_structure()
    
    print("\n" + "=" * 60)
    print("Examples complete! Check 'examples/output/' for generated files.")
    print("=" * 60)