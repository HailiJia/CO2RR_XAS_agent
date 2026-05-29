"""
ALCF Tests for CO2RR XAS Agent
===============================
Run on ALCF (Polaris/Aurora) with: python tests/test_alcf.py

These tests use natural language requests to test the full agent workflow.
They simulate how a user would interact with the agent in production.
"""

import sys
import tempfile
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test configuration
# =============================================================================

# Set your ALCF project here
ALCF_PROJECT = "YOUR_PROJECT_NAME"

# Output directory for test results
TEST_OUTPUT_DIR = Path.home() / "co2rr_xas_tests" / datetime.now().strftime("%Y%m%d_%H%M%S")


def setup_test_dir():
    """Create test output directory."""
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Test output directory: {TEST_OUTPUT_DIR}")
    return TEST_OUTPUT_DIR


# =============================================================================
# Natural language test scenarios
# =============================================================================

def test_scenario_1_cu_slab_vasp_xas():
    """
    Scenario 1: Generate Cu(111) slab and VASP XAS inputs
    
    User request: "I want to simulate Cu K-edge XANES for a Cu(111) surface 
    with 4 layers and 15 Angstrom vacuum"
    """
    print("\n" + "=" * 70)
    print("SCENARIO 1: Cu(111) VASP XAS")
    print("=" * 70)
    
    from agent.agent_core import CO2RRXASAgent
    
    output_dir = TEST_OUTPUT_DIR / "scenario_1_cu_vasp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = CO2RRXASAgent(hpc_system='polaris')
    
    # Request 1: Generate the slab
    print("\n--- Request 1: Generate slab ---")
    response = agent.process_request(
        "Create a Cu(111) surface slab with 4 layers and 15 Angstrom vacuum",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    if response['status'] != 'success':
        print(f"❌ Failed: {response.get('message')}")
        return False
    
    # Request 2: Generate VASP XAS inputs
    print("\n--- Request 2: Generate VASP XAS inputs ---")
    response = agent.process_request(
        "Now set up a Cu K-edge XANES calculation using VASP",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    # Request 3: Generate job script
    print("\n--- Request 3: Generate job script ---")
    response = agent.process_request(
        "Create a job script for Polaris with 2 nodes and 2 hour walltime",
        output_dir=str(output_dir),
        project=ALCF_PROJECT
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    # List generated files
    print("\n--- Generated files ---")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")
    
    print(f"\n✅ Scenario 1 complete. Files in: {output_dir}")
    return True


def test_scenario_2_au_co2_fdmnes():
    """
    Scenario 2: Au(111) with CO2 adsorbate, FDMNES calculation
    
    User request: "I need to calculate Au L3-edge XANES for Au(111) with 
    CO2 adsorbed on a bridge site using FDMNES"
    """
    print("\n" + "=" * 70)
    print("SCENARIO 2: Au(111) + CO2, FDMNES L3-edge")
    print("=" * 70)
    
    from agent.agent_core import CO2RRXASAgent
    
    output_dir = TEST_OUTPUT_DIR / "scenario_2_au_fdmnes"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = CO2RRXASAgent(hpc_system='polaris')
    
    # Request 1: Generate slab with adsorbate
    print("\n--- Request 1: Generate Au(111) with CO2 ---")
    response = agent.process_request(
        "Generate an Au(111) surface with CO2 adsorbed on a bridge site",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    if response['status'] != 'success':
        print(f"❌ Failed: {response.get('message')}")
        return False
    
    # Request 2: Generate FDMNES inputs for L3-edge
    print("\n--- Request 2: Generate FDMNES L3-edge inputs ---")
    response = agent.process_request(
        "Set up FDMNES calculation for Au L3-edge XANES",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    # Check that L3-edge specific settings were applied
    fdmfile = output_dir / "fdmfile.txt"
    if fdmfile.exists():
        # Read the actual input file
        with open(fdmfile) as f:
            content = f.read()
        
        # Find the referenced input file
        for line in content.strip().split('\n'):
            if line.strip() and not line.strip().isdigit():
                inp_file = output_dir / line.strip()
                if inp_file.exists():
                    inp_content = inp_file.read_text()
                    
                    if 'Spinorbit' in inp_content:
                        print("  ✓ Spinorbit enabled for L3-edge")
                    else:
                        print("  ⚠ Spinorbit not found (expected for L3-edge)")
                    
                    if 'Quadrupole' in inp_content:
                        print("  ✓ Quadrupole enabled for L3-edge")
                break
    
    # List generated files
    print("\n--- Generated files ---")
    for f in sorted(output_dir.iterdir()):
        print(f"  {f.name}")
    
    print(f"\n✅ Scenario 2 complete. Files in: {output_dir}")
    return True


def test_scenario_3_pt_feff_exafs():
    """
    Scenario 3: Pt(111) FEFF EXAFS calculation
    
    User request: "Calculate Pt L3-edge EXAFS for a Pt(111) surface using FEFF"
    """
    print("\n" + "=" * 70)
    print("SCENARIO 3: Pt(111) FEFF EXAFS")
    print("=" * 70)
    
    from agent.agent_core import CO2RRXASAgent
    
    output_dir = TEST_OUTPUT_DIR / "scenario_3_pt_feff"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = CO2RRXASAgent(hpc_system='polaris')
    
    # Request 1: Generate Pt slab
    print("\n--- Request 1: Generate Pt(111) slab ---")
    response = agent.process_request(
        "Create a Pt(111) surface with 4 layers",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    # Request 2: Generate FEFF inputs
    print("\n--- Request 2: Generate FEFF inputs ---")
    response = agent.process_request(
        "Generate FEFF input for Pt L3-edge EXAFS calculation",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    print(f"Message: {response['message']}")
    
    # Check feff.inp
    feff_inp = output_dir / "feff.inp"
    if feff_inp.exists():
        content = feff_inp.read_text()
        
        if 'EDGE L3' in content or 'EDGE  L3' in content:
            print("  ✓ L3 edge specified")
        
        if 'EXAFS' in content:
            print("  ✓ EXAFS calculation type")
        elif 'XANES' in content:
            print("  ⚠ XANES found (requested EXAFS)")
    
    print(f"\n✅ Scenario 3 complete. Files in: {output_dir}")
    return True


def test_scenario_4_modify_parameters():
    """
    Scenario 4: Modify parameters after initial generation
    
    User starts with defaults, then asks to change vacuum and add more layers
    """
    print("\n" + "=" * 70)
    print("SCENARIO 4: Modify Parameters")
    print("=" * 70)
    
    from agent.agent_core import CO2RRXASAgent
    
    output_dir = TEST_OUTPUT_DIR / "scenario_4_modify"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = CO2RRXASAgent(hpc_system='polaris')
    
    # Request 1: Initial slab
    print("\n--- Request 1: Initial Cu(111) slab ---")
    response = agent.process_request(
        "Create a Cu(111) slab",
        output_dir=str(output_dir)
    )
    print(f"Status: {response['status']}")
    
    initial_params = agent.current_params.copy()
    print(f"Initial vacuum: {initial_params.get('vacuum', 'default')}")
    print(f"Initial layers: {initial_params.get('layers', 'default')}")
    
    # Request 2: Modify parameters
    print("\n--- Request 2: Modify parameters ---")
    agent.modify_parameters({
        "vacuum": 25.0,
        "layers": 6
    })
    
    modified_params = agent.current_params
    print(f"Modified vacuum: {modified_params.get('vacuum')}")
    print(f"Modified layers: {modified_params.get('layers')}")
    
    if modified_params.get('vacuum') == 25.0:
        print("  ✓ Vacuum modified to 25 Å")
    else:
        print("  ❌ Vacuum modification failed")
        return False
    
    if modified_params.get('layers') == 6:
        print("  ✓ Layers modified to 6")
    else:
        print("  ❌ Layers modification failed")
        return False
    
    print(f"\n✅ Scenario 4 complete.")
    return True


def test_scenario_5_co2rr_workflow():
    """
    Scenario 5: Full CO2RR workflow
    
    Simulate a realistic CO2RR research workflow:
    1. Create Cu(111) surface
    2. Add CO2 adsorbate
    3. Generate VASP inputs for Cu K-edge
    4. Generate FEFF inputs for comparison
    5. Create job scripts for both
    """
    print("\n" + "=" * 70)
    print("SCENARIO 5: Full CO2RR Workflow")
    print("=" * 70)
    
    from agent.agent_core import CO2RRXASAgent
    
    output_dir = TEST_OUTPUT_DIR / "scenario_5_co2rr"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    agent = CO2RRXASAgent(hpc_system='polaris')
    
    # Step 1: Create Cu(111) with CO2
    print("\n--- Step 1: Create Cu(111) with CO2 adsorbate ---")
    response = agent.process_request(
        "I'm studying CO2 reduction on copper. Create a Cu(111) surface "
        "with CO2 adsorbed on a bridge site. Use 4 layers and 20 Angstrom vacuum.",
        output_dir=str(output_dir / "structure")
    )
    print(f"Status: {response['status']}")
    
    if response['status'] != 'success':
        print(f"❌ Failed at step 1")
        return False
    
    # Step 2: VASP XAS
    print("\n--- Step 2: Generate VASP Cu K-edge inputs ---")
    vasp_dir = output_dir / "vasp_xas"
    vasp_dir.mkdir(exist_ok=True)
    
    response = agent.process_request(
        "Set up VASP calculation for Cu K-edge XANES to study the electronic "
        "structure changes upon CO2 adsorption",
        output_dir=str(vasp_dir)
    )
    print(f"Status: {response['status']}")
    
    # Step 3: FEFF for comparison
    print("\n--- Step 3: Generate FEFF inputs for comparison ---")
    feff_dir = output_dir / "feff_xas"
    feff_dir.mkdir(exist_ok=True)
    
    # Reset agent to use same structure but different calculation
    response = agent.process_request(
        "Also generate FEFF input for Cu K-edge to compare with VASP results",
        output_dir=str(feff_dir)
    )
    print(f"Status: {response['status']}")
    
    # Step 4: Job scripts
    print("\n--- Step 4: Generate job scripts ---")
    
    response = agent.process_request(
        "Create a job script for the VASP calculation on Polaris",
        output_dir=str(vasp_dir),
        project=ALCF_PROJECT
    )
    print(f"VASP job script: {response['status']}")
    
    # Summary
    print("\n--- Generated directory structure ---")
    for subdir in ['structure', 'vasp_xas', 'feff_xas']:
        subpath = output_dir / subdir
        if subpath.exists():
            print(f"\n  {subdir}/")
            for f in sorted(subpath.iterdir()):
                print(f"    {f.name}")
    
    print(f"\n✅ Scenario 5 complete. Files in: {output_dir}")
    return True


def test_scenario_6_different_metals():
    """
    Scenario 6: Compare different metal surfaces
    
    Generate XAS inputs for Cu, Au, and Pt surfaces for comparison study
    """
    print("\n" + "=" * 70)
    print("SCENARIO 6: Compare Different Metals")
    print("=" * 70)
    
    from agent.agent_core import CO2RRXASAgent
    
    output_dir = TEST_OUTPUT_DIR / "scenario_6_metals"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metals = [
        ("Cu", "K", "8979 eV"),
        ("Au", "L3", "11919 eV"),
        ("Pt", "L3", "11564 eV"),
    ]
    
    for element, edge, energy in metals:
        print(f"\n--- {element} {edge}-edge ({energy}) ---")
        
        metal_dir = output_dir / f"{element}_{edge}_edge"
        metal_dir.mkdir(exist_ok=True)
        
        agent = CO2RRXASAgent(hpc_system='polaris')
        
        # Generate slab
        response = agent.process_request(
            f"Create a {element}(111) surface with 4 layers",
            output_dir=str(metal_dir)
        )
        
        if response['status'] == 'success':
            print(f"  ✓ {element}(111) slab generated")
        else:
            print(f"  ❌ Failed to generate {element} slab")
            continue
        
        # Generate FDMNES inputs (works for all edges)
        response = agent.process_request(
            f"Generate FDMNES input for {element} {edge}-edge XANES",
            output_dir=str(metal_dir)
        )
        
        if response['status'] == 'success':
            print(f"  ✓ FDMNES inputs generated")
        else:
            print(f"  ⚠ FDMNES generation: {response['status']}")
    
    print(f"\n✅ Scenario 6 complete. Files in: {output_dir}")
    return True


# =============================================================================
# Main test runner
# =============================================================================

def run_all_scenarios():
    """Run all ALCF test scenarios."""
    print("\n" + "=" * 70)
    print("CO2RR XAS AGENT - ALCF TESTS")
    print("=" * 70)
    print(f"Project: {ALCF_PROJECT}")
    print(f"Output: {TEST_OUTPUT_DIR}")
    
    setup_test_dir()
    
    scenarios = [
        ("Scenario 1: Cu VASP XAS", test_scenario_1_cu_slab_vasp_xas),
        ("Scenario 2: Au+CO2 FDMNES", test_scenario_2_au_co2_fdmnes),
        ("Scenario 3: Pt FEFF EXAFS", test_scenario_3_pt_feff_exafs),
        ("Scenario 4: Modify Parameters", test_scenario_4_modify_parameters),
        ("Scenario 5: Full CO2RR Workflow", test_scenario_5_co2rr_workflow),
        ("Scenario 6: Compare Metals", test_scenario_6_different_metals),
    ]
    
    results = []
    
    for name, test_func in scenarios:
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n❌ Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, "ERROR"))
    
    # Summary
    print("\n" + "=" * 70)
    print("SCENARIO SUMMARY")
    print("=" * 70)
    
    passed = 0
    for name, result in results:
        if result == "PASSED":
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: {result}")
    
    print(f"\n{passed}/{len(results)} scenarios passed")
    print(f"\nAll outputs saved to: {TEST_OUTPUT_DIR}")
    
    # Create summary file
    summary_file = TEST_OUTPUT_DIR / "test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "project": ALCF_PROJECT,
            "results": dict(results),
            "passed": passed,
            "total": len(results),
        }, f, indent=2)
    
    print(f"Summary saved to: {summary_file}")
    
    return passed == len(results)


def run_quick_test():
    """Run a quick single scenario for testing."""
    print("\n" + "=" * 70)
    print("QUICK TEST - Single Scenario")
    print("=" * 70)
    
    setup_test_dir()
    
    return test_scenario_1_cu_slab_vasp_xas()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ALCF tests for CO2RR XAS Agent")
    parser.add_argument("--quick", action="store_true", help="Run quick single test")
    parser.add_argument("--project", type=str, default="YOUR_PROJECT_NAME", 
                       help="ALCF project name")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory for test results")
    
    args = parser.parse_args()
    
    ALCF_PROJECT = args.project
    
    if args.output:
        TEST_OUTPUT_DIR = Path(args.output)
    
    if args.quick:
        success = run_quick_test()
    else:
        success = run_all_scenarios()
    
    sys.exit(0 if success else 1)