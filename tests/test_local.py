"""
Local Tests for CO2RR XAS Agent
================================
Run with: python tests/test_local.py

These tests verify each component works correctly with explicit parameters.
No LLM or natural language processing required.
"""

import sys
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Test utilities
# =============================================================================

def print_header(name: str) -> None:
    print("\n" + "=" * 70)
    print(f"TEST: {name}")
    print("=" * 70)


def print_success(msg: str) -> None:
    print(f"  ✓ {msg}")


def print_fail(msg: str) -> None:
    print(f"  ✗ {msg}")


# =============================================================================
# Skill Registry Tests
# =============================================================================

def test_skill_registry_loading():
    """Test that all skill JSON files load correctly."""
    print_header("Skill Registry - Loading")
    
    from skills.skill_registry import SkillRegistry
    
    skills_dir = Path(__file__).parent.parent / "skills"
    registry = SkillRegistry(skills_dir)
    
    expected_skills = [
        'vasp_slab_generation',
        'vasp_xas_calculation',
        'feff_xas_calculation',
        'fdmnes_xas_calculation',
    ]
    
    loaded_skills = registry.list_skills()
    print(f"  Loaded skills: {loaded_skills}")
    
    for skill_id in expected_skills:
        if skill_id in loaded_skills:
            print_success(f"Skill '{skill_id}' loaded")
        else:
            print_fail(f"Skill '{skill_id}' NOT found")
            return False
    
    return True


def test_skill_registry_defaults():
    """Test that skill defaults are returned correctly."""
    print_header("Skill Registry - Defaults")
    
    from skills.skill_registry import SkillRegistry
    
    skills_dir = Path(__file__).parent.parent / "skills"
    registry = SkillRegistry(skills_dir)
    
    # Test VASP slab defaults
    defaults = registry.get_defaults('vasp_slab_generation')
    
    expected_defaults = {
        'layers': 4,
        'vacuum': 15.0,
        'fix_layers': 2,
    }
    
    for key, expected_value in expected_defaults.items():
        actual_value = defaults.get(key)
        if actual_value == expected_value:
            print_success(f"Default '{key}' = {actual_value}")
        else:
            print_fail(f"Default '{key}' = {actual_value}, expected {expected_value}")
            return False
    
    return True


def test_skill_registry_merge_with_override():
    """Test that user parameters override defaults."""
    print_header("Skill Registry - Parameter Override")
    
    from skills.skill_registry import SkillRegistry
    
    skills_dir = Path(__file__).parent.parent / "skills"
    registry = SkillRegistry(skills_dir)
    
    # User wants vacuum = 20 Å instead of default 15 Å
    user_params = {
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "vacuum": 20.0,  # Override
        "layers": 6,     # Override
    }
    
    merged = registry.merge_with_defaults('vasp_slab_generation', user_params)
    
    # Check overrides applied
    if merged['vacuum'] == 20.0:
        print_success(f"User override 'vacuum' = {merged['vacuum']}")
    else:
        print_fail(f"User override 'vacuum' not applied: {merged['vacuum']}")
        return False
    
    if merged['layers'] == 6:
        print_success(f"User override 'layers' = {merged['layers']}")
    else:
        print_fail(f"User override 'layers' not applied: {merged['layers']}")
        return False
    
    # Check defaults still filled in
    if merged.get('fix_layers') == 2:
        print_success(f"Default 'fix_layers' = {merged['fix_layers']}")
    else:
        print_fail(f"Default 'fix_layers' not filled in")
        return False
    
    return True


def test_skill_registry_edge_specific_defaults():
    """Test edge-specific defaults for XAS calculations."""
    print_header("Skill Registry - Edge-Specific Defaults")
    
    from skills.skill_registry import SkillRegistry
    
    skills_dir = Path(__file__).parent.parent / "skills"
    registry = SkillRegistry(skills_dir)
    
    # K-edge should NOT have spinorbit by default
    k_params = registry.merge_with_defaults(
        'fdmnes_xas_calculation',
        {'edge': 'K'},
        context={'edge': 'K'}
    )
    
    if k_params.get('spinorbit') == False:
        print_success("K-edge: spinorbit = False (correct)")
    else:
        print_fail(f"K-edge: spinorbit = {k_params.get('spinorbit')}, expected False")
        return False
    
    # L3-edge SHOULD have spinorbit by default
    l3_params = registry.merge_with_defaults(
        'fdmnes_xas_calculation',
        {'edge': 'L3'},
        context={'edge': 'L3'}
    )
    
    if l3_params.get('spinorbit') == True:
        print_success("L3-edge: spinorbit = True (correct)")
    else:
        print_fail(f"L3-edge: spinorbit = {l3_params.get('spinorbit')}, expected True")
        return False
    
    return True


# =============================================================================
# Slab Generator Tests
# =============================================================================

def test_slab_generator_basic():
    """Test basic slab generation without adsorbate."""
    print_header("Slab Generator - Basic Cu(111)")
    
    from generators.slab_generator import SlabGenerator
    
    generator = SlabGenerator()
    
    params = {
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 4,
        "vacuum": 15.0,
        "fix_layers": 2,
        "supercell": [2, 2],
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = generator.run(params, output_dir=tmpdir)
        
        atoms = outputs['atoms']
        num_atoms = len(atoms)
        symbols = set(atoms.get_chemical_symbols())
        
        print(f"  Generated {num_atoms} atoms")
        print(f"  Elements: {symbols}")
        
        if num_atoms > 0:
            print_success(f"Generated {num_atoms} Cu atoms")
        else:
            print_fail("No atoms generated")
            return False
        
        if symbols == {'Cu'}:
            print_success("Only Cu atoms present")
        else:
            print_fail(f"Unexpected elements: {symbols}")
            return False
        
        # Check POSCAR written
        poscar_path = Path(tmpdir) / "POSCAR"
        if poscar_path.exists():
            print_success("POSCAR file written")
        else:
            print_fail("POSCAR file not written")
            return False
    
    return True


def test_slab_generator_with_adsorbate():
    """Test slab generation with CO2 adsorbate."""
    print_header("Slab Generator - Cu(111) + CO2")
    
    from generators.slab_generator import SlabGenerator
    
    generator = SlabGenerator()
    
    params = {
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 4,
        "vacuum": 15.0,
        "supercell": [2, 2],
        "adsorbate": {
            "molecule": "CO2",
            "binding_site": "bridge",
            "distance": 2.0,
        }
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = generator.run(params, output_dir=tmpdir)
        
        atoms = outputs['atoms']
        symbols = atoms.get_chemical_symbols()
        
        c_count = symbols.count('C')
        o_count = symbols.count('O')
        
        print(f"  Total atoms: {len(atoms)}")
        print(f"  C atoms: {c_count}")
        print(f"  O atoms: {o_count}")
        
        if c_count == 1:
            print_success("1 C atom from CO2")
        else:
            print_fail(f"Expected 1 C atom, got {c_count}")
            return False
        
        if o_count == 2:
            print_success("2 O atoms from CO2")
        else:
            print_fail(f"Expected 2 O atoms, got {o_count}")
            return False
    
    return True



def test_slab_generator_interface_cu_au_111():
    """Test Cu/Au(111) lateral interface slab generation."""
    print_header("Slab Generator - Cu/Au(111) Interface")

    from generators.slab_generator import SlabGenerator

    generator = SlabGenerator()
    outputs = generator.run({
        "element": "Cu(111)/Au(111)",
        "miller_index": [1, 1, 1],
        "layers": 4,
        "supercell": [2, 2],
    })

    atoms = outputs['atoms']
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    cu_count = symbols.count('Cu')
    au_count = symbols.count('Au')
    interface = outputs.get('interface') or {}

    print(f"  Cu atoms: {cu_count}")
    print(f"  Au atoms: {au_count}")
    print(f"  Cell lengths: {outputs['cell_parameters']['lengths']}")
    print(f"  Cell angles: {outputs['cell_parameters']['angles']}")
    print(f"  Interface: {interface}")

    if interface.get('type') != 'lateral':
        print_fail(f"Expected lateral interface metadata, got {interface}")
        return False

    if cu_count > 0 and au_count > 0:
        print_success("Cu/Au interface contains both metals")
    else:
        print_fail(f"Expected both Cu and Au, got Cu={cu_count}, Au={au_count}")
        return False

    split_coordinate = interface['split_coordinate']
    if all(positions[index][0] < split_coordinate for index, symbol in enumerate(symbols) if symbol == 'Cu'):
        print_success("Cu atoms occupy the lower-x side of the lateral interface")
    else:
        print_fail("Some Cu atoms are not on the lower-x side of the lateral interface")
        return False

    if all(positions[index][0] >= split_coordinate for index, symbol in enumerate(symbols) if symbol == 'Au'):
        print_success("Au atoms occupy the upper-x side of the lateral interface")
    else:
        print_fail("Some Au atoms are not on the upper-x side of the lateral interface")
        return False

    return True


def test_slab_generator_co2rr_adsorbates():
    """Test all requested CO2RR adsorbates are created with correct composition."""
    print_header("Slab Generator - CO2RR Adsorbates")

    from generators.slab_generator import SlabGenerator

    generator = SlabGenerator()
    expected_counts = {
        "CO": {"C": 1, "O": 1},
        "CH": {"C": 1, "H": 1},
        "CH2": {"C": 1, "H": 2},
        "CHO": {"C": 1, "H": 1, "O": 1},
        "CHOH": {"C": 1, "H": 2, "O": 1},
        "CH3": {"C": 1, "H": 3},
        "CH4": {"C": 1, "H": 4},
        "COCO": {"C": 2, "O": 2},
        "OCCO": {"C": 2, "O": 2},
    }

    for molecule, counts in expected_counts.items():
        outputs = generator.run({
            "element": "Cu/Au",
            "miller_index": [1, 1, 1],
            "layers": 4,
            "supercell": [2, 2],
            "adsorbate": {"molecule": molecule},
        })
        symbols = outputs['atoms'].get_chemical_symbols()
        for element, expected in counts.items():
            actual = symbols.count(element)
            if actual != expected:
                print_fail(f"{molecule}: expected {expected} {element}, got {actual}")
                return False
        print_success(f"{molecule}: composition {counts}")

    return True



def test_slab_generator_occo_on_cu_au_interface():
    """Test OCCO is placed across the Cu/Au lateral interface."""
    print_header("Slab Generator - OCCO Across Cu/Au Interface")

    from generators.slab_generator import SlabGenerator

    generator = SlabGenerator()
    outputs = generator.run({
        "element": "Cu/Au",
        "miller_index": [1, 1, 1],
        "layers": 4,
        "supercell": [2, 2],
        "adsorbate": {"molecule": "OCCO"},
    })

    atoms = outputs['atoms']
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    interface = outputs.get('interface') or {}
    placement = outputs.get('adsorbate_placement') or {}

    if placement.get('mode') == 'bidentate_across_lateral_interface':
        print_success("OCCO uses bidentate interface placement")
    else:
        print_fail(f"Unexpected OCCO placement metadata: {placement}")
        return False

    carbon_indices = placement.get('carbon_indices', [])
    if len(carbon_indices) != 2:
        print_fail(f"Expected two interface carbon indices, got {carbon_indices}")
        return False

    split_coordinate = interface['split_coordinate']
    carbon_x = [positions[index][0] for index in carbon_indices]
    if min(carbon_x) < split_coordinate < max(carbon_x):
        print_success("OCCO carbon atoms straddle the Cu/Au interface")
    else:
        print_fail(f"OCCO carbon x positions do not straddle interface: {carbon_x}")
        return False

    if symbols.count('C') == 2 and symbols.count('O') == 2:
        print_success("OCCO composition is correct")
    else:
        print_fail(f"Unexpected OCCO composition: C={symbols.count('C')}, O={symbols.count('O')}")
        return False

    return True

def test_slab_generator_different_surfaces():
    """Test different surface orientations."""
    print_header("Slab Generator - Different Surfaces")
    
    from generators.slab_generator import SlabGenerator
    
    generator = SlabGenerator()
    
    surfaces = [
        ("Cu", [1, 1, 1]),
        ("Cu", [1, 0, 0]),
        ("Au", [1, 1, 1]),
        ("Pt", [1, 1, 1]),
    ]
    
    for element, miller in surfaces:
        params = {
            "element": element,
            "miller_index": miller,
            "layers": 3,
            "supercell": [2, 2],
        }
        
        try:
            outputs = generator.run(params)
            num_atoms = len(outputs['atoms'])
            print_success(f"{element}({miller[0]}{miller[1]}{miller[2]}): {num_atoms} atoms")
        except Exception as e:
            print_fail(f"{element}({miller[0]}{miller[1]}{miller[2]}): {e}")
            return False
    
    return True


# =============================================================================
# VASP XAS Generator Tests
# =============================================================================

def test_vasp_xas_generator():
    """Test VASP XAS input generation."""
    print_header("VASP XAS Generator - Cu K-edge")
    
    from generators.slab_generator import SlabGenerator
    from generators.vasp_input_generator import VASPXASGenerator
    
    # First generate a structure
    slab_gen = SlabGenerator()
    slab_outputs = slab_gen.run({
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 4,
    })
    
    # Generate VASP XAS inputs
    vasp_gen = VASPXASGenerator()
    
    params = {
        "structure": slab_outputs["structure"],
        "absorbing_atom": {"element": "Cu"},
        "edge": "K",
        "spectroscopy_type": "XANES",
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = vasp_gen.run(
            params, 
            output_dir=tmpdir,
            context={"edge": "K", "spectroscopy_type": "XANES"}
        )
        
        # Check INCAR
        incar = outputs['incar']
        if 'ENCUT' in incar:
            print_success("INCAR contains ENCUT")
        else:
            print_fail("INCAR missing ENCUT")
            return False
        
        # Check files written
        for filename in ['INCAR', 'KPOINTS', 'POSCAR']:
            if (Path(tmpdir) / filename).exists():
                print_success(f"{filename} written")
            else:
                print_fail(f"{filename} not written")
                return False
    
    return True


# =============================================================================
# FEFF Generator Tests
# =============================================================================

def test_feff_generator():
    """Test FEFF input generation."""
    print_header("FEFF Generator - Cu K-edge")
    
    from generators.slab_generator import SlabGenerator
    from generators.feff_input_generator import FEFFXASGenerator
    
    # Generate structure
    slab_gen = SlabGenerator()
    slab_outputs = slab_gen.run({
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 3,
        "supercell": [2, 2],
    })
    
    # Generate FEFF input
    feff_gen = FEFFXASGenerator()
    
    params = {
        "structure": slab_outputs["structure"],
        "absorbing_atom": {"element": "Cu", "index": 0},
        "edge": "K",
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = feff_gen.run(
            params,
            output_dir=tmpdir,
            context={"edge": "K"}
        )
        
        feff_inp = outputs['feff_inp']
        
        # Check required cards
        required_cards = ['EDGE', 'POTENTIALS', 'ATOMS', 'END']
        for card in required_cards:
            if card in feff_inp:
                print_success(f"feff.inp contains {card}")
            else:
                print_fail(f"feff.inp missing {card}")
                return False
        
        # Check file written
        if (Path(tmpdir) / 'feff.inp').exists():
            print_success("feff.inp written")
        else:
            print_fail("feff.inp not written")
            return False
    
    return True


# =============================================================================
# FDMNES Generator Tests
# =============================================================================

def test_fdmnes_generator_k_edge():
    """Test FDMNES input generation for K-edge."""
    print_header("FDMNES Generator - Cu K-edge")
    
    from generators.slab_generator import SlabGenerator
    from generators.fdmnes_input_generator import FDMNESXASGenerator
    
    # Generate structure
    slab_gen = SlabGenerator()
    slab_outputs = slab_gen.run({
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 3,
    })
    
    # Generate FDMNES input
    fdmnes_gen = FDMNESXASGenerator()
    
    params = {
        "structure": slab_outputs["structure"],
        "absorbing_atom": {"element": "Cu"},
        "edge": "K",
        "radius": 6.0,
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        outputs = fdmnes_gen.run(
            params,
            output_dir=tmpdir,
            context={"edge": "K"}
        )
        
        fdmnes_inp = outputs['fdmnes_inp']
        
        # Check required keywords
        required_keywords = ['Filout', 'Range', 'Radius', 'Edge', 'Absorber']
        for keyword in required_keywords:
            if keyword in fdmnes_inp:
                print_success(f"Contains {keyword}")
            else:
                print_fail(f"Missing {keyword}")
                return False
        
        # K-edge should NOT have Spinorbit
        if 'Spinorbit' not in fdmnes_inp:
            print_success("K-edge: No Spinorbit (correct)")
        else:
            print_fail("K-edge should not have Spinorbit")
            return False
        
        # Check file written
        if (Path(tmpdir) / 'fdmfile.txt').exists():
            print_success("fdmfile.txt written")
        else:
            print_fail("fdmfile.txt not written")
            return False
    
    return True


def test_fdmnes_generator_l3_edge():
    """Test FDMNES input generation for L3-edge (should have spinorbit)."""
    print_header("FDMNES Generator - Cu L3-edge")
    
    from generators.slab_generator import SlabGenerator
    from generators.fdmnes_input_generator import FDMNESXASGenerator
    
    slab_gen = SlabGenerator()
    slab_outputs = slab_gen.run({
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 3,
    })
    
    fdmnes_gen = FDMNESXASGenerator()
    
    params = {
        "structure": slab_outputs["structure"],
        "absorbing_atom": {"element": "Cu"},
        "edge": "L3",
    }
    
    outputs = fdmnes_gen.run(params, context={"edge": "L3"})
    fdmnes_inp = outputs['fdmnes_inp']
    
    # L3-edge SHOULD have Spinorbit
    if 'Spinorbit' in fdmnes_inp:
        print_success("L3-edge: Spinorbit enabled (correct)")
    else:
        print_fail("L3-edge should have Spinorbit")
        return False
    
    # L3-edge SHOULD have Quadrupole
    if 'Quadrupole' in fdmnes_inp:
        print_success("L3-edge: Quadrupole enabled (correct)")
    else:
        print_fail("L3-edge should have Quadrupole")
        return False
    
    return True


def test_fdmnes_generator_user_override():
    """Test that user parameters override FDMNES defaults."""
    print_header("FDMNES Generator - User Override")
    
    from generators.slab_generator import SlabGenerator
    from generators.fdmnes_input_generator import FDMNESXASGenerator
    
    slab_gen = SlabGenerator()
    slab_outputs = slab_gen.run({
        "element": "Cu",
        "miller_index": [1, 1, 1],
        "layers": 3,
    })
    
    fdmnes_gen = FDMNESXASGenerator()
    
    # User specifies custom radius and energy range
    params = {
        "structure": slab_outputs["structure"],
        "absorbing_atom": {"element": "Cu"},
        "edge": "K",
        "radius": 8.0,  # Override default 6.0
        "range": {
            "emin": -20.0,  # Override default -10.0
            "emax": 100.0,  # Override default 50.0
        }
    }
    
    outputs = fdmnes_gen.run(params, context={"edge": "K"})
    fdmnes_inp = outputs['fdmnes_inp']
    
    # Check radius override
    if '8.00' in fdmnes_inp:
        print_success("Custom radius 8.0 applied")
    else:
        print_fail("Custom radius not applied")
        return False
    
    # Check energy range override
    if '-20.0' in fdmnes_inp:
        print_success("Custom emin -20.0 applied")
    else:
        print_fail("Custom emin not applied")
        return False
    
    if '100.0' in fdmnes_inp:
        print_success("Custom emax 100.0 applied")
    else:
        print_fail("Custom emax not applied")
        return False
    
    return True


# =============================================================================
# Job Script Generator Tests
# =============================================================================

def test_job_script_generator_pbs():
    """Test PBS job script generation."""
    print_header("Job Script Generator - PBS (Polaris)")
    
    from workflow.job_script_generator import JobScriptGenerator
    
    generator = JobScriptGenerator(system='polaris')
    
    script = generator.generate_script(
        software='vasp',
        job_name='cu_k_xas',
        work_dir='/path/to/calculation',
        nodes=2,
        walltime='02:00:00',
        project='my_project',
        sample_name='Cu_111_slab',
        sample_formula='Cu',
        absorber='Cu',
        edge='K',
        include_isaac_metadata=True,
    )
    
    # Check PBS directives
    if '#PBS -N cu_k_xas' in script:
        print_success("PBS job name directive")
    else:
        print_fail("Missing PBS job name")
        return False
    
    if '#PBS -l select=2' in script:
        print_success("PBS node count directive")
    else:
        print_fail("Missing PBS node count")
        return False
    
    # Check ISAAC metadata
    if 'ISAAC_SOFTWARE' in script:
        print_success("ISAAC_SOFTWARE variable")
    else:
        print_fail("Missing ISAAC_SOFTWARE")
        return False
    
    if 'isaac_run_metadata.json' in script:
        print_success("ISAAC metadata finalization")
    else:
        print_fail("Missing ISAAC metadata finalization")
        return False
    
    return True


def test_job_script_generator_slurm():
    """Test SLURM job script generation."""
    print_header("Job Script Generator - SLURM (Perlmutter)")
    
    from workflow.job_script_generator import JobScriptGenerator
    
    generator = JobScriptGenerator(system='perlmutter')
    
    script = generator.generate_script(
        software='feff',
        job_name='cu_feff',
        work_dir='/path/to/calculation',
        nodes=1,
        walltime='00:30:00',
        project='my_project',
        include_isaac_metadata=True,
    )
    
    # Check SLURM directives
    if '#SBATCH --job-name=cu_feff' in script:
        print_success("SLURM job name directive")
    else:
        print_fail("Missing SLURM job name")
        return False
    
    if '#SBATCH --nodes=1' in script:
        print_success("SLURM node count directive")
    else:
        print_fail("Missing SLURM node count")
        return False
    
    return True


# =============================================================================
# Result Parser Tests
# =============================================================================

def test_result_parser_isaac_record_structure():
    """Test that ISAAC record has correct structure."""
    print_header("Result Parser - ISAAC Record Structure")
    
    from tools.result_parser import ISAACRecordGenerator
    from tools.utils import get_timestamp
    
    generator = ISAACRecordGenerator()
    
    # Mock data
    structure = {
        'atoms': ['Cu'] * 16,
        'positions': [[0, 0, 0]] * 16,
        'cell': [[10, 0, 0], [0, 10, 0], [0, 0, 20]],
    }
    
    spectrum = {
        'energy': [8970, 8975, 8980, 8985, 8990],
        'intensity': [0.0, 0.02, 0.38, 1.35, 1.05],
        'energy_unit': 'eV',
        'intensity_unit': 'arb. units',
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a mock structure file
        poscar_path = Path(tmpdir) / "POSCAR"
        poscar_path.write_text("Cu slab\n1.0\n10 0 0\n0 10 0\n0 0 20\nCu\n16\nCartesian\n" + "0 0 0\n" * 16)
        
        record = generator.create_record(
            structure=structure,
            software='VASP',
            spectrum=spectrum,
            parameters={},
            absorber='Cu',
            edge='K',
            output_dir=tmpdir,
            structure_file=str(poscar_path),
        )
    
    # Check required top-level fields
    required_fields = [
        'isaac_record_version',
        'record_id',
        'record_type',
        'record_domain',
        'source_type',
        'timestamps',
        'sample',
        'context',
        'system',
        'computation',
        'measurement',
        'assets',
        'links',
        'descriptors',
    ]
    
    for field in required_fields:
        if field in record:
            print_success(f"Field '{field}' present")
        else:
            print_fail(f"Field '{field}' missing")
            return False
    
    # Check specific values
    if record['isaac_record_version'] == '1.05':
        print_success("Version 1.05")
    else:
        print_fail(f"Wrong version: {record['isaac_record_version']}")
        return False
    
    if record['record_type'] == 'evidence':
        print_success("record_type = evidence")
    else:
        print_fail(f"Wrong record_type: {record['record_type']}")
        return False
    
    if record['record_domain'] == 'simulation':
        print_success("record_domain = simulation")
    else:
        print_fail(f"Wrong record_domain: {record['record_domain']}")
        return False
    
    return True


def test_result_parser_measurement_series():
    """Test that measurement.series has correct structure."""
    print_header("Result Parser - Measurement Series")
    
    from tools.result_parser import ISAACRecordGenerator
    
    generator = ISAACRecordGenerator()
    
    structure = {'atoms': ['Cu'] * 4, 'positions': [], 'cell': []}
    spectrum = {
        'energy': [8970, 8975, 8980, 8985, 8990, 8995, 9000],
        'intensity': [0.0, 0.02, 0.38, 1.35, 1.05, 0.99, 1.01],
        'energy_unit': 'eV',
        'intensity_unit': 'arb. units',
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        poscar_path = Path(tmpdir) / "POSCAR"
        poscar_path.write_text("test\n")
        
        record = generator.create_record(
            structure=structure,
            software='VASP',
            spectrum=spectrum,
            parameters={},
            absorber='Cu',
            edge='K',
            output_dir=tmpdir,
            structure_file=str(poscar_path),
        )
    
    series = record['measurement']['series'][0]
    
    # Check series_id
    if series['series_id'] == 'computed_k_edge':
        print_success("series_id = computed_k_edge")
    else:
        print_fail(f"Wrong series_id: {series['series_id']}")
        return False
    
    # Check independent_variables
    indep = series['independent_variables'][0]
    if indep['name'] == 'energy' and indep['unit'] == 'eV':
        print_success("independent_variables: energy in eV")
    else:
        print_fail(f"Wrong independent_variables: {indep}")
        return False
    
    if indep['values'] == spectrum['energy']:
        print_success("Energy values match")
    else:
        print_fail("Energy values don't match")
        return False
    
    # Check channels
    channel = series['channels'][0]
    if channel['name'] == 'absorption_cross_section':
        print_success("channel name = absorption_cross_section")
    else:
        print_fail(f"Wrong channel name: {channel['name']}")
        return False
    
    if channel['role'] == 'simulated_observable':
        print_success("channel role = simulated_observable")
    else:
        print_fail(f"Wrong channel role: {channel['role']}")
        return False
    
    return True


def test_result_parser_descriptors():
    """Test that descriptors are computed correctly."""
    print_header("Result Parser - Descriptors")
    
    from tools.result_parser import ISAACRecordGenerator
    
    generator = ISAACRecordGenerator()
    
    # Spectrum with clear edge position
    spectrum = {
        'energy': [8970, 8975, 8980, 8985, 8990, 8995, 9000],
        'intensity': [0.0, 0.1, 0.5, 1.2, 1.0, 0.9, 0.85],
        'energy_unit': 'eV',
        'intensity_unit': 'arb. units',
    }
    
    descriptors = generator._compute_descriptors(spectrum, 'Cu', 'K')
    
    outputs = descriptors.get('outputs', [])
    if len(outputs) > 0:
        print_success("Descriptors outputs present")
    else:
        print_fail("No descriptors outputs")
        return False
    
    desc_list = outputs[0].get('descriptors', [])
    
    # Check edge position descriptor
    edge_desc = next((d for d in desc_list if d['name'] == 'xanes.edge_position'), None)
    if edge_desc:
        print_success(f"Edge position: {edge_desc['value']} eV")
        
        # Check uncertainty is numeric
        sigma = edge_desc.get('uncertainty', {}).get('sigma')
        if isinstance(sigma, (int, float)):
            print_success(f"Edge uncertainty sigma: {sigma} (numeric)")
        else:
            print_fail(f"Edge uncertainty sigma not numeric: {sigma}")
            return False
    else:
        print_fail("Edge position descriptor missing")
        return False
    
    # Check white line intensity descriptor
    wl_desc = next((d for d in desc_list if d['name'] == 'xanes.white_line_intensity'), None)
    if wl_desc:
        print_success(f"White line intensity: {wl_desc['value']}")
    else:
        print_fail("White line intensity descriptor missing")
        return False
    
    return True


# =============================================================================
# Parameter Extractor Tests
# =============================================================================

def test_parameter_extractor_slab():
    """Test parameter extraction for slab requests."""
    print_header("Parameter Extractor - Slab Parameters")
    
    from agent.parameter_extractor import ParameterExtractor
    
    extractor = ParameterExtractor()
    
    test_cases = [
        {
            "input": "Cu(111) with 6 layers",
            "expected_element": "Cu",
            "expected_miller": [1, 1, 1],
            "expected_layers": 6,
        },
        {
            "input": "Au(100) surface with 20 Å vacuum",
            "expected_element": "Au",
            "expected_miller": [1, 0, 0],
            "expected_vacuum": 20.0,
        },
        {
            "input": "Pt(111) with CO2 on bridge site",
            "expected_element": "Pt",
            "expected_miller": [1, 1, 1],
            "expected_adsorbate": "CO2",
            "expected_site": "bridge",
        },
    ]
    
    for case in test_cases:
        result = extractor.extract_slab_parameters(case["input"])
        
        if result.get('element') == case.get('expected_element'):
            print_success(f"Element: {result.get('element')}")
        else:
            print_fail(f"Element: got {result.get('element')}, expected {case.get('expected_element')}")
            return False
        
        if 'expected_miller' in case:
            if result.get('miller_index') == case['expected_miller']:
                print_success(f"Miller index: {result.get('miller_index')}")
            else:
                print_fail(f"Miller index: got {result.get('miller_index')}")
                return False
    
    return True


def test_parameter_extractor_xas():
    """Test parameter extraction for XAS requests."""
    print_header("Parameter Extractor - XAS Parameters")
    
    from agent.parameter_extractor import ParameterExtractor
    
    extractor = ParameterExtractor()
    
    test_cases = [
        {
            "input": "Cu K-edge XANES",
            "expected_edge": "K",
            "expected_element": "Cu",
            "expected_type": "XANES",
        },
        {
            "input": "Fe L3-edge EXAFS calculation",
            "expected_edge": "L3",
            "expected_element": "Fe",
            "expected_type": "EXAFS",
        },
    ]
    
    for case in test_cases:
        result = extractor.extract_xas_parameters(case["input"])
        
        if result.get('edge') == case.get('expected_edge'):
            print_success(f"Edge: {result.get('edge')}")
        else:
            print_fail(f"Edge: got {result.get('edge')}, expected {case.get('expected_edge')}")
            return False
        
        if result.get('absorbing_atom', {}).get('element') == case.get('expected_element'):
            print_success(f"Element: {result.get('absorbing_atom', {}).get('element')}")
        else:
            print_fail(f"Element extraction failed")
            return False
    
    return True


# =============================================================================
# Integration Tests
# =============================================================================

def test_full_workflow_vasp():
    """Test full workflow: slab → VASP XAS → job script."""
    print_header("Integration - Full VASP Workflow")
    
    from generators.slab_generator import SlabGenerator
    from generators.vasp_input_generator import VASPXASGenerator
    from workflow.job_script_generator import JobScriptGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Step 1: Generate slab
        slab_gen = SlabGenerator()
        slab_outputs = slab_gen.run(
            {"element": "Cu", "miller_index": [1, 1, 1], "layers": 4},
            output_dir=tmpdir
        )
        print_success("Step 1: Slab generated")
        
        # Step 2: Generate VASP XAS inputs
        vasp_gen = VASPXASGenerator()
        vasp_outputs = vasp_gen.run(
            {
                "structure": slab_outputs["structure"],
                "absorbing_atom": {"element": "Cu"},
                "edge": "K",
            },
            output_dir=tmpdir,
            context={"edge": "K"}
        )
        print_success("Step 2: VASP inputs generated")
        
        # Step 3: Generate job script
        job_gen = JobScriptGenerator(system='polaris')
        script = job_gen.generate_script(
            software='vasp',
            job_name='cu_k_xas',
            work_dir=str(tmpdir),
            project='test_project',
            absorber='Cu',
            edge='K',
        )
        
        script_path = tmpdir / 'submit.sh'
        job_gen.write_script(script, script_path)
        print_success("Step 3: Job script generated")
        
        # Verify all files exist
        expected_files = ['POSCAR', 'INCAR', 'KPOINTS', 'submit.sh']
        for filename in expected_files:
            if (tmpdir / filename).exists():
                print_success(f"File exists: {filename}")
            else:
                print_fail(f"File missing: {filename}")
                return False
    
    return True


def test_full_workflow_fdmnes():
    """Test full workflow: slab → FDMNES → job script."""
    print_header("Integration - Full FDMNES Workflow")
    
    from generators.slab_generator import SlabGenerator
    from generators.fdmnes_input_generator import FDMNESXASGenerator
    from workflow.job_script_generator import JobScriptGenerator
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Step 1: Generate slab
        slab_gen = SlabGenerator()
        slab_outputs = slab_gen.run(
            {"element": "Au", "miller_index": [1, 1, 1], "layers": 3},
            output_dir=tmpdir
        )
        print_success("Step 1: Au(111) slab generated")
        
        # Step 2: Generate FDMNES inputs
        fdmnes_gen = FDMNESXASGenerator()
        fdmnes_outputs = fdmnes_gen.run(
            {
                "structure": slab_outputs["structure"],
                "absorbing_atom": {"element": "Au"},
                "edge": "L3",
            },
            output_dir=tmpdir,
            context={"edge": "L3"}
        )
        print_success("Step 2: FDMNES inputs generated")
        
        # Verify L3-edge specific settings
        if 'Spinorbit' in fdmnes_outputs['fdmnes_inp']:
            print_success("L3-edge: Spinorbit enabled")
        else:
            print_fail("L3-edge: Spinorbit missing")
            return False
        
        # Step 3: Generate job script
        job_gen = JobScriptGenerator(system='perlmutter')
        script = job_gen.generate_script(
            software='fdmnes',
            job_name='au_l3_xas',
            work_dir=str(tmpdir),
            project='test_project',
            absorber='Au',
            edge='L3',
        )
        print_success("Step 3: Job script generated")
    
    return True


# =============================================================================
# Main test runner
# =============================================================================

def run_all_tests():
    """Run all local tests."""
    print("\n" + "=" * 70)
    print("CO2RR XAS AGENT - LOCAL TESTS")
    print("=" * 70)
    
    tests = [
        # Skill Registry
        ("Skill Registry - Loading", test_skill_registry_loading),
        ("Skill Registry - Defaults", test_skill_registry_defaults),
        ("Skill Registry - Override", test_skill_registry_merge_with_override),
        ("Skill Registry - Edge Defaults", test_skill_registry_edge_specific_defaults),
        
        # Generators
        ("Slab Generator - Basic", test_slab_generator_basic),
        ("Slab Generator - Adsorbate", test_slab_generator_with_adsorbate),
        ("Slab Generator - Interface", test_slab_generator_interface_cu_au_111),
        ("Slab Generator - CO2RR Adsorbates", test_slab_generator_co2rr_adsorbates),
        ("Slab Generator - OCCO Interface", test_slab_generator_occo_on_cu_au_interface),
        ("Slab Generator - Surfaces", test_slab_generator_different_surfaces),
        ("VASP XAS Generator", test_vasp_xas_generator),
        ("FEFF Generator", test_feff_generator),
        ("FDMNES Generator - K-edge", test_fdmnes_generator_k_edge),
        ("FDMNES Generator - L3-edge", test_fdmnes_generator_l3_edge),
        ("FDMNES Generator - Override", test_fdmnes_generator_user_override),
        
        # Job Scripts
        ("Job Script - PBS", test_job_script_generator_pbs),
        ("Job Script - SLURM", test_job_script_generator_slurm),
        
        # Result Parser
        ("Result Parser - ISAAC Structure", test_result_parser_isaac_record_structure),
        ("Result Parser - Measurement Series", test_result_parser_measurement_series),
        ("Result Parser - Descriptors", test_result_parser_descriptors),
        
        # Parameter Extractor
        ("Parameter Extractor - Slab", test_parameter_extractor_slab),
        ("Parameter Extractor - XAS", test_parameter_extractor_xas),
        
        # Integration
        ("Integration - VASP Workflow", test_full_workflow_vasp),
        ("Integration - FDMNES Workflow", test_full_workflow_fdmnes),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, "PASSED" if success else "FAILED"))
        except Exception as e:
            print(f"\n  ✗ Exception: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, f"ERROR"))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for name, result in results:
        if result == "PASSED":
            print(f"  ✓ {name}")
            passed += 1
        else:
            print(f"  ✗ {name}: {result}")
            failed += 1
    
    print(f"\n{passed}/{len(results)} tests passed")
    
    if failed > 0:
        print(f"\n⚠️  {failed} tests failed!")
        return False
    else:
        print("\n✅ All tests passed!")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)