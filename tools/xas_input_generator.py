"""
XAS Input Generator Tool for CO2RR XAS Agent
Skill 2 Implementation
"""

import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from .utils import (
    ATOMIC_NUMBERS, METALS_3D, METALS_4D, METALS_5D,
    EDGE_ENERGIES, METAL_DATA,
    get_edge_for_element, get_edge_energy, contains_carbon, get_nersc_defaults,
    read_poscar, read_structure_file, write_poscar, ensure_dir
)

def get_dipol_direct(structure: Dict) -> str:
    """
    Return DIPOL in direct lattice coordinates.

    VASP DIPOL expects direct coordinates. For asymmetric slabs, use the
    geometric center of the atoms, not the cell center, because vacuum is
    one-sided.
    """
    positions = np.array(structure["positions"])
    cell = np.array(structure["cell"])

    center_cart = positions.mean(axis=0)
    center_direct = np.dot(center_cart, np.linalg.inv(cell))
    center_direct = center_direct % 1.0

    return f"DIPOL = {center_direct[0]:.6f} {center_direct[1]:.6f} {center_direct[2]:.6f}"


def _isaac_submit_metadata_setup(
    software: str,
    code_version: str,
    technique: str,
    instrument_name: str,
    vendor_or_project: str,
    sample_name: str = "not_specified",
    sample_form: str = "slab_model",
    sample_notes: str = "not_specified",
    recipe_target: str = "not_specified",
) -> List[str]:
    """Shell lines placed near the top of submit.sh to start ISAAC metadata capture."""
    return [
        "",
        "# =============================================================================",
        "# ISAAC run metadata: initialized before the calculation starts",
        "# =============================================================================",
        f"export ISAAC_SOFTWARE=\"${{ISAAC_SOFTWARE:-{software}}}\"",
        f"export ISAAC_CODE_VERSION=\"${{ISAAC_CODE_VERSION:-{code_version}}}\"",
        f"export ISAAC_TECHNIQUE=\"${{ISAAC_TECHNIQUE:-{technique}}}\"",
        f"export ISAAC_INSTRUMENT_NAME=\"${{ISAAC_INSTRUMENT_NAME:-{instrument_name}}}\"",
        f"export ISAAC_VENDOR_OR_PROJECT=\"${{ISAAC_VENDOR_OR_PROJECT:-{vendor_or_project}}}\"",
        "export ISAAC_FACILITY_NAME=\"${ISAAC_FACILITY_NAME:-NERSC}\"",
        "export ISAAC_ORGANIZATION=\"${ISAAC_ORGANIZATION:-LBNL}\"",
        "export ISAAC_CLUSTER=\"${ISAAC_CLUSTER:-Perlmutter}\"",
        "export ISAAC_COMPUTE_ARCHITECTURE=\"${ISAAC_COMPUTE_ARCHITECTURE:-CPU}\"",
        f"export ISAAC_SAMPLE_NAME=\"${{ISAAC_SAMPLE_NAME:-{sample_name}}}\"",
        f"export ISAAC_SAMPLE_FORM=\"${{ISAAC_SAMPLE_FORM:-{sample_form}}}\"",
        f"export ISAAC_SAMPLE_NOTES=\"${{ISAAC_SAMPLE_NOTES:-{sample_notes}}}\"",
        "export ISAAC_SAMPLE_FORMULA=\"${ISAAC_SAMPLE_FORMULA:-not_specified}\"",
        "export ISAAC_MATERIAL_PROVENANCE=\"${ISAAC_MATERIAL_PROVENANCE:-theoretical}\"",
        "export ISAAC_CONTEXT_ENVIRONMENT=\"${ISAAC_CONTEXT_ENVIRONMENT:-in_silico}\"",
        "export ISAAC_TEMPERATURE_K=\"${ISAAC_TEMPERATURE_K:-0}\"",
        "export ISAAC_CONTEXT_NOTES=\"${ISAAC_CONTEXT_NOTES:-Static 0 K electronic structure unless otherwise specified.}\"",
        "export ISAAC_PROCESSING_TYPE=\"${ISAAC_PROCESSING_TYPE:-simulated_spectrum}\"",
        f"export ISAAC_RECIPE_TARGET=\"${{ISAAC_RECIPE_TARGET:-{recipe_target}}}\"",
        "export ISAAC_METADATA_FILE=\"${ISAAC_METADATA_FILE:-isaac_run_metadata.json}\"",
        "export ISAAC_ASSET_BASE_URI=\"${ISAAC_ASSET_BASE_URI:-}\"",
        "export ISAAC_REPO_ROOT=\"${ISAAC_REPO_ROOT:-$(pwd)}\"",
        "isaac_utc_now() { date -u +\"%Y-%m-%dT%H:%M:%SZ\"; }",
        "export ISAAC_START_UTC=\"$(isaac_utc_now)\"",
        "export ISAAC_RUN_COMMAND=\"not_started\"",
        "export ISAAC_EXIT_CODE=\"not_finished\"",
    ]


def _isaac_submit_metadata_finalize() -> List[str]:
    """Shell lines placed at the end of submit.sh to write isaac_run_metadata.json."""
    return [
        "",
        "# =============================================================================",
        "# ISAAC run metadata: finalized after the calculation ends",
        "# =============================================================================",
        "export ISAAC_END_UTC=\"$(isaac_utc_now)\"",
        "python - <<'PY'",
        "import hashlib",
        "import json",
        "import os",
        "from pathlib import Path",
        "",
        "def sha256_file(path):",
        "    p = Path(path)",
        "    if not p.exists() or not p.is_file():",
        "        return 'not_available'",
        "    h = hashlib.sha256()",
        "    with p.open('rb') as f:",
        "        for chunk in iter(lambda: f.read(1024 * 1024), b''):",
        "            h.update(chunk)",
        "    return h.hexdigest()",
        "",
        "def media_type_for(path):",
        "    name = Path(path).name.lower()",
        "    if name.endswith('.json'):",
        "        return 'application/json'",
        "    if name.endswith(('.tar.gz', '.tgz')):",
        "        return 'application/gzip'",
        "    if name.endswith('.xml'):",
        "        return 'application/xml'",
        "    return 'text/plain'",
        "",
        "def content_role_for(path):",
        "    name = Path(path).name",
        "    lower = name.lower()",
        "    if name in {'POSCAR', 'CONTCAR'} or lower.endswith(('.cif', '.xyz')):",
        "        return 'input_structure'",
        "    if name in {'INCAR', 'KPOINTS', 'POTCAR', 'POTCAR.spec', 'make_potcar.sh', 'submit.sh', 'feff.inp', 'HEADER', 'PARAMETERS', 'POTENTIALS', 'ATOMS', 'fdmfile.txt'} or name.endswith('_in.txt'):",
        "        return 'workflow_recipe'",
        "    if name == 'xmu.dat' or name.endswith('_conv.txt') or name == 'CORE_DIELECTRIC_IMAG.dat':",
        "        return 'reduction_product'",
        "    return 'raw_data_pointer'",
        "",
        "def note_for(path):",
        "    role = content_role_for(path)",
        "    if role == 'input_structure':",
        "        return 'Structure used for the XAS simulation.'",
        "    if role == 'workflow_recipe':",
        "        return 'Input deck or run script used by the XAS simulation workflow.'",
        "    if role == 'reduction_product':",
        "        return 'Parsed or parseable XAS spectrum output.'",
        "    return 'Raw code output or auxiliary file generated by the simulation.'",
        "",
        "def asset_uri(path):",
        "    p = Path(path)",
        "    base = os.environ.get('ISAAC_ASSET_BASE_URI', '').rstrip('/')",
        "    if base:",
        "        root = Path(os.environ.get('ISAAC_REPO_ROOT', '.')).resolve()",
        "        try:",
        "            rel = p.resolve().relative_to(root)",
        "        except ValueError:",
        "            rel = Path(p.name)",
        "        return f'{base}/{rel.as_posix()}'",
        "    return str(p.resolve())",
        "",
        "def make_asset(path, idx):",
        "    p = Path(path)",
        "    safe = p.as_posix().replace('/', '_').replace('.', '_')[:80]",
        "    return {",
        "        'asset_id': f'asset_{idx:03d}_{safe}',",
        "        'content_role': content_role_for(p),",
        "        'uri': asset_uri(p),",
        "        'media_type': media_type_for(p),",
        "        'sha256': sha256_file(p),",
        "        'notes': note_for(p),",
        "    }",
        "",
        "include_names = {",
        "    'POSCAR', 'CONTCAR', 'INCAR', 'KPOINTS', 'POTCAR', 'POTCAR.spec', 'make_potcar.sh', 'submit.sh',",
        "    'feff.inp', 'HEADER', 'PARAMETERS', 'POTENTIALS', 'ATOMS', 'fdmfile.txt', 'xmu.dat', 'chi.dat', 'paths.dat', 'files.dat', 'log1.dat',",
        "    'OUTCAR', 'vasprun.xml', 'OSZICAR', 'CONTCAR', 'WAVECAR', 'CHGCAR', 'vasp.out',",
        "    'CORE_DIELECTRIC_IMAG.dat'",
        "}",
        "paths = []",
        "for p in Path('.').rglob('*'):",
        "    if not p.is_file():",
        "        continue",
        "    if p.name == os.environ.get('ISAAC_METADATA_FILE', 'isaac_run_metadata.json'):",
        "        continue",
        "    if p.name in include_names or p.name.endswith('_in.txt') or p.name.endswith('_conv.txt') or p.name.endswith('_bav.txt'):",
        "        paths.append(p)",
        "paths = sorted(set(paths), key=lambda x: x.as_posix())",
        "assets = [make_asset(p, i) for i, p in enumerate(paths)]",
        "",
        "try:",
        "    temperature = float(os.environ.get('ISAAC_TEMPERATURE_K', '0'))",
        "except ValueError:",
        "    temperature = 0",
        "try:",
        "    exit_code = int(os.environ.get('ISAAC_EXIT_CODE', '-1'))",
        "except ValueError:",
        "    exit_code = -1",
        "",
        "metadata = {",
        "    'timestamps': {",
        "        'acquired_start_utc': os.environ.get('ISAAC_START_UTC', 'not_specified'),",
        "        'acquired_end_utc': os.environ.get('ISAAC_END_UTC', 'not_specified'),",
        "    },",
        "    'sample': {",
        "        'material': {",
        "            'name': os.environ.get('ISAAC_SAMPLE_NAME', 'not_specified'),",
        "            'formula': os.environ.get('ISAAC_SAMPLE_FORMULA', 'not_specified'),",
        "            'provenance': os.environ.get('ISAAC_MATERIAL_PROVENANCE', 'theoretical'),",
        "            'notes': os.environ.get('ISAAC_SAMPLE_NOTES', 'not_specified'),",
        "        },",
        "        'sample_form': os.environ.get('ISAAC_SAMPLE_FORM', 'slab_model'),",
        "    },",
        "    'context': {",
        "        'environment': os.environ.get('ISAAC_CONTEXT_ENVIRONMENT', 'in_silico'),",
        "        'temperature_K': temperature,",
        "        'notes': os.environ.get('ISAAC_CONTEXT_NOTES', 'Static 0 K electronic structure unless otherwise specified.'),",
        "    },",
        "    'system': {",
        "        'domain': 'computational',",
        "        'technique': os.environ.get('ISAAC_TECHNIQUE', 'not_specified'),",
        "        'facility': {",
        "            'facility_name': os.environ.get('ISAAC_FACILITY_NAME', 'NERSC'),",
        "            'organization': os.environ.get('ISAAC_ORGANIZATION', 'LBNL'),",
        "            'cluster': os.environ.get('ISAAC_CLUSTER', 'Perlmutter'),",
        "        },",
        "        'instrument': {",
        "            'instrument_type': 'simulation_engine',",
        "            'instrument_name': os.environ.get('ISAAC_INSTRUMENT_NAME', 'not_specified'),",
        "            'vendor_or_project': os.environ.get('ISAAC_VENDOR_OR_PROJECT', 'not_specified'),",
        "        },",
        "        'configuration': {",
        "            'code_version': os.environ.get('ISAAC_CODE_VERSION', 'not_specified'),",
        "            'compute_architecture': os.environ.get('ISAAC_COMPUTE_ARCHITECTURE', 'CPU'),",
        "        },",
        "    },",
        "    'run': {",
        "        'software': os.environ.get('ISAAC_SOFTWARE', 'not_specified'),",
        "        'scheduler': 'SLURM',",
        "        'command': os.environ.get('ISAAC_RUN_COMMAND', 'not_specified'),",
        "        'exit_code': exit_code,",
        "        'working_directory': str(Path('.').resolve()),",
        "        'hostname': os.environ.get('HOSTNAME', 'not_specified'),",
        "        'slurm_job_id': os.environ.get('SLURM_JOB_ID', 'not_specified'),",
        "        'slurm_job_name': os.environ.get('SLURM_JOB_NAME', 'not_specified'),",
        "        'slurm_partition': os.environ.get('SLURM_JOB_PARTITION', 'not_specified'),",
        "        'slurm_nodelist': os.environ.get('SLURM_NODELIST', 'not_specified'),",
        "        'nersc_host': os.environ.get('NERSC_HOST', 'not_specified'),",
        "        'cluster': os.environ.get('ISAAC_CLUSTER', 'Perlmutter'),",
        "    },",
        "    'measurement': {",
        "        'processing': {",
        "            'type': os.environ.get('ISAAC_PROCESSING_TYPE', 'simulated_spectrum'),",
        "            'recipe_link': {",
        "                'rel': 'processing_recipe',",
        "                'target': os.environ.get('ISAAC_RECIPE_TARGET', 'not_specified'),",
        "            },",
        "        },",
        "        'qc': {",
        "            'status': 'valid' if exit_code == 0 else 'failed',",
        "        },",
        "    },",
        "    'assets': assets,",
        "    'links': [],",
        "    'descriptors': {'outputs': []},",
        "}",
        "Path(os.environ.get('ISAAC_METADATA_FILE', 'isaac_run_metadata.json')).write_text(json.dumps(metadata, indent=2) + '\\n')",
        "print('Wrote', os.environ.get('ISAAC_METADATA_FILE', 'isaac_run_metadata.json'))",
        "PY",
        "",
    ]

class RelaxationInputGenerator:
    """Generate VASP relaxation inputs."""
    
    def __init__(self):
        self.potcar_map = {
            'H': 'H', 'C': 'C', 'N': 'N', 'O': 'O',
            'Fe': 'Fe_pv', 'Co': 'Co_pv', 'Ni': 'Ni_pv',
            'Cu': 'Cu_pv', 'Zn': 'Zn', 'Pd': 'Pd_pv',
            'Ag': 'Ag', 'Pt': 'Pt_pv', 'Au': 'Au',
            'Al': 'Al', 'Rh': 'Rh_pv', 'Ir': 'Ir'
        }
    
    def generate_incar(self, structure: Dict) -> str:
        """Generate INCAR for relaxation."""
        lines = [
            "# VASP INCAR for structure relaxation",
            "# Generated by CO2RR-XAS Agent",
            "",
            "# Electronic settings",
            "ENCUT = 520",
            "EDIFF = 1E-6",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "LREAL = Auto",
            "PREC = Accurate",
            "",
            "# Ionic relaxation",
            "IBRION = 2",
            "NSW = 200",
            "EDIFFG = -0.02",
            "ISIF = 2",
            "",    
            "# Dipole correction for asymmetric one-sided slab",
            "LDIPOL = .TRUE.",
            "IDIPOL = 3",
            get_dipol_direct(structure),
            "AMIN = 0.01",
            "",
            "# Spin",
            "ISPIN = 2",
            "",
            "# Output",
            "LWAVE = .FALSE.",
            "LCHARG = .FALSE.",
            "NCORE = 4",
        ]
        return "\n".join(lines)
    
    def generate_kpoints(self, structure: Dict) -> str:
        """Generate KPOINTS for relaxation."""
        cell = structure['cell']
        lengths = [np.linalg.norm(cell[i]) for i in range(3)]
        kpoints = [max(1, int(30 / l)) for l in lengths]
        # Reduce k-points in z-direction for slab
        kpoints[2] = 1
        
        lines = [
            "Automatic mesh",
            "0",
            "Gamma",
            f"  {kpoints[0]}  {kpoints[1]}  {kpoints[2]}",
            "  0  0  0"
        ]
        return "\n".join(lines)
    
    def generate_potcar_spec(self, structure: Dict) -> str:
        """Generate POTCAR specification file."""
        unique_elements = []
        for atom in structure['atoms']:
            if atom not in unique_elements:
                unique_elements.append(atom)
        
        lines = ["# POTCAR specification", "# Concatenate these POTCARs in order:"]
        for elem in unique_elements:
            potcar = self.potcar_map.get(elem, elem)
            lines.append(f"{potcar}")
        
        return "\n".join(lines)
    
    def write_inputs(self, structure: Dict, output_dir: str) -> List[str]:
        """Write all relaxation input files."""
        ensure_dir(output_dir)
        files = []
        
        # POSCAR
        poscar_path = os.path.join(output_dir, "POSCAR")
        write_poscar(
            structure['atoms'],
            structure['positions'],
            structure['cell'],
            poscar_path
        )
        files.append(poscar_path)
        
        # INCAR
        incar_path = os.path.join(output_dir, "INCAR")
        with open(incar_path, 'w') as f:
            f.write(self.generate_incar(structure))
        files.append(incar_path)
        
        # KPOINTS
        kpoints_path = os.path.join(output_dir, "KPOINTS")
        with open(kpoints_path, 'w') as f:
            f.write(self.generate_kpoints(structure))
        files.append(kpoints_path)
        
        # POTCAR.spec
        potcar_path = os.path.join(output_dir, "POTCAR.spec")
        with open(potcar_path, 'w') as f:
            f.write(self.generate_potcar_spec(structure))
        files.append(potcar_path)
        
        return files


class FEFFInputGenerator:
    """Generate FEFF input files.

    The FEFF calculation is written in the same split-file style used by
    Lightshow/pymatgen: ``HEADER``, ``PARAMETERS``, ``POTENTIALS``, and
    ``ATOMS`` are written alongside the concatenated ``feff.inp``. The
    absorber atom is always assigned ``ipot = 0``. Other scattering
    potentials, including atoms of the same element as the absorber, are assigned
    positive ``ipot`` values. The cluster is generated from periodic images of
    the input cell so surface supercells and small primitive cells both produce
    a physically meaningful local cluster.
    """

    DEFAULT_TAGS = {
        "CONTROL": "1 1 1 1 1 1",
        "COREHOLE": "RPA",
        "S02": "0",
        "EXCHANGE": "0 0.0 0.0 2",
        "XANES": "4 0.04 0.1",
        "RPATH": "-1",
    }

    def _fmt_float(self, value: float) -> str:
        """Compact FEFF-style floating point formatting."""
        value = float(value)
        if abs(value) < 5e-13:
            value = 0.0
        return f"{value:.8g}"

    def _cell_translation_limits(self, cell: np.ndarray, radius: float) -> Tuple[int, int, int]:
        """Return conservative image repeat limits for a cluster radius."""
        lengths = [np.linalg.norm(cell[i]) for i in range(3)]
        limits = []
        for length in lengths:
            if length <= 1e-8:
                limits.append(0)
            else:
                limits.append(int(np.ceil(radius / length)) + 1)
        return tuple(limits)

    def _build_periodic_cluster(
        self,
        structure: Dict,
        absorber: str,
        absorber_index: int,
        radius: float,
    ) -> Tuple[List[Dict], int]:
        """Build an absorber-centered periodic cluster.

        Returns:
            cluster: list of dictionaries with element, relative position,
                distance, original atom index, and is_absorber flag.
            abs_idx: absolute atom index of the selected absorber in the input cell.
        """
        atoms = list(structure["atoms"])
        positions = np.array(structure["positions"], dtype=float)
        cell = np.array(structure["cell"], dtype=float)

        abs_indices = [i for i, atom in enumerate(atoms) if atom == absorber]
        if not abs_indices:
            raise ValueError(f"No absorber atom {absorber!r} found in structure")
        if absorber_index >= len(abs_indices):
            absorber_index = 0
        abs_idx = abs_indices[absorber_index]
        abs_pos = positions[abs_idx]

        n1, n2, n3 = self._cell_translation_limits(cell, radius)
        cluster = []
        seen = set()

        for ia in range(-n1, n1 + 1):
            for ib in range(-n2, n2 + 1):
                for ic in range(-n3, n3 + 1):
                    translation = ia * cell[0] + ib * cell[1] + ic * cell[2]
                    image = (ia, ib, ic)

                    for atom_index, (atom, pos) in enumerate(zip(atoms, positions)):
                        rel = pos + translation - abs_pos
                        dist = float(np.linalg.norm(rel))

                        if dist > radius + 1e-8:
                            continue

                        is_absorber = atom_index == abs_idx and image == (0, 0, 0)

                        # Deduplicate atoms sitting on periodic boundaries. Round the
                        # relative coordinate so numerically identical images collapse.
                        key = (
                            atom,
                            round(float(rel[0]), 8),
                            round(float(rel[1]), 8),
                            round(float(rel[2]), 8),
                            is_absorber,
                        )
                        if key in seen:
                            continue
                        seen.add(key)

                        cluster.append({
                            "element": atom,
                            "rel": rel,
                            "distance": dist,
                            "atom_index": atom_index,
                            "image": image,
                            "is_absorber": is_absorber,
                        })

        # The true absorber must be first, then neighbors sorted by distance.
        cluster.sort(key=lambda item: (not item["is_absorber"], item["distance"], item["element"], item["atom_index"]))
        return cluster, abs_idx

    def _build_potentials(self, cluster: List[Dict], absorber: str) -> Tuple[List[Dict], Dict[str, int]]:
        """Build FEFF potentials and element-to-ipot mapping.

        FEFF convention:
            ipot 0 = absorbing atom
            ipot > 0 = scattering potentials

        If the same element appears as both absorber and scatterer, it gets two
        rows in POTENTIALS: ipot 0 for the absorber and another positive ipot for
        the non-absorbing atoms of that same element.
        """
        scatterer_elements = []
        for item in cluster:
            if item["is_absorber"]:
                continue
            elem = item["element"]
            if elem not in scatterer_elements:
                scatterer_elements.append(elem)

        # Put same-element scatterer first, matching common pymatgen/FEFF style.
        scatterer_elements.sort(key=lambda elem: (elem != absorber, elem))

        potentials = [{
            "ipot": 0,
            "element": absorber,
            "z": ATOMIC_NUMBERS.get(absorber),
            "xnatph": 0.0001,
        }]

        scatterer_ipot = {}
        for ipot, elem in enumerate(scatterer_elements, start=1):
            count = sum(1 for item in cluster if (not item["is_absorber"] and item["element"] == elem))
            scatterer_ipot[elem] = ipot
            potentials.append({
                "ipot": ipot,
                "element": elem,
                "z": ATOMIC_NUMBERS.get(elem),
                "xnatph": count,
            })

        for pot in potentials:
            if pot["z"] is None:
                raise ValueError(f"Unknown atomic number for element: {pot['element']}")

        return potentials, scatterer_ipot

    def _header_lines(self, structure: Dict, absorber: str, edge: str, radius: float) -> List[str]:
        atoms = structure.get("atoms", [])
        cell = np.array(structure.get("cell", np.eye(3)), dtype=float)
        metadata = structure.get("metadata", {})

        lengths = [np.linalg.norm(cell[i]) for i in range(3)]
        def angle(v1, v2):
            n1 = np.linalg.norm(v1)
            n2 = np.linalg.norm(v2)
            if n1 < 1e-12 or n2 < 1e-12:
                return 90.0
            return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))))

        alpha = angle(cell[1], cell[2])
        beta = angle(cell[0], cell[2])
        gamma = angle(cell[0], cell[1])

        formula_parts = []
        for elem in dict.fromkeys(atoms):
            formula_parts.append(f"{elem}{atoms.count(elem)}")
        formula = " ".join(formula_parts)

        lines = [
            "* This FEFF.inp file generated by CO2RR-XAS Agent",
            f"TITLE comment: absorber={absorber}, edge={edge}, radius={radius:.2f} Angstrom",
            f"TITLE Source: {metadata.get('source', 'CO2RR_XAS_agent')}",
            f"TITLE Structure Summary: {formula}",
            f"TITLE abc:  {lengths[0]:.6f}   {lengths[1]:.6f}   {lengths[2]:.6f}",
            f"TITLE angles: {alpha:.6f}  {beta:.6f}  {gamma:.6f}",
            f"TITLE sites: {len(atoms)}",
        ]
        for i, (atom, pos) in enumerate(zip(atoms, np.array(structure.get("positions", []), dtype=float))):
            lines.append(f"* {i + 1:4d} {atom:<2s} {pos[0]:12.6f} {pos[1]:12.6f} {pos[2]:12.6f}")
        lines.append("")
        return lines

    def generate_input_sections(
        self,
        structure: Dict,
        absorber: str,
        absorber_index: int = 0,
        edge: str = "K",
        radius: float = 6.0,
        scf: bool = True,
        fms: bool = True,
        corehole: str = "RPA",
        s02: float = 0,
        exchange: str = "0 0.0 0.0 2",
        xanes: str = "4 0.04 0.1",
        rpath: str = "-1",
        scf_radius: Optional[float] = None,
        fms_radius: Optional[float] = None,
    ) -> Dict[str, str]:
        """Generate Lightshow/pymatgen-style FEFF input sections.

        Args:
            structure: Structure dictionary with atoms, positions, and cell.
            absorber: Absorber element symbol.
            absorber_index: Which absorber atom of that element to center on.
            edge: Absorption edge, e.g. K, L2, L3, L23.
            radius: Cluster radius in Angstrom.
            scf: Whether to include the SCF card.
            fms: Whether to include the FMS card.
            corehole: FEFF COREHOLE model.
            s02: FEFF S02 value.
            exchange: FEFF EXCHANGE card values.
            xanes: FEFF XANES card values.
            rpath: FEFF RPATH card value.
            scf_radius: Optional SCF radius. Defaults to radius.
            fms_radius: Optional FMS radius. Defaults to radius.
        """
        radius = float(radius)
        scf_radius = radius if scf_radius is None else float(scf_radius)
        fms_radius = radius if fms_radius is None else float(fms_radius)

        cluster, abs_idx = self._build_periodic_cluster(
            structure=structure,
            absorber=absorber,
            absorber_index=absorber_index,
            radius=radius,
        )
        potentials, scatterer_ipot = self._build_potentials(cluster, absorber)

        header_lines = self._header_lines(structure, absorber, edge, radius)
        parameter_lines = [
            "CONTROL 1 1 1 1 1 1",
            f"COREHOLE {corehole}",
            f"S02 {self._fmt_float(s02)}",
            f"EXCHANGE {exchange}",
        ]

        if scf:
            parameter_lines.append(f"SCF {self._fmt_float(scf_radius)} 0 100 0.2 3")
        if fms:
            parameter_lines.append(f"FMS {self._fmt_float(fms_radius)} 0")

        parameter_lines.extend([
            f"XANES {xanes}",
            f"EDGE {edge}",
            f"RPATH {rpath}",
        ])

        potential_lines = [
            "POTENTIALS",
            "  *ipot    Z  tag      lmax1    lmax2    xnatph(stoichometry)    spinph",
            "******-  **-  ****-  ******-  ******-  **********************  ********",
        ]

        for pot in potentials:
            xnatph = pot["xnatph"]
            if pot["ipot"] == 0:
                xnatph_str = f"{float(xnatph):.4f}"
            else:
                xnatph_str = f"{int(xnatph)}"
            potential_lines.append(
                f"{pot['ipot']:7d} {pot['z']:4d}  {pot['element']:<4s}"
                f"{ -1:10d}{ -1:9d}{xnatph_str:>23s}{0:10d}"
            )

        atom_lines = [
            "ATOMS",
            "   *       x         y         z    ipot  Atom      Distance    Number",
            "************  ********  ********  ******  ******  **********  ********",
        ]

        for number, item in enumerate(cluster):
            elem = item["element"]
            rel = item["rel"]
            dist = item["distance"]
            ipot = 0 if item["is_absorber"] else scatterer_ipot[elem]
            atom_lines.append(
                f"{self._fmt_float(rel[0]):>12s} {self._fmt_float(rel[1]):>9s} {self._fmt_float(rel[2]):>9s}"
                f" {ipot:7d}  {elem:<4s} {self._fmt_float(dist):>11s} {number:8d}"
            )

        sections = {
            "HEADER": "\n".join(header_lines).rstrip() + "\n",
            "PARAMETERS": "\n".join(parameter_lines).rstrip() + "\n",
            "POTENTIALS": "\n".join(potential_lines).rstrip() + "\n",
            "ATOMS": "\n".join(atom_lines).rstrip() + "\n",
        }
        sections["feff.inp"] = (
            sections["HEADER"]
            + "\n"
            + sections["PARAMETERS"]
            + "\n"
            + sections["POTENTIALS"]
            + "\n"
            + sections["ATOMS"]
            + "\nEND\n"
        )
        return sections

    def generate_input(self, *args, **kwargs) -> str:
        """Generate complete ``feff.inp`` content."""
        return self.generate_input_sections(*args, **kwargs)["feff.inp"]

    def write_input(
        self,
        structure: Dict,
        output_dir: str,
        absorber: str,
        edge: str = "K",
        **kwargs
    ) -> str:
        """Write ``HEADER``, ``PARAMETERS``, ``POTENTIALS``, ``ATOMS``, and ``feff.inp``."""
        ensure_dir(output_dir)

        sections = self.generate_input_sections(structure, absorber, edge=edge, **kwargs)
        for filename in ["HEADER", "PARAMETERS", "POTENTIALS", "ATOMS", "feff.inp"]:
            with open(os.path.join(output_dir, filename), "w") as f:
                f.write(sections[filename])

        return os.path.join(output_dir, "feff.inp")

class FDMNESInputGenerator:
    """Generate FDMNES input files."""

    DEFAULT_OPTIONS = {
        "Quadrupole": False,
        "Relativism": False,
        "Spinorbit": False,
        "SCF": True,
        "SCFexc": False,
        "Screening": False,
        "Full_atom": False,
        "TDDFT": False,
        "Green": False,   # False = finite difference method; True = Green multiple scattering
        "Nonexc": False,  # Non-excited absorber: no core hole and no screening
        "Excited": False, # Force excited absorbing atom treatment
    }

    OPTION_ORDER = [
        "Quadrupole",
        "Relativism",
        "Spinorbit",
        "SCF",
        "SCFexc",
        "Screening",
        "Full_atom",
        "TDDFT",
        "Green",
        "Nonexc",
        "Excited",
    ]

    def _fmt_fdmnes_float(self, value: float) -> str:
        """Format values like -5. 0.2 50. for FDMNES style."""
        value = float(value)
        if value.is_integer():
            return f"{int(value)}."
        return f"{value:g}"

    def _merge_options(self, options: Optional[Dict] = None) -> Dict:
        merged = dict(self.DEFAULT_OPTIONS)
        if options:
            for key, value in options.items():
                if key in merged:
                    merged[key] = bool(value)
        return merged

    def generate_actual_input(
        self,
        structure: Dict,
        absorber: str,
        edge: str = "K",
        radius: float = 6.0,
        energy_range: Tuple[float, float, float] = (-5.0, 0.2, 50.0),
        options: Optional[Dict] = None,
        filout: Optional[str] = None,
    ) -> str:
        """
        Generate actual FDMNES input, e.g. Cu_in.txt.

        energy_range order is:
            (emin, step, emax)
        """
        atoms = structure["atoms"]
        positions = np.array(structure["positions"])
        cell = np.array(structure["cell"])

        absorber_z = ATOMIC_NUMBERS.get(absorber)
        if absorber_z is None:
            raise ValueError(f"Unknown absorber element: {absorber}")

        opts = self._merge_options(options)

        # Convert Cartesian positions to fractional coordinates.
        # This code stores lattice vectors as rows: cart = frac @ cell.
        frac_positions = np.dot(positions, np.linalg.inv(cell))
        frac_positions = frac_positions % 1.0

        # Cell parameters
        a = np.linalg.norm(cell[0])
        b = np.linalg.norm(cell[1])
        c = np.linalg.norm(cell[2])

        alpha = np.degrees(
            np.arccos(np.clip(np.dot(cell[1], cell[2]) / (b * c), -1.0, 1.0))
        )
        beta = np.degrees(
            np.arccos(np.clip(np.dot(cell[0], cell[2]) / (a * c), -1.0, 1.0))
        )
        gamma = np.degrees(
            np.arccos(np.clip(np.dot(cell[0], cell[1]) / (a * b), -1.0, 1.0))
        )

        emin, estep, emax = energy_range
        filout = filout or absorber

        lines = [
            "Filout",
            f"  {filout}",
            "",
            "Range",
            f"  {self._fmt_fdmnes_float(emin)} {self._fmt_fdmnes_float(estep)} {self._fmt_fdmnes_float(emax)}",
            "",
            "Radius",
            f"  {float(radius):.1f}",
            "",
            "Energpho",
            "",
            "Memory_save",
            "",
        ]

        for keyword in self.OPTION_ORDER:
            if opts.get(keyword, False):
                lines.extend([keyword, ""])

        lines.extend([
            "Perdew",
            "",
            "Crystal",
            f"{a:.6f} {b:.6f} {c:.6f} {alpha:.6f} {beta:.6f} {gamma:.6f}",
        ])

        for atom, frac in zip(atoms, frac_positions):
            z = ATOMIC_NUMBERS.get(atom)
            if z is None:
                raise ValueError(f"Unknown atomic number for element: {atom}")
            lines.append(f"  {z:3d} {frac[0]:.8f} {frac[1]:.8f} {frac[2]:.8f}")

        lines.extend([
            "",
            "Z_Absorber",
            f"  {absorber_z}",
            "",
            "Edge",
            f"  {edge}",
            "",
            "Convolution",
            "",
            "End",
            "",
        ])

        return "\n".join(lines)

    def generate_fdmfile(self, input_filename: str) -> str:
        """Generate driver file fdmfile.txt."""
        return f"1\n{input_filename}\n"

    def write_input(
        self,
        structure: Dict,
        output_dir: str,
        absorber: str,
        edge: str = "K",
        radius: float = 6.0,
        energy_range: Tuple[float, float, float] = (-5.0, 0.2, 50.0),
        options: Optional[Dict] = None,
    ) -> Dict[str, str]:
        """
        Write both FDMNES files:
        - fdmfile.txt
        - {absorber}_in.txt
        """
        ensure_dir(output_dir)

        input_filename = f"{absorber}_in.txt"
        input_path = os.path.join(output_dir, input_filename)
        fdmfile_path = os.path.join(output_dir, "fdmfile.txt")

        actual_input = self.generate_actual_input(
            structure=structure,
            absorber=absorber,
            edge=edge,
            radius=radius,
            energy_range=energy_range,
            options=options,
        )

        with open(input_path, "w") as f:
            f.write(actual_input)

        with open(fdmfile_path, "w") as f:
            f.write(self.generate_fdmfile(input_filename))

        return {
            "fdmfile": fdmfile_path,
            "input": input_path,
        }


class VASPXASInputGenerator:
    """Generate two-step VASP XAS inputs.

    Layout per absorber/edge:

        VASP/
          submit.sh
          01_scf/
            POSCAR INCAR KPOINTS POTCAR.spec make_potcar.sh [POTCAR]
          02_xas/
            POSCAR INCAR KPOINTS POTCAR.spec make_potcar.sh [POTCAR]

    The user still chooses one XAS method (PBE or GW/GW0). The SCF step is a
    conventional ground-state calculation that writes WAVECAR/CHGCAR; the XAS
    step reuses those files when running through submit.sh.
    """

    DEFAULT_POTCAR_MAP = {
        'H': 'H', 'C': 'C', 'N': 'N', 'O': 'O',
        'Fe': 'Fe_pv', 'Co': 'Co_pv', 'Ni': 'Ni_pv',
        'Cu': 'Cu_pv', 'Zn': 'Zn', 'Pd': 'Pd_pv',
        'Ag': 'Ag', 'Pt': 'Pt_pv', 'Au': 'Au',
        'Al': 'Al', 'Rh': 'Rh_pv', 'Ir': 'Ir',
        'Sc': 'Sc_sv', 'Ti': 'Ti_pv', 'V': 'V_pv', 'Cr': 'Cr_pv',
        'Mn': 'Mn_pv', 'Y': 'Y_sv', 'Zr': 'Zr_sv', 'Nb': 'Nb_pv',
        'Mo': 'Mo_pv', 'Ru': 'Ru_pv', 'Hf': 'Hf_pv', 'Ta': 'Ta_pv',
        'W': 'W_pv', 'Re': 'Re_pv', 'Os': 'Os_pv'
    }

    def __init__(self, potcar_map: Optional[Dict[str, str]] = None):
        self.potcar_map = dict(self.DEFAULT_POTCAR_MAP)
        if potcar_map:
            self.potcar_map.update(potcar_map)

    def _normalize_method(self, method: str) -> str:
        method = (method or "PBE").upper().replace("GW0", "GW")
        if method in {"PBE", "GW"}:
            return method
        raise ValueError(f"Unsupported VASP XAS method: {method}. Use 'PBE', 'GW', or 'GW0'.")

    def _unique_elements(self, structure: Dict) -> List[str]:
        unique_elements = []
        for atom in structure['atoms']:
            if atom not in unique_elements:
                unique_elements.append(atom)
        return unique_elements

    def _default_nbands(self, structure: Dict, method: str) -> int:
        n_atoms = len(structure['atoms'])
        if method == "GW":
            return n_atoms * 8 + 100
        return n_atoms * 6 + 60

    def generate_incar_scf(self, structure: Dict, method: str = "PBE", nbands: int = None) -> str:
        """Generate INCAR for the first SCF step used by VASP XAS."""
        method = self._normalize_method(method)
        if nbands is None:
            nbands = self._default_nbands(structure, method)
        encut = 400 if method == "GW" else 520
        lines = [
            "# VASP INCAR: step 1 ground-state SCF for later XAS",
            "# Generated by CO2RR-XAS Agent",
            "",
            "SYSTEM = VASP_XAS_SCF",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "LREAL = .FALSE.",
            "PREC = Accurate",
            f"NBANDS = {nbands}",
            "",
            "# Static SCF; write charge and wavefunction for the XAS step",
            "IBRION = -1",
            "NSW = 0",
            "NELM = 120",
            "LWAVE = .TRUE.",
            "LCHARG = .TRUE.",
            "",
            "# Dipole correction for asymmetric one-sided slab",
            "LDIPOL = .TRUE.",
            "IDIPOL = 3",
            get_dipol_direct(structure),
            "AMIN = 0.01",
            "",
            "# Spin",
            "ISPIN = 2",
            "NCORE = 4",
        ]
        return "\n".join(lines)

    def generate_incar_xas(
        self,
        structure: Dict,
        absorber: str,
        method: str = "PBE",
        absorber_index: int = 0,
        nbands: int = None,
    ) -> str:
        """Generate INCAR for the second VASP XAS step."""
        method = self._normalize_method(method)
        if nbands is None:
            nbands = self._default_nbands(structure, method)
        encut = 400 if method == "GW" else 520
        lines = [
            f"# VASP INCAR: step 2 {method}-based XAS",
            f"# Absorber: {absorber}",
            "# Generated by CO2RR-XAS Agent",
            "",
            "SYSTEM = VASP_XAS",
            f"ENCUT = {encut}",
            "EDIFF = 1E-6",
            "ISMEAR = 0",
            "SIGMA = 0.05",
            "LREAL = .FALSE.",
            "PREC = Accurate",
            f"NBANDS = {nbands}",
            "",
            "# Reuse the converged SCF result from 01_scf when available",
            "ISTART = 1",
            "ICHARG = 11",
            "IBRION = -1",
            "NSW = 0",
            "",
            "# XAS / optics",
            "LOPTICS = .TRUE.",
            "CSHIFT = 0.1",
            "NEDOS = 2001",
        ]
        if method == "GW":
            lines.extend([
                "",
                "# GW0 settings",
                "ALGO = GW0",
                "NOMEGA = 100",
            ])
        lines.extend([
            "",
            "# Core-level / core-hole tags",
            "ICORELEVEL = 2",
            f"CLNT = {absorber_index + 1}",
            "CLN = 1",
            "CLL = 0",
            "CLZ = 0.5",
            "",
            "# Dipole correction for asymmetric one-sided slab",
            "LDIPOL = .TRUE.",
            "IDIPOL = 3",
            get_dipol_direct(structure),
            "AMIN = 0.01",
            "",
            "# Spin and output",
            "ISPIN = 2",
            "LWAVE = .TRUE.",
            "LCHARG = .TRUE.",
            "NCORE = 4",
        ])
        return "\n".join(lines)

    def generate_kpoints(self, structure: Dict) -> str:
        """Generate KPOINTS for VASP XAS steps."""
        cell = structure['cell']
        lengths = [np.linalg.norm(cell[i]) for i in range(3)]
        kpoints = [max(1, int(25 / l)) for l in lengths]
        kpoints[2] = 1
        return "\n".join([
            "Automatic mesh",
            "0",
            "Gamma",
            f"  {kpoints[0]}  {kpoints[1]}  {kpoints[2]}",
            "  0  0  0",
        ])

    def get_potcar_symbols(self, structure: Dict) -> List[str]:
        """Return POTCAR symbols in POSCAR element order."""
        return [self.potcar_map.get(elem, elem) for elem in self._unique_elements(structure)]

    def generate_potcar_spec(self, structure: Dict, method: str = "PBE") -> str:
        method = self._normalize_method(method)
        lines = [
            "# POTCAR specification",
            f"# VASP XAS method: {method}",
            "# Concatenate these POTCAR directories in this order:",
        ]
        for elem, potcar in zip(self._unique_elements(structure), self.get_potcar_symbols(structure)):
            lines.append(f"{elem}: {potcar}")
        return "\n".join(lines)

    def generate_make_potcar_script(
        self,
        structure: Dict,
        potcar_dir: Optional[str] = None,
        method: str = "PBE",
    ) -> str:
        method = self._normalize_method(method)
        potcar_dir = potcar_dir or "${CO2RR_POTCAR_DIR:-${VASP_POTCAR_DIR:-}}"
        symbols = self.get_potcar_symbols(structure)
        lines = [
            "#!/bin/bash",
            "set -euo pipefail",
            "",
            f"# Generated by CO2RR-XAS Agent for VASP {method} XAS inputs.",
            "# Set CO2RR_POTCAR_DIR or VASP_POTCAR_DIR to the root folder containing",
            "# subdirectories like Cu_pv/POTCAR, C/POTCAR, O/POTCAR, etc.",
            "",
            f"POTCAR_DIR=\"{potcar_dir}\"",
            "if [ -z \"${POTCAR_DIR}\" ]; then",
            "  echo 'ERROR: Set CO2RR_POTCAR_DIR or VASP_POTCAR_DIR, or pass potcar_dir to the agent.' >&2",
            "  exit 1",
            "fi",
            "",
            "rm -f POTCAR",
        ]
        for symbol in symbols:
            lines.extend([
                f"if [ ! -f \"${{POTCAR_DIR}}/{symbol}/POTCAR\" ]; then",
                f"  echo 'ERROR: Missing POTCAR: '${{POTCAR_DIR}}'/{symbol}/POTCAR' >&2",
                "  exit 1",
                "fi",
                f"cat \"${{POTCAR_DIR}}/{symbol}/POTCAR\" >> POTCAR",
            ])
        lines.extend(["", "echo 'Wrote POTCAR from:'", "echo \"  ${POTCAR_DIR}\""])
        return "\n".join(lines)

    def write_potcar_if_available(self, structure: Dict, output_dir: str, potcar_dir: Optional[str] = None) -> Optional[str]:
        if not potcar_dir:
            potcar_dir = os.environ.get("CO2RR_POTCAR_DIR") or os.environ.get("VASP_POTCAR_DIR")
        if not potcar_dir or not os.path.isdir(os.path.expandvars(potcar_dir)):
            return None
        root = os.path.expandvars(potcar_dir)
        potcar_path = os.path.join(output_dir, "POTCAR")
        with open(potcar_path, "wb") as out:
            for symbol in self.get_potcar_symbols(structure):
                src = os.path.join(root, symbol, "POTCAR")
                if not os.path.isfile(src):
                    raise FileNotFoundError(f"Missing POTCAR file: {src}")
                with open(src, "rb") as f:
                    out.write(f.read())
        return potcar_path

    def _write_common_step_files(self, structure: Dict, output_dir: str, method: str, potcar_dir: Optional[str]) -> List[str]:
        ensure_dir(output_dir)
        files = []
        poscar_path = os.path.join(output_dir, "POSCAR")
        write_poscar(structure['atoms'], structure['positions'], structure['cell'], poscar_path, selective_dynamics=False)
        files.append(poscar_path)
        kpoints_path = os.path.join(output_dir, "KPOINTS")
        with open(kpoints_path, 'w') as f:
            f.write(self.generate_kpoints(structure))
        files.append(kpoints_path)
        potcar_spec_path = os.path.join(output_dir, "POTCAR.spec")
        with open(potcar_spec_path, 'w') as f:
            f.write(self.generate_potcar_spec(structure, method=method))
        files.append(potcar_spec_path)
        make_potcar_path = os.path.join(output_dir, "make_potcar.sh")
        with open(make_potcar_path, 'w') as f:
            f.write(self.generate_make_potcar_script(structure, potcar_dir=potcar_dir, method=method))
        os.chmod(make_potcar_path, 0o755)
        files.append(make_potcar_path)
        real_potcar = self.write_potcar_if_available(structure, output_dir, potcar_dir=potcar_dir)
        if real_potcar:
            files.append(real_potcar)
        return files

    def generate_two_step_submit(
        self,
        job_name: str,
        account: Optional[str] = None,
        queue: Optional[str] = None,
        nodes: Optional[int] = None,
        walltime: Optional[str] = None,
        email: Optional[str] = None,
        method: str = "PBE",
    ) -> str:
        """Generate a two-step VASP submit script that records ISAAC metadata."""
        defaults = get_nersc_defaults()
        account = account or defaults["account"]
        queue = queue or defaults["queue"]
        nodes = nodes or defaults["nodes"]
        walltime = walltime or defaults["walltime"]
        method = self._normalize_method(method)
        code_version = "VASP 6.3.2"
        lines = [
            "#!/bin/bash",
            f"#SBATCH -J {job_name}",
            f"#SBATCH -q {queue}",
            f"#SBATCH -A {account}",
            "#SBATCH -C cpu",
            f"#SBATCH -N {nodes}",
            f"#SBATCH -t {walltime}",
        ]
        if email:
            lines.extend(["#SBATCH --mail-type=ALL", f"#SBATCH --mail-user={email}"])
        lines.extend([
            "",
            "set -uo pipefail",
            "module load vasp/6.4.3-cpu",
            "export OMP_NUM_THREADS=2",
            "export OMP_PLACES=threads",
            "export OMP_PROC_BIND=spread",
        ])
        lines.extend(_isaac_submit_metadata_setup(
            software="VASP",
            code_version=code_version,
            technique="DFT",
            instrument_name="VASP_Standard",
            vendor_or_project="VASP",
            recipe_target="repo://CO2RR_XAS_agent/vasp_xas_two_step",
        ))
        lines.extend([
            "EXIT_CODE=0",
            "export ISAAC_RUN_COMMAND=\"srun vasp_std in 01_scf; srun vasp_std in 02_xas\"",
            "",
            "echo '== Step 1: SCF =='",
            "cd 01_scf",
            "if [ ! -f POTCAR ]; then ./make_potcar.sh; fi",
            "srun vasp_std > vasp.out 2>&1 || EXIT_CODE=$?",
            "cd ..",
            "",
            "if [ \"${EXIT_CODE}\" -eq 0 ]; then",
            "  echo '== Step 2: XAS =='",
            "  cd 02_xas",
            "  cp -f ../01_scf/WAVECAR . 2>/dev/null || true",
            "  cp -f ../01_scf/CHGCAR . 2>/dev/null || true",
            "  if [ ! -f POTCAR ]; then ./make_potcar.sh; fi",
            "  srun vasp_std > vasp.out 2>&1 || EXIT_CODE=$?",
            "  cd ..",
            "else",
            "  echo 'Skipping XAS step because SCF failed with exit code '${EXIT_CODE}",
            "fi",
            "export ISAAC_EXIT_CODE=\"${EXIT_CODE}\"",
        ])
        lines.extend(_isaac_submit_metadata_finalize())
        lines.extend(["exit \"${EXIT_CODE}\""])
        return "\n".join(lines)

    def write_inputs(
        self,
        structure: Dict,
        output_dir: str,
        absorber: str,
        method: str = "PBE",
        potcar_dir: Optional[str] = None,
        account: Optional[str] = None,
        queue: Optional[str] = None,
        nodes: Optional[int] = None,
        walltime: Optional[str] = None,
        email: Optional[str] = None,
        absorber_index: int = 0,
        nbands: int = None,
        job_name: str = "vasp_xas",
    ) -> Dict:
        """Write two-step VASP XAS inputs and return structured file paths."""
        defaults = get_nersc_defaults()
        account = account or defaults["account"]
        queue = queue or defaults["queue"]
        nodes = nodes or defaults["nodes"]
        walltime = walltime or defaults["walltime"]
        method = self._normalize_method(method)
        ensure_dir(output_dir)
        scf_dir = os.path.join(output_dir, "01_scf")
        xas_dir = os.path.join(output_dir, "02_xas")

        scf_files = self._write_common_step_files(structure, scf_dir, method, potcar_dir)
        xas_files = self._write_common_step_files(structure, xas_dir, method, potcar_dir)

        scf_incar = os.path.join(scf_dir, "INCAR")
        with open(scf_incar, "w") as f:
            f.write(self.generate_incar_scf(structure, method=method, nbands=nbands))
        scf_files.append(scf_incar)

        xas_incar = os.path.join(xas_dir, "INCAR")
        with open(xas_incar, "w") as f:
            f.write(self.generate_incar_xas(
                structure=structure,
                absorber=absorber,
                method=method,
                absorber_index=absorber_index,
                nbands=nbands,
            ))
        xas_files.append(xas_incar)

        submit_path = os.path.join(output_dir, "submit.sh")
        with open(submit_path, "w") as f:
            f.write(self.generate_two_step_submit(
                job_name=job_name,
                account=account,
                queue=queue,
                nodes=nodes,
                walltime=walltime,
                email=email,
                method=method,
            ))
        os.chmod(submit_path, 0o755)

        return {
            "method": method,
            "layout": "two_step_scf_then_xas",
            "scf_dir": scf_dir,
            "xas_dir": xas_dir,
            "scf_files": scf_files,
            "xas_files": xas_files,
            "submit": submit_path,
            "potcar_dir": potcar_dir,
            "potcar_generated": any(os.path.basename(path) == "POTCAR" for path in scf_files + xas_files),
        }


class NERSCScriptGenerator:
    """Generate NERSC SLURM submission scripts."""
    
    def generate_script(
        self,
        job_name: str,
        software: str,
        account: Optional[str] = None,
        queue: Optional[str] = None,
        nodes: Optional[int] = None,
        walltime: Optional[str] = None,
        email: Optional[str] = None
    ) -> str:
        """Generate SLURM submission script with ISAAC run metadata capture."""
        defaults = get_nersc_defaults()
        account = account or defaults["account"]
        queue = queue or defaults["queue"]
        nodes = nodes or defaults["nodes"]
        walltime = walltime or defaults["walltime"]
        software_upper = software.upper()
        if software_upper == "VASP":
            module_line = "module load vasp/6.4.3-cpu"
            run_command = "srun vasp_std"
            code_version = "VASP 6.3.2"
            technique = "DFT"
            instrument_name = "VASP_Standard"
            vendor = "VASP"
            recipe_target = "repo://CO2RR_XAS_agent/vasp"
        elif software_upper == "FEFF":
            module_line = "module load feff/10.0"
            run_command = "feff"
            code_version = "FEFF 10"
            technique = "multiple_scattering_xas"
            instrument_name = "FEFF10"
            vendor = "FEFF"
            recipe_target = "repo://CO2RR_XAS_agent/feff"
        elif software_upper == "FDMNES":
            module_line = "module load fdmnes/2025.11"
            run_command = "fdmnes < fdmfile.txt"
            code_version = "FDMNES 2025.11"
            technique = "finite_difference_or_multiple_scattering_xas"
            instrument_name = "FDMNES"
            vendor = "FDMNES"
            recipe_target = "repo://CO2RR_XAS_agent/fdmnes"
        else:
            module_line = ""
            run_command = "echo 'No command defined'"
            code_version = "not_specified"
            technique = "not_specified"
            instrument_name = software_upper
            vendor = software_upper
            recipe_target = "not_specified"

        lines = [
            "#!/bin/bash",
            f"#SBATCH -J {job_name}",
            f"#SBATCH -q {queue}",
            f"#SBATCH -A {account}",
            "#SBATCH -C cpu",
            f"#SBATCH -N {nodes}",
            f"#SBATCH -t {walltime}",
        ]
        if email:
            lines.extend(["#SBATCH --mail-type=ALL", f"#SBATCH --mail-user={email}"])
        lines.extend(["", "set -uo pipefail"])
        if module_line:
            lines.extend([module_line])
        if software_upper == "VASP":
            lines.extend([
                "export OMP_NUM_THREADS=2",
                "export OMP_PLACES=threads",
                "export OMP_PROC_BIND=spread",
            ])
        lines.extend(_isaac_submit_metadata_setup(
            software=software_upper,
            code_version=code_version,
            technique=technique,
            instrument_name=instrument_name,
            vendor_or_project=vendor,
            recipe_target=recipe_target,
        ))
        lines.extend([
            "EXIT_CODE=0",
            f"export ISAAC_RUN_COMMAND=\"{run_command}\"",
            f"{run_command} > {software_upper.lower()}.out 2>&1 || EXIT_CODE=$?",
            "export ISAAC_EXIT_CODE=\"${EXIT_CODE}\"",
        ])
        lines.extend(_isaac_submit_metadata_finalize())
        lines.extend(["exit \"${EXIT_CODE}\""])
        return "\n".join(lines)
    
    def write_script(
        self,
        output_dir: str,
        job_name: str,
        software: str,
        filename: str = "submit.sh",
        **kwargs
    ) -> str:
        """Write submission script to file."""
        ensure_dir(output_dir)
        
        content = self.generate_script(job_name, software, **kwargs)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            f.write(content)
        
        # Make executable
        os.chmod(filepath, 0o755)
        
        return filepath


class XASInputGenerator:
    """
    Main XAS Input Generator combining all methods.
    """
    
    def __init__(self):
        self.relax_gen = RelaxationInputGenerator()
        self.feff_gen = FEFFInputGenerator()
        self.fdmnes_gen = FDMNESInputGenerator()
        self.vasp_gen = VASPXASInputGenerator()
        self.script_gen = NERSCScriptGenerator()
    
    def determine_absorbers(self, structure: Dict) -> List[Dict]:
        """
        Determine all absorbers and their edges for a structure.
        
        Returns:
            List of dicts with 'element', 'edge', 'vasp_allowed'
        """
        absorbers = []
        seen = set()
        
        # Get unique elements
        for atom in structure['atoms']:
            if atom in seen:
                continue
            seen.add(atom)
            
            edge, vasp_allowed = get_edge_for_element(atom)
            
            # Only include metals and C
            if atom in METALS_3D + METALS_4D + METALS_5D or atom == 'C':
                absorbers.append({
                    'element': atom,
                    'edge': edge,
                    'vasp_allowed': vasp_allowed
                })
        
        return absorbers

    def generate_all_inputs(
        self,
        structure: Dict,
        output_dir: str,
        nersc_account: Optional[str] = None,
        nersc_queue: Optional[str] = None,
        nersc_nodes: Optional[int] = None,
        nersc_walltime: Optional[str] = None,
        email: Optional[str] = None,
        cluster_radius: float = 6.0,
        fdmnes_options: Optional[Dict] = None,
        fdmnes_energy_range: Tuple[float, float, float] = (-5.0, 0.2, 50.0),
        edge_override: Optional[str] = None,
        vasp_method: str = "PBE",
        potcar_dir: Optional[str] = None,
    ) -> Dict:
        """
        Generate all XAS inputs for a structure.
        
        Creates:
        - relax/ folder with VASP relaxation
        - xas/{Element}_{Edge}_edge/{Method}/ folders
        
        Returns:
            Dictionary with all generated file paths
        """
        results = {
            'relax': {},
            'xas': {}
        }
        
        # 1. Generate relaxation inputs
        relax_dir = os.path.join(output_dir, "relax")
        relax_files = self.relax_gen.write_inputs(structure, relax_dir)
        
        # Relaxation submit script
        meta = structure.get('metadata', {})
        job_base = f"{meta.get('element', meta.get('element1', 'struct'))}"
        if meta.get('adsorbate'):
            job_base += f"_{meta['adsorbate']}"
        
        relax_script = self.script_gen.write_script(
            relax_dir,
            f"{job_base}_relax",
            "VASP",
            "submit_relax.sh",
            account=nersc_account,
            queue=nersc_queue,
            nodes=nersc_nodes,
            walltime=nersc_walltime,
            email=email
        )
        
        results['relax'] = {
            'files': relax_files,
            'submit_script': relax_script
        }
        
        # 2. Determine absorbers
        absorbers = self.determine_absorbers(structure)
        
        # 3. Generate XAS inputs for each absorber
        for abs_info in absorbers:
            element = abs_info['element']
            edge = abs_info['edge']
            vasp_allowed = abs_info['vasp_allowed']

            if edge_override:
                edge = edge_override
                vasp_allowed = vasp_allowed and edge == "K"
            
            edge_key = f"{element}_{edge}_edge"
            edge_dir = os.path.join(output_dir, "xas", edge_key)
            
            results['xas'][edge_key] = {}
            

            # FDMNES: one folder only.
            # Finite difference if Green=False; multiple scattering if Green=True.
            fdmnes_options = fdmnes_options or FDMNESInputGenerator.DEFAULT_OPTIONS
            fdmnes_method = "green" if fdmnes_options.get("Green", False) else "fdm"

            fdmnes_dir = os.path.join(edge_dir, "FDMNES")
            fdmnes_files = self.fdmnes_gen.write_input(
                structure=structure,
                output_dir=fdmnes_dir,
                absorber=element,
                edge=edge,
                radius=cluster_radius,
                energy_range=fdmnes_energy_range,
                options=fdmnes_options,
            )

            fdmnes_script = self.script_gen.write_script(
                fdmnes_dir,
                f"{job_base}_{element}_{edge}_fdmnes_{fdmnes_method}",
                "FDMNES",
                account=nersc_account,
                queue=nersc_queue,
                email=email,
            )

            results["xas"][edge_key]["FDMNES"] = {
                "method": "green_multiple_scattering" if fdmnes_options.get("Green", False) else "finite_difference",
                "fdmfile": fdmnes_files["fdmfile"],
                "input": fdmnes_files["input"],
                "submit": fdmnes_script,
                "options": fdmnes_options,
                "radius_angstrom": cluster_radius,
                "energy_range": fdmnes_energy_range,
            }
            
            # FEFF
            feff_dir = os.path.join(edge_dir, "FEFF")
            feff_file = self.feff_gen.write_input(
                structure, feff_dir, element, edge=edge,
                radius=cluster_radius
            )
            feff_script = self.script_gen.write_script(
                feff_dir, f"{job_base}_{element}_{edge}_feff", "FEFF",
                account=nersc_account, queue=nersc_queue, email=email
            )
            results['xas'][edge_key]['FEFF'] = {
                'input': feff_file,
                'submit': feff_script,
                'section_files': {
                    'HEADER': os.path.join(feff_dir, 'HEADER'),
                    'PARAMETERS': os.path.join(feff_dir, 'PARAMETERS'),
                    'POTENTIALS': os.path.join(feff_dir, 'POTENTIALS'),
                    'ATOMS': os.path.join(feff_dir, 'ATOMS'),
                },
            }
            
            # VASP (only for K-edge): one selected method, but two execution steps.
            # Step 1: SCF. Step 2: XAS using CHGCAR/WAVECAR from SCF.
            if vasp_allowed:
                vasp_method_norm = (vasp_method or "PBE").upper().replace("GW0", "GW")
                vasp_dir = os.path.join(edge_dir, "VASP")
                vasp_result = self.vasp_gen.write_inputs(
                    structure=structure,
                    output_dir=vasp_dir,
                    absorber=element,
                    method=vasp_method_norm,
                    potcar_dir=potcar_dir,
                    account=nersc_account,
                    queue=nersc_queue,
                    nodes=nersc_nodes,
                    walltime=nersc_walltime,
                    email=email,
                    job_name=f"{job_base}_{element}_{edge}_vasp_{vasp_method_norm.lower()}",
                )
                results['xas'][edge_key]['VASP'] = vasp_result
        
        return results


# =============================================================================
# Skill Interface Function
# =============================================================================

def execute_xas_input_generation(
    structure_file: str,
    output_dir: str,
    nersc_account: Optional[str] = None,
    nersc_queue: Optional[str] = None,
    nersc_nodes: Optional[int] = None,
    nersc_walltime: Optional[str] = None,
    email: Optional[str] = None,
    cluster_radius: float = 6.0,
    fdmnes_options: Optional[Dict] = None,
    fdmnes_energy_range: Tuple[float, float, float] = (-5.0, 0.2, 50.0),
    edge_override: Optional[str] = None,
    vasp_method: str = "PBE",
    potcar_dir: Optional[str] = None,
    structure_metadata: Optional[Dict] = None
) -> Dict:
    """
    Execute Skill 2: XAS Input Generation
    
    Args:
        structure_file: Path to POSCAR/CONTCAR file
        output_dir: Output directory
        nersc_account: NERSC allocation account
        nersc_queue: SLURM queue
        nersc_nodes: Number of nodes
        nersc_walltime: Wall time
        email: Email for notifications
        cluster_radius: Cluster radius for FEFF/FDMNES
        structure_metadata: Optional metadata dict
        
    Returns:
        Dictionary with results and file paths
    """
    defaults = get_nersc_defaults()
    nersc_account = nersc_account or defaults["account"]
    nersc_queue = nersc_queue or defaults["queue"]
    nersc_nodes = nersc_nodes or defaults["nodes"]
    nersc_walltime = nersc_walltime or defaults["walltime"]

    # Read structure
    structure = read_structure_file(structure_file)
    
    # Add metadata if provided
    if structure_metadata:
        structure['metadata'] = structure_metadata
    elif 'metadata' not in structure:
        structure['metadata'] = {}
    
    # Generate all inputs
    generator = XASInputGenerator()
    results = generator.generate_all_inputs(
        structure=structure,
        output_dir=output_dir,
        nersc_account=nersc_account,
        nersc_queue=nersc_queue,
        nersc_nodes=nersc_nodes,
        nersc_walltime=nersc_walltime,
        email=email,
        fdmnes_options=fdmnes_options,
        fdmnes_energy_range=fdmnes_energy_range,
        edge_override=edge_override,
        cluster_radius=cluster_radius,
        vasp_method=vasp_method,
        potcar_dir=potcar_dir
    )
    
    results['status'] = 'success'
    results['structure_file'] = structure_file
    results['output_dir'] = output_dir
    
    return results
