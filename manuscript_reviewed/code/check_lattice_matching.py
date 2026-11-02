"""Recompute the two matching objectives from frozen PR18 source methods.

Run from any directory: python code/check_lattice_matching.py
Only the pure matching methods are evaluated; no structures/jobs are changed.
"""
from pathlib import Path
import ast
import hashlib
import json
from types import SimpleNamespace
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'provenance' / 'lattice_matching_source'
LATTICE = {'Cu': 3.615, 'Au': 4.078}


def method(filename, name):
    path = SOURCE / filename
    tree = ast.parse(path.read_text())
    node = next(n for n in ast.walk(tree)
                if isinstance(n, ast.FunctionDef) and n.name == name)
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {'np': np, 'LATTICE_CONSTANTS': LATTICE}
    exec(compile(module, str(path), 'exec'), namespace)
    return namespace[name]


def run():
    automatic = method('structure_generator.py', '_match_lateral_repeats')(
        SimpleNamespace(metal_data={s: {'a': a} for s, a in LATTICE.items()}),
        'Cu', 'Au', max_repeat=10)
    bounded = method('slab_generator.py', '_find_minimal_interface_match')(
        None, 'Cu', 'Au', max_strain=0.05, max_repeats=12,
        match_side='balanced')
    assert (automatic['element1_repeats'], automatic['element2_repeats']) == (9, 8)
    assert (bounded['left_repeats'], bounded['right_repeats']) == (6, 5)
    assert automatic['max_abs_strain'] < 0.0014
    assert bounded['max_abs_strain'] <= 0.05
    report = {
        'source_repository': 'https://github.com/HailiJia/CO2RR_XAS_agent',
        'source_commit': '390b1b899c4dca43685fda1380a494b1d5019fcd',
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(SOURCE.glob('*.py'))},
        'lattice_constants_A': LATTICE,
        'strain_definition': 'common_length / unstrained_length - 1',
        'automatic_minimum_strain_within_10_repeats': automatic,
        'minimum_total_repeats_within_5_percent_bound': bounded,
        'balanced_one_to_one_strain': {
            s: float(np.mean(list(LATTICE.values())) / a - 1)
            for s, a in LATTICE.items()},
        'scope': 'Initial imposed interface-parallel matching strain, not a '
                 'measurement of local relaxed strain or every cohort parent.',
        'documentation_discrepancy': 'The bounded matcher docstring/test_local '
                 'expects 7:6; evaluating its implemented balanced 5% objective '
                 'selects 6:5. Requires source-documentation reconciliation.'}
    output = ROOT / 'provenance' / 'lattice_matching_check.json'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    run()
