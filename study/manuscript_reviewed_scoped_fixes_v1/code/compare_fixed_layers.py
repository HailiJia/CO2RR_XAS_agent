"""Compare the supplied clean Cu/Au parents with two or three fixed layers."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from ase.io import read

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'figure_data' / 'fixed_layer_check'


def run():
    p3, p2 = DATA/'CONTCAR_3fixed.vasp', DATA/'CONTCAR_2fixed.vasp'
    a, b = read(p3, format='vasp'), read(p2, format='vasp')
    assert a.get_chemical_symbols() == b.get_chemical_symbols()
    assert np.allclose(a.cell.array, b.cell.array, atol=1e-12)
    symbols = np.array(a.get_chemical_symbols())
    z = np.sort(a.positions[:, 2])
    gaps = np.flatnonzero(np.diff(z) > .8)
    cuts = (z[gaps] + z[gaps+1])/2
    layers = np.digitize(a.positions[:, 2], cuts) + 1
    assert set(layers) == {1, 2, 3, 4, 5}
    for atoms, expected in [(a, 3), (b, 2)]:
        fixed = set(int(i) for constraint in atoms.constraints
                    for i in constraint.get_indices())
        assert fixed == set(np.flatnonzero(layers <= expected))
    frac_delta = (b.positions-a.positions) @ np.linalg.inv(a.cell.array)
    frac_delta[:, :2] -= np.round(frac_delta[:, :2])
    delta = frac_delta @ a.cell.array
    rows = []
    for element in ['Cu', 'Au']:
        ids4 = (symbols == element) & (layers == 4)
        ids5 = (symbols == element) & (layers == 5)
        d3 = float(a.positions[ids5, 2].mean()-a.positions[ids4, 2].mean())
        d2 = float(b.positions[ids5, 2].mean()-b.positions[ids4, 2].mean())
        rows.append(dict(element=element, n_surface=int(ids5.sum()),
                         d54_3fixed_A=d3, d54_2fixed_A=d2,
                         change_d54_A=d2-d3,
                         mean_surface_z_shift_A=float(delta[ids5, 2].mean()),
                         surface_RMS_displacement_A=float(np.sqrt(
                             np.mean(np.sum(delta[ids5]**2, axis=1)))),
                         max_surface_displacement_A=float(
                             np.linalg.norm(delta[ids5], axis=1).max())))
    table = pd.DataFrame(rows)
    table.to_csv(DATA/'surface_comparison.csv', index=False)
    report = dict(pair={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in [p2, p3]}, atom_count=len(a),
                  pairing='same element sequence, cell, and bottom-two anchor layers',
                  coordinate_difference='in-plane minimum image; vacuum nonperiodic; no rigid fit',
                  fixed_masks_verified=True,
                  conclusion='Uppermost interlayer spacings change by less than 0.008 A. '
                             'Au mean surface height shifts by about 0.138 A, so '
                             'the structures are not identical. No spectral or energy '
                             'convergence is established by this coordinate comparison.',
                  rows=rows)
    (DATA/'comparison.json').write_text(json.dumps(report, indent=2)+'\n')
    print(table.to_string(index=False))


if __name__ == '__main__':
    run()
