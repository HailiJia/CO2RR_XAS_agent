"""Numeric Cu references. Run from any directory: python path/to/process.py.

No simulation, portal access, ML refitting, extrapolation, or figure digitization.
"""
from pathlib import Path
import gzip
import hashlib
import io
import json
import platform
import re
import sys
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.ndimage import gaussian_filter1d
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RAW = ROOT/'raw'
OUT = ROOT/'processed'
MATERIALS = ['Cu', 'Cu2O', 'CuO']
METHODS = ['fdmnes_green', 'fdmnes_fdm', 'feff', 'vasp_pbe', 'vasp_gw']
COLORS = dict(zip(METHODS, ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']))
LABELS = dict(zip(METHODS, ['FDMNES Green', 'FDMNES FDM', 'FEFF', 'VASP PBE', 'VASP “GW”*']))
SETTINGS = {
    'protocol': 'reference-v1', 'pre_window_eV': [8974., 8976.],
    'post_window_eV': [9020., 9028.], 'metric_interpretation': 'conditional short-window local edge-jump comparison, not standardized absolute accuracy; conventional long pre-edge regression is unsupported by FDMNES range', 'metric_window_eV': [8980., 9020.],
    'grid_step_eV': .2, 'shift_sign': 'E_plotted = E_input + shift_eV',
    'common_calibration': 'one shift per backend fitted on Cu only; shared by all compounds and FDMNES modes',
    'common_shift_bounds_eV': {'fdmnes': [-20., 20.], 'feff': [-25., 25.], 'vasp': [0., 200.]},
    'shape_residual_shift_bounds_eV': [-20., 20.],
    'objective': 'edge-jump normalized RMSE on 8980–9020 eV, 0.2 eV linear grid; no extrapolation',
    'fit_degrees_of_freedom': 'common: 1 backend shift on Cu; optional shape: 1 residual shift per available compound/method; widths fixed',
    'normalization_nominal_offsets_eV': {'fdmnes': 0., 'feff': -12., 'vasp_pbe': 120., 'vasp_gw': 130., 'experiment': 0.},
    'normalization_energy_note': 'fixed windows evaluated on nominal axes once before shift fitting; FDMNES 0; FEFF -12 maps a below-E_F interval (native 8986–8988 eV); VASP PBE +120 follows notebook; GW alias +130 places window below first numeric onset; these offsets set normalization windows only; fitted shifts never refit background or scale',
    'preprocessing_order': ['parse numeric columns', 'average diagonal tensor components if VASP',
                            'average duplicate energy rows', 'remaining core-lifetime convolution where explicitly specified',
                            'constant local pre-edge subtraction on fixed nominal axes', 'local edge-jump scaling', 'energy shift',
                            'interpolate only within support', 'metrics'],
    'additional_lorentzian_FWHM_eV': 1.55, 'additional_lorentzian_HWHM_eV': .775,
    'additional_lifetime_applies_to': ['vasp core-hole dielectric outputs', 'unconvolved CuO FDM output'],
    'lifetime_source': 'explicit comparison prescription using Cu K Gamma_hole value in the five supplied FDMNES convolution logs; not a version-default claim; constant core lifetime only',
    'additional_gaussian_sigma_eV': 0., 'additional_gaussian_FWHM_eV': 0.,
    'instrument_resolution': 'not reported; no measured instrumental Gaussian imposed',
    'gaussian_sensitivity_sigma_eV': 1., 'gaussian_sensitivity_FWHM_eV': 2.354820045,
    'gaussian_sensitivity_source': 'original notebook cells 1–2; unvalidated sensitivity, not measured resolution or fitted width',
    'lorentzian_kernel': 'gamma/(pi*(dE**2+gamma**2)); quadrature on native grid, finite-domain kernel weights renormalized; no spectral extrapolation',
    'normalization_limit': '2 eV local pre-edge window, not conventional long-range pre-edge regression; post window is local, not asymptotic; all conclusions conditional on these short ranges',
    'experimental_anchor': 'supplied Energy column unchanged; no independent calibration/uncertainty metadata supplied',
    'vasp_label_limit': 'folder aliases retained; recorded ALGO Fast/Exact, ICORELEVEL=2, CH_LSPEC; no verified GW/BSE treatment',
    'references': ['https://vasp.at/wiki/index.php?title=CH_SIGMA&oldid=28409',
                   'https://vasp.at/wiki/Supercell_core-hole_calculations',
                   'https://feff.phys.washington.edu/FEFF/Docs/feff9/feff90/feff90_users_guide.pdf',
                   'https://data-analysis.synchrotron-soleil.fr/tmp/software/fdmnes/fdmnes_Linux/Doc/Manual_Eng.pdf'],
}


def read(path):
    b = path.read_bytes()
    return gzip.decompress(b).decode() if path.suffix == '.gz' else b.decode()


def numeric(path, skiprows=0):
    return np.loadtxt(io.StringIO(read(path)), skiprows=skiprows)


def deduplicate(x, y):
    d = pd.DataFrame({'energy_eV': x, 'intensity': y}).groupby('energy_eV', as_index=False).mean()
    return d.energy_eV.to_numpy(), d.intensity.to_numpy()


def lifetime(x, y, gamma=.775):
    # Recorded curves are absorption, not sticks: include integration weights.
    w = np.gradient(x)
    k = gamma/((x[:, None]-x[None, :])**2 + gamma**2)*w[None, :]
    return k@y/k.sum(axis=1)


def normalize(x, y):
    pre = (x >= 8974) & (x <= 8976)
    post = (x >= 9020) & (x <= 9028)
    if x.min() > 8974+.001 or x.max() < 9028-.001 or pre.sum() < 3 or post.sum() < 3:
        return None, {'status': 'insufficient_pre_or_post_range', 'pre_points': int(pre.sum()), 'post_points': int(post.sum())}
    bg = float(y[pre].mean())
    jump = float(y[post].mean()-bg)
    if jump <= 1e-14:
        return None, {'status': 'nonpositive_jump'}
    return (y-bg)/jump, {'status': 'local_edge_jump', 'background': bg, 'jump': jump,
                         'pre_points': int(pre.sum()), 'post_points': int(post.sum())}


def compare(curve, reference, shift):
    x, y = curve
    r, z = reference
    yy, zz = y, z
    lo, hi = max(8980., min(x+shift), min(r)), min(9020., max(x+shift), max(r))
    grid = np.arange(np.ceil(lo*5)/5, hi+1e-7, .2)
    if len(grid) < 20:
        return None
    a, b = np.interp(grid, x+shift, yy), np.interp(grid, r, zz)
    # RMSE is in experimental edge-jump units (not peak-max units).
    rmse = float(np.sqrt(np.mean((a-b)**2)))
    corr = float(np.corrcoef(a, b)[0, 1]) if min(np.std(a), np.std(b)) > 1e-10 else None
    # Only a well-contained global maximum is reported; tied/endpoint maxima are ambiguous.
    def peak(v):
        i = int(np.argmax(v))
        return float(grid[i]) if 2 < i < len(v)-3 and np.count_nonzero(np.isclose(v, v[i], rtol=1e-5, atol=1e-9)) == 1 else None
    pa, pb = peak(a), peak(b)
    return {'nrmse': rmse, 'correlation': corr, 'window_min_eV': float(grid[0]),
            'window_max_eV': float(grid[-1]), 'n_points': len(grid), 'peak_difference_eV': None if pa is None or pb is None else pa-pb,
            'edge_difference_eV': None, 'edge_status': 'not robust: short pre-edge support and multiple near-edge derivative maxima',
            'peak_status': 'global maximum on comparison window, not assigned transition; endpoint/ambiguous maxima omitted',
            'shift_eV': float(shift), 'energy_eV': grid, 'calculation': a, 'experiment': b}


def fit_shift(curve, reference, bounds):
    def score(s):
        m = compare(curve, reference, s)
        return m['nrmse'] if m else 1e9
    shifts = np.arange(bounds[0], bounds[1]+.125, .25)
    values = np.array([score(s) for s in shifts])
    i = int(values.argmin())
    if values[i] >= 1e9:
        return None
    lo, hi = max(bounds[0], shifts[i]-.3), min(bounds[1], shifts[i]+.3)
    f = minimize_scalar(score, bounds=(lo, hi), method='bounded', options={'xatol': .001})
    return float(f.x) if f.fun < values[i] else float(shifts[i])


def load_inputs():
    curves, rows = {}, []
    exp = pd.read_csv(RAW/'260324_export_XAS_Exp_ref_stds.dat', sep='\t')
    for i, material in enumerate(MATERIALS):
        for method in METHODS+['experiment']:
            row = {'material': material, 'formal_oxidation_state': [0, 1, 2][i], 'method': method,
                   'status': 'missing', 'source_file': '', 'sha256': '', 'numeric_columns_units': '',
                   'energy_convention': '', 'averaging': '', 'processing_present': '', 'settings': '', 'limitation': ''}
            path = None
            if method == 'experiment':
                path = RAW/'260324_export_XAS_Exp_ref_stds.dat'
                x, y = exp.iloc[:, 0].to_numpy(), exp.iloc[:, i+1].to_numpy()
                row.update(numeric_columns_units=f'column 0 Energy/eV; column {i+1} {exp.columns[i+1]}/arbitrary normalized absorption',
                           energy_convention='supplied photon energy, unchanged', averaging='merged traces per header; polarization not reported',
                           processing_present='merged and apparently normalized export; upstream background/resolution recipe not supplied',
                           limitation='Cu header identifies 100 nm Cu at -0.4 V, not a certified bulk metal foil; formal Cu(0) is nominal assignment')
            elif method.startswith('fdmnes'):
                # Oxide directory labels are reversed; classify from the deck and log.
                folder = ('fdm_green' if method == 'fdmnes_green' else 'fdm_fdm') if material == 'Cu' else ('fdm_fdm' if method == 'fdmnes_green' else 'fdm_green')
                base = RAW/material/folder
                deck = next(base.glob('*_in.txt'))
                text = read(deck)
                is_green = any(s.strip().lower() == 'green' for s in text.splitlines())
                assert is_green == (method == 'fdmnes_green')
                conv = list(base.glob('*conv.txt'))
                raw = [f for f in base.glob('*.txt') if f.name not in {deck.name, 'fdmfile.txt'} and 'conv' not in f.name]
                path = conv[0] if conv else raw[0]
                data = numeric(path, 1 if conv else 2)
                x, y = data[:, 0], data[:, -1]
                row.update(numeric_columns_units='Energy/eV, <xanes>/Mbarn per represented absorber (native code convention)',
                           energy_convention='Energpho photon-energy axis; native convolution site shifts retained',
                           averaging=('single saved CuO site 1, isotropic <xanes>; no complete two-site final convolution' if material == 'CuO' and not conv else '<xanes> isotropic; native represented absorber ensemble; CuO Green has 2-site convolution, native weights retained'),
                           processing_present='energy-dependent native convolution retained' if conv else 'raw absorption; no completed convolution in supplied log',
                           settings=f'Revision 11 April 2025; radius 7 Angstrom; SCF Perdew; input {deck.relative_to(ROOT)}',
                           limitation='oxide folder label reversed relative to input/log' if material != 'Cu' else '')
                if conv:
                    log = read(next(base.glob('*_bav.txt.gz')))
                    row['settings'] += '; ' + '; '.join(s.strip() for s in log.splitlines() if 'Gamma_max' in s or 'Gamma_hole' in s)
                else:
                    row['limitation'] += '; raw FDM run lacks completed convolution; add only constant core lifetime, final-state loss unavailable'
            elif method == 'feff':
                base = RAW/material/'feff'
                paths = list(base.glob('*xmu.dat'))
                if paths:
                    path = paths[0]
                    data = numeric(path)
                    x, y = data[:, 0], data[:, 3]
                    row.update(numeric_columns_units='column 0 omega/eV, column 3 mu/FEFF edge-normalized units; other columns preserved',
                               energy_convention='omega photon energy (not relative e column)', averaging='single symmetry-representative Cu absorber, unpolarized deck',
                               processing_present='intrinsic lifetime and Hedin-Lundqvist self energy already included; no repeat lifetime convolution',
                               settings='FEFF 9.6.4; COREHOLE RPA; SCF 7 Angstrom; FMS 9 Angstrom; header Gam_ch=1.729 eV, native convention preserved')
            else:
                base = RAW/material/('PBE' if method == 'vasp_pbe' else 'GW')/'XAS'
                path = base/'OUTCAR.gz'
                if path.is_file():
                    text = read(path)
                    pos = text.rfind('frequency dependent IMAGINARY DIELECTRIC FUNCTION')
                    data = []
                    if pos >= 0:
                        for line in text[pos:].splitlines()[3:]:
                            try:
                                v = [float(s) for s in line.split()]
                            except ValueError:
                                break
                            if len(v) != 7:
                                break
                            data.append(v)
                    if data:
                        data = np.array(data)
                        x, y = data[:, 0], data[:, 1:4].mean(axis=1)
                        row.update(numeric_columns_units='E/eV; arithmetic mean of X,Y,Z imaginary dielectric components/dimensionless; off-diagonals excluded',
                                   energy_convention='native VASP core-excitation energy; lacks absolute photon calibration',
                                   averaging='trace/3 dielectric tensor, single designated core-hole Cu atom; no multisite ensemble',
                                   processing_present='CH_SIGMA=0.1 eV built-in broadening; not a stick spectrum; no second Gaussian in primary view',
                                   settings=text.splitlines()[0].strip()+'; ICORELEVEL=2, CH_LSPEC; see preserved INCAR/KPOINTS/POSCAR',
                                   limitation='folder GW is an alias, no GW/BSE step verified; earlier notebook Cu_GW.dat has no verified lineage to these outputs')
                    else:
                        row['limitation'] = 'OUTCAR has no imaginary dielectric XAS block; not an available series'
                        path = None
                else:
                    row['limitation'] = 'no XAS OUTCAR spectrum supplied; empty Cu_PBE.dat is not a spectrum' if material == 'Cu' else 'no numeric XAS output supplied'
                    path = None
            if path is not None:
                b = path.read_bytes()
                uncompressed = gzip.decompress(b) if path.suffix == '.gz' else b
                row.update(status='available', source_file=path.relative_to(ROOT).as_posix(),
                           sha256=hashlib.sha256(uncompressed).hexdigest(), repository_sha256=hashlib.sha256(b).hexdigest())
                x, y = deduplicate(x, y)
                pd.DataFrame({'energy_eV': x, 'intensity': y}).to_csv(OUT/f'{material}_{method}_raw.csv', index=False)
                if method.startswith('vasp') or (method == 'fdmnes_fdm' and material == 'CuO'):
                    y = lifetime(x, y)
                curves[material, method] = x, y
                pd.DataFrame({'energy_eV': x, 'intensity': y}).to_csv(OUT/f'{material}_{method}_lifetime.csv', index=False)
            rows.append(row)
    return curves, rows


def run():
    OUT.mkdir(exist_ok=True)
    (ROOT/'figures').mkdir(exist_ok=True)
    curves, provenance = load_inputs()
    pd.DataFrame(provenance).to_csv(ROOT/'provenance.csv', index=False)
    matrix = pd.DataFrame(provenance).pivot(index='material', columns='method', values='status').reindex(MATERIALS)
    matrix.to_csv(ROOT/'missing_series_matrix.csv')
    normalization = []
    for (material, method), (x, y) in list(curves.items()):
        backend = method.split('_')[0]
        nominal = SETTINGS['normalization_nominal_offsets_eV'][method if method.startswith('vasp') else backend]
        yn, info = normalize(x+nominal, y)
        normalization.append({'material': material, 'method': method, 'nominal_shift_eV': nominal, 'actual_pre_min_eV': 8974-nominal, 'actual_pre_max_eV': 8976-nominal, 'actual_post_min_eV': 9020-nominal, 'actual_post_max_eV': 9028-nominal, **info})
        if yn is None:
            del curves[material, method]
        else:
            curves[material, method] = x, yn
    shifts = {}
    for backend, method in [('fdmnes', 'fdmnes_green'), ('feff', 'feff'), ('vasp', 'vasp_pbe')]:
        shifts[backend] = fit_shift(curves['Cu', method], curves['Cu', 'experiment'], SETTINGS['common_shift_bounds_eV'][backend])
    settings = {**SETTINGS, 'fitted_common_offsets_eV': shifts, 'process_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'raw_inventory_sha256': hashlib.sha256((ROOT/'source_inventory.json').read_bytes()).hexdigest(), 'environment': {'python': sys.version, 'platform': platform.platform(), 'numpy': np.__version__, 'pandas': pd.__version__}}
    (ROOT/'settings.json').write_text(json.dumps(settings, indent=2)+'\n')
    metrics, plots = [], []
    for material in MATERIALS:
        for method in METHODS:
            if (material, method) not in curves:
                continue
            backend = method.split('_')[0]
            common = shifts[backend]
            residual = fit_shift(curves[material, method], curves[material, 'experiment'], [common-20, common+20])
            for view, shift in [('native', 0.), ('common', common), ('shape_aligned', residual)]:
                if shift is None:
                    continue
                x, y = curves[material, method]
                yn = y
                if yn is not None:
                    pd.DataFrame({'energy_eV': x+shift, 'normalized_absorption': yn}).to_csv(OUT/f'{material}_{method}_{view}.csv', index=False)
                m = compare(curves[material, method], curves[material, 'experiment'], shift)
                if m:
                    for e, a, b in zip(m.pop('energy_eV'), m.pop('calculation'), m.pop('experiment')):
                        plots.append({'material': material, 'method': method, 'view': view, 'energy_eV': e, 'calculation': a, 'experiment': b})
                    metrics.append({'material': material, 'method': method, 'view': view, 'fitted_alignment': view != 'native', **m})
            # Optional notebook-width sensitivity: no changes to primary processing.
            x, y = curves[material, method]
            gx = np.arange(x.min(), x.max(), .1)
            gy = gaussian_filter1d(np.interp(gx, x, y), 10., mode='nearest', truncate=5.)
            m = compare((gx, gy), curves[material, 'experiment'], common)
            if m:
                for e, a, b in zip(m.pop('energy_eV'), m.pop('calculation'), m.pop('experiment')):
                    plots.append({'material': material, 'method': method, 'view': 'sigma1_sensitivity', 'energy_eV': e, 'calculation': a, 'experiment': b})
                metrics.append({'material': material, 'method': method, 'view': 'sigma1_sensitivity', 'fitted_alignment': True, **m})
    pd.DataFrame(normalization).to_csv(ROOT/'normalization.csv', index=False)
    pd.DataFrame(metrics).to_csv(ROOT/'metrics.csv', index=False)
    plotted = pd.DataFrame(plots)
    plotted.to_csv(ROOT/'plotted_arrays.csv', index=False)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    for view in ['native', 'common', 'shape_aligned', 'sigma1_sensitivity']:
        fig, axs = plt.subplots(3, 5, figsize=(14, 8), sharex=True, sharey=True)
        for i, material in enumerate(MATERIALS):
            for j, method in enumerate(METHODS):
                ax = axs[i, j]
                q = plotted[(plotted.material == material) & (plotted.method == method) & (plotted.view == view)]
                if not q.empty:
                    ax.plot(q.energy_eV-8979, q.experiment, c='black', lw=1.5, label='Experiment')
                    ax.plot(q.energy_eV-8979, q.calculation, c=COLORS[method], lw=1.5, label=LABELS[method])
                else:
                    ax.set_axis_off()  # Missing slots stay blank.
                if i == 0:
                    ax.set_title(LABELS[method], fontsize=12)
                if j == 0:
                    ax.set_ylabel(material+'\nLocal edge-jump absorption')
                if i == 2:
                    ax.set_xlabel('Photon energy − 8979 eV')
                ax.set_xlim(1, 41)
                ax.set_ylim(min(-.15, plotted.calculation.min()-.05), max(2.8, plotted.calculation.max()+.05))
        fig.suptitle({'native': 'Native axes (blank where comparison overlap is insufficient)', 'common': 'Cu-calibrated backend offsets shared across compounds', 'shape_aligned': 'Separately fitted shape alignment (chemical shifts removed)', 'sigma1_sensitivity': 'Sensitivity: additional Gaussian σ = 1 eV (not measured resolution)'}[view], fontsize=14)
        fig.text(.5, .015, 'Black: supplied experiment. Colors: calculations. *“GW” is a folder alias; no GW/BSE step verified. CuO FDM: core lifetime only, no recorded final-state convolution.', ha='center', fontsize=10)
        fig.tight_layout(rect=(0, .045, 1, .95))
        for ext in ['pdf', 'svg', 'png']:
            fig.savefig(ROOT/'figures'/f'reference_grid_{view}.{ext}', dpi=450)
        plt.close(fig)
    print(matrix.to_string())
    print(pd.DataFrame(metrics).query('view == "common"')[['material', 'method', 'nrmse', 'correlation', 'shift_eV']].to_string(index=False))


if __name__ == '__main__':
    run()
