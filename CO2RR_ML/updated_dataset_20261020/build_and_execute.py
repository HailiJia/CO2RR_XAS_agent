from pathlib import Path
import os,sys
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
import argparse
BASE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('--data-root',required=True);parser.add_argument('--output-dir',required=True);parser.add_argument('--split-manifest',default=str(BASE/'cv_manifest.csv'));parser.add_argument('--audit-only',action='store_true');parser.add_argument('--build-only',action='store_true');opts=parser.parse_args()
DEST=Path(opts.output_dir).resolve();DEST.mkdir(parents=True,exist_ok=True)
if DEST==BASE:raise ValueError('Archived notebook cannot be overwritten; choose a new output directory')
os.environ.update(CO2RR_DATA_ROOT=str(Path(opts.data_root).resolve()),CO2RR_OUTPUT_DIR=str(DEST),CO2RR_SPLIT_MANIFEST=str(Path(opts.split_manifest).resolve()))
os.chdir(BASE)
source=(BASE/'analysis.py').read_text()
parts=source.split('# %% ')
nb=nbformat.v4.new_notebook();cells=[]
cells.append(nbformat.v4.new_markdown_cell('''# Updated Cu K-edge benchmark: audited adsorbate identities and depth sensitivity
This notebook contains the complete current audit, optional record-correction procedure, inventory, spectral preparation, model fitting, nested cross-validation, five requested figure sets, and a short bullet-point interpretation. **Targets are adsorbate identities only.** Molecular counts and geometry descriptors are audited but never used as predictors or prediction targets.

**Setup:** Python 3.11+ with `numpy pandas scipy scikit-learn matplotlib nbformat ipykernel ipython`. Install using `python -m pip install numpy pandas scipy scikit-learn matplotlib nbformat ipykernel ipython`. Exact tested versions are saved in `environment.json`. Set CO2RR_DATA_ROOT, CO2RR_OUTPUT_DIR and optionally CO2RR_SPLIT_MANIFEST, or pass CLI flags to the builder. Use a new output directory; archived results remain immutable. Matching caches avoid refitting. Delete `cache/` after changing feature or model implementations to force fresh fits.

**Outputs:** `figures/01_…` through `05_…` are saved as PNG and PDF. CSV audits, per-species metrics, all feature/model/depth comparisons, fold assignments and held-out predictions are saved alongside the notebook. `ANALYSIS.md` provides the requested simple bullets. New figures use the current cohort only; prior scores are not treated as directly comparable independent tests.'''))
heads={
 'RECORD CORRECTION PROCEDURE':'''## Check and correct the new count schema
`adsorbate_counts` and `adsorbate_identities` are checked against periodic nonmetal connectivity in the simulated relaxed CONTCAR. COH/CHO and OCCOH/COCHO are separated by H–O versus H–C connectivity. Cutoffs are tested at ±10%. Every geometry entry must match its component formula before its species label can be corrected. Seven regenerated records were backed up and repaired before this run; the default below is read-only and confirms consistency. Set `apply=True` only when intentionally applying newly found corrections; backups precede writes. Measurements are preserved.''',
 'INVENTORY':'''## Inventory all structures, including unfinished configurations
Discover source folders containing POSCAR/CONTCAR, plus record-associated source folders. Exclude copied calculation-level structures as separate configurations. Use readable CONTCAR, falling back to POSCAR; include the standalone CuO structural reference separately. Deduplicate physical coordinates/atom identities while retaining every alias in CSV.

Count mixtures in mutually exclusive identity-combination groups (CO + H, CHO + CO + H, …), ignoring multiplicity as a new identity class. Also show a nonexclusive per-species presence panel. Source labels can change from POSCAR to CONTCAR after relaxation; that is recorded, not forced to agree.''',
 'RECORD AUDIT':'''## Audit completed records and layer populations
Only `isaac_record_layers.json` qualifies a completed record; all ML spectra are read from `measurement.series`. Check finite/nonconstant data, energy grids, local broadened outputs, identities/counts/site formulas, absorber indices/populations and FDMNES input coordinates. Duplicates are removed before splitting; conflicting duplicates are excluded.

Layer 5 is verified as the highest Cu layer with adsorbates above it. Each usable spectrum is a broadened multi-absorber layer ensemble. Weights are represented Cu populations. Compatible raw layer spectra are averaged before per-sample normalization. The current paired benchmark requires the four layers on an identical energy grid. Two recovered CO cases have subset recalculation decks but full 36-site convolution logs; site counts, exact source-file coverage, equal multiplicities and site E_cut values verify the full-layer outputs. One remaining record has an incomplete layer-5 grid and is excluded.

Source and simulated coordinates need not be byte-identical if atom counts and relaxed identities agree. The spectra are checked against the actual FDMNES input CONTCAR/decks, and any source-coordinate difference is flagged in CSV. Counts, site geometry, names and file paths are never ML predictors.''',
 'FOLDS AND FEATURES':'''## Configuration-level folds and fixed spectral features
Preserve the original outer folds for existing physical groups; assign new unique configurations using labels/count balance only. Copied folders and all layers of a structure stay in the same fold. All representations use these identical outer folds.

Compare layers 5/4/3/2, top-two/top-three and four-layer population-weighted raw averages. Features: minimum-subtracted unit-area shape, CDF, Savitzky–Golay first derivative, combined shape/CDF/first-and-second derivatives, and DCT24. Tree workflows also test top-layer/depth-contrast channels and multiscale derivatives. Signed contrasts are not normalized by their own area. Fixed transforms are per sample; learned scaling is fitted on training data only.''',
 'TRAINING':'''## Fit models and select settings inside training folds
Models: balanced logistic regression (C=1); balanced RBF SVM (C=3); joint random forest (96 trees, leaf size 2); joint extra-trees (96 trees); independent binary extra-trees (64 trees/species). SVM sigmoid decision scores are not calibrated probabilities.

For every fixed feature/model/depth entry, threshold selection uses three inner folds shared across the representations. Test predictions come from refitting on the full outer training partition. The primary procedure, specified before this run's results, selects among independent-species models/features/depths. Broader and per-family/per-depth procedures are also reported.

A shared threshold is compared with restrained species adjustments only when there are ≥5 training positives and ≥10 negatives; maximum adjustment is 0.075. Rare labels retain the shared threshold. Hyperparameters are intentionally limited after prior broad searches showed selection variance on this small cohort. No training/test score is used to define chemical labels.''',
 'RESULTS AND FIVE FIGURES':'''## Executed results and the five requested figures
1. All unique structures by exact species combination and individual species presence, including unfinished structures.
2. Eligible/excluded structures and actual training/testing counts in each outer fold.
3. Depth sensitivity on identical held-out configurations.
4. A single per-species F1 bar for the provisional best configuration, labeled with positive structure counts; full precision/recall metrics remain in CSV. Spectral summaries show energy-resolved present-minus-absent contrasts, strongest difference energies, and separate species mean/variability panels.
5. Feature/model comparison at both the top layer and top-two average, with thresholds selected inside training.

Metrics pool held-out predictions. Each multilabel species has its own binary confusion matrix. Macro F1 includes all observed species, including rare failures. Bootstrap intervals are conditional on fitted CV models and omit retraining/split uncertainty. A best fixed-entry score chosen after inspecting outer results is exploratory.''',
 'SIMPLE BULLET ANALYSIS AND VALIDATION':'''## Simple bullet-point interpretation and checks
The interpretation below is generated from the executed results, including what was tried, the primary workflow, observed rankings, why results remain limited and how to improve the data. It also verifies record/fold uniqueness and that every requested figure was saved.'''
}
cells.append(nbformat.v4.new_code_cell(parts[0]+"\nprint('Environment:',sys.version.split()[0], 'numpy',np.__version__,'sklearn',sklearn.__version__)"))
for part in parts[1:]:
    title,code=part.split('\n',1)
    code=code
    cells.append(nbformat.v4.new_markdown_cell(heads[title.strip()]))
    cells.append(nbformat.v4.new_code_cell(code))
nb.cells=cells;nb.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python'}}
p=DEST/'adsorbate_identity_depth_updated.ipynb';nbformat.write(nb,p)
if opts.build_only:raise SystemExit(0)
shell=InteractiveShell.instance()
import matplotlib
matplotlib.use('module://matplotlib_inline.backend_inline')
from matplotlib_inline.backend_inline import configure_inline_support
configure_inline_support(shell,'module://matplotlib_inline.backend_inline')
count=0
for cell in nb.cells:
    if cell.cell_type!='code':continue
    count+=1;print('Executing cell',count,flush=True)
    with capture_output(stdout=True,stderr=True,display=True) as cap:result=shell.run_cell(cell.source,store_history=True)
    cell.execution_count=count;cell.outputs=[]
    if cap.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=cap.stdout))
    if cap.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=cap.stderr))
    for o in cap.outputs:cell.outputs.append(nbformat.v4.new_output('display_data',data=o.data,metadata=o.metadata))
    err=result.error_before_exec or result.error_in_exec
    if err and not (isinstance(err,SystemExit) and err.code==0 and opts.audit_only):cell.outputs.append(nbformat.v4.new_output('error',ename=type(err).__name__,evalue=str(err),traceback=[]))
    nbformat.write(nb,p)
    if isinstance(err,SystemExit) and err.code==0 and opts.audit_only:break
    if err:raise err
nbformat.validate(nb)
print('Saved executed notebook:',p)
