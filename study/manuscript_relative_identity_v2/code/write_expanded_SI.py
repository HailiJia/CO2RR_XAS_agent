"""Generate current-cohort SI from pinned data and independently checked predictions."""
from pathlib import Path
import json
import pandas as pd
P=Path(__file__).resolve().parents[1];D=P/'figure_data';N=D/'dataset_20261029'
esc=lambda x:str(x).replace('_',r'\_').replace('&',r'\&')
def table(headers,rows,caption):
    return '\\begin{table}[!htbp]\\centering\\small\n\\caption{'+caption+'}\n\\begin{tabular}{'+'l'+'r'*(len(headers)-1)+'}\\toprule\n'+' & '.join(headers)+r'\\\midrule'+'\n'+'\n'.join(' & '.join(esc(x) for x in row)+r'\\' for row in rows)+'\n\\bottomrule\\end{tabular}\\end{table}\n'
s=r'''\clearpage
\section{Expanded spectral cohort and source correspondence}
The expanded record export contains 146 versioned records: 145 converted or merged supplied measurements and one reconstructed clean Cu(100) reference. The source audit verifies numerical correspondence for 553 retained native series and element-aware correspondence between each retained layer geometry and its FDMNES input. The record export is larger than each model cohort because incomplete required layers, uncertain lineage and unsupported molecular identities are treated separately. Record labels follow molecular connectivity in the XAS-used structure, including the distinction between H--C and H--O bonding. A root relaxed structure is not substituted for a different archived geometry that generated an existing spectrum.

Of 137 source-scaled configurations, 135 are adsorbate bearing and two are clean references. Figure 2b includes 118 adsorbed Cu--Au and 17 adsorbed Cu(100) configurations. Closed-label fitting instead uses 117 adsorbed Cu--Au configurations plus the existing clean Cu--Au negative, and 12 supported-label Cu(100) configurations. Thus the two Cu--Au counts of 118 have different membership. One Cu--Au CO/water case and five Cu(100) water/carbonate cases remain audited interference queries. The reconstructed Cu(100) reference is an input, not a thirteenth adsorbate example.

The OH configuration with reliable layers 4+5 is retained only for those layers; incompatible lower layers are removed from its versioned measurement. A roughened OH configuration remains excluded because its required physical-layer partition is ambiguous. The H root/XAS-copy discrepancy retains only the previously established exact archived lineage. The analogous unresolved CO/H discrepancy is excluded. Complete numeric output alone is insufficient when its geometry lineage cannot be established. These exclusions are recorded in the accompanying audit and required-layer tables.

The clean Cu(100) reconstruction uses three nonoverlapping symmetry representatives for four atoms per layer. A cell-metric-preserving x/y reflection maps the multiplicity-two representative to its partner. That multiplicity is retained, unlike full explicit lists that already enumerate every physical site. The symmetry check rejects arbitrary x/y swaps in rectangular cells. Convolution of saved electronic spectra is a local reconstruction, not a new electronic calculation or proof of ionic convergence.

\section{Relative differential intensity and experimental inputs}
For a reaction/reference pair, $d=(\mu_a-\mu_r)/S_r$ retains the signed relative response. If both spectra and the reference scale are multiplied by the same unknown factor $c$, then $(c\mu_a-c\mu_r)/(cS_r)=d$. Absolute cross-sections per Cu atom are consequently not required for adsorbate identity inference. Either a consistently processed pair or a precomputed normalized signed difference may be used, provided its convention is compatible with the model. Energy registration, sign, background treatment, reference state and noise remain relevant. This statement does not imply experimental accuracy for the current simulation-trained models.

The simulation grid is 8974--9029 eV at 0.2 eV increments. Its reference scale is the difference between clean-spectrum averages over 9018--9025 and 8974--8977 eV. This short-window near-edge proxy must not be interpreted as a demonstrated full experimental post-edge normalization. Established work on near-edge normalization motivates checking background and energy-range sensitivity.\cite{weng2005,bugarin2024} No independent peak alignment or per-difference minimum, area or norm scaling is applied. Source-weighted physical spectra establish comparable simulated absorber averages before this relative scaling. Absolute loading additionally depends on the mapping between the simulated surface ensemble and the measured active fraction. Identity, relative change and calibrated absolute coverage therefore have distinct input requirements.

For 37 adsorbed Cu--Au cases the initial metal model and constraints match the clean model. The six single-species groups contain CO 8, COH 3, COOH 3, H 11, O 6 and OH 2 configurations, totaling 33. Four matched mixtures are not used in Figure 4e. Eighty further adsorbed cases have weaker reference correspondence; the clean negative is self-referenced. These compatibility classes are saved by configuration. Only one reference family exists per substrate domain, so new-parent generalization is unmeasured.

\section{Current grouped identity and loading evaluation}
All spectra from a configuration remain in the same validation group. The compact comparison uses five outer folds and three grouped inner folds, selecting among Combined or DCT16 features with regularized independent logistic classifiers ($C=0.1,1$), or the specified tree classifier. Feature scaling and thresholds are training-only. This six-candidate single-seed study is separate from the three-seed fixed-Combined training-size control below. No model is selected by outer-test performance. Missing-positive training labels and domain-specific supports accompany the predictions.

The native-normalized representation starts from native convolution weights and applies the established minimum-subtracted unit-area transformation. Corrected raw and differential representations use reconstructed source weights and a shared clean-derived intensity scale. The comparison therefore varies both weighting and normalization. A corrected-and-normalized control is required to isolate them. Unit-area shape differences between native and corrected arrays are detectable in 8 of 118 Cu--Au and 6 of 12 Cu(100) configurations, with maximum integrated absolute differences of approximately 0.00185 and 0.00790. These diagnostics do not establish that the small shape changes caused the score differences.

'''
m=pd.read_csv(N/'ml/metrics.csv').query('task=="identity" and stage=="compact_nested"')
rows=[]
for r in m.itertuples():rows.append([('Cu--Au' if r.cohort=='refresh_CuAu' else 'Joint')+' / '+{'normalized_raw':'native normalized','physical_raw':'corrected raw','delta':r'$\Delta\mu$'}[r.representation],r.n,f'{r.micro_f1:.4f}',f'{r.macro_f1:.4f}',f'{r.exact_set_accuracy:.4f}'])
s+=table(['Representation','$N$','Micro F1','Macro F1','Exact set'],rows,'Current compact identity comparison. Macro F1 uses all eleven declared labels. The normalized-raw control differs in weighting as well as normalization.')
d=pd.read_csv(D/'Figure4b_current_per_species.csv');old=pd.read_csv(N/'label_support_comparison.csv').set_index('species')
s+=table(['Species','Old positive','New positive','New pure','Differential F1'],[[r.species,int(old.loc[r.species,'old99_positive_configurations']),r.support,int(old.loc[r.species,'new118_single_species_positive_configurations']),f'{r.F1:.4f}'] for r in d.itertuples()],'Positive configuration support and Cu--Au-only differential identity performance. Mixtures contribute to each constituent label; pure means single species, possibly several molecules.')
s+=r'''
The Cu--Au-only differential model gives micro F1 0.7668 on 101 single-species configurations and 0.5862 on 16 mixtures, with one exact mixture set. The joint model gives 0.7979 and 0.6333 on those same strata, with two exact mixture sets. The one clean negative is incorrectly assigned COOH and OCCOH by the Cu--Au-only differential model but correctly assigned the empty set by the joint model. This is a scoped null-control result, not an estimated experimental false-positive rate. A label with one positive has no positive training example in the fold that holds it out.

The target for loading is CO per top-surface metal atom. Cu-normalized loading and CO/nm$^2$ remain separate deterministic conversions; they are not inputs to the regressor. Only CO-only configurations train the coverage models. Results on CO-containing mixtures assess interference, not independently validated general mixture inference. Unknown experimental surface fraction and gain can confound absolute loading even when identity-related spectral shape is retained.
'''
from sklearn.metrics import mean_absolute_error,r2_score
import numpy as np
tr=json.loads((N/'training_manifest_compact.json').read_text());rr=[]
for rep in ['normalized_raw','physical_raw','delta']:
 z=np.load(N/f'ml/CO_coverage_refresh_joint_{rep}_compact_nested_20261027.npz')
 for dom in ['CuAu','Cu100']:
  ix=np.array([tr[i]['domain']==dom for i in z['pair_indices']]);rr.append([{'normalized_raw':'Native normalized','physical_raw':'Corrected raw','delta':r'$\Delta\mu$'}[rep],dom,int(ix.sum()),f"{mean_absolute_error(z['truth'][ix],z['prediction'][ix]):.5f}",f"{r2_score(z['truth'][ix],z['prediction'][ix]):.4f}"])
s+=table(['Input','Domain','$N$','MAE','$R^2$'],rr,'Joint CO-only loading results separated by substrate domain. Four Cu(100) cases cannot establish facet transfer; pooled correlation can reflect different domain loading ranges.')
s+=r'''
\section{Paired training-set enlargement and threshold sensitivity}
The control retains the original 99 Cu--Au outer-fold assignments and labels, adding 19 geometry-grouped configurations to the training pool. Original inner-training assignments remain paired, while additions receive deterministic geometry-hash folds. Both cohorts use the same current arrays and dependencies. All 198 original top-two source files are byte-identical; interpolation-grid roundoff relative to the historical model matrices is documented separately. The fixed recipe uses population-weighted top-two normalized-raw Combined features and 64 independent balanced ExtraTrees, leaf size one and square-root feature sampling, with three predeclared seeds. Depth, features and seed are not selected by held-out scores. These tests reuse established configurations and are not prospective confirmation.

Transferred thresholds are learned from the original training partitions; retuned thresholds are learned inside the enlarged training partitions. Fixed 0.5 results are also retained in the source study. Paired comparisons always use the same 99 queries, rather than comparing the full enlarged test population with the old one. Configuration bootstrap intervals condition on saved fitted predictions and omit refit, new-parent and experimental uncertainty.
'''
q=pd.read_csv(N/'review_paired_identity.csv');rr=[]
for r in q.itertuples():rr.append([str(r.seed),{'baseline99':'99 training','transferred_baseline_threshold':'118 / transferred','inner_retuned_threshold':'118 / retuned'}[r.stage],f'{r.micro_F1:.4f}',f'{r.macro_F1:.4f}',f'{r.exact_set:.4f}'])
s+=table(['Seed','Training / threshold','Micro F1','Macro F1','Exact set'],rr,'Fixed-recipe raw-XAS predictions on the same 99 configurations. No automatic depth selector is used.')
s+=r'''
Mean micro F1 changes from 0.7434 to 0.7519 with inner retuning, and to 0.7608 with transferred thresholds. Mean macro F1 changes from 0.5795 to 0.5812 with retuning; mean exact-set accuracy changes from 0.5993 to 0.6229. Each seed's paired retuned micro-F1 interval crosses zero. The same-25 CO-only comparison reduces MAE from 0.013971 to 0.011444, with a conditional difference interval of $[-0.006057,+0.000344]$. Both are encouraging point changes without established improvement. The 28-case MAE has a different query set. A corresponding frozen-recipe differential learning curve remains unperformed.

\section{Chemical interpretation and reproducible figure sources}
The retained OCCO/2H$_2$O spectral case is assigned 2CO plus two water molecules from its actual calculation geometry. It is an unsupported-label control, not a positive bonded-OCCO training example. Earlier unfinished checkpoints are historical snapshots. No retained matched OCCO/2CO spectral pair is available to isolate C--C-bond formation.

The accompanying code recomputes 25 saved prediction files and all 28 reference metric triplets, verifies array/manifest/fold correspondence and signed subtraction, and rebuilds Figures 2--5 without retraining. Current prediction arrays, compact manifests, source audit tables and model protocols are pinned to repository commit 798e2c81b7a9afa684dc2a5625607ec866430163 in the package provenance. The repository reports 153 focused tests and 110 saved-model replays at that commit; the manuscript editing pass independently checks stored predictions rather than repeating the full repository suite or loading all training artifacts. Historical studies remain separately identified.
'''
(P/'si').mkdir(exist_ok=True);(P/'si/expanded_differential_study.tex').write_text(s)
print('Expanded SI generated')
