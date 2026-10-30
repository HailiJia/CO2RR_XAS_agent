"""Frozen prediction diagnostics, independent model replay and review figure."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from tools.identity_comparison import transform,score_model
from tools.spectral_multilabel import SPECIES,f1_metrics
from tools.coverage_study import dump,sha
from tools.differential_route import infer
B=Path('study/relative_identity_guarded_v2');O=B/'results';D=Path('study/dataset_20261029');z=np.load(O/'inputs.npz');t=np.load(O/'targets_folds.npz');r=json.load(open(O/'source_manifest.json'));y=t['truth'];e=z['energy'];replay=[]
for p in sorted((O/'models').glob('*.joblib')):
 m=joblib.load(p);rep=m['representation'];ids=m['test_indices'];x=z['raw_native'] if rep=='native_normalized' else z['raw_physical'] if rep in ['corrected_normalized','common_raw'] else z['delta'];name=p.stem
 if name.startswith('augment_') and rep in ['relative_delta','signed_shape']:
  x=(z['raw_physical']-z['clean']).copy();margin=(e<e[0]+1)|(e>e[-1]-1);x[:,margin]=z['delta'][:,margin]
 s=score_model(m['model'],transform(x[ids],e,rep,m['feature'],m['floor']))
 if name.startswith('nested_') or name.startswith('fixed_rep_'):target=name.rsplit('_',1)[0]
 elif name.startswith('size_'):
  target=name.rsplit('_',1)[0]+'_common'
 elif name.startswith('curve_'):target=name.rsplit('_',1)[0]
 elif name.startswith('domain_'):target=name
 elif name.startswith('augment_'):target=name.rsplit('_',1)[0]+'_clean'
 q=np.load(O/(target+'.npz'));lookup={int(v):j for j,v in enumerate(q['indices'])};expected=q['score'][[lookup[int(v)] for v in ids]];np.testing.assert_array_equal(s,expected);assert not set(m['train_groups'])&set(m['test_groups']);replay.append({'model':str(p.relative_to(O)),'sha256':sha(p),'exact_heldout_scores':True,'train_test_groups_disjoint':True,'expected':target+'.npz'})
dump(B/'all_model_replay.json',replay)
# Fixed Combined/.5 comparisons isolate representation without changing selected features.
ci=[]
for cohort in ['expanded118','joint130']:
 for fam in ['logistic','trees']:
  for a,b in [('native_normalized','corrected_normalized'),('corrected_normalized','common_raw'),('common_raw','relative_delta'),('relative_delta','signed_shape')]:
   aa=np.load(O/f'fixed_rep_{cohort}_{a}_{fam}.npz');bb=np.load(O/f'fixed_rep_{cohort}_{b}_{fam}.npz');yy=aa['truth'];pa=aa['prediction'];pb=bb['prediction'];rng=np.random.default_rng(789);v=[]
   for _ in range(1000):
    ii=rng.integers(0,len(yy),len(yy));v.append(f1_metrics(yy[ii],pb[ii])['micro_f1']-f1_metrics(yy[ii],pa[ii])['micro_f1'])
   ci.append(dict(cohort=cohort,estimator=fam,baseline=a,intervention=b,delta_micro=f1_metrics(yy,pb)['micro_f1']-f1_metrics(yy,pa)['micro_f1'],lower=np.quantile(v,.025),upper=np.quantile(v,.975),uncertainty='conditional on fixed predictions, not model-refit variance'))
pd.DataFrame(ci).to_csv(B/'fixed_representation_intervals.csv',index=False)
# Actual unsupported records: predictions are review-only, never scored as eleven-label truth.
sm=json.load(open(D/'scale/configuration_manifest.json'));sz=np.load(D/'scale/per_Cu_spectra.npz');unsupported=[]
for row in json.load(open(D/'training_exclusions.json')):
 i=next(j for j,a in enumerate(sm) if a['configuration']==row['configuration']);ref=next(j for j,a in enumerate(sm) if a['domain']==sm[i]['domain'] and not a['identity_labels']);clean=sz['physical'][ref];scale=clean[(e>=9018)&(e<=9025)].mean()-clean[(e>=8974)&(e<=8977)].mean();delta=(sz['physical'][i]-clean)/scale;pred=[]
 for f in range(5):
  m=joblib.load(O/f'models/nested_joint130_signed_shape_logistic_{f}.joblib');s=score_model(m['model'],transform(delta[None],e,'signed_shape',m['feature'],m['floor']))[0];pred.append({'fold_model':f,'uncalibrated_scores':dict(zip(SPECIES,s.tolist())),'predicted_known_set':[sp for sp,v in zip(SPECIES,s>=m['thresholds']) if v]})
 unsupported.append({**row,'observed_connectivity':sm[i]['connectivity'],'status':'unsupported chemistry; mandatory review, not closed-label accuracy','predictions':pred})
dump(B/'unsupported_queries.json',unsupported)
# Nearest training examples for omitted CO/CHO/COH under the fixed relative-delta anchor.
q=np.load(O/'fixed_rep_expanded118_relative_delta_trees.npz');near=[]
for j,i in enumerate(q['indices']):
 omitted=[s for s in ['CO','CHO','COH'] if y[i,SPECIES.index(s)] and not q['prediction'][j,SPECIES.index(s)]]
 if not omitted:continue
 f=int(t['fold'][i]);m=joblib.load(O/f'models/fixed_rep_expanded118_relative_delta_trees_{f}.joblib');tr=m['train_indices'];x=z['delta'];norm=np.linalg.norm(x,axis=1);similar=x[tr]@x[i]/np.maximum(norm[tr]*norm[i],1e-30)
 for k in np.argsort(similar)[-3:][::-1]:near.append({'query':r[i]['configuration'],'omitted':';'.join(omitted),'training_configuration':r[tr[k]]['configuration'],'training_labels':';'.join(r[tr[k]]['identity_labels']),'cosine_signed_delta':similar[k],'query_CO_loading':r[i]['coverage_ML'],'training_CO_loading':r[tr[k]]['coverage_ML'],'training_fold':int(t['fold'][tr[k]]),'query_fold':f})
pd.DataFrame(near).to_csv(B/'omitted_species_neighbors.csv',index=False)
# Predeclared categories: observed failures guide proposals, not new performance claims.
priority=[{'rank':1,'proposal':'Independent pure CHO and COCHO environments','reason':'CHO9 positives but only2 pure; COCHO1 positive cannot support held-out positive learning','controls':'verify H-C vs H-O and isolated CO+H; varied independent parents'}, {'rank':2,'proposal':'Matched CO+H versus CHO/COH','reason':'omitted coadsorbates and hydrogenation confusion','controls':'same parent, loading, site, setting; relaxed connectivity'}, {'rank':3,'proposal':'Mixture composition at fixed total loading','reason':'composition/coverage confounding','controls':'CO/H/OH sweeps and clean control, no synthetic positives'}, {'rank':4,'proposal':'Independent clean/covered parents and batches','reason':'one reference family per domain cannot establish new-parent transfer','controls':'independent CuAu and Cu100 controls; replicated measurement references'}, {'rank':5,'proposal':'Supported CO/OCHO coordination and proximity sweeps','reason':'interface proximity differs from hollow composition; CuAu2/Au3 support absent','controls':'atop/bridge/hollow explicitly separated; no pure-Cu-to-Cu3 relabel'}, {'rank':6,'proposal':'More Cu100 loading/motif controls','reason':'four CO cases confound loading with motif','controls':'separate coverage and motif variation; no pooled-R2 transfer claim'}]
dump(B/'proposed_additions.json',priority)
# Review PNGs; exact arrays stay in results/*.npz and CSVs.
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,ax=plt.subplots(1,3,figsize=(12,3.7));d=pd.read_csv(O/'fixed_representation_metrics.csv');colors=['#4477aa','#66aacc','#228833','#aa3377','#ccbb44']
for k,fam in enumerate(['logistic','trees']):
 dd=d[(d.cohort=='expanded118')&(d.estimator==fam)];ax[k].barh(range(len(dd)),dd.micro_f1,color=colors);ax[k].set(yticks=range(len(dd)),yticklabels=dd.representation,xlim=(0,1),xlabel='Micro F1',title=f'Fixed Combined / {fam} / 118');ax[k].tick_params(labelsize=8)
md=pd.read_csv(O/'metrics.csv');c=md[(md.protocol=='learning_curve')&(md.subset=='all')];a=c.groupby('fraction').micro_f1.agg(['mean','std']);ax[2].errorbar(a.index,a['mean'],yerr=a['std'],marker='o',color='#228833');ax[2].set(xlabel='Fraction of outer-training groups',ylabel='Micro F1 on same old99 queries',title='Delta / Combined / trees / .5');ax[2].text(.05,.03,'Bars: refit/seed SD (3 seeds)',transform=ax[2].transAxes,fontsize=8);fig.tight_layout();fig.savefig(B/'controlled_comparisons.png',dpi=300);plt.close(fig)
print('Exact saved-model replays',len(replay),'unsupported records',len(unsupported))
