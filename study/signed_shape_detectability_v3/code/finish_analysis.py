"""Saved-prediction diagnostics and PNG-only analysis figures, with exact table inputs."""
from pathlib import Path
import json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support
from tools.spectral_multilabel import SPECIES
B=Path('study/signed_shape_detectability_v3');F=B/'figures';F.mkdir(exist_ok=True);I=Path('study/relative_identity_guarded_v2/results');t=np.load(I/'targets_folds.npz');r=json.load(open(I/'source_manifest.json'));families=np.array([json.load(open(I/'family_map.json'))['families'][x['configuration']] for x in r]);desc=pd.read_csv(B/'configuration_descriptors.csv');yy=t['truth'];mix=yy.sum(axis=1)>1
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
def metric(y,p,w):
 tp=(y*p*w[:,None]).sum();den=((y+p)*w[:,None]).sum();pr,re,f,s=precision_recall_fscore_support(y,p,sample_weight=w,average=None,zero_division=0);return dict(micro_f1=2*tp/den if den else 0,macro_f1=float(f.mean()),exact_set_accuracy=float(np.average(np.all(y==p,axis=1),weights=w)),false_positives=float(((1-y)*p*w[:,None]).sum()),omitted_labels=float((y*(1-p)*w[:,None]).sum()))
def weight(ids):return np.array([1/sum(families[ids]==f) for f in families[ids]])
# Conditional paired group intervals for the already-fixed joint representation contrast.
a=np.load(I/'fixed_rep_joint130_relative_delta_trees.npz')['prediction'];b=np.load(I/'fixed_rep_joint130_signed_shape_trees.npz')['prediction'];rows=[]
for dom in ['all','CuAu','Cu100']:
 ids=np.arange(len(r)) if dom=='all' else np.flatnonzero(t['domain']==dom);w=weight(ids);ma=metric(yy[ids],a[ids],w);mb=metric(yy[ids],b[ids],w);groups=sorted(set(families[ids]));rng=np.random.default_rng(20261030);boot=[]
 for _ in range(2000):
  sampled=rng.choice(groups,len(groups),replace=True);ix=np.concatenate([ids[families[ids]==g] for g in sampled]);ww=np.concatenate([np.full(sum(families[ids]==g),1/sum(families[ids]==g)) for g in sampled]);av=metric(yy[ix],a[ix],ww);bv=metric(yy[ix],b[ix],ww);boot.append([bv[k]-av[k] for k in ['micro_f1','macro_f1','exact_set_accuracy']])
 for j,k in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):rows.append(dict(domain=dom,metric=k,relative_delta=ma[k],signed_shape=mb[k],difference=mb[k]-ma[k],lower=np.quantile(boot,.025,axis=0)[j],upper=np.quantile(boot,.975,axis=0)[j],scope='conditional fixed-model group bootstrap; no refit or unseen-parent interval'))
pd.DataFrame(rows).to_csv(B/'paired_representation_intervals.csv',index=False)
# The one bounded dependence control and false-positive costs.
q=np.load(B/'label_dependence/predictions.npz');out=[]
for key in ['independent','inner_selected','fixed_strength_sensitivity']:
 p=q[key]
 for dom,mask in [('all',np.ones(len(r),bool)),('CuAu',t['domain']=='CuAu'),('Cu100',t['domain']=='Cu100'),('CuAu_mixtures',(t['domain']=='CuAu')&mix)]:
  ids=np.flatnonzero(mask)
  for weighting,w in [('record',np.ones(len(ids))),('near_group',weight(ids))]:out.append(dict(decoder=key,subset=dom,weighting=weighting,n_records=len(ids),n_groups=len(set(families[ids])),complete_sets=int(np.all(yy[ids]==p[ids],axis=1).sum()),**metric(yy[ids],p[ids],w)))
pd.DataFrame(out).to_csv(B/'label_dependence_metrics.csv',index=False)
# Fold training support is explicit: no positives means no learned class recognition.
miss=[]
import joblib
for f in range(5):
 m=joblib.load(I/'models'/f'fixed_rep_joint130_signed_shape_trees_{f}.joblib');tr=m['train_indices'];te=m['test_indices']
 for j,sp in enumerate(SPECIES):miss.append(dict(fold=f,species=sp,training_positive_records=int(yy[tr,j].sum()),training_positive_groups=len(set(families[tr[yy[tr,j]>0]])),test_positive_records=int(yy[te,j].sum()),test_positive_groups=len(set(families[te[yy[te,j]>0]])),missing_training_positive=not bool(yy[tr,j].any())))
pd.DataFrame(miss).to_csv(B/'joint_fold_label_support.csv',index=False)
# All mixture omissions and class-confusion paths retained, not just aggregate scores.
conf=[]
for rep in ['relative_delta','signed_shape']:
 p=np.load(I/f'fixed_rep_joint130_{rep}_trees.npz')['prediction']
 for j,sp in enumerate(['CO','CHO','COH']):
  jj=SPECIES.index(sp)
  for i in np.flatnonzero(yy[:,jj]):conf.append(dict(representation=rep,configuration=r[i]['configuration'],near_group=families[i],true_set=';'.join(np.array(SPECIES)[yy[i]>0]),prediction=';'.join(np.array(SPECIES)[p[i]>0]),target=sp,detected=bool(p[i,jj]),other_CO_CHO_COH_calls=';'.join(x for x in ['CO','CHO','COH'] if x!=sp and p[i,SPECIES.index(x)])))
pd.DataFrame(conf).to_csv(B/'CO_CHO_COH_confusions.csv',index=False)
# Identical-case improvement by exact-set flips, labels, substrate and mixing.
c=pd.read_csv(B/'paired_prediction_changes.csv').fillna('');changes=[]
for dom in ['CuAu','Cu100']:
 for part in ['single','mixed','clean']:
  ids=[i for i,x in enumerate(r) if x['domain']==dom and ((len(x['identity_labels'])==1) if part=='single' else (len(x['identity_labels'])>1) if part=='mixed' else not x['identity_labels'])]
  if not ids:continue
  changes.append(dict(domain=dom,subset=part,n_records=len(ids),n_groups=len(set(families[ids])),wrong_to_correct=int(sum(not x and y for x,y in zip(np.all(yy[ids]==a[ids],axis=1),np.all(yy[ids]==b[ids],axis=1)))),correct_to_wrong=int(sum(x and not y for x,y in zip(np.all(yy[ids]==a[ids],axis=1),np.all(yy[ids]==b[ids],axis=1))))))
pd.DataFrame(changes).to_csv(B/'exact_set_changes_by_domain_mix.csv',index=False)
# Plots use the saved tables, not a second calculation of scores.
v=pd.read_csv(B/'paired_size_intervals.csv');v=v[(v.seed=='mean_all_three')&(v.metric=='micro_f1')];fig,ax=plt.subplots(figsize=(7.5,3.7));labels=[]
for i,row in enumerate(v.itertuples()):
 ax.errorbar(row.difference,i,xerr=[[row.difference-row.lower],[row.upper-row.difference]],fmt='o',color='#23734d' if row.kind=='trees' else '#4375a5',capsize=3);labels.append(row.kind+' / '+row.threshold)
ax.axvline(0,color='.5',ls='--',lw=1);ax.set(yticks=range(len(labels)),yticklabels=labels,xlabel='Expanded − old training micro F1, same 96 query groups',title='All three seeds; conditional 95% group-bootstrap intervals');fig.tight_layout();fig.savefig(F/'paired_training_size.png',dpi=300);plt.close(fig)
v=pd.read_csv(B/'size_and_learning_metrics.csv');v=v[v.analysis.str.startswith('curve_')&(v.weighting=='near_group')];membership=pd.DataFrame(json.load(open(B/'ml/learning_membership.json')));curve=membership.groupby(['kind','seed','fraction']).training_groups.mean().reset_index();v=v.merge(curve,on=['kind','seed','fraction']);v.to_csv(B/'learning_plot_data.csv',index=False);fig,ax=plt.subplots(figsize=(6.8,3.6))
for kind,color in [('trees','#23734d'),('logistic','#4375a5')]:
 for seed,g in v[v.kind==kind].groupby('seed'):g=g.sort_values('fraction');ax.plot(g.training_groups,g.micro_f1,marker='o',alpha=.6,color=color,label=kind if seed==20261020 else None)
ax.set(xlabel='Mean actual training geometry groups per outer fold',ylabel='Micro F1, same old99 queries',title='Nested label-blind group subsampling; all seeds, fixed 0.5');ax.legend();fig.tight_layout();fig.savefig(F/'signed_shape_learning_curve.png',dpi=300);plt.close(fig)
v=pd.read_csv(B/'existing_representation_strata.csv');v=v[(v.weighting=='near_group')&v.subset.isin(['CuAu','Cu100','CuAu_mixed','CuAu_single'])];v.to_csv(B/'representation_plot_data.csv',index=False);fig,axes=plt.subplots(1,2,figsize=(8,3.5))
for ax,domain in zip(axes,['CuAu','Cu100']):
 d=v[v.subset==domain].set_index('representation');keys=['corrected_normalized','common_raw','relative_delta','signed_shape'];ax.bar(range(4),[d.loc[k,'micro_f1'] for k in keys],color=['#a4b6c0','#819d8f','#719ac1','#23734d']);ax.set(xticks=range(4),xticklabels=['Corrected\nnorm. raw','Common-scale\nraw','Relative\nΔμ','Signed\nshape'],ylim=(0,1),title=domain,ylabel='Group-weighted micro F1')
fig.tight_layout();fig.savefig(F/'representation_domains.png',dpi=300);plt.close(fig)
v=pd.read_csv(B/'robustness_review_metrics.csv');fig,axes=plt.subplots(2,1,figsize=(9,6),sharex=True)
for aug,color in [(False,'#4375a5'),(True,'#23734d')]:
 d=v[v.augmented_training==aug];x=np.arange(len(d));axes[0].plot(x,d.retained_fraction,'o-',c=color,label='Augmented training' if aug else 'Unaugmented');axes[1].plot(x,d.full_query_model_exact_error_before_gate,'o-',c=color)
axes[0].set(ylabel='Retained query fraction',ylim=(-.03,1.05));axes[0].legend();axes[1].set(ylabel='Full-query exact-set error\nbefore review gate',xticks=np.arange(len(d)),xticklabels=d.scenario);axes[1].tick_params(axis='x',rotation=40);fig.suptitle('Illustrative synthetic stress; DCT16 logistic control, no measured detection limit');fig.tight_layout();fig.savefig(F/'robustness_and_retention.png',dpi=300);plt.close(fig)
print(pd.DataFrame(out).query("subset=='CuAu_mixtures' and weighting=='record'").to_string(index=False));print(pd.DataFrame(rows).to_string(index=False));print(pd.DataFrame(changes).to_string(index=False))
