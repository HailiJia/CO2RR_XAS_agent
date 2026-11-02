"""Figures and conditional within-cohort comparisons from frozen paired predictions."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.fixed_top_two import SEEDS,CANDIDATES
from tools.spectral_multilabel import f1_metrics
B=ROOT/'study/top_two_20261023';E=B/'experiments';OUT=B/'figures';OUT.mkdir(exist_ok=True)
STAGES=['fixed_Combined_threshold0p5','fixed_Combined_transferred87_threshold','fixed_Combined_retuned_threshold','original_feature_selection','regularized_selection']
LABELS=['Combined\nthreshold 0.5','Combined\n87 thresholds','Combined\nretuned threshold','Original feature\nselection','Regularized\nselection']


def save(f,n):
 for ext in ['pdf','svg','png']:f.savefig(OUT/(n+'.'+ext),dpi=350,bbox_inches='tight')
 plt.close(f)


def main():
 plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
 rng=np.random.default_rng(20261023);paired=[];plot=[]
 for n in [87,90]:
  for seed in SEEDS:
   a=np.load(E/f'cohort{n}_seed{seed}/original_feature_selection.npz');b=np.load(E/f'cohort{n}_seed{seed}/regularized_selection.npz');y=a['truth'];p=a['prediction'];q=b['prediction'];diff={k:f1_metrics(y,q)[k]-f1_metrics(y,p)[k] for k in f1_metrics(y,p)};samples=[]
   for _ in range(1000):
    ix=rng.integers(0,n,n);u=f1_metrics(y[ix],q[ix]);v=f1_metrics(y[ix],p[ix]);samples.append([u[k]-v[k] for k in diff])
   samples=np.array(samples)
   for j,key in enumerate(diff):paired.append({'cohort':n,'seed':seed,'metric':key,'regularized_minus_original_selection':diff[key],'conditional95_low':np.quantile(samples[:,j],.025),'conditional95_high':np.quantile(samples[:,j],.975),'scope':'paired fixed-prediction interval, no refitting or multiple-comparison adjustment'})
 for n in [87,90]:
  for seed in SEEDS:
   for stage in STAGES:
    z=np.load(E/f'cohort{n}_seed{seed}'/(stage+'.npz'));values=f1_metrics(z['truth'][:87],z['prediction'][:87]);plot.append({'training_cohort':n,'seed':seed,'stage':stage,**values})
 pd.DataFrame(paired).to_csv(B/'paired_regularization_comparison.csv',index=False);plot=pd.DataFrame(plot);plot.to_csv(B/'paired_figure_arrays.csv',index=False);rec=pd.read_csv(B/'single_mixed_recall.csv');f,axs=plt.subplots(2,2,figsize=(12,8));x=np.arange(len(STAGES))
 for ax,metric,title in [(axs[0,0],'micro_f1','Micro F1: same 87 held-out configurations'),(axs[0,1],'exact_set_accuracy','Exact identity set: same 87 configurations')]:
  for dx,n,color in [(-.17,87,'#888888'),(.17,90,'#0072B2')]:
   q=plot[plot.training_cohort==n].groupby('stage')[metric].agg(['mean','std']).loc[STAGES];ax.bar(x+dx,q['mean'],.32,yerr=q['std'],capsize=3,color=color,label=f'Train cohort {n}')
  ax.set_xticks(x,LABELS,fontsize=7);ax.set(ylim=(0,1),ylabel=metric.replace('_',' '),title=title);ax.legend(fontsize=7)
 for ax,s in [(axs[1,0],'CO'),(axs[1,1],'H')]:
  for dx,n,color in [(-.17,87,'#888888'),(.17,90,'#0072B2')]:
   q=rec[(rec.cohort==n)&(rec.species==s)&(rec.kind=='mixed')].groupby('model').recall.agg(['mean','std']).loc[STAGES];ax.bar(x+dx,q['mean'],.32,yerr=q['std'],capsize=3,color=color,label=f'Train cohort {n}')
  ax.set_xticks(x,LABELS,fontsize=7);ax.set(ylim=(0,1),ylabel='Mixture recall',title=s+' mixtures: unchanged positive support');ax.legend(fontsize=7)
 f.suptitle('Depth fixed to population-weighted layers 5+4 in every model',fontweight='bold');f.tight_layout(rect=(0,.055,1,.95));f.text(.5,.012,'Bars: mean of three declared tree seeds; error bars: seed SD, not independent-sample confidence intervals. No winning seed selected.',ha='center',fontsize=8);save(f,'paired_top_two')
 ab=pd.read_csv(B/'feature_leaf_ablation.csv');features=['derivative','shape_CDF_derivatives','DCT8','DCT16','DCT24','DCT32'];f,axs=plt.subplots(1,2,figsize=(11,5),sharey=True)
 for ax,n in zip(axs,[87,90]):
  q=ab[ab.cohort==n].groupby(['feature','leaf']).micro_f1.agg(['mean','std']);a=np.array([[q.loc[(feature,leaf),'mean'] for leaf in [1,2,4]] for feature in features]);im=ax.imshow(a,vmin=.65,vmax=.83,cmap='Greens',aspect='auto')
  for i,feature in enumerate(features):
   for j,leaf in enumerate([1,2,4]):ax.text(j,i,f'{q.loc[(feature,leaf),"mean"]:.3f}\n±{q.loc[(feature,leaf),"std"]:.3f}',ha='center',va='center',fontsize=9)
  ax.set_xticks([0,1,2],[1,2,4]);ax.set_yticks(range(6),['d1','Combined','DCT8','DCT16','DCT24','DCT32']);ax.set(xlabel='Minimum leaf size',title=f'{n} configurations: micro F1 mean ± seed SD')
 f.colorbar(im,ax=axs.ravel().tolist(),fraction=.03,pad=.04,label='Micro F1');f.subplots_adjust(left=.12,right=.88,bottom=.15,wspace=.25);f.text(.5,.02,'Descriptive fixed-feature ablations; original and regularized selectors use inner-training scores only. Cohort totals differ; use paired plots for added-data effects.',ha='center',fontsize=8);save(f,'feature_leaf_ablation')
 # Report actual training-only leaf selections, without choosing a winning leaf from outer outcomes.
 selections=pd.concat([pd.read_csv(E/f'cohort{n}_seed{seed}/selections.csv') for n in [87,90] for seed in SEEDS],ignore_index=True);selections.to_csv(B/'all_selections.csv',index=False);selections.query("stage=='regularized_selection'").groupby(['cohort','feature','leaf']).size().rename('outer_fold_seed_selections').reset_index().to_csv(B/'regularized_selection_frequency.csv',index=False)
if __name__=='__main__':main()
