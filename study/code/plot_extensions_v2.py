"""Figures from frozen extension arrays; no fitting or outer-score selection."""
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];BASE=ROOT/'study/extensions_v2/ml';OUT=ROOT/'study/manuscript_package/figures'

def main():
    plt.rcParams.update({'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    d=pd.read_csv(BASE/'feature_ablation.csv');features=['normalized','CDF','derivative','descriptors','DCT8','DCT16','DCT24','DCT32','PCA8','PCA16'];models=d.model.unique();v=d.pivot(index='model',columns='feature',values='micro_f1').reindex(index=models,columns=features)
    f,a=plt.subplots(figsize=(9,4));im=a.imshow(v,vmin=.4,vmax=.8,cmap='Greens',aspect='auto');a.set_xticks(range(len(features)),features,rotation=35,ha='right');a.set_yticks(range(len(models)),[m.replace('_',' ') for m in models]);f.colorbar(im,ax=a,label='Micro F1')
    for i in range(len(models)):
        for j in range(len(features)):a.text(j,i,f'{v.iloc[i,j]:.2f}',ha='center',va='center',fontsize=7)
    f.tight_layout();save(f,'SI_extensions_features')
    d=pd.read_csv(BASE/'robustness.csv');f,axs=plt.subplots(2,2,figsize=(9,6.8));names=['nearest_neighbor','independent_trees','classifier_chain','xgboost']
    for a,name in zip(axs.flat,names):
        q=d[d.model.isin([name,name+'_augmented'])]
        scenarios=q.scenario.unique();x=np.arange(len(scenarios))
        for model,marker,label in [(name,'o','Unaugmented'),(name+'_augmented','s','Training augmentation')]:
            vals=q[q.model==model].set_index('scenario').loc[scenarios];a.plot(x,vals.micro_f1,marker+'-',ms=3,lw=1,label=label)
        a.set_xticks(x,scenarios,rotation=70,ha='right',fontsize=5.5);a.set(ylim=(0,1),ylabel='Micro F1');a.set_title(name.replace('_',' '));a.legend(fontsize=6)
    f.tight_layout();save(f,'SI_extensions_robustness')
    d=pd.read_csv(BASE/'learning_curves.csv');q=d.groupby('fraction').micro_f1.agg(['mean','std']);f,axs=plt.subplots(1,2,figsize=(8,3.4));a=axs[0];a.errorbar(q.index,q['mean'],yerr=q['std'],fmt='o-',capsize=3);a.set(xlabel='Outer-training fraction',ylabel='Mean fold micro F1',ylim=(0,1));a.set_title('Subsample/refit variation (SD)')
    d=pd.read_csv(BASE/'review_policy.csv');a=axs[1]
    for name in names:
        q=d[d.model==name].groupby('margin')[['accepted','total','exact_errors']].sum();a.plot(q.accepted/q.total,q.exact_errors/q.accepted.replace(0,np.nan),'o-',ms=3,label=name.replace('_',' '))
    a.set(xlabel='Accepted coverage',ylabel='Exact-set error among accepted',xlim=(0,1),ylim=(0,1));a.legend(fontsize=5.5);a.set_title('Sparse-label calibration unresolved')
    f.tight_layout();save(f,'SI_extensions_learning_review')

def save(f,name):
    for ext in ['pdf','svg','png']:f.savefig(OUT/(name+'.'+ext),dpi=320,bbox_inches='tight')
    plt.close(f)
if __name__=='__main__':main()
