"""Portable PNG from exact guarded group-weighted CSVs; no fitting."""
from pathlib import Path
import pandas as pd
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];D=R/'figure_data/controlled_identity';d=pd.read_csv(D/'near_group_metrics.csv');c=pd.read_csv(D/'group_learning_metrics.csv');fig=plt.figure(figsize=(7.2,6.4),layout='constrained');gs=fig.add_gridspec(2,2,height_ratios=[1.1,1]);ax=[fig.add_subplot(gs[0,0]),fig.add_subplot(gs[0,1]),fig.add_subplot(gs[1,:])];reps=['native_normalized','corrected_normalized','common_raw','relative_delta','signed_shape'];labels=['Native-normalized','Corrected-normalized','Common-scale raw','Relative delta','Signed shape'];colors=['#4477aa','#66aacc','#228833','#aa3377','#ccbb44']
for k,fam in enumerate(['logistic','trees']):
 values=[d[d.analysis==f'fixed_rep_joint130_{rep}_{fam}'].group_equal_micro_f1.iloc[0] for rep in reps];ax[k].barh(range(5),values,color=colors);ax[k].set(yticks=range(5),yticklabels=labels,xlim=(0,1),xlabel='Micro F1',title=f'Joint124 groups / {fam}');ax[k].tick_params(labelsize=8);ax[k].title.set_fontsize(9)
a=c.groupby('fraction').micro_f1.agg(['mean','std']);ax[2].errorbar(a.index,a['mean'],yerr=a['std'],marker='o',color='#228833');ax[2].set(xlabel='Fraction of available training groups',ylabel='Micro F1 on same96 old query groups',title='Delta / Combined / trees / threshold0.5');ax[2].text(.05,.03,'Bars: refit/seed SD (3 seeds)',transform=ax[2].transAxes,fontsize=8);ax[2].title.set_fontsize(9);ax[2].tick_params(labelsize=8);fig.savefig(R/'figures/SI_controlled_identity.png',dpi=300);plt.close(fig)
