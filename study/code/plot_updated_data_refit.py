"""Versioned updated-cohort figures from frozen prediction arrays; no fitting."""
from pathlib import Path
import numpy as np,pandas as pd,json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/retraining_20261022';OUT=B/'figures';OUT.mkdir(exist_ok=True)

def main():
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    m=json.load(open(B/'refit/models/metrics.json'));paired=pd.read_csv(B/'paired_old_subset_comparison.csv');r=pd.read_csv(B/'single_mixed_comparison.csv');fig,axs=plt.subplots(1,3,figsize=(12,3.8));a=axs[0];x=np.arange(3);keys=['micro_f1','macro_f1','exact_set_accuracy']
    a.bar(x-.18,paired.frozen_primary,.34,color='#888888',label='Frozen primary: 87');a.bar(x+.18,paired.updated_old_subset,.34,color='#0072B2',label='Refit: same 87 subset');a.set_xticks(x,['Micro F1','Macro F1','Exact set']);a.set(ylim=(0,1),ylabel='Held-out score');a.legend(fontsize=7);a.set_title('Paired old configurations')
    a=axs[1];a.axhline(0,lw=.8,c='#999999',label='No change');v=paired.difference.to_numpy();a.errorbar(x,v,yerr=[v-paired.conditional_95_low,paired.conditional_95_high-v],fmt='o',capsize=4,color='#0072B2');a.set_xticks(x,['Micro F1','Macro F1','Exact set']);a.set_ylabel('Refit − frozen primary');a.set_title('Conditional paired 95% intervals')
    a=axs[2];x=np.arange(2)
    for offset,name,color,label in [(-.18,'frozen_primary_87','#888888','Frozen primary'),(.18,'updated_refit_90','#0072B2','Updated refit')]:
        q=r[(r.model==name)&(r.kind=='mixed')].set_index('species').loc[['H','CO']];a.bar(x+offset,q.recall,.34,color=color,label=label)
        for xx,row in zip(x+offset,q.itertuples()):a.text(xx,row.recall+.025,f'{row.detected}/{row.support}',ha='center',fontsize=8)
    a.set_xticks(x,['H mixtures','CO mixtures']);a.set(ylim=(0,1.1),ylabel='Held-out recall');a.legend(fontsize=7);a.set_title('Unchanged mixture supports')
    fig.suptitle('Updated export v3: 90 configurations / 360 layer spectra',fontsize=12,fontweight='bold');fig.tight_layout(rect=(0,.09,1,.94));fig.text(.5,.01,'All paired intervals include zero; fixed-prediction intervals exclude refitting uncertainty. Added/recovered cases are three CO-only configurations.',ha='center',fontsize=8)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/('updated_refit_summary.'+ext),dpi=350,bbox_inches='tight')
    plt.close(fig)
if __name__=='__main__':main()
