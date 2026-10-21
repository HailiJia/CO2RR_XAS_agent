"""Rebuild five outline-role figures from committed numerical figure_data."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'svg.fonttype':'none'})

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data-dir',default=str(ROOT/'figure_data'));p.add_argument('--output-dir',default=str(ROOT/'figures'));a=p.parse_args(argv);data=Path(a.data_dir);out=Path(a.output_dir);out.mkdir(exist_ok=True,parents=True)
    def save(fig,name):
        for ext in ['pdf','svg','png']:fig.savefig(out/(name+'.'+ext),dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(12,5));ax.set_xlim(0,12);ax.set_ylim(0,5);ax.axis('off')
    boxes=[('Requested design\ncontrols + molecule mappings',.3,3.4),('Generation +\ngeometry checks',4.3,3.4),('Relaxation\nactual final coordinates',8.3,3.4),('XAS + final\npopulation verification',8.3,1.4),('ISAAC + immutable\nprovenance and hashes',4.3,1.4),('Grouped inference\n87 configurations',.3,1.4)]
    for label,x,y in boxes:ax.add_patch(FancyBboxPatch((x,y),3.3,.9,boxstyle='round,pad=.1',facecolor='#e0eff7',edgecolor='#2378ae'));ax.text(x+1.65,y+.45,label,ha='center',va='center')
    for start,end in [((3.7,3.85),(4.1,3.85)),((7.7,3.85),(8.1,3.85)),((9.95,3.25),(9.95,2.45)),((8.1,1.85),(7.7,1.85)),((4.1,1.85),(3.7,1.85))]:ax.annotate('',xy=end,xytext=start,arrowprops=dict(arrowstyle='->',lw=1.8))
    ax.text(6,.45,'Tested code and local replay are evidence of implementation.\nHPC submission, record deposition and merging require explicit authorization; execution-rate claims remain unmeasured.',ha='center',va='center',bbox=dict(facecolor='#fff2d6',edgecolor='none',pad=8));ax.set_title('1. Framework architecture and scope of autonomy',pad=15);save(fig,'figure_1_framework')
    inventory=pd.read_csv(data/'inventory_combinations.csv',index_col=0);fig,axes=plt.subplots(1,2,figsize=(13,7),gridspec_kw={'width_ratios':[1.25,1]});left=np.zeros(len(inventory))
    for key,color in [('usable','#2378ae'),('record excluded','#d08332'),('no completed record','#b8bdc5')]:axes[0].barh(range(len(inventory)),inventory[key],left=left,label=key,color=color);left+=inventory[key]
    axes[0].set_yticks(range(len(inventory)),inventory.index);axes[0].set_xlabel('Unique structures');axes[0].legend(fontsize=8);axes[0].set_title('155 structures: full inventory versus completion')
    plan=pd.read_csv(data/'planned_structural_matrix.csv').drop_duplicates('parent');axes[1].axis('off');table=axes[1].table(cellText=[[r.parent,f'{r.explicit_minimum_Cu_rows}:{r.explicit_minimum_Au_rows}',int(r.matching_ratio_unique_sources)] for r in plan.itertuples()],colLabels=['Planned parent','Minimum\nCu:Au rows','Sources with\ninferred ratio'],loc='upper center',cellLoc='center',colWidths=[.45,.25,.3]);table.auto_set_font_size(False);table.set_fontsize(9);table.scale(1,2)
    axes[1].text(.5,.25,'Ratio matches do not establish all requested sites,\ncoverages, distributions or independent parents.\nPure controls and absent cells remain priorities.\nRow counts are inferred from initial coordinates.',ha='center',transform=axes[1].transAxes);fig.suptitle('2. Planned controls and actual completion');fig.tight_layout();save(fig,'figure_2_inventory')
    spectra=pd.read_csv(data/'layer_spectra.csv');fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    for l,color in [(5,'#2378ae'),(2,'#d08332')]:
        t=spectra[spectra.layer==l];axes[0].plot(t.energy_eV,t['mean'],color=color,label=f'Layer {l} (n=87)');axes[0].fill_between(t.energy_eV,t['mean']-t.configuration_SD,t['mean']+t.configuration_SD,color=color,alpha=.15)
    axes[0].set_xlabel('Photon energy (eV)');axes[0].set_ylabel('Unit-area absorption (1/eV)');axes[0].legend();axes[0].set_title('Real layer spectra: mean ± configuration SD')
    axes[1].axis('off');axes[1].text(.5,.75,'Numerical Cu / Cu₂O / CuO reference metrics\nUNMEASURED',ha='center',va='center',fontsize=13);axes[1].text(.5,.42,'Raw experimental provenance and the named CuO PDF\nare unavailable. An image cannot establish exact RMSE/r.\n\nC/O complementary-edge contrasts: UNMEASURED.\nGenerator probes do not establish numerical convergence.',ha='center',va='center');fig.suptitle('3. Reference status, complementary edges and real Cu-layer spectra');fig.tight_layout();save(fig,'figure_3_spectra')
    per=pd.read_csv(data/'primary_species_performance.csv');depth=pd.read_csv(data/'depth_performance.csv');aggregate=pd.read_csv(data/'aggregate_performance.csv');fig,axes=plt.subplots(1,3,figsize=(16,6),gridspec_kw={'width_ratios':[1.15,1.2,1]})
    axes[0].barh(range(len(per)),per.f1,color=['#b8bdc5' if n<5 else '#2378ae' for n in per.support]);axes[0].set_yticks(range(len(per)),[f'{r.species} (n={r.support})' for r in per.itertuples()]);axes[0].invert_yaxis();axes[0].set_xlim(0,1);axes[0].set_xlabel('Primary held-out F1');axes[0].set_title('Primary species presence; rare n<5 in gray')
    keys=['layer_5','layer_4','layer_3','layer_2','top_2','top_3','all_4']
    for metric,color in [('micro','#2378ae'),('macro','#d08332')]:
        t=depth[depth.metric==metric].set_index('depth').loc[keys];axes[1].plot(range(7),t.f1,'o-',label=metric,color=color)
    axes[1].set_xticks(range(7),['5 (top)','4','3','2','Top 2','Top 3','Four'],rotation=40);axes[1].set_ylim(0,1);axes[1].set_ylabel('F1');axes[1].legend();axes[1].set_title('Fixed-depth nested feature selection')
    labels=['Primary nested','Exploratory fixed','Training-only baseline'];workflows=['independent_selected_micro','provisional_best','training_most_common_identity_set'];t=aggregate.set_index('workflow').loc[workflows];x=np.arange(3)
    axes[2].bar(x-.15,t.micro_f1,.3,label='micro',color='#2378ae');axes[2].bar(x+.15,t.macro_f1,.3,label='macro',color='#d08332');axes[2].set_xticks(x,labels,rotation=25,ha='right');axes[2].set_ylim(0,1);axes[2].legend();axes[2].set_title('Exploratory winner is not confirmation');fig.suptitle('4. Grouped multilabel inference on the repeatedly examined 87-configuration cohort');fig.text(.5,.01,'Identical frozen configuration folds. Supports overlap (99 positives); 9 mixed and 78 single-species configurations. New independent confirmation remains required.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.05,1,.96]);save(fig,'figure_4_inference')
    coords=pd.read_csv(data/'quartet_input_coordinates.csv');quartet=json.loads((data/'matched_quartet.json').read_text());fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True)
    colors={'Cu':'#c98543','Au':'#d8bd46','C':'#303030','O':'#d54d4d'}
    for i,(ax,row) in enumerate(zip(axes.flat,quartet['structures'])):
        panel=coords[coords.panel==i]
        for el in ['Cu','Au','C','O']:
            q=panel[panel.element==el];ax.scatter(q.x_A,q.y_A,s=18 if el in ['Cu','Au'] else 65,c=colors[el],label=el,alpha=.35 if el in ['Cu','Au'] else 1,edgecolors='none')
        ax.set_aspect('equal');ax.set_title(f"{row['state_requested']} at {row['region']}: input C–C {row['minimum_CC_A']:.2f} Å");ax.set_xlabel('x (Å)');ax.set_ylabel('y (Å)')
    axes[0,0].legend(ncol=4,fontsize=8);fig.suptitle('5. Matched INPUT quartet; relaxed coupling and Cu/C/O contrasts remain unmeasured');fig.text(.5,.015,'Shared cell and metal parent; 2 C + 2 O. Loading = 2/36 C equivalents per top-layer Cu. Inputs do not establish intact relaxed OCCO.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.96]);save(fig,'figure_5_matched_inputs')
if __name__=='__main__':main()
