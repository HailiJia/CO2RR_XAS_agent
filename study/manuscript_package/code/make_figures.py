"""Rebuild manuscript Figures 1-5 from bundled source data, without HPC access.

python code/make_figures.py
Outputs PDF, SVG, and 300-dpi PNG. No model fitting or synthetic spectra.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'figure_data'; FIG=ROOT/'figures'; FIG.mkdir(exist_ok=True)
C={'blue':'#246282','teal':'#26877b','orange':'#d38135','red':'#be4a46','gray':'#88929b', 'Cu':'#bc754b','Au':'#dcb645','C':'#313844','O':'#d84b46','H':'#f4f4ef'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
                     'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
def save(fig,name):
    for ext in ['pdf','svg','png']: fig.savefig(FIG/f'{name}.{ext}',dpi=300,bbox_inches='tight',pad_inches=.12)
    plt.close(fig)
def title(ax,letter,text): ax.set_title(f'{letter}  {text}',loc='left',fontweight='bold',pad=10)
def box(ax,xy,w,h,heading,body,color=C['blue']):
    x,y=xy; ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=.015',facecolor='#f3f6f8',edgecolor=color,lw=1.2))
    import textwrap
    body='\n'.join(textwrap.fill(line,width=32) for line in body.split('\n'))
    ax.text(x+.020,y+h-.04,heading,ha='left',va='top',fontweight='bold',fontsize=10,color=color)
    ax.text(x+.020,y+h-.105,body,ha='left',va='top',fontsize=8.2,linespacing=1.4)
def arrow(ax,a,b): ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='-|>',color='#627180',lw=1.4,shrinkA=3,shrinkB=3))
def fig1():
    f,ax=plt.subplots(figsize=(10,6.0)); ax.axis('off'); ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,(.02,.68),.28,.27,'Request and plan','Natural language / settings\nSchema-checked tool arguments\nMetal pair, adsorbates, edge')
    box(ax,(.36,.68),.28,.27,'Structure and chemistry','Interface, sites, coverage\nPeriodic contact checks\nRequested and observed labels')
    box(ax,(.70,.68),.28,.27,'Relaxation and XAS','VASP relaxation → CONTCAR\nFEFF / FDMNES / VASP\nSite and layer provenance')
    box(ax,(.70,.25),.28,.27,'Execute and validate','NERSC API + Slurm dependencies\nPersistent state and restart\nOutput and population checks',C['teal'])
    box(ax,(.36,.25),.28,.27,'ISAAC records','Spectra + structural descriptors\nConfiguration and asset links\nPortal validation / retrieval',C['teal'])
    box(ax,(.02,.25),.28,.27,'Inference and refinement','Grouped training and selection\nPredictions and diagnostics\nTargeted calculations',C['teal'])
    arrow(ax,(.30,.82),(.36,.82));arrow(ax,(.64,.82),(.70,.82));arrow(ax,(.84,.68),(.84,.52));arrow(ax,(.70,.38),(.64,.38));arrow(ax,(.36,.38),(.30,.38))
    ax.annotate('Candidate refinement',xy=(.16,.68),xytext=(.16,.56),ha='center',va='center',color=C['teal'],arrowprops=dict(arrowstyle='-|>',color=C['teal']))
    ax.text(.5,.12,'Executed scientific demonstration: Cu K edge · adsorbate presence · 87 configurations',ha='center',fontweight='bold',color=C['blue'])
    ax.text(.5,.05,'Planning is model assisted; calculations and validation use explicit software and recorded settings.',ha='center',fontsize=9)
    save(f,'Figure_1_framework')
def load_poscar(path):
    l=path.read_text().splitlines();scale=float(l[1]);cell=np.array([list(map(float,s.split())) for s in l[2:5]])*scale
    names=l[5].split();n=list(map(int,l[6].split()));sp=np.repeat(names,n);i=7
    if l[i].lower().startswith('s'):i+=1
    direct=l[i].lower().startswith('d');i+=1
    q=np.array([list(map(float,s.split()[:3])) for s in l[i:i+sum(n)]])
    frac=q if direct else q*scale@np.linalg.inv(cell)
    return sp,frac@cell,cell,frac
def surface(ax,file,zoom=False):
    sp,xyz,cell,frac=load_poscar(DATA/'structures'/file)
    metal=np.isin(sp,['Cu','Au']);top=xyz[metal,2].max();selected=(metal&(xyz[:,2]>top-1.6))|~metal
    if zoom:
        center=xyz[~metal,:2].mean(axis=0); lim=[center[0]-4.8,center[0]+4.8,center[1]-4.8,center[1]+4.8]
    else: lim=[-.8,max(xyz[:,0])+1,-.8,max(xyz[:,1])+1]
    for el,size in [('Cu',180),('Au',180),('C',70),('O',90),('H',35)]:
        m=(sp==el)&selected;ax.scatter(xyz[m,0],xyz[m,1],s=size,c=C[el],edgecolors='#666666',linewidths=.5,zorder=3 if el in ['C','O','H'] else 1,alpha=1 if el in ['C','O','H'] else .62)
    ids=np.where(~metal)[0]
    for i,a in enumerate(ids):
        for b in ids[:i]:
            d=frac[a]-frac[b];d[:2]-=np.round(d[:2]);vec=d@cell;length=np.linalg.norm(vec)
            if length<1.9 and np.linalg.norm(xyz[a]-xyz[b])<1.9:
                ax.plot(xyz[[a,b],0],xyz[[a,b],1],color='#394550',lw=3,zorder=2)
    ax.set(xlim=lim[:2],ylim=lim[2:]);ax.set_aspect('equal');ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
    carbon=ids[sp[ids]=='C'];cc=[]
    for i,a in enumerate(carbon):
        for b in carbon[:i]:
            d=frac[a]-frac[b];d[:2]-=np.round(d[:2]);cc.append(np.linalg.norm(d@cell))
    return min(cc) if cc else np.nan
def fig2():
    f=plt.figure(figsize=(10,8));gs=f.add_gridspec(2,2,width_ratios=[1,1.2],hspace=.34,wspace=.30)
    a=f.add_subplot(gs[0,0]);surface(a,'2CO_adjacent_interface.vasp');title(a,'a','Cu–Au interface input')
    a.legend(handles=[Line2D([],[],marker='o',ls='',color=C[x],label=x) for x in ['Cu','Au','C','O']],loc='lower center',ncol=4,frameon=False,bbox_to_anchor=(.5,-.06))
    a=f.add_subplot(gs[0,1]);d=pd.read_csv(DATA/'figure1_species_presence.csv');inv=pd.read_csv(DATA/'all_unique_structures.csv')
    support=pd.read_csv(DATA/'species_support.csv').set_index('species')['support'];species=sorted(set(s for v in inv.labels.dropna() for s in v.split(';'))-{'clean','CuO'})
    allcount=[sum(s in str(v).split(';') for v in inv.labels) for s in species];current=[support.get(s,0) for s in species];y=np.arange(len(species))
    a.barh(y+.16,allcount,height=.3,color='#c9d3db',label='Full inventory');a.barh(y-.16,current,height=.3,color=C['blue'],label='ML eligible');a.set_yticks(y,species);a.invert_yaxis();a.set_xlabel('Unique positive configurations');a.legend(frameon=False,fontsize=8,loc='lower right');title(a,'b','Species inventory')
    a=f.add_subplot(gs[1,0]);a.axis('off');title(a,'c','Independent dataset controls')
    lines=[('Parent','Pure metal or matching fcc(111) pair'),('Region','Interface · neighboring row · terrace'),('Binding','Atop · bridge · multicoordinated'),('Coverage','Molecules / declared surface reference'),('Distribution','Uniform · biased · coadsorbed'),('Chemistry','Verify components after relaxation')]
    for i,(k,v) in enumerate(lines):
        a.text(0,.93-i*.14,k,fontweight='bold',color=C['blue']);a.text(0,.88-i*.14,v,fontsize=9)
    a=f.add_subplot(gs[1,1]);combo=pd.read_csv(DATA/'figure1_combination_counts.csv').sort_values('usable',ascending=False).head(10).iloc[::-1]
    left=np.zeros(len(combo))
    for col,c,lab in [('usable',C['teal'],'ML eligible'),('record excluded',C['red'],'Excluded record'),('no completed record','#c9d3db','No completed record')]:
        a.barh(np.arange(len(combo)),combo[col],left=left,color=c,label=lab);left+=combo[col]
    a.set_yticks(np.arange(len(combo)),combo.combo);a.set_xlabel('Unique configurations');title(a,'d','Ten most populated identity sets');a.legend(frameon=False,fontsize=7,loc='lower right')
    f.text(.5,.01,'155 unique inventory structures; 87 ML eligible. Species counts overlap for coadsorption; identity sets are exclusive.',ha='center',fontsize=9)
    save(f,'Figure_2_dataset')
def fig3():
    f=plt.figure(figsize=(10,8));gs=f.add_gridspec(2,2,hspace=.40,wspace=.32)
    a=f.add_subplot(gs[0,0]);a.imshow(plt.imread(DATA/'CuO_benchmark.png'));a.axis('off');title(a,'a','CuO reference comparison')
    a.text(.5,-.08,'Supplied comparison plot; numeric benchmark audit pending',transform=a.transAxes,ha='center',fontsize=8,color=C['gray'])
    a=f.add_subplot(gs[0,1]);d=pd.read_csv(DATA/'representative_layer_spectra.csv');config='CO/CO_0p11/interface_biased';d=d[d.configuration==config]
    for layer,c in zip([5,4,3,2],[C['red'],C['orange'],C['teal'],C['blue']]):
        s=d[d.layer==layer];a.plot(s.energy_eV,s.absorption,lw=1.3,label=f'Layer {layer}'+(' (top)' if layer==5 else ''),color=c)
    a.set(xlim=(8974,9029),xlabel='Recorded photon energy (eV)',ylabel='Absorption (recorded arbitrary units)');a.legend(frameon=False,fontsize=8);title(a,'b','Representative CO layer spectra')
    a=f.add_subplot(gs[1,0]);d=pd.read_csv(DATA/'spectral_identity_contrasts.csv');cols=list(d.columns[1:]);v=d[cols].to_numpy().T;lim=np.max(abs(v));im=a.imshow(v,aspect='auto',origin='upper',extent=[d.energy_eV.min(),d.energy_eV.max(),len(cols)-.5,-.5],cmap='RdBu_r',vmin=-lim,vmax=lim)
    a.set_yticks(range(len(cols)),cols);a.set_xlabel('Recorded photon energy (eV)');cb=f.colorbar(im,ax=a,fraction=.045,pad=.03);cb.set_label('Mean presence − absence (eV⁻¹)',fontsize=8);title(a,'c','Descriptive top-two-layer contrasts')
    a=f.add_subplot(gs[1,1]);a.axis('off');title(a,'d','Complementary edge plan')
    for y,heading,body,c in [(.9,'Cu K edge','Completed identity and depth benchmark',C['blue']),(.67,'C K edge','Planned adsorbate and C–C bonding contrast',C['teal']),(.44,'O K edge','Planned oxygen coordination contrast',C['teal'])]:
        a.text(.03,y,heading,fontweight='bold',color=c,fontsize=12);a.text(.03,y-.09,body,fontsize=9)
    a.text(.03,.19,'Presence/absence groups are not matched for\ncoverage, coadsorbates, or geometry.\nThe heatmap is not model attribution.',va='top',fontsize=9,color='#5a6670',linespacing=1.5)
    save(f,'Figure_3_spectroscopy')
def fig4():
    f=plt.figure(figsize=(10,8));gs=f.add_gridspec(2,2,hspace=.52,wspace=.30)
    a=f.add_subplot(gs[0,0]);d=pd.read_csv(DATA/'nested_performance.csv');d=d[d.objective=='micro'].set_index('selector');keys=['independent_selected','depth_layer_5','depth_top_2','depth_all_4'];names=['Automatic\nselector','Top layer','Top two','Four layers'];x=np.arange(4)
    a.bar(x-.17,[d.loc[k,'micro_f1'] for k in keys],.32,color=C['blue'],label='Micro F1');a.bar(x+.17,[d.loc[k,'macro_f1'] for k in keys],.32,color=C['teal'],label='Macro F1')
    a.set_xticks(x,names);a.set(ylim=(0,1),ylabel='Pooled held-out F1');a.legend(frameon=False,fontsize=8);title(a,'a','Nested selection results')
    a=f.add_subplot(gs[0,1]);d=pd.read_csv(DATA/'per_species.csv');d=d[(d.selector=='independent_selected')&(d.objective=='micro')].copy();x=np.arange(len(d));a.bar(x,d.f1,color=[C['gray'] if n<5 else C['blue'] for n in d.support]);a.set_xticks(x,[s+('*' if n<5 else '') for s,n in zip(d.species,d.support)],rotation=45,ha='right');a.set(ylim=(0,1.15),ylabel='Held-out F1')
    for i,r in enumerate(d.itertuples()):a.text(i,r.f1+.025,str(r.support),ha='center',fontsize=8)
    a.text(.02,.95,'Counts = positive configurations; * fewer than 5',transform=a.transAxes,fontsize=8);title(a,'b','Automatic selector by species')
    a=f.add_subplot(gs[1,0]);d=pd.read_csv(DATA/'layer_sensitivity.csv');order=['layer_5','layer_4','layer_3','layer_2','top_2','top_3','all_4'];labels=['Top','Layer 4','Layer 3','Layer 2','Top 2','Top 3','Four']
    for metric,c,dx in [('micro',C['blue'],-.05),('macro',C['teal'],.05)]:
        s=d[d.metric==metric].set_index('depth').loc[order];a.errorbar(np.arange(7)+dx,s.f1,yerr=[s.f1-s.ci_low,s.ci_high-s.f1],fmt='o-',color=c,capsize=3,label=metric.title())
    a.set_xticks(range(7),labels,rotation=35,ha='right');a.set(ylim=(0,1),ylabel='Held-out F1');a.legend(frameon=False,fontsize=8);title(a,'c','Depth sensitivity with conditional intervals')
    a=f.add_subplot(gs[1,1]);d=pd.read_csv(DATA/'feature_model_depth_performance.csv');d=d[(d.depth=='layer_5')&(d.objective=='micro')];feats=['normalized','CDF','derivative','shape_CDF_derivatives','DCT24'];models=['logistic','SVM','random_forest','extra_trees','independent_trees'];table=d.pivot(index='model',columns='feature',values='micro_f1').reindex(index=models,columns=feats);im=a.imshow(table,aspect='auto',vmin=.3,vmax=.85,cmap='viridis')
    for i in range(5):
        for j in range(5):a.text(j,i,f'{table.iloc[i,j]:.2f}',ha='center',va='center',color='white' if table.iloc[i,j]<.65 else '#172b35',fontsize=8)
    a.set_xticks(range(5),['Shape','CDF','d1','Combined','DCT24'],rotation=30,ha='right');a.set_yticks(range(5),['Logistic','SVM','RF','Extra trees','Independent']);title(a,'d','Exploratory top-layer micro F1')
    f.text(.5,.01,'Five outer configuration folds; training-only feature/threshold selection. Intervals condition on fitted predictions.',ha='center',fontsize=9)
    save(f,'Figure_4_ML')
def fig5():
    f,axs=plt.subplots(2,2,figsize=(9,8));rows=[]
    for ax,letter,(family,region) in zip(axs.flat,'abcd',[('2CO_adjacent','interface'),('OCCO','interface'),('2CO_adjacent','interior'),('OCCO','interior')]):
        file=f'{family}_{region}.vasp';cc=surface(ax,file,zoom=True);label='2CO' if family=='2CO_adjacent' else 'OCCO';title(ax,letter,f'{label} · '+('Cu-side interface' if region=='interface' else 'Cu terrace'))
        ax.text(.02,.02,f'Input C–C distance: {cc:.2f} Å',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',edgecolor='none',alpha=.9))
        rows.append(dict(file=file,minimum_input_CC_distance_A=cc,stage='input'))
    f.subplots_adjust(hspace=.25,wspace=.12,top=.92,bottom=.10)
    f.suptitle('Matched 2CO/OCCO comparison design',fontsize=14,fontweight='bold')
    f.text(.5,.045,'Actual input coordinates, top view · 2 C and 2 O per panel · relaxed matched spectra and discrimination pending',ha='center',fontsize=9)
    pd.DataFrame(rows).to_csv(DATA/'Figure_5_input_distances.csv',index=False);save(f,'Figure_5_CO_coupling')
if __name__=='__main__':
    for fn in [fig1,fig2,fig3,fig4,fig5]:fn()
    print('Figures 1–5 written to',FIG)
