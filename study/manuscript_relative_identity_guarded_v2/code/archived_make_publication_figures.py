"""Publication figures from recorded spectra, predictions and atom coordinates.

Run in the extracted package: python code/make_figures.py
All panels are code-native; sphere shading preserves the supplied coordinates.
The reference panels use supplied numeric inputs processed in study/reference_benchmark.
Legacy vector-path rendering data are archival context only.
"""
from pathlib import Path
import json, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch, Ellipse, Rectangle, Arc
from matplotlib.lines import Line2D

ROOT=Path(__file__).resolve().parents[1]; DATA=ROOT/'figure_data'; FIG=ROOT/'figures'
P={'blue':'#4477AA','orange':'#EE7733','teal':'#228833','navy':'#15334F','gray':'#7D8790','light':'#D6DEE6',
   'Cu':'#4169B3','Au':'#E2B346','C':'#414B57','O':'#D84C48','H':'#EDEEF1'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,'axes.titlesize':10,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'axes.linewidth':.7,'xtick.major.width':.7,'ytick.major.width':.7,'legend.frameon':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def save(fig,name):
    FIG.mkdir(exist_ok=True)
    for ext in ['pdf','svg','png']:
        temp=FIG/f'.{name}.{ext}.tmp'
        fig.savefig(temp,format=ext,dpi=450,bbox_inches='tight',pad_inches=.06)
        temp.replace(FIG/f'{name}.{ext}')
    plt.close(fig)
def panel(ax,letter,heading):ax.set_title(f'{letter}  {heading}',loc='left',fontweight='bold',pad=9)
def blank(ax):
    ax.set_xticks([]);ax.set_yticks([])
    for s in ax.spines.values():s.set_visible(False)
def sphere(ax,x,y,r,element,z=1):
    """Vector concentric discs approximate a lit sphere in data coordinates."""
    base=np.array(mcolors.to_rgb(P[element]));edge=base*.35
    ax.add_patch(Circle((x,y),r,facecolor=base*.58,edgecolor=edge,lw=.25,zorder=z))
    for i,t in enumerate(np.linspace(.98,.04,32)):
        center=np.array([x,y])+r*np.array([-.27,.30])*(1-t)
        w=(1-t)**1.7*.82;col=base*(1-w)+np.ones(3)*w
        ax.add_patch(Circle(center,r*t,facecolor=col,edgecolor='none',zorder=z+(i+1)*.0005))

def load_poscar(path):
    ls=Path(path).read_text().splitlines();scale=float(ls[1]);cell=np.array([list(map(float,s.split())) for s in ls[2:5]])*scale
    sp=np.repeat(ls[5].split(),list(map(int,ls[6].split())));i=7
    if ls[i].lower().startswith('s'):i+=1
    direct=ls[i].lower().startswith('d');i+=1
    q=np.array([list(map(float,s.split()[:3])) for s in ls[i:i+len(sp)]])
    frac=q if direct else q*scale@np.linalg.inv(cell)
    return sp,frac@cell,cell,frac

def project(xyz,elevation=65,azimuth=0):
    a=np.deg2rad(azimuth);e=np.deg2rad(elevation)
    x=xyz[:,0]*np.cos(a)-xyz[:,1]*np.sin(a);y=xyz[:,0]*np.sin(a)+xyz[:,1]*np.cos(a)
    return np.column_stack([x,y*np.sin(e)+xyz[:,2]*np.cos(e)]),-y*np.cos(e)+xyz[:,2]*np.sin(e)

def structure(ax,file,local=False,substrate_only=False):
    sp,xyz,cell,frac=load_poscar(DATA/'structures'/file);metal=np.isin(sp,['Cu','Au']);ads=~metal
    # Include the two outer layers, then let depth sorting provide occlusion.
    selected=(metal&(xyz[:,2]>xyz[metal,2].max()-4.1))|ads
    if substrate_only:selected &= metal
    center=xyz[metal if substrate_only else ads].mean(axis=0);q=xyz.copy();q[:,:2]-=center[:2]
    if local:
        # Periodic in-plane copies prevent a cropped cell edge from looking vacant.
        qq=[];ss=[]
        for i in [-1,0,1]:
            for j in [-1,0,1]:
                shift=i*cell[0]+j*cell[1];qq.append(q[selected]+shift);ss.extend(sp[selected])
        q=np.concatenate(qq);sp2=np.array(ss);m=np.isin(sp2,['Cu','Au']);keep=(abs(q[:,0])<6)&(abs(q[:,1])<5)
        q=q[keep];sp2=sp2[keep]
    else:q=q[selected];sp2=sp[selected]
    q[:,2]-=xyz[metal,2].max();xy,depth=project(q,65,0)
    rad={'Cu':1.19,'Au':1.28,'C':.67,'O':.65,'H':.40}
    ai=np.where(~np.isin(sp2,['Cu','Au']))[0]
    for i,a in enumerate(ai):
        for b in ai[:i]:
            if sp2[a]==sp2[b]=='O':continue
            cutoff=1.85 if sp2[a]==sp2[b]=='C' else 1.7
            if np.linalg.norm(q[a]-q[b])<cutoff:
                ax.plot(xy[[a,b],0],xy[[a,b],1],color='#515B66',lw=3.5,zorder=min(depth[a],depth[b])+49.99,solid_capstyle='round')
    for n in np.argsort(depth):sphere(ax,*xy[n],rad[sp2[n]],sp2[n],z=float(depth[n]+50))
    if local:
        ad=np.where(~np.isin(sp2,['Cu','Au']))[0];mid=xy[ad].mean(axis=0)
        ax.set_xlim(mid[0]-4.4,mid[0]+4.4);ax.set_ylim(mid[1]-3.8,mid[1]+3.5)
    else:
        ax.set_xlim(xy[:,0].min()-1.5,xy[:,0].max()+1.5);ax.set_ylim(xy[:,1].min()-1.5,xy[:,1].max()+1.6)
    ax.set_aspect('equal');blank(ax)
    ids=np.where(sp=='C')[0];dist=[]
    for i,a in enumerate(ids):
        for b in ids[:i]:
            d=frac[a]-frac[b];d[:2]-=np.round(d[:2]);dist.append(np.linalg.norm(d@cell))
    return min(dist) if dist else np.nan

def molecule(ax,name):
    d=json.loads((DATA/'adsorbate_illustrations.json').read_text())[name]
    sp=d['atoms'];q=np.asarray(d['xy'],float);q=q-q.mean(axis=0);r={'C':.30,'O':.30,'H':.30}
    for i,j in d['bonds']:ax.plot(q[[i,j],0],q[[i,j],1],lw=2.0,color='#69727B',zorder=1,solid_capstyle='round')
    for i,el in enumerate(sp):sphere(ax,*q[i],r[el],el,z=2+i*.01)
    # One fixed viewport for every molecule prevents isolated atoms being enlarged.
    ax.set(xlim=(-1.9,1.9),ylim=(-1.9,1.9));ax.set_aspect('equal');blank(ax)
    ax.text(.5,-.06,name.replace('CO2','CO₂'),transform=ax.transAxes,ha='center',fontsize=8)


def interface_sites(ax,legend_ax):
    """One 3D scene and common oblique projection for surface and vertical CO."""
    points=[];elements=[];lookup={}
    for row in range(7):
        for col in range(7):
            x=(col-3)*2.5+(row%2)*1.25;y=row*2.5*np.sqrt(3)/2
            lookup[row,col]=len(points);points.append([float(x),float(y),0.])
            elements.append('Cu' if x<.625 else 'Au')
    points=np.asarray(points);sites=[];view=dict(elevation=52,azimuth=0)
    choices=[('Cu atop',[(1,1)]),('Cu–Cu bridge',[(4,1),(4,2)]),
             ('Cu–Au bridge: across boundary',[(1,2),(1,3)]),
             ('CuCuAu hollow',[(3,2),(3,3),(4,3)]),('Au atop',[(4,5)]),
             ('CuCuCu hollow',[(5,0),(5,1),(6,1)]),
             ('CuAuAu hollow',[(4,3),(4,4),(5,3)])]
    for k,(label,contacts) in enumerate(choices,1):
        ids=[lookup[q] for q in contacts];anchor=points[ids].mean(axis=0);anchor[2]=1.85
        sites.append(dict(number=k,label=label,contact_indices=ids,contact_elements=[elements[i] for i in ids],C_xyz=anchor.tolist(),O_xyz=(anchor+[0,0,1.16]).tolist()))
    xy,depth=project(points,**view)
    for n in np.argsort(depth):sphere(ax,*xy[n],1.11,elements[n],z=float(depth[n]+25))
    boundary,_=project(np.array([[.625,-1.4,0],[.625,14.1,0]]),**view)
    ax.plot(boundary[:,0],boundary[:,1],ls=(0,(3,2)),lw=1,c='#48576A',zorder=45)
    ax.text(-4.8,13.5,'Cu',ha='center',color=P['Cu'],fontweight='bold',fontsize=9)
    ax.text(5,13.5,'Au',ha='center',color='#9A741B',fontweight='bold',fontsize=9)
    ax.text(.625,13.5,'Boundary',ha='center',color=P['navy'],fontsize=6.3)
    for site in sites:
        molecule_xyz=np.array([site['C_xyz'],site['O_xyz']]);q,md=project(molecule_xyz,**view)
        for i in site['contact_indices']:
            ax.plot([q[0,0],xy[i,0]],[q[0,1],xy[i,1]],ls=(0,(2,1)),c='white',lw=2.2,zorder=47)
            ax.plot([q[0,0],xy[i,0]],[q[0,1],xy[i,1]],ls=(0,(2,1)),c=P['navy'],lw=1,zorder=48)
        ax.plot(q[:,0],q[:,1],lw=2,c=P['navy'],zorder=49)
        for pos,el in zip(q,['C','O']):
            ax.add_patch(Circle(pos,.43,fc='white',ec='white',zorder=50))
            sphere(ax,*pos,.38,el,z=51)
        tag=q[1]+[-.75,.12]
        ax.add_patch(Circle(tag,.34,fc=P['navy'],ec='white',lw=.6,zorder=55))
        ax.text(*tag,str(site['number']),ha='center',va='center',fontsize=6.2,color='white',fontweight='bold',zorder=56)
    ax.set(xlim=(-9,10),ylim=(-1.8,14.4));ax.set_aspect('equal');blank(ax)
    panel(ax,'c','Local coordination at the interface')
    blank(legend_ax);legend_ax.axis('off')
    for i,site in enumerate(sites):
        label=site['label'].replace('Cu–Au bridge: across boundary','Cu–Au bridge (across boundary)')
        legend_ax.text(.02,.98-i*.14,f"{site['number']}  {label}",transform=legend_ax.transAxes,va='top',fontsize=7.3)
    (DATA/'interface_site_illustration.json').write_text(json.dumps(dict(status='ideal schematic surface; not relaxed or used for inference',surface_xyz=points.tolist(),surface_elements=elements,boundary_x=.625,projection=view,sites=sites),indent=2)+'\n')

def workflow_icon(ax,index):
    ax.set(xlim=(-1,1),ylim=(-.85,.85));ax.set_facecolor('none');ax.set_aspect('equal');blank(ax)
    navy=P['navy'];teal='#008C91'
    if index==0:
        ax.add_patch(FancyBboxPatch((-.75,-.35),1.5,.9,boxstyle='round,pad=.02,rounding_size=.2',fill=False,ec=navy,lw=1.7))
        ax.plot([-.45,-.62,-.14],[-.35,-.62,-.35],color=navy,lw=1.7)
        for x in [-.35,0,.35]:ax.add_patch(Circle((x,.1),.07,color=P['blue']))
    elif index==1:
        for j in range(4):
            for i in range(6):sphere(ax,(i-2.5)*.28+(j%2)*.13,(j-1.5)*.26,.16,'Au' if j>1 else 'Cu',z=j+2)
        sphere(ax,-.1,.0,.11,'C',10);sphere(ax,-.1,.22,.11,'O',11)
    elif index==2:
        for y in [-.42,-.02,.38]:
            ax.add_patch(FancyBboxPatch((-.72,y),1.44,.28,boxstyle='round,pad=.015',facecolor='#EAF0F6',edgecolor=navy,lw=1.2))
            for x in [.35,.5,.65]:ax.add_patch(Circle((x,y+.14),.025,color=teal))
        ax.text(0,-.69,'NERSC HPC',ha='center',fontweight='bold',fontsize=8,color=navy)
    elif index==3:
        ax.add_patch(Rectangle((-.6,-.45),1.2,.92,fc='#E6F3F2',ec=teal,lw=1.4))
        for y in [-.45,-.15,.15,.47]:ax.add_patch(Ellipse((0,y),1.2,.28,fc='white' if y==.47 else '#E6F3F2',ec=teal,lw=1.4))
    elif index==4:
        nodes=[(-.65,y) for y in [-.4,0,.4]]+[(0,y) for y in [-.5,-.17,.17,.5]]+[(.65,y) for y in [-.25,.25]]
        for i in range(3):
            for j in range(3,7):ax.plot([nodes[i][0],nodes[j][0]],[nodes[i][1],nodes[j][1]],c='#A2B5C5',lw=.7)
        for i in range(3,7):
            for j in range(7,9):ax.plot([nodes[i][0],nodes[j][0]],[nodes[i][1],nodes[j][1]],c='#A2B5C5',lw=.7)
        for x,y in nodes:ax.add_patch(Circle((x,y),.09,fc=P['blue'],ec=navy,lw=.5))
    else:
        x=np.linspace(-.8,.55,120);y=-.4+.7*np.exp(-((x+.3)/.13)**2)+.35*np.exp(-((x-.2)/.12)**2)
        ax.plot(x,y,c=teal,lw=1.4);ax.plot([-.85,.7],[-.55,-.55],c=navy,lw=.7)
        ax.add_patch(Circle((.35,.02),.3,fill=False,ec=navy,lw=1.7));ax.plot([.56,.86],[-.21,-.55],c=navy,lw=3)

def fig1():
    f=plt.figure(figsize=(7.1,7.1));ax=f.add_axes([0,0,1,1]);ax.set(xlim=(-4.4,4.4),ylim=(-4.4,4.4));ax.set_aspect('equal');ax.axis('off')
    theta=np.deg2rad([90,30,-30,-90,-150,150]);R=3.0;r=1.06;centers=np.column_stack([R*np.cos(theta),R*np.sin(theta)])
    texts=['Natural-language\nrequest','Catalyst\nmodels','Automated\ncalculations','AI-ready records\nand data portal','ML training\nand advisor','CO₂RR\nhypotheses']
    for i,(x,y) in enumerate(centers):
        col=P['blue'] if i%2==0 else '#008C91'
        ax.add_patch(Circle((x,y),r,fc='#F6FAFC',ec=col,lw=2.4))
        ax.add_patch(Circle((x,y+r-.05),.19,fc=col,ec='white',lw=.6));ax.text(x,y+r-.05,str(i+1),color='white',ha='center',va='center',fontsize=11,fontweight='bold')
        ax.text(x,y+.52,texts[i],ha='center',va='center',fontsize=9.5,fontweight='bold',color=col,linespacing=1.12)
        ia=f.add_axes([.5+(x-r*.74)/8.8,.5+(y-r*.66)/8.8,1.48*r/8.8,.90*r/8.8]);workflow_icon(ia,i)
        details=['Controlled alternatives','Site · coverage · distance','Relaxation → XAS','Structure + absorbers','Grouped validation','Test C–C coupling']
        ax.text(x,y-.70,details[i],ha='center',va='center',fontsize=7.2,color=P['navy'])
        j=(i+1)%6;a=centers[i];b=centers[j];direction=(b-a)/np.linalg.norm(b-a)
        ax.add_patch(FancyArrowPatch(a+direction*r,b-direction*r,connectionstyle='arc3,rad=-.14',arrowstyle='-|>',mutation_scale=18,lw=2.4,color=col))
    ax.text(0,.22,'XAS Agent',ha='center',fontsize=25,fontweight='bold',color=P['navy'])
    ax.text(0,-.37,'Catalytic interfaces',ha='center',fontsize=12,color=P['navy'])
    ax.text(0,-.70,'from structures to inference',ha='center',fontsize=9,color=P['gray'])
    save(f,'Figure_1_framework')

def fig2():
    f=plt.figure(figsize=(7.2,9.1));gs=f.add_gridspec(2,2,height_ratios=[1,1.55],width_ratios=[1,1.1],hspace=.35,wspace=.28)
    a=f.add_subplot(gs[0,0]);structure(a,'2CO_adjacent_interface.vasp',substrate_only=True);panel(a,'a','Substrate and adsorbates')
    a.text(.5,-.04,'Cu–Au substrate interface',transform=a.transAxes,ha='center',fontsize=8)
    f.legend(handles=[Line2D([],[],ls='',marker='o',ms=5.5,color=P[z],label=z) for z in ['Cu','Au','C','O','H']],ncol=5,loc='center',bbox_to_anchor=(.54,.586),fontsize=8,handletextpad=.25,columnspacing=1.4)
    sub=gs[0,1].subgridspec(3,4,hspace=.38,wspace=.20)
    for k,name in enumerate(['CO','CO2','CHO','COH','COOH','OCHO','COCHO','OCCOH','OCCO','H','O','OH']):molecule(f.add_subplot(sub[k//4,k%4]),name)
    inv=pd.read_csv(DATA/'all_unique_structures.csv');ml=pd.read_csv(DATA/'cv_manifest.csv')
    singles=['CO','H','COOH','OCHO','O','COH','CHO','OH','OCCOH','CO2','COCHO']
    multi=['CO;H','CO;OH','CHO;CO;H','CO;COH','CO;OCCO','CHO;CO']
    rows=[];sub=gs[1,0].subgridspec(2,1,height_ratios=[11,6],hspace=.33)
    for k,(cats,title) in enumerate([(singles,'Single species'),(multi,'Coadsorbed species')]):
        a=f.add_subplot(sub[k]);y=np.arange(len(cats))
        alln=[int((inv.labels==c).sum()) for c in cats];used=[int((ml.labels==c).sum()) for c in cats]
        for c,x,z in zip(cats,alln,used):rows.append(dict(category='single' if k==0 else 'coadsorbed',identity_set=c,inventory=x,used_for_ML=z))
        a.barh(y,alln,height=.75,color=P['light'],label='All structures')
        a.barh(y,used,height=.38,color=P['blue'],label='Used for ML')
        for j,n in enumerate(alln):a.text(n+.5,j,str(n),va='center',fontsize=6.7,color='#6C7781')
        a.set_yticks(y,[c.replace(';',' + ').replace('CO2','CO₂') for c in cats],fontsize=7.2);a.invert_yaxis();a.set_xlim(0,35);a.set_xticks([0,10,20,30]);a.tick_params(axis='y',length=0)
        a.set_title(title,loc='left',fontsize=8.3,fontweight='bold',pad=5)
        if k==0:
            a.text(0,1.17,'b  Adsorbate populations',transform=a.transAxes,fontsize=10,fontweight='bold')
            a.legend(fontsize=6.8,loc='lower right',handlelength=1,labelspacing=.3)
        else:a.set_xlabel('Independent configurations',fontsize=8)
    counts=pd.DataFrame(rows)
    assert counts.inventory.sum()==152 and counts.used_for_ML.sum()==87
    assert counts.query("category=='coadsorbed'").used_for_ML.sum()==9
    counts.to_csv(DATA/'adsorbate_configuration_counts.csv',index=False)
    sub=gs[1,1].subgridspec(2,1,height_ratios=[3.6,1.6],hspace=.04)
    interface_sites(f.add_subplot(sub[0]),f.add_subplot(sub[1]))
    save(f,'Figure_2_dataset')

def fig3():
    f=plt.figure(figsize=(7.2,5.2));gs=f.add_gridspec(2,1,height_ratios=[1.35,1],hspace=.60,top=.80,bottom=.10)
    ref=gs[0].subgridspec(1,3,wspace=.30)
    d=pd.read_csv(DATA/'reference_benchmark_v2'/'plotted_arrays.csv')
    colors={'fdmnes_green':'#0072B2','fdmnes_fdm':'#D55E00','feff':'#009E73','vasp_pbe':'#CC79A7','vasp_gw':'#E69F00'}
    labels={'fdmnes_green':'FDMNES green','fdmnes_fdm':'FDMNES fdm','feff':'FEFF','vasp_pbe':'VASP PBE','vasp_gw':'VASP GW'}
    handles=[Line2D([],[],color='black',lw=1.5,label='Experiment')]+[Line2D([],[],color=c,lw=1.2,label=labels[m]) for m,c in colors.items()]
    handles.append(Line2D([],[],color=colors['fdmnes_fdm'],lw=1.2,ls='--',label='CuO fdm: supplied single site'))
    f.text(.125,.98,'a  Reference benchmark: Cu-calibrated alignment',fontweight='bold',fontsize=10)
    f.legend(handles=handles,loc='center',bbox_to_anchor=(.54,.895),ncol=3,fontsize=7.1,handlelength=2,columnspacing=1.2,labelspacing=.45)
    display=[]
    for i,material in enumerate(['Cu','Cu2O','CuO']):
        a=f.add_subplot(ref[0,i]);subset=d[(d.material==material)&(d.comparison_type=='Cu_calibrated')]
        first=True
        for method,col in colors.items():
            q=subset[subset.method==method]
            if q.empty:continue
            if first:
                a.plot(q.energy_eV,q.experiment,color='black',lw=1.5);first=False
                display.extend(dict(material=material,method='experiment',energy_eV=e,intensity_au=v) for e,v in zip(q.energy_eV,q.experiment))
            a.plot(q.energy_eV,q.calculation,color=col,lw=1.15,ls='--' if material=='CuO' and method=='fdmnes_fdm' else '-')
            display.extend(dict(material=material,method=method,energy_eV=e,intensity_au=v) for e,v in zip(q.energy_eV,q.calculation))
        a.set(xlim=(8980,9020),ylim=(-.035,1.08),xlabel='Photon energy (eV)');a.set_xticks([8980,9000,9020]);a.set_yticks([0,.5,1]);a.ticklabel_format(axis='x',style='plain',useOffset=False);a.tick_params(labelsize=7)
        if i==0:a.set_ylabel('Intensity (a.u.)')
        a.set_title({'Cu':'Cu','Cu2O':'Cu₂O','CuO':'CuO'}[material],loc='left',fontweight='bold',pad=8)
    pd.DataFrame(display).to_csv(DATA/'Figure_3a_display.csv',index=False)
    a=f.add_subplot(gs[1]);d=pd.read_csv(DATA/'representative_layer_spectra.csv');d=d[d.configuration=='CO/CO_0p11/interface_biased']
    for l,col in zip([5,4,3,2],[P['orange'],P['blue'],'#228833','#AA3377']):
        q=d[d.layer==l];a.plot(q.energy_eV,q.absorption,lw=1.7,color=col,label='Surface' if l==5 else f'Layer {l}')
    a.set(xlim=(8974,9029),xlabel='Photon energy (eV)',ylabel='Intensity (a.u.)');a.legend(fontsize=7.5,loc='lower right');a.ticklabel_format(axis='x',style='plain',useOffset=False);panel(a,'b','Layer-resolved Cu K edge')
    save(f,'Figure_3_spectroscopy')

def fig4():
    f=plt.figure(figsize=(7.2,6.5));gs=f.add_gridspec(2,2,hspace=.46,wspace=.30)
    a=f.add_subplot(gs[0,0]);d=pd.read_csv(DATA/'nested_performance.csv').query('objective=="micro"').set_index('selector');keys=['independent_selected','depth_layer_5','depth_top_2','depth_all_4'];x=np.arange(4)
    for delta,col,metric,lab in [(-.18,P['blue'],'micro_f1','Micro F1'),(.18,P['orange'],'macro_f1','Macro F1')]:
        v=[d.loc[k,metric] for k in keys];a.bar(x+delta,v,.34,color=col,label=lab)
    a.set_xticks(x,['Primary\nselector','Surface','Top two','Four layers']);a.set(ylim=(0,1),ylabel='Held-out F1');a.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.03),handlelength=1);panel(a,'a','Automated inference')
    a=f.add_subplot(gs[0,1]);d=pd.read_csv(DATA/'per_species.csv').query('selector=="independent_selected" and objective=="micro"');x=np.arange(len(d));a.bar(x,d.f1,color=[P['blue'] if n>=5 else '#A8B5C3' for n in d.support],width=.72)
    a.set_xticks(x,[s.replace('CO2','CO₂')+('*' if n<5 else '') for s,n in zip(d.species,d.support)],rotation=50,ha='right');a.set(ylim=(0,1.12),ylabel='Held-out F1')
    for i,q in enumerate(d.itertuples()):a.text(i,q.f1+.035,str(q.support),ha='center',fontsize=7.5)
    panel(a,'b','Adsorbate identification')
    sub=gs[1,0].subgridspec(1,2,wspace=.25);d=pd.read_csv(DATA/'layer_sensitivity.csv')
    groups=[(['layer_5','layer_4','layer_3','layer_2'],'Individual layers','Layer from surface',[P['blue'],P['orange']]),
            (['layer_5','top_2','top_3','all_4'],'Top-n-layer average','Number of top layers, n',['#008577','#AA3377'])]
    for g,(order,title,xlabel,cols) in enumerate(groups):
        a=f.add_subplot(sub[g]);x=np.arange(1,5)
        for (metric,dx,marker),col in zip([('micro',-.06,'o'),('macro',.06,'s')],cols):
            q=d[d.metric==metric].set_index('depth').loc[order]
            a.errorbar(x+dx,q.f1,yerr=[q.f1-q.ci_low,q.ci_high-q.f1],fmt=marker+'-',ms=5.1,mew=.5,mec='white',lw=1.4,elinewidth=.7,capsize=2,color=col,label=metric.title()+' F1')
        a.set_xticks(x);a.set(xlim=(.7,4.3),ylim=(0,1),xlabel=xlabel);a.set_title(title,fontsize=7.8,pad=7,fontweight='bold');a.set_xlabel(xlabel,fontsize=7.1);a.tick_params(labelsize=7)
        a.set_yticks([0,.2,.4,.6,.8,1]);a.set_axisbelow(True);a.yaxis.grid(True,color='#E6E9ED',lw=.5)
        if g==0:
            a.set_ylabel('Held-out F1');a.text(0,1.21,'c  Depth sensitivity',transform=a.transAxes,fontsize=10,fontweight='bold')
        else:a.set_yticklabels([])
        a.legend(fontsize=6.3,loc='lower left',handlelength=1.5,labelspacing=.3,borderaxespad=.3)
    a=f.add_subplot(gs[1,1]);d=pd.read_csv(DATA/'feature_model_depth_performance.csv').query('depth=="layer_5" and objective=="micro"');features=['normalized','CDF','derivative','shape_CDF_derivatives','DCT24'];models=['logistic','SVM','random_forest','extra_trees','independent_trees'];v=d.pivot(index='model',columns='feature',values='micro_f1').reindex(index=models,columns=features)
    cmap=mcolors.LinearSegmentedColormap.from_list('gray_to_green',['#E5E5E5','#CFDCD3','#80B89B','#137548']);a.imshow(v,aspect='auto',vmin=float(v.min().min()),vmax=float(v.max().max()),cmap=cmap)
    for i in range(5):
        for j in range(5):a.text(j,i,f'{v.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if v.iloc[i,j]>.745 else '#1F2E3C')
    a.set_xticks(range(5),['Normalized energy\nspectrum','CDF','d1','Combined','DCT24'],rotation=40,ha='right');a.set_yticks(range(5),['Logistic','SVM','RF','Extra trees','Independent'],fontsize=7.5);panel(a,'d','Feature–model comparison')
    save(f,'Figure_4_ML')

def fig5():
    f,axs=plt.subplots(2,2,figsize=(7.2,6.4),gridspec_kw={'hspace':.30,'wspace':.08});rows=[]
    for ax,letter,(family,region) in zip(axs.flat,'abcd',[('2CO_adjacent','interface'),('OCCO','interface'),('2CO_adjacent','interior'),('OCCO','interior')]):
        cc=structure(ax,f'{family}_{region}.vasp',local=True);name='2CO' if family=='2CO_adjacent' else 'OCCO';panel(ax,letter,f'{name} · '+('interface' if region=='interface' else 'terrace'))
        ax.text(.5,-.055,f'C–C = {cc:.2f} Å',ha='center',va='top',transform=ax.transAxes,fontsize=9);rows.append(dict(family=family,region=region,input_CC_A=cc))
    pd.DataFrame(rows).to_csv(DATA/'Figure_5_input_distances.csv',index=False);save(f,'Figure_5_CO_coupling')

def main():
    for fn in [fig1,fig2,fig3,fig4,fig5]:fn()
    print('Rebuilt five publication figure sets')
if __name__=='__main__':main()
