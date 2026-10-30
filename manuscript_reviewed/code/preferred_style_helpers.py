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
    for ext in ['png']:
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

def substrate_views(ax_top,ax_side):
    """ASE-read geometry, shaded consistently with the other structure panels.

    Top: actual outermost metal layer. Side: every metal layer, viewed along
    the boundary. Both retain the same Cu-left/Au-right orientation. No atom
    positions or chemical identities are changed to improve the appearance.
    """
    from ase.io import read
    import ase
    path=DATA/'structures'/'2CO_adjacent_interface.vasp'
    atoms=read(path,format='vasp')
    original_positions=atoms.positions.copy()
    # Equivalent atoms at a periodic cell edge belong to the same domain.
    # Wrap in-plane for display; leave the vacuum coordinate unchanged.
    atoms.set_pbc([True,True,False])
    # A cut 0.5 angstrom below the original y origin keeps the slightly
    # relaxed Cu boundary row together instead of splitting it across images.
    atoms.wrap(center=(.5,.5-.5/atoms.cell.lengths()[1],.5))
    image_shifts=(atoms.positions-original_positions)@np.linalg.inv(atoms.cell.array)
    assert np.allclose(image_shifts[:,:2],np.round(image_shifts[:,:2]),atol=1e-10)
    assert np.allclose(image_shifts[:,2],0,atol=1e-10)
    metal_indices=np.flatnonzero(np.isin(atoms.get_chemical_symbols(),['Cu','Au']))
    z=np.sort(atoms.positions[metal_indices,2])
    gaps=np.flatnonzero(np.diff(z)>1.0)
    if not len(gaps):raise ValueError('Cannot identify a separated surface layer')
    threshold=float((z[gaps[-1]]+z[gaps[-1]+1])/2)
    indices=metal_indices[atoms.positions[metal_indices,2]>threshold]
    symbols=np.asarray(atoms.get_chemical_symbols())
    top_symbols=symbols[indices]
    if set(top_symbols)!={'Cu','Au'}:raise ValueError('Surface must contain both metals')
    # Rotate the viewing basis only: transverse y becomes horizontal, and
    # original x (parallel to the interface) becomes the viewing direction.
    positions=atoms.positions.copy()
    q=np.column_stack([positions[:,1],-positions[:,0],positions[:,2]])
    radius=1.18
    for ax,chosen,view in [(ax_top,indices,'top'),(ax_side,metal_indices,'side')]:
        local=q[chosen]
        if view=='top':xy=local[:,:2];depth=local[:,2]
        else:xy=local[:,[0,2]];depth=-local[:,1]
        for n in np.argsort(depth,kind='stable'):
            sphere(ax,*xy[n],radius,symbols[chosen[n]],z=float(depth[n]+100))
        ax.set_xlim(float(xy[:,0].min()-1.5),float(xy[:,0].max()+1.5))
        ax.set_ylim(float(xy[:,1].min()-1.5),float(xy[:,1].max()+1.5))
        ax.set_aspect('equal');blank(ax)
        ax.text(.5,-.045,'Top view' if view=='top' else 'Side view',
                transform=ax.transAxes,ha='center',va='top',fontsize=7.5)
    audit=dict(renderer='shared shaded-sphere renderer; geometry read with ASE',ase_version=ase.__version__,
               source=str(path.relative_to(ROOT)),view='top and side; Cu left and Au right',
               selection='outermost metal layer above highest interlayer gap > 1 angstrom',
               z_threshold_A=threshold,source_atom_indices_zero_based=indices.tolist(),
               surface_atom_counts={s:int(np.count_nonzero(top_symbols==s)) for s in ['Cu','Au']},
               side_view_atom_indices_zero_based=metal_indices.tolist(),
               side_view_atom_counts={s:int(np.count_nonzero(symbols[metal_indices]==s)) for s in ['Cu','Au']},
               coordinates_modified=False,
               display_periodic_wrapping='integer cell translations in x/y only; z unchanged',
               transverse_cell_cut_A=-.5,
               display_radius_A=radius)
    (DATA/'Figure_2a_surface_render.json').write_text(json.dumps(audit,indent=2)+'\n')


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
    ax.set(xlim=(-9,10),ylim=(-1.8,14.4));ax.set_aspect('equal');ax.set_anchor('N');blank(ax)
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
    left=gs[0,0].subgridspec(2,1,height_ratios=[1.35,.75],hspace=.26)
    a=f.add_subplot(left[0]);side=f.add_subplot(left[1]);substrate_views(a,side)
    # Panel labels use the common figure grid, independent of atom-axis aspect.
    f.text(.125,.925,'a  Substrate and adsorbates',fontsize=10,fontweight='bold')
    f.text(.125,.548,'b  Adsorbate populations',fontsize=10,fontweight='bold')
    f.text(.5439014202,.548,'c  Adsorption sites',fontsize=10,fontweight='bold')
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
            a.legend(fontsize=6.8,loc='lower right',handlelength=1,labelspacing=.3)
        else:a.set_xlabel('Independent configurations',fontsize=8)
    counts=pd.DataFrame(rows)
    assert counts.inventory.sum()==152 and counts.used_for_ML.sum()==87
    assert counts.query("category=='coadsorbed'").used_for_ML.sum()==9
    counts.to_csv(DATA/'adsorbate_configuration_counts.csv',index=False)
    sub=gs[1,1].subgridspec(2,1,height_ratios=[3.6,1.6],hspace=.04)
    interface_sites(f.add_subplot(sub[0]),f.add_subplot(sub[1]))
    save(f,'Figure_2_dataset')

def fig3_archived():
    f=plt.figure(figsize=(7.2,7.6));gs=f.add_gridspec(3,1,height_ratios=[1.15,.80,1.65],hspace=.51,top=.855,bottom=.08)
    ref=gs[0].subgridspec(1,3,wspace=.30)
    d=pd.read_csv(DATA/'reference_benchmark'/'plotted_arrays.csv')
    colors={'fdmnes_green':'#0072B2','fdmnes_fdm':'#D55E00','feff':'#009E73','vasp_pbe':'#CC79A7','vasp_gw':'#E69F00'}
    labels={'fdmnes_green':'FDMNES green','fdmnes_fdm':'FDMNES fdm','feff':'FEFF','vasp_pbe':'VASP PBE','vasp_gw':'VASP GW'}
    handles=[Line2D([],[],color='black',lw=1.5,label='Experiment')]+[Line2D([],[],color=c,lw=1.2,label=labels[m]) for m,c in colors.items()]
    f.text(.125,.98,'a  Reference benchmark',fontweight='bold',fontsize=10)
    f.legend(handles=handles,loc='center',bbox_to_anchor=(.54,.931),ncol=3,fontsize=7.3,handlelength=2,columnspacing=1.5,labelspacing=.45)
    display=[]
    for i,material in enumerate(['Cu','Cu2O','CuO']):
        a=f.add_subplot(ref[0,i]);subset=d[(d.material==material)&(d.view=='common')]
        first=True
        for method,col in colors.items():
            q=subset[subset.method==method]
            if q.empty:continue
            if first:
                div=float(q.experiment.max());a.plot(q.energy_eV,q.experiment/div,color='black',lw=1.5);first=False
                display.extend(dict(material=material,method='experiment',energy_eV=e,intensity_au=v/div,unit_max_divisor=div) for e,v in zip(q.energy_eV,q.experiment))
            div=float(q.calculation.max());a.plot(q.energy_eV,q.calculation/div,color=col,lw=1.15,ls='--' if material=='CuO' and method=='fdmnes_fdm' else '-')
            display.extend(dict(material=material,method=method,energy_eV=e,intensity_au=v/div,unit_max_divisor=div) for e,v in zip(q.energy_eV,q.calculation))
        a.set(xlim=(8980,9020),ylim=(-.25,1.10),xlabel='Photon energy (eV)');a.set_xticks([8980,9000,9020]);a.set_yticks([0,.5,1]);a.ticklabel_format(axis='x',style='plain',useOffset=False);a.tick_params(labelsize=7)
        if i==0:a.set_ylabel('Intensity (a.u.)')
        a.set_title({'Cu':'Cu','Cu2O':'Cu₂O','CuO':'CuO'}[material],loc='left',fontweight='bold',pad=8)
    pd.DataFrame(display).to_csv(DATA/'Figure_3a_display.csv',index=False)
    a=f.add_subplot(gs[1]);d=pd.read_csv(DATA/'representative_layer_spectra.csv');d=d[d.configuration=='CO/CO_0p11/interface_biased']
    for l,col in zip([5,4,3,2],[P['orange'],P['blue'],'#228833','#AA3377']):
        q=d[d.layer==l];a.plot(q.energy_eV,q.absorption,lw=1.7,color=col,label='Surface' if l==5 else f'Layer {l}')
    a.set(xlim=(8974,9029),xlabel='Photon energy (eV)',ylabel='Intensity (a.u.)');a.legend(fontsize=7.5,loc='lower right');a.ticklabel_format(axis='x',style='plain',useOffset=False);panel(a,'b','Layer-resolved Cu K edge')
    sub=gs[2].subgridspec(3,4,hspace=.30,wspace=.16);d=pd.read_csv(DATA/'spectral_identity_contrasts.csv');order=['CO','CO2','CHO','COH','COOH','OCHO','COCHO','OCCOH','H','O','OH'];scale=float(np.max(np.abs(d.iloc[:,1:].values)));x=d.energy_eV.values
    assert scale>0
    relative=d.copy();relative.iloc[:,1:]=relative.iloc[:,1:]/scale;relative.to_csv(DATA/'relative_spectral_identity_contrasts.csv',index=False)
    (DATA/'spectral_contrast_display.json').write_text(json.dumps(dict(definition='mean unit-area spectrum among configurations containing species minus mean among configurations not containing species',reference='species-absent configurations, not clean surface',display_scale='one common maximum absolute contrast across all species and energies',common_divisor_per_eV=scale,range=[-1,1],preserves_relative_amplitudes=True),indent=2)+'\n')
    for i,name in enumerate(order):
        a=f.add_subplot(sub[i//4,i%4]);y=relative[name].values
        a.axhline(0,color='#A9B0B8',lw=.6);a.fill_between(x,0,y,where=y>=0,color=P['blue'],alpha=.26,interpolate=True);a.fill_between(x,0,y,where=y<=0,color=P['orange'],alpha=.30,interpolate=True);a.plot(x,y,c=P['navy'],lw=1.05)
        a.set(xlim=(8974,9029),ylim=(-1.10,1.10));a.set_xticks([8980,9000,9020]);a.set_yticks([-1,0,1]);a.tick_params(labelsize=7)
        a.text(.95,.87,name.replace('CO2','CO₂'),transform=a.transAxes,ha='right',fontsize=8,fontweight='bold')
        if i%4:a.set_yticklabels([])
        if i<8:a.set_xticklabels([])
        else:a.set_xlabel('Photon energy (eV)',fontsize=7)
        if i==0:a.text(0,1.26,'c  Mean spectra: species present − absent',transform=a.transAxes,fontweight='bold',fontsize=10)
    a=f.add_subplot(sub[2,3]);blank(a);a.axis('off');a.text(.03,.91,'Relative contrast\nShared scale: −1 to 1\nReference: species absent\n(not a clean surface)',transform=a.transAxes,va='top',fontsize=7.1,linespacing=1.35)
    save(f,'Figure_3_spectroscopy')

def fig3():
    """Reprocessed references and visible agreement metrics; no false clean control."""
    f=plt.figure(figsize=(7.2,7.0))
    gs=f.add_gridspec(3,1,height_ratios=[1.05,1.75,.72],hspace=.62,
                     top=.855,bottom=.065)
    ref=gs[0].subgridspec(1,3,wspace=.30)
    root=ROOT/'reference_benchmark_v2'
    data=pd.read_csv(root/'plotted_arrays.csv')
    data=data[data.comparison_type=='peak_aligned']
    colors={'fdmnes_green':'#0072B2','fdmnes_fdm':'#D55E00','feff':'#009E73',
            'vasp_pbe':'#CC79A7','vasp_gw':'#C59000'}
    labels={'fdmnes_green':'FDMNES green','fdmnes_fdm':'FDMNES fdm',
            'feff':'FEFF','vasp_pbe':'VASP PBE','vasp_gw':'VASP GW'}
    styles={'fdmnes_green':'-','fdmnes_fdm':'--','feff':':',
            'vasp_pbe':'-.','vasp_gw':'-'}
    handles=[Line2D([],[],color='black',lw=1.5,label='Experiment')]+[
        Line2D([],[],color=colors[m],ls=styles[m],lw=1.25,label=labels[m])
        for m in colors]
    f.text(.125,.985,'a  Reference spectra aligned by the first peak',fontsize=10,fontweight='bold')
    f.legend(handles=handles,loc='center',bbox_to_anchor=(.54,.934),ncol=3,
             fontsize=7.3,handlelength=2.6,columnspacing=1.7)
    display=[]
    for i,material in enumerate(['Cu','Cu2O','CuO']):
        ax=f.add_subplot(ref[0,i]);subset=data[data.material==material]
        for j,(method,color) in enumerate(colors.items()):
            q=subset[subset.method==method]
            if q.empty:continue
            if not any(x['material']==material and x['method']=='experiment' for x in display):
                ax.plot(q.energy_eV,q.experiment,c='black',lw=1.5)
                display.extend(dict(material=material,method='experiment',energy_eV=e,
                                    intensity_au=v) for e,v in zip(q.energy_eV,q.experiment))
            ax.plot(q.energy_eV,q.calculation,c=color,ls=styles[method],lw=1.25)
            display.extend(dict(material=material,method=method,energy_eV=e,
                                intensity_au=v) for e,v in zip(q.energy_eV,q.calculation))
        limits=(8975,9015) if material=='Cu2O' else (8980,9020)
        ax.set(xlim=limits,ylim=(-.025,1.10),xlabel='Photon energy (eV)')
        ax.set_xticks([limits[0],sum(limits)/2,limits[1]]);ax.set_yticks([0,.5,1])
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        ax.tick_params(labelsize=7)
        ax.set_title({'Cu':'Cu','Cu2O':'Cu₂O','CuO':'CuO'}[material],
                     loc='left',fontweight='bold',pad=7)
        if i==0:ax.set_ylabel('Intensity (a.u.)')
    pd.DataFrame(display).to_csv(DATA/'Figure_3a_display.csv',index=False)
    layers=pd.read_csv(DATA/'representative_layer_spectra.csv')
    layers=layers[layers.configuration=='CO/CO_0p11/interface_biased']
    surface=layers[layers.layer==5].sort_values('energy_eV')
    sub=gs[2].subgridspec(1,3,wspace=.18);contrasts=[]
    for i,(layer,color) in enumerate(zip([4,3,2],[P['blue'],'#228833','#AA3377'])):
        ax=f.add_subplot(sub[0,i]);q=layers[layers.layer==layer].sort_values('energy_eV')
        np.testing.assert_allclose(q.energy_eV,surface.energy_eV)
        delta=q.absorption.to_numpy()-surface.absorption.to_numpy()
        ax.axhline(0,color='#9299A0',lw=.6)
        ax.plot(q.energy_eV,delta,c=color,lw=1.6)
        ax.fill_between(q.energy_eV,0,delta,color=color,alpha=.12)
        ax.set(xlim=(8974,9029),ylim=(-.06,.06),xlabel='Photon energy (eV)')
        ax.set_title(f'Layer {layer} − surface',fontsize=8,pad=6)
        ax.set_xticks([8980,9000,9020]);ax.set_yticks([-.05,0,.05]);ax.tick_params(labelsize=7)
        ax.ticklabel_format(axis='x',style='plain',useOffset=False)
        if i==0:
            ax.set_ylabel('Δ intensity (a.u.)')
            ax.text(0,1.39,'c  Depth contrast relative to the surface',transform=ax.transAxes,
                    fontweight='bold',fontsize=10,ha='left')
        else:ax.tick_params(labelleft=False)
        contrasts.extend(dict(layer=layer,reference_layer=5,energy_eV=float(e),delta_intensity_au=float(v)) for e,v in zip(q.energy_eV,delta))
    pd.DataFrame(contrasts).to_csv(DATA/'Figure_3c_depth_contrasts.csv',index=False)
    ax=f.add_subplot(gs[1]);ax.axis('off')
    ax.set_title('b  Benchmark performance',loc='left',fontweight='bold',pad=10)
    metrics=pd.read_csv(root/'similarity_metrics.csv').query('comparison_type=="peak_aligned"')
    timing=pd.read_csv(ROOT/'reference_benchmark'/'reference_v2'/'timing_audit.csv').fillna('')
    rows=[]
    for material in ['Cu','Cu2O','CuO']:
        for method in colors:
            q=metrics[(metrics.material==material)&(metrics.method_key==method)]
            if q.empty:continue
            q=q.iloc[0]
            timer=timing[(timing.material==material)&(timing.method==method)]
            time_label=timer.iloc[0].display_time if len(timer)==1 else ''
            rows.append([{'Cu':'Cu','Cu2O':'Cu₂O','CuO':'CuO'}[material],labels[method],
                         f'{q.cosine_similarity:.3f}',f'{q.pearson_r:.3f}',f'{q.spearman_rho:.3f}',time_label])
    table=ax.table(cellText=rows,colLabels=['Reference','Method','Cosine','Pearson r','Spearman ρ','CPU time (s)¹'],
                   colWidths=[.11,.25,.14,.14,.16,.20],cellLoc='center',bbox=[0,.055,1,.925])
    table.auto_set_font_size(False);table.set_fontsize(7.3)
    for (row,col),cell in table.get_celld().items():
        cell.set_linewidth(.35);cell.set_edgecolor('#D6DEE6')
        if row==0:cell.set_facecolor('#E8EFF5');cell.set_text_props(fontweight='bold')
        elif rows[row-1][0]=='Cu₂O':cell.set_facecolor('#F5F8FA')
        else:cell.set_facecolor('white')
    ax.text(0,.008,'¹ Reported CPU; aggregation differs or is unverified. VASP excludes preceding SCF; blank = unavailable.',
            transform=ax.transAxes,fontsize=6.1,va='bottom')
    save(f,'Figure_3_spectroscopy')

def fig4():
    f=plt.figure(figsize=(7.2,9.4));gs=f.add_gridspec(3,2,hspace=.78,wspace=.30,height_ratios=[1,1,1.05])
    a=f.add_subplot(gs[0,0]);d=pd.read_csv(DATA/'nested_performance.csv').query('objective=="micro"').set_index('selector');keys=['depth_layer_5','depth_top_2','depth_all_4'];x=np.arange(3)
    for delta,col,metric,lab in [(-.18,P['blue'],'micro_f1','Micro F1'),(.18,P['orange'],'macro_f1','Macro F1')]:
        v=[d.loc[k,metric] for k in keys];a.bar(x+delta,v,.34,color=col,label=lab)
    a.set_xticks(x,['Surface only','Top-two\naverage','Four-layer\naverage']);a.set(ylim=(0,1),ylabel='Held-out F1');a.legend(fontsize=8,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.03),handlelength=1);panel(a,'a','Fixed-depth inference')
    a=f.add_subplot(gs[0,1]);d=pd.read_csv(DATA/'top_two_per_species.csv');x=np.arange(len(d));a.bar(x,d.f1,color=[P['blue'] if n>=5 else '#A8B5C3' for n in d.support],width=.72)
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
    a=f.add_subplot(gs[1,1]);d=pd.read_csv(DATA/'feature_model_depth_performance.csv').query('depth=="top_2" and objective=="micro"');features=['normalized','CDF','derivative','shape_CDF_derivatives','DCT24'];models=['logistic','SVM','random_forest','extra_trees','independent_trees'];v=d.pivot(index='model',columns='feature',values='micro_f1').reindex(index=models,columns=features)
    cmap=mcolors.LinearSegmentedColormap.from_list('gray_to_green',['#E5E5E5','#CFDCD3','#80B89B','#137548']);a.imshow(v,aspect='auto',vmin=float(v.min().min()),vmax=float(v.max().max()),cmap=cmap)
    for i in range(5):
        for j in range(5):a.text(j,i,f'{v.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if v.iloc[i,j]>.745 else '#1F2E3C')
    a.set_xticks(range(5),['Normalized\nspectrum','CDF','d1','Combined','DCT24'],rotation=30,ha='right');a.set_yticks(range(5),['Logistic','SVM','RF','Extra trees','Independent'],fontsize=7.5);panel(a,'d','Feature–model comparison')
    # Descriptive energy localization, on the same two-layer representation.
    d=pd.read_csv(DATA/'Figure_4e_energy_contrasts.csv')
    protocol=json.loads((DATA/'Figure_4e_protocol.json').read_text())
    order=protocol['species'];energy=np.sort(d.energy_eV.unique())
    values=d.pivot(index='species',columns='energy_eV',values='contrast_percent').loc[order,energy]
    sub=gs[2,:].subgridspec(1,2,width_ratios=[30,1],wspace=.08)
    a=f.add_subplot(sub[0]);colorbar=f.add_subplot(sub[1])
    limit=protocol['color_limit_percent'];step=float(np.median(np.diff(energy)))
    im=a.imshow(values,origin='upper',aspect='auto',interpolation='nearest',cmap='RdBu_r',
        vmin=-limit,vmax=limit,extent=[energy[0]-step/2,energy[-1]+step/2,len(order)-.5,-.5])
    a.set_yticks(range(len(order)),[f'{s} ({protocol["counts"][s]})' for s in order],fontsize=7.8)
    a.set_xticks([8975,8980,8990,9000,9010,9020]);a.set_xlim(8974,9029)
    a.set_xlabel('Photon energy (eV)');a.set_ylabel('Adsorbate (n)')
    a.set_yticks(np.arange(.5,len(order)-.5,1),minor=True)
    a.grid(which='minor',axis='y',color='white',lw=.65);a.tick_params(which='minor',left=False)
    for lo,hi in protocol['highlighted_windows_eV']:
        a.plot([lo,hi],[-.79,-.79],color='#555F69',lw=1.5,clip_on=False)
        a.text((lo+hi)/2,-.94,f'{lo}–{hi}',ha='center',va='bottom',fontsize=6.5,clip_on=False)
    a.set_title('e  Adsorbate-dependent energy regions',loc='left',fontweight='bold',fontsize=10,pad=30)
    a.text(0,1.17,'Top-two-layer means − equal-species mean',transform=a.transAxes,
        ha='left',va='bottom',fontsize=6.7,color='#53606C')
    cb=f.colorbar(im,cax=colorbar,ticks=[-3,-1.5,0,1.5,3]);cb.ax.tick_params(labelsize=7)
    cb.set_label('Δ intensity\n(% of reference maximum)',fontsize=7.2,labelpad=5)
    save(f,'Figure_4_ML')

def fig5():
    f,axs=plt.subplots(2,2,figsize=(6.4,4.9),gridspec_kw={'hspace':.32,'wspace':.08});rows=[]
    for ax,letter,(family,region) in zip(axs.flat,'abcd',[('2CO_adjacent','interface'),('OCCO','interface'),('2CO_adjacent','interior'),('OCCO','interior')]):
        cc=structure(ax,f'{family}_{region}.vasp',local=True);name='2CO' if family=='2CO_adjacent' else 'OCCO';panel(ax,letter,f'{name} · '+('interface' if region=='interface' else 'terrace'))
        ax.text(.5,-.055,f'C–C = {cc:.2f} Å',ha='center',va='top',transform=ax.transAxes,fontsize=9);rows.append(dict(family=family,region=region,input_CC_A=cc))
    pd.DataFrame(rows).to_csv(DATA/'Figure_5_input_distances.csv',index=False);save(f,'Figure_5_CO_coupling')

def main(figure=None):
    functions=[fig1,fig2,fig3,fig4,fig5]
    for fn in (functions if figure is None else [functions[figure-1]]):fn()
    print('Rebuilt publication figure PNGs' if figure is None else f'Rebuilt Figure {figure} PNG')
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--figure',type=int,choices=range(1,6))
    main(parser.parse_args().figure)
