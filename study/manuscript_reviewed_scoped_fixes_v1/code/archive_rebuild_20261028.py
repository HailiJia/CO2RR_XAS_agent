"""Rebuild Figures 2–5 from committed data without retraining or spectral refitting.
Run from the extracted package: python code/rebuild_requested_figures.py
All outputs are PNG. Scientific controls and cohort memberships are saved as CSV/JSON.
"""
from pathlib import Path
import importlib.util,json,shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors as mc
from matplotlib.lines import Line2D
from sklearn.metrics import mean_absolute_error,r2_score

ROOT=Path(__file__).resolve().parents[1];D=ROOT/'figure_data';F=ROOT/'figures'
spec=importlib.util.spec_from_file_location('style',ROOT/'code/preferred_style_helpers.py');h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
h.ROOT=ROOT;h.DATA=D/'preferred_style';h.FIG=F
P=h.P;BLUE='#4477AA';GREEN='#228873';ORANGE='#D58A45';GRAY='#BBC6CF'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
rows=json.load(open(D/'pair_manifest.json'));paired=sorted([r for r in rows if 'pair_index' in r],key=lambda r:r['pair_index']);Z=np.load(D/'paired_arrays.npz')
for i,r in enumerate(paired):assert r['configuration']==Z['configuration'][i]
np.testing.assert_allclose(Z['delta'],Z['raw_physical']-Z['clean'],atol=1e-13)
def save(f,name):f.savefig(F/(name+'.png'),dpi=450,bbox_inches='tight',pad_inches=.06);plt.close(f)
def panel(a,l,t):a.set_title(l+'  '+t,loc='left',fontweight='bold',pad=8)
def extra(n):return pd.read_csv(D/n)
def label(s):return s.replace('Cu2O','Cu₂O').replace('CO2','CO₂').replace('CO3','CO₃').replace('H2O','H₂O').replace(';',' + ')

def dataset():
    f=plt.figure(figsize=(7.6,9.7));gs=f.add_gridspec(2,2,height_ratios=[1,1.65],width_ratios=[1,1.1],hspace=.34,wspace=.28)
    l=gs[0,0].subgridspec(2,1,height_ratios=[1.35,.75],hspace=.26)
    h.substrate_views(f.add_subplot(l[0]),f.add_subplot(l[1]))
    sub=gs[0,1].subgridspec(3,4,hspace=.38,wspace=.2)
    for i,n in enumerate(['CO','CO2','CHO','COH','COOH','OCHO','COCHO','OCCOH','OCCO','H','O','OH']):h.molecule(f.add_subplot(sub[i//4,i%4]),n)
    f.text(.125,.925,'a  Substrate and adsorbates',fontsize=10,fontweight='bold')
    f.text(.125,.544,'b  Adsorbate populations',fontsize=10,fontweight='bold')
    f.text(.544,.544,'c  Adsorption sites',fontsize=10,fontweight='bold')
    f.legend(handles=[Line2D([],[],ls='',marker='o',ms=5.5,color=P[z],label=z) for z in ['Cu','Au','C','O','H']],ncol=5,loc='center',bbox_to_anchor=(.54,.582),fontsize=8,handletextpad=.25,columnspacing=1.4)
    # Preferred broad gray background / narrow blue overlay; no tiny subset ticks.
    domains=[('Cu–Au, available',lambda r:r['domain']=='CuAu',BLUE),('Cu(100), available',lambda r:r['domain']=='Cu100',GRAY),('Cu–Au, matched',lambda r:r.get('eligible_primary',False),GREEN)]
    combos=sorted({';'.join(sorted(r['labels'])) for r in rows})
    singles=['CO','H','COOH','OCHO','O','COH','CHO','OH','OCCOH','CO2','COCHO','CO3'];mixed=[c for c in combos if ';' in c]
    sub=gs[1,0].subgridspec(2,1,height_ratios=[len(singles),len(mixed)],hspace=.40)
    ct=[]
    for k,(cats,title) in enumerate([(singles,'Single species'),(mixed,'Coadsorbed species')]):
        a=f.add_subplot(sub[k]);y=np.arange(len(cats))
        current={name:np.array([sum(include(r) and ';'.join(sorted(r['labels']))==c for r in rows) for c in cats]) for name,include,col in domains}
        for name,include,col in domains:
            ct.extend(dict(category=title,identity_set=c,cohort=name,n=int(n)) for c,n in zip(cats,current[name]))
        total=current['Cu–Au, available']+current['Cu(100), available']
        a.barh(y,total,height=.75,color='#D6DEE6',label='All completed (115)')
        a.barh(y,current['Cu–Au, available'],height=.38,color=BLUE,label='Cu–Au (98)')
        for j,n in enumerate(total):
            if n:a.text(n+.5,j,str(n),va='center',fontsize=7.1,color='#4B5660')
        a.set_yticks(y,[label(c) for c in cats],fontsize=7.2);a.invert_yaxis();a.set(xlim=(0,35),xticks=[0,10,20,30]);a.tick_params(axis='y',length=0)
        a.set_title(title,loc='left',fontsize=8.3,fontweight='bold',pad=5)
        if k==0:a.legend(loc='lower right',fontsize=6.7,handlelength=1.1,labelspacing=.35)
        else:a.set_xlabel('Independent configurations',fontsize=8)
    counts=pd.DataFrame(ct);counts.to_csv(D/'Figure2b_current_counts.csv',index=False)
    assert counts[counts.cohort=='Cu–Au, available'].n.sum()==98
    assert counts[counts.cohort=='Cu(100), available'].n.sum()==17
    assert counts[counts.cohort=='Cu–Au, matched'].n.sum()==34
    sub=gs[1,1].subgridspec(2,1,height_ratios=[3.6,1.6],hspace=.04);h.interface_sites(f.add_subplot(sub[0]),f.add_subplot(sub[1]))
    save(f,'Figure_2_dataset')

def benchmark():
    f=plt.figure(figsize=(7.6,8.1));g=f.add_gridspec(3,1,height_ratios=[1,1.65,.77],hspace=.64,top=.855,bottom=.07)
    methods=['fdmnes_green','fdmnes_fdm','feff','vasp_pbe','vasp_gw'];cols=['#0072B2','#D55E00','#009E73','#CC79A7','#C59000'];names=['FDMNES green','FDMNES fdm','FEFF','VASP PBE','VASP GW-PAW'];styles=['-','--',':','-.','-']
    arrays=extra('reference_plotted_arrays.csv').query('comparison_type=="first_feature_shape_diagnostic"');met=extra('similarity_metrics.csv').query('comparison_type=="first_feature_shape_diagnostic"')
    f.text(.125,.985,'a  Reference-material spectra',fontsize=10,fontweight='bold')
    f.legend(handles=[Line2D([],[],c='black',lw=1.5,label='Experiment')]+[Line2D([],[],c=c,ls=ls,lw=1.2,label=n) for n,c,ls in zip(names,cols,styles)],ncol=3,loc='center',bbox_to_anchor=(.54,.938),fontsize=7.5,handlelength=2.8)
    sub=g[0].subgridspec(1,3,wspace=.29)
    for j,material in enumerate(['Cu','Cu2O','CuO']):
        a=f.add_subplot(sub[j]);d=arrays[arrays.material==material];first=True
        for method,c,ls in zip(methods,cols,styles):
            q=d[d.method==method]
            if q.empty:continue
            if first:a.plot(q.energy_eV,q.experiment_normalized,c='black',lw=1.5);first=False
            a.plot(q.energy_eV,q.calculation_normalized,c=c,ls=ls,lw=1.15)
        a.set(xlim=(8975,9015) if material=='Cu2O' else (8980,9015),ylim=(0,1.08),xlabel='Photon energy (eV)',xticks=[8980,9000,9015],yticks=[0,.5,1]);a.ticklabel_format(axis='x',style='plain',useOffset=False)
        a.set_title(label(material),loc='left',fontweight='bold',fontsize=9)
        if j==0:a.set_ylabel('Intensity (a.u.)')
    old=extra('legacy_timing_audit.csv');prior=extra('legacy_scf_timings.csv');new=extra('reference_timing_summary.csv');table=[];audit=[]
    aliases={'VASP PBE':'vasp_pbe','VASP GW':'vasp_gw','FDMNES fdm':'fdmnes_fdm'}
    new['method_key']=new.method.map(aliases)
    for mat in ['Cu','Cu2O','CuO']:
        for method,name in zip(methods,names):
            q=met[(met.material==mat)&(met.method==method)];t=old[(old.material==mat)&(old.method==method)]
            xas=float(t.iloc[0].reported_cpu_s) if len(t) and pd.notna(t.iloc[0].reported_cpu_s) else np.nan
            source=t.iloc[0].timing_source if len(t) else ''
            sc=prior[(prior.material==mat)&(prior.method==method)];scf=float(sc.iloc[0].reported_cpu_s) if len(sc) and pd.notna(sc.iloc[0].reported_cpu_s) else np.nan
            for stage in ['SCF','XAS']:
                u=new[(new.material==mat)&(new.method_key==method)&(new.stage==stage)]
                if len(u):
                    value=u.iloc[0].reported_CPU_time_s
                    if stage=='SCF':scf=float(value) if pd.notna(value) else np.nan
                    else:xas=float(value) if pd.notna(value) else np.nan;source=u.iloc[0].source if pd.notna(u.iloc[0].source) else u.iloc[0].source_basis
            fmt=lambda v:'—' if not np.isfinite(v) else f'{v:,.0f}'
            time=fmt(xas)  # One latest available XAS-stage CPU counter; SCF remains in SI.
            scores=['—']*3 if q.empty else [f'{q.iloc[0][k]:.3f}' for k in ['cosine_similarity','Pearson_r','Spearman_rho']]
            table.append([label(mat),name,*scores,time]);audit.append(dict(material=mat,method=method,numeric_spectrum_available=not q.empty,logged_scf_CPU_s=scf,logged_XAS_CPU_s=xas,source=source,scope='code-reported CPU; cross-code aggregation/hardware not harmonized'))
    a=f.add_subplot(g[1]);a.axis('off');panel(a,'b','Benchmark scores and logged CPU time')
    tb=a.table(cellText=table,colLabels=['Reference','Method','Cosine','Pearson r','Spearman ρ','CPU time (s)¹'],colWidths=[.10,.225,.115,.12,.13,.21],cellLoc='center',bbox=[0,.05,1,.95]);tb.auto_set_font_size(False);tb.set_fontsize(7.7)
    for (i,j),c in tb.get_celld().items():
        c.set_edgecolor('#D6DEE6');c.set_linewidth(.35)
        if i==0:c.set_facecolor('#E8EFF5');c.set_text_props(fontweight='bold')
        else:c.set_facecolor('#F4F7F9' if table[i-1][0]=='Cu₂O' else 'white')
    a.text(0,-.005,'¹ Latest available XAS CPU timer. Preceding VASP SCF times and accounting scope are reported in the SI.',transform=a.transAxes,fontsize=6.1,va='top')
    pd.DataFrame(audit).to_csv(D/'Figure3_logged_CPU_provenance.csv',index=False)
    ld=extra('layer_differential_arrays.csv');sub=g[2].subgridspec(1,3,wspace=.18)
    cases=[('2CO_adjacent/2CO_adjacent_0p111_interface','Interface'),('CO/CO_0p11/interior_biased','Cu interior'),('CO/CO_0p11/uniform','Distributed')]
    for j,(case,title) in enumerate(cases):
        a=f.add_subplot(sub[j]);q=ld[ld.configuration==case]
        assert q.primary_matched.all()
        for l,c in [(5,'#D77741'),(4,BLUE)]:
            u=q[q.layer==l];a.plot(u.energy_eV,u.delta_common_scale*100,c=c,lw=1.4,label='Surface (5)' if l==5 else 'Subsurface (4)')
        a.axhline(0,c='.5',lw=.7,ls='--',label='Zero difference');a.set(xlim=(8974,9029),ylim=(-3.25,3.25),xticks=[8980,9000,9020],xlabel='Photon energy (eV)');a.ticklabel_format(axis='x',style='plain',useOffset=False);a.set_title('4 CO: '+title,fontsize=8)
        if j==0:a.set_ylabel('Δμ / clean edge step (%)');a.text(0,1.52,'c  Layer-resolved adsorbate-minus-clean spectra',transform=a.transAxes,fontweight='bold',fontsize=10)
        else:a.tick_params(labelleft=False)
        if j==1:a.legend(loc='upper center',bbox_to_anchor=(.5,-.52),ncol=3,fontsize=6.2,handlelength=2.,columnspacing=1.5,frameon=False)
    save(f,'Figure_3_spectroscopy')

def differences():
    order=['CO','COH','COOH','H','O','OH'];out=[];members=[]
    for name in order:
        r=[r for r in paired if r['eligible_primary'] and r['labels']==[name]];idx=[r['pair_index'] for r in r];v=Z['delta'][idx]*100
        for e,m,sd in zip(Z['energy'],v.mean(axis=0),v.std(axis=0,ddof=1)):out.append(dict(species=name,n=len(r),energy_eV=e,delta_percent=m,between_configuration_sd_percent=sd))
        members.extend(dict(species=name,configuration=r['configuration'],pair_index=r['pair_index'],n_CO=r['n_CO'],coverage_metal=r['coverage_metal']) for r in r)
    pd.DataFrame(out).to_csv(D/'Figure4e_clean_reference_species.csv',index=False);pd.DataFrame(members).to_csv(D/'Figure4e_membership.csv',index=False)
    # Companion plots retain individual configurations and separate weaker references.
    for primary,title,file in [(True,'Initial-model-matched clean references','SI_differential_species_matched'),(False,'Same-substrate reference sensitivity','SI_differential_species_sensitivity')]:
        names=order if primary else sorted({r['labels'][0] for r in paired if len(r['labels'])==1});fig,axs=plt.subplots(int(np.ceil(len(names)/3)),3,figsize=(7.4,2.05*np.ceil(len(names)/3)),sharex=True,sharey=True,squeeze=False)
        for a,name in zip(axs.flat,names):
            rr=[r for r in paired if (r['eligible_primary'] or not primary) and r['labels']==[name]];v=Z['delta'][[r['pair_index'] for r in rr]]*100
            for vv in v:a.plot(Z['energy'],vv,c=BLUE,alpha=.16,lw=.7)
            a.plot(Z['energy'],v.mean(axis=0),c=GREEN,lw=1.6);a.axhline(0,c='.5',ls='--',lw=.6);a.set_title(f'{name}, n={len(rr)}',fontsize=8);a.set(xlim=(8974,9029),xlabel='Energy (eV)',ylabel='Δμ / clean step (%)');a.ticklabel_format(axis='x',style='plain',useOffset=False)
        for a in list(axs.flat)[len(names):]:a.axis('off')
        fig.suptitle(title,fontsize=10);fig.tight_layout();save(fig,file)
    return pd.DataFrame(out),order

def identity():
    delta,order=differences();f=plt.figure(figsize=(7.6,9.4));g=f.add_gridspec(3,2,hspace=.83,wspace=.34,height_ratios=[1,1,1.04])
    a=f.add_subplot(g[0,0]);v=extra('preferred_style/nested_performance.csv').query('objective=="micro"').set_index('selector');keys=['depth_layer_5','depth_top_2','depth_all_4'];x=np.arange(3)
    for dx,col,metric,lab in [(-.18,BLUE,'micro_f1','Micro F1'),(.18,ORANGE,'macro_f1','Macro F1')]:a.bar(x+dx,[v.loc[k,metric] for k in keys],.34,color=col,label=lab)
    a.set(xticks=x,xticklabels=['Surface only','Top-two\naverage','Four-layer\naverage'],ylim=(0,1),ylabel='Held-out F1');a.legend(ncol=2,loc='upper center',bbox_to_anchor=(.5,1.04),fontsize=7,handlelength=1);panel(a,'a','Fixed-depth benchmark (87)')
    a=f.add_subplot(g[0,1]);d=pd.DataFrame(json.load(open(D/'study__differential_v1__ml__same_substrate_sensitivity_raw_physical_per_species.json')));x=np.arange(len(d));a.bar(x,d.F1,color=[BLUE if n>=5 else '#A8B5C3' for n in d.support],width=.72)
    for i,r in enumerate(d.itertuples()):
        expected=sum(r.species in q['labels'] for q in paired);assert expected==r.support
        a.text(i,r.F1+.035,str(r.support),ha='center',fontsize=7)
    a.set(xticks=x,xticklabels=[label(s)+('*' if n<5 else '') for s,n in zip(d.species,d.support)],ylim=(0,1.13),ylabel='Held-out F1');a.tick_params(axis='x',rotation=50);panel(a,'b','Adsorbate identification (98)');d.to_csv(D/'Figure4b_current_per_species.csv',index=False)
    sub=g[1,0].subgridspec(1,2,wspace=.24);ls=extra('preferred_style/layer_sensitivity.csv')
    for j,(order2,title,xlab,cols) in enumerate([(['layer_5','layer_4','layer_3','layer_2'],'Individual layers','Layer from surface',[BLUE,ORANGE]),(['layer_5','top_2','top_3','all_4'],'Top-n-layer average','Number of top layers, n',[GREEN,'#AA3377'])]):
        a=f.add_subplot(sub[j]);x=np.arange(1,5)
        for metric,dx,marker,col in [('micro',-.06,'o',cols[0]),('macro',.06,'s',cols[1])]:
            q=ls[ls.metric==metric].set_index('depth').loc[order2];a.errorbar(x+dx,q.f1,yerr=[q.f1-q.ci_low,q.ci_high-q.f1],fmt=marker+'-',ms=5,mew=.5,mec='white',lw=1.3,elinewidth=.6,capsize=2,c=col,label=metric.title()+' F1')
        a.set(xlim=(.7,4.3),ylim=(0,1),xticks=x,xlabel=xlab);a.set_title(title,fontsize=7.6,fontweight='bold');a.xaxis.label.set_size(7);a.yaxis.grid(True,c='#E6E9ED',lw=.5);a.set_axisbelow(True);a.legend(loc='lower left',fontsize=6.1,handlelength=1.3)
        if j==0:a.set_ylabel('Held-out F1');a.text(0,1.24,'c  Depth sensitivity (87)',transform=a.transAxes,fontsize=10,fontweight='bold')
        else:a.tick_params(labelleft=False)
    a=f.add_subplot(g[1,1]);d=extra('preferred_style/feature_model_depth_performance.csv').query('depth=="top_2" and objective=="micro"');feats=['normalized','CDF','derivative','shape_CDF_derivatives','DCT24'];models=['logistic','SVM','random_forest','extra_trees','independent_trees'];v=d.pivot(index='model',columns='feature',values='micro_f1').reindex(index=models,columns=feats)
    cmap=mc.LinearSegmentedColormap.from_list('gray_green',['#E5E5E5','#CFDCD3','#80B89B','#137548']);a.imshow(v,aspect='auto',vmin=v.min().min(),vmax=v.max().max(),cmap=cmap)
    for i in range(5):
        for j in range(5):a.text(j,i,f'{v.iloc[i,j]:.2f}',ha='center',va='center',fontsize=8,c='white' if v.iloc[i,j]>.745 else '#1F2E3C')
    a.set_xticks(range(5),['Normalized\nspectrum','CDF','d1','Combined','DCT24'],rotation=30,ha='right');a.set_yticks(range(5),['Logistic','SVM','RF','Extra trees','Independent'],fontsize=7.3);panel(a,'d','Feature–model comparison (87)')
    sub=g[2,:].subgridspec(1,2,width_ratios=[30,1],wspace=.08);a=f.add_subplot(sub[0]);c=f.add_subplot(sub[1]);v=delta.pivot(index='species',columns='energy_eV',values='delta_percent').loc[order];im=a.imshow(v,aspect='auto',cmap='RdBu_r',vmin=-6,vmax=6,extent=[8973.9,9029.1,len(order)-.5,-.5],interpolation='nearest')
    n=delta.groupby('species').n.first();a.set(yticks=np.arange(len(order)),yticklabels=[f'{sp} ({n[sp]})' for sp in order],xlim=(8974,9029),xlabel='Photon energy (eV)',ylabel='Adsorbate (n)',xticks=[8980,8990,9000,9010,9020]);a.set_yticks(np.arange(.5,len(order)-.5),minor=True);a.grid(which='minor',axis='y',c='white',lw=.7);a.tick_params(which='minor',left=False);panel(a,'e','Differential XAS by adsorbate')
    cb=f.colorbar(im,cax=c,ticks=[-6,-3,0,3,6]);cb.set_label('Δμ / clean edge step (%)',fontsize=7.4)
    save(f,'Figure_4_ML')

def new_results():
    f,axs=plt.subplots(2,2,figsize=(7.6,6.2),gridspec_kw={'hspace':.57,'wspace':.36})
    a=axs[0,0];d=extra('identity_metrics.csv').query('scope=="matched"').set_index('representation');reps=['corrected_normalized','raw_physical','delta'];xx=np.arange(3)
    for dx,metric,col,lab in [(-.17,'micro_f1',GREEN,'Micro F1'),(.17,'exact_set_accuracy',GRAY,'Exact set')]:a.bar(xx+dx,[d.loc[r,metric] for r in reps],.31,color=col,label=lab)
    a.set(xticks=xx,xticklabels=['Normalized\nraw','Physical\nraw','Differential\nΔμ'],ylim=(0,1.02),ylabel='Held-out score');a.legend(loc='upper left',fontsize=7);panel(a,'a','Reference comparison: 34 Cu–Au')
    a=axs[0,1];df=extra('study__differential_v1__ml__coverage_predictions.csv');d=df.query('scope=="same_substrate_sensitivity" and representation=="raw_physical"').copy();d['prediction_Cu']=d.prediction_metal*68/36
    assert len(d)==25
    a.scatter(d.truth_Cu_loading,d.prediction_Cu,s=30,c=BLUE,ec='white',lw=.5,zorder=3);a.plot([0,.8],[0,.8],c='.5',ls='--',lw=.8,label='Ideal');a.set(xlim=(0,.8),ylim=(-.025,.825),xlabel='True CO / surface Cu',ylabel='Predicted CO / surface Cu');mae=mean_absolute_error(d.truth_Cu_loading,d.prediction_Cu);r2=r2_score(d.truth_Cu_loading,d.prediction_Cu);a.text(.04,.94,f'MAE = {mae:.3f}\nR² = {r2:.3f}',transform=a.transAxes,va='top',fontsize=8);panel(a,'b','CO loading: 25 Cu–Au, raw XAS')
    a=axs[1,0];d=extra('study__differential_v1__raw_augmentation__heldout.csv').query('domain=="Cu100"')
    for mode,col,marker,lab in [('CuAu_only','#8B9297','o','Cu–Au training'),('CuAu_plus_Cu100',GREEN,'s','+ three Cu(100) cases')]:
        q=d[d.training==mode].sort_values('truth_Cu_loading');a.plot(q.truth_Cu_loading,q.prediction_Cu_loading,marker=marker,ls='-',lw=1.1,ms=5.6,c=col,mec='white',mew=.5,label=lab)
    a.plot([.2,1.05],[.2,1.05],c='.45',ls='--',lw=.8,label='Ideal');a.set(xlim=(.2,1.05),ylim=(.1,1.1),xlabel='True CO / surface Cu',ylabel='Predicted CO / surface Cu');a.legend(fontsize=6.4,loc='upper left');panel(a,'c','Cu(100) loading: four held-out cases')
    a=axs[1,1];d=extra('study__differential_v1__raw_identity_augmentation__metrics.csv').query('domain=="Cu100"').set_index('training');keys=['micro_f1','macro_f1','exact_set_accuracy'];xx=np.arange(3)
    for dx,mode,col,lab in [(-.17,'CuAu_only','#8B9297','Cu–Au training'),(.17,'CuAu_plus_Cu100',GREEN,'+ Cu(100) training')]:
        a.bar(xx+dx,[d.loc[mode,k] for k in keys],.31,color=col,label=lab)
    a.set(xticks=xx,xticklabels=['Micro F1','Macro F1','Exact set'],ylim=(0,1.02),ylabel='Held-out score');a.legend(fontsize=6.4,loc='upper left');panel(a,'d','Cu(100) identity: 12 cases, four labels')
    save(f,'Figure_5_reference_and_transfer')

if __name__=='__main__':
    dataset();benchmark();identity();new_results();print('Rebuilt four manuscript PNGs and two differential-species SI figures; no model refit.')
