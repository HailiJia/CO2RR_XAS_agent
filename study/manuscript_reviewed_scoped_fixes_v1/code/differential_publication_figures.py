"""Publication PNGs from saved paired arrays and OOF predictions; no refitting."""
from pathlib import Path
import sys,json
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import f1_score,accuracy_score,mean_absolute_error
ROOT=next(p for p in Path(__file__).resolve().parents if (p/'tools/differential_xas.py').is_file());sys.path.insert(0,str(ROOT))
from tools.progress_snapshots import write_json
B=ROOT/'study/differential_v1';F=B/'figures';F.mkdir(exist_ok=True);z=np.load(B/'paired_arrays.npz');rows=sorted([r for r in json.load(open(B/'pair_manifest.json')) if 'pair_index'in r],key=lambda r:r['pair_index']);legacy=ROOT/'study/manuscript_package/figure_data';plt.rcParams.update({'font.size':8,'axes.labelsize':8,'axes.titlesize':9,'legend.fontsize':7,'xtick.labelsize':7,'ytick.labelsize':7,'savefig.dpi':400})
labels={'raw_normalized':'Legacy norm.','corrected_normalized':'Corrected norm.','raw_physical':'Raw per Cu','delta':'Δμ','clean_plus_delta':'[clean, Δμ]'};colors=['#777777','#b2b2b2','#447aa1','#25834e','#80b69a'];reps=list(labels);identity=pd.read_csv(B/'ml/identity_metrics.csv');coverage=pd.read_csv(B/'ml/coverage_metrics.csv');arch=pd.read_csv(legacy/'layer_sensitivity.csv');feature=pd.read_csv(legacy/'feature_model_depth_performance.csv')
fig=plt.figure(figsize=(6.5,8.4));g=fig.add_gridspec(3,2,height_ratios=[1,1.1,1.12],hspace=.54,wspace=.46)
ax=fig.add_subplot(g[0,0]);x=np.arange(5);d=identity[(identity.scope=='matched')].set_index('representation');ax.bar(x-.16,[d.loc[r,'micro_f1'] for r in reps],.30,color='#25834e',label='Micro F1');ax.bar(x+.16,[d.loc[r,'exact_set_accuracy'] for r in reps],.30,color='#a8b9b1',label='Exact set');ax.set(xticks=x,xticklabels=[labels[r] for r in reps],ylim=(0,1),title='(a) Matched identity: 34 configurations',ylabel='Held-out score');ax.tick_params(axis='x',rotation=40);ax.legend(loc='upper left',fontsize=6.7)
ax=fig.add_subplot(g[0,1]);d=coverage[(coverage.scope=='matched')&(coverage.target=='CO/all_top_surface_metals')].set_index('representation');names=reps+['training_mean'];xx=np.arange(6);ax.bar(xx,[d.loc[r,'MAE'] for r in names],color=colors+['#d6d6d6']);ax.set(xticks=xx,xticklabels=[labels.get(r,'Train mean') for r in names],title='(b) Matched CO-only coverage: 7 cases',ylabel='MAE (CO / all surface metal sites)');ax.tick_params(axis='x',rotation=40)
sub=g[1,0].subgridspec(1,2,wspace=.43)
for j,(depths,ticks,title) in enumerate([(['layer_2','layer_3','layer_4','layer_5'],['2','3','4','5'],'Individual layer'),(['layer_5','top_2','top_3','all_4'],['1','2','3','4'],'Top-n average')]):
 ax=fig.add_subplot(sub[0,j]);vals=[arch[(arch.depth==d)&(arch.metric=='micro')].iloc[0].f1 for d in depths];ax.plot(np.arange(4),vals,'o-',c='#25834e',lw=1.2,ms=3);ax.set(xticks=np.arange(4),xticklabels=ticks,ylim=(0,1),xlabel='Layer index' if j==0 else 'n',title=('(c) ' if j==0 else '')+title)
 if j==0:ax.set_ylabel('Archived micro F1 (87 cases)')
ax=fig.add_subplot(g[1,1]);deps=['layer_5','top_2','top_3','all_4'];feats=['normalized','CDF','derivative','shape_CDF_derivatives','DCT24'];arr=np.array([[feature[(feature.depth==d)&(feature.feature==f)&(feature.objective=='micro')].micro_f1.max() for f in feats] for d in deps]);cmap=LinearSegmentedColormap.from_list('gray_green',['#eeeeee','#bfc8bf','#217842']);im=ax.imshow(arr,cmap=cmap,vmin=.45,vmax=.85,aspect='auto');ax.set(xticks=np.arange(5),xticklabels=['Norm.','CDF','d1','Comb.','DCT24'],yticks=np.arange(4),yticklabels=['Surface','Top 2','Top 3','All 4'],title='(d) Archived exploratory score map');ax.tick_params(axis='x',labelsize=6.5)
for i in range(4):
 for j in range(5):ax.text(j,i,f'{arr[i,j]:.2f}',ha='center',va='center',fontsize=6.5,color='white' if arr[i,j]>.73 else 'black')
fig.colorbar(im,ax=ax,fraction=.04,pad=.03).ax.tick_params(labelsize=6)
sub=g[2,:].subgridspec(1,2,wspace=.32);axes=[fig.add_subplot(sub[0,i]) for i in range(2)];chosen=['2CO_adjacent/2CO_adjacent_0p111_interface','CO/CO_0p11/interior_biased','CO/CO_0p11/uniform'];cs=['#0072B2','#D55E00','#009E73'];E=z['energy'];idx=[next(i for i,r in enumerate(rows) if r['configuration']==a) for a in chosen];axes[0].plot(E,z['clean'][idx[0]],c='black',lw=1.4,label='Clean Cu–Au')
for k,i in enumerate(idx):
 axes[0].plot(E,z['raw_physical'][i],c=cs[k],lw=1,label=f'4 CO, configuration {chr(65+k)}');axes[1].plot(E,z['delta'][i],c=cs[k],lw=1,label=f'4 CO, configuration {chr(65+k)}')
axes[1].axhline(0,c='.5',lw=.8,ls='--',label='Zero difference')
for ax in axes:ax.set(xlabel='Photon energy (eV)',xlim=(8974,9029));ax.legend(fontsize=6.4,loc='best')
axes[0].set(title='(e) Actual clean/adsorbed spectra',ylabel='μ / shared clean edge step');axes[1].set(title='Equal loading, different geometries',ylabel='Signed Δμ / shared clean edge step');fig.savefig(F/'Figure_4_differential_ML.png',bbox_inches='tight');plt.close(fig)
# All layer-resolved physical differences retain the top-two clean scale.
layer_rows=[];S=B/'scale/reconvolution'
for r in rows:
 for l in [4,5]:
  a=pd.read_csv(S/r['configuration']/f'layer_{l}/scale_comparison.csv');b=pd.read_csv(S/'plain'/f'layer_{l}/scale_comparison.csv');diff=(a.corrected_per_Cu-b.corrected_per_Cu)/r['pair_scale']
  for e,v in zip(a.energy_eV,diff):layer_rows.append({'configuration':r['configuration'],'layer':l,'energy_eV':e,'delta_common_scale':v,'scale_Mbarn_per_Cu':r['pair_scale'],'primary_matched':r['eligible_primary']})
pd.DataFrame(layer_rows).to_csv(B/'layer_differential_arrays.csv',index=False)
fig,axes=plt.subplots(1,3,figsize=(9.6,3))
ld=pd.DataFrame(layer_rows)
for ax,i in zip(axes,idx):
 for l,col in [(4,'#6a9fb5'),(5,'#d77741')]:
  t=ld[(ld.configuration==rows[i]['configuration'])&(ld.layer==l)];ax.plot(t.energy_eV,t.delta_common_scale,c=col,label=f'Layer {l}')
 ax.axhline(0,c='.5',ls='--',lw=.7,label='Zero difference');ax.set(title=rows[i]['configuration'].split('/')[-1],xlabel='Photon energy (eV)',ylabel='Signed layer Δμ / top-two clean scale');ax.legend(fontsize=7)
fig.tight_layout();fig.savefig(F/'SI_layer_differences.png',bbox_inches='tight');plt.close(fig)
# Raw scale correction is source-resolved, not a guessed aggregate divisor.
fig,axes=plt.subplots(1,2,figsize=(6.5,2.7))
for l,col in [(4,'#0072B2'),(5,'#D55E00')]:
 d=pd.read_csv(S/'plain'/f'layer_{l}/scale_comparison.csv');axes[0].plot(d.energy_eV,d.native_aggregate_per_unique_Cu_without_symmetry_correction,c=col,ls=':',label=f'Layer {l}, native/N');axes[0].plot(d.energy_eV,d.corrected_per_Cu,c=col,label=f'Layer {l}, source-corrected');axes[1].plot(d.energy_eV,d.corrected_per_Cu/d.native_aggregate_per_unique_Cu_without_symmetry_correction,c=col,label=f'Layer {l}')
axes[0].set(xlabel='Photon energy (eV)',ylabel='Absorption (Mbarn / Cu)',title='Clean source-scale audit');axes[1].set(xlabel='Photon energy (eV)',ylabel='Corrected / native-per-index',title='Energy-dependent correction ratio')
for ax in axes:ax.legend(fontsize=6.5)
fig.tight_layout();fig.savefig(F/'SI_source_scale.png',bbox_inches='tight');plt.close(fig)
# Stress tests and the clearly separate fixed raw-input augmentation pilot.
r=pd.read_csv(B/'robustness/metrics.csv');fig,axes=plt.subplots(2,2,figsize=(6.5,5.7));styles={'unaugmented':'-','training_only_augmented':'--'}
for rep,col,label in [('raw_normalized','#447aa1','Corrected norm. raw'),('delta','#25834e','Δμ')]:
 for training,style in styles.items():
  a=r[(r.task=='CO_coverage')&(r.representation==rep)&(r.training==training)&(r.kind=='independent_pair_noise')].groupby('value').MAE.mean();axes[0,0].plot(a.index*100,a.values,ls=style,c=col,marker='o',ms=3,label=label+(' + train aug.' if training!='unaugmented' else ''))
  a=r[(r.task=='CO_coverage')&(r.representation==rep)&(r.training==training)&(r.kind=='relative_shift')].groupby('value').MAE.mean();axes[0,1].plot(a.index,a.values,ls=style,c=col,marker='o',ms=3,label=label+(' + train aug.' if training!='unaugmented' else ''))
axes[0,0].set(title='(a) Synthetic independent pair noise',xlabel='Noise amplitude (% of peak)',ylabel='MAE (CO / all surface sites)');axes[0,1].set(title='(b) Residual relative reference shift',xlabel='Reference-only shift error (eV)',ylabel='MAE (CO / all surface sites)')
a=r[(r.task=='CO_coverage')&(r.representation=='delta')&(r.training=='unaugmented')&(r.kind=='surface_fraction')].sort_values('value');axes[1,0].plot(a.value,a.MAE,'o-',c='#25834e',ms=3,label='Error versus full surface coverage');axes[1,0].plot(a.value,a.MAE_vs_response_proxy_f_times_theta,'s--',c='.45',ms=3,label='Error versus response proxy fθ');axes[1,0].set(title='(c) Stable-background dilution scenario',xlabel='Participating Cu fraction f',ylabel='MAE (CO / all surface sites)')
pilot=pd.read_csv(B/'raw_augmentation/heldout.csv');a=pilot[pilot.domain=='Cu100']
for mode,col,label in [('CuAu_only','.55','Cu–Au only'),('CuAu_plus_Cu100','#25834e','Cu–Au + other 3 Cu100')]:
 t=a[a.training==mode];axes[1,1].plot(t.truth_Cu_loading,t.prediction_Cu_loading,'o-',ms=3,c=col,label=label)
axes[1,1].plot([.2,1.05],[.2,1.05],c='.3',ls=':',lw=.8,label='y = x');axes[1,1].set(title='(d) Raw-input Cu100 pilot (4 cases)',xlabel='True CO / surface Cu loading',ylabel='Predicted CO / surface Cu loading')
for ax in axes.flat:ax.legend(fontsize=6.1)
fig.tight_layout();fig.savefig(F/'Figure_5_robustness_transfer.png',bbox_inches='tight');plt.close(fig)
# Conditional paired uncertainty; configurations resampled, not energy points.
intervals=[];rng=np.random.default_rng(20261027)
for task in ['identity','CO_coverage']:
 zz={rep:np.load(B/'ml'/f'matched_{task}_{rep}_oof.npz') for rep in reps};truth=zz['delta']['truth'];n=len(truth)
 for baseline in ['raw_normalized','corrected_normalized','raw_physical']:
  a=zz['delta']['prediction'];b=zz[baseline]['prediction'];vals=[]
  def score(y,p):return f1_score(y,p,average='micro',zero_division=0) if task=='identity' else mean_absolute_error(y,p)
  observed=score(truth,a)-score(truth,b)
  for _ in range(1000):
   k=rng.integers(n,size=n);vals.append(score(truth[k],a[k])-score(truth[k],b[k]))
  intervals.append({'task':task,'contrast':'delta minus '+baseline,'n_configurations':n,'observed_difference':observed,'conditional_95_low':np.quantile(vals,.025),'conditional_95_high':np.quantile(vals,.975),'scope':'fixed predictions; no refitting or independent-parent uncertainty'})
pd.DataFrame(intervals).to_csv(B/'paired_uncertainty.csv',index=False)
# Preserve identities/paths behind the representative clean-reference panel.
write_json(B/'figure4e_provenance.json',[rows[i] for i in idx]);write_json(B/'figure_source_manifest.json',{'Figure_4_differential_ML.png':['ml/identity_metrics.csv','ml/coverage_metrics.csv','paired_arrays.npz','figure4e_provenance.json','archived layer_sensitivity.csv','archived feature_model_depth_performance.csv'],'Figure_5_robustness_transfer.png':['robustness/metrics.csv','raw_augmentation/heldout.csv'],'independent_samples':'configurations only; no energy-point p-values or molecular pseudo-replication'})
print('Saved publication PNGs and conditional paired intervals')
