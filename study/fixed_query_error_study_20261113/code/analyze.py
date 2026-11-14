"""Saved-prediction metrics, paired geometry bootstrap, errors and score ranking."""
import json,sys
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.metrics import f1_score,accuracy_score,precision_recall_fscore_support,roc_auc_score,average_precision_score
O=Path(__file__).resolve().parents[1];REPO=O.parents[1];sys.path.insert(0,str(REPO))
from tools.spectral_multilabel import SPECIES
SEEDS=[20261020,20261021,20261022];METHODS=['old_only_locked','expanded_locked','old_only_recalibrated','expanded_recalibrated','expanded_H_threshold','expanded_residual']
def dump(n,x):(O/n).write_text(json.dumps(x,indent=2,default=lambda v:v.tolist() if isinstance(v,np.ndarray) else str(v))+'\n')
def main():
 rows=json.load(open(O/'dataset_manifest.json'));n=len(rows);fam=np.array([r['near_group'] for r in rows]);domain=np.array([r['domain'] for r in rows]);mixture=np.array(['+'.join(r['identity_labels']) or 'clean' for r in rows]);countmix=np.array(['+'.join(str(v)+k for k,v in sorted(r['observed_counts'].items())) or 'clean' for r in rows]);pure=np.array(['mixed' if len(r['identity_labels'])>1 else 'pure' if r['identity_labels'] else 'clean' for r in rows]);y=np.load(O/'predictions/old_only_locked_20261020.npz')['truth'];cache={};metrics=[];perlabel=[];cases=[];errors=[];rank=[];hist=[];transitions=[];h=SPECIES.index('H');co=SPECIES.index('CO');cho=SPECIES.index('CHO');oh=SPECIES.index('OH')
 def wts(ids):return np.array([1/np.sum(fam[ids]==fam[i]) for i in ids])
 def met(yy,pp,w):return dict(micro_f1=float(f1_score(yy,pp,average='micro',sample_weight=w,zero_division=0)),macro_f1=float(f1_score(yy,pp,average='macro',sample_weight=w,zero_division=0)),exact_set_accuracy=float(accuracy_score(yy,pp,sample_weight=w)))
 for m in METHODS:
  for seed in SEEDS:
   z=np.load(O/'predictions'/f'{m}_{seed}.npz');cache[m,seed]=(z['prediction'],z['score'])
  vote=np.sum([cache[m,s][0] for s in SEEDS],axis=0)>=2;vote_score=np.mean([cache[m,s][1] for s in SEEDS],axis=0);cache[m,'voting']=(vote.astype(int),vote_score)
  np.savez_compressed(O/'predictions'/f'{m}_voting.npz',truth=y,prediction=vote,families=fam,configuration=np.array([r['configuration'] for r in rows]),score_mean=vote_score)
  for seed in SEEDS+['voting']:
   p,s=cache[m,seed]
   for population,popids in [('original141',np.arange(141)),('new_queries',np.arange(141,n))]:
    masks=[('all','all',popids)]
    for typ,arr in [('substrate',domain),('species_set',mixture),('exact_stoichiometry',countmix),('pure_mixed',pure)]:
     masks.extend((typ,v,popids[arr[popids]==v]) for v in sorted(set(arr[popids])))
    for typ,v,ids in masks:
     w=wts(ids);r=dict(protocol=m,seed=seed,population=population,stratum=typ,value=v,records=len(ids),groups=len(set(fam[ids])));metrics.append({**r,**met(y[ids],p[ids],w)})
     pr,re,f1,supp=precision_recall_fscore_support(y[ids],p[ids],average=None,sample_weight=w,zero_division=0)
     for j,sp in enumerate(SPECIES):perlabel.append(dict(**r,species=sp,precision=float(pr[j]),recall=float(re[j]),f1=float(f1[j]),weighted_support=float(supp[j]),positive_records=int(y[ids,j].sum()),positive_groups=len(set(fam[ids[y[ids,j]==1]])),false_positives=int(np.sum((y[ids,j]==0)&(p[ids,j]==1))),omissions=int(np.sum((y[ids,j]==1)&(p[ids,j]==0)))))
    ids=popids;hm=ids[(y[ids,h]==1)&(pure[ids]=='mixed')];triple=ids[mixture[ids]=='CHO+CO+H'];coonly=ids[mixture[ids]=='CO'];hist.append(dict(protocol=m,seed=seed,population=population,H_containing_mixtures=len(hm),H_mixture_omissions=int(np.sum(p[hm,h]==0)),H_mixture_hits=int(np.sum(p[hm,h]==1)),CHO_CO_H_sets=len(triple),CHO_CO_H_exact=int(np.all(p[triple]==y[triple],axis=1).sum()),CO_only_cases=len(coonly),false_H_CO_only=int(p[coonly,h].sum()),H_positive_records=int(y[ids,h].sum()),H_omissions_all=int(np.sum((y[ids,h]==1)&(p[ids,h]==0)))))
    if population=='original141':
     w=wts(ids)
     for j,sp in enumerate(SPECIES):
      for scope,mask in [('all',np.ones(len(ids),bool)),('CO_containing',y[ids,co]==1)]:
       yy=y[ids[mask],j];ss=s[ids[mask],j];ww=w[mask]
       if len(np.unique(yy))==2:rank.append(dict(protocol=m,seed=seed,species=sp,scope=scope,positive_records=int(yy.sum()),negative_records=int((1-yy).sum()),auc=float(roc_auc_score(yy,ss,sample_weight=ww)),average_precision=float(average_precision_score(yy,ss,sample_weight=ww)),positive_score_median=float(np.median(ss[yy==1])),negative_score_median=float(np.median(ss[yy==0]))))
   for i,r in enumerate(rows):
    omitted=[sp for j,sp in enumerate(SPECIES) if y[i,j] and not p[i,j]];fp=[sp for j,sp in enumerate(SPECIES) if p[i,j] and not y[i,j]];cases.append(dict(protocol=m,seed=seed,index=i,configuration=r['configuration'],near_group=fam[i],fold=r['fold'],population='original141' if i<141 else 'new_queries',substrate=domain[i],species_set=mixture[i],exact_stoichiometry=countmix[i],pure_mixed=pure[i],truth=';'.join(r['identity_labels']),prediction=';'.join(sp for j,sp in enumerate(SPECIES) if p[i,j]),omissions=';'.join(omitted),false_positives=';'.join(fp),exact=not(omitted or fp),**{f'score_{sp}':float(s[i,j]) for j,sp in enumerate(SPECIES)}))
    for j,sp in enumerate(SPECIES):
     if y[i,j]!=p[i,j]:errors.append(dict(protocol=m,seed=seed,configuration=r['configuration'],near_group=fam[i],population='original141' if i<141 else 'new_queries',substrate=domain[i],species_set=mixture[i],exact_stoichiometry=countmix[i],pure_mixed=pure[i],species=sp,error='omission' if y[i,j] else 'false_positive',score=float(s[i,j])))
 for seed in SEEDS+['voting']:
  for a,b in [('old_only_locked','expanded_locked'),('expanded_locked','expanded_H_threshold'),('expanded_locked','expanded_residual')]:
   pa=cache[a,seed][0];pb=cache[b,seed][0]
   for i,r in enumerate(rows):
    for j,sp in enumerate(SPECIES):
     if pa[i,j]!=pb[i,j]:transitions.append(dict(seed=seed,before=a,after=b,population='original141' if i<141 else 'new_queries',configuration=r['configuration'],species=sp,truth=int(y[i,j]),before_prediction=int(pa[i,j]),after_prediction=int(pb[i,j]),change='corrected_omission' if y[i,j] and pb[i,j] else 'added_omission' if y[i,j] else 'added_false_positive' if pb[i,j] else 'corrected_false_positive'))
 mf=pd.DataFrame(metrics);mf.to_csv(O/'metrics.csv',index=False);pd.DataFrame(perlabel).to_csv(O/'per_label_metrics.csv',index=False);pd.DataFrame(cases).to_csv(O/'per_case_predictions_errors.csv',index=False);ef=pd.DataFrame(errors);ef.to_csv(O/'errors_long.csv',index=False);pd.DataFrame(hist).to_csv(O/'historical_error_checks.csv',index=False);pd.DataFrame(rank).to_csv(O/'score_separation.csv',index=False);pd.DataFrame(transitions).to_csv(O/'error_transitions.csv',index=False)
 for cols,name in [(['population','substrate','species'],'species_substrate'),(['population','species_set','species'],'species_set'),(['population','exact_stoichiometry','species'],'exact_stoichiometry'),(['population','pure_mixed','species'],'pure_mixed')]:ef.groupby(['protocol','seed',*cols,'error']).size().reset_index(name='errors').to_csv(O/f'error_breakdown_{name}.csv',index=False)
 overall=mf[(mf.stratum=='all')&(mf.seed!='voting')];means=overall.groupby(['population','protocol'])[['micro_f1','macro_f1','exact_set_accuracy']].agg(['mean','std','min','max']);means.to_csv(O/'seed_variation.csv')
 # Vectorized paired bootstrap: sample whole near groups; same draws for both arms and all seeds.
 rng=np.random.default_rng(20261113);boot=[]
 for population,ids in [('original141',np.arange(141)),('new_queries',np.arange(141,n))]:
  groups=sorted(set(fam[ids]));gi=[ids[fam[ids]==g] for g in groups];draws=rng.multinomial(len(groups),np.full(len(groups),1/len(groups)),size=5000)
  stat={}
  for m in METHODS:
   for seed in SEEDS+['voting']:
    p=cache[m,seed][0];st=[]
    for ix in gi:
     yy=y[ix];pp=p[ix];st.append(np.r_[np.mean(yy*pp,axis=0),np.mean((1-yy)*pp,axis=0),np.mean(yy*(1-pp),axis=0),np.mean(np.all(yy==pp,axis=1))])
    st=np.array(st);tot=draws@st;k=len(SPECIES);tp=tot[:,:k];fp=tot[:,k:2*k];fn=tot[:,2*k:3*k];den=2*tp+fp+fn;f1=np.divide(2*tp,den,out=np.zeros_like(tp),where=den>0);micro=2*tp.sum(axis=1)/(2*tp+fp+fn).sum(axis=1);stat[m,seed]=np.column_stack([micro,f1.mean(axis=1),tot[:,-1]/len(groups)])
  for before,after in [('old_only_locked','expanded_locked'),('old_only_locked','old_only_recalibrated'),('expanded_locked','expanded_recalibrated'),('expanded_locked','expanded_H_threshold'),('expanded_locked','expanded_residual')]:
   for scope in ['seed_mean','voting']:
    d=np.mean([stat[after,s]-stat[before,s] for s in SEEDS],axis=0) if scope=='seed_mean' else stat[after,'voting']-stat[before,'voting']
    for k,key in enumerate(['micro_f1','macro_f1','exact_set_accuracy']):
     mask=(mf.population==population)&(mf.stratum=='all');q=mf[mask];av=float(q[(q.protocol==after)&(q.seed!='voting')][key].mean()-q[(q.protocol==before)&(q.seed!='voting')][key].mean()) if scope=='seed_mean' else float(q[(q.protocol==after)&(q.seed=='voting')][key].iloc[0]-q[(q.protocol==before)&(q.seed=='voting')][key].iloc[0]);boot.append(dict(population=population,before=before,after=after,scope=scope,metric=key,difference=av,lower=float(np.quantile(d[:,k],.025)),upper=float(np.quantile(d[:,k],.975)),resamples=5000,groups=len(groups),interpretation='conditional on fixed saved OOF predictions; not refit or independent-parent uncertainty'))
 pd.DataFrame(boot).to_csv(O/'paired_bootstrap.csv',index=False)
 # Per-new-case status: new to baseline versus newly completed in latest export.
 inv={r['configuration']:r for r in json.load(open(O/'audit_inventory.json'))};new=[]
 for r in rows[141:]:
  a=inv[r['configuration']];new.append(dict(configuration=r['configuration'],domain=r['domain'],species_set='+'.join(r['identity_labels']),exact_stoichiometry='+'.join(str(v)+k for k,v in sorted(r['observed_counts'].items())),near_group=r['near_group'],added_layers=a['added_layers'],source_status='top-two completed November13' if any(l in a['added_layers'] for l in [4,5]) else 'top-two already available November3; absent from frozen November1 cohort',new_near_group=r['near_group'] not in set(fam[:141])))
 dump('new_case_version_classification.json',new)
 cf=pd.DataFrame(new);cf.groupby(['domain','species_set','exact_stoichiometry','source_status']).agg(records=('configuration','nunique'),near_groups=('near_group','nunique'),new_near_groups=('new_near_group','sum')).reset_index().to_csv(O/'new_cohort_counts.csv',index=False)
 primary=[r for r in boot if r['population']=='original141' and r['before']=='old_only_locked' and r['after']=='expanded_locked'];summary=dict(primary=primary,overall=mf[(mf.stratum=='all')].to_dict('records'),historical=hist,targeted_changes= [r for r in boot if r['population']=='original141' and r['scope']=='seed_mean' and r['after'] in ['expanded_H_threshold','expanded_residual']],note='Seed variation is reported separately; voting is majority of three predeclared binary predictions. No best seed is used for model selection.');dump('analysis_summary.json',summary)
 print('PRIMARY',json.dumps(primary,indent=2));print(pd.DataFrame(hist).query('population=="original141" and protocol in ["old_only_locked","expanded_locked","expanded_H_threshold","expanded_residual"]').to_string(index=False));print('Analysis complete',flush=True)
if __name__=='__main__':main()
