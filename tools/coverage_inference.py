"""Spectra-only configuration-level CO loading and local-contact inference.

Cu100 is an external input-geometry domain; no pooling with the CuAu cohort.
Selection is nested by exact physical geometry. Molecules/layers are not samples.
"""
from pathlib import Path
import json,hashlib,platform
from collections import Counter
import numpy as np
import pandas as pd
import joblib,sklearn,scipy
from scipy.fft import dct
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.dummy import DummyRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold,LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from .coverage_study import dump,sha,SEED
from .spectral_multilabel import feature_bank,fit_binary,binary_scores,encode_labels,SPECIES,f1_metrics
from .updated_data_training import threshold_fit
from .contact_checks import minimum_image_vectors

FEATURES=('Combined','DCT8','DCT16','DCT24')
RECIPES=tuple((f,kind,leaf) for f in FEATURES for kind,leaf in [('trees',2),('trees',4),('ridge',10),('ridge',100)])

def bank(raw,energy):
    b=feature_bank(raw,energy);return {'Combined':b['shape_CDF_derivatives'],**{f'DCT{k}':dct(b['normalized'],type=2,norm='ortho',axis=1)[:,:k] for k in [8,16,24]}}
def estimator(recipe):
    f,kind,p=recipe
    return ExtraTreesRegressor(n_estimators=128,min_samples_leaf=p,max_features=1.,random_state=SEED,n_jobs=1) if kind=='trees' else make_pipeline(StandardScaler(),Ridge(alpha=p))
def metrics(y,p):return {'n':len(y),'MAE':float(mean_absolute_error(y,p)),'RMSE':float(np.sqrt(mean_squared_error(y,p))),'R2':float(r2_score(y,p)) if len(y)>1 and np.ptp(y)>0 else None}
def contacts(s,cut):
    a=np.array(s['atoms']);pos=np.asarray(s['positions']);cell=np.asarray(s['cell']);ids=np.flatnonzero(a=='C');m=np.flatnonzero(np.isin(a,['Cu','Au']));result=[]
    for i in ids:
        ds=np.linalg.norm(minimum_image_vectors(pos[m]-pos[i],cell,(True,True,False)),axis=1);els=a[m[ds<=cut]];cu=int(sum(els=='Cu'));au=int(sum(els=='Au'));result.append((cu,au))
    return result

def run(out):
    out=Path(out).resolve();allrows=json.loads((out/'configuration_manifest.json').read_text());z=np.load(out/'spectra.npz');geos={x['configuration']:x for x in json.loads((out/'geometry_arrays.json').read_text())};seen=set();rows=[];raw=[];grids=[];duplicates=[]
    for i,r in enumerate(allrows):
        if r['group'] in seen:duplicates.append(r['configuration']);continue
        seen.add(r['group']);rows.append(r);raw.append(z[f'raw_{i}']);grids.append(z[f'energy_{i}'])
    # Fixed common support established without labels. No extrapolation or shifts.
    lo=max(e[0] for e in grids);hi=min(e[-1] for e in grids);energy=np.arange(np.ceil(lo*5)/5,np.floor(hi*5)/5+.01,.2);raw=np.array([np.interp(energy,e,x) for x,e in zip(raw,grids)]);b=bank(raw,energy)
    pure=np.array([i for i,r in enumerate(rows) if r['domain']=='CuAu' and r['CO_only']],int);external=np.array([i for i,r in enumerate(rows) if r['domain']=='Cu100' and r['CO_only']],int);groups=np.array([r['group'] for r in rows]);y=np.array([r['coverage_ML'] for r in rows]);density=np.array([r['coverage_CO_per_nm2'] for r in rows]);models=out/'models';models.mkdir(exist_ok=True);protocol={'version':'CO-coverage-v1','seed':SEED,'features':FEATURES,'candidates':RECIPES,'n_estimators':128,'outer':'5-fold GroupKFold by geometry; separate leave-coverage-level-out and contact-motif-held-out tests','inner':'3-fold GroupKFold by geometry; minimum coverage MAE; same folds across all recipes','target':'intact CO molecules / all top-surface metal sites','additional_denominators':'per surface Cu and molecules/nm2 saved separately; names never define loading','predictors':'spectra only; population-weighted native top-two average; archived minimum-subtracted unit area; Combined or DCT compression','Cu100_use':'no Cu100 in fitting/selection; external input-geometry diagnostic only with matched original ASE snapshots; production relaxation termination not supplied','source_hashes':{p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('coverage_study.py'),Path(__file__).with_name('cu100_lineage.py')]},'dependencies':{'python':platform.python_version(),'numpy':np.__version__,'scipy':scipy.__version__,'sklearn':sklearn.__version__},'energy_window_eV':[float(energy[0]),float(energy[-1])],'energy_points':len(energy),'duplicate_aliases_excluded':duplicates,'confirmation':'exploratory reused configurations; not locked prospective validation'};dump(out/'protocol.json',protocol)
    if len(pure)<10:raise ValueError('Insufficient audited CO-only configurations')
    selections=[];predictions=[];summary=[];recipes_oof={};folds=np.full(len(pure),-1);outer=list(GroupKFold(min(5,len(set(groups[pure])))).split(pure,groups=groups[pure]))
    tasks={'coverage_configuration_folds':outer}
    if len(set(y[pure]))>=3:tasks['coverage_level_held_out']=list(LeaveOneGroupOut().split(pure,groups=np.round(y[pure],10)))
    motif=[]
    for i in pure:motif.append(';'.join(f'{cu}Cu{au}Au' for cu,au in sorted(contacts(geos[rows[i]['configuration']],2.25))))
    if len(set(motif))>=3:tasks['contact_motif_held_out']=list(GroupKFold(min(5,len(set(motif)))).split(pure,groups=motif))
    for task,splits in tasks.items():
        pp={name:np.full(len(pure),np.nan) for name in ['nested','fixed_Combined_leaf2','training_mean','training_median']}
        for fold,(tr0,te0) in enumerate(splits):
            tr=pure[tr0];te=pure[te0];inner=list(GroupKFold(min(3,len(tr))).split(tr,groups=groups[tr]));ranks=[]
            for recipe in RECIPES:
                f,_,_=recipe;ip=np.full(len(tr),np.nan)
                for a,v in inner:
                    m=estimator(recipe).fit(b[f][tr[a]],y[tr[a]]);ip[v]=m.predict(b[f][tr[v]])
                ranks.append(float(mean_absolute_error(y[tr],ip)))
            best=RECIPES[int(np.argmin(ranks))];selected=estimator(best).fit(b[best[0]][tr],y[tr]);fixed=estimator(('Combined','trees',2)).fit(b['Combined'][tr],y[tr]);values={'nested':selected.predict(b[best[0]][te]),'fixed_Combined_leaf2':fixed.predict(b['Combined'][te]),'training_mean':np.full(len(te),y[tr].mean()),'training_median':np.full(len(te),np.median(y[tr]))}
            if task=='coverage_configuration_folds':
                folds[te0]=fold;joblib.dump({'estimator':selected,'feature':best[0],'energy':energy,'population_rule':'physical top two; explicit populations required','train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'target':'CO/top-surface-metal','training_range':[float(y[tr].min()),float(y[tr].max())]},models/f'coverage_fold{fold}.joblib')
            for name,p in values.items():
                pp[name][te0]=p
                for i,pv in zip(te,p):predictions.append({'protocol':task,'model':name,'configuration':rows[i]['configuration'],'group':groups[i],'fold':fold,'truth_ML':y[i],'prediction_ML':pv,'n_CO':rows[i]['n_CO'],'surface_sites':rows[i]['surface_sites'],'truth_per_nm2':density[i],'prediction_per_nm2':pv*rows[i]['surface_sites']/(rows[i]['cell_area_A2']/100),'training_target_min':float(y[tr].min()),'training_target_max':float(y[tr].max()),'outside_training_coverage':bool(y[i]<y[tr].min() or y[i]>y[tr].max())})
            selections.append({'protocol':task,'fold':fold,'selected':best,'inner_MAE':ranks,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist()})
        for name,p in pp.items():summary.append({'protocol':task,'model':name,**metrics(y[pure],p)})
        recipes_oof[task]=pp
    # Final fit used only for external-domain diagnosis, not as a scored OOF model.
    rank=[]
    for rec in RECIPES:
        pp=np.zeros(len(pure))
        for a,v in outer:pp[v]=estimator(rec).fit(b[rec[0]][pure[a]],y[pure[a]]).predict(b[rec[0]][pure[v]])
        rank.append(mean_absolute_error(y[pure],pp))
    rec=RECIPES[int(np.argmin(rank))];final=estimator(rec).fit(b[rec[0]][pure],y[pure]);joblib.dump({'estimator':final,'feature':rec[0],'energy':energy,'train_groups':groups[pure].tolist(),'test_groups':groups[external].tolist(),'target':'CO/top-surface-metal','training_range':[float(y[pure].min()),float(y[pure].max())]},models/'CuAu_only_external_Cu100.joblib')
    external_rows=[]
    for domain,ids in [('Cu100_input_geometry_external',external),('CuAu_CO_mixtures_diagnostic',np.array([i for i,r in enumerate(rows) if r['domain']=='CuAu' and r['n_CO']>0 and not r['CO_only']],int))]:
        if not len(ids):continue
        p=final.predict(b[rec[0]][ids]);summary.append({'protocol':domain,'model':'CuAu_CO_only_nested_recipe',**metrics(y[ids],p)})
        for i,pv in zip(ids,p):external_rows.append({'domain':domain,'configuration':rows[i]['configuration'],'truth_ML':y[i],'prediction_ML':pv,'n_CO':rows[i]['n_CO'],'surface_sites':rows[i]['surface_sites'],'CONTCAR_verified':rows[i]['CONTCAR_available'],'outside_training_coverage':bool(y[i]<y[pure].min() or y[i]>y[pure].max())})
    # Contact regression predicts configuration-averaged coordination counts,
    # not a fictitious unique molecular site in multicarbon configurations.
    coord=np.array([[np.mean([c for c,a in contacts(geos[rows[i]['configuration']],2.25)]),np.mean([a for c,a in contacts(geos[rows[i]['configuration']],2.25)])] for i in pure]);cp=np.zeros_like(coord);cb=np.zeros_like(coord);contact_rows=[]
    for fold,(tr,te) in enumerate(outer):
        m=ExtraTreesRegressor(n_estimators=128,min_samples_leaf=2,random_state=SEED,n_jobs=1).fit(b['Combined'][pure[tr]],coord[tr]);cp[te]=m.predict(b['Combined'][pure[te]]);cb[te]=coord[tr].mean(axis=0);joblib.dump({'estimator':m,'feature':'Combined','energy':energy,'train_groups':groups[pure[tr]].tolist(),'test_groups':groups[pure[te]].tolist(),'target':['mean_CO_Cu_contacts_2p25A','mean_CO_Au_contacts_2p25A']},models/f'contacts_fold{fold}.joblib')
    support=Counter()
    for j,i in enumerate(pure):
        s=geos[rows[i]['configuration']];c=contacts(s,2.25);support.update(c);stable=contacts(s,2.15)==c==contacts(s,2.35);contact_rows.append({'configuration':rows[i]['configuration'],'contacts_per_CO':c,'cutoff_stable_2p15_2p35':stable,'truth_mean_Cu':coord[j,0],'truth_mean_Au':coord[j,1],'prediction_mean_Cu':cp[j,0],'prediction_mean_Au':cp[j,1],'baseline_mean_Cu':cb[j,0],'baseline_mean_Au':cb[j,1]})
    dump(out/'contact_support.json',{'contacts_Cu_Au_molecule_counts':{str(k):v for k,v in support.items()},'hollow_series_molecule_support':{f'Cu{3-a}Au{a}':support[(3-a,a)] for a in range(4)},'caution':'molecule counts are not independent configuration support; single fixed Combined/leaf2 contact model, no outer-score selection; cutoff sensitivity is reported, contact count is not electronic bond order','metrics':{'Cu':metrics(coord[:,0],cp[:,0]),'Au':metrics(coord[:,1],cp[:,1]),'Cu_mean_baseline':metrics(coord[:,0],cb[:,0]),'Au_mean_baseline':metrics(coord[:,1],cb[:,1])}})
    dump(out/'contact_predictions.json',contact_rows)
    # Updated identity evaluation on all supported CuAu labels, fixed top-two
    # Combined and training-only thresholds. Sparse species support is explicit.
    ident=np.array([i for i,r in enumerate(rows) if r['domain']=='CuAu' and set(r['identity_labels'])<=set(SPECIES)],int);truth=encode_labels([rows[i]['identity_labels'] for i in ident]);score=np.zeros_like(truth,float);prediction=np.zeros_like(truth);identity_splits=[]
    for fold,(tr0,te0) in enumerate(GroupKFold(5).split(ident,groups=groups[ident])):
        tr=ident[tr0];te=ident[te0];inside=np.zeros((len(tr),len(SPECIES)))
        for a,v in GroupKFold(3).split(tr,groups=groups[tr]):
            ms=fit_binary(b['Combined'][tr[a]],truth[tr0[a]],'independent_trees');inside[v]=binary_scores(ms,b['Combined'][tr[v]])
        _,thresholds,_=threshold_fit(truth[tr0],inside);ms=fit_binary(b['Combined'][tr],truth[tr0],'independent_trees');score[te0]=binary_scores(ms,b['Combined'][te]);prediction[te0]=(score[te0]>=thresholds).astype(int);identity_splits.append({'fold':fold,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'training_positive_support':truth[tr0].sum(axis=0).tolist(),'test_positive_support':truth[te0].sum(axis=0).tolist(),'thresholds':thresholds.tolist()});joblib.dump({'estimators':ms,'thresholds':thresholds,'feature':'Combined','energy':energy,'train_groups':groups[tr].tolist(),'test_groups':groups[te].tolist(),'species':SPECIES},models/f'identity_fold{fold}.joblib')
    dump(out/'identity_metrics.json',{'protocol':'new fixed-top-two Combined grouped evaluation; not same-fold update of archived87/90 scores','species':SPECIES,'n':len(ident),**f1_metrics(truth,prediction),'splits':identity_splits});np.savez_compressed(out/'identity_oof.npz',truth=truth,prediction=prediction,score=score,configuration=np.array([rows[i]['configuration'] for i in ident]),species=np.array(SPECIES))
    # Zero-shot identity transfer: train/calibrate exclusively on CuAu. Cu100
    # scores never select a recipe or threshold. Unsupported chemistry excluded.
    extid=np.array([i for i,r in enumerate(rows) if r['domain']=='Cu100' and set(r['identity_labels'])<=set(SPECIES)],int)
    _,it,_=threshold_fit(truth,score);im=fit_binary(b['Combined'][ident],truth,'independent_trees')
    joblib.dump({'estimators':im,'thresholds':it,'feature':'Combined','energy':energy,'train_groups':groups[ident].tolist(),'test_groups':groups[extid].tolist(),'species':SPECIES},models/'identity_CuAu_only_external_Cu100.joblib')
    if len(extid):
        es=binary_scores(im,b['Combined'][extid]);ey=encode_labels([rows[i]['identity_labels'] for i in extid]);ep=(es>=it).astype(int)
        np.savez_compressed(out/'identity_Cu100_external.npz',truth=ey,prediction=ep,score=es,thresholds=it,configuration=np.array([rows[i]['configuration'] for i in extid]),species=np.array(SPECIES))
        dump(out/'identity_Cu100_external_metrics.json',{'n':len(extid),**f1_metrics(ey,ep),'interpretation':'zero-shot input-geometry diagnostic, no Cu100 fitting or calibration; 11-label macro includes unsupported labels','positive_support':ey.sum(axis=0).tolist(),'species':SPECIES})
    nn=[];allcu=np.array([i for i,r in enumerate(rows) if r['domain']=='CuAu']);norm=feature_bank(raw,energy)['normalized']
    for i in [i for i,r in enumerate(rows) if r['domain']=='Cu100']:
        distance=np.linalg.norm(norm[allcu]-norm[i],axis=1);nearest=np.argsort(distance)[:3]
        nn.append({'configuration':rows[i]['configuration'],'neighbors':[{'configuration':rows[allcu[j]]['configuration'],'distance_unit_area_L2':float(distance[j]),'labels':rows[allcu[j]]['identity_labels']} for j in nearest],'domain':'Cu100; facet/composition/cell/layer-population change; no validated parent transfer'})
    dump(out/'Cu100_training_neighbors.json',nn)
    # Conditional paired bootstrap for coverage MAE change vs training-mean.
    p=recipes_oof['coverage_configuration_folds'];rng=np.random.default_rng(SEED);delta=[]
    for _ in range(2000):
        ids=rng.integers(len(pure),size=len(pure));delta.append(float(np.mean(abs(p['nested'][ids]-y[pure][ids]))-np.mean(abs(p['training_mean'][ids]-y[pure][ids]))))
    dump(out/'uncertainty.json',{'conditional_paired_MAE_delta_vs_training_mean':float(np.mean(abs(p['nested']-y[pure]))-np.mean(abs(p['training_mean']-y[pure]))),'percentile95':np.quantile(delta,[.025,.975]).tolist(),'scope':'configuration resampling conditional on these fixed predictions; does not include refitting or selection variation'})
    pd.DataFrame(summary).to_csv(out/'coverage_metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'coverage_heldout_predictions.csv',index=False);pd.DataFrame(external_rows).to_csv(out/'coverage_external_predictions.csv',index=False);dump(out/'selection_audit.json',selections);dump(out/'model_configuration_manifest.json',rows)
    np.savez_compressed(out/'model_inputs.npz',raw=raw,energy=energy,groups=groups,pure=pure,external=external,coverage_ML=y,identity_indices=ident)
    return {'configurations':len(rows),'pure_CO_CuAu':len(pure),'pure_CO_Cu100_external':len(external),'identity_CuAu':len(ident),'metrics':summary}


def predict(record,model,populations):
    """Require explicit physical populations; never assume CuAu layer numbering."""
    series=record['measurement']['series'];arrays={int(s['series_id'].rsplit('_',1)[1]):np.column_stack([s['independent_variables'][0]['values'],s['channels'][0]['values']]) for s in series if s['series_id'].startswith('Cu_K_layer_')};layers=record.get('geometry_provenance',{}).get('top_two_layers',[])
    if len(layers)!=2 or any(l not in arrays for l in layers):raise ValueError('Audited physical top-two layers required; missing layers cannot be substituted')
    if len(layers)!=2 or any(l not in populations or populations[l]<=0 for l in layers):raise ValueError('Two physical layers and verified populations required')
    energy=np.asarray(model['energy']);raw=[]
    for l in layers:
        a=arrays[l]
        if a[0,0]>energy[0]+1e-8 or a[-1,0]<energy[-1]-1e-8:raise ValueError('Query lacks trained energy support; no extrapolation')
        raw.append(np.interp(energy,a[:,0],a[:,1]))
    x=sum(a*populations[l] for a,l in zip(raw,layers))/sum(populations[l] for l in layers);value=model['estimator'].predict(bank(x[None,:],energy)[model['feature']]);return value

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output-dir',required=True);a=p.parse_args();print(json.dumps(run(a.output_dir),indent=2))
