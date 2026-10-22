"""Read-only updated-export audit and versioned, training-only selector refit."""
from pathlib import Path
from collections import Counter
import gzip,hashlib,json,sys
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support
from .chemical_lineage import scan,correspondence
from .component_audit import read_poscar,audit_components,legacy_benchmark_geometry_hash
from .dataset_labels import geometry_hash
from .layer_population import verify_layer_population,aggregate_layers
from .spectral_multilabel import (SPECIES,DEPTHS,record_identity,spectrum_layers,encode_labels,
    feature_bank,feature_for_records,fit_binary,binary_scores,f1_metrics,predict_records,implementation_fingerprint)
from .scientific_extensions import single_mixed

SEED=20261020
THRESHOLDS=np.array([.15,.25,.35,.45,.5,.55,.65,.75])


def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def threshold_fit(y,score):
    """Same micro objective/global+shrunk decoder as frozen primary; training only."""
    def quality(p):
        tp=((y==1)&p).sum(axis=0);fp=((y==0)&p).sum(axis=0);fn=((y==1)&~p).sum(axis=0);den=2*tp+fp+fn
        return float(2*tp.sum()/den.sum()) if den.sum() else 0.
    order=sorted(THRESHOLDS,key=lambda t:abs(t-.5));q=[quality(score>=t) for t in order];i=int(np.argmax(q));t=float(order[i]);v=np.full(y.shape[1],t);best=q[i];adjusted=v.copy();counts=y.sum(axis=0)
    for j in range(y.shape[1]):
        if counts[j]<5 or len(y)-counts[j]<10:continue
        choices=np.array([a for a in THRESHOLDS if abs(a-t)<=.15000001]);p=score[:,j][None,:]>=choices[:,None];tp=(p&(y[:,j][None,:]==1)).sum(axis=1);den=p.sum(axis=1)+counts[j];per=np.divide(2*tp,den,out=np.zeros(len(choices)),where=den>0);adjusted[j]=t+.5*(choices[np.argmax(per)]-t)
    new=quality(score>=adjusted)
    return (new,adjusted,'shrunk') if new>best+1e-12 else (best,v,'global')


def inner_folds(y,fold):
    best=None
    for trial in range(60):
        splits=list(KFold(3,shuffle=True,random_state=SEED+100*fold+trial).split(y))
        loss=(sum((y[a].sum(axis=0)==0).sum() for a,b in splits),sum(np.sum((y[b].sum(axis=0)-y.sum(axis=0)/3)**2/(y.sum(axis=0)+1)) for a,b in splits))
        if best is None or loss<best[0]:best=(loss,splits)
    return best[1]


def audit_updated(root,out,repo):
    root=Path(root);out=Path(out);repo=Path(repo);out.mkdir(parents=True,exist_ok=True);objects=out/'source_objects';objects.mkdir(exist_ok=True)
    old=pd.read_csv(repo/'study/manuscript_package/figure_data/cv_manifest.csv');oldgeom={}
    export=repo/'study/extensions_v2/ml/cohort_spectra.npz';z=np.load(export)
    # Map by geometry, not the mutable aliases. Old row hashes are kept separately.
    legacy=dict(zip(old.structure_hash,old.to_dict('records')));files=scan(root)
    candidates={p.parent: p for p in files if p.name=='isaac_record_layers.json' and '/FDMNES/green/' in str(p)}
    for p in files:
        if p.name=='Cu_layer5_conv.txt' and '/FDMNES/green/' in str(p):candidates.setdefault(p.parent.parent,None)
    inventory=[];samples=[];source_manifest=[];seen=set();energy=None
    def capture(path,role):
        p=Path(path);b=p.read_bytes();h=hashlib.sha256(b).hexdigest()
        if h not in seen:
            with (objects/(h+'.gz')).open('wb') as f:f.write(gzip.compress(b,mtime=0))
            seen.add(h)
        source_manifest.append({'path':str(p.relative_to(root)),'role':role,'bytes':len(b),'sha256':h,'object':h+'.gz'})
    for folder,path in sorted(candidates.items(),key=lambda item:str(item[0])):
        alias=str(folder.relative_to(root)).split('/XAS/')[0];row={'configuration':alias,'record':str(path.relative_to(root)) if path else None,'record_origin':'supplied' if path else 'derived_from_verified_native_outputs'}
        try:
            structure=read_poscar(folder/'layer_5/CONTCAR');a=audit_components(structure)
            if a['status']!='verified':raise ValueError('Ambiguous or cutoff-sensitive molecular graph')
            labels=set(a['identities'])
            if labels-set(SPECIES):raise ValueError('Unsupported target species '+str(labels-set(SPECIES)))
            final=root/alias/'CONTCAR'
            if not final.exists() or not final.stat().st_size:raise ValueError('No valid saved relaxed geometry; initial geometry cannot replace it')
            link=correspondence(read_poscar(final),structure)
            lineage=None
            if link['status']!='matched':
                prior_lineage=repo/'study/retraining_20261022/historical_lineage.json'
                records=json.loads(prior_lineage.read_text())['records_by_legacy_geometry'] if prior_lineage.exists() else {}
                lineage=records.get(legacy_benchmark_geometry_hash(structure))
                if not lineage or digest(final)!=lineage['root_relaxed_sha256'] or digest(folder/'layer_5/CONTCAR')!=lineage['XAS_used_geometry_sha256'] or sorted(labels)!=sorted(lineage['archived_labels']):
                    raise ValueError('Root/XAS geometry mismatch without verified historical calculation lineage')
                for l in [2,3,4,5]:
                    if digest(folder/f'layer_{l}/Cu_layer{l}_conv.txt')!=lineage['convolved_output_sha256'][str(l)]:raise ValueError('Historical output changed; cannot reuse old lineage')
                row['lineage_warning']='Root and exact XAS geometry differ, already in the archived export. Both saved geometry files and all four native outputs retain exact historical byte hashes. Labels derive only from the actual XAS-used geometry; no root-XAS equality or new relaxed state is asserted.'
                row['historical_lineage']=lineage
            population={};evidence={};layers={}
            for l in [2,3,4,5]:
                f=folder/f'layer_{l}';evidence[l]=verify_layer_population(f,l);population[l]=evidence[l]['population']
                if correspondence(structure,read_poscar(f/'CONTCAR'))['status']!='matched':raise ValueError('Different XAS-used geometries between layers')
                layers[l]=np.loadtxt(f/f'Cu_layer{l}_conv.txt',skiprows=1)
            if not all(np.array_equal(layers[2][:,0],arr[:,0]) for arr in layers.values()):raise ValueError('Incompatible layer grids')
            if energy is not None and not np.array_equal(energy,layers[2][:,0]):raise ValueError('Incompatible cohort grid; no extrapolation')
            if path:
                d=json.loads(path.read_text());record_layers=spectrum_layers(d)
                if set(record_identity(d))!=labels:raise ValueError('Record identity differs from XAS-used full graph')
                c=d['sample']['composition']
                if 'adsorbate_counts' in c and c['adsorbate_counts']!=a['counts']:raise ValueError('Record molecule counts differ from graph')
                if c.get('atom_counts')!=dict(Counter(structure['atoms'])):raise ValueError('Record atom counts differ from actual geometry')
                for l in layers:
                    if not np.array_equal(record_layers[l][:,0],layers[l][:,0]) or not np.allclose(record_layers[l][:,1],layers[l][:,1],rtol=1e-7,atol=1e-9):raise ValueError('Record/native numeric output mismatch')
                capture(path,'unaltered supplied AI-ready record')
            else:
                d={'isaac_record_version':'recovered-numeric-v1','record_id':hashlib.sha256(alias.encode()).hexdigest(),'sample':{'composition':{'atom_counts':dict(Counter(structure['atoms'])),'adsorbate_counts':a['counts'],'adsorbate_identities':sorted(labels),'adsorbate_count_total':sum(a['counts'].values())},'structure_descriptors':{'component_audit':a}},'measurement':{'series':[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'name':'photon_energy','unit':'eV','values':layers[l][:,0].tolist()}],'channels':[{'name':'absorption_cross_section','unit':'not_reported','values':layers[l][:,1].tolist()}]} for l in [2,3,4,5]]},'provenance':{'origin':'Recovered existing native outputs after full geometry/layer/population audit; no new spectrum or inferred core-hole default','source_configuration':alias,'core_hole':'not_reported','native_broadening':'retained as supplied'}}
                recovered=out/'recovered_records';recovered.mkdir(exist_ok=True);(recovered/(alias.replace('/','__')+'.json')).write_text(json.dumps(d,indent=2)+'\n')
            stack=np.stack([layers[l][:,1] for l in [2,3,4,5]]);g=geometry_hash(structure);h=legacy_benchmark_geometry_hash(structure);prior=legacy.get(h);sh=hashlib.sha256(np.round(stack,10).tobytes()).hexdigest()
            row.update(status='eligible',geometry_hash=g,legacy_geometry_hash=h,labels=';'.join(sorted(labels)) or 'clean',counts=a['counts'],prior_configuration=prior['configuration'] if prior else None,prior_fold=int(prior['fold']) if prior else None,record_sha256=digest(path) if path else None,spectrum_sha256=sh,populations=population,component_audit=a,root_XAS_correspondence=link,layer_evidence=evidence)
            samples.append({'row':row,'record':d,'layers':layers,'population':population,'group':g,'prior':prior});energy=layers[2][:,0]
            # Compact lossless evidence export; large site files/logs stay local with hashes in layer evidence.
            for p in [root/alias/'POSCAR',final]:
                if p.exists():capture(p,'source initial or relaxed geometry')
            for l in [2,3,4,5]:
                f=folder/f'layer_{l}'
                for name in ['CONTCAR','absorbers.tsv',f'Cu_layer{l}_in.txt',f'Cu_layer{l}_convolution_in.txt',f'Cu_layer{l}_conv.txt']:
                    if (f/name).exists():capture(f/name,'actual XAS input, absorber definition or convolved output')
                if evidence[l]['convolution_log']:capture(f/evidence[l]['convolution_log'],'native convolution log')
        except (ValueError,KeyError,IndexError,FileNotFoundError) as exc:row.update(status='excluded',reason=str(exc))
        inventory.append(row)
    samples.sort(key=lambda s:(s['prior'] is None,s['row']['configuration']));selected=[];geom={};spectra={}
    for s in samples:
        row=s['row'];dup=geom.get(s['group']) or spectra.get(row['spectrum_sha256'])
        if dup:
            if dup['row']['labels']!=row['labels']:raise ValueError('Conflicting labels for duplicate data')
            row.update(status='duplicate',duplicate_of=dup['row']['configuration']);continue
        geom[s['group']]=s;spectra[row['spectrum_sha256']]=s;selected.append(s)
    selected.sort(key=lambda s:s['row']['configuration']);y=encode_labels([s['row']['labels'].split(';') for s in selected]);folds=np.array([s['row']['prior_fold'] if s['prior'] else -1 for s in selected]);rng=np.random.default_rng(SEED)
    for i in np.flatnonzero(folds<0):
        losses=[2*(folds==f).sum()/len(y)+np.sum(y[folds==f].sum(axis=0)[y[i]==1]/np.maximum(y.sum(axis=0)[y[i]==1],1)) for f in range(5)];choices=np.flatnonzero(np.isclose(losses,min(losses)));folds[i]=int(rng.choice(choices))
    manifest=[];comparison=[]
    for i,s in enumerate(selected):
        row=s['row'];manifest.append({'configuration':row['configuration'],'fold':int(folds[i]),'structure_hash':row['legacy_geometry_hash'],'geometry_group':s['group'],'labels':row['labels'],'record_sha256':row['record_sha256'],'record_origin':row['record_origin'],'prior_configuration':row['prior_configuration'],'spectrum_sha256':row['spectrum_sha256']})
        if s['prior']:
            j=old.index[old.configuration==s['prior']['configuration']][0]
            if not np.array_equal(y[i],z['truth'][j]):raise ValueError('Changed old target; separate label-correction protocol required')
            if not np.array_equal(s['layers'][5][:,1],z['raw'][j]):raise ValueError('Old spectral data changed; separate measurement-correction comparison required')
            comparison.append({'current_configuration':row['configuration'],'prior_configuration':s['prior']['configuration'],'old_fold':int(s['prior']['fold']),'current_fold':int(folds[i]),'label_identical':True,'surface_array_identical':True})
    pd.DataFrame(manifest).to_csv(out/'cv_manifest.csv',index=False);pd.DataFrame(comparison).to_csv(out/'before_after_cohort.csv',index=False)
    (out/'record_audit.json').write_text(json.dumps(inventory,indent=2)+'\n');pd.DataFrame([{k:v for k,v in x.items() if k not in ['layer_evidence','component_audit','root_XAS_correspondence']} for x in inventory]).to_csv(out/'record_audit.csv',index=False)
    (out/'source_manifest.json').write_text(json.dumps(source_manifest,indent=2)+'\n')
    np.savez_compressed(out/'cohort.npz',stack=np.array([np.stack([s['layers'][l][:,1] for l in [2,3,4,5]]) for s in selected]),populations=np.array([[s['population'][l] for l in [2,3,4,5]] for s in selected]),energy=energy,truth=y,fold=folds,groups=np.array([s['group'] for s in selected]),configuration=np.array([s['row']['configuration'] for s in selected]),species=SPECIES)
    summary={'configurations':len(selected),'layer_spectra':4*len(selected),'prior_configurations':len(comparison),'new_or_recovered_configurations':[s['row']['configuration'] for s in selected if not s['prior']],'supplied_record_attempts':sum(p is not None for p in candidates.values()),'raw_without_record_attempts':sum(p is None for p in candidates.values()),'statuses':dict(Counter(x['status'] for x in inventory)),'frozen_archive_changed':False,'source_mutated':False,'training_groups_are_independent_geometries':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2),flush=True);return summary


def fit_snapshot(dataset,out):
    dataset=Path(dataset);out=Path(out);out.mkdir(parents=True,exist_ok=True);z=np.load(dataset/'cohort.npz');raw=z['stack'];weights=z['populations'];e=z['energy'];y=z['truth'];fold=z['fold'];groups=z['groups'];configs=z['configuration'];n=len(y)
    if len(set(groups))!=n:raise ValueError('Deduplicate physical configurations before fitting')
    if tuple(z['species'])!=SPECIES or raw.shape!=(n,4,len(e)) or y.shape!=(n,len(SPECIES)):raise ValueError('Invalid snapshot dimensions')
    banks={};keys=[]
    for depth,layers in DEPTHS.items():
        ix=np.array(layers)-2;averaged=np.array([np.average(a[ix],axis=0,weights=w[ix]) for a,w in zip(raw,weights)])
        for feature,a in feature_bank(averaged,e).items():banks[depth+'::'+feature]=a;keys.append(depth+'::'+feature)
    from scipy.signal import savgol_filter
    positive=raw-raw.min(axis=2,keepdims=True);scaled=positive/np.trapezoid(positive[:,3,:],e,axis=1)[:,None,None]
    def block(a,w=11):return np.hstack([savgol_filter(a,w,3,axis=1),savgol_filter(a,w,3,deriv=1,delta=float(np.median(np.diff(e))),axis=1),savgol_filter(a,w,3,deriv=2,delta=float(np.median(np.diff(e))),axis=1)])
    banks['depth_channels::contrast']=np.hstack([block(scaled[:,3,:]),block(scaled[:,3,:]-scaled[:,2,:])]);banks['depth_channels::multiscale']=np.hstack([block(scaled[:,3,:],w) for w in [7,11,21]]);keys+=['depth_channels::contrast','depth_channels::multiscale']
    protocol={'version':'updated-data-v3','seed':SEED,'candidate_count':len(keys),'candidates':keys,'model':'independent per-target Extra Trees,64 trees,min leaf1,balanced,sqrt features; constants for missing classes','thresholds':THRESHOLDS.tolist(),'selection':'micro F1; same global+shrunk decoder and training-label-balanced three-fold inner split search as frozen primary; only independent tree family refitted','outer':'five configuration folds; all matched prior geometry folds preserved; only added/recovered configurations assigned deterministically using target balance and counts','inner':'60 candidate KFold(3) partitions selected by training-label support/balance only; geometry-deduplicated rows are independent groups, explicitly checked','normalization':'unchanged minimum-subtracted unit-area; raw population averages precede normalization','feature_scaling':'no cohort-level transforms; fixed per-spectrum features only','calibration':'not calibrated; tree scores are not asserted reliable probabilities','source_sha256':digest(__file__),'cohort_sha256':digest(dataset/'cohort.npz'),'fingerprint':implementation_fingerprint(),'new_holdout':'newly usable configurations excluded from every training/selection step for this diagnostic; not prospectively registered and not independent confirmation','deployment':'separate full-cohort training-only selection and fit; no held-out performance claimed for deployment model'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');pred=np.zeros_like(y);score=np.zeros(y.shape,float);selection=[];supports=[];all_inner=[]
    def fit_one(tr,te,name,seedfold):
        inner=inner_folds(y[tr],seedfold);ranking=[];allscore=[]
        for j,key in enumerate(keys):
            a=banks[key];inside=np.zeros((len(tr),y.shape[1]))
            for it,iv in inner:
                if set(groups[tr[it]])&set(groups[tr[iv]]):raise ValueError('Inner configuration leakage')
                inside[iv]=binary_scores(fit_binary(a[tr[it]],y[tr[it]],'independent_trees'),a[tr[iv]])
            q,ts,decoder=threshold_fit(y[tr],inside);ranking.append({'candidate':key,'inner_micro_f1':q,'thresholds':ts.tolist(),'decoder':decoder});allscore.append(inside)
            if (j+1)%10==0:print(name,'candidate',j+1,'/',len(keys),flush=True)
        win=max(ranking,key=lambda row:row['inner_micro_f1']);key=win['candidate'];depth,feature=key.split('::');fit=fit_binary(banks[key][tr],y[tr],'independent_trees');ss=binary_scores(fit,banks[key][te]) if len(te) else np.empty((0,y.shape[1]));threshold=np.array(win['thresholds'])
        model={'format':'co2rr-multilabel-v1','fit':fit,'energy':e,'species':list(SPECIES),'depth':depth,'feature':feature,'thresholds':threshold,'fingerprint':implementation_fingerprint(),'provenance':{'recipe':'updated-data-v3','training_indices':tr.tolist(),'heldout_indices':te.tolist(),'training_configuration_groups':groups[tr].tolist(),'selection':'inner training only','cohort_sha256':protocol['cohort_sha256']},'normalization':protocol['normalization'],'prediction_function':'tools.spectral_multilabel.predict_records'}
        p=out/(name+'.joblib');joblib.dump(model,p,compress=3);loaded=joblib.load(p);reload=binary_scores(loaded['fit'],banks[key][te]) if len(te) else ss
        if not np.array_equal(reload,ss):raise RuntimeError('Saved-model replay mismatch')
        np.savez_compressed(out/(name+'_inner.npz'),scores=np.array(allscore),truth=y[tr],training_indices=tr,candidates=keys)
        (out/(name+'_ranking.json')).write_text(json.dumps(ranking,indent=2)+'\n');(out/(name+'_splits.json')).write_text(json.dumps([{'training_indices':tr[it].tolist(),'validation_indices':tr[iv].tolist(),'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr[it]].sum(axis=0)==0)]} for it,iv in inner],indent=2)+'\n')
        report={'partition':name,'training_count':len(tr),'test_count':len(te),'candidate':key,'inner_micro_f1':win['inner_micro_f1'],'thresholds':win['thresholds'],'decoder':win['decoder'],'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)],'model_sha256':digest(p),'scores_reloaded_identically':True}
        print('Completed',name,key,flush=True);return ss,threshold,report
    for k in sorted(set(fold)):
        tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k)
        if set(groups[tr])&set(groups[te]):raise ValueError('Outer configuration leakage')
        ss,ts,row=fit_one(tr,te,f'outer_fold_{k}',int(k));score[te]=ss;pred[te]=ss>=ts;row.update(f1_metrics(y[te],pred[te]));selection.append(row)
        supports += [{'fold':int(k),'species':s,'training_positive':int(y[tr,j].sum()),'test_positive':int(y[te,j].sum())} for j,s in enumerate(SPECIES)]
        (out/'selections.json').write_text(json.dumps(selection,indent=2)+'\n')
    np.savez_compressed(out/'oof.npz',truth=y,prediction=pred,score=score,fold=fold,groups=groups,configuration=configs,species=SPECIES)
    pp,rr,ff,su=precision_recall_fscore_support(y,pred,zero_division=0);pd.DataFrame({'species':SPECIES,'precision':pp,'recall':rr,'f1':ff,'support':su}).to_csv(out/'per_species.csv',index=False);pd.DataFrame(single_mixed(y,pred,'updated_primary_refit')).to_csv(out/'single_mixed_recall.csv',index=False);pd.DataFrame(supports).to_csv(out/'fold_support.csv',index=False)
    manifest=pd.read_csv(dataset/'cv_manifest.csv');new=manifest.prior_configuration.isna().to_numpy();old=~new;results={'all_updated_configurations':f1_metrics(y,pred),'matched_old_subset':f1_metrics(y[old],pred[old]),'added_or_recovered_subset_in_updated_CV':f1_metrics(y[new],pred[new]),'old_subset_count':int(old.sum()),'added_or_recovered_count':int(new.sum())}
    # Explicit diagnostic before allowing new configurations into a training partition.
    ss,ts,row=fit_one(np.flatnonzero(old),np.flatnonzero(new),'new_configuration_holdout',5);hp=ss>=ts;results['new_configuration_holdout']=f1_metrics(y[new],hp);selection.append(row);np.savez_compressed(out/'new_configuration_holdout.npz',truth=y[new],prediction=hp,score=ss,indices=np.flatnonzero(new),configuration=configs[new],species=SPECIES)
    _,_,row=fit_one(np.arange(n),np.array([],int),'deployment_full_cohort',6);selection.append(row)
    (out/'selections.json').write_text(json.dumps(selection,indent=2)+'\n');results.update(configurations=n,layer_spectra=4*n,model_count=len(selection),all_saved_model_replays_identical=True,interpretation='new versioned fit; old 87-config/348-spectrum archived benchmark unchanged; existing folds examined repeatedly; no prospective validation or catalytic mechanism claim')
    (out/'metrics.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(results,indent=2),flush=True);return results


def run(data_root,output_dir,split_manifest=None):
    import tarfile,shutil
    repo=Path(__file__).resolve().parents[1];out=Path(output_dir)
    if out.exists() and any(out.iterdir()):raise ValueError('Use a new empty versioned output directory')
    out.mkdir(parents=True,exist_ok=True);dataset=out/'dataset';root=Path(data_root)
    if (root/'cohort.npz').exists():
        shutil.copytree(root,dataset)
        expected=json.loads((dataset/'snapshot_hashes.json').read_text())
        for p,h in expected.items():
            if digest(dataset/p)!=h:raise ValueError('Snapshot hash mismatch: '+p)
    else:
        audit_updated(root,dataset,repo)
        historical=repo/'study/retraining_20261022/historical_lineage.json'
        if historical.exists():shutil.copyfile(historical,dataset/'historical_lineage.json')
        with tarfile.open(dataset/'source_evidence.tar','w') as tar:
            for p in sorted((dataset/'source_objects').iterdir()):
                info=tar.gettarinfo(str(p),arcname='source_objects/'+p.name);info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
                with p.open('rb') as f:tar.addfile(info,f)
        shutil.rmtree(dataset/'source_objects')
        hashes={str(p.relative_to(dataset)):digest(p) for p in dataset.rglob('*') if p.is_file()};(dataset/'snapshot_hashes.json').write_text(json.dumps(hashes,indent=2)+'\n')
    metrics=fit_snapshot(dataset,out/'models')
    trace={'action':'updated_data_v3','raw_sources_mutated':False,'new_HPC_jobs':False,'portal_writes':False,'source_sha256':digest(__file__),'snapshot_manifest_sha256':digest(dataset/'snapshot_hashes.json'),'metrics':metrics,'output_dir':out.name}
    (out/'invocation_trace.json').write_text(json.dumps(trace,indent=2)+'\n');return {'status':'success','recipe':'updated_data_v3','output_dir':str(out),'metrics':metrics,'trace':str(out/'invocation_trace.json')}


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--data-root',required=True);p.add_argument('--output-dir',required=True);a=p.parse_args();run(a.data_root,a.output_dir)
