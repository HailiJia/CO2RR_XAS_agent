"""Recompute archived OOF metrics and portable fitted-model replay."""
import json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support,multilabel_confusion_matrix
from .spectral_multilabel import (SPECIES,feature_for_records,fit_binary,binary_scores,
    f1_metrics,save_model,predict_records,implementation_fingerprint)
from .study_cohort import load_cohort,sha256
import joblib

EXPECTED={'independent_selected':(.7790697674418605,.6382765882765882,.6896551724137931),
          'provisional_best':(.7978142076502732,.6597402597402597,.7011494252873564)}

def replay_oof(analysis_dir,output_dir,split_manifest=None):
    base=Path(analysis_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(split_manifest or base/'cv_manifest.csv')
    archived_manifest=pd.read_csv(base/'cv_manifest.csv');keys=['configuration','record_sha256','structure_hash','labels','fold']
    if not manifest[keys].equals(archived_manifest[keys]):raise ValueError('Replay manifest differs from archived cohort/order; create a new version and comparison')
    reference=np.load(base/'oof_independent_selected_micro.npz')
    truth=reference['truth'];fold=reference['fold'];species=reference['species'].astype(str)
    if truth.shape!=(len(manifest),len(species)) or not np.array_equal(species,SPECIES) or not np.array_equal(fold,manifest.fold):raise ValueError('OOF species/order/fold mismatch')
    labels=np.array([[int(s in row.split(';')) for s in species] for row in manifest.labels])
    if not np.array_equal(labels,truth):raise ValueError('Truth differs from frozen labels')
    if sum(truth.sum(axis=1)>1)!=9 or truth.sum()!=99:raise ValueError('Unexpected cohort/mixture composition')
    rows=[];per=[];mixtures=[];confusion={};baseline_preds={}
    reports=pd.read_csv(base/'nested_performance.csv')
    for p in sorted(base.glob('oof_*.npz')):
        if p.name.startswith('oof_fold_'):continue
        z=np.load(p);pred=z['prediction']
        if not np.array_equal(z['truth'],truth) or not np.array_equal(z['fold'],fold) or not np.array_equal(z['species'].astype(str),species):raise ValueError('OOF alignment error: '+p.name)
        key=p.stem[4:];metrics=f1_metrics(truth,pred);rows.append(dict(workflow=key,**metrics))
        if key in ['independent_selected_micro','provisional_best']:
            expected=EXPECTED['independent_selected' if key.startswith('independent') else key]
            if not np.allclose(list(metrics.values()),expected,atol=1e-10,rtol=0):raise ValueError('Published metric mismatch')
        if key!='provisional_best':
            selector,objective=key.rsplit('_',1);r=reports[(reports.selector==selector)&(reports.objective==objective)].iloc[0]
            if not np.allclose([metrics['micro_f1'],metrics['macro_f1']],[r.micro_f1,r.macro_f1],atol=1e-12,rtol=0):raise ValueError('Reported aggregate CSV mismatch')
        precision,recall,f1,support=precision_recall_fscore_support(truth,pred,average=None,zero_division=0)
        for j,s in enumerate(species):per.append(dict(workflow=key,species=s,precision=precision[j],recall=recall[j],f1=f1[j],support=int(support[j])))
        confusion[key]=multilabel_confusion_matrix(truth,pred).tolist()
        for j,s in enumerate(species):
            for kind,mask in [('single',(truth[:,j]==1)&(truth.sum(axis=1)==1)),('mixed',(truth[:,j]==1)&(truth.sum(axis=1)>1))]:
                n=int(mask.sum());detected=int(pred[mask,j].sum());mixtures.append(dict(workflow=key,species=s,kind=kind,support=n,detected=detected,recall=detected/n if n else None))
        if key=='provisional_best':
            csv=pd.read_csv(base/'provisional_best_per_species.csv')
            if not np.allclose(csv[['precision','recall','f1']],np.column_stack([precision,recall,f1])) or not np.array_equal(csv.support,support):raise ValueError('Per-species CSV mismatch')
    # Reconcile every archived per-species and confusion CSV row directly.
    per_frame=pd.DataFrame(per)
    for row in pd.read_csv(base/'per_species.csv').itertuples():
        key=f'{row.selector}_{row.objective}'
        found=per_frame[(per_frame.workflow==key)&(per_frame.species==row.species)].iloc[0]
        if not np.allclose([found.precision,found.recall,found.f1,found.support],[row.precision,row.recall,row.f1,row.support],atol=1e-12,rtol=0):raise ValueError('Per-species CSV discrepancy')
    for row in pd.read_csv(base/'confusion_matrices.csv').itertuples():
        matrix=np.array(confusion[f'{row.selector}_{row.objective}'])[list(species).index(row.species)]
        if not np.array_equal(matrix.ravel(),[row.TN,row.FP,row.FN,row.TP]):raise ValueError('Confusion CSV discrepancy')
    for row in pd.read_csv(base/'pure_mixed_recall.csv').itertuples():
        found=next(r for r in mixtures if r['workflow']=='provisional_best' and r['species']==row.species and r['kind']==('single' if row.kind=='single-species' else 'mixed'))
        if found['support']!=row.support or found['detected']!=row.detected:raise ValueError('Pure/mixed CSV discrepancy')
    for baseline in ['training_majority_binary_constant','training_most_common_identity_set']:
        pred=np.zeros_like(truth)
        for f in np.unique(fold):
            tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
            if baseline.startswith('training_majority'):constant=(truth[tr].mean(axis=0)>.5).astype(int)
            else:
                unique,counts=np.unique(truth[tr],axis=0,return_counts=True);constant=unique[np.argmax(counts)]
            pred[te]=constant
        rows.append(dict(workflow=baseline,**f1_metrics(truth,pred)));baseline_preds[baseline]=pred
    missing=[]
    for f in np.unique(fold):
        for j,s in enumerate(species):missing.append(dict(fold=int(f),species=s,training_positives=int(truth[fold!=f,j].sum()),test_positives=int(truth[fold==f,j].sum())))
    pd.DataFrame(rows).to_csv(out/'recomputed_metrics.csv',index=False)
    pd.DataFrame(per).to_csv(out/'recomputed_per_species.csv',index=False)
    pd.DataFrame(mixtures).to_csv(out/'recomputed_pure_mixed_recall.csv',index=False)
    pd.DataFrame(missing).to_csv(out/'fold_positive_support.csv',index=False)
    (out/'recomputed_confusion_matrices.json').write_text(json.dumps(dict(species=species.tolist(),matrices=confusion),indent=2)+'\n')
    np.savez_compressed(out/'baseline_oof.npz',truth=truth,fold=fold,species=species,**baseline_preds)
    (out/'replay_provenance.json').write_text(json.dumps(dict(archived_result_preserved=True,source_files={p.name:sha256(p) for p in base.glob('oof_*.npz')},fingerprint=implementation_fingerprint(),uncertainty='Existing bootstrap intervals condition on fitted OOF predictions; not retraining confidence intervals',confirmation='Repeatedly examined cohort; reserve genuinely new configurations for confirmation'),indent=2)+'\n')
    return pd.DataFrame(rows)

def replay_models(data_root,analysis_dir,output_dir,split_manifest=None):
    base=Path(analysis_dir);out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    cohort=load_cohort(data_root,split_manifest or base/'cv_manifest.csv')
    choices=pd.read_csv(base/'nested_selections.csv');choices=choices[(choices.selector=='independent_selected')&(choices.objective=='micro')]
    reference=np.load(base/'oof_independent_selected_micro.npz');fold=cohort['split'].fold.to_numpy();reports=[];predicted=np.zeros_like(cohort['Y'])
    for choice in choices.itertuples():
        f=int(choice.fold);tr=np.flatnonzero(fold!=f);te=np.flatnonzero(fold==f)
        X=feature_for_records(cohort['records'],cohort['populations'],cohort['energy'],choice.depth,choice.feature)
        fitted=fit_binary(X[tr],cohort['Y'][tr],choice.model);thresholds=np.array(json.loads(choice.thresholds));path=out/f'primary_fold_{f}.joblib'
        provenance=dict(outer_fold=f,train_record_hashes=cohort['split'].iloc[tr].record_sha256.tolist(),test_record_hashes=cohort['split'].iloc[te].record_sha256.tolist(),train_geometry_hashes=cohort['split'].iloc[tr].structure_hash.tolist(),split_sha256=sha256(split_manifest or base/'cv_manifest.csv'),selection='archived inner-selected primary workflow, replay only; not new selection',represented_training_populations=[cohort['populations'][i] for i in tr])
        save_model(path,fitted,cohort['energy'],SPECIES,choice.depth,choice.feature,thresholds,provenance)
        loaded=joblib.load(path);p,s=predict_records(loaded,[cohort['records'][i] for i in te],[cohort['populations'][i] for i in te])
        before=binary_scores(fitted,X[te]);assert np.array_equal(s,before)
        if not np.array_equal(p,reference['prediction'][te]):raise ValueError('Primary OOF prediction replay differs in fold '+str(f))
        predicted[te]=p;reports.append(dict(fold=f,heldout_configurations=len(te),export_import_scores_identical=True,archived_predictions_identical=True,missing_positive_training=[SPECIES[j] for j in fitted['no_positive_training']],model_sha256=sha256(path)))
        print('Replayed fitted model fold',f,flush=True)
    report=dict(folds=reports,metrics=f1_metrics(cohort['Y'],predicted),fingerprint=implementation_fingerprint())
    (out/'saved_model_replay.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
