"""Predeclared leave-local-motif-out extension; structural data define groups only."""
from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from sklearn.model_selection import GroupKFold
from .study_cohort import load_cohort
from .spectral_multilabel import SPECIES,spectrum_layers
from .scientific_extensions import fit_recipe,scores,thresholds,metrics,candidates,single_mixed


def run(data_root,output_dir,split_manifest=None):
    repo=Path(__file__).resolve().parents[1];out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('New empty output directory required')
    audit=json.loads((repo/'study/extensions_v2/chemical_audit/summary.json').read_text())
    if audit['demonstrated_retained_label_mismatches'] or audit['ambiguous_retained']:raise ValueError('Resolve chemical labels first')
    cohort=load_cohort(data_root,split_manifest or repo/'study/manuscript_package/figure_data/cv_manifest.csv');split=cohort['split'];y=cohort['Y'];groups=cohort['groups'];e=cohort['energy'];raw=np.array([spectrum_layers(r)[5][:,1] for r in cohort['records']])
    inventory=repo/'study/extensions_v2/ml/generalization_inventory.csv';domain=pd.read_csv(inventory).set_index('configuration').loc[split.configuration];motifs=domain.motif.to_numpy()
    protocol={'version':'motif-holdout-v2','group_definition':'initial top-metal-layer three-row Cu/Au composition, nearest fractional-y row; cyclic row windows and 0.6 A clustering in tools.study_design; not a relaxed adsorption binding-site label',
        'outer':'leave each observed local motif out completely; separate from frozen configuration folds','inner':'3 grouped configuration folds within outer training for thresholds only','recipe':'predeclared DCT16 independent Extra Trees, 32 trees, min leaf2, balanced class weights, threshold grid inherited from scientific_extensions; no outer-score feature/model selection',
        'predictors':'surface-layer spectrum only, same 8976-9027 eV crop as extensions; no motif/count/name/graph predictors','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'inventory_sha256':hashlib.sha256(inventory.read_bytes()).hexdigest(),
        'sparse_groups':'CuAu2 has one held-out configuration; report count and avoid general population claims','independence':'same repeatedly examined cohort, not prospective confirmation or ancestral-parent transfer'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');prediction=np.zeros_like(y);allscore=np.zeros(y.shape,float);rows=[];splits=[];support=[];cfg=candidates('independent_trees')[0]
    for motif in sorted(set(motifs)):
        tr=np.flatnonzero(motifs!=motif);te=np.flatnonzero(motifs==motif);assert not set(motifs[tr])&set(motifs[te]);inner=np.zeros((len(tr),y.shape[1]));inner_rows=[]
        for k,(it,iv) in enumerate(GroupKFold(3).split(raw[tr],y[tr],groups[tr])):
            m=fit_recipe(raw[tr[it]],y[tr[it]],groups[tr[it]],e,'DCT16','independent_trees',cfg);inner[iv]=scores(m,raw[tr[iv]])
            inner_rows.append({'fold':k,'training':tr[it].tolist(),'validation':tr[iv].tolist(),'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr[it]].sum(axis=0)==0)]})
        ts=thresholds(y[tr],inner);model=fit_recipe(raw[tr],y[tr],groups[tr],e,'DCT16','independent_trees',cfg);ss=scores(model,raw[te]);prediction[te]=ss>=ts;allscore[te]=ss
        model.update(thresholds=ts,training_indices=tr,heldout_indices=te,heldout_motif=motif);path=out/f'{motif}.joblib';joblib.dump(model,path,compress=3);assert np.array_equal(scores(joblib.load(path),raw[te]),ss)
        rows.append({'heldout_motif':motif,'training_configurations':len(tr),'test_configurations':len(te),**metrics(y[te],prediction[te]),'model_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'save_reload_identical':True})
        support += [{'heldout_motif':motif,'species':s,'training_positives':int(y[tr,j].sum()),'test_positives':int(y[te,j].sum())} for j,s in enumerate(SPECIES)]
        splits.append({'heldout_motif':motif,'training':tr.tolist(),'test':te.tolist(),'thresholds':ts.tolist(),'missing_positive_training':[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)],'inner_splits':inner_rows})
    rows.append({'heldout_motif':'pooled','training_configurations':None,'test_configurations':len(y),**metrics(y,prediction),'save_reload_identical':True})
    pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(support).to_csv(out/'support.csv',index=False);pd.DataFrame(single_mixed(y,prediction,'leave_motif_out')).to_csv(out/'single_mixed_recall.csv',index=False);(out/'splits.json').write_text(json.dumps(splits,indent=2)+'\n')
    np.savez_compressed(out/'oof.npz',truth=y,prediction=prediction,score=allscore,motif=motifs.astype(str),configuration=split.configuration.to_numpy(dtype=str),species=SPECIES)
    return {'status':'success','recipe':'motif_holdout_v2','output_dir':str(out),'metrics':rows,'trace':str(out/'protocol.json')}
