from pathlib import Path
import json,hashlib,tarfile
import numpy as np,pandas as pd,nbformat
from sklearn.metrics import f1_score,precision_recall_fscore_support
BASE=Path(__file__).resolve().parent;DATA=BASE.parent.parent
summary=json.loads((BASE/'run_summary.json').read_text());config=json.loads((BASE/'run_config.json').read_text())
inventory=pd.read_csv(BASE/'all_unique_structures.csv');folders=pd.read_csv(BASE/'all_structure_folders.csv');records=pd.read_csv(BASE/'record_audit.csv');manifest=pd.read_csv(BASE/'cv_manifest.csv')
assert len(inventory)==summary['unique_inventory']==155 and len(folders)==summary['source_locations']==172
assert len(records)==102 and len(manifest)==87 and manifest.structure_hash.nunique()==87
assert records.status.value_counts().to_dict()=={'usable':87,'duplicate':14,'excluded':1}
assert inventory.ml_status.value_counts().to_dict()=={'usable':87,'no completed record':67,'record excluded':1}
assert pd.read_csv(BASE/'figure1_combination_counts.csv',index_col=0).to_numpy().sum()==155
for r in manifest.itertuples():assert hashlib.sha256((DATA/r.record).read_bytes()).hexdigest()==r.record_sha256
assert config['manifest_hash']==hashlib.sha256((BASE/'cv_manifest.csv').read_bytes()).hexdigest()
old=pd.read_csv(BASE/'prior_fold_assignments.csv').set_index('configuration')
for r in inventory.itertuples():
    previous=[int(old.loc[c,'fold']) for c in json.loads(r.aliases) if c in old.index]
    if not previous or r.ml_status!='usable':continue
    candidates=manifest[manifest.configuration.isin(json.loads(r.aliases))]
    assert len(candidates)==1 and int(candidates.iloc[0].fold)==previous[0]
primary=np.load(BASE/'oof_independent_selected_micro.npz');Y=primary['truth'];fold=primary['fold'];species=primary['species']
metrics=pd.read_csv(BASE/'nested_performance.csv');assert len(metrics)==28
for row in metrics.itertuples():
    d=np.load(BASE/f'oof_{row.selector}_{row.objective}.npz');assert np.array_equal(d['truth'],Y) and np.array_equal(d['fold'],fold)
    assert abs(f1_score(Y,d['prediction'],average='micro',zero_division=0)-row.micro_f1)<1e-12
    assert abs(f1_score(Y,d['prediction'],average='macro',zero_division=0)-row.macro_f1)<1e-12
for f in range(5):
    train=np.where(fold!=f)[0];test=np.where(fold==f)[0]
    inner=pd.read_csv(BASE/f'inner_folds_{f}.csv');assert set(inner.manifest_row)==set(train)
    for p in (BASE/'cache').glob(f'{config["tag"]}_{f}_*.npz'):
        d=np.load(p);assert np.array_equal(d['train_indices'],train) and np.array_equal(d['test_indices'],test)
        assert np.isfinite(d['inner_score']).all() and np.isfinite(d['test_score']).all()
assert len(pd.read_csv(BASE/'feature_model_depth_performance.csv'))==179*2
chosen=np.load(BASE/'oof_provisional_best.npz');chosen_spec=json.loads((BASE/'provisional_best_model.json').read_text())
assert np.array_equal(chosen['truth'],Y) and np.array_equal(chosen['prediction'],chosen['score']>=chosen['threshold'])
assert abs(f1_score(Y,chosen['prediction'],average='micro',zero_division=0)-chosen_spec['micro_f1'])<1e-12
chosen_report=pd.read_csv(BASE/'provisional_best_per_species.csv')
p,r,f,s=precision_recall_fscore_support(Y,chosen['prediction'],average=None,zero_division=0)
assert np.allclose(chosen_report.f1,f) and np.array_equal(chosen_report.support,s)
# Check every correction against the actual pre-edit JSON backup.
corrections=json.loads((BASE/'record_corrections.json').read_text());assert len(corrections['changes'])==7
with tarfile.open(corrections['backup']) as tar:
    for row in corrections['changes']:
        before=json.load(tar.extractfile(row['record']));after=json.loads((DATA/row['record']).read_text())
        assert before['measurement']==after['measurement']
        assert after['sample']['composition']['adsorbate_counts']==row['after']['adsorbate_counts']
nb=nbformat.read(BASE/'adsorbate_identity_depth_updated.ipynb',4);nbformat.validate(nb)
code=images=0
for c in nb.cells:
    if c.cell_type=='code':
        code+=1;assert c.execution_count is not None and not any(o.output_type=='error' for o in c.outputs)
        images+=sum('image/png' in o.get('data',{}) for o in c.outputs);compile(c.source,'notebook cell','exec')
assert code==8 and images==8,(code,images)
for n in ['01_all_structures_by_adsorbate','02_training_testing_structure_counts','03_layer_sensitivity','04_adsorbate_identity_performance','05_feature_model_comparison']:
    for ext in ['png','pdf']:assert (BASE/'figures'/f'{n}.{ext}').stat().st_size>1000
print('Validated: 155 unique inventory structures; 87 configurations; 7 reversible record corrections; preserved prior folds; 358 feature/model/depth entries; 28 nested reports; 8 executed code cells; 8 embedded figures; five requested PNG/PDF figure pairs.')
