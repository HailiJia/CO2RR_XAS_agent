"""Fit final models to ALL141 records; selection CV is training, not test evidence."""
from pathlib import Path
import json
import numpy as np, joblib
from sklearn.model_selection import GroupKFold
from .full_cohort_tuning import CANDIDATES, group_weights, fit_weighted, choose_threshold
from .identity_comparison import transform, score_model, SEEDS
from .differential_identity import VERSION, shape_floor
from .spectral_multilabel import SPECIES, encode_labels
from .coverage_study import dump, sha


def train(data_root, output_dir):
    B, O = Path(data_root), Path(output_dir)
    O.mkdir(parents=True, exist_ok=True)
    if any(O.iterdir()):
        raise ValueError('Use a new empty output directory')
    rows = json.load(open(B/'training_manifest.json'))
    z = np.load(B/'training_arrays.npz')
    fm = json.load(open(B/'family_map.json'))
    assert fm['training_manifest_sha256'] == sha(B/'training_manifest.json')
    families = np.array([fm['families'][r['configuration']] for r in rows])
    y = encode_labels([r['identity_labels'] for r in rows])
    ids = np.arange(len(rows))
    assert len(ids) == 141
    inner = list(GroupKFold(3).split(ids, groups=families))
    protocol = dict(scope='final full-data training AFTER evaluation; not held-out evidence',
        inputs=len(rows), near_groups=len(set(families)), all_training_views=True,
        candidates=CANDIDATES, seeds=SEEDS,
        selection='same locked candidates and training-only group-weighted common threshold',
        performance='no independent score is computed from these final models',
        source_sha256=sha(Path(__file__)),
        fitting_dependency_sha256=sha(Path(__file__).with_name('full_cohort_tuning.py')),
        input_sha256=sha(B/'training_arrays.npz'))
    dump(O/'protocol.json', protocol)
    dump(O/'selection_splits.json', [dict(train=a.tolist(), validation=v.tolist()) for a,v in inner])
    receipts = []
    for seed in SEEDS:
        scores = np.zeros((len(CANDIDATES), len(rows), len(SPECIES)))
        for a, v in inner:
            assert not set(families[a]) & set(families[v])
            floor = shape_floor(z['delta'][a])
            for k, recipe in enumerate(CANDIDATES):
                X = transform(z['delta'][a], z['energy'], 'signed_shape', recipe[0], floor)
                V = transform(z['delta'][v], z['energy'], 'signed_shape', recipe[0], floor)
                scores[k,v] = score_model(fit_weighted(X,y[a],families[a],recipe,seed),V)
        thresholds, quality = zip(*[choose_threshold(y,s,families) for s in scores])
        win = int(np.argmax(quality))
        recipe = CANDIDATES[win]
        floor = shape_floor(z['delta'])
        X = transform(z['delta'], z['energy'], 'signed_shape', recipe[0], floor)
        model = fit_weighted(X,y,families,recipe,seed)
        bundle = dict(model=model,input_contract=VERSION,representation='signed_shape',
            feature=recipe[0],recipe=recipe,floor=floor,energy=z['energy'],species=SPECIES,
            thresholds=np.full(len(SPECIES),thresholds[win]),domains=sorted({r['domain'] for r in rows}),
            train_indices=ids,test_indices=np.array([],int),
            train_families=families.tolist(),group_weights=group_weights(families),
            reference_ids=sorted({r['reference_id'] for r in rows}),
            reference_record_hashes=sorted({r['reference_record_sha256'] for r in rows}),
            seed=seed,scope='final full141 fit; no independent test score',
            source_sha256=sha(Path(__file__)))
        path=O/f'final_full141_seed{seed}.joblib'
        joblib.dump(bundle,path,compress=3)
        np.testing.assert_array_equal(score_model(model,X),score_model(joblib.load(path)['model'],X))
        np.savez_compressed(O/f'training_selection_seed{seed}.npz',score=scores,
                            truth=y,thresholds=thresholds,quality=quality)
        receipts.append(dict(seed=seed,recipe=recipe,threshold=thresholds[win],
            training_records=len(rows),training_groups=len(set(families)),path=path.name,
            sha256=sha(path),exact_reload=True))
        print('Final fit',seed,recipe,'all',len(rows),'records',flush=True)
    dump(O/'models.json',receipts)
    return dict(status='success',output=str(O),final_models=len(receipts),training_records=141,
                performance_claim='none; use separate full-cohort outer predictions')
