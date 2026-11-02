"""Full-cohort model tuning: all permitted views, group weights, nested selection."""
from pathlib import Path
import json
import numpy as np
import joblib
from sklearn.ensemble import ExtraTreesClassifier
from .coverage_study import dump, sha
from .differential_identity import VERSION, shape_floor
from .identity_comparison import transform, score_model, SEEDS
from .spectral_multilabel import SPECIES, encode_labels

# Declared before fitting; selection uses inner predictions only.
CANDIDATES = tuple((feature, leaf, 64) for feature in ['Combined', 'DCT16']
                   for leaf in [1, 2, 4]) + (('Combined', 1, 256),)
THRESHOLDS = (.3, .4, .5, .6)


def group_weights(families):
    families = np.asarray(families)
    return np.array([1. / np.sum(families == value) for value in families])


def balanced_weights(target, weights):
    target = np.asarray(target, int)
    weights = np.asarray(weights, float)
    mass = np.bincount(target, weights=weights, minlength=2)
    if np.any(mass == 0):
        return weights.copy()
    return weights * (weights.sum() / (2 * mass))[target]


def fit_weighted(X, y, families, recipe, seed):
    feature, leaf, trees = recipe
    weights = group_weights(families)
    estimators = []
    for target in y.T:
        if len(np.unique(target)) < 2:
            estimators.append(int(target[0]))
        else:
            model = ExtraTreesClassifier(n_estimators=trees,
                min_samples_leaf=leaf, max_features='sqrt',
                random_state=seed, n_jobs=1)
            model.fit(X, target, sample_weight=balanced_weights(target, weights))
            estimators.append(model)
    return dict(models=estimators, scaler=None, kind='independent_trees',
                n_outputs=y.shape[1])


def weighted_micro(y, prediction, weights):
    numerator = 2 * np.sum(y * prediction * weights[:, None])
    denominator = np.sum((y + prediction) * weights[:, None])
    return float(numerator / denominator) if denominator else 0.


def choose_threshold(y, scores, families):
    weights = group_weights(families)
    quality = [weighted_micro(y, scores >= t, weights) - .02 * (t - .5)**2
               for t in THRESHOLDS]
    winner = int(np.argmax(quality))
    return THRESHOLDS[winner], quality[winner]


def expand_inner(outer_train, source_splits, families):
    pairs = []
    for split in source_splits:
        val_groups = set(families[split['validation']])
        v = np.array([i for i in outer_train if families[i] in val_groups], int)
        a = np.array([i for i in outer_train if families[i] not in val_groups], int)
        assert not set(families[a]) & set(families[v])
        pairs.append((a, v))
    assert sorted(np.concatenate([v for a, v in pairs]).tolist()) == outer_train.tolist()
    return pairs


def train(data_root, output_dir):
    B, O = Path(data_root), Path(output_dir)
    O.mkdir(parents=True, exist_ok=True)
    if any(O.iterdir()):
        raise ValueError('Use a new empty output directory')
    rows = json.load(open(B / 'training_manifest.json'))
    z = np.load(B / 'training_arrays.npz')
    fm = json.load(open(B / 'family_map.json'))
    assert fm['training_manifest_sha256'] == sha(B / 'training_manifest.json')
    previous = json.load(open(B / 'ml/protocol.json'))
    assert previous['training_arrays_sha256'] == sha(B / 'training_arrays.npz')
    families = np.array([fm['families'][r['configuration']] for r in rows])
    y = encode_labels([r['identity_labels'] for r in rows])
    folds = np.array([r['fold'] for r in rows])
    assert len(rows) == 141 and len(set(families)) == 135
    splits = json.load(open(B / 'ml/splits.json'))
    source = {s['fold']: s for s in splits if s['kind'] == 'trees'
              and s['seed'] == SEEDS[0] and s['protocol'] == 'updated_full_cohort'}
    protocol = dict(version='full-cohort-model-tuning-v1', candidates=CANDIDATES,
        seeds=SEEDS, primary='seven-candidate nested selection; no outer-score winner',
        controls=['all-views Combined/leaf1/64', 'all-views Combined/leaf1/256',
                  'previous canonical-view Combined/leaf1/64, read-only'],
        data='ALL141 supported records queried exactly once per seed; no abstention or difficulty filter',
        training='ALL records outside held-out near groups; no canonical-view pruning',
        grouping='135 conservative near groups, inherited outer folds and inner group assignments',
        weights='each training geometry group has total base weight1; binary class masses balanced within training',
        leaf='minimum leaf size counts record views; weighted impurity/class balance uses group weights',
        threshold='common .3/.4/.5/.6, inner group micro F1 minus .02(t-.5)^2; fixed.5 separately',
        representations='fixed population-weighted top-two signed shape; Combined or DCT16; no new depth/energy selection',
        normalization='training-only simulation floor; every query retained; no amplitude/quality selection',
        uncertainty='paired group bootstrap conditional on saved predictions; seed/refit variability separate',
        experimental_validation=False, manuscript_changes=False,
        source_sha256=sha(Path(__file__)),
        input_hashes={name: sha(B/name) for name in ['training_arrays.npz',
                         'training_manifest.json','family_map.json','ml/splits.json']})
    dump(O / 'protocol.json', protocol)  # frozen before the first fit
    (O / 'models').mkdir()
    audit, selections, replays = [], [], []
    for seed in SEEDS:
        scores = {name: np.zeros_like(y, float) for name in ['all_views64', 'all_views256', 'nested']}
        predictions = {(name, stage): np.zeros_like(y) for name in scores
                       for stage in ['fixed0p5', 'common']}
        for f in range(5):
            te = np.flatnonzero(folds == f)
            tr = np.flatnonzero(~np.isin(families, families[te]))
            assert set(families[tr]) == set(families[source[f]['train']])
            inner = expand_inner(tr, source[f]['inner'], families)
            inside = np.zeros((len(CANDIDATES), len(tr), y.shape[1]))
            location = {int(i): j for j, i in enumerate(tr)}
            for a, v in inner:
                floor = shape_floor(z['delta'][a])
                for k, recipe in enumerate(CANDIDATES):
                    X = transform(z['delta'][a], z['energy'], 'signed_shape', recipe[0], floor)
                    V = transform(z['delta'][v], z['energy'], 'signed_shape', recipe[0], floor)
                    model = fit_weighted(X, y[a], families[a], recipe, seed)
                    inside[k, [location[int(i)] for i in v]] = score_model(model, V)
            thresholds, quality = zip(*[choose_threshold(y[tr], s, families[tr]) for s in inside])
            winner = int(np.argmax(quality))  # ties follow predeclared candidate order
            chosen = {'all_views64': 0, 'all_views256': 6, 'nested': winner}
            np.savez_compressed(O / f'inner_{seed}_fold{f}.npz', scores=inside,
                truth=y[tr], indices=tr, weights=group_weights(families[tr]),
                thresholds=thresholds, quality=quality)
            floor = shape_floor(z['delta'][tr])
            cache = {}
            for k in sorted(set(chosen.values())):
                recipe = CANDIDATES[k]
                X = transform(z['delta'][tr], z['energy'], 'signed_shape', recipe[0], floor)
                V = transform(z['delta'][te], z['energy'], 'signed_shape', recipe[0], floor)
                model = fit_weighted(X, y[tr], families[tr], recipe, seed)
                score = score_model(model, V)
                bundle = dict(model=model, input_contract=VERSION,
                    representation='signed_shape', feature=recipe[0], recipe=recipe,
                    floor=floor, energy=z['energy'], species=SPECIES,
                    thresholds=np.full(len(SPECIES), thresholds[k]),
                    domains=sorted({r['domain'] for r in rows}),
                    train_indices=tr, test_indices=te,
                    train_families=families[tr].tolist(), test_families=families[te].tolist(),
                    group_weights=group_weights(families[tr]),
                    reference_ids=sorted({rows[i]['reference_id'] for i in tr}),
                    reference_record_hashes=sorted({rows[i]['reference_record_sha256'] for i in tr}),
                    source_sha256=sha(Path(__file__)), seed=seed)
                path = O / 'models' / f'seed{seed}_fold{f}_candidate{k}.joblib'
                joblib.dump(bundle, path, compress=3)
                np.testing.assert_array_equal(score, score_model(joblib.load(path)['model'], V))
                replays.append(dict(path=str(path.relative_to(O)), sha256=sha(path), exact_reload=True))
                cache[k] = score
            for name, k in chosen.items():
                scores[name][te] = cache[k]
                predictions[name, 'fixed0p5'][te] = cache[k] >= .5
                predictions[name, 'common'][te] = cache[k] >= thresholds[k]
                selections.append(dict(seed=seed, fold=f, protocol=name, candidate=k,
                    recipe=CANDIDATES[k], threshold=thresholds[k], inner_quality=quality[k],
                    outer_model=f'models/seed{seed}_fold{f}_candidate{k}.joblib'))
            audit.append(dict(seed=seed, fold=f, train=tr.tolist(), test=te.tolist(),
                training_records=len(tr), training_groups=len(set(families[tr])),
                missing_positive=[SPECIES[j] for j in np.flatnonzero(y[tr].sum(axis=0)==0)],
                inner=[dict(train=a.tolist(), validation=v.tolist(),
                    missing_positive=[SPECIES[j] for j in np.flatnonzero(y[a].sum(axis=0)==0)]) for a,v in inner]))
            print('completed seed', seed, 'fold', f, 'selected', CANDIDATES[winner], flush=True)
        for name in scores:
            for stage in ['fixed0p5', 'common']:
                np.savez_compressed(O / f'{name}_{seed}_{stage}.npz',
                    indices=np.arange(len(rows)), truth=y, prediction=predictions[name, stage],
                    score=scores[name], fold=folds, families=families)
    dump(O/'splits.json', audit)
    dump(O/'selections.json', selections)
    dump(O/'model_replay.json', replays)
    return dict(status='success', output=str(O), records=len(rows), near_groups=len(set(families)),
                saved_models=len(replays), all_queries_retained=True)
