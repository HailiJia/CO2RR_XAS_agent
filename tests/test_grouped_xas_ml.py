import io
from copy import deepcopy

import joblib
import numpy as np
import pytest

from tools.ml_splits import grouped_holdout, groups_for_rows
from tools.xas_ml_utils import auto_advisor, export_model, run_auto_advisor, train_manual, train_manual_rows


def rows():
    x = np.linspace(0, 10, 24)
    return [{'record_id': f'record_{i}_{j}', 'series_id': 'xas', 'n_points': len(x), 'x': x,
             'y': np.exp(-((x - (3 if i % 2 else 6)) / 1.2)**2) + .01*i + .001*j,
             'record': {'sample': {'configuration_id': f'cfg_{i}', 'parent_structure_id': 'one_clean_parent',
                                   'ml_labels': {'identity': 'CO' if i % 2 else 'OH', 'coverage': i / 10}}}}
            for i in range(8) for j in range(3)]


def test_manual_fit_keeps_all_replicates_together_and_fits_scaler_on_train_only():
    data = rows()
    result = train_manual_rows(data, 'sample.ml_labels.identity', 'classification', 'Logistic regression', normalization='none', n_grid=24)
    manifest = result['split_manifest']
    assert set(manifest['train_groups']).isdisjoint(manifest['test_groups'])
    assert len(manifest['test_row_ids']) == 6
    fitted = result['model']
    training = [data[i] for i in result['train_indices']]
    expected = fitted['features'].transform(training).mean(axis=0)
    np.testing.assert_allclose(fitted['scale'].mean_, expected)
    predictions = fitted.predict(data)
    bundle = joblib.load(io.BytesIO(export_model(result)))
    np.testing.assert_array_equal(bundle['pipeline'].predict(data), predictions)
    assert bundle['split_manifest'] == manifest
    assert bundle['target'] == 'sample.ml_labels.identity'
    assert 'test_balanced_accuracy' in bundle['metrics']


def test_advisor_does_not_use_holdout_features_or_labels_for_ranking():
    data = rows()
    configs = [{'feature_set':['raw'], 'normalization':'none', 'model':name} for name in ['Dummy baseline', 'Ridge']]
    first, _ = auto_advisor(data, 'sample.ml_labels.coverage', 'regression', 24, 5, candidate_configs=configs)
    assert first
    modified = deepcopy(data)
    for i in first[0]['split_manifest']['test_indices']:
        modified[i]['y'] = np.full(24, 1e6)
        modified[i]['record']['sample']['ml_labels']['coverage'] = 1e6
    second, _ = auto_advisor(modified, 'sample.ml_labels.coverage', 'regression', 24, 5, candidate_configs=configs,
                             split_manifest=first[0]['split_manifest'])
    assert [r['model'] for r in first] == [r['model'] for r in second]
    assert [r['validation_RMSE'] for r in first] == [r['validation_RMSE'] for r in second]
    for recommendation in second:
        assert recommendation['test_evaluated'] is False
        held = set(recommendation['split_manifest']['test_groups'])
        for fold in recommendation['cv_folds']:
            assert set(fold['train_groups']).isdisjoint(fold['validation_groups'])
            assert held.isdisjoint(fold['train_groups'] + fold['validation_groups'])
    run = run_auto_advisor(data, 'sample.ml_labels.coverage', 'regression', 24, 5, candidate_configs=configs)
    assert 'test_RMSE' in run['result']['metrics']
    assert run['result']['split_manifest']['split_id'] == first[0]['split_manifest']['split_id']


def test_small_or_unlinked_dataset_is_never_reported_as_validated():
    with pytest.raises(ValueError, match='at least three'):
        grouped_holdout(['same_config'] * 20)
    with pytest.raises(ValueError, match='Supply configuration groups'):
        train_manual(np.ones((5,2)), np.array(['a','a','b','b','b']), 'classification', 'Dummy baseline', False, 2)
    with pytest.raises(ValueError, match='configuration_id'):
        groups_for_rows([{'record': {'record_id':'independent_backend_record', 'record_domain':'simulation'}}])
    groups, sources = groups_for_rows(rows())
    assert len(set(groups)) == 8
    assert sources == ['sample.configuration_id']


def test_split_manifest_rejects_overlap_and_fitted_grid_rejects_extrapolation():
    with pytest.raises(ValueError, match='leaks groups'):
        grouped_holdout(['a','b','c'], manifest={'train_groups':['a','b'], 'test_groups':['b','c']})
    result = train_manual_rows(rows(), 'sample.ml_labels.coverage', 'regression', 'Ridge')
    short = deepcopy(rows()[:1]); short[0]['x'] = np.linspace(1, 9, 24)
    with pytest.raises(ValueError, match='fitted energy grid'):
        result['model'].predict(short)


def test_identical_relaxed_geometries_from_different_initial_configs_are_grouped():
    data = rows()
    for row in data:
        row['record']['sample']['structure_descriptors'] = {'geometry_hash': 'same_relaxed_geometry' if row['record']['sample']['configuration_id'] in {'cfg_0', 'cfg_1'} else row['record']['sample']['configuration_id']}
    groups, sources = groups_for_rows(data)
    assert groups[0] == groups[3]
    assert len(set(groups)) == 7
    assert 'geometry_hash_deduplication' in sources
