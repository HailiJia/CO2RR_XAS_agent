"""Updated-data provenance, fold and decoder invariants."""
from pathlib import Path
import ast,json
import numpy as np
import pandas as pd
import pytest
from tools.updated_data_training import threshold_fit,inner_folds
ROOT=Path(__file__).resolve().parents[1]


def test_updated_decoder_matches_frozen_primary_algorithm():
    source=ast.parse((ROOT/'CO2RR_ML/updated_dataset_20261020/analysis.py').read_text());functions=[n for n in source.body if isinstance(n,ast.FunctionDef) and n.name in ['f1_values','threshold_fit']];scope={'np':np,'THRESHOLDS':np.array([.15,.25,.35,.45,.5,.55,.65,.75])};exec(compile(ast.Module(body=functions,type_ignores=[]),'<frozen decoder>','exec'),scope)
    rng=np.random.default_rng(42)
    for n in [12,70,90]:
        y=(rng.random((n,11))<.2).astype(int);s=rng.random(y.shape);actual=threshold_fit(y,s);expected=scope['threshold_fit'](y,s,'micro',True)
        assert actual[0]==expected[0] and actual[2]==expected[2];np.testing.assert_array_equal(actual[1],expected[1])


def test_refit_retains_frozen_groups_labels_and_source_lineage():
    b=ROOT/'study/retraining_20261022/refit/dataset';d=np.load(b/'cohort.npz');cv=pd.read_csv(b/'cv_manifest.csv');old=pd.read_csv(ROOT/'study/manuscript_package/figure_data/cv_manifest.csv');summary=json.loads((b/'summary.json').read_text())
    assert summary['configurations']==90 and summary['prior_configurations']==87 and summary['layer_spectra']==360
    assert len(set(d['groups']))==90 and d['stack'].shape==(90,4,276)
    matched=cv[cv.prior_configuration.notna()].set_index('prior_configuration')
    assert set(matched.index)==set(old.configuration)
    for row in old.itertuples():assert matched.loc[row.configuration,'fold']==row.fold and matched.loc[row.configuration,'labels']==row.labels
    audit=json.loads((b/'record_audit.json').read_text());excluded=[r for r in audit if r['status']=='excluded'];assert len(excluded)==1 and 'convolution' in excluded[0]['reason']
    warnings=[r for r in audit if r.get('historical_lineage')];assert len(warnings)==1 and warnings[0]['root_XAS_correspondence']['status']=='geometry_changed'
    assert len(warnings[0]['historical_lineage']['convolved_output_sha256'])==4
    assert len(summary['new_or_recovered_configurations'])==3


def test_snapshot_folds_and_training_only_inner_partitions():
    d=np.load(ROOT/'study/retraining_20261022/refit/dataset/cohort.npz');y=d['truth'];groups=d['groups'];fold=d['fold']
    for k in range(5):
        tr=np.flatnonzero(fold!=k);te=np.flatnonzero(fold==k);assert not set(groups[tr])&set(groups[te])
        for it,iv in inner_folds(y[tr],k):assert not set(groups[tr[it]])&set(groups[tr[iv]]) and set(tr[it])|set(tr[iv])==set(tr)


def test_updated_data_natural_language_calls_new_versioned_route():
    from agent.co2rr_xas_agent import LocalIntentParser
    p=LocalIntentParser().parse('Audit updated data and retrain the adsorbate identity model, data-root=/tmp/export')
    assert p.action=='audited_training' and p.parameters['training_recipe']=='dataset_refresh_v1'
