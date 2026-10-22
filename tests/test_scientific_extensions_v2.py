"""Checks scientific invariants without repeating the expensive nested fit."""
import importlib.util,json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools.controlled_comparisons import equivalent_candidates
from tools.scientific_extensions import transform_spectra,fit_target,probability,features
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reference_v2',ROOT/'study/reference_benchmark_v2/process.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)

def test_all_reference_arrays_and_metrics_share_exact_rule_grid():
    base=ROOT/'study/reference_benchmark_v2';d=pd.read_csv(base/'plotted_arrays.csv');m=pd.read_csv(base/'similarity_metrics.csv')
    for (material,method,view),q in d.groupby(['material','method','comparison_type']):
        np.testing.assert_allclose(q.energy_eV,ref.GRID,atol=1e-9)
        for key in ['calculation','experiment']:
            assert q[key].between(0,1).all()
            assert np.isclose(q[key].min(),0) and np.isclose(q[key].max(),1)
        row=m[(m.material==material)&(m.method_key==method)&(m.comparison_type==view)].iloc[0]
        for key,value in ref.similarity(q.calculation,q.experiment).items():assert np.isclose(row[key],value,atol=1e-12)

def test_identity_and_undefined_similarities():
    x=np.r_[np.zeros(10),np.linspace(0,1,30)]
    for key in ['cosine_similarity','pearson_r','spearman_rho']:assert np.isclose(ref.similarity(x,x)[key],1)
    assert ref.similarity(np.zeros(40),x)['cosine_similarity'] is None
    assert ref.similarity(np.ones(40),x)['pearson_r'] is None
    with pytest.raises(ValueError):ref.normalize(np.ones(40))
    with pytest.raises(ValueError):ref.normalize(np.r_[np.nan,x])

def test_fixed_window_rejects_partial_support_and_positive_kernel():
    with pytest.raises(ValueError):ref.sample((np.array([8990,9021]),np.ones(2)),0)
    x=np.linspace(8970,9030,301);y=np.exp(-((x-9000)/3)**2)
    assert (ref.gaussian(x,y,1)>=0).all()

def test_equivalent_duplicate_candidates_are_one_observation():
    a={'geometry_hash':'same','path':'a'};b={'geometry_hash':'same','path':'copy'}
    assert len(equivalent_candidates([a,b]))==1
    assert len(equivalent_candidates([a,{'geometry_hash':'different'}]))==2

def test_recovered_terrace_is_not_reported_as_unrun():
    d=json.loads((ROOT/'study/extensions_v2/controlled_comparisons/matched_quartet_v2.json').read_text())
    row=next(r for r in d['structures'] if r['requested_state']=='2CO' and r['region']=='terrace')
    assert row['relaxed_available'] and row['records']
    assert row['status']=='completed_but_incompatible_with_pinned_initial_version'

def test_synthetic_stress_never_extrapolates_and_constants_work():
    e=np.linspace(8974,9029,276);raw=np.exp(-((e-9000)/3)**2)[None,:]
    for case in ['shift:-2','shift:2','gaussian:1','noise:0.05','augmentation']:
        z,grid=transform_spectra(raw,e,case);assert np.isfinite(z).all() and grid.min()>=8976 and grid.max()<=9027
    with pytest.raises(ValueError):transform_spectra(raw[:,20:-20],e[20:-20],'shift:2')
    model=fit_target(np.arange(8)[:,None],np.ones(8,int),'xgboost',{})
    assert np.array_equal(probability(model,np.zeros((2,1))),[1,1])
    assert features(raw,e,'DCT8').shape==(1,8)


def test_main_figure_uses_the_metric_arrays_without_rescaling():
    d=pd.read_csv(ROOT/'study/reference_benchmark_v2/plotted_arrays.csv');display=pd.read_csv(ROOT/'study/manuscript_package/figure_data/Figure_3a_display.csv')
    for (material,method),q in d[d.comparison_type=='Cu_calibrated'].groupby(['material','method']):
        actual=display[(display.material==material)&(display.method==method)]
        np.testing.assert_allclose(actual.intensity_au,q.calculation,atol=1e-12)
        experiment=display[(display.material==material)&(display.method=='experiment')]
        np.testing.assert_allclose(experiment.intensity_au,q.experiment,atol=1e-12)


def test_new_natural_language_recipe_routes_to_shared_training():
    from agent.co2rr_xas_agent import LocalIntentParser
    parsed=LocalIntentParser().parse('Train the adsorbate identity model using scientific extensions v2 for coadsorbate recovery, data-root=/tmp/local-export')
    assert parsed.parameters['training_recipe']=='scientific_extensions_v2'


def test_review_rejects_out_of_grid_without_extrapolation():
    from tools.scientific_review import predict_with_review
    result=predict_with_review({'energy':np.arange(10)},np.ones((1,9)),np.arange(9))
    assert result['status']=='review_required' and result['predictions']==[]
