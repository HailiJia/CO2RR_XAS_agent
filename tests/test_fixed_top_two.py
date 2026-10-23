"""Fixed-depth and paired-evaluation invariants."""
from pathlib import Path
import json,copy
import numpy as np
import pytest
from scipy.fft import dct
from tools.fixed_top_two import load_data,paired_inner,CANDIDATES,SEEDS
from tools.spectral_multilabel import feature_bank,feature_for_records
ROOT=Path(__file__).resolve().parents[1]

def test_author_anchor_and_search_are_declared():
 a=json.loads((ROOT/'study/top_two_20261023/author_anchor.json').read_text());assert a['feature']=='shape_CDF_derivatives' and a['depth'].startswith('fixed raw')
 assert len(CANDIDATES)==20 and set(SEEDS)=={20261020,20261021,20261022}
 assert {leaf for f,leaf in CANDIDATES}=={1,2,4}

def test_inner_and_outer_groups_preserved():
 d=load_data(ROOT)
 for k in range(5):
  tr87,a=paired_inner(d,k,87);tr90,b=paired_inner(d,k,90)
  assert set(tr87)==set(tr90[tr90<87])
  for (ta,va),(tb,vb) in zip(a,b):assert set(ta)==set(tb[tb<87]) and set(va)==set(vb[vb<87]) and not set(d['groups'][tb])&set(d['groups'][vb])

def test_archived_feature_bank_not_expanded_by_DCT_inference():
 e=np.linspace(8974,9029,276);y=(1+np.tanh((e-8985)/2))[None,:];assert list(feature_bank(y,e))==['normalized','CDF','derivative','shape_CDF_derivatives','DCT24']

@pytest.mark.parametrize('name',['DCT8','DCT16','DCT32','shape_CDF_derivatives'])
def test_top_two_inference_ignores_deeper_layers(name):
 e=np.linspace(8974,9029,276);y=1+np.tanh((e-8985)/2);record={'measurement':{'series':[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'unit':'eV','values':e.tolist()}],'channels':[{'name':'absorption_cross_section','values':(y*(1+l/10)).tolist()}]} for l in [2,3,4,5]]}};other=copy.deepcopy(record)
 for series in other['measurement']['series'][:2]:series['channels'][0]['values']=(100+np.sin(e)).tolist()
 p=[{2:36,3:36,4:36,5:36}];a=feature_for_records([record],p,e,'top_2',name);b=feature_for_records([other],p,e,'top_2',name);assert np.array_equal(a,b)
 if name.startswith('DCT'):
  raw=np.mean([np.array(record['measurement']['series'][i]['channels'][0]['values']) for i in [2,3]],axis=0)[None,:];expected=dct(feature_bank(raw,e)['normalized'],type=2,norm='ortho',axis=1)[:,:int(name[3:])];np.testing.assert_allclose(a,expected,rtol=1e-12,atol=1e-14)

def test_old_macro_correction_and_prediction_tradeoffs():
 x=json.loads((ROOT/'study/top_two_20261023/historical_v3_reporting_correction.json').read_text());assert x['CO_precision']==1 and x['CO_recall']==2/3 and x['CO_f1']==.8 and x['other_positive_species_tested'] is False
 assert (x['became_correct'],x['became_wrong'],x['CO_only_cases_acquiring_false_H_OH'])==(4,5,2)
 assert np.isclose(x['legacy_eleven_label_macro_F1'],.8/11)

def test_combined_fixed_controls_use_old_thresholds_and_full_cohorts():
 b=ROOT/'study/top_two_20261023';meta=__import__('pandas').read_csv(b/'Combined_control_thresholds.csv')
 for seed in SEEDS:
  for k in range(5):
   a=meta[(meta.cohort==87)&(meta.seed==seed)&(meta.fold==k)&(meta.stage=='fixed_Combined_transferred87_threshold')].iloc[0];c=meta[(meta.cohort==90)&(meta.seed==seed)&(meta.fold==k)&(meta.stage=='fixed_Combined_transferred87_threshold')].iloc[0];assert a.thresholds==c.thresholds and a.feature==c.feature=='shape_CDF_derivatives'

def test_natural_language_route_is_fixed_top_two():
 from agent.co2rr_xas_agent import LocalIntentParser
 p=LocalIntentParser().parse('Train adsorbate identity using the fixed top-two paired benchmark');assert p.parameters['training_recipe']=='fixed_top_two_v4'
