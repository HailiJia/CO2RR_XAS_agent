from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import pearsonr,spearmanr
from scipy.spatial.distance import cosine
B=Path(__file__).resolve().parents[1]/'study/reference_benchmark_v3'
def test_same_final_arrays_normalization_and_metrics():
 a=pd.read_csv(B/'plotted_arrays.csv');m=pd.read_csv(B/'similarity_metrics.csv');assert len(m)==28
 grids=[]
 for key,d in a.groupby(['material','method','comparison_type'],sort=False):
  x=d.calculation_normalized.to_numpy();y=d.experiment_normalized.to_numpy();e=d.energy_eV.to_numpy();grids.append(e)
  assert len(e)==176 and np.isfinite(x).all() and np.isfinite(y).all()
  assert min(x.min(),y.min())>=-1e-12 and max(x.max(),y.max())<=1+1e-12
  np.testing.assert_allclose([x.min(),x.max(),y.min(),y.max()],[0,1,0,1],atol=1e-12)
  row=m[(m.material==key[0])&(m.method==key[1])&(m.comparison_type==key[2])].iloc[0]
  np.testing.assert_allclose([1-cosine(x,y),pearsonr(x,y).statistic,spearmanr(x,y).statistic],[row.cosine_similarity,row.Pearson_r,row.Spearman_rho],atol=2e-14)
  np.testing.assert_allclose([1-cosine(x,x),pearsonr(x,x).statistic,spearmanr(x,x).statistic],[1,1,1],atol=2e-14)
 for grid in grids:np.testing.assert_array_equal(grid,grids[0])
 assert set(m.normalization)=={'(I-Imin)/(Imax-Imin), extrema over identical fixed 8980–9015 eV interval'}
def test_missing_slots_and_broadening_prescription():
 m=pd.read_csv(B/'similarity_metrics.csv');assert len(m[['material','method']].drop_duplicates())==14
 assert not ((m.material=='CuO')&(m.method=='vasp_pbe')).any()
 assert np.all(m.Lorentzian_additional_FWHM_eV==0)
 np.testing.assert_allclose(m[m.method.str.startswith('vasp')].Gaussian_sigma_eV,1)
 assert json.load(open(B/'settings.json'))['CuO_fdm_CPU_time_s']==129600
