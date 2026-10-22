"""Verify released predictions/counts and record checks for the manuscript bundle."""
from pathlib import Path
import json, sys, platform, importlib.metadata
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score
ROOT=Path(__file__).resolve().parents[1]; D=ROOT/'figure_data'; results={}
expected={'oof_independent_selected_micro.npz':(0.7790697674418605,0.6382765882765882,0.6896551724137931),
          'oof_provisional_best.npz':(0.7978142076502732,0.6597402597402597,0.7011494252873564)}
for name,target in expected.items():
    z=np.load(D/name); y=z['truth'];p=z['prediction'];assert y.shape==p.shape==(87,11)
    values=(f1_score(y,p,average='micro',zero_division=0),f1_score(y,p,average='macro',zero_division=0),accuracy_score(y,p))
    assert np.allclose(values,target,atol=1e-12)
    results[name]=dict(micro_f1=values[0],macro_f1=values[1],exact_match=values[2])
manifest=pd.read_csv(D/'cv_manifest.csv');assert len(manifest)==87
assert manifest.structure_hash.nunique()==87
assert manifest.configuration.nunique()==87
assert sorted(manifest.groupby('fold').size())==[17,17,17,18,18]
support=pd.read_csv(D/'species_support.csv');assert support.support.sum()==99
combo=pd.read_csv(D/'figure1_combination_counts.csv');assert combo.usable.sum()==87
assert combo['no completed record'].sum()+combo['record excluded'].sum()+combo.usable.sum()==155
assert combo.loc[combo.combo.str.contains(r'\+'),'usable'].sum()==9
inventory=pd.read_csv(D/'all_unique_structures.csv');assert len(inventory)==155
layers=pd.read_csv(D/'representative_layer_spectra.csv')
assert len(layers.configuration.unique())==2
assert sorted(layers.layer.unique())==[2,3,4,5]
for _,g in layers.groupby(['configuration','layer']):
    assert len(g)==276 and np.isfinite(g[['energy_eV','absorption']]).all().all()
    assert (g.energy_eV.diff().dropna()>0).all()
results['counts']=dict(unique_inventory=155,configurations=87,layer_spectra=348,mixed_configurations=9)
results['verification']='Saved predictions and plotted data checked; no retraining or raw-HPC audit performed.'
(ROOT/'provenance/verification.json').write_text(json.dumps(results,indent=2)+'\n')
env=dict(python=sys.version,platform=platform.platform(),dependencies={x:importlib.metadata.version(x) for x in ['numpy','pandas','matplotlib','scikit-learn']})
(ROOT/'provenance/plot_environment.json').write_text(json.dumps(env,indent=2)+'\n')
print(json.dumps(results,indent=2))
