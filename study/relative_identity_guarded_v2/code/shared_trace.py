from pathlib import Path
import json
import numpy as np
from tools.coverage_study import dump
from tools.differential_route import infer
B=Path('study/relative_identity_guarded_v2');z=np.load(B/'results/inputs.npz');r=json.load(open(B/'results/source_manifest.json'));t=json.load(open(B/'shared_route_trace.json'));i=next(i for i,a in enumerate(r) if a['configuration']==t['heldout_configuration']);c={'input_kind':'precomputed_difference','sign':'reaction-minus-reference','energy_registration':'shared simulated photon-energy scale','normalization':'shared_reference_near_edge_scale','reference_chemistry':'audited substrate-specific clean simulated slab','domain':r[i]['domain'],'noise':'not measured; simulation','processing':'audited physical top two layers; near-edge common divisor','input_energy_eV':z['energy'].tolist()};t['input_convention']=c;t['result']=infer(B/'results/models/nested_joint130_signed_shape_logistic_0.joblib',z['delta'][i],None,{},{},r[i]['domain'],r[i]['domain'],experimental_convention=c);dump(B/'shared_route_trace.json',t)
