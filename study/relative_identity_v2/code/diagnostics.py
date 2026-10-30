from pathlib import Path
import json,hashlib
import numpy as np,pandas as pd,joblib
from tools.identity_comparison import transform,score_model,SEEDS
from tools.differential_identity import signed_shape,VERSION
from tools.spectral_multilabel import SPECIES,encode_labels,f1_metrics
from tools.coverage_study import dump,sha
from tools.differential_route import infer
from agent.co2rr_xas_agent import LocalIntentParser
B=Path('study/relative_identity_v2');O=B/'results';D=Path('study/dataset_20261029');r=json.load(open(O/'source_manifest.json'));z=np.load(O/'inputs.npz');e=z['energy'];y=encode_labels([a['identity_labels'] for a in r]);pair=[];chemical=[];neg=[];low=[];nearest=[];replays=[]
# Paired intervals conditional on the already frozen predictions; all comparisons shown.
for cohort in ['expanded118','joint130']:
 for family in ['logistic','trees']:
  for a,b in [('native_normalized','corrected_normalized'),('corrected_normalized','common_raw'),('common_raw','relative_delta'),('relative_delta','signed_shape')]:
   aa=np.load(O/f'nested_{cohort}_{a}_{family}.npz');bb=np.load(O/f'nested_{cohort}_{b}_{family}.npz');assert np.array_equal(aa['indices'],bb['indices']);yy=aa['truth'];pa=aa['prediction'];pb=bb['prediction'];rng=np.random.default_rng(20261030);v=[]
   for _ in range(1000):
    ix=rng.integers(0,len(yy),len(yy));v.append(f1_metrics(yy[ix],pb[ix])['micro_f1']-f1_metrics(yy[ix],pa[ix])['micro_f1'])
   pair.append(dict(cohort=cohort,family=family,baseline=a,intervention=b,difference=f1_metrics(yy,pb)['micro_f1']-f1_metrics(yy,pa)['micro_f1'],lower=np.quantile(v,.025),upper=np.quantile(v,.975),uncertainty='conditional fixed-prediction configuration bootstrap; not refit or new-parent confirmation'))
# Connectivity-derived descriptors define reporting strata only; never predictors.
molecules=json.load(open(D/'molecule_audit.json'));mol={}
for a in molecules:mol.setdefault(a['configuration'],[]).append(a)
for p in O.glob('nested_*.npz'):
 if 'fixed0p5' in p.stem:continue
 q=np.load(p);ids=q['indices'];pp=q['prediction'];ss=q['score'];strata={}
 for j,i in enumerate(ids):
  ms=mol.get(r[i]['configuration'],[]);dist=[a['interface_distance_A'] for a in ms if a.get('interface_distance_A') is not None];values={'interface_le1p5A':bool(min(dist)<=1.5) if dist else None}
  contacts=[]
  for a in ms:
   if a['species'] not in ['CO','CHO','COH','OCHO']:continue
   cc=[t for t in a.get('metal_contacts_under_3A',[]) if t[3]<=2.5];cs={t[0] for t in cc};counts=[(sum(t[0]==v and t[2]=='Cu' for t in cc),sum(t[0]==v and t[2]=='Au' for t in cc)) for v in cs]
   contacts+=counts
  for nc,na in contacts:values[f'adsorbate_anchor_contacts_Cu{nc}_Au{na}']=True
  for k,v in values.items():
   if v is not None:strata.setdefault(k+'='+str(v),[]).append(j)
  for sp in ['CO','CHO','COH']:
   k=SPECIES.index(sp)
   if y[i,k] and not pp[j,k]:chemical.append({'analysis':p.stem,'configuration':r[i]['configuration'],'omitted':sp,'true_set':';'.join(r[i]['identity_labels']),'predicted_set':';'.join(s for s,v in zip(SPECIES,pp[j]) if v),'score':float(ss[j,k]),'coverage_CO_per_all_metals':r[i]['coverage_ML'],'minimum_interface_distance_A':min(dist) if dist else None,'molecule_graphs':r[i]['connectivity']})
 for name,jj in strata.items():
  yy=y[ids[jj]];pred=pp[jj];pairrow={'analysis':p.stem,'stratum':name,'n':len(jj),**f1_metrics(yy,pred)};nearest.append(pairrow)
# Per-model low/null synthetic diagnostic and exact saved-model replay.
for p in sorted((O/'models').glob('*.joblib')):
 m=joblib.load(p);ids=np.array(m['test_indices']);rep=m['representation'];x=z['raw_native'][ids] if rep=='native_normalized' else z['raw_physical'][ids] if rep in ['corrected_normalized','common_raw'] else z['delta'][ids];s=score_model(m['model'],transform(x,e,rep,m['feature'],m['floor']));assert not set(m['train_groups'])&set(m['test_groups']);replays.append({'path':str(p.relative_to(O)),'sha256':sha(p),'reload_finite':bool(np.isfinite(s).all()),'groups_disjoint':True})
 if p.name.startswith('nested_') and rep in ['relative_delta','signed_shape']:
  rms=np.sqrt(np.mean(z['delta'][ids]**2,axis=1));keep=rms>m['floor'];low.append({'model':p.name,'n_queries':len(ids),'retained_fraction_signal_floor':float(keep.mean()),'floor':m['floor'],'floor_definition':'training heuristic; not a measured detection limit','n_clean_queries':int((y[ids].sum(axis=1)==0).sum()),'clean_false_positive_labels_before_abstention':int(((s>=m['thresholds'])&(y[ids].sum(axis=1)==0)[:,None]).sum())})
  for sigma in [0.,1e-5,1e-3]:
   noise=np.random.default_rng(23).normal(0,sigma,(8,len(e)));scores=score_model(m['model'],transform(noise,e,rep,m['feature'],m['floor']));valid=np.sqrt(np.mean(noise**2,axis=1))>m['floor'];neg.append({'model':p.name,'synthetic_null_noise_RMS':sigma,'n_synthetic_views':8,'false_positive_labels_before_abstention':int((scores>=m['thresholds']).sum()),'retained_fraction':float(valid.mean()),'not_independent_clean_examples':True})
# Shared route demonstration: real held-out saved geometry, supplied as precomputed normalized delta.
p=O/'models/nested_joint130_signed_shape_logistic_0.joblib';m=joblib.load(p);i=int(next(j for j in m['test_indices'] if np.linalg.norm(z['delta'][j])>0));convention={'input_kind':'precomputed_difference','sign':'reaction-minus-reference','energy_registration':'shared simulation photon-energy convention; no independent peak shifts','normalization':'shared_reference_near_edge_scale','reference_chemistry':'audited substrate-specific clean simulation','domain':r[i]['domain'],'noise':'not measured; simulated native data','processing':'top-two population weighted; source correction; shared short near-edge reference divisor','input_energy_eV':e.tolist()};prediction=infer(p,z['delta'][i],None,{},{},r[i]['domain'],r[i]['domain'],experimental_convention=convention)
trace={'natural_language_request':'Train relative_identity_v2 on normalized signed differential XAS','parsed_recipe':LocalIntentParser().parse('Train relative_identity_v2 on normalized signed differential XAS').parameters['training_recipe'],'fit_entrypoint':'tools.audited_training.run_audited_training (actual fit log)','inference_entrypoint':'tools.differential_route.infer','heldout_configuration':r[i]['configuration'],'training_groups_exclude_query':r[i]['group'] not in m['train_groups'],'sample_record_sha256':r[i]['sample_record_sha256'],'reference_sha256':r[i]['reference_record_sha256'],'query_delta_sha256':hashlib.sha256(z['delta'][i].tobytes()).hexdigest(),'result':prediction,'HPC_jobs':False,'portal_writes':False,'scope':'local simulated held-out query, not measured experimental validation'};dump(B/'shared_route_trace.json',trace)
# Unsupported chemistry remains review-only. No unsupported species are removed from targets.
sm=json.load(open(D/'scale/configuration_manifest.json'));sz=np.load(D/'scale/per_Cu_spectra.npz');print('scale arrays',sz.files)
unsupported=[]
for row in json.load(open(D/'training_exclusions.json')):
 ix=next(j for j,t in enumerate(sm) if t['configuration']==row['configuration']);ref=next(j for j,t in enumerate(sm) if t['domain']==sm[ix]['domain'] and not t['identity_labels']);# data array key inspected below
 unsupported.append({**row,'status':'unsupported chemistry; review required; excluded from all fits','connectivity':sm[ix].get('connectivity')})
dump(B/'unsupported_queries.json',unsupported)
pd.DataFrame(pair).to_csv(B/'paired_representation_intervals.csv',index=False);pd.DataFrame(nearest).to_csv(B/'geometry_strata.csv',index=False);dump(B/'CO_CHO_COH_omissions.json',chemical);pd.DataFrame(low).to_csv(B/'signal_abstention.csv',index=False);pd.DataFrame(neg).to_csv(B/'synthetic_negative_queries.csv',index=False);dump(B/'saved_model_recheck.json',replays)
