from copy import deepcopy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from tools.component_audit import audit_components,write_metadata_correction
from tools.layer_population import verify_layer_population,aggregate_layers
from tools.spectral_multilabel import (SPECIES,identity_set,encode_labels,fit_binary,binary_scores,rank_outer_training,feature_bank)
from tools.isaac_roundtrip import FixturePortal,roundtrip_records
from tools.reference_benchmark import compare_reference

def structure(atoms,positions,cell=None):return dict(atoms=atoms,positions=np.asarray(positions,float),cell=np.diag([20.,20.,25.]) if cell is None else np.asarray(cell,float))

@pytest.mark.parametrize('atoms,positions,counts',[
 (['C','O','H'],[[3,3,10],[4.2,3,10],[8,3,10]],{'CO':1,'H':1}),
 (['C','O','H'],[[3,3,10],[4.2,3,10],[3,4.1,10]],{'CHO':1}),
 (['C','O','H'],[[3,3,10],[4.2,3,10],[5.1,3,10]],{'COH':1}),
 (['C','O','O','H'],[[3,3,10],[4.2,3,10],[1.8,3,10],[3,4.1,10]],{'OCHO':1}),
 (['C','O','O','H'],[[3,3,10],[4.2,3,10],[1.8,3,10],[5.1,3,10]],{'COOH':1}),
 (['C','C','O','O'],[[3,3,10],[4.45,3,10],[1.8,3,10],[5.65,3,10]],{'OCCO':1}),
 (['C','C','O','O'],[[3,3,10],[6,3,10],[1.8,3,10],[7.2,3,10]],{'CO':2}),
 (['C','C','O','O','H'],[[3,3,10],[4.45,3,10],[1.8,3,10],[5.65,3,10],[3,4.1,10]],{'COCHO':1}),
 (['C','C','O','O','H'],[[3,3,10],[4.45,3,10],[1.8,3,10],[5.65,3,10],[.9,3,10]],{'OCCOH':1}),
])
def test_component_distinctions(atoms,positions,counts):
    audit=audit_components(structure(atoms,positions));assert audit['counts']==counts and audit['status']=='verified'

@pytest.mark.parametrize('atoms,positions,expected,status',[
 (['C','O','C','O'],[[0,0,5],[1.2,0,5],[2.4,0,5],[3.6,0,5]],{},'review_required'), # C-O-C-O chain
 (['C','C','O','O'],[[2,2,5],[8,2,5],[3.2,2,5],[9.2,2,5]],{'CO':2},'verified'), # two disconnected COs
 (['C','C','O','O'],[[2,2,5],[3.45,2,5],[2,3.3,5],[3.45,3.3,5]],{},'review_required'), # O-O-bridged graph
])
def test_c2o2_formula_does_not_substitute_for_bond_topology(atoms,positions,expected,status):
    audit=audit_components(structure(atoms,positions))
    assert audit['status']==status and audit['counts']==expected
    assert audit['counts'].get('OCCO',0)==0

def test_expected_component_graph_is_checked_and_requested_identity_retained():
    s=structure(['O','C','C','O'],[[2,2,5],[3.2,2,5],[4.65,2,5],[5.85,2,5]])
    audit=audit_components(s,[dict(species='OCCO',atom_indices=[0,1,2,3],formula_counts={'O':2,'C':2},expected_edges=[[1,2],[2,3],[3,4]])])
    assert audit['status']=='verified'
    mismatch=audit_components(s,[dict(species='OCCO',atom_indices=[0,1,2,3],expected_edges=[[1,2],[2,4],[3,4]])])
    assert mismatch['status']=='review_required' and mismatch['mapping_evidence'][0]['edge_graph_matches'] is False

def test_periodic_skew_and_nonperiodic_vacuum_and_cutoff_boundary():
    skew=[[10,0,0],[9,1,0],[0,0,25]]
    audit=audit_components(structure(['C','O'],[[0,0,10],[9.31,.49,10]],skew));assert audit['counts']=={'CO':1}
    audit=audit_components(structure(['C','O'],[[0,0,.1],[0,0,24.9]]));assert audit['counts']!={'CO':1}
    audit=audit_components(structure(['C','O'],[[3,3,10],[4.65,3,10]]));assert audit['status']=='review_required'

def test_mapping_provenance_keeps_requested_and_observed():
    s=structure(['C','O','H'],[[3,3,10],[4.2,3,10],[8,3,10]])
    a=audit_components(s,[dict(species='CHO',atom_indices=[0,1,2],formula_counts={'C':1,'O':1,'H':1})]);assert a['counts']=={'CO':1,'H':1}
    assert a['mapping_evidence'][0]['requested']=='CHO' and not a['mapping_evidence'][0]['exact_component']
    with pytest.raises(ValueError):audit_components(s,[dict(atom_indices=[0,0])])

def layer_fixture(tmp_path):
    p=tmp_path;cell=np.diag([10.,10.,25.]);pos=np.array([[1.,1.,2*l] for l in range(5) for _ in range(2)]);pos[1::2,0]=4
    lines=['fixture','1']+[' '.join(map(str,v)) for v in cell]+['Cu','10','Direct']+[' '.join(map(str,v)) for v in pos@np.linalg.inv(cell)]
    (p/'CONTCAR').write_text('\n'.join(lines)+'\n')
    pd.DataFrame(dict(atom_index=[9,10],height_A=[8.,8.])).to_csv(p/'absorbers.tsv',sep='\t',index=False)
    crystal='\n'.join('29 '+' '.join(map(str,v)) for v in pos@np.linalg.inv(cell))
    (p/'Cu_layer5_in.txt').write_text('Crystal\n10 10 25 90 90 90\n'+crystal+'\nAbsorber\n9 10\nConvolution\nEnd\n')
    (p/'Cu_layer5_convolution_in.txt').write_text('Calculation\nCu_layer5\nConv_out\nCu_layer5_conv\nConvolution\nEnd\n')
    energy=np.arange(8974.,8980.,.2)
    for atom in [9,10]:
        header=[8979,29,1,1,8,-4,-1.2344,1,1,8826,2500,0,132,1,0]
        with (p/f'Cu_layer5_{atom}.txt').open('w') as f:
            f.write(' '.join(map(str,header))+' = header\nEnergy xanes\n');np.savetxt(f,np.column_stack([energy,np.linspace(.1,1,len(energy))]))
    with (p/'Cu_layer5_conv.txt').open('w') as f:f.write('Energy xanes\n');np.savetxt(f,np.column_stack([energy,np.linspace(.1,1,len(energy))]))
    (p/'fdmnes.out').write_text('\n'.join(f'Gamma_hole = 1.55, E_cut = -1.234, Shift = 0.0 eV, site {i}' for i in [1,2]))
    (p/'Cu_layer5_bav.txt').write_text('Absorbing atom 9 beginning\nAbsorbing atom 10 beginning\n');return p

def test_full_and_subset_retry_with_final_population(tmp_path):
    p=layer_fixture(tmp_path);assert verify_layer_population(p,5)['population']==2
    deck=p/'Cu_layer5_in.txt';deck.write_text(deck.read_text().replace('9 10\nConvolution','9\nConvolution'))
    (p/'Cu_layer5_bav.txt').write_text('Absorbing atom 9 beginning\n')
    assert verify_layer_population(p,5)['population']==2
    (p/'fdmnes.out').rename(p/'fdmnes_convolution-123.out');assert verify_layer_population(p,5)['population']==2

def test_fdmnes_crystal_cell_must_match_relaxed_cell(tmp_path):
    p=layer_fixture(tmp_path);deck=p/'Cu_layer5_in.txt'
    deck.write_text(deck.read_text().replace('10 10 25 90 90 90','20 10 25 90 90 90'))
    with pytest.raises(ValueError,match='cell lengths'):
        verify_layer_population(p,5)

def test_fdmnes_skew_cell_and_rigid_cartesian_frame_are_supported(tmp_path):
    from tools.component_audit import read_poscar
    p=layer_fixture(tmp_path);old=read_poscar(p/'CONTCAR');frac=old['positions']@np.linalg.inv(old['cell'])
    cell=np.array([[10.,0,0],[5.,8.660254037844386,0],[0,0,25.]])
    theta=np.deg2rad(31);rotation=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1.]])
    rotated=cell@rotation;pos=frac@rotated
    lines=['fixture','1']+[' '.join(map(str,v)) for v in rotated]+['Cu','10','Direct']+[' '.join(map(str,v)) for v in frac]
    (p/'CONTCAR').write_text('\n'.join(lines)+'\n')
    (p/'absorbers.tsv').write_text('atom_index\theight_A\n9\t8\n10\t8\n')
    params='10 10 25 90 90 60'
    cr='\n'.join('29 '+' '.join(map(str,v)) for v in frac)
    (p/'Cu_layer5_in.txt').write_text(f'Crystal\n{params}\n{cr}\nAbsorber\n9 10\nConvolution\nEnd\n')
    result=verify_layer_population(p,5)
    assert result['population']==2 and result['coordinate_frame'].endswith('in-plane PBC only')

@pytest.mark.parametrize('fault',['partial','duplicate','element','height','ecut','grid','physical_subset'])
def test_reject_population_faults(tmp_path,fault):
    p=layer_fixture(tmp_path)
    if fault=='partial':(p/'fdmnes.out').write_text('Gamma_hole = 1.55, E_cut = -1.234, Shift = 0.0 eV, site 1\n')
    elif fault=='duplicate':(p/'absorbers.tsv').write_text('atom_index\theight_A\n9\t8\n9\t8\n')
    elif fault=='height':(p/'absorbers.tsv').write_text('atom_index\theight_A\n9\t7\n10\t8\n')
    elif fault=='physical_subset':(p/'absorbers.tsv').write_text('atom_index\theight_A\n9\t8\n')
    elif fault in ['element','ecut']:
        file=p/'Cu_layer5_9.txt';file.write_text(file.read_text().replace('29 1 1','8 1 1') if fault=='element' else file.read_text().replace('-1.2344','-1.2400'))
    elif fault=='grid':
        file=p/'Cu_layer5_10.txt';file.write_text(file.read_text().replace('8.974000000000000000e+03','8.973000000000000000e+03'))
    with pytest.raises(ValueError):verify_layer_population(p,5)

def test_population_average_before_normalization():
    a={5:np.array([[1,0],[2,10],[3,0]]),4:np.array([[1,0],[2,0],[3,1]])}
    e,y=aggregate_layers(a,{5:1,4:3},[5,4]);np.testing.assert_array_equal(y,[0,2.5,.75])
    a[4][0,0]=0
    with pytest.raises(ValueError,match='grid'):aggregate_layers(a,{5:1,4:3},[5,4])

def test_species_order_clean_unknown_and_no_positive_training():
    y=encode_labels(['CO_H','CO_OH','COCHO','clean'])
    assert y[0,list(SPECIES).index('CO')]==y[0,list(SPECIES).index('H')]==1 and y[-1].sum()==0
    assert y[2].sum()==1
    with pytest.raises(ValueError,match='Unseen'):identity_set('made_up_species')
    X=np.arange(12).reshape(6,2);Y=np.zeros((6,3),int);Y[:3,0]=1
    b=fit_binary(X,Y);assert b['no_positive_training']==[1,2];assert (binary_scores(b,X)[:,1:]==0).all()

def test_agent_natural_language_dispatches_to_shared_audited_training(monkeypatch,tmp_path):
    from agent.co2rr_xas_agent import CO2RRXASAgent
    from agent.schemas import XASPlan
    import tools.audited_training as training
    calls=[]
    def fake(**kwargs):
        calls.append(kwargs)
        return {'status':'success','recipe':kwargs['recipe'],'output_dir':kwargs['output_dir']}
    monkeypatch.setattr(training,'run_audited_training',fake)
    request='train the adsorbate multilabel model data-root=/local/export split-manifest=/local/export/splits.csv'
    result=CO2RRXASAgent(default_output_dir=str(tmp_path),use_llm=False).process_request(request)
    assert result['status']=='success' and calls[0]['data_root']=='/local/export'
    assert calls[0]['split_manifest']=='/local/export/splits.csv'
    assert calls[0]['recipe']=='archived_primary'
    plan=XASPlan(action='audited_training',data_root='/local/export')
    assert plan.to_parsed_intent(str(tmp_path)).action=='audited_training'

def test_archived_agent_export_writes_recipe_and_named_heldout_predictions(tmp_path):
    from tools.audited_training import _write_archived_recipe_and_heldout
    run=tmp_path/'run';run.mkdir()
    (run/'run_config.json').write_text(json.dumps({'candidates':[{'feature':'CDF'}],'thresholds':[.35,.5]}))
    pd.DataFrame({'configuration':['cfg-a','cfg-b']}).to_csv(run/'cv_manifest.csv',index=False)
    truth=np.array([[1,0],[0,1]]);pred=truth.copy();fold=np.array([0,1]);species=np.array(['CO','H'])
    np.savez_compressed(run/'oof_independent_selected_micro.npz',truth=truth,prediction=pred,fold=fold,species=species)
    _write_archived_recipe_and_heldout(run)
    z=np.load(run/'heldout_predictions.npz')
    assert z['configuration'].tolist()==['cfg-a','cfg-b'] and np.array_equal(z['prediction'],pred)
    recipe=json.loads((run/'recipe.json').read_text())
    assert recipe['name']=='archived_primary_nested_selector' and recipe['candidate_space']==[{'feature':'CDF'}]

def test_advisor_ranking_cannot_see_outer_test():
    rng=np.random.default_rng(1);Xs={'shape':rng.normal(size=(12,4)),'cdf':rng.normal(size=(12,3))};Y=np.zeros((12,2),int);Y[::2,0]=1;Y[::3,1]=1;groups=np.array([str(i//2) for i in range(12)]);tr=np.arange(8);te=np.arange(8,12)
    first=rank_outer_training(Xs,Y,groups,tr,te,kinds=('logistic',))
    altered={k:v.copy() for k,v in Xs.items()};Y2=Y.copy()
    for X in altered.values():X[te]=1e8
    Y2[te]=1-Y2[te]
    assert first==rank_outer_training(altered,Y2,groups,tr,te,kinds=('logistic',))
    with pytest.raises(ValueError,match='leakage'):rank_outer_training(Xs,Y,groups,np.arange(9),np.arange(9,12))

def test_metadata_correction_is_opt_in_reversible_and_array_preserving(tmp_path):
    record={'measurement':{'series':[{'x':[1,2],'y':[3,4]}]},'sample':{'composition':{'adsorbate_counts':{'CO':1}}}}
    p=tmp_path/'record.json';p.write_text(json.dumps(record));raw=p.read_bytes();corrected=deepcopy(record);corrected['sample']['composition']['adsorbate_counts']={'CO':1,'H':1}
    report=write_metadata_correction(p,corrected,tmp_path/'logs');assert not report['applied'] and p.read_bytes()==raw
    report=write_metadata_correction(p,corrected,tmp_path/'logs',apply=True);assert (tmp_path/'logs'/report['backup']).read_bytes()==raw
    corrected['measurement']['series'][0]['y'][0]=99
    with pytest.raises(ValueError,match='arrays'):write_metadata_correction(p,corrected,tmp_path/'logs',apply=True)

def test_portal_fixture_is_not_deposition_and_checks_prediction_replay():
    record={'measurement':{'series':[{'energy':[1,2],'signal':[3,4]}]},'sample':{'composition':{'adsorbate_counts':{'CO':1,'H':1}}}}
    with pytest.raises(PermissionError):roundtrip_records([record],FixturePortal())
    receipts=roundtrip_records([record],FixturePortal(),allow_upload=True,predict=lambda r:encode_labels([r['sample']['composition']['adsorbate_counts']]))
    assert receipts[0]['transport']=='local_fixture_only' and receipts[0]['prediction_identical']

def test_reference_protocol_requires_raw_data_and_no_extrapolation():
    x=np.linspace(0,10,30);a=np.column_stack([x,np.sin(x)])
    protocol={k:'specified' for k in ['source','license','backend','mode','version','structure','core_hole','convergence','broadening','energy_calibration']}
    protocol.update(interval_eV=[2,8],allowed_shift_eV=1,allowed_scale=[.5,1.5],allowed_offset=[-.1,.1],data_kind='plot')
    with pytest.raises(ValueError,match='raw numeric'):compare_reference(a,a,protocol)
    protocol['data_kind']='raw_numeric';result=compare_reference(a,a,protocol);assert result['unshifted']['normalized_RMSE']==0
    protocol['allowed_shift_eV']=5
    with pytest.raises(ValueError,match='extrapolate'):compare_reference(a,a,protocol)

def test_fit_cache_rejects_wrong_implementation_and_test_partition(tmp_path):
    from tools.spectral_multilabel import validate_fit_cache
    p=tmp_path/'fit.npz';np.savez(p,fingerprint='code-v1',train_indices=[0,1],test_indices=[2],inner_score=np.zeros((2,1)),test_score=np.zeros((1,1)))
    validate_fit_cache(p,[0,1],[2],'code-v1')
    with pytest.raises(ValueError,match='fingerprint'):validate_fit_cache(p,[0,1],[2],'code-v2')
    with pytest.raises(ValueError,match='partition'):validate_fit_cache(p,[0,2],[1],'code-v1')

@pytest.mark.parametrize('element',['C','O'])
def test_vasp_light_element_kpoint_serialization(element):
    from pymatgen.core import Structure,Lattice
    from pymatgen.io.vasp import Kpoints
    from generators.vasp_input_generator import VASPXASGenerator
    st=Structure(Lattice.cubic(12),['Cu','C','O'],[[.5,.5,.4],[.5,.5,.55],[.5,.5,.65]])
    generated=VASPXASGenerator().generate(dict(structure=st,absorbing_atom={'element':element},edge='K'))
    assert st[generated['absorbing_index']].specie.symbol==element
    assert Kpoints.from_str(generated['kpoints']).kpts==[(2,2,1)]
    assert generated['incar_dict']['CLNT']==generated['absorbing_index']+1

def test_execution_log_separates_unknown_validity_and_detects_rewrite(tmp_path):
    from tools.execution_evidence import append_event,read_events
    p=tmp_path/'events.jsonl';base=dict(workflow_id='w',configuration_id='c',attempt_id='job1',event_type='submission_result')
    first=append_event(p,base);assert first['ml_eligible'] is None and first['scheduler_terminal_state'] is None
    append_event(p,dict(base,event_type='process_end',process_exit_code=0))
    rows=read_events(p);assert len(rows)==2 and rows[-1]['output_validator'] is None
    p.write_text(p.read_text().replace('"process_exit_code": 0','"process_exit_code": 1'))
    with pytest.raises(ValueError,match='hash'):read_events(p)

def test_scattering_method_does_not_invent_family_functional_or_core_hole():
    from tools.result_parser import ISAACRecordGenerator
    for backend in ['FDMNES','FEFF']:
        method=ISAACRecordGenerator()._default_method(backend,{}, {}, {})
        assert 'family' not in method and 'functional_name' not in method and 'core_hole_treatment' not in method
        explicit=ISAACRecordGenerator()._default_method(backend,{}, {'core_hole_treatment':'explicit_test_value'}, {})
        assert explicit['core_hole_treatment']=='explicit_test_value'

def test_actual_fdmnes_coordinates_override_unrelated_source_contcar(tmp_path):
    from tools.component_audit import calculation_structure
    p=layer_fixture(tmp_path)
    (p/'POSCAR').write_text('unrelated parent')
    observed,source=calculation_structure(p)
    assert source=='Cu_layer5_in.txt' and len(observed['atoms'])==10

def test_batch_cli_local_contract_and_external_write_guard(tmp_path):
    from batch_validate_upload_isaac_records import main
    record={'measurement':{'series':[{'energy':[1,2],'signal':[3,4]}]},'sample':{'composition':{'adsorbate_counts':{'CO':1}}}}
    p=tmp_path/'record.json';p.write_text(json.dumps(record));original=p.read_bytes();out=tmp_path/'report.json'
    rows=main(['--input',str(p),'--output',str(out)])
    assert not rows[0]['deposited'] and p.read_bytes()==original
    with pytest.raises(SystemExit):main(['--input',str(p),'--output',str(out),'--upload'])

def test_explicit_group_field_cannot_split_identical_current_geometries():
    from tools.ml_splits import groups_for_rows
    rows=[{'record':{'sample':{'configuration_id':f'c{i}','parent_structure_id':f'p{i}','structure_descriptors':{'geometry_hash':'same-final'}}}} for i in range(2)]
    groups,_=groups_for_rows(rows,group_field='sample.parent_structure_id');assert groups[0]==groups[1]

def test_execution_backfill_deduplicates_copied_scheduler_attempts(tmp_path):
    from tools.execution_evidence import backfill_events
    for name in ['alias_a','alias_b']:
        folder=tmp_path/name/'XAS/layer_5';folder.mkdir(parents=True);(folder/'slurm-123.out').write_text('Start: 2026-01-01T00:00:00Z\nExit code: 0\n')
    summary=backfill_events(tmp_path,tmp_path/'report')
    assert summary['observed_attempts_with_surviving_scheduler_stdout']==1 and summary['observed_scheduler_stdout_files']==2
    row=json.loads((tmp_path/'report/execution_events.jsonl').read_text().splitlines()[0]);assert len(row['configuration_aliases'])==2
    assert summ