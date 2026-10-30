"""Shared, verified training API for the UI, agent and local CLI.

``archived_primary`` delegates to the frozen nested selector in analysis.py;
``training_only_global_threshold`` is a separate new integration recipe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .study_cohort import load_cohort, sha256, verify_export
from .spectral_multilabel import (
    SPECIES, feature_for_records, fit_binary, binary_scores, f1_metrics,
    save_model, predict_records, train_cohort_holdout, implementation_fingerprint,
)


def _hash_tree(paths):
    return {str(Path(p).name): sha256(p) for p in sorted(map(Path, paths)) if Path(p).is_file()}


def _run_new_global(cohort, output_dir):
    """Fit the explicitly new grouped training-only recipe and persist replay artifacts."""
    result = train_cohort_holdout(cohort, fold=0)
    bundle = result['bundle']
    model_path = output_dir / 'global_threshold_fold0.joblib'
    import joblib
    joblib.dump(bundle, model_path)
    loaded = joblib.load(model_path)
    ids = result['test_indices']
    pred, scores = predict_records(loaded, [cohort['records'][i] for i in ids],
                                   [cohort['populations'][i] for i in ids])
    if not np.array_equal(pred, result['prediction']):
        raise RuntimeError('Saved-model held-out prediction replay mismatch')
    np.savez_compressed(output_dir / 'heldout_predictions.npz',
                        truth=cohort['Y'][ids], prediction=pred.astype(int),
                        scores=scores, species=np.asarray(SPECIES),
                        fold=cohort['split'].iloc[ids].fold.to_numpy())
    (output_dir / 'recipe.json').write_text(json.dumps({
        'recipe': 'training_only_global_threshold',
        'status': 'new integration recipe; not the archived primary selector',
        'metrics': result['metrics'], 'ranking': result['ranking'],
        'heldout_configurations': cohort['split'].iloc[ids].configuration.tolist(),
        'missing_positive_training': result['missing_positive_training'],
        'model_sha256': sha256(model_path),
    }, indent=2) + '\n')
    return {'metrics': result['metrics'], 'prediction_replay_identical': True,
            'model': model_path.name, 'heldout_count': len(ids)}


def _write_archived_recipe_and_heldout(output_dir):
    """Write a concise importable recipe summary and the frozen held-out OOF rows."""
    output_dir = Path(output_dir)
    cfg = json.loads((output_dir / 'run_config.json').read_text())
    z = np.load(output_dir / 'oof_independent_selected_micro.npz')
    split = pd.read_csv(output_dir / 'cv_manifest.csv')
    np.savez_compressed(output_dir / 'heldout_predictions.npz',
                        truth=z['truth'], prediction=z['prediction'],
                        fold=z['fold'], species=z['species'],
                        configuration=split.configuration.to_numpy(dtype=str))
    recipe = {
        'name': 'archived_primary_nested_selector',
        'selection': 'outer folds frozen in cv_manifest.csv; candidate selection and threshold fitting use label-aware grouped inner folds within each outer training partition',
        'candidate_space': cfg.get('candidates'),
        'threshold_candidates': cfg.get('thresholds'),
        'preprocessing': 'per-spectrum normalization/features; any learned estimator preprocessing is fit on training folds only',
        'grouping': 'current physical geometry; layer/site/backend variants share group',
        'species_order': z['species'].astype(str).tolist(),
        'metrics': {'micro_f1': 0.7790697674418605,
                    'macro_f1': 0.6382765882765882,
                    'exact_set_accuracy': 0.6896551724137931},
        'run_config_sha256': sha256(output_dir / 'run_config.json'),
        'implementation_dependency_fingerprint': implementation_fingerprint(),
        'heldout_predictions_sha256': sha256(output_dir / 'heldout_predictions.npz'),
        'archived_outputs_are_immutable': True,
    }
    (output_dir / 'recipe.json').write_text(json.dumps(recipe, indent=2) + '\n')


def run_audited_training(data_root, output_dir, split_manifest=None,
                         recipe='archived_primary'):
    """Audit a full local export, fit the requested recipe and write a sanitized trace.

    The export root may be given either as its directory or as its ``data`` child.
    A new, empty output directory is required so cache provenance is unambiguous.
    """
    if recipe == 'signed_shape_size_v3':
        b=Path(data_root);manifest=json.loads((b/'artifact_sha256.json').read_text())
        for name in ['training_arrays.npz','training_manifest.json']:
            if sha256(b/name)!=manifest[name]['sha256']:raise ValueError('Prepared identity export hash mismatch: '+name)
        families=json.loads((Path(__file__).resolve().parents[1]/'study/relative_identity_guarded_v2/near_duplicate_family_map.json').read_text())
        if families['training_manifest_sha256']!=sha256(b/'training_manifest.json') or families['geometry_sha256']!=sha256(b/'geometry_arrays.json'):raise ValueError('Near-family audit belongs to another export')
        from .signed_shape_size import train
        return train(data_root,output_dir)
    if recipe == 'relative_identity_v2':
        # This recipe consumes a prepared audited export, never an arbitrary folder.
        b=Path(data_root);manifest=json.loads((b/'artifact_sha256.json').read_text())
        for name in ['training_arrays.npz','training_manifest.json']:
            if hashlib.sha256((b/name).read_bytes()).hexdigest()!=manifest[name]['sha256']:
                raise ValueError('Prepared identity export hash mismatch: '+name)
        from .identity_comparison import train
        families=json.loads((Path(__file__).resolve().parents[1]/'study/relative_identity_guarded_v2/near_duplicate_family_map.json').read_text())
        if families['training_manifest_sha256']!=hashlib.sha256((b/'training_manifest.json').read_bytes()).hexdigest() or families['geometry_sha256']!=hashlib.sha256((b/'geometry_arrays.json').read_bytes()).hexdigest():raise ValueError('Near-family audit belongs to another export')
        result=train(data_root,output_dir,family_map=families)
        from .grouped_identity_learning import run as group_learning
        result.update(group_learning(output_dir))
        return result
    if recipe == 'dataset_refresh_v1':
        from .dataset_refresh_training import train
        return train(data_root, output_dir)
    if recipe == 'differential_v1':
        from .differential_route import train
        return train(data_root,output_dir)
    if recipe == 'coverage_v1':
        from .coverage_study import convert, audit
        from .coverage_inference import run
        repo=Path(__file__).resolve().parents[1]
        target=Path(output_dir).resolve()
        if target.exists() and any(target.iterdir()):raise ValueError('New empty output directory required; archived runs cannot be overwritten')
        if not data_root:raise ValueError('Local export data_root required')
        convert(data_root,repo,target)
        audit(data_root,repo,target)
        result=run(target)
        import joblib
        from .coverage_inference import predict
        manifest={r['group']:r for r in json.loads((target/'model_configuration_manifest.json').read_text())}
        expected=pd.read_csv(target/'coverage_heldout_predictions.csv')
        expected=expected[(expected.protocol=='coverage_configuration_folds')&(expected.model=='nested')].set_index('configuration')
        replayed=0
        for path in (target/'models').glob('coverage_fold*.joblib'):
            model=joblib.load(path)
            for group in model['test_groups']:
                r=manifest[group];record=json.loads((target/'records'/r['configuration']/'isaac_record_layers.json').read_text());pop={int(k):v['population'] for k,v in r['layer_evidence'].items()}
                if not np.isclose(predict(record,model,pop)[0],expected.loc[r['configuration'],'prediction_ML'],rtol=0,atol=1e-11):raise RuntimeError('Coverage saved-model replay mismatch')
                replayed+=1
        trace=target/'invocation_trace.json'
        trace.write_text(json.dumps({'action':'audited_training','recipe':recipe,'scope':'local conversion/audit/fit only','coverage_heldout_records_replayed':replayed,'protocol_sha256':sha256(target/'protocol.json'),'portal_writes':False,'HPC_jobs':False},indent=2)+'\n')
        result.update(output_dir=str(target),trace=str(trace),prediction_replay_identical=True)
        return result
    if recipe == 'fixed_top_two_v4':
        from .fixed_top_two import run
        result=run(data_root, output_dir, split_manifest)
        import runpy
        report=runpy.run_path(str(Path(__file__).resolve().parents[1]/'study/code/report_fixed_top_two.py'))
        report['main'](output_dir,Path(output_dir)/'report')
        result['metrics']=str(Path(output_dir)/'report/metrics.csv')
        return result
    if recipe == 'updated_data_v3':
        from .updated_data_training import run
        return run(data_root, output_dir, split_manifest)
    if recipe == 'motif_holdout_v2':
        from .scientific_motif import run
        return run(data_root, output_dir, split_manifest)
    if recipe == 'scientific_extensions_v2':
        from .scientific_extensions import run
        return run(data_root, output_dir, split_manifest)
    if recipe not in {'archived_primary', 'training_only_global_threshold'}:
        raise ValueError('Unsupported audited training recipe')
    if not data_root:
        raise ValueError('audited_training requires data_root pointing to a verified cohort export')
    supplied = Path(data_root).expanduser().resolve()
    export_root = supplied.parent if supplied.name == 'data' else supplied
    manifest_path = export_root / 'dataset_manifest.json'
    if not manifest_path.is_file():
        raise ValueError('A complete verified cohort export with dataset_manifest.json is required')
    verified = verify_export(export_root)
    if int(verified.get('configuration_count', -1)) != 87:
        raise ValueError('This archived primary recipe requires the frozen 87-configuration export')
    records_root = export_root / 'data'
    split = Path(split_manifest).expanduser().resolve() if split_manifest else export_root / 'splits.csv'
    if not split.is_file():
        raise ValueError('Frozen split manifest is missing')
    output = Path(output_dir).expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Training output must be new and empty; use a fresh cache/output directory')
    output.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort(records_root, split)
    trace = {
        'request_action': 'audited_training',
        'arguments': {'data_root': 'verified-export/data', 'split_manifest': split.name,
                      'recipe': recipe, 'output_dir': output.name},
        'data_manifest_sha256': sha256(manifest_path),
        'split_manifest_sha256': sha256(split),
        'configuration_count': len(cohort['records']),
        'geometry_group_count': len(set(cohort['groups'].tolist())),
        'code_and_dependency_fingerprint': implementation_fingerprint(),
        'credentials_included': False,
    }
    if recipe == 'archived_primary':
        repo = Path(__file__).resolve().parents[1]
        script = repo / 'CO2RR_ML/updated_dataset_20261020/analysis.py'
        env = dict(os.environ)
        env.update(CO2RR_REPO_ROOT=str(repo))
        run = subprocess.run([sys.executable, str(script), '--data-root', str(records_root),
                              '--output-dir', str(output), '--split-manifest', str(split)],
                             cwd=repo, env=env, capture_output=True, text=True)
        if run.returncode:
            trace['training_failure'] = 'Archived analysis exited nonzero; stderr was omitted from the sanitized trace.'
            (output / 'invocation_trace.json').write_text(json.dumps(trace, indent=2) + '\n')
            raise RuntimeError('Archived primary fitting failed; inspect sanitized invocation_trace.json')
        from .study_replay import replay_oof, replay_models
        replay_oof(output, output)
        replay = replay_models(records_root, output, output, split)
        _write_archived_recipe_and_heldout(output)
        trace.update({
            'recipe': 'archived_primary_nested_selector',
            'archived_score_reproduced': bool(np.allclose(
                [replay['metrics']['micro_f1'], replay['metrics']['macro_f1']],
                [0.7790697674418605, 0.6382765882765882], atol=1e-10, rtol=0)),
            'metrics': replay['metrics'],
            'heldout_model_roundtrip_identical': all(x['export_import_scores_identical'] and x['archived_predictions_identical'] for x in replay['folds']),
            'model_count': len(replay['folds']),
            'outputs': _hash_tree([output/'run_config.json', output/'nested_selections.csv',
                                   output/'oof_independent_selected_micro.npz', output/'saved_model_replay.json',
                                   output/'recipe.json', output/'heldout_predictions.npz']),
        })
        if not trace['archived_score_reproduced'] or not trace['heldout_model_roundtrip_identical']:
            raise RuntimeError('Archived primary score or saved-model replay did not match')
    else:
        report = _run_new_global(cohort, output)
        trace.update({'recipe': 'training_only_global_threshold_new_recipe', **report,
                      'outputs': _hash_tree([output/'recipe.json', output/'heldout_predictions.npz', output/'global_threshold_fold0.joblib'])})
    (output / 'invocation_trace.json').write_text(json.dumps(trace, indent=2) + '\n')
    return {'status': 'success', 'recipe': trace['recipe'], 'output_dir': str(output),
            'metrics': trace.get('metrics'), 'prediction_replay_identical': trace.get('heldout_model_roundtrip_identical', trace.get('prediction_replay_identical')),
            'trace': str(output / 'invocation_trace.json')}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--split-manifest')
    p.add_argument('--recipe', choices=['archived_primary', 'training_only_global_threshold', 'scientific_extensions_v2', 'motif_holdout_v2', 'updated_data_v3', 'fixed_top_two_v4', 'coverage_v1', 'differential_v1', 'dataset_refresh_v1', 'relative_identity_v2', 'signed_shape_size_v3'], default='archived_primary')
    a = p.parse_args(argv)
    print(json.dumps(run_audited_training(a.data_root, a.output_dir, a.split_manifest, a.recipe), indent=2))


def review_extension_model(model_path, raw_spectra, energy, **domain):
    """Shared callable for agent/UI review of a trusted extension model bundle."""
    import joblib
    from .scientific_review import predict_with_review
    return predict_with_review(joblib.load(model_path), raw_spectra, energy, **domain)


if __name__ == '__main__':
    main()
