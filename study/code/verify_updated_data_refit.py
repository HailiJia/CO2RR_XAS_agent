"""Replay uploaded numeric snapshot and every v3 model through shared prediction."""
from pathlib import Path
import sys,json,gzip,hashlib,tarfile
import numpy as np,pandas as pd,joblib
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.spectral_multilabel import predict_records,f1_metrics


def main():
    b=ROOT/'study/retraining_20261022/refit';data=b/'dataset';z=np.load(data/'cohort.npz');checks=json.loads((data/'snapshot_hashes.json').read_text())
    for name,h in checks.items():assert hashlib.sha256((data/name).read_bytes()).hexdigest()==h,name
    source=json.loads((data/'source_manifest.json').read_text());expected={x['sha256']:x['bytes'] for x in source};found={}
    with tarfile.open(data/'source_evidence.tar') as t:
        for m in t.getmembers():
            name=Path(m.name).name;assert m.name=='source_objects/'+name and name.endswith('.gz')
            h=name[:-3];content=gzip.decompress(t.extractfile(m).read());assert h in expected and hashlib.sha256(content).hexdigest()==h and len(content)==expected[h];found[h]=True
    assert set(found)==set(expected)
    records=[];pops=[]
    for stack,pop in zip(z['stack'],z['populations']):
        records.append({'measurement':{'series':[{'series_id':f'Cu_K_layer_{l}','independent_variables':[{'unit':'eV','values':z['energy'].tolist()}],'channels':[{'name':'absorption_cross_section','values':stack[l-2].tolist()}]} for l in [2,3,4,5]]}});pops.append({l:int(pop[l-2]) for l in [2,3,4,5]})
    oof=np.load(b/'models/oof.npz');results=[]
    for p in sorted((b/'models').glob('*.joblib')):
        model=joblib.load(p);ids=np.array(model['provenance']['heldout_indices'],int);ids=ids if len(ids) else np.arange(len(records));pred,score=predict_records(model,[records[i] for i in ids],[pops[i] for i in ids]);pred2,score2=predict_records(joblib.load(p),[records[i] for i in ids],[pops[i] for i in ids]);assert np.array_equal(pred,pred2) and np.array_equal(score,score2)
        if p.stem.startswith('outer_fold_'):assert np.array_equal(score,oof['score'][ids]) and np.array_equal(pred,oof['prediction'][ids])
        elif p.stem=='new_configuration_holdout':
            h=np.load(b/'models/new_configuration_holdout.npz');assert np.array_equal(ids,h['indices']) and np.array_equal(score,h['score']) and np.array_equal(pred,h['prediction'])
        results.append({'model':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'evaluated_rows':len(ids),'shared_predict_records_replay_identical':True,'deployment_not_heldout':p.stem=='deployment_full_cohort'})
    expected_metrics=json.loads((b/'models/metrics.json').read_text())['all_updated_configurations'];assert f1_metrics(oof['truth'],oof['prediction'])==expected_metrics
    print(json.dumps({'models_verified':len(results),'all_identical':True,'snapshot_files_verified':len(checks),'source_files_verified':len(source),'unique_source_objects_verified':len(found),'metrics':expected_metrics,'models':results},indent=2))
if __name__=='__main__':main()
