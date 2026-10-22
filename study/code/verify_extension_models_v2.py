"""Verify all saved extension fold models against frozen held-out scores."""
from pathlib import Path
import argparse,sys,json,hashlib
import numpy as np,joblib

def main(root):
    root=Path(root);sys.path.insert(0,str(root))
    from tools.scientific_extensions import scores,calibrated
    from tools.audited_training import review_extension_model
    base=root/'study/extensions_v2/ml';data=np.load(base/'cohort_spectra.npz');rows=[]
    for p in sorted(base.glob('*_fold*.joblib')):
        m=joblib.load(p);ids=m['heldout_indices'];expected=np.load(base/('oof_'+p.stem.rsplit('_fold',1)[0]+'.npz'))['score'][ids]
        observed=calibrated(m['calibrators'],scores(m,data['raw'][ids],ids=ids))
        assert np.array_equal(observed,expected),p.name
        assert not set(m['training_indices'])&set(ids)
        rows.append({'model':p.name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'heldout_count':len(ids),'scores_identical':True})
    p=base/'nearest_neighbor_fold0.joblib';m=joblib.load(p);i=int(m['heldout_indices'][0]);review=review_extension_model(p,data['raw'][i:i+1],data['energy'])
    assert review['status']=='success' and review['predictions']
    print(json.dumps({'saved_models_verified':len(rows),'all_identical':True,'shared_review_callable_executed':True,'models':rows},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default=str(Path(__file__).resolve().parents[2]));a=p.parse_args();main(a.root)
