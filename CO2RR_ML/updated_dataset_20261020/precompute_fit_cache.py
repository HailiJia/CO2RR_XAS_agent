"""Populate the notebook's deterministic fit cache in four local worker processes.
The full scientific workflow remains embedded in analysis.py / the notebook.
"""
from pathlib import Path
import os,time,multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed
BASE=Path(__file__).resolve().parent
os.chdir(BASE)
source=(BASE/'analysis.py').read_text().split('# %% TRAINING')[0]
exec(compile(source,str(BASE/'analysis.py'),'exec'),globals())
training=(BASE/'analysis.py').read_text().split('# %% TRAINING\n',1)[1].split('all_selections=[];all_fixed=[];start=time.time()',1)[0]
exec(compile(training,'training functions','exec'),globals())
def compute(task):
    fold,candidate=task;spec=CANDIDATES[candidate];tr,te=FOLDS[fold];path=CACHE/f'{TAG}_{fold}_{candidate}.npz'
    if path.exists():return False
    inside=np.zeros((len(tr),Y.shape[1]))
    for a,b in inner_folds(Y[tr],fold):inside[b]=fit_scores(spec,tr[a],tr[b])
    outside=fit_scores(spec,tr,te)
    np.savez_compressed(path,inner_score=inside,test_score=outside,train_indices=tr,test_indices=te)
    return True
if __name__=='__main__':
    start=time.time();tasks=[(fold,spec['id']) for fold in range(5) for spec in CANDIDATES]
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('fork')) as pool:
        futures=[pool.submit(compute,task) for task in tasks]
        for i,future in enumerate(as_completed(futures),1):
            future.result()
            if i%25==0 or i==len(tasks):print(f'Fit caches {i}/{len(tasks)}; elapsed {(time.time()-start)/60:.1f} min',flush=True)
