"""Extract exact plotted values from archived OOF and audited records.

No-data mode reuses committed layer_spectra.csv; it does not claim an audit of
raw structures. All model metrics come from saved held-out arrays.
"""
from pathlib import Path
import argparse,json,sys,shutil,hashlib
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from tools.study_replay import replay_oof
from tools.spectral_multilabel import feature_bank,spectrum_layers
from tools.component_audit import read_poscar

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--data-root');p.add_argument('--output-dir',default=str(ROOT/'study/figure_data'));p.add_argument('--analysis-dir',default=str(ROOT/'CO2RR_ML/updated_dataset_20261020'));a=p.parse_args(argv)
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);base=Path(a.analysis_dir)
    replay_oof(base,out/'replay')
    for src,dest in [('figure1_combination_counts.csv','inventory_combinations.csv'),('layer_sensitivity.csv','depth_performance.csv')]:shutil.copy2(base/src,out/dest)
    per=pd.read_csv(out/'replay/recomputed_per_species.csv');per[per.workflow=='independent_selected_micro'].to_csv(out/'primary_species_performance.csv',index=False)
    metrics=pd.read_csv(out/'replay/recomputed_metrics.csv');metrics[metrics.workflow.isin(['independent_selected_micro','provisional_best','training_most_common_identity_set'])].to_csv(out/'aggregate_performance.csv',index=False)
    if a.data_root:
        rows=[];split=pd.read_csv(base/'cv_manifest.csv');means={l:[] for l in [2,3,4,5]}
        for item in split.itertuples():
            arrays=spectrum_layers(json.loads((Path(a.data_root)/item.record).read_text()))
            for l in means:means[l].append(feature_bank(arrays[l][:,1],arrays[l][:,0])['normalized'][0])
        for l,values in means.items():
            values=np.array(values)
            for e,m,s in zip(arrays[l][:,0],values.mean(axis=0),values.std(axis=0)):rows.append(dict(layer=l,energy_eV=e,mean=m,configuration_SD=s,n=len(values)))
        pd.DataFrame(rows).to_csv(out/'layer_spectra.csv',index=False)
    if not a.data_root and out.resolve()!=(ROOT/'study/figure_data').resolve():
        shutil.copy2(ROOT/'study/figure_data/layer_spectra.csv',out/'layer_spectra.csv')
    design=ROOT/'study/results/design'
    for name in ['planned_structural_matrix.csv','actual_structural_matrix.csv','matched_quartet.json']:
        if (design/name).exists():shutil.copy2(design/name,out/name)
    if (design/'matched_quartet.json').exists():
        quartet=json.loads((design/'matched_quartet.json').read_text());coords=[]
        for i,row in enumerate(quartet['structures']):
            structure=read_poscar(ROOT/row['input']);h=structure['positions'][:,2];atoms=np.array(structure['atoms']);metals=np.isin(atoms,['Cu','Au']);top_limit=np.sort(h[metals])[-68] # verified 36 Cu + 32 Au for this fixed quartet
            for index,(atom,pos) in enumerate(zip(atoms,structure['positions'])):
                if not metals[index] or h[index]>=top_limit-1e-7:coords.append(dict(panel=i,atom_index=index+1,element=atom,x_A=pos[0],y_A=pos[1],z_A=pos[2]))
        pd.DataFrame(coords).to_csv(out/'quartet_input_coordinates.csv',index=False)
    sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir() if p.is_file() and p.name!='figure_sources.json'}
    provenance=dict(reviewed_commit='71bb0a20b369faba842b413a09544dd66c127a78',source_package_status='Named manuscript.tex/provenance/figure_sources.json package unavailable; reconstructed from accessible two-page outline and benchmark artifacts',figures={'1':'implementation architecture; explicit authorization boundary','2':'155-structure inventory and separate planned controls; counts do not imply matrix completion','3':'87-configuration layer means and configuration SD; reference/C/O panels unmeasured','4':'primary OOF metrics/support, fixed-depth nested metrics, exploratory fixed winner and training-only baseline','5':'input quartet coordinates and C-C distances; matched relaxed spectra unmeasured'},plotted_data_sha256=sources)
    (out/'figure_sources.json').write_text(json.dumps(provenance,indent=2)+'\n')
    prov=ROOT/'study/provenance';prov.mkdir(exist_ok=True);shutil.copy2(out/'figure_sources.json',prov/'figure_sources.json')
if __name__=='__main__':main()
