"""Copy pinned repository outputs and extract unchanged measurements for plotting.

Run once: python code/prepare_figure_data.py --repo /path/to/CO2RR_XAS_agent
          --benchmark /path/to/CuO_Kedge_benchmark.pdf
Final plotting uses the bundled figure_data and does not need the raw HPC dataset.
"""
import argparse, hashlib, json, pathlib, shutil, subprocess, tarfile
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--repo', type=pathlib.Path, required=True)
    ap.add_argument('--benchmark', type=pathlib.Path, required=True); args = ap.parse_args()
    dst = ROOT/'figure_data'; dst.mkdir(exist_ok=True); sources = []
    def copy(src, rel):
        out = dst/rel; out.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src,out)
        sources.append(dict(file=rel, source=str(src.relative_to(args.repo)) if src.is_relative_to(args.repo) else src.name,
                            sha256=hashlib.sha256(out.read_bytes()).hexdigest()))
    base = args.repo/'CO2RR_ML/updated_dataset_20261020'
    for n in ['all_unique_structures.csv','figure1_combination_counts.csv','figure1_species_presence.csv',
              'species_support.csv','nested_performance.csv','per_species.csv','provisional_best_per_species.csv',
              'feature_model_depth_performance.csv','layer_sensitivity.csv','paired_depth_comparisons.csv',
              'cv_manifest.csv','fold_counts.csv','layer_population_audit.csv','run_summary.json','environment.json',
              'spectral_identity_contrasts.csv','spectral_difference_peaks.csv','provisional_best_model.json',
              'pure_mixed_recall.csv','oof_independent_selected_micro.npz','oof_provisional_best.npz']:
        copy(base/n,n)
    records = []
    with tarfile.open(base/'records_before_convolution_correction_20261020T215321Z.tar.gz') as t:
        for member in t.getmembers():
            d=json.load(t.extractfile(member)); config=member.name.split('/XAS/')[0]
            for series in d['measurement']['series']:
                for e,y in zip(series['independent_variables'][0]['values'],series['channels'][0]['values']):
                    records.append(dict(configuration=config,layer=int(series['series_id'].split('_')[-1]),energy_eV=e,absorption=y))
    pd.DataFrame(records).to_csv(dst/'representative_layer_spectra.csv',index=False)
    sources.append(dict(file='representative_layer_spectra.csv',source='CO2RR_ML/updated_dataset_20261020/records_before_convolution_correction_20261020T215321Z.tar.gz',
        note='Measurement arrays only, from backed-up records. The population annotations were superseded; current layer_population_audit.csv supplies populations. Spectra were unchanged by the documented correction. Labels are CO in current cv_manifest.csv.'))
    structure_rows=[]
    for family in ['2CO_adjacent','OCCO']:
        for region in ['interface','interior']:
            rel=f'data/CuAu_dataset/{family}/{family}_0p056_{region}/POSCAR'
            name=f'{family}_{region}.vasp'; copy(args.repo/rel,'structures/'+name)
            structure_rows.append(dict(file=name,family=family,region=region,status='input structure; not a verified relaxed matched spectral pair'))
    pd.DataFrame(structure_rows).to_csv(dst/'structure_panels.csv',index=False)
    copy(args.benchmark,'CuO_Kedge_benchmark_source.pdf')
    subprocess.run(['pdftoppm','-scale-to','2000','-singlefile','-png',str(args.benchmark),str(dst/'CuO_benchmark')],check=True)
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=args.repo,text=True).strip()
    (ROOT/'provenance/figure_sources.json').write_text(json.dumps(dict(repository='https://github.com/HailiJia/CO2RR_XAS_agent',commit=sha,files=sources),indent=2)+'\n')
if __name__=='__main__': main()
