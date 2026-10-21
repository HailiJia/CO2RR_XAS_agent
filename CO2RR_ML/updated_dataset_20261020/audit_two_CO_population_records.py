"""Read-only final-population audit of the recovered CO configurations."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.layer_population import verify_layer_population
p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--split-manifest');p.add_argument('--audit-only',action='store_true');a=p.parse_args();rows=[]
for name in ['interface_biased','interior_biased']:
    for l in [4,5]:
        folder=Path(a.data_root)/'CO/CO_0p11'/name/'XAS/FDMNES/green'/f'layer_{l}'
        row=verify_layer_population(folder,l);assert row['population']==36;rows.append(dict(configuration=name,**row))
out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);(out/'recovered_CO_population.json').write_text(json.dumps(rows,indent=2)+'\n')
print('Four final 36-site convolution populations verified')
