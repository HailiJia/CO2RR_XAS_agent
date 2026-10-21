"""Read-only reproduction of the corrected final-convolution population audit."""
from pathlib import Path
import json,re
ROOT=Path('/Users/hjia/Downloads/CO2RR')
for name in ['interface_biased','interior_biased']:
    for layer in [4,5]:
        folder=ROOT/'CO/CO_0p11'/name/'XAS/FDMNES/green'/f'layer_{layer}'
        sites=re.findall(r'Gamma_hole\s*=.*?E_cut\s*=\s*([-\d.]+).*?site\s+(\d+)',(folder/'fdmnes.out').read_text())
        assert [int(s) for c,s in sites]==list(range(1,37))
        print(name,layer,'36 sites in final convolution')
