"""Compile the editable package twice with an installed TeX engine."""
from pathlib import Path
import os,shutil,subprocess
B=Path(__file__).resolve().parents[1];P=B/'provenance/rebuild';P.mkdir(parents=True,exist_ok=True)
engine=os.environ.get('TECTONIC_BINARY') or shutil.which('pdflatex') or shutil.which('tectonic')
if not engine:raise SystemExit('Install pdflatex or tectonic, or set TECTONIC_BINARY to an installed executable')
for file in ['manuscript.tex','Supporting_Information.tex']:
 for n in [1,2]:
  args=[engine,'--keep-logs',file] if 'tectonic' in Path(engine).name else [engine,'-interaction=nonstopmode','-halt-on-error',file]
  r=subprocess.run(args,cwd=B,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(P/f'{Path(file).stem}_pass{n}.log').write_text(r.stdout)
  if r.returncode:raise SystemExit(r.stdout)
 print('Compiled twice:',file)
