"""Compile twice and render every page, or record the exact unavailable tool."""
from pathlib import Path
import argparse,json,shutil,subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(Path(__file__).resolve().parents[1]/'manuscript.tex'));p.add_argument('--output-dir',required=True);p.add_argument('--engine',default='pdflatex');a=p.parse_args();out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);source=Path(a.source).resolve();engine=shutil.which(a.engine)
    if engine is None:
        (out/'compile_status.json').write_text(json.dumps(dict(status='blocked',reason='TeX engine '+a.engine+' unavailable',passes_completed=0,original_supplied_manuscript_available=False),indent=2)+'\n');return 2
    passes=[]
    for i in range(2):
        result=subprocess.run([engine,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),str(source)],cwd=source.parent,capture_output=True,text=True)
        (out/f'pass_{i+1}.log').write_text(result.stdout+result.stderr);passes.append(result.returncode)
        if result.returncode:break
    pdf=out/(source.stem+'.pdf');rendered=[]
    if passes==[0,0]:
        import fitz
        with fitz.open(pdf) as doc:
            for i,page in enumerate(doc):
                filename=f'page_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/filename);rendered.append(filename)
    (out/'compile_status.json').write_text(json.dumps(dict(status='compiled' if passes==[0,0] else 'failed',passes=passes,rendered_pages=rendered),indent=2)+'\n');return 0 if passes==[0,0] else 1
if __name__=='__main__':raise SystemExit(main())
