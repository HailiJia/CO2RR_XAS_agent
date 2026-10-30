"""Compile the authoritative revised manuscript twice and render each page."""
from pathlib import Path
import argparse, json, shutil, subprocess

DEFAULT_SOURCE = Path(__file__).resolve().parents[1] / 'manuscript_package' / 'manuscript.tex'

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',default=str(DEFAULT_SOURCE));p.add_argument('--output-dir',required=True);p.add_argument('--engine',default='pdflatex');a=p.parse_args()
    out=Path(a.output_dir).resolve();out.mkdir(parents=True,exist_ok=True);source=Path(a.source).resolve();engine=shutil.which(a.engine)
    if engine is None:
        status={'status':'blocked','reason':f'TeX engine {a.engine} unavailable','source_available':source.is_file(),'source':source.name,'passes_completed':0,'page_inspection':'not run; no freshly compiled PDF'}
        (out/'compile_status.json').write_text(json.dumps(status,indent=2)+'\n');return 2
    passes=[]
    for i in range(2):
        result=subprocess.run([engine,'-interaction=nonstopmode','-halt-on-error','-output-directory='+str(out),str(source)],cwd=source.parent,capture_output=True,text=True)
        (out/f'pass_{i+1}.log').write_text(result.stdout+result.stderr);passes.append(result.returncode)
        if result.returncode:break
    pdf=out/(source.stem+'.pdf');rendered=[]
    if passes==[0,0]:
        try:
            import fitz
        except ImportError:
            status={'status':'compiled_render_blocked','passes':passes,'pdf':pdf.name,'page_inspection':'blocked; PyMuPDF unavailable'}
        else:
            with fitz.open(pdf) as doc:
                for i,page in enumerate(doc):
                    filename=f'page_{i+1:02d}.png';page.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(out/filename);rendered.append(filename)
            status={'status':'compiled','passes':passes,'pdf':pdf.name,'rendered_pages':rendered,'page_inspection':'rendered every page'}
    else:status={'status':'failed','passes':passes,'source':source.name}
    (out/'compile_status.json').write_text(json.dumps(status,indent=2)+'\n');return 0 if status['status']=='compiled' else 1
if __name__=='__main__':raise SystemExit(main())
