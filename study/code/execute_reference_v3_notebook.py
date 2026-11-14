from pathlib import Path
import os
import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output
ROOT=Path(__file__).resolve().parents[2];B=ROOT/'study/reference_benchmark_v3';path=B/'Cu_XAS_revised.ipynb'
notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Numeric Cu/Cu2O/CuO benchmark v3\nOriginal Cu_XAS.ipynb is preserved in reference-v1/raw. This notebook rebuilds the portable numeric processing and all scores. New native Cu2O arrays were independently extracted from matching OUTCAR and HDF5 diagonals. Widths are fixed, not fitted.'),nbformat.v4.new_code_cell("from pathlib import Path\nimport runpy, json, pandas as pd\nrepo=Path.cwd().resolve()\nwhile not (repo/'tools').is_dir():\n    repo=repo.parent\nrunpy.run_path(str(repo/'study/code/process_reference_v3.py'),run_name='__main__')\nrunpy.run_path(str(repo/'study/code/reference_v3_supplement.py'),run_name='__main__')"),nbformat.v4.new_code_cell("settings=json.loads((repo/'study/reference_benchmark_v3/settings.json').read_text())\nprint(json.dumps(settings,indent=2))"),nbformat.v4.new_code_cell("metrics=pd.read_csv(repo/'study/reference_benchmark_v3/similarity_metrics.csv')\nprint(metrics[['material','method','comparison_type','shift_eV','cosine_similarity','Pearson_r','Spearman_rho']].to_string(index=False))"),nbformat.v4.new_code_cell("missing=pd.read_csv(repo/'study/reference_benchmark_v3/missing_series.csv')\nprint(missing.to_string(index=False))\nprint('All available calculations and Experiment use the identical minmax rule and fixed supported scoring window; no extrapolation.')"),nbformat.v4.new_markdown_cell('![Reference comparison and score panels](figures/reference_comparison.png)\n\nThe first-feature view removes per-compound energy differences and is a shape diagnostic; transferable Cu offsets are retained separately. Cu2O/CuO early reference features are measured rather than forcing an arbitrary later peak. Native data, fixed Gaussian stages, offsets/divisors and scored arrays are versioned.')])
shell=InteractiveShell.instance();old=Path.cwd();os.chdir(B)
try:
 for cell in notebook.cells:
  if cell.cell_type!='code':continue
  count=shell.execution_count
  with capture_output() as captured:result=shell.run_cell(cell.source,store_history=True)
  result.raise_error();cell.execution_count=count;cell.outputs=[]
  if captured.stdout:cell.outputs.append(nbformat.v4.new_output('stream',name='stdout',text=captured.stdout))
  if captured.stderr:cell.outputs.append(nbformat.v4.new_output('stream',name='stderr',text=captured.stderr))
 notebook.metadata['execution_route']='All code cells executed in order using IPython, no network kernel';nbformat.write(notebook,path)
finally:os.chdir(old)
print('Executed notebook',path)
