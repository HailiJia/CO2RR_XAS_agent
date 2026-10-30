from pathlib import Path
import subprocess,ast,py_compile,hashlib,json,importlib,sys
root=Path.cwd();sys.path.insert(0,str(root));files=subprocess.check_output(['git','ls-files','-z']).decode().split('\0');rows=[]
for name in files:
 if not name.endswith('.py'):continue
 p=root/name;b=p.read_bytes();ast.parse(b,filename=name);py_compile.compile(str(p),cfile='/tmp/pr18_v2_compile/'+name.replace('/','__')+'c',doraise=True)
 blob=subprocess.check_output(['git','hash-object',str(p)]).decode().strip();index=subprocess.check_output(['git','rev-parse',':'+name]).decode().strip();assert blob==index,(name,blob,index)
 rows.append({'path':name,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'git_blob':blob,'index_matches_working_bytes':True})
mods=['workflow.nersc_workflow','agent.co2rr_xas_agent','tools.audited_training','tools.study_design','tools.scientific_extensions','tools.scientific_review','tools.chemical_lineage','tools.controlled_comparisons','tools.scientific_motif','tools.updated_data_training','tools.fixed_top_two','tools.coverage_study','tools.coverage_inference','tools.cu100_lineage','tools.differential_xas','tools.differential_training','tools.differential_route','tools.differential_geometry','tools.differential_robustness','tools.fdmnes_scale','tools.prepare_clean_Cu100','tools.dataset_refresh','tools.dataset_refresh_training','tools.dataset_refresh_sources','tools.differential_identity','tools.identity_comparison','tools.grouped_identity_learning','tools.signed_shape_size','tools.spectral_label_decoder']
for name in mods:importlib.import_module(name)
report={'scope':'every tracked Python file AST+py_compile and index/working byte/hash equality','python_files':len(rows),'imports':mods,'files':rows,'SHA_scope':'pre-commit staged tree; exact pushed clean-checkout SHA receipt follows in PR'}
print(json.dumps(report,indent=2))
