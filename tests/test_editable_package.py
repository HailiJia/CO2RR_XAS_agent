"""A refreshed editable archive must not hash its own previous manifest."""
import importlib.util,json,hashlib,zipfile
from pathlib import Path

def test_existing_manifest_and_output_zip_are_not_recursively_hashed(tmp_path):
 source=Path(__file__).resolve().parents[1]/'study/manuscript_relative_identity_guarded_v2/code/package_editable.py';spec=importlib.util.spec_from_file_location('editable_package',source);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);module.B=tmp_path
 (tmp_path/'manuscript.tex').write_text('Editable scientific source\n');(tmp_path/'package_manifest.json').write_text('{"package_manifest.json":{"sha256":"stale"}}');output=tmp_path/'editable.zip';output.write_bytes(b'old output');module.build(output)
 m=json.loads((tmp_path/'package_manifest.json').read_text());assert 'package_manifest.json' not in m and 'editable.zip' not in m
 for name,row in m.items():assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest()==row['sha256']
 with zipfile.ZipFile(output) as z:
  assert z.testzip() is None;assert z.read('CO2RR_Reviewed_Manuscript/package_manifest.json')==(tmp_path/'package_manifest.json').read_bytes();assert z.read('CO2RR_Reviewed_Manuscript/manuscript.tex')==(tmp_path/'manuscript.tex').read_bytes()
