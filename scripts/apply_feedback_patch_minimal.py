#!/usr/bin/env python3
from pathlib import Path

root = Path(__file__).resolve().parents[1]
main = root / "web_app" / "main.py"
s = main.read_text()

# title / browser tab
s = s.replace('st.set_page_config(page_title="CO2RR XAS Agent", layout="wide")', 'st.set_page_config(page_title="Structure and XAS calculation", layout="wide")')
s = s.replace('st.title("CO2RR XAS Agent")', 'st.title("Structure and XAS calculation")')
s = s.replace('st.header("Workflow workspace")', 'st.header("Structure and XAS calculation")')

# contact-check import
imp_anchor = 'from tools.utils import read_structure_file\n'
imp_add = 'from tools.contact_checks import minimum_pair as element_aware_minimum_pair\n'
if imp_add not in s:
    s = s.replace(imp_anchor, imp_anchor + imp_add, 1)

old = '''    dmin, pair = min_pair_distance(positions)
    if dmin < 1.2:
        st.error(f"Possible severe overlap: minimum distance = {dmin:.3f} Å between atoms {pair}")
    elif dmin < 2.0:
        st.warning(f"Possible close contact: minimum distance = {dmin:.3f} Å between atoms {pair}")
    else:
        st.info(f"Minimum non-PBC atom distance: {dmin:.3f} Å")
'''
new = '''    contact_report = element_aware_minimum_pair(atoms, positions)
    if contact_report["severity"] == "error":
        st.error(contact_report["message"])
    elif contact_report["severity"] == "warning":
        st.warning(contact_report["message"])
    else:
        st.info(contact_report["message"])
'''
if old in s:
    s = s.replace(old, new, 1)

# deterministic full-workflow route, applied before LLM fallback/answer handling
func_anchor = '\ndef _classify_chat_intent_rule_based(message):\n'
route_func = r'''

def _full_workflow_intent_from_text(text):
    low = str(text or "").lower().strip()
    is_full = any(x in low for x in ["full workflow", "workflow package", "relax-to-xas", "relax to xas", "relaxation to xas"])
    if is_full:
        if any(x in low for x in ["cancel", "stop"]):
            return "cancel_full_workflow"
        if any(x in low for x in ["restart", "continue", "resume"]):
            return "restart_full_workflow"
        if any(x in low for x in ["status", "refresh", "check"]):
            return "refresh_full_workflow_status"
        has_upload = any(x in low for x in ["upload", "uploade", "uploaded", "send", "copy"])
        has_start = any(x in low for x in ["start", "run", "submit"])
        has_prepare = any(x in low for x in ["prepare", "generate", "make", "build"])
        if has_upload and has_start:
            return "prepare_upload_start_full_workflow"
        if has_upload:
            return "prepare_upload_full_workflow"
        if has_start:
            return "start_full_workflow"
        if has_prepare:
            return "prepare_full_workflow"
    if "upload" in low and any(x in low for x in ["did", "have", "was", "is", "where", "path", "status"]):
        return "show_paths"
    return None
'''
if route_func.strip() not in s:
    s = s.replace(func_anchor, route_func + func_anchor, 1)

class_anchor = '    low = str(message or "").lower().strip()\n'
class_add = '    full_workflow_override = _full_workflow_intent_from_text(message)\n    if full_workflow_override:\n        return full_workflow_override\n'
if class_add not in s:
    s = s.replace(class_anchor, class_anchor + class_add, 1)

plan_anchor = '        low_text = text.lower()\n'
plan_add = '        full_workflow_override = _full_workflow_intent_from_text(text)\n        if full_workflow_override:\n            intent = full_workflow_override\n'
if plan_add not in s:
    s = s.replace(plan_anchor, plan_anchor + plan_add, 1)

s = s.replace('"nersc_remote_submit_script": "01_structure/submit_relax.sh"', '"nersc_remote_submit_script": "workflow_submit.sh"')
s = s.replace('`change queue to debug`, `upload to NERSC under test03`, or `submit clean surface relaxation job`.', '`change queue to debug`, `prepare full workflow and upload to NERSC under test05`, or `refresh full workflow status`.')
s = s.replace('Try `Generate full CO2RR pathway on Cu`, `generate structure and relaxation files`, `upload to NERSC under test03`, `what files are in that folder`, `change to 8 hrs`, or `submit clean surface relaxation job`.', 'Try `prepare full workflow and upload to NERSC under test05`, `prepare, upload, and start full workflow under test05`, `refresh full workflow status`, `what files are in that folder`, or `change to 8 hrs`.')

main.write_text(s)
print("patched", main)
