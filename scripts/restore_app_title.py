#!/usr/bin/env python3
from pathlib import Path
p = Path(__file__).resolve().parents[1] / "web_app" / "main.py"
s = p.read_text()
s = s.replace('st.set_page_config(page_title="Structure and XAS calculation", layout="wide")', 'st.set_page_config(page_title="CO2RR XAS Agent", layout="wide")')
s = s.replace('st.title("Structure and XAS calculation")', 'st.title("CO2RR XAS Agent")')
s = s.replace('st.header("Structure and XAS calculation")', 'st.header("Workflow workspace")')
p.write_text(s)
print("restored CO2RR XAS Agent title in", p)
