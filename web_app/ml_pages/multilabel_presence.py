"""Explicit audited species-presence route, separate from the generic advisor."""
import io,json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from tools.audited_training import run_audited_training

def render():
    st.subheader('Audited adsorbate presence (multilabel)')
    st.write('CO + H has two positive outputs. Counts, geometry and names are never spectral predictors. This route requires the audited cohort export and frozen folds; the general categorical advisor is a separate workflow.')
    st.info('Archived primary nested-selector micro/macro F1: 0.7791 / 0.6383. The post-hoc fixed winner (0.7978 / 0.6597) is exploratory. Runs below are new integration recipes and do not replace those results.')
    root=st.text_input('Local verified cohort export directory')
    recipe=st.radio('Training protocol',['archived_primary','training_only_global_threshold','scientific_extensions_v2', 'motif_holdout_v2'],format_func=lambda x: 'Archived nested selector (0.7791 / 0.6383)' if x=='archived_primary' else 'Scientific extensions v2: nested coadsorbate/feature/robustness study' if x=='scientific_extensions_v2' else 'Leave-local-motif-out v2: separate generalization protocol' if x=='motif_holdout_v2' else 'New training-only global-threshold recipe')
    output=st.text_input('New output directory (must be empty)')
    if st.button('Audit, fit and verify saved-model replay',disabled=not(root and output)):
        try:
            result=run_audited_training(root,output,recipe=recipe)
            st.success(f"Saved training run: {result['output_dir']}")
            st.write(result['metrics'])
            st.write('Held-out saved-model replay identical:',result['prediction_replay_identical'])
            st.write('Sanitized invocation trace:',result['trace'])
        except Exception as exc:st.error(str(exc))
