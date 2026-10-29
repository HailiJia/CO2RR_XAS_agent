"""Explicit audited species-presence route, separate from the generic advisor."""
import io,json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from tools.audited_training import run_audited_training

def render():
    st.subheader('Audited adsorbate presence (multilabel)')
    st.write('CO + H has two positive outputs. Counts, geometry and names are never spectral predictors. Archived recipes require a verified export. Updated export v3 audits the local data, freezes a versioned snapshot and preserves matched prior folds.')
    st.info('Historical automatic-depth selector micro/macro F1: 0.7791 / 0.6383; these do not update the fixed-top-two benchmark. The post-hoc fixed winner (0.7978 / 0.6597) is exploratory. Runs below are new integration recipes and do not replace those results.')
    root=st.text_input('Local data or verified export directory')
    recipe=st.radio('Training protocol',['archived_primary','training_only_global_threshold','scientific_extensions_v2', 'motif_holdout_v2', 'updated_data_v3', 'fixed_top_two_v4', 'coverage_v1', 'differential_v1', 'dataset_refresh_v1'],format_func=lambda x: 'Archived nested selector (0.7791 / 0.6383)' if x=='archived_primary' else 'Scientific extensions v2: nested coadsorbate/feature/robustness study' if x=='scientific_extensions_v2' else 'Leave-local-motif-out v2: separate generalization protocol' if x=='motif_holdout_v2' else 'Historical auto-depth v3 (not fixed top-two)' if x=='updated_data_v3' else 'Dataset refresh: fixed top-two on a prepared audited export' if x=='dataset_refresh_v1' else 'Paired differential v1: explicit clean reference required' if x=='differential_v1' else 'CO coverage v1: audit and separate Cu100 transfer' if x=='coverage_v1' else 'Fixed top-two v4: paired cohorts and regularization' if x=='fixed_top_two_v4' else 'New training-only global-threshold recipe')
    output=st.text_input('New output directory (must be empty)')
    if st.button('Audit, fit and verify saved-model replay',disabled=not(root and output)):
        try:
            result=run_audited_training(root,output,recipe=recipe)
            st.success(f"Saved training run: {result['output_dir']}")
            st.write(result['metrics'])
            st.write('Held-out saved-model replay identical:',result.get('prediction_replay_identical', 'See the saved invocation trace'))
            st.write('Sanitized invocation trace:',result['trace'])
        except Exception as exc:st.error(str(exc))

    st.divider()
    st.subheader('Predict from an explicit clean-reference pair')
    st.write('Use the same catalyst/facet and a compatible Cu intensity and energy basis. Missing references are rejected. Experimental use remains an evaluation of simulation transfer.')
    model_path=st.text_input('Paired model file',key='paired_model_path')
    ads_path=st.text_input('Adsorbed ISAAC record or experimental array JSON',key='paired_ads_path')
    clean_path=st.text_input('Clean-reference ISAAC record or experimental array JSON',key='paired_clean_path')
    domain=st.selectbox('Catalyst domain',['CuAu','Cu100','other'],key='paired_domain')
    reference_domain=st.selectbox('Clean-reference domain',['CuAu','Cu100','other'],key='paired_reference_domain')
    experimental=st.checkbox('Paired experimental evaluation (not validated experimental accuracy)',key='paired_experimental')
    convention_text=st.text_area('Experimental calibration/population conventions JSON, if applicable',value='{}',key='paired_conventions') if experimental else '{}'
    if st.button('Check pair and predict',disabled=not(model_path and ads_path and clean_path)):
        try:
            from tools.differential_route import infer
            a=json.loads(Path(ads_path).read_text());b=json.loads(Path(clean_path).read_text())
            if experimental:
                convention=json.loads(convention_text)
                convention.update(sample_energy_eV=a['energy_eV'],reference_energy_eV=b['energy_eV'])
                result=infer(model_path,a['intensity'],b['intensity'],{}, {},domain,reference_domain,experimental_convention=convention)
            else:
                p={int(k):v for k,v in a['geometry_provenance']['top_two_populations'].items()};q={int(k):v for k,v in b['geometry_provenance']['top_two_populations'].items()}
                result=infer(model_path,a,b,p,q,domain,reference_domain)
            st.json(result)
        except Exception as exc:st.error(str(exc))
