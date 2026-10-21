"""Explicit audited species-presence route, separate from the generic advisor."""
import io,json
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st
from tools.study_cohort import verify_export,load_cohort
from tools.spectral_multilabel import train_cohort_holdout,predict_records

def render():
    st.subheader('Audited adsorbate presence (multilabel)')
    st.write('CO + H has two positive outputs. Counts, geometry and names are never spectral predictors. This route requires the audited cohort export and frozen folds; the general categorical advisor is a separate workflow.')
    st.info('Archived primary nested-selector micro/macro F1: 0.7791 / 0.6383. The post-hoc fixed winner (0.7978 / 0.6597) is exploratory. Runs below are new integration recipes and do not replace those results.')
    root=st.text_input('Local verified cohort export directory')
    fold=st.selectbox('Frozen outer test fold',list(range(5)))
    depth=st.selectbox('Depth representation',['layer_5','layer_4','layer_3','layer_2','top_2','top_3','all_4'])
    features=st.multiselect('Training-only candidate features',['normalized','CDF','derivative','shape_CDF_derivatives','DCT24'],default=['normalized','CDF','derivative'])
    kinds=st.multiselect('Training-only candidate models',['independent_trees','extra_trees','random_forest','logistic','SVM'],default=['independent_trees'])
    if st.button('Audit and fit species-presence model',disabled=not(root and features and kinds)):
        try:
            base=Path(root);verify_export(base)
            cohort=load_cohort(base/'data',base/'splits.csv');result=train_cohort_holdout(cohort,fold,depth,features,kinds)
            buffer=io.BytesIO();joblib.dump(result['bundle'],buffer);loaded=joblib.load(io.BytesIO(buffer.getvalue()))
            ids=result['test_indices'];p,_=predict_records(loaded,[cohort['records'][i] for i in ids],[cohort['populations'][i] for i in ids])
            if not (p==result['prediction']).all():raise ValueError('Export/import prediction mismatch')
            st.write(result['metrics']);st.dataframe(pd.DataFrame(result['ranking']))
            st.write('Species without positive training examples:',result['missing_positive_training'])
            st.download_button('Download fitted multilabel model and recipe',buffer.getvalue(),'co2rr_species_presence.joblib')
            st.download_button('Download held-out predictions',json.dumps(dict(species=result['bundle']['species'],configuration=cohort['split'].iloc[ids].configuration.tolist(),truth=cohort['Y'][ids].tolist(),prediction=p.astype(int).tolist()),indent=2),'heldout_species_presence.json')
        except Exception as exc:st.error(str(exc))
