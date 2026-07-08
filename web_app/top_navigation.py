from __future__ import annotations

from typing import Literal

import streamlit as st

AgentPage = Literal["calculation", "ml"]


def render_top_navigation(active_page: AgentPage) -> None:
    """Render the app-level top navigation and hide Streamlit's sidebar page list."""
    st.markdown(
        """
<style>
/* Move the app controls closer to the top of the page. */
.block-container {
    padding-top: 1.2rem !important;
}

/* Hide Streamlit's default multipage navigation in the sidebar. */
[data-testid="stSidebarNav"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"],
section[data-testid="stSidebar"] nav[aria-label="Pages"] {
    display: none !important;
}

.co2rr-active-agent {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.95rem;
    font-weight: 750;
    color: rgba(49, 51, 63, 0.74);
    margin: 0 0 0.35rem 0;
}

/* Make Streamlit page links look like full-width product tabs. */
div[data-testid="stPageLink"] {
    width: 100% !important;
}

div[data-testid="stPageLink"] a {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    min-height: 54px !important;
    white-space: nowrap !important;
    overflow: visible !important;
    font-size: 1.16rem !important;
    font-weight: 800 !important;
    line-height: 1.15 !important;
    padding: 0.7rem 1.15rem !important;
    border: 1px solid rgba(49, 51, 63, 0.18) !important;
    border-radius: 0.85rem !important;
    background: rgba(250, 250, 250, 0.90) !important;
    text-decoration: none !important;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.03);
}

div[data-testid="stPageLink"] a:hover {
    border-color: rgba(255, 75, 75, 0.55) !important;
    background: rgba(255, 75, 75, 0.06) !important;
}
</style>
        """.strip(),
        unsafe_allow_html=True,
    )

    active_label = "Calculation Agent" if active_page == "calculation" else "Machine Learning Agent"
    st.markdown(f'<div class="co2rr-active-agent">Active mode: {active_label}</div>', unsafe_allow_html=True)

    nav_col1, nav_col2, nav_spacer = st.columns([3.2, 3.6, 3.2])
    with nav_col1:
        st.page_link("CO2RR_XAS_Agent.py", label="Calculation Agent")
    with nav_col2:
        st.page_link("pages/2_Machine_Learning_Agent.py", label="Machine Learning Agent")
    with nav_spacer:
        st.empty()

    st.divider()


def install_top_navigation(active_page: AgentPage) -> None:
    """Install a hook that renders navigation immediately after page setup.

    Streamlit keeps Python objects alive across reruns. Reset to the original
    set_page_config before wrapping so the navigation is rendered once per page
    run rather than once per historical wrapper.
    """
    original = getattr(st, "_co2rr_original_set_page_config", None)
    if original is None:
        original = st.set_page_config
        setattr(st, "_co2rr_original_set_page_config", original)
    else:
        st.set_page_config = original

    def wrapped_set_page_config(*args, **kwargs):
        result = original(*args, **kwargs)
        render_top_navigation(active_page)
        return result

    st.set_page_config = wrapped_set_page_config
