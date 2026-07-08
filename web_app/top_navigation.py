from __future__ import annotations

from typing import Literal

import streamlit as st

AgentPage = Literal["calculation", "ml"]


def render_top_navigation(active_page: AgentPage) -> None:
    """Render a fixed full-width app header and hide Streamlit's sidebar page list."""
    st.markdown(
        """
<style>
:root {
    --co2rr-topbar-height: 74px;
}

/* Reserve space for the fixed app bar. */
.block-container {
    padding-top: calc(var(--co2rr-topbar-height) + 1.6rem) !important;
}

/* Keep the sidebar visually aligned below the app bar. */
section[data-testid="stSidebar"] > div:first-child {
    padding-top: calc(var(--co2rr-topbar-height) + 1.2rem) !important;
}

/* Hide Streamlit's default multipage navigation in the sidebar. */
[data-testid="stSidebarNav"],
section[data-testid="stSidebar"] [data-testid="stSidebarNav"],
section[data-testid="stSidebar"] nav[aria-label="Pages"] {
    display: none !important;
}

.co2rr-topbar {
    position: fixed;
    top: 0;
    left: 0;
    right: 0;
    height: var(--co2rr-topbar-height);
    z-index: 999999;
    display: grid;
    grid-template-columns: minmax(260px, 390px) 1fr minmax(120px, 220px);
    align-items: center;
    gap: 1.0rem;
    padding: 0 1.45rem;
    background: rgba(255, 255, 255, 0.96);
    border-bottom: 1px solid rgba(49, 51, 63, 0.13);
    box-shadow: 0 2px 14px rgba(15, 23, 42, 0.05);
    backdrop-filter: blur(10px);
}

.co2rr-brand {
    display: flex;
    flex-direction: column;
    gap: 0.05rem;
    min-width: 0;
}

.co2rr-brand-title {
    font-size: 1.28rem;
    font-weight: 850;
    line-height: 1.1;
    color: #262b3a;
    white-space: nowrap;
}

.co2rr-brand-subtitle {
    font-size: 0.76rem;
    font-weight: 650;
    color: rgba(49, 51, 63, 0.58);
    white-space: nowrap;
}

.co2rr-tabbar-wrap {
    display: flex;
    justify-content: center;
    min-width: 0;
}

.co2rr-tabbar {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.32rem;
    border-radius: 999px;
    background: rgba(245, 247, 250, 0.96);
    border: 1px solid rgba(49, 51, 63, 0.14);
}

.co2rr-tab {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 190px;
    height: 42px;
    padding: 0 1.15rem;
    border-radius: 999px;
    color: rgba(39, 43, 58, 0.82) !important;
    font-size: 1.0rem;
    font-weight: 760;
    text-decoration: none !important;
    white-space: nowrap;
    transition: background 0.12s ease, box-shadow 0.12s ease, color 0.12s ease;
}

.co2rr-tab:hover {
    background: rgba(255, 75, 75, 0.08);
    color: #262b3a !important;
}

.co2rr-nav-note {
    display: flex;
    justify-content: flex-end;
    color: rgba(49, 51, 63, 0.50);
    font-size: 0.78rem;
    font-weight: 650;
    white-space: nowrap;
}

@media (max-width: 920px) {
    .co2rr-topbar {
        grid-template-columns: 1fr;
        height: 112px;
        align-items: center;
        padding: 0.7rem 1.0rem;
    }
    :root {
        --co2rr-topbar-height: 112px;
    }
    .co2rr-brand, .co2rr-nav-note {
        display: none;
    }
    .co2rr-tab {
        min-width: 150px;
        font-size: 0.92rem;
    }
}
</style>
<div class="co2rr-topbar">
  <div class="co2rr-brand">
    <div class="co2rr-brand-title">CO2RR XAS Agent</div>
    <div class="co2rr-brand-subtitle">Agentic XAS workflow for ISAAC</div>
  </div>
  <div class="co2rr-tabbar-wrap">
    <div class="co2rr-tabbar">
      <a class="co2rr-tab" href="./" target="_self">Calculation Agent</a>
      <a class="co2rr-tab" href="./Machine_learning_for_XAS" target="_self">Machine Learning Agent</a>
    </div>
  </div>
  <div class="co2rr-nav-note">Agent mode</div>
</div>
        """.strip(),
        unsafe_allow_html=True,
    )


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
