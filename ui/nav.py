"""
ui/nav.py
---------
Navigation helpers that never crash a page.

st.page_link / st.switch_page raise an error if the target page can't be found (for example if a
page file is renamed, or the app is started from a different entry file). A missing link is a
small problem; a crashed results screen is a big one. So we catch it and say where to click instead.
"""
import streamlit as st


def link(path: str, label: str, icon: str | None = None) -> None:
    try:
        st.page_link(path, label=label, icon=icon)
    except Exception:  # StreamlitPageNotFoundError and friends
        st.caption(f"{icon or ''} {label.rstrip(' →')}: open it from the sidebar.".strip())


def go(path: str, fallback: str) -> None:
    """Jump to a page. If that is impossible, show `fallback` instead of crashing.
    (A successful jump works by raising a special control-flow exception that this `except Exception` lets through.)"""
    try:
        st.switch_page(path)
    except Exception:
        st.success(fallback)
