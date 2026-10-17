"""
ui/session.py
-------------
Streamlit-specific helpers: login state, page guards, sidebar.

Why a separate `ui/` package? `core/` must stay free of Streamlit. Anything
that touches st.session_state or st.stop() belongs here.

IMPORTANT concept: Streamlit runs each page script on the SERVER. So a guard
like require_admin() is real security: a student can't skip it by editing
their browser. (Merely hiding a menu item would NOT be security.)
"""
import os

import streamlit as st

from core.auth import CurrentUser

SESSION_KEY = "current_user"


# ---------- Page setup shared by EVERY page ----------
def _bridge_secrets() -> None:
    """Copy Streamlit Cloud secrets into environment variables for core/.

    Lives here (not only in app.py) because on a multipage app a visitor can
    land directly on /Admin and app.py would never run.
    """
    try:
        for key in ("DATABASE_URL", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", "TUTOR_DAILY_EXPLANATIONS",
                    "TUTOR_DAILY_MESSAGES", "TUTOR_GLOBAL_DAILY_TOKENS", "TUTOR_WEB_SEARCH",
                    "TUTOR_DAILY_WEB_SEARCHES", "TUTOR_GLOBAL_DAILY_WEB_SEARCHES", "TUTOR_WEB_MAX_USES",
                    "TUTOR_WEB_DOMAINS", "NOTES_WEAK_SCORE"):
            if key in st.secrets:
                os.environ.setdefault(key, st.secrets[key])
    except Exception:
        pass  # no secrets file in local development: that's fine


@st.cache_resource  # runs once per server process, not on every click
def _ensure_tables() -> bool:
    from core.db import get_engine
    from core.orm import Base
    from core.security import ensure_rls

    engine = get_engine()
    # create_all adds NEW tables but never alters existing ones. (Changing an
    # existing table needs a migration tool like Alembic: a later lesson.)
    Base.metadata.create_all(engine)
    # Lock down any table that doesn't have Row Level Security yet (Supabase).
    ensure_rls(engine)
    return True


def init_page(title: str, icon: str = "🎯", layout: str = "wide") -> None:
    """Call this FIRST on every page."""
    st.set_page_config(page_title=f"{title} | AdaptPrep", page_icon=icon, layout=layout)
    _bridge_secrets()
    _ensure_tables()
    render_sidebar()


# ---------- Login state ----------
def get_current_user() -> CurrentUser | None:
    return st.session_state.get(SESSION_KEY)


def login(user: CurrentUser) -> None:
    st.session_state[SESSION_KEY] = user


def logout() -> None:
    st.session_state.pop(SESSION_KEY, None)


def render_sidebar() -> None:
    user = get_current_user()
    if user is None:
        st.sidebar.info("You are not logged in.")
        return
    st.sidebar.write(f"**{user.full_name}**")
    st.sidebar.caption(f"{user.email} · {user.role.value}")
    if st.sidebar.button("Log out"):
        logout()
        st.rerun()


# ---------- Guards ----------
def require_login() -> CurrentUser:
    """Use at the top of a page. Stops the page if nobody is logged in."""
    user = get_current_user()
    if user is None:
        st.warning("Please log in from the Home page to view this page.")
        st.stop()  # nothing below this line runs
    return user


def require_admin() -> CurrentUser:
    """Like require_login, but only admins get through."""
    user = require_login()
    if not user.is_admin:
        st.error("You don't have permission to view this page.")
        st.stop()
    return user
