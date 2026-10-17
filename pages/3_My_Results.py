"""Student page: look back at finished tests and review every answer."""
import streamlit as st

from core.cbt import CbtError, get_result, list_sessions
from core.db import get_session
from ui.results import local_time, render_result
from ui.session import init_page, require_login

init_page("My Results", icon="📈")
user = require_login()

st.title("📈 My results")

with get_session() as db:
    sessions = list_sessions(db, user)

if not sessions:
    st.info("You haven't finished a test yet. Open **Take a Test** in the sidebar to start one.")
    st.stop()

st.dataframe(
    [{"Date": f"{local_time(s.started_at):%d %b %Y, %H:%M}", "Test": s.title,
      "Score": f"{s.score} / {s.total}", "%": s.percent, "Ended by timer": "yes" if s.auto_submitted else ""}
     for s in sessions],
    hide_index=True,
)

chosen = st.selectbox(
    "Open a test to review",
    sessions,
    format_func=lambda s: f"{local_time(s.started_at):%d %b %Y, %H:%M} · {s.title} · {s.score}/{s.total}",
)
try:
    with get_session() as db:
        result = get_result(db, user, chosen.id)
except CbtError as e:
    st.error(str(e))
    st.stop()
render_result(result)
