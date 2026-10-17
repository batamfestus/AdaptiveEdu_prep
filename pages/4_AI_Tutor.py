"""Student page: saved AI tutor conversations. Start one from a question in My Results."""
import streamlit as st

from core.db import get_session
from core.tutor import list_threads
from ui import tutor_panel
from ui.results import local_time
from ui.session import init_page, require_login

init_page("AI Tutor", icon="🤖")
user = require_login()

st.title("🤖 My AI tutor chats")
st.caption("Your conversations are saved here. To start a new one, open a finished test in "
           "**My Results**, open a question, and choose **Ask a follow-up question**.")

with get_session() as db:
    threads = list_threads(db, user)
if not threads:
    st.info("No conversations yet.")
    st.stop()

chosen = st.selectbox(
    "Conversation", threads,
    format_func=lambda t: f"{local_time(t.updated_at):%d %b, %H:%M} · {t.title[:70]} ({t.n_messages} messages)",
)
tutor_panel.render_thread(chosen.id, user)
