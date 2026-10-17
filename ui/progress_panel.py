"""
ui/progress_panel.py
--------------------
Small pieces of the progress screens that are shared between pages:
  * the "what this test changed" box shown on a result screen,
  * the study notes for a competency (read from the notes files; no AI, no cost).
"""
import streamlit as st

from core.cbt import CbtError
from core.config import BASE_DIR
from core.db import get_session
from core.knowledge import KnowledgeError, load_chunks
from core.progress import profile_change
from ui.nav import link


@st.cache_resource(show_spinner=False)
def _notes() -> dict[tuple[str, str], str]:
    try:
        return {(c.topic.lower(), c.competency.lower()): c.text for c in load_chunks(BASE_DIR / "data" / "knowledge")}
    except (KnowledgeError, OSError):
        return {}


def study_note(topic: str, competency: str) -> str | None:
    return _notes().get((topic.lower(), competency.lower()))


def render_change(session_id: int, user) -> None:
    """Show how finishing THIS test moved the student's profile (the 'competency update' step of the loop)."""
    try:
        with get_session() as db:
            change = profile_change(db, user, session_id)
    except CbtError:
        return
    before, after = change.before, change.after
    st.markdown("#### 📈 What this test changed")
    if before.attempts == 0:
        st.metric("Expected exam score", f"{after.score:.0f}%")
        st.caption("This was your first test, so there is nothing to compare with yet. "
                   "Each test you finish sharpens this estimate.")
    else:
        st.metric("Expected exam score", f"{after.score:.0f}%", f"{after.score - before.score:+.0f} points")
    for c in change.changes[:6]:
        arrow = "🔼" if c.after > c.before + 0.005 else "🔽" if c.after < c.before - 0.005 else "➖"
        st.markdown(f"{arrow} **{c.competency}** ({c.topic}): {c.before:.0%} → {c.after:.0%} "
                    f"· {c.correct} of {c.answered} correct")
    link("pages/1_My_Dashboard.py", "See my full dashboard and plan →", "📊")
