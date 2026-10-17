"""Admin home: a snapshot of the question pipeline."""
import streamlit as st
from sqlalchemy import func, select

from core.config import (
    BASE_DIR, get_anthropic_key, get_anthropic_model, tutor_limits, web_domains, web_search_enabled,
)
from core.db import get_session
from core.knowledge import KnowledgeError, coverage, load_chunks
from core.orm import Competency, QuestionStatus, Role, Topic, User
from core.review import count_by_status
from core.tutor import usage_summary
from ui.session import init_page, require_admin

init_page("Admin", icon="🛠️")
admin = require_admin()  # students are stopped here, on the server

st.title("🛠️ Admin")

with get_session() as db:
    n_students = db.scalar(select(func.count(User.id)).where(User.role == Role.STUDENT))
    n_admins = db.scalar(select(func.count(User.id)).where(User.role == Role.ADMIN))
    counts = count_by_status(db)
    ai = usage_summary(db)
    syllabus = {}
    for topic_name, comp_name in db.execute(
        select(Topic.name, Competency.name).join(Competency, Competency.topic_id == Topic.id)
    ):
        syllabus.setdefault(topic_name, []).append(comp_name)

st.subheader("Question pipeline")
labels = [
    ("Pending review", QuestionStatus.PENDING_REVIEW), ("Approved", QuestionStatus.APPROVED),
    ("Published", QuestionStatus.PUBLISHED), ("Returned (draft)", QuestionStatus.DRAFT),
    ("Retired", QuestionStatus.RETIRED),
]
for col, (label, status) in zip(st.columns(len(labels)), labels):
    col.metric(label, counts.get(status, 0))

st.subheader("People")
c1, c2 = st.columns(2)
c1.metric("Students", n_students)
c2.metric("Admins", n_admins)

st.info("Use **Upload Questions** to add questions, then **Review Queue** to approve and publish them.")

# ---------------- AI tutor: cost, credit protection and note coverage ----------------
st.subheader("AI tutor (last 24 hours)")
limits = tutor_limits()
if not get_anthropic_key():
    st.warning("No ANTHROPIC_API_KEY is set, so the AI tutor is switched off.")
a1, a2, a3, a4 = st.columns(4)
a1.metric("Paid AI calls", ai.calls)
a2.metric("Tokens used", f"{ai.input_tokens + ai.output_tokens:,}")
a3.metric("Daily token budget", f"{limits['global_tokens']:,}")
a4.metric("Saved explanations", ai.cached_explanations)
st.caption(f"Model: `{get_anthropic_model()}` · per-student daily limits: {limits['explanations']} new explanations, "
           f"{limits['messages']} chat messages. Saved explanations are shared and free.")
w1, w2, w3, w4 = st.columns(4)
w1.metric("Web search", "ON" if web_search_enabled() else "off")
w2.metric("Web searches", ai.web_searches)
w3.metric("Search fees (approx.)", f"${ai.web_searches * 0.01:.2f}")   # $10 per 1,000 searches
w4.metric("Allowed sites", len(web_domains()))
st.caption(f"Per-student: {limits['web_per_student']} searches a day · app-wide: {limits['web_global']} a day · "
           f"{limits['web_per_message']} per message. Search fees are extra: the pages the tutor reads also use tokens.")
if ai.flagged:
    st.warning("The tutor doubted the answer key on these questions. Please check them:")
    st.dataframe([{"Question": qid, "Starts with": text} for qid, text in ai.flagged], hide_index=True)

st.subheader("Study notes coverage")
try:
    covered, total, missing, orphans = coverage(load_chunks(BASE_DIR / "data" / "knowledge"), syllabus)
    st.metric("Competencies with notes", f"{covered} / {total}")
    if missing:
        st.caption("No notes yet for: " + "; ".join(missing))
    if orphans:
        st.warning("These notes match no syllabus competency (check the spelling): " + "; ".join(orphans))
except KnowledgeError as e:
    st.error(str(e))
