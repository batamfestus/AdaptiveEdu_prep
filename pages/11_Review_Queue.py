"""
Admin page: the review queue. Approve, reject, publish and retire questions,
and fix diagrams while a question is still editable.

Every button calls core/, which enforces the rules and writes an audit entry.
"""
import streamlit as st
from sqlalchemy import select

from core.db import get_session
from core.images import VALID_ROLES, ImageError, remove_question_image, set_question_image
from core.orm import EDITABLE_STATUSES, QuestionStatus, Topic
from core.permissions import PermissionDenied
from core.review import (
    MIN_NOTE_LENGTH, ReviewError, available_actions, count_by_status,
    edit_explanation, list_questions, transition_question,
)
from ui.components import render_question
from ui.session import init_page, require_admin

init_page("Review Queue", icon="✅")
admin = require_admin()

ACTION_LABELS = {
    "submit": "Submit for review", "approve": "✅ Approve", "reject": "↩️ Reject (return to draft)",
    "publish": "🚀 Publish", "send_back": "↩️ Send back for review",
    "retire": "🗄️ Retire", "restore": "♻️ Restore for review",
}
DONE_LABELS = {
    "submit": "submitted", "approve": "approved", "reject": "returned to draft",
    "publish": "published", "send_back": "sent back", "retire": "retired", "restore": "restored",
}
TABS = [
    ("Pending review", QuestionStatus.PENDING_REVIEW), ("Approved", QuestionStatus.APPROVED),
    ("Published", QuestionStatus.PUBLISHED), ("Returned (draft)", QuestionStatus.DRAFT),
    ("Retired", QuestionStatus.RETIRED),
]
ROLE_LABELS = {"stem": "The question itself", "A": "Option A", "B": "Option B", "C": "Option C", "D": "Option D"}

st.title("✅ Review queue")
flash = st.session_state.pop("review_flash", None)
if flash:
    st.success(flash)

with get_session() as db:
    counts = count_by_status(db)
    topic_names = list(db.scalars(select(Topic.name).order_by(Topic.name)))
topic_filter = st.selectbox("Filter by topic", ["All topics"] + topic_names)
topic_name = None if topic_filter == "All topics" else topic_filter


def diagram_tools(q) -> str | None:
    """Add, replace or remove diagrams while the question is editable. Returns an error message, if any."""
    error = None
    with st.expander("🖼️ Diagrams"):
        for img in q.images:
            left, right = st.columns([4, 1])
            left.caption(f"{ROLE_LABELS[img.role]}: {img.alt_text}")
            if right.button("Remove", key=f"rmimg_{img.id}"):
                try:
                    with get_session() as db:
                        remove_question_image(db, img.id, admin)
                    st.session_state["review_flash"] = f"Removed a diagram from question #{q.id}."
                    st.rerun()
                except (ImageError, PermissionDenied) as e:
                    error = str(e)

        n = st.session_state.get(f"imgn_{q.id}", 0)  # changing n empties the widgets after a save
        role = st.selectbox("Attach to", list(VALID_ROLES), format_func=ROLE_LABELS.get, key=f"imgrole_{q.id}_{n}")
        file = st.file_uploader("Image (replaces any existing image in that position)",
                                type=["png", "jpg", "jpeg", "webp"], key=f"imgfile_{q.id}_{n}")
        alt = st.text_input("Describe the image in words", key=f"imgalt_{q.id}_{n}")
        if st.button("Save diagram", key=f"imgsave_{q.id}_{n}"):
            if file is None:
                error = "Choose an image file first."
            else:
                try:
                    with get_session() as db:
                        set_question_image(db, q.id, admin, role, file.getvalue(), alt, file.name)
                    st.session_state["review_flash"] = f"Diagram saved on question #{q.id}."
                    st.session_state[f"imgn_{q.id}"] = n + 1
                    st.rerun()
                except (ImageError, PermissionDenied) as e:
                    error = str(e)
    return error


def show_question(q) -> None:
    badge = " 🖼️" if q.images else ""
    title = f"#{q.id} · {q.topic} › {q.competency} · difficulty {q.difficulty}{badge}"
    with st.expander(title):
        render_question(q.text, q.options, q.correct_option, q.explanation, q.images)
        st.caption(f"Source: {q.source} · Added by {q.created_by_name or 'unknown'} · version {q.version}")

        if q.history:
            with st.container(border=True):
                st.caption("History")
                for h in q.history[-5:]:
                    note = f": {h.note}" if h.note else ""
                    st.caption(f"{h.created_at:%d %b %Y %H:%M} · {h.actor_name} · {h.action}{note}")

        error = None
        # ---- editing (only while the question is still editable) ----
        if q.status in EDITABLE_STATUSES:
            new_text = st.text_area("Explanation", value=q.explanation, key=f"expl_{q.id}")
            if st.button("Save explanation", key=f"save_expl_{q.id}"):
                try:
                    with get_session() as db:
                        edit_explanation(db, q.id, admin, new_text)
                    st.session_state["review_flash"] = f"Explanation saved for question #{q.id}."
                    st.rerun()
                except (ReviewError, PermissionDenied) as e:
                    error = str(e)
            error = diagram_tools(q) or error

        # ---- status buttons: only the moves the rules allow ----
        actions = available_actions(q.status)
        note = ""
        if any(a in ("reject", "send_back") for a in actions):
            note = st.text_input(f"Reason (needed to reject or send back, min {MIN_NOTE_LENGTH} characters)",
                                 key=f"note_{q.id}")
        for col, action in zip(st.columns(len(actions)), actions):
            if col.button(ACTION_LABELS[action], key=f"{action}_{q.id}"):
                try:
                    with get_session() as db:
                        transition_question(db, q.id, action, admin, note)
                    st.session_state["review_flash"] = f"Question #{q.id} {DONE_LABELS[action]}."
                    st.rerun()  # works by raising a special exception, so keep the except clauses narrow
                except (ReviewError, PermissionDenied) as e:
                    error = str(e)
        if error:
            st.error(error)


for tab, (label, status) in zip(st.tabs([f"{l} ({counts.get(s, 0)})" for l, s in TABS]), TABS):
    with tab:
        with get_session() as db:
            questions = list_questions(db, status, topic_name)
        if not questions:
            st.caption("Nothing here.")
        for q in questions:
            show_question(q)
