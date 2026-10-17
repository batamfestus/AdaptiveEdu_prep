"""
ui/tutor_panel.py
-----------------
Everything the student SEES of the AI tutor: the "Explain with AI" card and the saved chat.
All the rules (limits, caching, grounding, ownership) live in core/tutor.py.

Two expensive objects are created ONCE per server process with st.cache_resource:
  * the knowledge base (it builds the vector index from the notes), and
  * the Claude client.
(Tests replace get_llm / get_kb with fakes, so call them through this module.)
"""
from pathlib import Path

import streamlit as st

from core.cbt import CbtError
from core.config import BASE_DIR, get_anthropic_key, get_anthropic_model
from core.db import get_session
from core.knowledge import KnowledgeBase, KnowledgeError, load_chunks
from core.llm import AnthropicClient, LLMError
from core.tutor import (
    MAX_MESSAGE_CHARS, TutorError, ask_tutor, explain_answer, find_thread, get_or_create_thread, get_thread,
    web_search_status,
)

KNOWLEDGE_DIR = BASE_DIR / "data" / "knowledge"
FRIENDLY_ERRORS = (TutorError, LLMError, CbtError)


@st.cache_resource(show_spinner="Loading the study notes…")
def get_kb() -> KnowledgeBase | None:
    try:
        return KnowledgeBase(load_chunks(Path(KNOWLEDGE_DIR)))
    except (KnowledgeError, OSError):
        return None  # the tutor still works, just without retrieved notes


@st.cache_resource
def get_llm():
    key = get_anthropic_key()
    return AnthropicClient(key, get_anthropic_model()) if key else None


# ---------------------------------------------------------------- explanation card
def _show_explanation(e) -> None:
    st.markdown("##### 🤖 AI explanation")
    st.markdown(e.explanation)
    if e.why_wrong:
        st.markdown(f"**Why your answer was wrong:** {e.why_wrong}")
    if e.misconception:
        st.warning(f"**Possible misconception:** {e.misconception}")
    st.success(f"**Next step:** {e.study_tip}")
    if e.sources:
        with st.expander("Study notes used"):
            for s in e.sources:
                st.caption(f"**{s.competency}** — {s.snippet}")
    if not e.notes_sufficient:
        st.info("📚 Your study notes don't fully cover this one. Ask a follow-up question below and tick the "
                "web search option, and the tutor can look it up on trusted sites.")
    if e.needs_review:
        st.caption("🔎 The tutor wasn't fully sure about this question, so it has been flagged for your teacher to check.")
    st.caption("AI can make mistakes. Check important points with your teacher or textbook."
               + (" · saved explanation" if e.cached else ""))


# ---------------------------------------------------------------- chat
def _show_chat(thread_id: int, user, key: str) -> None:
    """Show a saved conversation and a box to continue it."""
    try:
        with get_session() as db:
            thread = get_thread(db, user, thread_id)
    except FRIENDLY_ERRORS as e:
        st.error(str(e))
        return
    for message in thread.messages:
        with st.chat_message("user" if message.role == "user" else "assistant"):
            st.markdown(message.content)

    llm = get_llm()
    if llm is None:
        st.caption("The AI tutor is not available right now.")
        return
    # Consent for web search is the STUDENT'S choice, made per conversation view, and it is off by default.
    allow_web = False
    with get_session() as db:
        web = web_search_status(db, user, llm)
    if web.enabled:
        allow_web = st.checkbox(
            "🌐 If my notes don't cover this, let the tutor search trusted websites", key=f"web_{key}",
            help="The tutor only looks at a short list of trusted education sites, and only when your study notes "
                 "don't cover your question. Web answers are not checked by your teacher.")
        st.caption(f"{web.remaining} web search(es) left today · please don't type personal information")
    with st.form(f"ask_{key}", clear_on_submit=True):
        text = st.text_input("Ask a follow-up question", max_chars=MAX_MESSAGE_CHARS, key=f"text_{key}")
        sent = st.form_submit_button("Send")
    if sent:
        with st.spinner("The tutor is thinking…"):
            try:
                with get_session() as db:
                    ask_tutor(db, user, thread_id, text, llm=llm, kb=get_kb(), allow_web=allow_web)
            except FRIENDLY_ERRORS as e:
                st.error(str(e))
                return
        st.rerun()


def render_thread(thread_id: int, user) -> None:
    """Used by the AI Tutor history page."""
    _show_chat(thread_id, user, key=f"thread_{thread_id}")


# ---------------------------------------------------------------- the part shown under each reviewed question
def render_for_item(session_id: int, position: int, user) -> None:
    """Explain button + follow-up chat for one question in a finished test."""
    if get_llm() is None:
        st.caption("🤖 The AI tutor is not switched on yet.")
        return

    key = f"{session_id}_{position}"
    state_key = f"ai_explanation_{key}"
    if st.button("🤖 Explain with AI", key=f"ai_btn_{key}"):
        with st.spinner("The tutor is thinking…"):
            try:
                with get_session() as db:
                    st.session_state[state_key] = explain_answer(
                        db, user, session_id, position, llm=get_llm(), kb=get_kb())
            except FRIENDLY_ERRORS as e:
                st.session_state.pop(state_key, None)
                st.error(str(e))
    if state_key in st.session_state:
        _show_explanation(st.session_state[state_key])

    chat_key = f"chat_open_{key}"
    if st.button("💬 Ask a follow-up question", key=f"chat_btn_{key}"):
        st.session_state[chat_key] = True
    if st.session_state.get(chat_key):
        try:
            with get_session() as db:
                thread_id = find_thread(db, user, session_id, position)
                if thread_id is None:
                    thread_id = get_or_create_thread(db, user, session_id, position)
        except FRIENDLY_ERRORS as e:
            st.error(str(e))
            return
        _show_chat(thread_id, user, key=key)
