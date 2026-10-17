"""
Student page: take a timed test (CBT).

Three screens live on this page, chosen by the state saved in the DATABASE:
    1. no test running   -> the "start a test" form
    2. a test running    -> the test runner (timer, navigator, question)
    3. just submitted    -> the graded result

Streamlit re-runs this whole script on every click, so NOTHING important lives
only in the browser: every answer is saved to the database the moment it is chosen.
All rules (ownership, deadline, grading) live in core/cbt.py.
"""
import math
from datetime import datetime, timezone

import streamlit as st

from core.cbt import (
    MAX_QUESTIONS, QUESTION_COUNTS, SECONDS_PER_QUESTION, CbtError, TimeUp,
    abandon_test, available_counts, check_active, load_active_test, save_answer,
    set_flag, start_test, submit_test, get_result,
)
from core.db import get_session
from ui.components import image_bytes, show_image
from ui.results import render_result
from ui.session import init_page, require_login

init_page("Take a Test", icon="📝")
user = require_login()

st.title("📝 Take a test")


# ============================================================ screen 3: result
def show_result_screen(session_id: int) -> None:
    notice = st.session_state.pop("result_notice", None)
    if notice:
        st.warning(notice)
    try:
        with get_session() as db:
            result = get_result(db, user, session_id)
    except CbtError as e:
        st.session_state.pop("show_result_id", None)
        st.error(str(e))
        return
    render_result(result)
    if st.button("Start another test", type="primary"):
        st.session_state.pop("show_result_id", None)
        st.rerun()


# ============================================================ screen 1: start form
def show_start_form() -> None:
    with get_session() as db:
        counts = available_counts(db)
    if not counts:
        st.info("There are no published questions yet. Please check back soon.")
        return

    kinds = {
        "Adaptive practice (recommended)": "adaptive",
        "Diagnostic test (find my level)": "diagnostic",
        "Mixed (all topics)": "mixed",
        "One topic": "topic",
    }
    descriptions = {
        "adaptive": "Questions chosen for YOU: your weak spots, topics worth a lot in the exam, and ones we need to know more about.",
        "diagnostic": "A first look at every competency (about two questions on each), so we can work out where you stand.",
        "mixed": "A random mix from every topic.",
        "topic": "Questions from one topic only.",
    }
    mode = kinds[st.radio("Test type", list(kinds), horizontal=True)]
    st.caption(descriptions[mode])

    topic = None
    if mode == "topic":
        topic = st.selectbox("Topic", list(counts), format_func=lambda t: f"{t}  ({counts[t]} questions)")
    available = counts[topic] if topic else sum(counts.values())

    if mode == "diagnostic":
        n = MAX_QUESTIONS            # the engine picks about two per competency, up to this limit
        st.info(f"You get **{SECONDS_PER_QUESTION} seconds per question**. The clock runs on our server, so it keeps counting "
                "even if you close the page. Your answers are saved as you go.")
    else:
        choices = sorted({c for c in QUESTION_COUNTS if c <= available} | {min(available, MAX_QUESTIONS)})
        n = st.selectbox("Number of questions", choices, index=min(1, len(choices) - 1) if mode == "adaptive" else len(choices) - 1)
        minutes = math.ceil(n * SECONDS_PER_QUESTION / 60)
        st.info(f"You will have **{minutes} minutes**. The clock runs on our server, so it keeps counting "
                "even if you close the page. Your answers are saved as you go.")

    if st.button("Start test", type="primary"):
        try:
            with get_session() as db:
                session_id = start_test(db, user, mode, topic, n)
        except CbtError as e:
            st.error(str(e))
            return
        st.session_state[f"pos_{session_id}"] = 0
        st.rerun()


# ============================================================ screen 2: the test runner
@st.fragment(run_every=1)
def countdown(deadline: datetime) -> None:
    """Redraws ONLY itself, once a second. It does no database work: the remaining
    time is just (deadline - now). When it reaches zero it reruns the whole page,
    and the server then grades the test (see load_active_test)."""
    remaining = int((deadline - datetime.now(timezone.utc)).total_seconds())
    if remaining <= 0:
        st.rerun()  # scope="app" by default: the whole page reloads
    minutes, seconds = divmod(remaining, 60)
    if remaining <= 300:
        st.error(f"⏱️ {minutes:02d}:{seconds:02d}")  # last 5 minutes: red
    else:
        st.info(f"⏱️ {minutes:02d}:{seconds:02d}")


# ---- callbacks: Streamlit runs these BEFORE the page reruns, when a widget changes ----
def _remember_error(e: Exception) -> None:
    st.session_state["cbt_error"] = str(e)


def _on_answer(item_id: int, widget_key: str) -> None:
    try:
        with get_session() as db:
            save_answer(db, user, item_id, st.session_state.get(widget_key))
    except CbtError as e:      # includes TimeUp: the next rerun will show the result
        _remember_error(e)


def _clear_answer(item_id: int, widget_key: str) -> None:
    st.session_state.pop(widget_key, None)
    try:
        with get_session() as db:
            save_answer(db, user, item_id, None)
    except CbtError as e:
        _remember_error(e)


def _on_flag(item_id: int, widget_key: str) -> None:
    try:
        with get_session() as db:
            set_flag(db, user, item_id, bool(st.session_state.get(widget_key)))
    except CbtError as e:
        _remember_error(e)


def _goto(pos_key: str, position: int) -> None:
    st.session_state[pos_key] = position


def run_test(session_id: int) -> None:
    try:
        with get_session() as db:
            exam = load_active_test(db, user, session_id)
    except TimeUp as e:  # the server just graded it
        st.session_state["show_result_id"] = e.session_id
        st.session_state["result_notice"] = str(e)
        st.rerun()

    items = exam.items
    n = len(items)
    pos_key = f"pos_{session_id}"
    pos = min(max(st.session_state.get(pos_key, 0), 0), n - 1)
    item = items[pos]

    error = st.session_state.pop("cbt_error", None)
    if error:
        st.error(error)

    # ---- header: title, progress, countdown ----
    left, right = st.columns([3, 1])
    left.subheader(exam.title)
    answered = sum(1 for it in items if it.selected)
    flagged = sum(1 for it in items if it.flagged)
    left.caption(f"Question {pos + 1} of {n} · answered {answered} · flagged {flagged}")
    with right:
        countdown(exam.deadline_at)

    # ---- navigator: ✅ answered, 🚩 flagged, ⬜ not answered yet ----
    per_row = 10
    for start in range(0, n, per_row):
        cols = st.columns(per_row)
        for i, col in zip(range(start, min(start + per_row, n)), cols):
            it = items[i]
            mark = "🚩" if it.flagged else ("✅" if it.selected else "⬜")
            col.button(f"{mark}{i + 1}", key=f"nav_{session_id}_{i}", on_click=_goto, args=(pos_key, i),
                       type="primary" if i == pos else "secondary")
    st.divider()

    # ---- the question itself ----
    by_role = {img.role: img for img in item.images}
    st.markdown(f"**{item.position}.** {item.text}")
    if "stem" in by_role:
        show_image(by_role["stem"], show_caption=False)

    option_pictures = [k for k in item.options if k in by_role]
    if option_pictures:
        for col, key in zip(st.columns(len(option_pictures)), option_pictures):
            col.image(image_bytes(by_role[key]), caption=f"Option {key}")

    widget_key = f"ans_{item.item_id}"
    keys = list(item.options)
    st.radio(
        "Choose your answer", keys, key=widget_key,
        index=keys.index(item.selected) if item.selected in keys else None,  # the saved answer, from the database
        format_func=lambda k: f"{k}.  {item.options[k]}",
        on_change=_on_answer, args=(item.item_id, widget_key),
        label_visibility="collapsed",
    )
    c1, c2 = st.columns(2)
    flag_key = f"flag_{item.item_id}"
    c1.checkbox("🚩 Flag for review", value=item.flagged, key=flag_key, on_change=_on_flag,
                args=(item.item_id, flag_key))
    c2.button("Clear my answer", key=f"clear_{item.item_id}", on_click=_clear_answer,
              args=(item.item_id, widget_key), disabled=item.selected is None)

    # ---- previous / next ----
    b1, _, b3 = st.columns(3)
    b1.button("◀ Previous", key="prev", disabled=pos == 0, on_click=_goto, args=(pos_key, pos - 1))
    b3.button("Next ▶", key="next", disabled=pos == n - 1, on_click=_goto, args=(pos_key, pos + 1))

    st.divider()

    # ---- submit (two steps, so nobody submits by accident) ----
    confirm_key = f"confirm_{session_id}"
    if not st.session_state.get(confirm_key):
        st.button("Submit test", type="primary", key="submit_start",
                  on_click=lambda: st.session_state.__setitem__(confirm_key, True))
    else:
        unanswered = n - answered
        st.warning(
            f"Submit now? You have **{unanswered}** unanswered question(s) and **{flagged}** flagged. "
            "You cannot change your answers after submitting."
        )
        yes, no = st.columns(2)
        if yes.button("Yes, submit my test", type="primary", key="submit_yes"):
            try:
                with get_session() as db:
                    submit_test(db, user, session_id)
            except CbtError as e:
                st.error(str(e))
            else:
                st.session_state.pop(confirm_key, None)
                st.session_state["show_result_id"] = session_id
                st.rerun()
        no.button("Keep working", key="submit_no",
                  on_click=lambda: st.session_state.__setitem__(confirm_key, False))

    with st.expander("Discard this test"):
        st.caption("A discarded test is never graded and does not count toward your progress.")
        if st.button("Discard test", key="discard"):
            with get_session() as db:
                abandon_test(db, user, session_id)
            st.rerun()


# ============================================================ which screen?
shown = st.session_state.get("show_result_id")
if shown:
    show_result_screen(shown)
    st.stop()

with get_session() as db:
    active_id, expired_id = check_active(db, user)  # also grades a test that ran out while the student was away

if expired_id:
    st.session_state["show_result_id"] = expired_id
    st.session_state["result_notice"] = "Time ran out while you were away, so your test was submitted automatically."
    st.rerun()
elif active_id is None:
    show_start_form()
else:
    run_test(active_id)
