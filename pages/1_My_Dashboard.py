"""
Student dashboard: where am I, how sure are we, and what should I do next?

This page only DRAWS what core/progress.py computes. The ideas behind the numbers
(a belief about each competency, forgetting, exam weights) are explained in core/mastery.py.
"""
from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from core.cbt import CbtError, available_counts, start_test
from core.db import get_session
from core.progress import build_dashboard
from ui.nav import go, link
from ui.progress_panel import study_note
from ui.results import local_time
from ui.session import init_page, require_login

init_page("My Dashboard", icon="📊")
user = require_login()

with get_session() as db:
    dash = build_dashboard(db, user)
    published = sum(available_counts(db).values())
ready = dash.readiness
now = datetime.now(timezone.utc)

st.title("📊 My dashboard")

# ---------------------------------------------------------------- headline
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected exam score", f"{ready.score:.0f}%",
          help="The percentage we expect you to score on the real exam today, based on every answer you have given. "
               "Recent answers count more than old ones, and each topic counts as much as it is worth in the exam.")
c2.metric("Likely range", f"{ready.low:.0f}% to {ready.high:.0f}%", help="We are about 80% sure your score is in this range.")
c3.metric("Questions answered", ready.attempts)
c4.metric("Competencies assessed", f"{ready.assessed} / {ready.total}")

if ready.attempts == 0:
    st.info("👋 Welcome! Take the **Diagnostic test** first. It asks a few questions on every competency, so we can "
            "work out where you stand and build your plan. (Open **Take a Test** in the sidebar.)")
elif ready.provisional:
    st.warning(f"⏳ **Provisional.** This estimate rests on {ready.attempts} answers across {ready.assessed} of "
               f"{ready.total} competencies. A **Diagnostic test** will make it far more reliable.")
else:
    st.caption(f"Confidence in this estimate: **{ready.confidence}**. It becomes more certain with every test you finish.")

# ---------------------------------------------------------------- the plan
st.subheader("🎯 Your plan: what to practise next")
st.caption("We rank competencies by how much you stand to gain: weak spots, topics worth a lot in the exam, "
           "topics we don't know enough about yet, and things you haven't practised for a while.")
for number, item in enumerate(dash.plan, start=1):
    state = item.state
    with st.container(border=True):
        left, right = st.columns([4, 1])
        left.markdown(f"**{number}. {state.meta.name}**  \n{state.meta.topic}")
        left.caption(item.reason)
        right.metric("Mastery", f"{state.mastery:.0%}")
        left.progress(min(max(state.mastery, 0.0), 1.0))
        note = study_note(state.meta.topic, state.meta.name)
        if note:
            with st.expander("📖 Study notes"):
                st.markdown(note)

if published == 0:
    st.info("There are no published questions yet, so practice tests can't start.")
else:
    options = [n for n in (5, 10, 20) if n <= published] or [published]
    count_col, start_col = st.columns([1, 2])       # (named so they can't shadow the go() helper)
    n = count_col.selectbox("Questions", options, index=min(1, len(options) - 1), label_visibility="collapsed")
    if start_col.button(f"🚀 Start adaptive practice ({n} questions)", type="primary"):
        try:
            with get_session() as db:
                session_id = start_test(db, user, "adaptive", None, n)
        except CbtError as e:
            st.error(str(e))
            link("pages/2_Take_a_Test.py", "Go to Take a Test", "📝")
        else:
            st.session_state[f"pos_{session_id}"] = 0
            go("pages/2_Take_a_Test.py", "Your practice test is ready! Open **Take a Test** in the sidebar to begin.")

# ---------------------------------------------------------------- progress over time
if len(dash.history) >= 2:
    st.subheader("📈 Progress over time")
    frame = pd.DataFrame(
        {"Expected score": [p.score for p in dash.history], "Low": [p.low for p in dash.history],
         "High": [p.high for p in dash.history]},
        index=pd.DatetimeIndex([local_time(p.when).replace(tzinfo=None) for p in dash.history], name="Test finished"),
    )
    st.line_chart(frame)
    st.caption("The solid line is your expected exam score after each test; the other two lines are the range we were "
               "80% sure about. The range narrows as we learn more about you.")

# ---------------------------------------------------------------- topics and competencies
st.subheader("🧭 By topic")
st.dataframe(
    [{"Topic": t.topic, "Expected score": t.expected_score, "Share of exam": 100 * t.weight,
      "Assessed": f"{t.assessed} / {t.total}"} for t in dash.topics],
    hide_index=True,
    column_config={
        "Expected score": st.column_config.ProgressColumn("Expected score", min_value=0, max_value=100, format="%.0f%%"),
        "Share of exam": st.column_config.NumberColumn("Share of exam", format="%.0f%%"),
    },
)


def how_sure(uncertainty: float) -> str:
    return "high" if uncertainty <= 0.10 else "medium" if uncertainty <= 0.18 else "not sure yet"


def last_practised(moment) -> str:
    if moment is None:
        return "never"
    days = (now - moment).days
    return "today" if days <= 0 else f"{days} day{'s' if days != 1 else ''} ago"


with st.expander("All competencies"):
    st.dataframe(
        [{"Competency": s.meta.name, "Topic": s.meta.topic, "Mastery": 100 * s.mastery,
          "How sure we are": how_sure(s.uncertainty), "Answers": s.attempts, "Last practised": last_practised(s.last_attempt)}
         for s in sorted(dash.states, key=lambda s: s.mastery)],
        hide_index=True,
        column_config={"Mastery": st.column_config.ProgressColumn("Mastery", min_value=0, max_value=100, format="%.0f%%")},
    )

# ---------------------------------------------------------------- mistakes the AI tutor spotted
if dash.mistakes:
    st.subheader("⚠️ Mistakes to watch out for")
    st.caption("Misconceptions the AI tutor found in your wrong answers.")
    for m in dash.mistakes:
        st.markdown(f"- **{m.competency}**: {m.misconception}" + (f" *(seen {m.times} times)*" if m.times > 1 else ""))

# ---------------------------------------------------------------- recent tests
if dash.recent:
    st.subheader("🕒 Recent tests")
    st.dataframe(
        [{"When": f"{local_time(s.started_at):%d %b, %H:%M}", "Test": s.title, "Score": f"{s.score} / {s.total}", "%": s.percent}
         for s in dash.recent],
        hide_index=True,
    )
    link("pages/3_My_Results.py", "Review my tests and ask the AI tutor →", "📈")
