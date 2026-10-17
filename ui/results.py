"""
ui/results.py
-------------
Draws a graded test. Used by BOTH the test page (right after submitting) and the
results page (looking back later), so the two always look the same.
"""
from datetime import datetime, timezone

import streamlit as st

from core.cbt import ExamResult
from ui import progress_panel, tutor_panel
from ui.components import render_review_item
from ui.session import get_current_user


def local_time(moment: datetime) -> datetime:
    """Show times in the visitor's own time zone (we STORE everything in UTC)."""
    try:
        from zoneinfo import ZoneInfo
        return moment.astimezone(ZoneInfo(st.context.timezone))
    except Exception:  # old Streamlit, or no tz database: fall back to UTC
        return moment.astimezone(timezone.utc)


def format_duration(seconds: int) -> str:
    minutes, secs = divmod(seconds, 60)
    return f"{minutes}m {secs:02d}s"


def render_result(result: ExamResult) -> None:
    st.subheader(result.title)
    st.caption(f"Taken on {local_time(result.started_at):%d %b %Y, %H:%M}")
    if result.auto_submitted:
        st.warning("Time ran out, so this test was submitted automatically.")

    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{result.score} / {result.total}")
    c2.metric("Percentage", f"{result.percent:g}%")
    c3.metric("Time used", format_duration(result.seconds_taken))

    progress_panel.render_change(result.session_id, get_current_user())

    st.markdown("**By topic**")
    st.dataframe(
        [{"Topic": t.topic, "Correct": t.correct, "Out of": t.total, "Score %": t.percent} for t in result.by_topic],
        hide_index=True,
    )

    st.markdown("**Review your answers**")
    show = st.radio("Show", ["All questions", "Wrong or unanswered", "Flagged"], horizontal=True,
                    key=f"result_filter_{result.session_id}")
    for item in result.items:
        if show == "Wrong or unanswered" and item.is_correct:
            continue
        if show == "Flagged" and not item.flagged:
            continue
        icon = "✅" if item.is_correct else "❌"
        flag = " 🚩" if item.flagged else ""
        with st.expander(f"{icon} Question {item.position} · {item.topic}{flag}"):
            render_review_item(item)
            st.divider()
            tutor_panel.render_for_item(result.session_id, item.position, get_current_user())
