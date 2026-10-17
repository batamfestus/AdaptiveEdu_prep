"""
Admin page: how much of the real exam is each topic worth?

Readiness is a WEIGHTED average, so these numbers decide how much each topic matters to a
student's score and to what they are told to practise. Set them to match the real JAMB paper
(for example, the number of questions on each topic).
"""
import streamlit as st

from core.db import get_session
from core.permissions import PermissionDenied
from core.syllabus import WeightError, list_topic_weights, set_topic_weights
from ui.session import init_page, require_admin

init_page("Exam Weights", icon="⚖️")
admin = require_admin()

st.title("⚖️ Exam weights")
st.caption("Give each topic a number that reflects how much of the real exam it is worth. Only the proportions matter: "
           "weights of 2 and 1 mean the first topic is worth twice as much as the second. Check the official JAMB "
           "syllabus and past papers; the starter values (all 1) treat every topic equally.")

flash = st.session_state.pop("weights_flash", None)
if flash:
    st.success(flash)

with get_session() as db:
    rows = list_topic_weights(db)

with st.form("weights_form"):
    values = {
        row.id: st.number_input(row.name, min_value=0.0, max_value=100.0, value=float(row.weight), step=0.5,
                                key=f"weight_{row.id}")
        for row in rows
    }
    saved = st.form_submit_button("Save weights", type="primary")

if saved:
    try:
        with get_session() as db:
            set_topic_weights(db, admin, values)
    except (WeightError, PermissionDenied) as e:
        st.error(str(e))
    else:
        st.session_state["weights_flash"] = "Weights saved. Student dashboards use them straight away."
        st.rerun()

st.subheader("Current share of the exam")
st.dataframe([{"Topic": r.name, "Weight": r.weight, "Share of exam": r.share} for r in rows], hide_index=True,
             column_config={"Share of exam": st.column_config.ProgressColumn("Share of exam", min_value=0, max_value=100, format="%.0f%%")})
