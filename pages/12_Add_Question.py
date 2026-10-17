"""
Admin page: add ONE question with a form, including diagrams for the question
and/or its options.

Streamlit detail worth learning: the Topic and Competency pickers are OUTSIDE the
form on purpose. Widgets inside st.form don't update the page until the form is
submitted, so the competency list could not react to the chosen topic.
"""
import streamlit as st
from pydantic import ValidationError
from sqlalchemy import select

from core.db import get_session
from core.images import ImageError
from core.orm import Competency, Topic
from core.permissions import PermissionDenied
from core.question_create import ImageUpload, QuestionCreateError, create_question
from core.question_import import format_validation_error
from core.schemas import QuestionIn
from ui.session import init_page, require_admin

init_page("Add Question", icon="➕")
admin = require_admin()

SOURCES = ["original", "past_paper_inspired", "ai_drafted"]
IMAGE_TYPES = ["png", "jpg", "jpeg", "webp"]

st.title("➕ Add a question")
flash = st.session_state.pop("add_flash", None)
if flash:
    st.success(flash)

# Changing n gives every widget a brand-new key, which blanks the form after a successful save
# (but keeps what you typed if there was an error).
n = st.session_state.get("add_q_n", 0)


def key(name: str) -> str:
    return f"{name}_{n}"


with get_session() as db:
    syllabus: dict[str, list[str]] = {}
    for topic_name, comp_name in db.execute(
        select(Topic.name, Competency.name).join(Competency, Competency.topic_id == Topic.id)
        .order_by(Topic.name, Competency.name)
    ):
        syllabus.setdefault(topic_name, []).append(comp_name)
if not syllabus:
    st.error("The syllabus is empty. Run `python -m scripts.init_db` first.")
    st.stop()

topic = st.selectbox("Topic", list(syllabus), key=key("topic"))
competency = st.selectbox("Competency", syllabus[topic], key=key("competency"))

with st.form(f"add_question_form_{n}"):
    text = st.text_area("Question", key=key("text"),
                        help="Formulas go between dollar signs, e.g. $F = ma$.")

    st.markdown("**Options**")
    st.caption("If your options are diagrams, type a short label such as 'See diagram A', "
               "then attach the pictures under 'Diagrams' below.")
    left, right = st.columns(2)
    options = {}
    for i, letter in enumerate("ABCD"):
        options[letter] = (left if i % 2 == 0 else right).text_input(f"Option {letter}", key=key(f"opt_{letter}"))

    correct = st.radio("Correct option", list("ABCD"), horizontal=True, key=key("correct"))
    difficulty = st.slider("Difficulty (1 = easy, 5 = hard)", 1, 5, 2, key=key("difficulty"))
    explanation = st.text_area("Explanation (shown to the student after answering)", key=key("explanation"))
    source = st.selectbox("Source", SOURCES, key=key("source"),
                          help="Be honest. Reviewers see this.")

    st.markdown("**Diagrams (optional)**")
    stem_file = st.file_uploader("Diagram for the question", type=IMAGE_TYPES, key=key("stem_file"))
    stem_alt = st.text_input("Describe the diagram in words", key=key("stem_alt"),
                             help="Helps screen readers, reviewers and the AI tutor. At least 10 characters.")
    with st.expander("Diagrams for the options"):
        option_files, option_alts = {}, {}
        for letter in "ABCD":
            option_files[letter] = st.file_uploader(f"Diagram for option {letter}", type=IMAGE_TYPES,
                                                    key=key(f"file_{letter}"))
            option_alts[letter] = st.text_input(f"Describe option {letter}'s diagram", key=key(f"alt_{letter}"))

    submitted = st.form_submit_button("Save for review", type="primary")

if submitted:
    uploads = []
    if stem_file is not None:
        uploads.append(ImageUpload("stem", stem_file.name, stem_file.getvalue(), stem_alt))
    for letter in "ABCD":
        if option_files[letter] is not None:
            uploads.append(ImageUpload(letter, option_files[letter].name,
                                       option_files[letter].getvalue(), option_alts[letter]))
    try:
        payload = QuestionIn(topic=topic, competency=competency, text=text, options=options,
                             correct_option=correct, difficulty=difficulty,
                             explanation=explanation, source=source)
        with get_session() as db:
            question_id = create_question(db, admin, payload, uploads)
    except ValidationError as e:
        for message in format_validation_error(e):
            st.error(message)
    except (QuestionCreateError, ImageError, PermissionDenied) as e:
        st.error(str(e))
    else:
        st.session_state["add_flash"] = (
            f"Saved as question #{question_id}. Find it in the Review Queue (pending review)."
        )
        st.session_state["add_q_n"] = n + 1
        st.rerun()
