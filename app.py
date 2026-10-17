"""
app.py - Home page: login / sign-up, then a welcome screen.

Still thin: it draws forms and calls core/auth.py. All the rules
(password strength, duplicate emails, hashing) live in core/.
"""
import streamlit as st
from sqlalchemy import func, select

from core.auth import AuthError, authenticate, register_user
from core.db import get_session
from core.orm import Competency, Question, Topic
from ui.session import get_current_user, init_page, login

init_page("Home")  # must be the first Streamlit call on the page


def render_auth_forms() -> None:
    st.title("🎯 AdaptPrep")
    st.caption("Adaptive JAMB Physics exam preparation, with an AI tutor.")

    login_tab, signup_tab = st.tabs(["Log in", "Create account"])

    with login_tab:
        # st.form batches the inputs: the page only reruns when the button is
        # pressed, not on every keystroke.
        with st.form("login_form"):
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Log in")
        if submitted:
            with get_session() as db:
                user = authenticate(db, email, password)
            if user is None:
                st.error("Invalid email or password.")  # vague on purpose
            else:
                login(user)
                st.rerun()  # redraw the page as a logged-in user

    with signup_tab:
        with st.form("signup_form"):
            name = st.text_input("Full name", key="signup_name")
            email = st.text_input("Email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password", help="At least 8 characters, with a letter and a number.")
            confirm = st.text_input("Confirm password", type="password", key="signup_confirm")
            submitted = st.form_submit_button("Create account")
        if submitted:
            if password != confirm:
                st.error("Passwords do not match.")
            else:
                try:
                    with get_session() as db:
                        user = register_user(db, email, name, password)  # always a STUDENT
                    login(user)
                    st.rerun()
                except AuthError as e:  # safe, user-fixable messages only
                    st.error(str(e))


def render_home(user) -> None:
    st.title(f"Welcome, {user.full_name.split()[0]} 👋")
    if user.is_admin:
        st.info("You are logged in as an admin. Use the Admin page in the sidebar.")

    with get_session() as db:
        n_topics = db.scalar(select(func.count(Topic.id)))
        n_comps = db.scalar(select(func.count(Competency.id)))
        n_questions = db.scalar(select(func.count(Question.id)))

    c1, c2, c3 = st.columns(3)
    c1.metric("Topics", n_topics)
    c2.metric("Competencies", n_comps)
    c3.metric("Questions", n_questions)
    st.caption("Practice tests and your dashboard are coming in later milestones.")


user = get_current_user()
if user is None:
    render_auth_forms()
else:
    render_home(user)
