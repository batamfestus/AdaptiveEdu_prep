"""Tests for core/review.py (the review state machine)."""
import pytest
from sqlalchemy import select

from core.orm import Competency, Question, QuestionReview, QuestionStatus as S
from core.permissions import PermissionDenied
from core.review import (
    InvalidTransition, ReviewError, available_actions, count_by_status,
    edit_explanation, list_questions, transition_question,
)


def add_question(db, status=S.PENDING_REVIEW, explanation="I = V/R = 3 A.", text="What current flows?"):
    comp = db.scalars(select(Competency).where(Competency.name == "Apply Ohm's law")).one()
    q = Question(competency_id=comp.id, text=text, options={"A": "1", "B": "2", "C": "3", "D": "4"},
                 correct_option="C", difficulty=2, explanation=explanation, status=status)
    db.add(q)
    db.commit()
    return q


def test_full_happy_path_is_logged(db, seeded, admin):
    q = add_question(db)
    transition_question(db, q.id, "approve", admin)
    transition_question(db, q.id, "publish", admin)
    db.refresh(q)
    assert q.status == S.PUBLISHED and q.reviewed_by == admin.id
    actions = [e.action for e in db.scalars(select(QuestionReview).order_by(QuestionReview.id))]
    assert actions == ["approve", "publish"]


def test_cannot_skip_steps(db, seeded, admin):
    q = add_question(db, status=S.PENDING_REVIEW)
    with pytest.raises(InvalidTransition):
        transition_question(db, q.id, "publish", admin)  # must be approved first
    with pytest.raises(InvalidTransition):
        transition_question(db, q.id, "no_such_action", admin)


def test_reject_needs_a_reason_and_returns_to_draft(db, seeded, admin):
    q = add_question(db)
    with pytest.raises(ReviewError, match="reason"):
        transition_question(db, q.id, "reject", admin, note="no")
    transition_question(db, q.id, "reject", admin, note="Option C is ambiguous")
    db.refresh(q)
    assert q.status == S.DRAFT


def test_cannot_publish_without_an_explanation(db, seeded, admin):
    q = add_question(db, status=S.APPROVED, explanation="")
    with pytest.raises(ReviewError, match="explanation"):
        transition_question(db, q.id, "publish", admin)


def test_student_cannot_review(db, seeded, student):
    q = add_question(db)
    with pytest.raises(PermissionDenied):
        transition_question(db, q.id, "approve", student)
    db.refresh(q)
    assert q.status == S.PENDING_REVIEW


def test_available_actions_follow_the_rules():
    assert available_actions(S.PENDING_REVIEW) == ["approve", "reject"]
    assert available_actions(S.PUBLISHED) == ["retire"]


def test_edit_explanation_bumps_version_and_is_logged(db, seeded, admin):
    q = add_question(db, explanation="")
    edit_explanation(db, q.id, admin, "  Use V = IR.  ")
    db.refresh(q)
    assert q.explanation == "Use V = IR." and q.version == 2
    assert db.scalars(select(QuestionReview)).one().action == "edit"


def test_cannot_edit_approved_or_published(db, seeded, admin):
    q = add_question(db, status=S.PUBLISHED)
    with pytest.raises(ReviewError, match="Only draft or pending"):
        edit_explanation(db, q.id, admin, "changed")


def test_list_and_count(db, seeded, admin):
    add_question(db, S.PENDING_REVIEW, text="first")
    add_question(db, S.PENDING_REVIEW, text="second")
    add_question(db, S.PUBLISHED, text="third")
    pending = list_questions(db, S.PENDING_REVIEW)
    assert [q.text for q in pending] == ["first", "second"]
    assert pending[0].topic == "Current Electricity" and pending[0].competency == "Apply Ohm's law"
    assert list_questions(db, S.PENDING_REVIEW, topic_name="Motion") == []
    assert count_by_status(db) == {S.PENDING_REVIEW: 2, S.PUBLISHED: 1}


def test_history_is_attached_to_listed_questions(db, seeded, admin):
    q = add_question(db)
    transition_question(db, q.id, "reject", admin, note="Needs a clearer stem")
    row = list_questions(db, S.DRAFT)[0]
    assert row.history[0].action == "reject" and row.history[0].actor_name == "The Boss"
