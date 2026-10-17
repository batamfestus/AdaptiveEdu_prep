"""
Tests for core/cbt.py. Run with:  pytest -q

Notice how time is handled: we never wait for a real timer. Every function takes
a `now`, so a test can say "it is now 2 hours later" in one line.
"""
import dataclasses
import random
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from core.auth import register_user
from core.cbt import (
    GRACE_SECONDS, SECONDS_PER_QUESTION, CbtError, TimeUp, abandon_test, available_counts,
    check_active, get_result, list_sessions, load_active_test, save_answer, set_flag,
    start_test, submit_test,
)
from core.orm import (
    Competency, ExamItem, ExamSession, Question, QuestionImage, QuestionStatus as Q, SessionStatus, Topic,
)

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def minutes(m: float) -> timedelta:
    return timedelta(minutes=m)


def add_questions(db, topic, competency, n, status=Q.PUBLISHED, correct="B", prefix="q"):
    comp = db.scalars(select(Competency).join(Topic).where(Topic.name == topic, Competency.name == competency)).one()
    made = []
    for i in range(n):
        q = Question(competency_id=comp.id, text=f"{prefix} {topic} {i}?",
                     options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option=correct,
                     difficulty=2, explanation=f"Because {prefix}{i}.", status=status)
        db.add(q)
        made.append(q)
    db.commit()
    return made


@pytest.fixture()
def bank(db, seeded):
    """4 published Electricity questions, 2 published Motion, plus 1 pending and 1 retired (never testable)."""
    add_questions(db, "Current Electricity", "Apply Ohm's law", 4, prefix="e")
    add_questions(db, "Motion", "Apply equations of uniform acceleration", 2, prefix="m")
    add_questions(db, "Motion", "Apply equations of uniform acceleration", 1, status=Q.PENDING_REVIEW, prefix="pend")
    add_questions(db, "Motion", "Apply equations of uniform acceleration", 1, status=Q.RETIRED, prefix="old")


@pytest.fixture()
def other(db):
    return register_user(db, "tunde@example.com", "Tunde Bello", "Physics2026")


def begin(db, student, n=6, mode="mixed", topic=None, now=T0, seed=1):
    return start_test(db, student, mode, topic, n, rng=random.Random(seed), now=now)


# ------------------------------------------------------------------ choosing questions
def test_available_counts_only_counts_published(db, bank):
    assert available_counts(db) == {"Current Electricity": 4, "Motion": 2}


def test_topic_test_uses_only_published_questions_of_that_topic(db, bank, student):
    sid = begin(db, student, n=10, mode="topic", topic="Motion")
    texts = [i.text for i in load_active_test(db, student, sid, T0).items]
    assert len(texts) == 2                       # only 2 published Motion questions exist
    assert all(t.startswith("m Motion") for t in texts)


def test_mixed_test_spreads_across_topics(db, bank, student):
    sid = begin(db, student, n=3)
    topics = {t.split()[1] for t in (i.text for i in load_active_test(db, student, sid, T0).items)}
    assert topics == {"Current", "Motion"}  # both topics appear even though Electricity is bigger


def test_same_seed_gives_same_test(db, bank, student):
    first = [i.text for i in load_active_test(db, student, begin(db, student, seed=7), T0).items]
    abandon_test(db, student, db.scalars(select(ExamSession.id)).one())
    second = [i.text for i in load_active_test(db, student, begin(db, student, seed=7), T0).items]
    assert first == second


def test_no_published_questions_means_no_test(db, seeded, student):
    with pytest.raises(CbtError, match="no published questions"):
        begin(db, student)


def test_bad_inputs(db, bank, student):
    with pytest.raises(CbtError):
        start_test(db, student, "nonsense", None, 5)
    with pytest.raises(CbtError):
        start_test(db, student, "mixed", None, 0)
    with pytest.raises(CbtError, match="topic"):
        start_test(db, student, "topic", "No Such Topic", 5)


# ------------------------------------------------------------------ one active test
def test_only_one_test_in_progress(db, bank, student):
    begin(db, student)
    with pytest.raises(CbtError, match="already have a test in progress"):
        begin(db, student)


def test_database_itself_refuses_a_second_active_test(db, bank, student):
    """The rule is enforced by the DATABASE, not just by Python code (protects against double-clicks)."""
    sid = begin(db, student)
    clone = ExamSession(user_id=student.id, title="x", mode="mixed", status=SessionStatus.IN_PROGRESS,
                        n_questions=1, duration_seconds=90, started_at=T0, deadline_at=T0 + minutes(2))
    db.add(clone)
    with pytest.raises(IntegrityError):
        db.flush()
    db.rollback()
    assert db.get(ExamSession, sid) is not None


def test_two_students_can_each_have_a_test(db, bank, student, other):
    begin(db, student)
    begin(db, other)
    assert db.scalars(select(ExamSession)).all().__len__() == 2


# ------------------------------------------------------------------ what the student sees
def test_running_test_contains_no_answer_key(db, bank, student):
    exam = load_active_test(db, student, begin(db, student), T0)
    forbidden = {"correct_option", "explanation", "is_correct"}
    for item in exam.items:
        assert not forbidden & {f.name for f in dataclasses.fields(item)}
        assert not forbidden & set(vars(item))
    assert [i.position for i in exam.items] == list(range(1, len(exam.items) + 1))


def test_deadline_is_start_plus_duration_and_timezone_aware(db, bank, student):
    sid = begin(db, student, n=4)
    exam = load_active_test(db, student, sid, T0)
    assert exam.deadline_at == T0 + timedelta(seconds=4 * SECONDS_PER_QUESTION)
    assert exam.deadline_at.tzinfo is not None   # SQLite returns naive times; we normalise them


def test_question_images_are_included(db, bank, student):
    q = db.scalars(select(Question).where(Question.status == Q.PUBLISHED)).first()
    db.add(QuestionImage(question_id=q.id, role="stem", content_type="image/png", width=10, height=10,
                         sha256="x" * 64, alt_text="A circuit diagram of a battery.", data=b"png"))
    db.commit()
    exam = load_active_test(db, student, begin(db, student, n=6), T0)
    with_images = [i for i in exam.items if i.images]
    assert len(with_images) == 1 and with_images[0].images[0].role == "stem"


# ------------------------------------------------------------------ answering
def test_answers_are_saved_immediately_and_survive_a_reload(db, bank, student):
    sid = begin(db, student)
    first = load_active_test(db, student, sid, T0).items[0]
    save_answer(db, student, first.item_id, "c", T0 + minutes(1))      # lower-case is accepted
    reloaded = load_active_test(db, student, sid, T0 + minutes(2)).items[0]
    assert reloaded.selected == "C"


def test_answer_can_be_changed_and_cleared(db, bank, student):
    sid = begin(db, student)
    item = load_active_test(db, student, sid, T0).items[0]
    save_answer(db, student, item.item_id, "A", T0)
    save_answer(db, student, item.item_id, "D", T0)
    assert db.get(ExamItem, item.item_id).selected_option == "D"
    save_answer(db, student, item.item_id, None, T0)
    assert db.get(ExamItem, item.item_id).selected_option is None


def test_invalid_option_rejected(db, bank, student):
    item = load_active_test(db, student, begin(db, student), T0).items[0]
    with pytest.raises(CbtError, match="valid option"):
        save_answer(db, student, item.item_id, "E", T0)


def test_flagging(db, bank, student):
    item = load_active_test(db, student, begin(db, student), T0).items[0]
    set_flag(db, student, item.item_id, True, T0)
    assert db.get(ExamItem, item.item_id).flagged is True


def test_a_student_cannot_touch_another_students_test(db, bank, student, other):
    """The IDOR test: changing the number in the request must not work."""
    sid = begin(db, student)
    item = load_active_test(db, student, sid, T0).items[0]
    with pytest.raises(CbtError, match="not found"):
        save_answer(db, other, item.item_id, "A", T0)
    with pytest.raises(CbtError, match="not found"):
        set_flag(db, other, item.item_id, True, T0)
    with pytest.raises(CbtError, match="not found"):
        load_active_test(db, other, sid, T0)
    with pytest.raises(CbtError, match="not found"):
        submit_test(db, other, sid, T0)
    with pytest.raises(CbtError, match="not found"):
        abandon_test(db, other, sid)
    assert db.get(ExamItem, item.item_id).selected_option is None


# ------------------------------------------------------------------ grading
def finish_with(db, student, answers_by_position, n=6, now=T0 + minutes(5)):
    sid = begin(db, student, n=n)
    for item in load_active_test(db, student, sid, T0).items:
        choice = answers_by_position.get(item.position)
        if choice:
            save_answer(db, student, item.item_id, choice, T0 + minutes(1))
    return sid, submit_test(db, student, sid, now)


def test_grading_counts_correct_wrong_and_unanswered(db, bank, student):
    # every question's correct answer is "B"
    sid, result = finish_with(db, student, {1: "B", 2: "B", 3: "A", 4: "B"})  # 5 and 6 unanswered
    assert (result.score, result.total) == (3, 6) and result.percent == 50.0
    by_pos = {i.position: i for i in result.items}
    assert by_pos[1].is_correct and not by_pos[3].is_correct
    assert by_pos[5].selected is None and not by_pos[5].is_correct
    assert not result.auto_submitted


def test_result_includes_answer_key_and_explanations_only_after_submission(db, bank, student):
    sid, result = finish_with(db, student, {1: "A"})
    assert all(i.correct_option == "B" and i.explanation.startswith("Because") for i in result.items)


def test_topic_breakdown(db, bank, student):
    sid, result = finish_with(db, student, {p: "B" for p in range(1, 7)})
    assert {t.topic: (t.correct, t.total) for t in result.by_topic} == {"Current Electricity": (4, 4), "Motion": (2, 2)}
    assert all(t.percent == 100.0 for t in result.by_topic)


def test_double_submit_is_harmless(db, bank, student):
    sid, first = finish_with(db, student, {1: "B"})
    second = submit_test(db, student, sid, T0 + minutes(9))
    assert second.score == first.score and second.submitted_at == first.submitted_at


def test_cannot_answer_after_submitting(db, bank, student):
    sid, _ = finish_with(db, student, {1: "B"})
    item_id = db.scalars(select(ExamItem.id).where(ExamItem.session_id == sid)).first()
    with pytest.raises(CbtError, match="already finished"):
        save_answer(db, student, item_id, "A", T0 + minutes(6))


def test_results_not_available_while_test_is_running(db, bank, student):
    sid = begin(db, student)
    with pytest.raises(CbtError, match="only available after"):
        get_result(db, student, sid)


def test_results_belong_to_their_owner(db, bank, student, other):
    sid, _ = finish_with(db, student, {1: "B"})
    with pytest.raises(CbtError, match="not found"):
        get_result(db, other, sid)
    assert get_result(db, student, sid).session_id == sid


# ------------------------------------------------------------------ the clock
def test_time_runs_out_while_answering(db, bank, student):
    sid = begin(db, student, n=4)   # 6 minutes
    items = load_active_test(db, student, sid, T0).items
    save_answer(db, student, items[0].item_id, "B", T0 + minutes(1))
    with pytest.raises(TimeUp) as caught:
        load_active_test(db, student, sid, T0 + minutes(7))
    assert caught.value.session_id == sid

    result = get_result(db, student, sid)       # it was graded automatically
    assert result.auto_submitted and result.score == 1 and result.total == 4
    assert result.submitted_at == T0 + minutes(6)   # recorded at the DEADLINE, not when we noticed


def test_answer_in_grace_window_still_counts_but_later_does_not(db, bank, student):
    sid = begin(db, student, n=4)
    deadline = T0 + minutes(6)
    item = load_active_test(db, student, sid, T0).items[0]
    save_answer(db, student, item.item_id, "B", deadline + timedelta(seconds=GRACE_SECONDS - 1))  # accepted
    assert db.get(ExamItem, item.item_id).selected_option == "B"
    with pytest.raises(TimeUp):
        save_answer(db, student, item.item_id, "A", deadline + timedelta(seconds=GRACE_SECONDS + 1))
    assert db.get(ExamItem, item.item_id).selected_option == "B"   # the late change was ignored


def test_expired_test_is_graded_when_the_student_returns(db, bank, student):
    sid = begin(db, student, n=4)
    assert check_active(db, student, T0 + minutes(1)) == (sid, None)
    assert check_active(db, student, T0 + minutes(60)) == (None, sid)   # away for an hour
    assert get_result(db, student, sid).auto_submitted
    assert check_active(db, student, T0 + minutes(61)) == (None, None)
    begin(db, student, now=T0 + minutes(62))                            # and they can start a new one


def test_submit_after_deadline_is_marked_automatic(db, bank, student):
    sid = begin(db, student, n=4)
    result = submit_test(db, student, sid, T0 + minutes(30))
    assert result.auto_submitted


# ------------------------------------------------------------------ discarding and history
def test_discarded_test_is_never_graded_or_listed(db, bank, student):
    sid = begin(db, student)
    abandon_test(db, student, sid)
    assert list_sessions(db, student) == []
    with pytest.raises(CbtError, match="discarded"):
        submit_test(db, student, sid, T0)
    with pytest.raises(CbtError):
        get_result(db, student, sid)
    begin(db, student, now=T0 + minutes(1))   # free to start another


def test_history_lists_only_own_finished_tests_newest_first(db, bank, student, other):
    for hour in (1, 2):
        sid = begin(db, student, now=T0 + timedelta(hours=hour))
        submit_test(db, student, sid, T0 + timedelta(hours=hour, minutes=3))
    submit_test(db, other, begin(db, other), T0 + minutes(3))
    mine = list_sessions(db, student)
    assert len(mine) == 2 and mine[0].started_at > mine[1].started_at
    assert len(list_sessions(db, other)) == 1
