"""Tests for core/progress.py: the dashboard data, the replayed history, and "what did this test change?"."""
import random
from datetime import timedelta

import pytest
from sqlalchemy import select

from core.auth import register_user
from core.cbt import CbtError, start_test
from core.mastery import compute_readiness, compute_states, load_catalog, load_evidence
from core.orm import ExamItem, Question, TutorExplanationRow
from core.progress import (
    build_dashboard, profile_change, readiness_history, recent_mistakes, topic_summaries,
)
from tests.adaptive_helpers import T0, comp_name, publish_bank, take_test


@pytest.fixture()
def bank(db, seeded):
    publish_bank(db, per_competency=4)


@pytest.fixture()
def other(db):
    return register_user(db, "tunde@example.com", "Tunde Bello", "Physics2026")


DAY = timedelta(days=1)


# ------------------------------------------------------------------ the dashboard
def test_a_brand_new_student_gets_a_sensible_dashboard(db, bank, student):
    d = build_dashboard(db, student, T0)
    assert d.readiness.provisional and d.readiness.attempts == 0 and d.readiness.assessed == 0
    assert d.history == [] and d.recent == [] and d.mistakes == []
    assert len(d.plan) == 4 and all("haven't tried" in item.reason for item in d.plan)
    assert len(d.states) == 4 and abs(sum(t.weight for t in d.topics) - 1) < 1e-9


def test_the_dashboard_reflects_finished_tests(db, bank, student):
    take_test(db, student, lambda q: True, now=T0)
    d = build_dashboard(db, student, T0 + DAY)
    assert d.readiness.attempts == 16 and d.readiness.assessed == 4
    assert not d.readiness.provisional and d.readiness.score > 70
    assert len(d.recent) == 1 and d.recent[0].score == 16 and len(d.history) == 1


def test_topic_summaries_use_exam_weights(db, bank, student):
    take_test(db, student, lambda q: comp_name(db, q).startswith("Apply Ohm"), now=T0)
    topics = {t.topic: t for t in topic_summaries(build_dashboard(db, student, T0 + DAY).states)}
    assert topics["Current Electricity"].expected_score > topics["Motion"].expected_score
    assert topics["Motion"].assessed == 2 and topics["Motion"].total == 2


# ------------------------------------------------------------------ progress over time (replay)
def test_history_has_one_point_per_test_in_order(db, bank, student):
    take_test(db, student, lambda q: False, now=T0)
    take_test(db, student, lambda q: True, now=T0 + 3 * DAY)
    take_test(db, student, lambda q: True, now=T0 + 6 * DAY)
    points = readiness_history(db, student)
    assert len(points) == 3 and [p.when for p in points] == sorted(p.when for p in points)
    assert points[0].score < points[1].score and points[1].low <= points[1].score <= points[1].high


def test_each_history_point_only_knows_what_existed_then(db, bank, student):
    take_test(db, student, lambda q: False, now=T0)
    only_first = readiness_history(db, student)[0].score
    take_test(db, student, lambda q: True, now=T0 + 3 * DAY)          # a later, very different test
    assert readiness_history(db, student)[0].score == pytest.approx(only_first)     # the past did not change


def test_a_history_point_equals_a_direct_calculation(db, bank, student):
    sid = take_test(db, student, lambda q: q.difficulty >= 3, now=T0)
    point = readiness_history(db, student)[0]
    evidence = [e for e in load_evidence(db, student) if e.session_id == sid]
    direct = compute_readiness(compute_states(load_catalog(db), evidence, point.when))
    assert point.score == pytest.approx(direct.score) and point.provisional == direct.provisional


def test_history_can_be_limited_to_the_latest_tests(db, bank, student):
    for i in range(4):
        take_test(db, student, lambda q: True, now=T0 + i * DAY)
    assert len(readiness_history(db, student, limit=2)) == 2
    assert readiness_history(db, student, limit=2)[-1].when > readiness_history(db, student, limit=2)[0].when


def test_history_is_private(db, bank, student, other):
    take_test(db, student, lambda q: True, now=T0)
    assert readiness_history(db, other) == []


# ------------------------------------------------------------------ mistakes the AI has already spotted
def explanation(db, question_id, option, text, version=1):
    db.add(TutorExplanationRow(question_id=question_id, selected_option=option, question_version=version, model="m",
                               content={"explanation": "x", "study_tip": "y", "misconception": text}))
    db.commit()


def test_misconceptions_are_collected_from_wrong_answers(db, bank, student, other):
    sid = take_test(db, student, lambda q: False, now=T0)
    take_test(db, other, lambda q: False, now=T0)
    ohm = [i.question_id for i in db.scalars(select(ExamItem).where(ExamItem.session_id == sid))
           if comp_name(db, db.get(Question, i.question_id)) == "Apply Ohm's law"]
    explanation(db, ohm[0], "A", "Mixing up V/R with R/V")
    explanation(db, ohm[1], "A", "mixing up v/r with r/v")             # the same idea, other question
    explanation(db, ohm[2], "A", "")                                    # nothing useful to show
    notes = recent_mistakes(db, student)
    assert len(notes) == 1 and notes[0].times == 2 and notes[0].competency == "Apply Ohm's law"
    assert notes[0].misconception.lower() == "mixing up v/r with r/v"


def test_correct_answers_and_other_options_are_not_mistakes(db, bank, student):
    sid = take_test(db, student, lambda q: True, now=T0)
    qid = db.scalars(select(ExamItem.question_id).where(ExamItem.session_id == sid)).first()
    explanation(db, qid, "B", "should not appear")                       # B was the correct choice
    explanation(db, qid, "C", "was never chosen")
    assert recent_mistakes(db, student) == []


# ------------------------------------------------------------------ "what did this test change?"
def test_a_good_test_after_a_bad_one_raises_the_profile(db, bank, student):
    bad = take_test(db, student, lambda q: False, now=T0)
    good = take_test(db, student, lambda q: True, now=T0 + 3 * DAY)
    first, second = profile_change(db, student, bad), profile_change(db, student, good)
    assert first.before.attempts == 0 and first.after.attempts == 16 and first.after.score < first.before.score
    assert second.after.score > second.before.score
    assert second.changes and all(c.after > c.before for c in second.changes)
    assert all(c.answered == c.correct == 4 for c in second.changes)
    assert [abs(c.after - c.before) for c in second.changes] == sorted((abs(c.after - c.before) for c in second.changes), reverse=True)


def test_reviewing_an_old_test_ignores_what_came_later(db, bank, student):
    old = take_test(db, student, lambda q: False, now=T0)
    take_test(db, student, lambda q: True, now=T0 + 3 * DAY)
    change = profile_change(db, student, old)
    assert change.before.attempts == 0 and change.after.attempts == 16       # the later test is not part of THAT story


def test_only_touched_competencies_are_listed(db, bank, student):
    sid = take_test(db, student, lambda q: True, mode="topic", topic="Motion", n=40, now=T0)
    assert {c.topic for c in profile_change(db, student, sid).changes} == {"Motion"}


def test_profile_change_is_private_and_needs_a_finished_test(db, bank, student, other):
    sid = take_test(db, student, lambda q: True, now=T0)
    with pytest.raises(CbtError, match="not found"):
        profile_change(db, other, sid)
    running = start_test(db, student, "mixed", None, 5, rng=random.Random(1), now=T0 + 5 * DAY)
    with pytest.raises(CbtError, match="only available after"):
        profile_change(db, student, running)
