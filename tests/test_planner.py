"""
Tests for core/planner.py: what to practise, and which questions to serve.
Run with:  pytest -q
"""
import random
from datetime import timedelta

import pytest
from sqlalchemy import select

from core import planner
from core.cbt import CbtError, start_test
from core.mastery import CompetencyMeta, Evidence, compute_states
from core.orm import ExamItem, Question, QuestionStatus as Q
from core.planner import (
    MAX_PER_COMPETENCY, PlanItem, allocate, build_plan, pick_adaptive_questions, pick_diagnostic_questions,
    priority, reason_for,
)
from tests.adaptive_helpers import T0, add_question, publish_bank, take_test

NOW = T0 + timedelta(days=60)


def ev(comp, correct, days=0.0, n=1, difficulty=3):
    when = NOW - timedelta(days=days)
    return [Evidence(comp, 1000 + i, 1, difficulty, correct, when, when) for i in range(n)]


def profile(weights, evidence):
    catalog = [CompetencyMeta(i + 1, "T", f"C{i + 1}", w) for i, w in enumerate(weights)]
    return compute_states(catalog, evidence, NOW)


# ================================================================== what to practise
def test_weak_and_heavy_beats_strong_and_light():
    states = profile([0.5, 0.5], ev(1, False, n=8) + ev(2, True, n=8))
    plan = build_plan(states, NOW)
    assert [p.state.meta.id for p in plan] == [1, 2]
    assert plan[0].priority > plan[1].priority


def test_exam_weight_changes_the_order():
    same = ev(1, False, n=6) + ev(2, False, n=6)
    light_first = build_plan(profile([0.9, 0.1], same), NOW)
    heavy_first = build_plan(profile([0.1, 0.9], same), NOW)
    assert light_first[0].state.meta.id == 1 and heavy_first[0].state.meta.id == 2


def test_not_knowing_enough_is_worth_exploring():
    states = profile([0.5, 0.5], ev(1, True, n=15) + [])        # competency 1 is well known, 2 is untouched
    assert build_plan(states, NOW)[0].state.meta.id == 2


def test_competencies_you_stopped_practising_rise_back_up():
    fresh = profile([0.5, 0.5], ev(1, True, n=6) + ev(2, True, n=6))
    stale = profile([0.5, 0.5], ev(1, True, n=6) + ev(2, True, n=6, days=120))
    assert priority(stale[1]) > priority(fresh[1])               # forgetting alone made it a priority again
    assert build_plan(stale, NOW)[0].state.meta.id == 2


def test_ties_break_the_same_way_every_time():
    states = profile([0.25] * 4, [])
    assert [p.state.meta.id for p in build_plan(states, NOW)] == [1, 2, 3, 4]


def test_plan_size_is_respected():
    assert len(build_plan(profile([0.1] * 9, []), NOW)) == planner.PLAN_SIZE
    assert len(build_plan(profile([0.1] * 9, []), NOW, size=2)) == 2


# ================================================================== the "why"
def test_every_kind_of_reason_is_reachable():
    never = profile([1.0], [])[0]
    few = profile([1.0], ev(1, True, n=2))[0]
    weak = profile([1.0], ev(1, False, n=12))[0]
    stale = profile([1.0], ev(1, True, n=14, days=20))[0]                 # solid, but not touched for 20 days
    middling = profile([1.0], ev(1, True, n=10) + ev(1, False, n=3))[0]
    strong = profile([1.0], ev(1, True, n=30))[0]
    assert "haven't tried" in reason_for(never, NOW)
    assert "don't know enough" in reason_for(few, NOW)
    assert reason_for(weak, NOW).startswith("Weak spot")
    assert "20 days" in reason_for(stale, NOW)
    assert "Getting there" in reason_for(middling, NOW)
    assert reason_for(strong, NOW).startswith("Strong")


# ================================================================== sharing questions out
@pytest.mark.parametrize("n", [1, 3, 5, 10, 20, 40])
def test_allocation_always_adds_up(n):
    plan = build_plan(profile([0.3, 0.25, 0.2, 0.15, 0.1], ev(1, False, n=5) + ev(3, False, n=2)), NOW)
    shares = allocate(plan, n)
    assert sum(shares.values()) == min(n, sum(MAX_PER_COMPETENCY for _ in plan))
    assert all(1 <= v <= MAX_PER_COMPETENCY for v in shares.values())


def test_bigger_priority_gets_more_questions():
    plan = build_plan(profile([0.7, 0.3], ev(1, False, n=6) + ev(2, False, n=6)), NOW)
    shares = allocate(plan, 10)
    assert shares[1] > shares[2]


def test_allocation_with_nothing_to_allocate():
    assert allocate([], 5) == {}


def test_never_more_competencies_than_questions():
    plan = build_plan(profile([0.2] * 5, []), NOW)
    assert len(allocate(plan, 2)) == 2


# ================================================================== picking real questions
def test_only_published_questions_are_ever_chosen(db, seeded, student):
    publish_bank(db)
    drafts = [add_question(db, "Apply Ohm's law", 3, status=Q.PENDING_REVIEW), add_question(db, "Apply Ohm's law", 3, status=Q.RETIRED)]
    chosen = pick_adaptive_questions(db, student, 12, random.Random(1), NOW)
    assert chosen and not set(chosen) & set(drafts)
    assert not set(pick_diagnostic_questions(db, student, random.Random(1), NOW)) & set(drafts)


def test_the_requested_number_is_returned_without_repeats(db, seeded, student):
    publish_bank(db)
    chosen = pick_adaptive_questions(db, student, 8, random.Random(2), NOW)
    assert len(chosen) == 8 == len(set(chosen))


def test_no_questions_means_an_empty_list(db, seeded, student):
    assert pick_adaptive_questions(db, student, 5, random.Random(1), NOW) == []
    assert pick_diagnostic_questions(db, student, random.Random(1), NOW) == []


def test_the_student_is_steered_towards_their_weak_competency(db, seeded, student):
    publish_bank(db, per_competency=6, difficulties=(2, 3, 4))
    # right on everything about Ohm's law, wrong on everything else
    take_test(db, student, lambda q: db.get(Question, q.id).competency_id == db.scalars(
        select(Question.competency_id).where(Question.text.like("Apply Ohm's law%"))).first(), n=40, now=T0)
    chosen = pick_adaptive_questions(db, student, 5, random.Random(3), T0 + timedelta(days=1))
    texts = [db.get(Question, q).text for q in chosen]
    assert sum(t.startswith("Apply Ohm's law") for t in texts) <= 1          # the strong one gets (almost) no attention
    assert len(texts) == 5


def test_unseen_questions_come_before_repeats(db, seeded, student):
    publish_bank(db, per_competency=8)                  # 32 questions; the first test uses 12, so 20 stay unseen
    take_test(db, student, lambda q: True, n=12, now=T0, seed=5)
    done = set(db.scalars(select(ExamItem.question_id)))
    chosen = pick_adaptive_questions(db, student, 8, random.Random(4), T0 + timedelta(days=1))
    assert not set(chosen) & done                       # plenty of unseen ones exist, so no repeats


def history_then_three_targets(db, user, right: bool) -> dict[int, int]:
    """Give the student six answered questions (all right, or all wrong) in ONE competency, then add three fresh
    questions of difficulty 1, 3 and 5 so the planner has a real choice."""
    comp = "Apply Ohm's law"
    for _ in range(6):
        add_question(db, comp, 3)
    take_test(db, user, lambda q: right, n=40, now=T0)
    return {d: add_question(db, comp, d) for d in (1, 3, 5)}


def test_a_struggling_student_is_given_an_easier_question(db, seeded, student):
    targets = history_then_three_targets(db, student, right=False)
    chosen = pick_adaptive_questions(db, student, 1, random.Random(1), T0 + timedelta(days=1))
    assert chosen == [targets[1]]                       # difficulty 1 gives about the 65% target chance of success


def test_a_strong_student_is_given_a_harder_question(db, seeded, student):
    targets = history_then_three_targets(db, student, right=True)
    chosen = pick_adaptive_questions(db, student, 1, random.Random(1), T0 + timedelta(days=1))
    assert chosen == [targets[5]]


def test_a_short_competency_is_topped_up_from_the_next_best(db, seeded, student):
    publish_bank(db, per_competency=1)               # only ONE question per competency
    chosen = pick_adaptive_questions(db, student, 4, random.Random(1), NOW)
    assert len(chosen) == 4 == len(set(chosen))


def test_diagnostic_covers_every_competency_with_at_most_two_questions_each(db, seeded, student):
    publish_bank(db, per_competency=5)
    chosen = pick_diagnostic_questions(db, student, random.Random(1), NOW)
    per = {}
    for qid in chosen:
        per.setdefault(db.get(Question, qid).competency_id, []).append(qid)
    assert len(per) == 4 and all(len(v) == planner.DIAGNOSTIC_PER_COMPETENCY for v in per.values())


def test_diagnostic_prefers_medium_difficulty(db, seeded, student):
    comp = "Apply Ohm's law"
    mid = add_question(db, comp, 3)
    add_question(db, comp, 1)
    add_question(db, comp, 5)
    chosen = pick_diagnostic_questions(db, student, random.Random(1), NOW)
    assert mid in chosen


def test_diagnostic_is_capped_to_the_most_valuable_competencies(db, seeded, student, monkeypatch):
    publish_bank(db, per_competency=3)
    monkeypatch.setattr(planner, "MAX_TEST_QUESTIONS", 3)
    assert len(pick_diagnostic_questions(db, student, random.Random(1), NOW)) == 3


# ================================================================== the new test modes
def test_diagnostic_and_adaptive_tests_can_be_started(db, seeded, student):
    publish_bank(db, per_competency=3)
    sid = start_test(db, student, "diagnostic", None, 40, rng=random.Random(1), now=T0)
    from core.orm import ExamSession
    session = db.get(ExamSession, sid)
    assert session.title == "Diagnostic test" and session.mode == "diagnostic" and session.n_questions == 8
    from core.cbt import abandon_test
    abandon_test(db, student, sid)
    sid2 = start_test(db, student, "adaptive", None, 5, rng=random.Random(1), now=T0)
    assert db.get(ExamSession, sid2).title == "Adaptive practice" and db.get(ExamSession, sid2).n_questions == 5
    assert len(list(db.scalars(select(ExamItem).where(ExamItem.session_id == sid2)))) == 5


def test_adaptive_mode_needs_published_questions_and_valid_input(db, seeded, student):
    with pytest.raises(CbtError, match="no published questions"):
        start_test(db, student, "adaptive", None, 5, now=T0)
    with pytest.raises(CbtError, match="Unknown test type"):
        start_test(db, student, "chaos", None, 5, now=T0)
