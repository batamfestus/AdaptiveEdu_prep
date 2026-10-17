"""
Tests for core/mastery.py. Run with:  pytest -q

For a statistical model the best tests are PROPERTIES: rules that must hold for any data
(more correct answers never lower mastery; evidence fades with time; a guess can't push the
expected score below 25%). Exact numbers would break whenever we retune the model.
"""
import math
import random
from datetime import timedelta

import pytest

from core.auth import register_user
from core.cbt import start_test
from core.mastery import (
    GUESS, HALF_LIFE_DAYS, MIN_ATTEMPTS_FOR_READINESS, UNANSWERED_WEIGHT, CompetencyMeta, Evidence,
    compute_readiness, compute_states, evidence_weight, load_catalog, load_evidence, p_correct,
)
from core.orm import Topic
from sqlalchemy import select
from tests.adaptive_helpers import T0, publish_bank, take_test

NOW = T0 + timedelta(days=60)
META = CompetencyMeta(1, "Motion", "Graphs", 1.0)


def ev(correct, difficulty=3, days=0.0, weight=1.0, comp=1, question=1, session=1):
    when = NOW - timedelta(days=days)
    return Evidence(comp, question, session, difficulty, correct, when, when, weight)


def state(items, now=NOW):
    return compute_states([META], items, now)[0]


# ------------------------------------------------------------------ the answer model
def test_a_stronger_student_is_more_likely_to_be_right_and_guessing_sets_a_floor():
    chances = [p_correct(theta, 3) for theta in (-4, -2, 0, 2, 4)]
    assert chances == sorted(chances)
    assert chances[0] >= GUESS and chances[-1] <= 1.0
    assert p_correct(0, 1) > p_correct(0, 3) > p_correct(0, 5)        # easier questions are easier


def test_out_of_range_difficulty_is_clamped_not_a_crash():
    assert p_correct(0, 0) == p_correct(0, 1) and p_correct(0, 99) == p_correct(0, 5)


# ------------------------------------------------------------------ properties of the belief
def test_no_evidence_means_a_wide_open_belief():
    s = state([])
    assert 0.3 < s.mastery < 0.6 and s.attempts == 0 and s.last_attempt is None
    assert s.uncertainty > 0.2 and (s.high - s.low) > 0.4
    assert not s.assessed


def test_more_correct_answers_never_lower_mastery():
    levels = [state([ev(True)] * n).mastery for n in range(0, 10)]
    assert levels == sorted(levels) and levels[-1] > levels[0] + 0.3


def test_more_wrong_answers_never_raise_mastery():
    levels = [state([ev(False)] * n).mastery for n in range(0, 10)]
    assert levels == sorted(levels, reverse=True) and levels[-1] < levels[0] - 0.2


def test_difficulty_matters_both_ways():
    hard, mid, easy = (state([ev(True, d)] * 3).mastery for d in (5, 3, 1))
    assert hard > mid > easy                                           # beating a hard question says more
    wrong_easy, wrong_hard = (state([ev(False, d)] * 3).mastery for d in (1, 5))
    assert wrong_easy < wrong_hard                                     # missing an easy one says more


def test_the_guess_floor_holds_even_after_many_wrong_answers():
    s = state([ev(False)] * 30)
    assert s.expected_score >= GUESS and s.low >= GUESS


def test_uncertainty_shrinks_as_evidence_grows():
    mixed = [ev(i % 3 != 0) for i in range(30)]
    sds = [state(mixed[:n]).uncertainty for n in (0, 2, 8, 30)]
    assert sds == sorted(sds, reverse=True)


def test_the_credible_range_brackets_the_estimate():
    for items in ([], [ev(True)] * 4, [ev(False)] * 4, [ev(True), ev(False)] * 5):
        s = state(items)
        assert 0 <= s.low <= s.expected_score <= s.high <= 1


def test_order_of_evidence_does_not_matter():
    items = [ev(bool(i % 2), difficulty=1 + i % 5, days=i) for i in range(12)]
    shuffled = items[:]
    random.Random(4).shuffle(shuffled)
    a, b = state(items), state(shuffled)
    assert math.isclose(a.mastery, b.mastery, abs_tol=1e-12) and math.isclose(a.uncertainty, b.uncertainty, abs_tol=1e-12)


def test_extreme_evidence_stays_numerically_sane():
    for items in ([ev(False)] * 500, [ev(True, 5)] * 500):
        s = state(items)
        assert all(math.isfinite(x) for x in (s.mastery, s.uncertainty, s.expected_score, s.theta))
        assert 0 <= s.mastery <= 1


def test_future_evidence_is_treated_as_fresh():
    assert evidence_weight(ev(True, days=-5), NOW) == 1.0


# ------------------------------------------------------------------ forgetting
def test_evidence_loses_half_its_weight_every_half_life():
    assert evidence_weight(ev(True, days=0), NOW) == pytest.approx(1.0)
    assert evidence_weight(ev(True, days=HALF_LIFE_DAYS), NOW) == pytest.approx(0.5)
    assert evidence_weight(ev(True, days=2 * HALF_LIFE_DAYS), NOW) == pytest.approx(0.25)


def test_old_evidence_drifts_back_towards_unknown():
    fresh, old, none = state([ev(True)] * 5), state([ev(True, days=90)] * 5), state([])
    assert none.mastery < old.mastery < fresh.mastery                  # still some credit, but less
    assert old.uncertainty > fresh.uncertainty                          # and we are less sure
    assert old.evidence_weight < fresh.evidence_weight
    assert old.attempts == fresh.attempts == 5                          # the history itself is not erased


def test_unanswered_questions_count_as_half_a_wrong_answer():
    assert UNANSWERED_WEIGHT == 0.5
    blanks, wrongs = state([ev(False, weight=0.5)] * 4), state([ev(False)] * 2)
    assert math.isclose(blanks.mastery, wrongs.mastery, abs_tol=1e-9)


# ------------------------------------------------------------------ readiness
def two_competencies(heavy=0.8):
    return [CompetencyMeta(1, "Motion", "Graphs", heavy), CompetencyMeta(2, "Heat", "Gas laws", 1 - heavy)]


def test_readiness_is_the_weighted_expected_score():
    catalog = two_competencies(0.8)
    strong_heavy = [ev(True, comp=1)] * 8 + [ev(False, comp=2)] * 8
    weak_heavy = [ev(False, comp=1)] * 8 + [ev(True, comp=2)] * 8
    a = compute_readiness(compute_states(catalog, strong_heavy, NOW))
    b = compute_readiness(compute_states(catalog, weak_heavy, NOW))
    assert a.score > b.score + 20                                       # the heavier topic dominates
    states = compute_states(catalog, strong_heavy, NOW)
    assert a.score == pytest.approx(100 * sum(s.meta.weight * s.expected_score for s in states))


def test_readiness_range_stays_between_guessing_and_perfect():
    for items in ([], [ev(False)] * 50, [ev(True, 5)] * 50):
        r = compute_readiness(compute_states(two_competencies(), items, NOW))
        assert 25 <= r.low <= r.score <= r.high <= 100


def test_readiness_is_provisional_until_there_is_enough_evidence():
    catalog = two_competencies()
    few = compute_readiness(compute_states(catalog, [ev(True, comp=1)] * 3, NOW))
    assert few.provisional and few.assessed == 1 and few.coverage == 0.5
    lots = [ev(True, comp=c, question=i) for c in (1, 2) for i in range(MIN_ATTEMPTS_FOR_READINESS)]
    assert not compute_readiness(compute_states(catalog, lots, NOW)).provisional
    assert compute_readiness(compute_states(catalog, [], NOW)).provisional


def test_confidence_tracks_how_narrow_the_range_is():
    catalog = two_competencies()
    none = compute_readiness(compute_states(catalog, [], NOW))
    lots = compute_readiness(compute_states(catalog, [ev(i % 4 != 0, comp=1 + i % 2) for i in range(300)], NOW))
    assert none.confidence == "low" and lots.confidence in ("medium", "high")
    assert (lots.high - lots.low) < (none.high - none.low)


# ------------------------------------------------------------------ reading the database
def test_exam_weights_are_split_across_a_topics_competencies_and_normalised(db, seeded):
    ce, motion = (db.scalars(select(Topic).where(Topic.name == n)).one() for n in ("Current Electricity", "Motion"))
    ce.exam_weight, motion.exam_weight = 3.0, 1.0
    db.commit()
    catalog = load_catalog(db)
    weights = {m.name: m.weight for m in catalog}
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["Apply Ohm's law"] == pytest.approx(0.375) and weights["Distinguish scalars and vectors"] == pytest.approx(0.125)


def test_zero_weights_fall_back_to_equal_shares(db, seeded):
    for topic in db.scalars(select(Topic)):
        topic.exam_weight = 0.0
    db.commit()
    assert {round(m.weight, 6) for m in load_catalog(db)} == {0.25}


def test_evidence_comes_only_from_finished_tests_of_this_student(db, seeded, student):
    publish_bank(db)
    other = register_user(db, "tunde@example.com", "Tunde Bello", "Physics2026")
    take_test(db, student, lambda q: True, now=T0)
    take_test(db, other, lambda q: False, now=T0)
    start_test(db, student, "mixed", None, 5, rng=random.Random(2), now=T0 + timedelta(days=1))   # left in progress

    mine = load_evidence(db, student)
    assert len(mine) == 12 and all(e.correct for e in mine)
    assert not any(e.correct for e in load_evidence(db, other))


def test_blank_answers_are_half_weight_evidence_and_use_the_submission_time(db, seeded, student):
    publish_bank(db)
    take_test(db, student, lambda q: None, now=T0)
    evidence = load_evidence(db, student)
    assert all(e.weight == UNANSWERED_WEIGHT and not e.correct for e in evidence)
    assert all(e.when == e.submitted_at == T0 + timedelta(minutes=5) for e in evidence)
    take_test(db, student, lambda q: True, now=T0 + timedelta(days=1))
    answered = [e for e in load_evidence(db, student) if e.correct]
    assert all(e.weight == 1.0 and e.when == T0 + timedelta(days=1, minutes=1) for e in answered)
