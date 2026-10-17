"""Tests for core/syllabus.py: the admin's exam weights."""
import pytest
from sqlalchemy import select

from core.orm import Topic
from core.permissions import PermissionDenied
from core.syllabus import WeightError, list_topic_weights, set_topic_weights


def ids(db):
    return {t.name: t.id for t in db.scalars(select(Topic))}


def test_shares_add_up_to_100(db, seeded):
    shares = list_topic_weights(db)
    assert sum(w.share for w in shares) == pytest.approx(100) and {w.share for w in shares} == {50.0}


def test_admin_can_change_weights_and_shares_follow(db, seeded, admin):
    topic = ids(db)
    set_topic_weights(db, admin, {topic["Current Electricity"]: 3.0, topic["Motion"]: 1.0})
    shares = {w.name: w.share for w in list_topic_weights(db)}
    assert shares == {"Current Electricity": 75.0, "Motion": 25.0}


@pytest.mark.parametrize("bad", [-1, 101, float("nan"), float("inf")])
def test_invalid_weights_are_refused_and_nothing_changes(db, seeded, admin, bad):
    topic = ids(db)
    with pytest.raises(WeightError, match="between"):
        set_topic_weights(db, admin, {topic["Current Electricity"]: 5.0, topic["Motion"]: bad})
    assert {w.weight for w in list_topic_weights(db)} == {1.0}       # even the valid half was not saved


def test_a_syllabus_cannot_be_weighted_to_nothing(db, seeded, admin):
    topic = ids(db)
    with pytest.raises(WeightError, match="above zero"):
        set_topic_weights(db, admin, {topic["Current Electricity"]: 0.0, topic["Motion"]: 0.0})


def test_one_topic_may_be_zero_if_another_is_not(db, seeded, admin):
    topic = ids(db)
    set_topic_weights(db, admin, {topic["Current Electricity"]: 1.0, topic["Motion"]: 0.0})
    assert {w.name: w.share for w in list_topic_weights(db)}["Motion"] == 0.0


def test_unknown_topics_are_refused(db, seeded, admin):
    with pytest.raises(WeightError, match="no longer exists"):
        set_topic_weights(db, admin, {9999: 2.0})


def test_students_cannot_change_weights(db, seeded, student):
    with pytest.raises(PermissionDenied):
        set_topic_weights(db, student, {ids(db)["Motion"]: 9.0})
    assert {w.weight for w in list_topic_weights(db)} == {1.0}
