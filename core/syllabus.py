"""
core/syllabus.py
----------------
The admin's control over EXAM WEIGHTS: how much of the real exam each topic is worth.
Readiness is a weighted average, so these numbers decide how much each topic matters.
Set them to match the real JAMB paper (for example, the number of questions per topic).
"""
from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.orm import Topic
from core.permissions import ensure_admin

MIN_WEIGHT, MAX_WEIGHT = 0.0, 100.0


class WeightError(Exception):
    """A weight was invalid. The message is safe to show."""


@dataclass(frozen=True)
class TopicWeight:
    id: int
    name: str
    weight: float
    share: float          # this topic's percentage of the whole exam


def list_topic_weights(db: Session) -> list[TopicWeight]:
    topics = db.execute(select(Topic.id, Topic.name, Topic.exam_weight).order_by(Topic.name)).all()
    total = sum(max(0.0, w or 0.0) for _, _, w in topics)
    return [TopicWeight(tid, name, w or 0.0, (100 * (w or 0.0) / total) if total else 100 / len(topics))
            for tid, name, w in topics]


def set_topic_weights(db: Session, actor: CurrentUser, weights: dict[int, float]) -> None:
    """Save new weights for several topics at once. Nothing is saved unless every value is valid."""
    ensure_admin(actor)
    for topic_id, value in weights.items():
        if value != value or not MIN_WEIGHT <= value <= MAX_WEIGHT:      # `value != value` catches NaN
            raise WeightError(f"Weights must be between {MIN_WEIGHT:g} and {MAX_WEIGHT:g}.")
    if all(v == 0 for v in weights.values()) and len(weights) == db.query(Topic).count():
        raise WeightError("At least one topic needs a weight above zero.")
    for topic_id, value in weights.items():
        topic = db.get(Topic, topic_id)
        if topic is None:
            raise WeightError("That topic no longer exists.")
        topic.exam_weight = float(value)
    db.commit()
