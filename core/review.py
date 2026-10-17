"""
core/review.py
--------------
The review queue's brain: who may move a question from which status to which.

This is a small STATE MACHINE. Every allowed move is listed in ACTION_RULES;
anything not listed is refused. Putting the rules in ONE table (instead of
scattered if-statements) makes them easy to read, test and change.

    draft --submit--> pending_review --approve--> approved --publish--> published --retire--> retired
      ^                    |                         |                                          |
      +------reject--------+                         +--send_back--> pending_review   restore --+--> pending_review
"""
from dataclasses import dataclass, field
from datetime import datetime

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.images import ImageInfo, list_image_infos
from core.orm import (
    EDITABLE_STATUSES, Competency, Question, QuestionReview, QuestionStatus, Topic, User,
)
from core.permissions import ensure_admin

S = QuestionStatus

# action -> (statuses it may start from, status it leads to, is a written reason required?)
ACTION_RULES: dict[str, tuple[set[QuestionStatus], QuestionStatus, bool]] = {
    "submit":    ({S.DRAFT}, S.PENDING_REVIEW, False),
    "approve":   ({S.PENDING_REVIEW}, S.APPROVED, False),
    "reject":    ({S.PENDING_REVIEW}, S.DRAFT, True),
    "publish":   ({S.APPROVED}, S.PUBLISHED, False),
    "send_back": ({S.APPROVED}, S.PENDING_REVIEW, True),
    "retire":    ({S.PUBLISHED}, S.RETIRED, False),
    "restore":   ({S.RETIRED}, S.PENDING_REVIEW, False),
}

MIN_NOTE_LENGTH = 5


class ReviewError(Exception):
    """A rule was broken. The message is safe to show the admin."""


class InvalidTransition(ReviewError):
    pass


# ---------- plain data handed to the UI (never ORM objects; see Lesson 2) ----------
@dataclass
class ReviewEntry:
    created_at: datetime
    actor_name: str
    action: str
    to_status: str
    note: str


@dataclass
class QuestionRow:
    id: int
    topic: str
    competency: str
    text: str
    options: dict
    correct_option: str
    difficulty: int
    explanation: str
    source: str
    status: QuestionStatus
    version: int
    created_by_name: str
    history: list[ReviewEntry] = field(default_factory=list)
    images: list[ImageInfo] = field(default_factory=list)


def available_actions(status: QuestionStatus) -> list[str]:
    """Which buttons should the UI show for a question in this status?"""
    return [name for name, (allowed_from, _, _) in ACTION_RULES.items() if status in allowed_from]


# ---------- changing things ----------
def transition_question(db: Session, question_id: int, action: str, actor: CurrentUser, note: str = "") -> None:
    ensure_admin(actor)
    rule = ACTION_RULES.get(action)
    if rule is None:
        raise InvalidTransition(f"Unknown action '{action}'.")
    allowed_from, to_status, note_required = rule

    question = db.get(Question, question_id)
    if question is None:
        raise ReviewError("That question no longer exists.")
    if question.status not in allowed_from:
        raise InvalidTransition(f"Cannot {action.replace('_', ' ')} a question that is {question.status.value}.")

    note = note.strip()
    if note_required and len(note) < MIN_NOTE_LENGTH:
        raise ReviewError(f"Please give a reason (at least {MIN_NOTE_LENGTH} characters).")
    # Quality gate: the AI tutor will rely on the explanation, so it must exist.
    if action == "publish" and not question.explanation.strip():
        raise ReviewError("Add an explanation before publishing.")

    previous = question.status
    question.status = to_status
    question.reviewed_by = actor.id
    db.add(QuestionReview(
        question_id=question.id, actor_id=actor.id, action=action,
        from_status=previous.value, to_status=to_status.value, note=note,
    ))
    db.commit()


def edit_explanation(db: Session, question_id: int, actor: CurrentUser, explanation: str) -> None:
    """Edit the explanation of a draft/pending question (it is logged and versioned)."""
    ensure_admin(actor)
    question = db.get(Question, question_id)
    if question is None:
        raise ReviewError("That question no longer exists.")
    if question.status not in EDITABLE_STATUSES:
        raise ReviewError("Only draft or pending questions can be edited. Send it back or retire it first.")
    explanation = explanation.strip()
    if not explanation:
        raise ReviewError("The explanation cannot be empty.")

    question.explanation = explanation
    question.version += 1
    db.add(QuestionReview(
        question_id=question.id, actor_id=actor.id, action="edit",
        from_status=question.status.value, to_status=question.status.value,
        note=f"Explanation edited (now version {question.version})",
    ))
    db.commit()


# ---------- reading things ----------
def count_by_status(db: Session) -> dict[QuestionStatus, int]:
    rows = db.execute(select(Question.status, func.count(Question.id)).group_by(Question.status)).all()
    return {status: n for status, n in rows}


def list_questions(db: Session, status: QuestionStatus, topic_name: str | None = None, limit: int = 50) -> list[QuestionRow]:
    stmt = (
        select(Question, Competency.name, Topic.name, User.full_name)
        .join(Competency, Competency.id == Question.competency_id)
        .join(Topic, Topic.id == Competency.topic_id)
        .outerjoin(User, User.id == Question.created_by)   # outer: creator may be missing
        .where(Question.status == status)
        .order_by(Question.id)                              # oldest first: a fair queue
        .limit(limit)
    )
    if topic_name:
        stmt = stmt.where(Topic.name == topic_name)
    found = db.execute(stmt).all()
    ids = [q.id for q, *_ in found]
    history = _load_history(db, ids)
    images = list_image_infos(db, ids)

    return [
        QuestionRow(
            id=q.id, topic=topic, competency=comp, text=q.text, options=q.options,
            correct_option=q.correct_option, difficulty=q.difficulty, explanation=q.explanation,
            source=q.source, status=q.status, version=q.version,
            created_by_name=creator or "", history=history.get(q.id, []), images=images.get(q.id, []),
        )
        for q, comp, topic, creator in found
    ]


def _load_history(db: Session, question_ids: list[int]) -> dict[int, list[ReviewEntry]]:
    if not question_ids:
        return {}
    stmt = (
        select(QuestionReview, User.full_name)
        .outerjoin(User, User.id == QuestionReview.actor_id)
        .where(QuestionReview.question_id.in_(question_ids))
        .order_by(QuestionReview.id)
    )
    history: dict[int, list[ReviewEntry]] = {}
    for entry, actor_name in db.execute(stmt):
        history.setdefault(entry.question_id, []).append(
            ReviewEntry(entry.created_at, actor_name or "system", entry.action, entry.to_status, entry.note)
        )
    return history
