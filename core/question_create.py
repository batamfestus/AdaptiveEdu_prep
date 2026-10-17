"""
core/question_create.py
-----------------------
Create ONE question (with optional diagrams) from the admin's form.

Compare with core/question_import.py, which handles many questions from a CSV.
Both end the same way: the question is saved as PENDING_REVIEW with an audit
entry, and a human must approve it before it can go live.
"""
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.images import (
    VALID_ROLES, ImageError, attach_image, process_image, validate_alt_text,
)
from core.orm import Competency, Question, QuestionReview, QuestionStatus, Topic
from core.permissions import ensure_admin
from core.question_import import normalise_text
from core.schemas import QuestionIn


class QuestionCreateError(Exception):
    """A rule was broken. The message is safe to show the admin."""


@dataclass
class ImageUpload:
    role: str          # "stem", "A", "B", "C" or "D"
    filename: str
    data: bytes
    alt_text: str


def find_competency_id(db: Session, topic: str, competency: str) -> int | None:
    return db.scalar(
        select(Competency.id)
        .join(Topic, Topic.id == Competency.topic_id)
        .where(func.lower(Topic.name) == topic.strip().lower(),
               func.lower(Competency.name) == competency.strip().lower())
    )


def create_question(db: Session, actor: CurrentUser, payload: QuestionIn,
                    images: list[ImageUpload] | None = None) -> int:
    """Validate everything FIRST, then save the question and its images in one transaction."""
    ensure_admin(actor)

    competency_id = find_competency_id(db, payload.topic, payload.competency)
    if competency_id is None:
        raise QuestionCreateError(f"'{payload.competency}' is not a competency under '{payload.topic}'.")

    key = normalise_text(payload.text)
    for existing_id, existing_text in db.execute(select(Question.id, Question.text)):
        if normalise_text(existing_text) == key:
            raise QuestionCreateError(f"This looks like a duplicate of question #{existing_id}.")

    # Clean every image BEFORE writing anything: a bad image must not leave a half-saved question.
    prepared, seen_roles = [], set()
    for upload in images or []:
        if upload.role not in VALID_ROLES:
            raise ImageError(f"Unknown image position '{upload.role}'.")
        if upload.role in seen_roles:
            raise ImageError(f"Two images were given for position '{upload.role}'. Only one is allowed.")
        seen_roles.add(upload.role)
        label = "the question" if upload.role == "stem" else f"option {upload.role}"
        try:
            prepared.append((upload, process_image(upload.data), validate_alt_text(upload.alt_text)))
        except ImageError as e:
            raise ImageError(f"Image for {label}: {e}")

    try:
        question = Question(
            competency_id=competency_id, text=payload.text, options=payload.options,
            correct_option=payload.correct_option, difficulty=payload.difficulty,
            explanation=payload.explanation, source=payload.source,
            status=QuestionStatus.PENDING_REVIEW, created_by=actor.id,
        )
        db.add(question)
        db.flush()  # gives the question an id so images and the log can point to it
        for upload, processed, alt in prepared:
            attach_image(db, question.id, upload.role, processed, alt, upload.filename)
        db.add(QuestionReview(
            question_id=question.id, actor_id=actor.id, action="create",
            from_status="-", to_status=QuestionStatus.PENDING_REVIEW.value,
            note="Created with the form" + (f" ({len(prepared)} image(s))" if prepared else ""),
        ))
        db.commit()
    except Exception:
        db.rollback()
        raise
    return question.id
