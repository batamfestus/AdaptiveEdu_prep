"""Tests for core/question_create.py (the single-question form)."""
import io

import pytest
from PIL import Image
from sqlalchemy import select

from core.images import ImageError
from core.orm import Question, QuestionImage, QuestionReview, QuestionStatus
from core.permissions import PermissionDenied
from core.question_create import ImageUpload, QuestionCreateError, create_question
from core.schemas import QuestionIn

ALT = "Circuit diagram: a 12 V battery across a 6 ohm and a 3 ohm resistor in parallel."


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (120, 80), "white").save(buf, "PNG")
    return buf.getvalue()


def payload(**overrides) -> QuestionIn:
    data = dict(
        topic="Current Electricity", competency="Apply Ohm's law",
        text="What is the total current drawn from the battery in the circuit shown?",
        options={"A": "2 A", "B": "4 A", "C": "6 A", "D": "9 A"},
        correct_option="C", difficulty=3, explanation="I = V/R = 12/2 = 6 A.", source="original",
    )
    data.update(overrides)
    return QuestionIn(**data)


def test_create_without_images(db, seeded, admin):
    qid = create_question(db, admin, payload())
    q = db.get(Question, qid)
    assert q.status == QuestionStatus.PENDING_REVIEW and q.created_by == admin.id
    assert db.scalars(select(QuestionReview)).one().action == "create"


def test_create_with_stem_and_option_diagrams(db, seeded, admin):
    uploads = [ImageUpload("stem", "circuit.png", png(), ALT),
               ImageUpload("B", "optionb.png", png(), "Option B shows two resistors in series.")]
    qid = create_question(db, admin, payload(), uploads)
    roles = sorted(i.role for i in db.scalars(select(QuestionImage).where(QuestionImage.question_id == qid)))
    assert roles == ["B", "stem"]
    assert "(2 image(s))" in db.scalars(select(QuestionReview)).one().note


def test_competency_must_belong_to_topic(db, seeded, admin):
    with pytest.raises(QuestionCreateError, match="not a competency"):
        create_question(db, admin, payload(competency="Use wave quantities"))


def test_duplicate_question_rejected(db, seeded, admin):
    create_question(db, admin, payload())
    with pytest.raises(QuestionCreateError, match="duplicate"):
        create_question(db, admin, payload(text="  WHAT is the total current drawn from the battery in the circuit shown  "))


def test_bad_image_saves_nothing_at_all(db, seeded, admin):
    with pytest.raises(ImageError, match="option C"):
        create_question(db, admin, payload(), [ImageUpload("stem", "ok.png", png(), ALT),
                                               ImageUpload("C", "bad.png", b"not an image", ALT)])
    assert db.scalars(select(Question)).all() == []       # not even the question
    assert db.scalars(select(QuestionImage)).all() == []


def test_missing_description_is_refused(db, seeded, admin):
    with pytest.raises(ImageError, match="the question"):
        create_question(db, admin, payload(), [ImageUpload("stem", "c.png", png(), "")])


def test_two_images_for_one_position_refused(db, seeded, admin):
    with pytest.raises(ImageError, match="Only one"):
        create_question(db, admin, payload(), [ImageUpload("stem", "a.png", png(), ALT),
                                               ImageUpload("stem", "b.png", png(), ALT)])


def test_student_cannot_create(db, seeded, student):
    with pytest.raises(PermissionDenied):
        create_question(db, student, payload())
    assert db.scalars(select(Question)).all() == []
