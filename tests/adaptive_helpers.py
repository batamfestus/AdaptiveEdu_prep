"""Shared helpers for the adaptive-learning tests: a question bank and a way to take tests quickly."""
import random
from datetime import datetime, timedelta, timezone

from sqlalchemy import select

from core.cbt import save_answer, start_test, submit_test
from core.orm import Competency, ExamItem, Question, QuestionStatus, Topic

T0 = datetime(2026, 10, 1, 9, 0, tzinfo=timezone.utc)


def competency_id(db, name: str) -> int:
    return db.scalars(select(Competency.id).where(Competency.name == name)).one()


def publish_bank(db, per_competency: int = 3, difficulties=(2, 3, 4)) -> None:
    """Published questions for EVERY competency in the seeded syllabus. The right answer is always 'B'."""
    for comp in db.scalars(select(Competency).order_by(Competency.id)):
        for i in range(per_competency):
            db.add(Question(
                competency_id=comp.id, text=f"{comp.name} question {i + 1}?", difficulty=difficulties[i % len(difficulties)],
                options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option="B",
                explanation="Because.", status=QuestionStatus.PUBLISHED))
    db.commit()


def add_question(db, comp: str, difficulty: int, status=QuestionStatus.PUBLISHED, text=None) -> int:
    q = Question(competency_id=competency_id(db, comp), text=text or f"{comp} d{difficulty} {random.random()}",
                 options={"A": "a", "B": "b", "C": "c", "D": "d"}, correct_option="B", difficulty=difficulty,
                 explanation="Because.", status=status)
    db.add(q)
    db.commit()
    return q.id


def take_test(db, user, decide, *, mode="mixed", n=40, topic=None, now=T0, seed=1) -> int:
    """Start a test, answer every question with decide(question) -> True (right) / False (wrong) / None (blank),
    then submit it. Returns the session id."""
    sid = start_test(db, user, mode, topic, n, rng=random.Random(seed), now=now)
    for item in list(db.scalars(select(ExamItem).where(ExamItem.session_id == sid))):
        verdict = decide(db.get(Question, item.question_id))
        if verdict is not None:
            save_answer(db, user, item.id, "B" if verdict else "A", now + timedelta(minutes=1))
    submit_test(db, user, sid, now + timedelta(minutes=5))
    return sid


def comp_name(db, question: Question) -> str:
    return db.get(Competency, question.competency_id).name


def topic_name(db, question: Question) -> str:
    return db.scalars(select(Topic.name).join(Competency, Competency.topic_id == Topic.id)
                      .where(Competency.id == question.competency_id)).one()
