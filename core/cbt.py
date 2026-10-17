"""
core/cbt.py
-----------
The computer-based test (CBT) engine: start a timed test, save answers, grade it,
and show results. No Streamlit in here, so every rule can be unit-tested.

THE IDEAS TO UNDERSTAND (all of them are interview-worthy):

1. THE SERVER OWNS THE CLOCK. The deadline is stored in the database. The browser
   shows a countdown, but only the server decides when time is up. Expiry is
   checked "lazily": whenever anyone touches an expired test, it is graded then.

2. THE ANSWER KEY NEVER LEAVES THE SERVER DURING A TEST. The objects we send to the
   UI while a test is running (ExamQuestion) simply have no `correct_option` or
   `explanation` field. You can't leak what you never loaded.

3. EVERY ACTION CHECKS OWNERSHIP. The UI sends an id ("save an answer on item 57").
   An attacker can change that number, so the core re-checks that the item belongs to
   the logged-in student. Skipping this check is a classic bug called IDOR
   (insecure direct object reference). Not-yours and not-found give the SAME error,
   so ids can't be probed.

4. SAVE EVERY ANSWER IMMEDIATELY. Nothing lives only in the browser, so a refresh,
   a dropped connection or a closed laptop loses nothing.

5. INJECT THE CLOCK. Functions accept an optional `now`, so tests can travel through
   time instead of waiting for a real timer.
"""
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.images import ImageInfo, list_image_infos
from core.orm import (
    Competency, ExamItem, ExamSession, Question, QuestionStatus, SessionStatus, Topic, utcnow,
)
from core.planner import pick_adaptive_questions, pick_diagnostic_questions
from core.timeutil import as_utc  # noqa: F401  (still importable from core.cbt)

SECONDS_PER_QUESTION = 90      # 40 questions -> 60 minutes
GRACE_SECONDS = 5              # lets an answer clicked at the last moment still count
MAX_QUESTIONS = 40
QUESTION_COUNTS = (5, 10, 20, 40)
OPTION_KEYS = ("A", "B", "C", "D")
MODES = ("topic", "mixed", "diagnostic", "adaptive")
MODE_TITLES = {"diagnostic": "Diagnostic test", "adaptive": "Adaptive practice", "mixed": "Mixed test"}


class CbtError(Exception):
    """A rule was broken. The message is safe to show the student."""


class TimeUp(CbtError):
    """The deadline passed. By the time you see this, the test has already been graded."""
    def __init__(self, session_id: int):
        super().__init__("Time is up. Your test was submitted automatically.")
        self.session_id = session_id


# ============================================================ what the UI receives
@dataclass(frozen=True)
class ExamQuestion:
    """A question AS SEEN DURING A TEST. Note what is missing: no correct answer, no explanation."""
    item_id: int
    position: int
    text: str
    options: dict
    images: list[ImageInfo]
    selected: str | None
    flagged: bool


@dataclass(frozen=True)
class ActiveExam:
    session_id: int
    title: str
    deadline_at: datetime
    items: list[ExamQuestion]


@dataclass(frozen=True)
class ResultItem:
    """A question AFTER the test is graded, so now the answer and explanation are included."""
    position: int
    topic: str
    competency: str
    text: str
    options: dict
    images: list[ImageInfo]
    selected: str | None
    correct_option: str
    is_correct: bool
    flagged: bool
    explanation: str


@dataclass(frozen=True)
class TopicScore:
    topic: str
    correct: int
    total: int

    @property
    def percent(self) -> float:
        return round(100 * self.correct / self.total, 1) if self.total else 0.0


@dataclass(frozen=True)
class ExamResult:
    session_id: int
    title: str
    mode: str
    started_at: datetime
    submitted_at: datetime
    auto_submitted: bool
    score: int
    total: int
    items: list[ResultItem]
    by_topic: list[TopicScore]

    @property
    def percent(self) -> float:
        return round(100 * self.score / self.total, 1) if self.total else 0.0

    @property
    def seconds_taken(self) -> int:
        return max(0, int((self.submitted_at - self.started_at).total_seconds()))


@dataclass(frozen=True)
class SessionSummary:
    id: int
    title: str
    mode: str
    started_at: datetime
    score: int
    total: int
    auto_submitted: bool

    @property
    def percent(self) -> float:
        return round(100 * self.score / self.total, 1) if self.total else 0.0


# ============================================================ small helpers
def _own_session(db: Session, session_id: int, user: CurrentUser) -> ExamSession:
    session = db.get(ExamSession, session_id)
    if session is None or session.user_id != user.id:
        raise CbtError("Test not found.")  # same message either way: no probing other students' ids
    return session


def _own_active_item(db: Session, item_id: int, user: CurrentUser, now: datetime) -> tuple[ExamItem, ExamSession]:
    """Fetch an item the student may change right now, or raise."""
    item = db.get(ExamItem, item_id)
    if item is None:
        raise CbtError("Question not found.")
    session = _own_session(db, item.session_id, user)
    if session.status != SessionStatus.IN_PROGRESS:
        raise CbtError("This test is already finished.")
    if now >= as_utc(session.deadline_at) + timedelta(seconds=GRACE_SECONDS):
        _grade(db, session, now)
        db.commit()
        raise TimeUp(session.id)
    return item, session


# ============================================================ choosing questions
def available_counts(db: Session) -> dict[str, int]:
    """Published questions per topic. Only PUBLISHED questions can ever appear in a test."""
    rows = db.execute(
        select(Topic.name, func.count(Question.id))
        .join(Competency, Competency.topic_id == Topic.id)
        .join(Question, Question.competency_id == Competency.id)
        .where(Question.status == QuestionStatus.PUBLISHED)
        .group_by(Topic.name).order_by(Topic.name)
    ).all()
    return {name: count for name, count in rows}


def select_questions(db: Session, topic_name: str | None, n: int, rng: random.Random) -> list[int]:
    """Pick n published question ids. For a mixed test, cycle through the topics so
    one big topic can't crowd out the others; then shuffle so topics aren't grouped."""
    stmt = (
        select(Question.id, Topic.name)
        .join(Competency, Competency.id == Question.competency_id)
        .join(Topic, Topic.id == Competency.topic_id)
        .where(Question.status == QuestionStatus.PUBLISHED)
        .order_by(Question.id)  # a fixed starting order, so a seeded rng gives reproducible tests
    )
    if topic_name:
        stmt = stmt.where(Topic.name == topic_name)

    by_topic: dict[str, list[int]] = {}
    for question_id, topic in db.execute(stmt):
        by_topic.setdefault(topic, []).append(question_id)
    for ids in by_topic.values():
        rng.shuffle(ids)

    chosen: list[int] = []
    topics = list(by_topic)
    rng.shuffle(topics)
    while len(chosen) < n and any(by_topic.values()):
        for topic in topics:
            if by_topic[topic] and len(chosen) < n:
                chosen.append(by_topic[topic].pop())
    rng.shuffle(chosen)
    return chosen


# ============================================================ starting and finding tests
def check_active(db: Session, user: CurrentUser, now: datetime | None = None) -> tuple[int | None, int | None]:
    """Returns (active_session_id, just_expired_session_id).

    If the student's test ran out of time while they were away, it is graded NOW
    and its id comes back in the second slot so the UI can tell them."""
    now = now or utcnow()
    session = db.scalar(select(ExamSession).where(
        ExamSession.user_id == user.id, ExamSession.status == SessionStatus.IN_PROGRESS))
    if session is None:
        return None, None
    if now >= as_utc(session.deadline_at):
        _grade(db, session, now)
        db.commit()
        return None, session.id
    return session.id, None


def start_test(db: Session, user: CurrentUser, mode: str, topic_name: str | None, n_questions: int,
               *, rng: random.Random | None = None, now: datetime | None = None) -> int:
    if mode not in MODES:
        raise CbtError("Unknown test type.")
    if not 1 <= n_questions <= MAX_QUESTIONS:
        raise CbtError(f"Choose between 1 and {MAX_QUESTIONS} questions.")
    now = now or utcnow()

    active, _ = check_active(db, user, now)
    if active is not None:
        raise CbtError("You already have a test in progress. Finish or discard it first.")

    topic = None
    if mode == "topic":
        topic = db.scalar(select(Topic).where(Topic.name == topic_name))
        if topic is None:
            raise CbtError("Please choose a topic.")

    rng = rng or random.Random()
    if mode == "diagnostic":        # a first look at every competency
        ids = pick_diagnostic_questions(db, user, rng, now)
    elif mode == "adaptive":        # aimed at this student's biggest opportunities
        ids = pick_adaptive_questions(db, user, n_questions, rng, now)
    else:
        ids = select_questions(db, topic.name if topic else None, n_questions, rng)
    if not ids:
        raise CbtError("There are no published questions for this test yet.")

    duration = len(ids) * SECONDS_PER_QUESTION
    session = ExamSession(
        user_id=user.id, title=f"{topic.name} test" if topic else MODE_TITLES[mode], mode=mode,
        topic_id=topic.id if topic else None, status=SessionStatus.IN_PROGRESS,
        n_questions=len(ids), duration_seconds=duration,
        started_at=now, deadline_at=now + timedelta(seconds=duration),
    )
    db.add(session)
    try:
        db.flush()  # the database's one-active-test-per-student rule is enforced here
    except IntegrityError:
        db.rollback()
        raise CbtError("You already have a test in progress. Finish or discard it first.")
    for position, question_id in enumerate(ids, start=1):
        db.add(ExamItem(session_id=session.id, question_id=question_id, position=position))
    db.commit()
    return session.id


def load_active_test(db: Session, user: CurrentUser, session_id: int, now: datetime | None = None) -> ActiveExam:
    """Everything the UI needs to draw a running test, with NO answer key in it."""
    now = now or utcnow()
    session = _own_session(db, session_id, user)
    if session.status != SessionStatus.IN_PROGRESS:
        raise CbtError("This test is already finished.")
    if now >= as_utc(session.deadline_at):
        _grade(db, session, now)
        db.commit()
        raise TimeUp(session.id)

    # Select only the columns a student may see. correct_option and explanation are never loaded.
    rows = db.execute(
        select(ExamItem.id, ExamItem.position, ExamItem.selected_option, ExamItem.flagged,
               Question.id, Question.text, Question.options)
        .join(Question, Question.id == ExamItem.question_id)
        .where(ExamItem.session_id == session_id)
        .order_by(ExamItem.position)
    ).all()
    images = list_image_infos(db, [row[4] for row in rows])
    items = [
        ExamQuestion(item_id=item_id, position=position, text=text, options=options,
                     images=images.get(question_id, []), selected=selected, flagged=bool(flagged))
        for item_id, position, selected, flagged, question_id, text, options in rows
    ]
    return ActiveExam(session.id, session.title, as_utc(session.deadline_at), items)


# ============================================================ answering
def save_answer(db: Session, user: CurrentUser, item_id: int, option: str | None,
                now: datetime | None = None) -> None:
    """Save (or clear, with None) the student's answer. Called on every click."""
    now = now or utcnow()
    item, _ = _own_active_item(db, item_id, user, now)
    if option is not None:
        option = option.strip().upper()
        if option not in OPTION_KEYS:
            raise CbtError("That is not a valid option.")
    item.selected_option = option
    item.answered_at = now if option else None
    db.commit()


def set_flag(db: Session, user: CurrentUser, item_id: int, flagged: bool, now: datetime | None = None) -> None:
    item, _ = _own_active_item(db, item_id, user, now or utcnow())
    item.flagged = bool(flagged)
    db.commit()


# ============================================================ grading
def _grade(db: Session, session: ExamSession, now: datetime) -> None:
    """Mark every answer and close the test. Does NOT commit: the caller does."""
    deadline = as_utc(session.deadline_at)
    items = db.scalars(select(ExamItem).where(ExamItem.session_id == session.id)).all()
    correct = {
        qid: answer for qid, answer in db.execute(
            select(Question.id, Question.correct_option).where(Question.id.in_([i.question_id for i in items]))
        )
    }
    score = 0
    for item in items:
        # An unanswered question is simply wrong (there is no negative marking).
        item.is_correct = item.selected_option is not None and item.selected_option == correct[item.question_id]
        score += int(item.is_correct)

    session.score = score
    session.status = SessionStatus.SUBMITTED
    session.auto_submitted = now >= deadline
    session.submitted_at = min(now, deadline)  # a late discovery doesn't make the test "longer"


def submit_test(db: Session, user: CurrentUser, session_id: int, now: datetime | None = None) -> ExamResult:
    """Grade the test. Safe to call twice (a double-click): the second call just returns the result."""
    now = now or utcnow()
    session = _own_session(db, session_id, user)
    if session.status == SessionStatus.ABANDONED:
        raise CbtError("This test was discarded.")
    if session.status == SessionStatus.IN_PROGRESS:
        _grade(db, session, now)
        db.commit()
    return _build_result(db, session)


def abandon_test(db: Session, user: CurrentUser, session_id: int) -> None:
    """Throw a test away (it is never graded and never counts toward progress)."""
    session = _own_session(db, session_id, user)
    if session.status != SessionStatus.IN_PROGRESS:
        raise CbtError("Only a test in progress can be discarded.")
    session.status = SessionStatus.ABANDONED
    db.commit()


# ============================================================ results
def get_result(db: Session, user: CurrentUser, session_id: int) -> ExamResult:
    session = _own_session(db, session_id, user)
    if session.status != SessionStatus.SUBMITTED:
        raise CbtError("Results are only available after the test is submitted.")
    return _build_result(db, session)


def _build_result(db: Session, session: ExamSession) -> ExamResult:
    rows = db.execute(
        select(ExamItem, Question, Competency.name, Topic.name)
        .join(Question, Question.id == ExamItem.question_id)
        .join(Competency, Competency.id == Question.competency_id)
        .join(Topic, Topic.id == Competency.topic_id)
        .where(ExamItem.session_id == session.id)
        .order_by(ExamItem.position)
    ).all()
    images = list_image_infos(db, [q.id for _, q, _, _ in rows])

    items, tally = [], {}
    for item, question, competency, topic in rows:
        items.append(ResultItem(
            position=item.position, topic=topic, competency=competency, text=question.text,
            options=question.options, images=images.get(question.id, []), selected=item.selected_option,
            correct_option=question.correct_option, is_correct=bool(item.is_correct),
            flagged=bool(item.flagged), explanation=question.explanation,
        ))
        got, out_of = tally.get(topic, (0, 0))
        tally[topic] = (got + int(bool(item.is_correct)), out_of + 1)

    return ExamResult(
        session_id=session.id, title=session.title, mode=session.mode,
        started_at=as_utc(session.started_at), submitted_at=as_utc(session.submitted_at),
        auto_submitted=bool(session.auto_submitted), score=session.score or 0, total=len(items),
        items=items, by_topic=[TopicScore(t, g, n) for t, (g, n) in sorted(tally.items())],
    )


def list_sessions(db: Session, user: CurrentUser, limit: int = 50) -> list[SessionSummary]:
    """The student's finished tests, newest first. Discarded tests are not shown."""
    sessions = db.scalars(
        select(ExamSession)
        .where(ExamSession.user_id == user.id, ExamSession.status == SessionStatus.SUBMITTED)
        .order_by(ExamSession.started_at.desc(), ExamSession.id.desc()).limit(limit)
    ).all()
    return [
        SessionSummary(s.id, s.title, s.mode, as_utc(s.started_at), s.score or 0, s.n_questions, bool(s.auto_submitted))
        for s in sessions
    ]


def graded_item_ref(db: Session, user: CurrentUser, session_id: int, position: int) -> tuple[int, str | None]:
    """For the AI tutor: which question was number `position` in this FINISHED test, and what
    did the student choose? Same ownership and 'only after submission' rules as get_result."""
    session = _own_session(db, session_id, user)
    if session.status != SessionStatus.SUBMITTED:
        raise CbtError("Results are only available after the test is submitted.")
    item = db.scalar(select(ExamItem).where(ExamItem.session_id == session_id, ExamItem.position == position))
    if item is None:
        raise CbtError("Question not found.")
    return item.question_id, item.selected_option
