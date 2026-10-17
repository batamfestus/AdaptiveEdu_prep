"""
core/orm.py
-----------
Database tables, written as Python classes (SQLAlchemy 2.0 style).

Tables:
    users            -> who can log in (student or admin)
    topics           -> e.g. "Current Electricity"
    competencies     -> skills inside a topic, e.g. "Apply Ohm's law"
    questions        -> the question bank (with a review status)
    question_reviews -> audit log of everything that happens to a question (Lesson 3)
    question_images  -> diagrams attached to questions or to their options (Lesson 4)
    exam_sessions    -> NEW in Lesson 5: one row per test a student starts
    exam_items       -> the questions in a test, with the student's answers (Lesson 5)
    tutor_*          -> NEW in Lesson 6: cached AI explanations, usage log, saved chat threads and messages

The learning-loop chain from the job description starts here:
    exam -> topic -> competency -> question  (attempts/mastery come later)
"""
from __future__ import annotations

import enum
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, Boolean, DateTime, Enum, Float, ForeignKey, Index, Integer, LargeBinary, String, Text,
    UniqueConstraint, text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware 'now'. Always store UTC; convert only when displaying."""
    return datetime.now(timezone.utc)


class Base(DeclarativeBase):
    """All table classes inherit from this so SQLAlchemy can track them."""


# ---------- Enums (fixed sets of allowed values) ----------
class Role(str, enum.Enum):
    STUDENT = "student"
    ADMIN = "admin"


class QuestionStatus(str, enum.Enum):
    """The review pipeline. A student only ever sees PUBLISHED questions.
    RETIRED hides a question without deleting it, so old attempts stay valid."""
    DRAFT = "draft"
    PENDING_REVIEW = "pending_review"
    APPROVED = "approved"
    PUBLISHED = "published"
    RETIRED = "retired"


class SessionStatus(str, enum.Enum):
    IN_PROGRESS = "in_progress"
    SUBMITTED = "submitted"      # graded (by the student, or automatically when time ran out)
    ABANDONED = "abandoned"      # discarded by the student; never graded or counted


# Content may only be edited while a question is still being prepared. Once a
# question is approved or published, it must stay exactly as it was reviewed.
EDITABLE_STATUSES = {QuestionStatus.DRAFT, QuestionStatus.PENDING_REVIEW}


# ---------- Tables ----------
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    full_name: Mapped[str] = mapped_column(String(120))
    # We NEVER store plain passwords, only a hash (see core/auth.py).
    password_hash: Mapped[str] = mapped_column(String(255))
    # native_enum=False stores the role as plain text. That avoids painful
    # migrations on Postgres if we add a new role later.
    role: Mapped[Role] = mapped_column(Enum(Role, native_enum=False), default=Role.STUDENT)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    # Relative importance in the real exam. The adaptive scheduler will use
    # this later ("exam weighting" in the job description).
    exam_weight: Mapped[float] = mapped_column(Float, default=1.0)

    competencies: Mapped[list[Competency]] = relationship(
        back_populates="topic", cascade="all, delete-orphan"
    )


class Competency(Base):
    __tablename__ = "competencies"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), index=True)
    name: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text, default="")

    topic: Mapped[Topic] = relationship(back_populates="competencies")
    questions: Mapped[list[Question]] = relationship(back_populates="competency")


class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    competency_id: Mapped[int] = mapped_column(ForeignKey("competencies.id"), index=True)

    text: Mapped[str] = mapped_column(Text)
    # Options live in one JSON column: {"A": "...", "B": "...", ...}.
    # Flexible if another exam uses 5 options, and one field to validate.
    options: Mapped[dict] = mapped_column(JSON)
    correct_option: Mapped[str] = mapped_column(String(1))  # "A".."D"
    difficulty: Mapped[int] = mapped_column(Integer, default=3)  # 1 (easy) .. 5 (hard)
    explanation: Mapped[str] = mapped_column(Text, default="")
    image_path: Mapped[str | None] = mapped_column(String(255), nullable=True)

    # Provenance: where did this question come from? (quality control + honesty)
    source: Mapped[str] = mapped_column(String(30), default="original")
    status: Mapped[QuestionStatus] = mapped_column(
        Enum(QuestionStatus, native_enum=False), default=QuestionStatus.DRAFT, index=True
    )
    version: Mapped[int] = mapped_column(Integer, default=1)

    created_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    reviewed_by: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    competency: Mapped[Competency] = relationship(back_populates="questions")


class QuestionReview(Base):
    """Audit log: ONE ROW PER EVENT in a question's life (imported, approved,
    rejected, edited, published, retired...).

    Why a separate table instead of a `review_note` column on questions?
      1. A column keeps only the LAST note; a log keeps the whole history.
      2. A brand-new table is created automatically by `create_all`, while
         adding a column to an existing table would need a migration (ALTER TABLE).
    Rows are only ever ADDED, never edited, which is what makes it an audit trail.
    """
    __tablename__ = "question_reviews"

    id: Mapped[int] = mapped_column(primary_key=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), index=True)
    actor_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(20))        # "import", "approve", "reject", ...
    from_status: Mapped[str] = mapped_column(String(20))
    to_status: Mapped[str] = mapped_column(String(20))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class QuestionImage(Base):
    """A diagram attached to a question's stem or to one of its options.

    DESIGN CHOICE: images are stored IN THE DATABASE (bytea in Postgres), not in a
    separate file service. For a question bank of a few hundred small diagrams
    this is the simplest thing that works: no extra service or secret key, it is
    saved in the same transaction as the question, it works the same on SQLite
    locally, and it is covered by the same Row Level Security. All access goes
    through core/images.py, so moving to Supabase Storage later only changes that
    one file.

    We avoided changing the `questions` table (which would need a migration) by
    linking images to questions with a foreign key from THIS new table instead.
    (The unused `questions.image_path` column from Lesson 1 can be dropped when we
    learn migrations.)
    """
    __tablename__ = "question_images"
    # At most ONE image per role per question: "stem", "A", "B", "C" or "D".
    __table_args__ = (UniqueConstraint("question_id", "role", name="uq_question_image_role"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))
    filename: Mapped[str] = mapped_column(String(120), default="")
    content_type: Mapped[str] = mapped_column(String(30))
    width: Mapped[int] = mapped_column(Integer)
    height: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64))
    # Describes the diagram in words: for screen readers, for reviewers, and as
    # context for the AI tutor. Required, so no diagram is ever "invisible".
    alt_text: Mapped[str] = mapped_column(Text)
    # deferred=True: listing questions does NOT download every image's bytes.
    data: Mapped[bytes] = mapped_column(LargeBinary, deferred=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class ExamSession(Base):
    """One test taken by one student (a timed set of questions).

    (Named ExamSession, not TestSession, because pytest treats any class whose
    name starts with "Test" as a test to collect.)

    The deadline is stored as an absolute time ON THE SERVER. The browser only
    DISPLAYS a countdown; it can never extend or shorten the real deadline.
    """
    __tablename__ = "exam_sessions"
    __table_args__ = (
        # A DATABASE-level rule: one student can have at most ONE in-progress test.
        # Even if two browser tabs click "Start" at the same instant, the second
        # insert is refused by the database itself.
        Index("uq_one_active_exam_per_user", "user_id", unique=True,
              sqlite_where=text("status = 'in_progress'"),
              postgresql_where=text("status = 'in_progress'")),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(160))
    mode: Mapped[str] = mapped_column(String(20))                       # "topic" or "mixed"
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"), nullable=True)
    # values_callable stores the readable value ("in_progress"), which the index above relies on.
    status: Mapped[SessionStatus] = mapped_column(
        Enum(SessionStatus, native_enum=False, length=20, values_callable=lambda e: [m.value for m in e]),
        default=SessionStatus.IN_PROGRESS,
    )
    n_questions: Mapped[int] = mapped_column(Integer)
    duration_seconds: Mapped[int] = mapped_column(Integer)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deadline_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    auto_submitted: Mapped[bool] = mapped_column(Boolean, default=False)  # True = time ran out
    score: Mapped[int | None] = mapped_column(Integer, nullable=True)     # filled in when graded


class ExamItem(Base):
    """One question inside one test, plus the student's answer to it.

    The set and ORDER of questions is fixed when the test starts, so a refresh
    (or coming back tomorrow) shows exactly the same test.
    """
    __tablename__ = "exam_items"
    __table_args__ = (UniqueConstraint("session_id", "position", name="uq_exam_item_position"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("exam_sessions.id"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), index=True)
    position: Mapped[int] = mapped_column(Integer)                        # 1-based question number
    selected_option: Mapped[str | None] = mapped_column(String(1), nullable=True)
    flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    answered_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)  # NULL until graded


# ============================================================ AI tutor tables (Lesson 6)
class TutorExplanationRow(Base):
    """A saved AI explanation, SHARED by every student.

    The explanation of "why option C is wrong for question 42" doesn't depend on
    WHO chose C. So we generate it once and every later student gets it from the
    database for free. This one table is the biggest credit saver in the project.
    """
    __tablename__ = "tutor_explanations"
    __table_args__ = (UniqueConstraint("question_id", "selected_option", name="uq_tutor_explanation"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"), index=True)
    selected_option: Mapped[str] = mapped_column(String(1))     # "A".."D", or "-" for "no answer"
    question_version: Mapped[int] = mapped_column(Integer)      # a changed question invalidates old explanations
    model: Mapped[str] = mapped_column(String(60))
    content: Mapped[dict] = mapped_column(JSON)                 # the validated structured explanation
    needs_review: Mapped[bool] = mapped_column(Boolean, default=False)  # the AI doubts the answer key
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class TutorUsage(Base):
    """One row per PAID AI call (cached answers cost nothing and aren't logged here).
    The daily limits are computed by counting these rows."""
    __tablename__ = "tutor_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    kind: Mapped[str] = mapped_column(String(12))               # "explain" or "chat"
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, index=True)


class TutorThread(Base):
    """A saved conversation about ONE question in ONE of a student's tests."""
    __tablename__ = "tutor_threads"
    __table_args__ = (UniqueConstraint("user_id", "session_id", "question_id", name="uq_tutor_thread"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), index=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("exam_sessions.id"))
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id"))
    title: Mapped[str] = mapped_column(String(160))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class TutorMessage(Base):
    """One message in a thread. Only what the student SEES is saved (their question and
    the tutor's final answer), never the behind-the-scenes tool calls."""
    __tablename__ = "tutor_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    thread_id: Mapped[int] = mapped_column(ForeignKey("tutor_threads.id"), index=True)
    role: Mapped[str] = mapped_column(String(10))               # "user" or "assistant"
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
