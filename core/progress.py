"""
core/progress.py
----------------
Everything the student DASHBOARD needs, built from the mastery model.
Still no Streamlit: the UI only draws what this module returns.

Two techniques worth knowing:

* REPLAY (event sourcing). To draw "readiness over time" we do NOT store old scores. We re-run the
  mastery model on the evidence that existed after each test, pretending it is that moment. The
  chart is therefore always consistent with the current model and can never drift out of date.

* BEFORE / AFTER. After a test, "what changed?" is the profile with the test's answers minus
  the profile with them. The difference is exactly what that test taught us.
"""
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.cbt import CbtError, SessionSummary, _own_session, list_sessions
from core.mastery import (
    CompetencyState, Readiness, compute_readiness, compute_states, load_catalog, load_evidence,
)
from core.orm import (
    Competency, ExamItem, ExamSession, Question, SessionStatus, Topic, TutorExplanationRow, utcnow,
)
from core.planner import PlanItem, build_plan
from core.timeutil import as_utc

HISTORY_LIMIT = 30


# ============================================================ what the dashboard shows
@dataclass(frozen=True)
class HistoryPoint:
    when: datetime
    title: str
    score: float
    low: float
    high: float
    provisional: bool


@dataclass(frozen=True)
class TopicSummary:
    topic: str
    weight: float             # share of the exam
    expected_score: float     # 0..100, weighted over the topic's competencies
    assessed: int
    total: int


@dataclass(frozen=True)
class MistakeNote:
    topic: str
    competency: str
    misconception: str
    times: int


@dataclass(frozen=True)
class Dashboard:
    readiness: Readiness
    states: list[CompetencyState]
    topics: list[TopicSummary]
    plan: list[PlanItem]
    history: list[HistoryPoint]
    recent: list[SessionSummary]
    mistakes: list[MistakeNote]


def topic_summaries(states: list[CompetencyState]) -> list[TopicSummary]:
    by_topic: dict[str, list[CompetencyState]] = {}
    for state in states:
        by_topic.setdefault(state.meta.topic, []).append(state)
    summaries = []
    for topic, items in sorted(by_topic.items()):
        weight = sum(s.meta.weight for s in items)
        score = sum(s.meta.weight * s.expected_score for s in items) / weight if weight else 0.0
        summaries.append(TopicSummary(topic, weight, 100 * score, sum(1 for s in items if s.assessed), len(items)))
    return summaries


def readiness_history(db: Session, user: CurrentUser, limit: int = HISTORY_LIMIT) -> list[HistoryPoint]:
    """One point per finished test: the readiness we would have shown right after it."""
    catalog = load_catalog(db)
    evidence = load_evidence(db, user)
    sessions = db.execute(
        select(ExamSession.id, ExamSession.title, ExamSession.submitted_at)
        .where(ExamSession.user_id == user.id, ExamSession.status == SessionStatus.SUBMITTED)
        .order_by(ExamSession.submitted_at, ExamSession.id)
    ).all()
    points = []
    for _, title, submitted in sessions[-limit:]:
        moment = as_utc(submitted)
        known = [e for e in evidence if e.submitted_at <= moment]       # only what existed THEN
        ready = compute_readiness(compute_states(catalog, known, moment))
        points.append(HistoryPoint(moment, title, ready.score, ready.low, ready.high, ready.provisional))
    return points


def recent_mistakes(db: Session, user: CurrentUser, limit: int = 5) -> list[MistakeNote]:
    """Misconceptions the AI tutor has already identified for this student's wrong answers.
    (This closes the loop: AI explanation -> what to watch out for next time.)"""
    rows = db.execute(
        select(Topic.name, Competency.name, TutorExplanationRow.content)
        .select_from(ExamItem)
        .join(ExamSession, ExamSession.id == ExamItem.session_id)
        .join(Question, Question.id == ExamItem.question_id)
        .join(Competency, Competency.id == Question.competency_id)
        .join(Topic, Topic.id == Competency.topic_id)
        .join(TutorExplanationRow, (TutorExplanationRow.question_id == ExamItem.question_id)
              & (TutorExplanationRow.selected_option == ExamItem.selected_option))
        .where(ExamSession.user_id == user.id, ExamSession.status == SessionStatus.SUBMITTED,
               ExamItem.is_correct.is_(False))
        .order_by(ExamSession.submitted_at.desc())
    ).all()
    seen: dict[tuple[str, str], MistakeNote] = {}
    for topic, competency, content in rows:
        text = (content or {}).get("misconception", "").strip()
        if not text:
            continue
        key = (competency, text.lower())
        old = seen.get(key)
        seen[key] = MistakeNote(topic, competency, text, (old.times if old else 0) + 1)
    return list(seen.values())[:limit]          # newest first, because of the ordering above


def build_dashboard(db: Session, user: CurrentUser, now: datetime | None = None) -> Dashboard:
    now = now or utcnow()
    states = compute_states(load_catalog(db), load_evidence(db, user), now)
    return Dashboard(
        readiness=compute_readiness(states), states=states, topics=topic_summaries(states),
        plan=build_plan(states, now), history=readiness_history(db, user),
        recent=list_sessions(db, user, limit=5), mistakes=recent_mistakes(db, user),
    )


# ============================================================ "what did this test change?"
@dataclass(frozen=True)
class CompetencyChange:
    topic: str
    competency: str
    before: float             # mastery before the test (0..1)
    after: float
    answered: int             # questions on this competency in the test
    correct: int


@dataclass(frozen=True)
class ProfileChange:
    before: Readiness
    after: Readiness
    changes: list[CompetencyChange]     # biggest movement first


def profile_change(db: Session, user: CurrentUser, session_id: int) -> ProfileChange:
    session = _own_session(db, session_id, user)           # ownership check: only YOUR tests
    if session.status != SessionStatus.SUBMITTED:
        raise CbtError("Results are only available after the test is submitted.")
    moment = as_utc(session.submitted_at)

    catalog = load_catalog(db)
    evidence = load_evidence(db, user)
    this_test = [e for e in evidence if e.session_id == session_id]
    earlier = [e for e in evidence if e.session_id != session_id and e.submitted_at < moment]

    before = compute_states(catalog, earlier, moment)
    after = compute_states(catalog, earlier + this_test, moment)
    touched = {e.competency_id for e in this_test}
    changes = []
    for b, a in zip(before, after):
        if a.meta.id in touched:
            mine = [e for e in this_test if e.competency_id == a.meta.id]
            changes.append(CompetencyChange(a.meta.topic, a.meta.name, b.mastery, a.mastery,
                                            len(mine), sum(1 for e in mine if e.correct)))
    changes.sort(key=lambda c: -abs(c.after - c.before))
    return ProfileChange(compute_readiness(before), compute_readiness(after), changes)
