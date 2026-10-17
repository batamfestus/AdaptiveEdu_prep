"""
core/mastery.py
---------------
The MASTERY MODEL: from a student's graded answers, estimate how well they know each
competency, how SURE we are, and what score they could expect in the real exam.
No Streamlit and no database writes in here, so every rule can be tested.

THE BIG IDEAS (all interview-worthy):

1. WE STORE EVIDENCE, NOT SCORES. Every graded answer is already in the database. A student's
   profile is RECOMPUTED from that evidence whenever we need it. We never store "mastery = 63%",
   because a stored number goes stale and can't be explained. Bonus: replaying the evidence
   up to an earlier date gives us a progress-over-time chart for free.

2. A BELIEF, NOT A NUMBER. For each competency we keep a probability distribution over the
   student's ability (theta) on a grid of 81 values. Each answer updates it by Bayes' rule.
   From the distribution we get an estimate AND an honest measure of uncertainty. Few answers
   means a wide distribution; many answers means a narrow one.

3. HOW ONE ANSWER COUNTS (a simplified item-response-theory model, "3PL with fixed slope"):
       P(correct) = GUESS + (1 - GUESS) * sigmoid( DISCRIMINATION * (theta - b) )
   - b is the question's difficulty. Getting a HARD question right is strong evidence; getting
     an EASY one wrong is strong evidence of a gap.
   - GUESS = 0.25 because JAMB questions have four options: a blind guess is right a quarter
     of the time, so a correct answer is never proof of knowledge.

4. FORGETTING. Older evidence is down-weighted by 0.5 ** (age / HALF_LIFE_DAYS). Fading evidence
   flattens the belief back towards the prior, so uncertainty GROWS when you stop practising,
   and the planner naturally brings the competency back for a refresh.

5. READINESS = the score we expect on the real exam: each competency's expected score,
   weighted by how much of the exam it is worth (the admin's exam weights).
"""
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.orm import Competency, ExamItem, ExamSession, Question, SessionStatus, Topic
from core.timeutil import as_utc

# ---------------------------------------------------------------- model settings (tune and re-run the tests)
GRID_SIZE = 81
THETA_MIN, THETA_MAX = -4.0, 4.0
GRID = tuple(THETA_MIN + (THETA_MAX - THETA_MIN) * k / (GRID_SIZE - 1) for k in range(GRID_SIZE))
DISCRIMINATION = 1.2      # how sharply a question separates strong from weak students
GUESS = 0.25              # four options: a blind guess is right 25% of the time
DIFFICULTY_STEP = 0.8     # question difficulty 1..5 maps to b = (difficulty - 3) * 0.8
PRIOR_MEAN = -0.3         # "not mastered until shown otherwise": a slightly cautious starting belief
PRIOR_SD = 1.5            # ...but a wide one, because we know nothing yet
HALF_LIFE_DAYS = 30.0     # evidence loses half its weight every 30 days
UNANSWERED_WEIGHT = 0.5   # a blank is weak evidence of a gap (the student may simply have run out of time)
INTERVAL = 0.80           # we report 80% credible ranges
ASSESSED_MIN_ATTEMPTS = 2           # a competency counts as "assessed" after this many answers
MIN_ATTEMPTS_FOR_READINESS = 10     # below this, the headline readiness is only "provisional"
MIN_COVERAGE = 0.5                  # ...and so is a readiness based on under half the syllabus
_Z80 = 1.2816             # the normal-distribution multiplier for an 80% range


# ---------------------------------------------------------------- the maths
def _sigmoid(x: float) -> float:
    x = max(-30.0, min(30.0, x))   # clamp: exp() of a huge number would overflow
    return 1.0 / (1.0 + math.exp(-x))


def _clamp_difficulty(difficulty: int) -> int:
    return max(1, min(5, int(difficulty)))


def p_correct(theta: float, difficulty: int) -> float:
    """Chance a student of ability `theta` answers a question of this difficulty correctly."""
    b = (_clamp_difficulty(difficulty) - 3) * DIFFICULTY_STEP
    return GUESS + (1.0 - GUESS) * _sigmoid(DISCRIMINATION * (theta - b))


def _loglik_table() -> dict[tuple[int, bool], tuple[float, ...]]:
    """log P(this answer | theta) for every grid point, computed ONCE per (difficulty, correct)."""
    table = {}
    for difficulty in range(1, 6):
        for correct in (True, False):
            values = []
            for theta in GRID:
                p = min(max(p_correct(theta, difficulty), 1e-9), 1 - 1e-9)
                values.append(math.log(p if correct else 1.0 - p))
            table[(difficulty, correct)] = tuple(values)
    return table


_LOGLIK = _loglik_table()
_LOG_PRIOR = tuple(-0.5 * ((theta - PRIOR_MEAN) / PRIOR_SD) ** 2 for theta in GRID)
_MASTERY = tuple(_sigmoid(DISCRIMINATION * theta) for theta in GRID)   # "chance of a mid-difficulty question, no guessing"
_SCORE = tuple(GUESS + (1.0 - GUESS) * m for m in _MASTERY)            # expected score on that question, guessing included


@dataclass(frozen=True)
class Evidence:
    """One graded answer."""
    competency_id: int
    question_id: int
    session_id: int
    difficulty: int
    correct: bool
    when: datetime            # when it was answered (UTC)
    submitted_at: datetime    # when its test was submitted (UTC)
    weight: float = 1.0       # 1.0 for an answer, UNANSWERED_WEIGHT for a blank


def evidence_weight(item: Evidence, now: datetime) -> float:
    age_days = max(0.0, (now - item.when).total_seconds() / 86400.0)
    return item.weight * 0.5 ** (age_days / HALF_LIFE_DAYS)


def posterior(evidence: list[Evidence], now: datetime) -> tuple[list[float], float]:
    """Bayes' rule on the grid. Returns (probabilities over the grid, total decayed weight)."""
    logp = list(_LOG_PRIOR)
    total_weight = 0.0
    for item in evidence:
        w = evidence_weight(item, now)
        if w < 1e-6:
            continue
        total_weight += w
        loglik = _LOGLIK[(_clamp_difficulty(item.difficulty), item.correct)]
        for k in range(GRID_SIZE):
            logp[k] += w * loglik[k]
    top = max(logp)
    probs = [math.exp(x - top) for x in logp]     # subtract the max first, so nothing overflows
    norm = sum(probs)
    return [p / norm for p in probs], total_weight


@dataclass(frozen=True)
class CompetencyMeta:
    id: int
    topic: str
    name: str
    weight: float             # share of the exam (all weights add up to 1)


@dataclass(frozen=True)
class CompetencyState:
    """Everything we believe about ONE competency for ONE student."""
    meta: CompetencyMeta
    mastery: float            # 0..1 chance of a mid-difficulty question WITHOUT guessing (50% = unknown)
    uncertainty: float        # standard deviation of that belief (big = not sure yet)
    expected_score: float     # 0..1 expected score on a mid-difficulty question, guessing included
    low: float                # 80% credible range of expected_score
    high: float
    theta: float              # the underlying ability estimate (the planner picks question difficulty from it)
    attempts: int             # how many answers (all time)
    correct: int
    evidence_weight: float    # how much evidence is still "fresh" after forgetting
    last_attempt: datetime | None

    @property
    def assessed(self) -> bool:
        return self.attempts >= ASSESSED_MIN_ATTEMPTS


def summarise(probs: list[float]) -> tuple[float, float, float, float, float, float]:
    """(mastery, uncertainty, expected score, low, high, theta) from a belief over the grid."""
    mastery = sum(p * m for p, m in zip(probs, _MASTERY))
    variance = sum(p * (m - mastery) ** 2 for p, m in zip(probs, _MASTERY))
    theta = sum(p * t for p, t in zip(probs, GRID))
    tail = (1.0 - INTERVAL) / 2.0
    cumulative, low, high = 0.0, _SCORE[0], _SCORE[-1]
    got_low = False
    for p, score in zip(probs, _SCORE):          # the score rises with theta, so quantiles map straight across
        cumulative += p
        if not got_low and cumulative >= tail:
            low, got_low = score, True
        if cumulative >= 1.0 - tail:
            high = score
            break
    return mastery, math.sqrt(variance), GUESS + (1 - GUESS) * mastery, low, high, theta


def compute_states(catalog: list[CompetencyMeta], evidence: list[Evidence], now: datetime) -> list[CompetencyState]:
    """The student's profile: one CompetencyState per competency (including ones never tried)."""
    grouped: dict[int, list[Evidence]] = defaultdict(list)
    for item in evidence:
        grouped[item.competency_id].append(item)

    states = []
    for meta in catalog:
        items = grouped.get(meta.id, [])
        probs, weight = posterior(items, now)
        mastery, sd, score, low, high, theta = summarise(probs)
        states.append(CompetencyState(
            meta=meta, mastery=mastery, uncertainty=sd, expected_score=score, low=low, high=high, theta=theta,
            attempts=len(items), correct=sum(1 for i in items if i.correct), evidence_weight=weight,
            last_attempt=max((i.when for i in items), default=None),
        ))
    return states


# ---------------------------------------------------------------- readiness
@dataclass(frozen=True)
class Readiness:
    """The headline number: the percentage score we expect on the real exam."""
    score: float              # 0..100
    low: float                # 80% range
    high: float
    assessed: int             # competencies with enough answers to judge
    total: int
    attempts: int

    @property
    def coverage(self) -> float:
        return self.assessed / self.total if self.total else 0.0

    @property
    def provisional(self) -> bool:
        """True while the number rests on too little: show it, but say so."""
        return self.attempts < MIN_ATTEMPTS_FOR_READINESS or self.coverage < MIN_COVERAGE

    @property
    def confidence(self) -> str:
        half_width = (self.high - self.low) / 2
        return "high" if half_width <= 6 else "medium" if half_width <= 12 else "low"


def compute_readiness(states: list[CompetencyState]) -> Readiness:
    score = sum(s.meta.weight * s.expected_score for s in states)
    # The competencies are judged independently, so their variances add (weighted by weight squared).
    variance = sum((s.meta.weight * (1 - GUESS) * s.uncertainty) ** 2 for s in states)
    margin = _Z80 * math.sqrt(variance)
    floor, ceiling = GUESS, 1.0
    return Readiness(
        score=100 * score, low=100 * max(floor, score - margin), high=100 * min(ceiling, score + margin),
        assessed=sum(1 for s in states if s.assessed), total=len(states), attempts=sum(s.attempts for s in states),
    )


# ---------------------------------------------------------------- reading the database
def load_catalog(db: Session) -> list[CompetencyMeta]:
    """Every competency with its share of the exam. A topic's weight is split equally between its
    competencies, then everything is scaled so the shares add up to 1."""
    rows = db.execute(
        select(Competency.id, Topic.name, Competency.name, Topic.exam_weight, Topic.id)
        .join(Topic, Topic.id == Competency.topic_id).order_by(Topic.name, Competency.id)
    ).all()
    per_topic: dict[int, int] = defaultdict(int)
    for *_, topic_id in rows:
        per_topic[topic_id] += 1
    raw = [(cid, topic, name, max(0.0, weight or 0.0) / per_topic[topic_id]) for cid, topic, name, weight, topic_id in rows]
    total = sum(w for *_, w in raw)
    if total <= 0:                                   # no weights set: treat every competency alike
        return [CompetencyMeta(cid, topic, name, 1 / len(raw)) for cid, topic, name, _ in raw]
    return [CompetencyMeta(cid, topic, name, w / total) for cid, topic, name, w in raw]


def load_evidence(db: Session, user: CurrentUser) -> list[Evidence]:
    """All of a student's graded answers, from FINISHED tests only."""
    rows = db.execute(
        select(Question.competency_id, Question.id, ExamSession.id, Question.difficulty, ExamItem.is_correct,
               ExamItem.selected_option, ExamItem.answered_at, ExamSession.submitted_at)
        .select_from(ExamItem)
        .join(ExamSession, ExamSession.id == ExamItem.session_id)
        .join(Question, Question.id == ExamItem.question_id)
        .where(ExamSession.user_id == user.id, ExamSession.status == SessionStatus.SUBMITTED)
        .order_by(ExamSession.submitted_at, ExamItem.position)
    ).all()
    evidence = []
    for comp_id, question_id, session_id, difficulty, is_correct, selected, answered_at, submitted_at in rows:
        submitted = as_utc(submitted_at)
        evidence.append(Evidence(
            competency_id=comp_id, question_id=question_id, session_id=session_id, difficulty=difficulty,
            correct=bool(is_correct), when=as_utc(answered_at) if answered_at else submitted, submitted_at=submitted,
            weight=1.0 if selected else UNANSWERED_WEIGHT,
        ))
    return evidence
