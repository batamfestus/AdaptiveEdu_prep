"""
core/planner.py
---------------
THE PLANNER: decides what a student should practise next and picks the questions.
It reads the mastery model (core/mastery.py) and chooses, using the factors named in
the job description: competency gaps, exam weighting, uncertainty, recency and previous performance.

HOW IT DECIDES (explain this in an interview):

  priority = exam_weight * ( gap + UNCERTAINTY_BONUS * uncertainty )
               gap = 1 - mastery        "how much is there to gain?"
               uncertainty              "how little do we know?"  (exploration)
               exam_weight              "how much is it worth?"

  * A weak, heavily-weighted competency scores high: big gap, big prize.
  * A competency we know little about scores high: we need a first look. (Exploration, as in a
    "multi-armed bandit": don't ignore options you haven't tried enough.)
  * RECENCY needs no special rule: old evidence fades in the mastery model, so a competency you
    haven't touched in weeks drifts back towards "unknown", its uncertainty rises, and so does its priority.

  Questions are then chosen to be "just hard enough": we aim for about a 65% chance of a correct
  answer, the zone where learning is fastest (too easy is boring, too hard is discouraging),
  and unseen questions come before repeats.
"""
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from core.auth import CurrentUser
from core.mastery import (
    CompetencyState, Evidence, compute_states, load_catalog, load_evidence, p_correct,
)
from core.orm import Competency, Question, QuestionStatus, utcnow

UNCERTAINTY_BONUS = 0.5
TARGET_SUCCESS = 0.65          # the chance of a correct answer we aim for
PLAN_SIZE = 5                  # competencies in a plan
MAX_PER_COMPETENCY = 6
DIAGNOSTIC_PER_COMPETENCY = 2
MAX_TEST_QUESTIONS = 40
STALE_DAYS = 14                # "not practised for N days" starts being mentioned after this
WEAK_MASTERY = 0.45


@dataclass(frozen=True)
class PlanItem:
    state: CompetencyState
    priority: float
    reason: str


# ============================================================ what to practise
def priority(state: CompetencyState) -> float:
    return state.meta.weight * ((1.0 - state.mastery) + UNCERTAINTY_BONUS * state.uncertainty)


def reason_for(state: CompetencyState, now: datetime) -> str:
    """A short, honest explanation the student can read ("why this one?")."""
    if state.attempts == 0:
        return "You haven't tried this yet, so we need a first look."
    if state.attempts < 3 or state.uncertainty > 0.18:
        return f"We don't know enough yet (about {state.mastery:.0%} so far). A few more answers will settle it."
    if state.mastery < WEAK_MASTERY:
        return f"Weak spot: your mastery is about {state.mastery:.0%}."
    idle = (now - state.last_attempt).days if state.last_attempt else 0
    if idle >= STALE_DAYS:
        return f"You haven't practised this for {idle} days, so it needs a refresh."
    if state.mastery < 0.75:
        return f"Getting there (about {state.mastery:.0%}): more practice will lock it in."
    return "Strong. A little practice keeps it sharp."


def build_plan(states: list[CompetencyState], now: datetime, size: int = PLAN_SIZE) -> list[PlanItem]:
    """The competencies to work on, most valuable first."""
    ranked = sorted(states, key=lambda s: (-priority(s), s.meta.id))   # the id breaks ties the same way every time
    return [PlanItem(s, priority(s), reason_for(s, now)) for s in ranked[:size]]


# ============================================================ choosing questions
def allocate(plan: list[PlanItem], n: int) -> dict[int, int]:
    """Share n questions between the plan's competencies in proportion to priority
    (the "largest remainder" method), at least 1 each, at most MAX_PER_COMPETENCY."""
    items = plan[:n]                                  # never more competencies than questions
    if not items:
        return {}
    total = sum(i.priority for i in items) or 1.0
    exact = {i.state.meta.id: n * i.priority / total for i in items}
    shares = {cid: max(1, min(MAX_PER_COMPETENCY, int(math.floor(x)))) for cid, x in exact.items()}
    # hand out (or take back) questions until the total is exactly n
    order = sorted(exact, key=lambda cid: exact[cid] - math.floor(exact[cid]), reverse=True)
    while sum(shares.values()) < n:
        grew = False
        for cid in order:
            if sum(shares.values()) >= n:
                break
            if shares[cid] < MAX_PER_COMPETENCY:
                shares[cid] += 1
                grew = True
        if not grew:
            break
    while sum(shares.values()) > n:
        for cid in reversed(order):
            if sum(shares.values()) <= n:
                break
            if shares[cid] > 1:
                shares[cid] -= 1
    return shares


def _published_by_competency(db: Session) -> dict[int, list[tuple[int, int]]]:
    """{competency id: [(question id, difficulty), ...]} for published questions only."""
    pool: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for qid, comp_id, difficulty in db.execute(
        select(Question.id, Question.competency_id, Question.difficulty)
        .where(Question.status == QuestionStatus.PUBLISHED).order_by(Question.id)
    ):
        pool[comp_id].append((qid, difficulty))
    return pool


def _times_seen(evidence: list[Evidence]) -> dict[int, int]:
    seen: dict[int, int] = defaultdict(int)
    for item in evidence:
        seen[item.question_id] += 1
    return seen


def pick_adaptive_questions(db: Session, user: CurrentUser, n: int, rng: random.Random,
                            now: datetime | None = None) -> list[int]:
    """Choose n questions aimed at this student's biggest opportunities."""
    now = now or utcnow()
    evidence = load_evidence(db, user)
    states = compute_states(load_catalog(db), evidence, now)
    by_id = {s.meta.id: s for s in states}
    pool = _published_by_competency(db)
    seen = _times_seen(evidence)

    # Only plan around competencies we actually have questions for.
    candidates = [s for s in states if pool.get(s.meta.id)]
    ranked = build_plan(candidates, now, size=len(candidates))
    shares = allocate(ranked[:PLAN_SIZE], n)

    chosen: list[int] = []
    taken: set[int] = set()

    def take(competency_id: int, how_many: int) -> None:
        state = by_id[competency_id]
        # unseen first; then the question whose predicted success is closest to the target; random last
        options = sorted(
            (q for q in pool[competency_id] if q[0] not in taken),
            key=lambda q: (seen.get(q[0], 0), abs(p_correct(state.theta, q[1]) - TARGET_SUCCESS), rng.random()),
        )
        for qid, _ in options[:how_many]:
            chosen.append(qid)
            taken.add(qid)

    for competency_id, how_many in shares.items():
        take(competency_id, how_many)
    # A competency may have had too few questions: fill the shortfall from the next best, in priority order.
    for item in ranked:
        if len(chosen) >= n:
            break
        take(item.state.meta.id, n - len(chosen))
    rng.shuffle(chosen)           # so questions from one competency aren't served in a block
    return chosen[:n]


def pick_diagnostic_questions(db: Session, user: CurrentUser, rng: random.Random,
                              now: datetime | None = None) -> list[int]:
    """A first look at EVERY competency: up to two questions each, near medium difficulty."""
    now = now or utcnow()
    evidence = load_evidence(db, user)
    pool = _published_by_competency(db)
    seen = _times_seen(evidence)
    catalog = [m for m in load_catalog(db) if pool.get(m.id)]

    per_competency: list[tuple[float, list[int]]] = []
    for meta in catalog:
        options = sorted(pool[meta.id], key=lambda q: (seen.get(q[0], 0), abs(q[1] - 3), rng.random()))
        per_competency.append((meta.weight, [qid for qid, _ in options[:DIAGNOSTIC_PER_COMPETENCY]]))

    chosen = [qid for _, ids in per_competency for qid in ids]
    if len(chosen) > MAX_TEST_QUESTIONS:               # a very big syllabus: keep the most valuable competencies
        per_competency.sort(key=lambda pair: -pair[0])
        chosen = [qid for _, ids in per_competency for qid in ids][:MAX_TEST_QUESTIONS]
    rng.shuffle(chosen)
    return chosen
