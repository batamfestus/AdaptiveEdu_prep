"""
Shared test fixtures (pytest finds this file automatically).

Every test gets a FRESH in-memory database, so tests never touch your real
data and never affect each other.
"""
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from core import auth
from core.auth import register_user
from core.orm import Base, Competency, Role, Topic

# Remember the REAL setting before we lower it for the tests (tests/test_hashing_cost.py checks it).
PRODUCTION_BCRYPT_ROUNDS = auth.BCRYPT_ROUNDS

PASSWORD = "Physics2026"


@pytest.fixture(autouse=True, scope="session")
def fast_password_hashing():
    """Hash passwords with the cheapest bcrypt setting during tests (about 250x faster).
    This changes ONLY the test run: the app itself keeps the production value."""
    auth.BCRYPT_ROUNDS = 4
    yield
    auth.BCRYPT_ROUNDS = PRODUCTION_BCRYPT_ROUNDS


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    engine.dispose()


@pytest.fixture()
def seeded(db):
    """A tiny syllabus: 2 topics, 2 competencies each."""
    syllabus = {
        "Current Electricity": ["Apply Ohm's law", "Analyse series and parallel circuits"],
        "Motion": ["Apply equations of uniform acceleration", "Distinguish scalars and vectors"],
    }
    for topic_name, comps in syllabus.items():
        topic = Topic(name=topic_name)
        topic.competencies = [Competency(name=c) for c in comps]
        db.add(topic)
    db.commit()
    return syllabus


@pytest.fixture()
def admin(db):
    return register_user(db, "boss@example.com", "The Boss", PASSWORD, role=Role.ADMIN)


@pytest.fixture()
def student(db):
    return register_user(db, "ada@example.com", "Ada Obi", PASSWORD)
