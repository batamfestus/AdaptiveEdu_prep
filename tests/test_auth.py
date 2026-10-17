"""
Tests for core/auth.py. Run with:  pytest -q

Each test gets a FRESH in-memory database (see the `db` fixture), so tests
never touch your real adaptprep.db and never affect each other.
"""
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from core.auth import (
    EmailTaken, InvalidInput, WeakPassword,
    authenticate, hash_password, register_user, verify_password,
)
from core.orm import Base, Role, User

GOOD_PW = "Physics2026"


@pytest.fixture()
def db():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        yield session  # the test runs here
    engine.dispose()


# ---- hashing ----
def test_hash_is_not_the_password_and_is_salted():
    h1, h2 = hash_password(GOOD_PW), hash_password(GOOD_PW)
    assert h1 != GOOD_PW
    assert h1 != h2  # different salt each time


def test_verify_password():
    h = hash_password(GOOD_PW)
    assert verify_password(GOOD_PW, h)
    assert not verify_password("wrong-password1", h)


# ---- registration ----
def test_register_creates_student_by_default(db):
    user = register_user(db, "Ada@Example.com", "Ada Obi", GOOD_PW)
    assert user.role == Role.STUDENT
    assert user.email == "ada@example.com"  # normalised
    stored = db.scalar(select(User))
    assert stored.password_hash != GOOD_PW  # only the hash is stored


def test_duplicate_email_rejected_case_insensitively(db):
    register_user(db, "ada@example.com", "Ada Obi", GOOD_PW)
    with pytest.raises(EmailTaken):
        register_user(db, "ADA@example.com", "Someone Else", GOOD_PW)


@pytest.mark.parametrize("bad_pw", ["short1", "onlyletters", "12345678", "x" * 80 + "1"])
def test_weak_passwords_rejected(db, bad_pw):
    with pytest.raises(WeakPassword):
        register_user(db, "ada@example.com", "Ada Obi", bad_pw)


def test_invalid_email_rejected(db):
    with pytest.raises(InvalidInput):
        register_user(db, "not-an-email", "Ada Obi", GOOD_PW)


# ---- login ----
def test_authenticate_success(db):
    register_user(db, "ada@example.com", "Ada Obi", GOOD_PW)
    user = authenticate(db, " ADA@example.com ", GOOD_PW)  # messy input still works
    assert user is not None and user.full_name == "Ada Obi"


def test_authenticate_wrong_password(db):
    register_user(db, "ada@example.com", "Ada Obi", GOOD_PW)
    assert authenticate(db, "ada@example.com", "WrongPass9") is None


def test_authenticate_unknown_email(db):
    assert authenticate(db, "ghost@example.com", GOOD_PW) is None


def test_admin_can_only_be_created_explicitly(db):
    admin = register_user(db, "boss@example.com", "The Boss", GOOD_PW, role=Role.ADMIN)
    assert admin.is_admin
