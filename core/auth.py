"""
core/auth.py
------------
All authentication LOGIC lives here: hashing, registering, logging in.
No Streamlit imports, so every rule can be unit-tested.

Security ideas used in this file (know these for interviews):
  1. Passwords are HASHED with bcrypt, never stored. A hash is one-way:
     you can check a guess against it, but you cannot recover the password.
  2. bcrypt adds a random "salt" per password, so two users with the same
     password get different hashes, and pre-computed tables don't work.
  3. bcrypt is deliberately SLOW (~0.2s). That's harmless for one login
     but makes mass password-guessing very expensive for an attacker.
  4. Login errors are deliberately vague ("invalid email or password") so
     an attacker can't learn which emails have accounts.
"""
import re
from dataclasses import dataclass

import bcrypt
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from core.orm import Role, User

# How much work one password hash costs: each +1 DOUBLES the time. 12 is about a quarter of a
# second per login, which is nothing for a student but ruinous for someone guessing millions of
# passwords. NEVER lower this for the real app. (The test suite lowers it, see tests/conftest.py,
# because creating hundreds of test users at full strength made the tests take minutes.)
BCRYPT_ROUNDS = 12
MIN_PASSWORD_LEN = 8
MAX_PASSWORD_BYTES = 72  # bcrypt silently ignores anything past 72 bytes, so we refuse it
# Intentionally simple email check (one @, a dot in the domain). The real proof
# an email is valid is sending a verification mail, which is a later upgrade.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# ---------- Errors ----------
# One base class lets the UI write `except AuthError` and show e's message.
class AuthError(Exception):
    """Base class for problems the user can fix (safe to show on screen)."""


class InvalidInput(AuthError):
    pass


class WeakPassword(AuthError):
    pass


class EmailTaken(AuthError):
    pass


# ---------- What we keep in the browser session ----------
@dataclass(frozen=True)
class CurrentUser:
    """A small, plain snapshot of the logged-in user.

    Why not store the SQLAlchemy `User` object in the session instead?
    ORM objects are tied to a database session that closes after each
    request, and they carry the password hash. This snapshot has neither problem.
    """
    id: int
    email: str
    full_name: str
    role: Role

    @property
    def is_admin(self) -> bool:
        return self.role == Role.ADMIN


def _to_current_user(user: User) -> CurrentUser:
    return CurrentUser(id=user.id, email=user.email, full_name=user.full_name, role=user.role)


# ---------- Password helpers ----------
def hash_password(password: str) -> str:
    """Turn a plain password into a salted bcrypt hash (safe to store)."""
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=BCRYPT_ROUNDS)).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    """Check a login attempt against a stored hash."""
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:  # malformed hash in the database
        return False


# A throwaway hash used to keep login timing the same whether or not the email
# exists (see authenticate). Computed once when this module loads.
_DUMMY_HASH = hash_password("timing-equaliser-not-a-real-password")


def check_password_strength(password: str) -> None:
    if len(password) < MIN_PASSWORD_LEN:
        raise WeakPassword(f"Password must be at least {MIN_PASSWORD_LEN} characters.")
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        raise WeakPassword("Password is too long (maximum 72 bytes).")
    if not (re.search(r"[A-Za-z]", password) and re.search(r"\d", password)):
        raise WeakPassword("Password must contain at least one letter and one number.")


def normalise_email(email: str) -> str:
    """Emails are case-insensitive in practice, so store them lower-cased."""
    return email.strip().lower()


# ---------- Public API ----------
def register_user(
    db: Session,
    email: str,
    full_name: str,
    password: str,
    role: Role = Role.STUDENT,
) -> CurrentUser:
    """Create an account.

    SECURITY: the signup form never passes `role`, so everyone who registers
    through the website is a STUDENT. Only trusted scripts (create_admin.py)
    pass Role.ADMIN. Never let a web form choose its own role.
    """
    email = normalise_email(email)
    full_name = full_name.strip()

    if not EMAIL_RE.match(email):
        raise InvalidInput("Please enter a valid email address.")
    if len(full_name) < 2:
        raise InvalidInput("Please enter your full name.")
    check_password_strength(password)

    # Friendly pre-check. (Telling people an email is taken does reveal it has
    # an account; most signup forms accept that trade-off for usability.)
    if db.scalar(select(User.id).where(User.email == email)) is not None:
        raise EmailTaken("An account with this email already exists.")

    user = User(email=email, full_name=full_name, password_hash=hash_password(password), role=role)
    db.add(user)
    try:
        db.commit()
    except IntegrityError:
        # Two people registering the same email at the same instant can both
        # pass the pre-check; the database's UNIQUE constraint is the real guard.
        db.rollback()
        raise EmailTaken("An account with this email already exists.")
    return _to_current_user(user)


def authenticate(db: Session, email: str, password: str) -> CurrentUser | None:
    """Return the user if email+password are right, otherwise None."""
    user = db.scalar(select(User).where(User.email == normalise_email(email)))

    # Always run ONE bcrypt check, even when the email is unknown. Otherwise an
    # unknown email returns instantly and a known one takes ~0.2s, and that
    # timing difference lets an attacker discover which emails are registered.
    hash_to_check = user.password_hash if user else _DUMMY_HASH
    password_ok = verify_password(password, hash_to_check)

    if user is None or not password_ok:
        return None
    return _to_current_user(user)
