"""
A guard test. The test suite deliberately uses cheap password hashing to run fast
(see tests/conftest.py). This test makes sure that cheap setting can never be the
one the real app uses.
"""
import bcrypt

from core import auth
from tests.conftest import PRODUCTION_BCRYPT_ROUNDS


def test_production_password_hashing_is_still_strong():
    assert PRODUCTION_BCRYPT_ROUNDS >= 12


def test_tests_use_the_cheap_setting_and_hashes_still_verify():
    assert auth.BCRYPT_ROUNDS == 4                      # the fast setting is active during tests
    stored = auth.hash_password("Physics2026")
    assert stored.startswith("$2b$04$")                 # the cost is recorded inside the hash itself
    assert auth.verify_password("Physics2026", stored)
    # A hash made at full strength still verifies too, so nothing about real accounts changes.
    strong = bcrypt.hashpw(b"Physics2026", bcrypt.gensalt(rounds=12)).decode()
    assert auth.verify_password("Physics2026", strong)
