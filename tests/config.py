"""
Tests for core/config.py. Run with:  pytest -q

These tests exist because of a REAL bug: a bare "postgresql://" URL picked
a different driver on different SQLAlchemy versions. A bug that bit you once
deserves a test so it can never come back silently.
"""
import pytest

from core import config

SAME = "unchanged"


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("postgresql://u:p@host:5432/db", "postgresql+psycopg2://u:p@host:5432/db"),
        ("postgres://u:p@host:5432/db", "postgresql+psycopg2://u:p@host:5432/db"),
        ("postgresql+psycopg2://u:p@host/db", SAME),   # already explicit
        ("postgresql+psycopg://u:p@host/db", SAME),    # a deliberate choice is respected
        ("sqlite:///local.db", SAME),                  # not Postgres: untouched
    ],
)
def test_normalise_database_url(raw, expected):
    assert config.normalise_database_url(raw) == (raw if expected == SAME else expected)


def test_empty_database_url_falls_back_to_sqlite(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    assert config.get_database_url().startswith("sqlite:///")


def test_env_url_is_normalised(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host:5432/db")
    assert config.get_database_url() == "postgresql+psycopg2://u:p@host:5432/db"