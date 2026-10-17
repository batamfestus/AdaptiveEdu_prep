"""ensure_rls must be a harmless no-op on SQLite (it only acts on Postgres)."""
from core.security import ensure_rls


def test_ensure_rls_is_a_noop_on_sqlite(db):
    assert ensure_rls(db.get_bind()) == []
