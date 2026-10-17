"""
core/security.py
----------------
Database hardening helpers.

ensure_rls() turns on Row Level Security for any of OUR tables that don't have
it yet (Postgres/Supabase only). Why: Supabase exposes tables in the public
schema through a REST API guarded only by RLS. A table created WITHOUT RLS is
readable by anyone who has your project's public "anon" key.

Because new tables appear in every lesson (Lesson 3 added question_reviews),
the app calls this at startup, so a new table can never be left exposed.
"""
from sqlalchemy import text
from sqlalchemy.engine import Engine

from core.orm import Base


def ensure_rls(engine: Engine) -> list[str]:
    """Enable RLS where missing. Returns the names of tables it changed.

    It first READS which tables already have RLS and only ALTERs the others,
    because ALTER TABLE briefly locks the table and we don't want to do that
    on every app start.
    """
    if engine.dialect.name != "postgresql":
        return []  # RLS is a Postgres feature; SQLite has nothing to secure

    quote = engine.dialect.identifier_preparer.quote  # safely quotes table names
    changed: list[str] = []
    with engine.begin() as conn:  # one transaction: all or nothing
        rows = conn.execute(
            text("SELECT tablename, rowsecurity FROM pg_tables WHERE schemaname = current_schema()")
        )
        has_rls = {name: enabled for name, enabled in rows}
        for table in Base.metadata.sorted_tables:
            if has_rls.get(table.name) is False:  # exists, but RLS is off
                conn.execute(text(f"ALTER TABLE {quote(table.name)} ENABLE ROW LEVEL SECURITY"))
                changed.append(table.name)
    return changed
