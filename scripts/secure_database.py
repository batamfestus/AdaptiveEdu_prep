"""
scripts/secure_database.py
--------------------------
Turns on Row Level Security (RLS) for every table that lacks it:

    python -m scripts.secure_database

The app now does this automatically at startup (see ui/session.py), so you
rarely need to run it by hand, but it is a handy way to check and see the result.

WHY RLS: Supabase exposes `public` tables through a REST API protected only by
RLS. With RLS ON and NO policies, that public API sees nothing, while our own
connection (the `postgres` role, which bypasses RLS) keeps working.
Safe to run repeatedly. Does nothing on SQLite.
"""
from core.db import get_engine
from core.orm import Base
from core.security import ensure_rls


def main() -> None:
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        print("Not a Postgres database, so there is nothing to secure (RLS is a Postgres feature).")
        return

    Base.metadata.create_all(engine)  # make sure every table exists first
    changed = ensure_rls(engine)
    for name in changed:
        print(f"Row Level Security enabled on: {name}")
    if not changed:
        print("All tables already have Row Level Security enabled.")


if __name__ == "__main__":
    main()
