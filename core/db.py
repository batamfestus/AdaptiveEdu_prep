"""
core/db.py
----------
Creates the database engine and hands out sessions.

Concepts:
  * Engine  = the connection factory/pool (create it ONCE per process).
  * Session = a short-lived "unit of work" (open, do work, close).
"""
from functools import lru_cache

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from core.config import get_database_url


@lru_cache(maxsize=1)  # lru_cache => the engine is built once and reused
def get_engine() -> Engine:
    url = get_database_url()
    connect_args = {}
    if url.startswith("sqlite"):
        # Streamlit serves each user in its own thread; SQLite must be told
        # that sharing a connection across threads is okay.
        connect_args["check_same_thread"] = False
    # pool_pre_ping tests a connection before using it, which avoids
    # "connection closed" errors on hosted Postgres that sleeps when idle.
    return create_engine(url, pool_pre_ping=True, connect_args=connect_args)


def get_session() -> Session:
    """Use as a context manager so the session always closes:

        with get_session() as db:
            db.query(...)
    """
    return Session(get_engine())
