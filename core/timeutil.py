"""
core/timeutil.py
----------------
One tiny helper shared by several modules (it lives here so they don't import each other in a circle).
"""
from datetime import datetime, timezone


def as_utc(moment: datetime) -> datetime:
    """SQLite forgets time zones (it returns 'naive' datetimes); Postgres keeps them.
    We always store UTC, so a naive value IS UTC. Normalising makes both behave the same."""
    return moment if moment.tzinfo else moment.replace(tzinfo=timezone.utc)
