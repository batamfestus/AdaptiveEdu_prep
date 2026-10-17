"""
core/permissions.py
-------------------
Authorisation checks that live in `core/`, NOT only in the UI.

Defence in depth: ui/session.py already blocks students from admin PAGES.
But if a future page (or an API) forgets that guard, the core functions that
change data check the role themselves. Two locks are safer than one.
"""
from core.auth import CurrentUser


class PermissionDenied(Exception):
    """Raised when someone tries an action their role doesn't allow."""


def ensure_admin(actor: CurrentUser | None) -> None:
    if actor is None or not actor.is_admin:
        raise PermissionDenied("This action requires an admin account.")
