"""
scripts/create_admin.py
-----------------------
Create the first admin (or promote an existing account). Run in your terminal:

    python -m scripts.create_admin

Why a script and not a website button? Because "become admin" must never be
reachable from the web. Only someone with access to the server/database
(you) can make admins.
"""
from getpass import getpass

from sqlalchemy import select

from core.auth import AuthError, normalise_email, register_user
from core.db import get_engine, get_session
from core.orm import Base, Role, User


def main() -> None:
    Base.metadata.create_all(get_engine())  # make sure tables exist
    email = normalise_email(input("Admin email: "))

    with get_session() as db:
        existing = db.scalar(select(User).where(User.email == email))

        if existing is not None:
            if existing.role == Role.ADMIN:
                print("That account is already an admin.")
                return
            if input(f"{email} exists as a student. Promote to admin? [y/N] ").lower() == "y":
                existing.role = Role.ADMIN
                db.commit()
                print("Promoted to admin.")
            return

        name = input("Full name: ")
        password = getpass("Password (hidden): ")  # getpass doesn't echo what you type
        try:
            register_user(db, email, name, password, role=Role.ADMIN)
            print("Admin account created.")
        except AuthError as e:
            print(f"Could not create admin: {e}")


if __name__ == "__main__":
    main()
