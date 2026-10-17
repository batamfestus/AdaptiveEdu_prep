"""
scripts/check_connection.py
---------------------------
Diagnoses database connection problems WITHOUT ever printing your password.

    python -m scripts.check_connection

It shows the parts of DATABASE_URL that are safe to see (driver, host, port,
username, database name), checks for the most common mistakes, then tries a
real connection and explains the result in plain English.
"""
import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError

from core.config import get_database_url  # importing this also loads your .env


def explain(error_text: str) -> None:
    """Map the usual Postgres/Supabase error messages to a likely fix."""
    e = error_text.lower()
    if "password authentication failed" in e:
        print("\nLikely cause: the password in DATABASE_URL is wrong.")
        print(" - In Supabase: Project Settings > Database > Reset database password.")
        print(" - Use letters and numbers only (no @ # / : ? %) to avoid URL-encoding problems.")
        print(" - Paste the new password into .env, save, and run this script again.")
    elif "tenant or user not found" in e:
        print("\nLikely cause: the username. On the pooler it must be postgres.<project-ref>,")
        print("not just 'postgres'. Copy the whole string from Connect > Session pooler.")
    elif "could not translate host" in e or "unreachable" in e:
        print("\nLikely cause: wrong host, or you used the direct connection string.")
        print("Use the SESSION POOLER string from Supabase's Connect panel.")
    elif "timeout" in e or "timed out" in e:
        print("\nLikely cause: network or firewall blocking the connection. Try another network.")
    else:
        print("\nUnrecognised error. Copy the first line above (without any password) and ask for help.")


def main() -> None:
    url = make_url(get_database_url())

    if url.drivername.startswith("sqlite"):
        print("DATABASE_URL is empty, so the app is using local SQLite.")
        print("Check that your .env file is named exactly '.env' (not '.env.txt') and has DATABASE_URL=...")
        return

    user, host, password = url.username or "", url.host or "", url.password or ""
    print(f"Driver   : {url.drivername}")
    print(f"Host     : {host}")
    print(f"Port     : {url.port}")
    print(f"Username : {user}")
    print(f"Database : {url.database}")
    print(f"Password : {len(password)} characters (value hidden)")

    # ---- common mistakes, checked before we even try to connect ----
    problems = []
    if "pooler.supabase.com" in host and "." not in user:
        problems.append("Username is '%s'. On the Supabase pooler it should look like "
                        "postgres.<project-ref>." % user)
    if password.startswith("[") or password.endswith("]"):
        problems.append("The password still has [square brackets]. Remove the brackets: "
                        "[YOUR-PASSWORD] was a placeholder.")
    if not password:
        problems.append("The password is empty.")
    raw = os.getenv("DATABASE_URL", "")
    if raw.count("@") > 1:
        problems.append("The URL contains more than one '@'. If your password has an '@', "
                        "URL-encode it as %40, or reset to a simpler password.")
    if raw.startswith(('"', "'")):
        problems.append("Remove the quotation marks around the URL in .env.")
    for p in problems:
        print(f"\n[!] {p}")

    # ---- try a real connection (10 second limit instead of waiting forever) ----
    print("\nTrying to connect...")
    engine = create_engine(url, connect_args={"connect_timeout": 10})
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("SUCCESS: connected to the database.")
    except OperationalError as err:
        first_line = str(err.orig).strip().splitlines()[0]
        print(f"FAILED: {first_line}")
        explain(first_line)


if __name__ == "__main__":
    main()