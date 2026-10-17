"""
core/config.py
--------------
Central place for settings. Everything else in `core/` asks THIS file
for configuration instead of reading environment variables itself.

Design rule for the whole project:
    `core/` never imports Streamlit. That keeps the business logic
    testable and reusable (e.g. by a future FastAPI backend).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Project root = the folder that contains `core/`
BASE_DIR = Path(__file__).resolve().parent.parent

# Reads a local `.env` file (if present) into environment variables.
load_dotenv(BASE_DIR / ".env")

# The Postgres driver we install in requirements.txt (psycopg2-binary).
POSTGRES_DRIVER = "postgresql+psycopg2"


def normalise_database_url(url: str) -> str:
    """Make sure Postgres URLs name the driver we actually installed.

    THE BUG THIS FIXES: Supabase gives you a URL starting with "postgresql://".
    That bare form means "use the default driver", and the DEFAULT DIFFERS
    BETWEEN SQLAlchemy VERSIONS (2.0 -> psycopg2, newer 2.1 -> psycopg v3).
    On a newer SQLAlchemy you get "No module named 'psycopg'".
    Naming the driver explicitly makes the app behave the same everywhere.

    Rules:
      * "postgres://..." and "postgresql://..." -> "postgresql+psycopg2://..."
      * URLs that already name a driver ("postgresql+psycopg://") are left
        alone, so you can still choose a different driver on purpose.
      * Anything else (e.g. sqlite:///...) is returned unchanged.
    """
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return f"{POSTGRES_DRIVER}://" + url[len(prefix):]
    return url


def get_database_url() -> str:
    """Return the DB connection string.

    We read the environment at CALL time (not import time) so that app.py can
    copy Streamlit secrets into the environment first, on the deployed app.
    Falls back to a local SQLite file so the project runs with zero setup.
    """
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        return f"sqlite:///{BASE_DIR / 'adaptprep.db'}"
    return normalise_database_url(url)


def get_anthropic_key() -> str:
    """Return the Anthropic API key (empty string if not configured)."""
    return os.getenv("ANTHROPIC_API_KEY", "").strip()


# ============================================================ AI tutor settings (Lesson 6)
# Every limit below can be changed in .env (or Streamlit secrets) WITHOUT touching code.
DEFAULT_MODEL = "claude-haiku-5-5"   # a small, fast, low-cost model is plenty for tutoring


def get_anthropic_model() -> str:
    return os.getenv("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL


def get_int_setting(name: str, default: int) -> int:
    """Read a whole-number setting from the environment; a bad value falls back to the default."""
    try:
        return int(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def tutor_limits() -> dict[str, int]:
    """Credit protection. Counted over a rolling 24 hours.

    explanations: NEW AI explanations one student may trigger (cached ones are free and unlimited)
    messages:     follow-up chat messages one student may send
    global_tokens: total AI tokens (input + output) across ALL students: the emergency brake
    """
    return {
        "explanations": get_int_setting("TUTOR_DAILY_EXPLANATIONS", 15),
        "messages": get_int_setting("TUTOR_DAILY_MESSAGES", 25),
        "global_tokens": get_int_setting("TUTOR_GLOBAL_DAILY_TOKENS", 400_000),
        "web_per_student": get_int_setting("TUTOR_DAILY_WEB_SEARCHES", 5),
        "web_global": get_int_setting("TUTOR_GLOBAL_DAILY_WEB_SEARCHES", 100),
        "web_per_message": get_int_setting("TUTOR_WEB_MAX_USES", 2),
    }


# ============================================================ web search agent settings (Lesson 7)
# Web search is OFF until you switch it on. Each search costs real money (about 1 cent plus the
# tokens for the pages it reads), and the tutor may only look at the sites listed below.
DEFAULT_WEB_DOMAINS = (
    "hyperphysics.phy-astr.gsu.edu",   # HyperPhysics
    "physicsclassroom.com",
    "khanacademy.org",
    "bbc.co.uk/bitesize",
    "openstax.org",
    "phys.libretexts.org",
    "physics.info",                    # The Physics Hypertextbook
    "britannica.com",
    "wikipedia.org",
    "jamb.gov.ng",                     # the official exam board
)


def get_float_setting(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, "").strip() or default)
    except ValueError:
        return default


def web_search_enabled() -> bool:
    return os.getenv("TUTOR_WEB_SEARCH", "").strip().lower() in ("1", "true", "yes", "on")


def clean_domains(domains) -> tuple[str, ...]:
    """Keep only well-formed entries. Anthropic's rules (and basic safety):
    bare domains with an optional path, no 'https://', no wildcard in the domain part, and
    ASCII only (look-alike letters such as a Cyrillic 'a' could otherwise sneak a fake site in)."""
    good = []
    for entry in domains:
        entry = entry.strip().lower()
        if not entry or "://" in entry or entry.startswith(".") or not entry.isascii():
            continue
        if "*" in entry.split("/")[0] or " " in entry:
            continue
        if entry not in good:
            good.append(entry)
    return tuple(good)


def web_domains() -> tuple[str, ...]:
    """The sites the tutor may search. Override with TUTOR_WEB_DOMAINS="a.com,b.org/path"."""
    raw = os.getenv("TUTOR_WEB_DOMAINS", "").strip()
    return clean_domains(raw.split(",")) if raw else DEFAULT_WEB_DOMAINS


def notes_weak_score() -> float:
    """If the best study note scores below this (0 to 1), the notes are treated as NOT covering
    the question. Lower it if the tutor searches the web too often; raise it if too rarely."""
    return get_float_setting("NOTES_WEAK_SCORE", 0.30)
