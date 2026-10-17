"""Tests for the web-search settings in core/config.py."""
import pytest

from core import config


@pytest.mark.parametrize("value, expected", [("1", True), ("true", True), ("YES", True), ("on", True),
                                              ("", False), ("0", False), ("no", False), ("maybe", False)])
def test_web_search_is_off_unless_switched_on(monkeypatch, value, expected):
    monkeypatch.setenv("TUTOR_WEB_SEARCH", value)
    assert config.web_search_enabled() is expected


def test_web_search_is_off_by_default(monkeypatch):
    monkeypatch.delenv("TUTOR_WEB_SEARCH", raising=False)
    assert config.web_search_enabled() is False


def test_unsafe_domain_entries_are_dropped():
    cleaned = config.clean_domains([
        "Khanacademy.org", " bbc.co.uk/bitesize ", "khanacademy.org",        # lower-cased, trimmed, de-duplicated
        "https://evil.example", "*.example.com", "ex*ample.com", "",         # scheme and wildcards are not allowed
        "аmazon.com",                                                         # a Cyrillic 'а': a look-alike of amazon.com
        "has space.com", ".leading-dot.com",
        "example.com/*/articles",                                             # a wildcard in the PATH is allowed
    ])
    assert cleaned == ("khanacademy.org", "bbc.co.uk/bitesize", "example.com/*/articles")


def test_the_default_site_list_is_valid_and_unchanged_by_cleaning():
    assert config.clean_domains(config.DEFAULT_WEB_DOMAINS) == config.DEFAULT_WEB_DOMAINS
    assert "jamb.gov.ng" in config.DEFAULT_WEB_DOMAINS


def test_domains_can_be_overridden_in_the_environment(monkeypatch):
    monkeypatch.delenv("TUTOR_WEB_DOMAINS", raising=False)
    assert config.web_domains() == config.DEFAULT_WEB_DOMAINS
    monkeypatch.setenv("TUTOR_WEB_DOMAINS", "openstax.org, https://bad.example ,physics.info")
    assert config.web_domains() == ("openstax.org", "physics.info")


def test_budget_defaults_and_overrides(monkeypatch):
    for name in ("TUTOR_DAILY_WEB_SEARCHES", "TUTOR_GLOBAL_DAILY_WEB_SEARCHES", "TUTOR_WEB_MAX_USES"):
        monkeypatch.delenv(name, raising=False)
    limits = config.tutor_limits()
    assert (limits["web_per_student"], limits["web_global"], limits["web_per_message"]) == (5, 100, 2)
    monkeypatch.setenv("TUTOR_DAILY_WEB_SEARCHES", "9")
    monkeypatch.setenv("TUTOR_WEB_MAX_USES", "not a number")
    limits = config.tutor_limits()
    assert limits["web_per_student"] == 9 and limits["web_per_message"] == 2     # a bad value falls back


def test_notes_threshold(monkeypatch):
    monkeypatch.delenv("NOTES_WEAK_SCORE", raising=False)
    assert config.notes_weak_score() == 0.30
    monkeypatch.setenv("NOTES_WEAK_SCORE", "0.45")
    assert config.notes_weak_score() == 0.45
    monkeypatch.setenv("NOTES_WEAK_SCORE", "high")
    assert config.notes_weak_score() == 0.30
