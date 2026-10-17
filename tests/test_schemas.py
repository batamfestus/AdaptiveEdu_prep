"""
Tests for the question validator. Run with:  pytest -q

Good habit: every rule in schemas.py gets one test that proves it works,
and one that proves bad data is rejected.
"""
import pytest
from pydantic import ValidationError

from core.schemas import QuestionIn


def make_valid(**overrides) -> dict:
    """Build a valid question dict, then let each test break ONE thing."""
    base = {
        "topic": "Current Electricity",
        "competency": "Apply Ohm's law",
        "text": "A 12 V battery drives a current through a 4 ohm resistor. Find the current.",
        "options": {"A": "1 A", "B": "2 A", "C": "3 A", "D": "48 A"},
        "correct_option": "C",
        "difficulty": 2,
    }
    base.update(overrides)
    return base


def test_valid_question_passes():
    q = QuestionIn(**make_valid())
    assert q.correct_option == "C"


def test_correct_option_is_normalised():
    assert QuestionIn(**make_valid(correct_option=" c ")).correct_option == "C"


def test_missing_option_rejected():
    with pytest.raises(ValidationError):
        QuestionIn(**make_valid(options={"A": "1 A", "B": "2 A", "C": "3 A"}))


def test_duplicate_options_rejected():
    with pytest.raises(ValidationError):
        QuestionIn(**make_valid(options={"A": "3 A", "B": "3 A", "C": "1 A", "D": "2 A"}))


def test_correct_option_outside_range_rejected():
    with pytest.raises(ValidationError):
        QuestionIn(**make_valid(correct_option="E"))


def test_difficulty_out_of_range_rejected():
    with pytest.raises(ValidationError):
        QuestionIn(**make_valid(difficulty=9))
