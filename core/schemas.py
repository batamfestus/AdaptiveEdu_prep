"""
core/schemas.py
---------------
Pydantic models = the "gatekeepers" for incoming data.

Rule of thumb:
    ORM classes (orm.py)   -> how data is STORED
    Pydantic (this file)   -> whether data is VALID before we store it

The SAME QuestionIn model will validate:
    1. rows from an admin's CSV upload,
    2. the admin's single-question form,
    3. JSON that Claude returns when it drafts questions.
One gatekeeper, three entry points.
"""
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

ALLOWED_OPTION_KEYS = {"A", "B", "C", "D"}


class QuestionIn(BaseModel):
    topic: str = Field(min_length=2)
    competency: str = Field(min_length=2)
    text: str = Field(min_length=10, description="The question stem")
    options: dict[str, str]
    correct_option: str
    difficulty: int = Field(ge=1, le=5)
    explanation: str = ""
    source: Literal["original", "past_paper_inspired", "ai_drafted"] = "original"
    image_path: str | None = None

    @field_validator("text", "topic", "competency", "explanation")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        # Spreadsheets often contain stray spaces; clean them here once.
        return v.strip()

    @field_validator("correct_option")
    @classmethod
    def normalise_correct_option(cls, v: str) -> str:
        return v.strip().upper()  # accept " b " and store "B"

    @field_validator("options")
    @classmethod
    def check_options(cls, v: dict[str, str]) -> dict[str, str]:
        if set(v.keys()) != ALLOWED_OPTION_KEYS:
            raise ValueError("options must have exactly the keys A, B, C and D")
        if any(not str(text).strip() for text in v.values()):
            raise ValueError("every option needs text")
        if len({str(t).strip().lower() for t in v.values()}) < 4:
            raise ValueError("options must all be different")
        return {k: str(t).strip() for k, t in v.items()}

    @model_validator(mode="after")
    def correct_option_exists(self) -> "QuestionIn":
        # Runs after single fields pass; checks rules that involve 2+ fields.
        if self.correct_option not in self.options:
            raise ValueError(f"correct_option '{self.correct_option}' is not one of A-D")
        return self
