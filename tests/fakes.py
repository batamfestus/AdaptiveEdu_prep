"""
Test doubles for the AI. Tests NEVER call the real Claude API: a fake gives the same
answer every time, costs nothing, needs no internet, and lets us check exactly what we SENT.
"""
import hashlib
import math
import re

from core.llm import LLMReply, Usage


class FakeLLM:
    """Stands in for core.llm.AnthropicClient. Records every call it receives."""
    model = "fake-model"
    supports_web_search = True

    def __init__(self, structured: dict | None = None, replies: list[LLMReply] | None = None,
                 usage: Usage = Usage(100, 50), error: Exception | None = None,
                 web_error: Exception | None = None):
        self.structured_data = structured if structured is not None else good_explanation()
        self.replies = list(replies or [])
        self.usage = usage
        self.error = error
        self.web_error = web_error          # raised only on calls that include the web search tool
        self.structured_calls: list[dict] = []
        self.converse_calls: list[dict] = []

    def structured(self, *, system, user, schema, tool_name, max_tokens):
        self.structured_calls.append(dict(system=system, user=user, schema=schema, tool_name=tool_name, max_tokens=max_tokens))
        if self.error:
            raise self.error
        return dict(self.structured_data), self.usage

    def converse(self, *, system, messages, tools, max_tokens, web_search=None):
        # keep a COPY: the code under test keeps appending to its own list
        self.converse_calls.append(dict(system=system, messages=[dict(m) for m in messages], tools=list(tools),
                                        max_tokens=max_tokens, web_search=web_search))
        if self.error:
            raise self.error
        if self.web_error and web_search is not None:
            raise self.web_error
        if not self.replies:
            return LLMReply("Default answer.", [], self.usage)
        return self.replies.pop(0)


def good_explanation(**overrides) -> dict:
    data = {
        "explanation": "Use $I = V/R$. With 12 V across 4 ohm, $I = 3$ A.",
        "why_your_answer_is_wrong": "You divided the wrong way round.",
        "misconception": "Mixing up V/R with R/V.",
        "study_tip": "Practise three more Ohm's law problems.",
        "sources": [],
        "needs_review": False,
    }
    data.update(overrides)
    return data


def hash_embed(texts: list[str], dims: int = 64) -> list[list[float]]:
    """A tiny deterministic 'embedding': hash each word into one of `dims` buckets.
    Texts that share words end up close together. Good enough to test the plumbing offline."""
    vectors = []
    for text in texts:
        v = [0.0] * dims
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % dims] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        vectors.append([x / norm for x in v])
    return vectors
