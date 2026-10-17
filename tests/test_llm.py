"""
Tests for core/llm.py. We hand AnthropicClient a STUB in place of the real SDK client, so
we can check exactly what it sends and how it reacts to every kind of failure, offline.
"""
from types import SimpleNamespace

import anthropic
import httpx
import pytest
from pydantic import BaseModel

from core.llm import (
    AnthropicClient, LLMError, LLMRateLimited, LLMUnavailable, LLMWebSearchUnavailable, Usage, WebSearchConfig,
)

REQUEST = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def api_error(cls, status: int, message: str = "boom"):
    return cls(message, response=httpx.Response(status, request=REQUEST), body=None)


class Stub:
    """Mimics `anthropic.Anthropic(...).messages`."""
    def __init__(self, response=None, error=None):
        self.response, self.error, self.kwargs = response, error, None
        self.messages = self

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def reply(*blocks, tokens=(11, 7)):
    return SimpleNamespace(content=list(blocks), usage=SimpleNamespace(input_tokens=tokens[0], output_tokens=tokens[1]))


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(name, **input):
    return SimpleNamespace(type="tool_use", id="toolu_1", name=name, input=input)


class Schema(BaseModel):
    answer: str


def client_for(stub) -> AnthropicClient:
    return AnthropicClient("sk-test", "model-x", client=stub)


# ------------------------------------------------------------------ structured output
def test_structured_forces_the_tool_and_returns_its_input():
    stub = Stub(reply(tool_use("submit", answer="42")))
    data, usage = client_for(stub).structured(system="sys", user="hi", schema=Schema, tool_name="submit", max_tokens=100)
    assert data == {"answer": "42"} and usage == Usage(11, 7) and usage.total == 18
    sent = stub.kwargs
    assert sent["model"] == "model-x" and sent["max_tokens"] == 100 and sent["system"] == "sys"
    assert sent["tool_choice"] == {"type": "tool", "name": "submit"}          # the answer is FORCED into the schema
    assert sent["tools"][0]["input_schema"]["properties"]["answer"]["type"] == "string"


def test_structured_without_a_tool_reply_is_an_error():
    with pytest.raises(LLMError, match="structured"):
        client_for(Stub(reply(text("just prose")))).structured(system="s", user="u", schema=Schema, tool_name="submit", max_tokens=10)


# ------------------------------------------------------------------ conversation with tools
def test_converse_parses_text_and_tool_calls():
    stub = Stub(reply(text("Let me check. "), tool_use("search_notes", query="ohm")))
    out = client_for(stub).converse(system="s", messages=[{"role": "user", "content": "hi"}],
                                    tools=[{"name": "search_notes"}], max_tokens=50)
    assert out.text == "Let me check. "
    assert out.tool_calls[0].name == "search_notes" and out.tool_calls[0].input == {"query": "ohm"}
    assert out.content_blocks == list(stub.response.content)    # handed back RAW: the API needs them unchanged
    assert stub.kwargs["tools"] == [{"name": "search_notes"}]


def test_converse_omits_tools_when_there_are_none():
    stub = Stub(reply(text("hi")))
    client_for(stub).converse(system="s", messages=[], tools=[], max_tokens=5)
    assert "tools" not in stub.kwargs


# ------------------------------------------------------------------ every failure becomes OUR error type
@pytest.mark.parametrize("error, expected, fragment", [
    (api_error(anthropic.RateLimitError, 429), LLMRateLimited, "busy"),
    (api_error(anthropic.AuthenticationError, 401), LLMUnavailable, "API key"),
    (anthropic.APIConnectionError(request=REQUEST), LLMUnavailable, "reach"),
    (api_error(anthropic.BadRequestError, 400, "Your credit balance is too low"), LLMUnavailable, "credits"),
    (api_error(anthropic.BadRequestError, 400, "something else"), LLMError, "could not process"),
    (api_error(anthropic.InternalServerError, 500), LLMError, "500"),
])
def test_api_failures_are_translated(error, expected, fragment):
    with pytest.raises(expected, match=fragment):
        client_for(Stub(error=error)).converse(system="s", messages=[], tools=[], max_tokens=5)


def test_the_api_key_never_appears_in_error_messages():
    with pytest.raises(LLMError) as caught:
        client_for(Stub(error=api_error(anthropic.AuthenticationError, 401, "bad key sk-test"))).converse(
            system="s", messages=[], tools=[], max_tokens=5)
    assert "sk-test" not in str(caught.value)


# ================================================================== web search (Lesson 7)
def server_tool_use(query):
    return SimpleNamespace(type="server_tool_use", id="srvtoolu_1", name="web_search", input={"query": query})


def search_results(*pages):
    items = [SimpleNamespace(type="web_search_result", url=url, title=title) for title, url in pages]
    return SimpleNamespace(type="web_search_tool_result", tool_use_id="srvtoolu_1", content=items)


def cited_text(words, *pages):
    cites = [SimpleNamespace(type="web_search_result_location", url=url, title=title) for title, url in pages]
    return SimpleNamespace(type="text", text=words, citations=cites)


def with_web_usage(response, searches, stop_reason="end_turn"):
    response.usage.server_tool_use = SimpleNamespace(web_search_requests=searches)
    response.stop_reason = stop_reason
    return response


def test_the_web_tool_is_added_with_a_cap_and_an_allow_list():
    stub = Stub(reply(text("hi")))
    client = client_for(stub)
    assert client.supports_web_search is True
    client.converse(system="s", messages=[], tools=[{"name": "search_notes"}], max_tokens=5,
                    web_search=WebSearchConfig(2, ("khanacademy.org", "bbc.co.uk/bitesize")))
    assert stub.kwargs["tools"] == [
        {"name": "search_notes"},
        {"type": "web_search_20250305", "name": "web_search", "max_uses": 2,
         "allowed_domains": ["khanacademy.org", "bbc.co.uk/bitesize"]},
    ]


@pytest.mark.parametrize("config", [None, WebSearchConfig(0, ("a.com",))])
def test_no_web_tool_without_permission(config):
    stub = Stub(reply(text("hi")))
    client_for(stub).converse(system="s", messages=[], tools=[], max_tokens=5, web_search=config)
    assert "tools" not in stub.kwargs


def test_a_web_search_turn_is_parsed():
    response = with_web_usage(reply(
        text("I'll look that up. "), server_tool_use("hooke's law spring constant"),
        search_results(("Hooke's law", "https://phys.libretexts.org/hooke"), ("Springs", "https://physics.info/springs")),
        cited_text("F = kx.", ("Hooke's law", "https://phys.libretexts.org/hooke")),
    ), searches=1)
    out = client_for(Stub(response)).converse(system="s", messages=[], tools=[], max_tokens=5,
                                              web_search=WebSearchConfig(1, ("physics.info",)))
    assert out.text == "I'll look that up. F = kx."
    assert out.web_searches == 1 and out.web_queries == ["hooke's law spring constant"]
    assert [(s.title, s.url) for s in out.web_sources] == [("Hooke's law", "https://phys.libretexts.org/hooke")]  # cited pages win
    assert out.stop_reason == "end_turn" and out.tool_calls == []


def test_uncited_results_are_the_fallback_sources():
    response = with_web_usage(reply(server_tool_use("q"), search_results(
        ("A", "https://a.org/1"), ("B", "https://b.org/2"), ("C", "https://c.org/3"), ("D", "https://d.org/4")), text("answer")), 1)
    out = client_for(Stub(response)).converse(system="s", messages=[], tools=[], max_tokens=5, web_search=WebSearchConfig(1, ()))
    assert [s.title for s in out.web_sources] == ["A", "B", "C"]


def test_a_failed_search_result_is_not_a_source():
    error_result = SimpleNamespace(type="web_search_tool_result", tool_use_id="srvtoolu_1",
                                   content=SimpleNamespace(type="web_search_tool_result_error", error_code="max_uses_exceeded"))
    out = client_for(Stub(with_web_usage(reply(server_tool_use("q"), error_result, text("sorry")), 0))).converse(
        system="s", messages=[], tools=[], max_tokens=5, web_search=WebSearchConfig(1, ()))
    assert out.web_sources == [] and out.web_searches == 0 and out.text == "sorry"


def test_a_paused_turn_is_reported_with_its_raw_blocks():
    blocks = [server_tool_use("q")]
    out = client_for(Stub(with_web_usage(reply(*blocks), 0, stop_reason="pause_turn"))).converse(
        system="s", messages=[], tools=[], max_tokens=5, web_search=WebSearchConfig(1, ()))
    assert out.stop_reason == "pause_turn" and out.content_blocks == blocks


@pytest.mark.parametrize("message", ["web search is not enabled for this organization", "The web_search tool is not supported by this model"])
def test_web_search_being_unavailable_has_its_own_error(message):
    with pytest.raises(LLMWebSearchUnavailable, match="isn't available"):
        client_for(Stub(error=api_error(anthropic.BadRequestError, 400, message))).converse(
            system="s", messages=[], tools=[], max_tokens=5, web_search=WebSearchConfig(1, ()))
