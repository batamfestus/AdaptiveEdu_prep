"""
core/llm.py
-----------
The ONLY file that talks to the Anthropic API.

Everything else (core/tutor.py) depends on the small `LLMClient` interface below, not on
the Anthropic SDK. That has three payoffs:
  * TESTS use a fake client, so they cost nothing, need no internet, and are repeatable.
  * Changing model or provider later touches this one file.
  * Error handling is in one place: the rest of the app only sees our own LLMError types.

Three techniques from the job description live here:
  * STRUCTURED OUTPUT: we give Claude a "tool" whose input schema is our Pydantic model and
    force it to call that tool. The reply is then a JSON object that already matches the
    schema, instead of free text we would have to parse and hope for the best.
  * TOOL CALLING: in the chat tutor, Claude may call a `search_notes` tool whenever it
    decides it needs a fact; our code runs the search and returns the result.
  * A SERVER TOOL: Anthropic's built-in web search. Unlike our own tools, Anthropic runs the search
    itself and the answer comes back with citations. We cap it (max_uses), restrict it to a list
    of trusted sites, and treat everything it brings back as untrusted.
"""
from dataclasses import dataclass, field
from typing import Protocol

from pydantic import BaseModel


class LLMError(Exception):
    """The AI call failed. The message is safe to show the student."""


class LLMUnavailable(LLMError):
    """Not configured, can't be reached, or out of credit."""


class LLMRateLimited(LLMError):
    """The provider asked us to slow down."""


class LLMWebSearchUnavailable(LLMError):
    """Web search is switched off for this account, or this model can't use it."""


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(self.input_tokens + other.input_tokens, self.output_tokens + other.output_tokens)

    @property
    def total(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass(frozen=True)
class WebSearchConfig:
    """How the web search tool is allowed to run for ONE request."""
    max_uses: int                        # hard cap on searches in this request
    allowed_domains: tuple[str, ...]     # the ONLY sites it may use


@dataclass(frozen=True)
class WebSource:
    title: str
    url: str


@dataclass(frozen=True)
class LLMReply:
    text: str
    tool_calls: list[ToolCall]
    usage: Usage
    # The assistant turn EXACTLY as received. When we send it back (after running a tool, or to
    # continue a paused turn) the API requires it unchanged, including encrypted search results.
    content_blocks: list = field(default_factory=list)
    stop_reason: str = ""                                    # "end_turn", "tool_use", "pause_turn"...
    web_searches: int = 0                                    # billed web searches in this reply
    web_queries: list[str] = field(default_factory=list)     # what Claude searched for
    web_sources: list[WebSource] = field(default_factory=list)  # pages it cited (or, failing that, found)


class LLMClient(Protocol):
    model: str
    supports_web_search: bool

    def structured(self, *, system: str, user: str, schema: type[BaseModel], tool_name: str,
                   max_tokens: int) -> tuple[dict, Usage]: ...

    def converse(self, *, system: str, messages: list[dict], tools: list[dict],
                 max_tokens: int, web_search: WebSearchConfig | None = None) -> LLMReply: ...


class AnthropicClient:
    """Real implementation. `client` can be passed in to test this class without a network."""

    supports_web_search = True
    WEB_TOOL_TYPE = "web_search_20250305"   # the basic version: no code sandbox, zero-data-retention eligible

    def __init__(self, api_key: str, model: str, *, client=None, timeout: float = 30.0):
        import anthropic

        self._anthropic = anthropic
        self.model = model
        self._client = client or anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=2)

    # ---- one place for every API call, so errors are translated once ----
    def _create(self, **kwargs):
        a = self._anthropic
        try:
            return self._client.messages.create(model=self.model, **kwargs)
        except a.RateLimitError:
            raise LLMRateLimited("The AI tutor is very busy right now. Please try again in a minute.")
        except a.AuthenticationError:
            raise LLMUnavailable("The AI tutor is not set up correctly (the API key was rejected).")
        except a.APIConnectionError:  # includes timeouts
            raise LLMUnavailable("Could not reach the AI service. Please try again.")
        except a.BadRequestError as e:
            message = str(e).lower()
            if "credit balance" in message:
                raise LLMUnavailable("The AI tutor has run out of credits. Please tell your teacher.")
            if "web search" in message or "web_search" in message:
                raise LLMWebSearchUnavailable(
                    "Web search isn't available (it may be switched off for this account, or this model can't use it).")
            raise LLMError("The AI service could not process that request.")
        except a.APIStatusError as e:
            raise LLMError(f"The AI service returned an error ({e.status_code}). Please try again.")

    @staticmethod
    def _usage(response) -> Usage:
        u = getattr(response, "usage", None)
        return Usage(getattr(u, "input_tokens", 0) or 0, getattr(u, "output_tokens", 0) or 0)

    def structured(self, *, system, user, schema, tool_name, max_tokens):
        tool = {
            "name": tool_name,
            "description": "Submit the answer in exactly this structure.",
            "input_schema": schema.model_json_schema(),
        }
        response = self._create(
            system=system, max_tokens=max_tokens, tools=[tool],
            tool_choice={"type": "tool", "name": tool_name},   # FORCE the structured answer
            messages=[{"role": "user", "content": user}],
        )
        for block in response.content:
            if block.type == "tool_use" and block.name == tool_name:
                return dict(block.input), self._usage(response)
        raise LLMError("The AI did not return a structured answer. Please try again.")

    def converse(self, *, system, messages, tools, max_tokens, web_search=None):
        all_tools = list(tools)
        if web_search is not None and web_search.max_uses > 0:
            web_tool = {"type": self.WEB_TOOL_TYPE, "name": "web_search", "max_uses": web_search.max_uses}
            if web_search.allowed_domains:
                web_tool["allowed_domains"] = list(web_search.allowed_domains)
            all_tools.append(web_tool)
        kwargs = dict(system=system, max_tokens=max_tokens, messages=messages)
        if all_tools:
            kwargs["tools"] = all_tools
        response = self._create(**kwargs)

        texts, calls, queries, cited, found = [], [], [], [], []
        for block in response.content:
            if block.type == "text":
                texts.append(block.text)
                for citation in getattr(block, "citations", None) or []:
                    if getattr(citation, "type", "") == "web_search_result_location":
                        cited.append(WebSource(citation.title or citation.url, citation.url))
            elif block.type == "tool_use":
                calls.append(ToolCall(block.id, block.name, dict(block.input)))
            elif block.type == "server_tool_use" and block.name == "web_search":
                queries.append(str(dict(block.input).get("query", "")))
            elif block.type == "web_search_tool_result":
                results = block.content if isinstance(block.content, list) else []   # an error is a single object
                found += [WebSource(r.title or r.url, r.url) for r in results if getattr(r, "type", "") == "web_search_result"]

        server_usage = getattr(getattr(response, "usage", None), "server_tool_use", None)
        return LLMReply(
            text="".join(texts), tool_calls=calls, usage=self._usage(response),
            content_blocks=list(response.content),        # raw, unchanged: the API insists on that
            stop_reason=getattr(response, "stop_reason", "") or "",
            web_searches=getattr(server_usage, "web_search_requests", 0) or 0,
            web_queries=queries, web_sources=cited or found[:3],
        )
