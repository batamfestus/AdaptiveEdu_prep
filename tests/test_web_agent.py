"""
Tests for the research agent in core/tutor.py: when the web tool unlocks, how it is capped,
how answers are labelled, and how the API's tool-calling rules are followed.
Run with:  pytest -q

The model is fake, so we can script exactly what it "decides" and inspect exactly what we
sent it on every step of the loop.
"""
from datetime import timedelta

import pytest
from sqlalchemy import select

from core.config import web_domains
from core.llm import LLMReply, LLMWebSearchUnavailable, ToolCall, Usage, WebSource
from core.orm import TutorMessage, TutorUsage
from core.tutor import (
    MAX_PAUSES, REQUEST_WEB_TOOL, SEARCH_TOOL, TutorError, WebStatus, _web_footer, ask_tutor, explain_answer,
    get_thread, usage_summary, web_search_status,
)
from tests.fakes import FakeLLM, good_explanation
# shared helpers and fixtures from the tutor tests
from tests.test_tutor import KB, NOW, bank, finished_test, other, search_call, text_reply, thread  # noqa: F401

HOOKE = [WebSource("Hooke's law", "https://phys.libretexts.org/hooke")]


@pytest.fixture(autouse=True)
def web_on(monkeypatch):
    """Most tests here run with web search switched on and clean budgets."""
    monkeypatch.setenv("TUTOR_WEB_SEARCH", "1")
    for name in ("TUTOR_DAILY_WEB_SEARCHES", "TUTOR_GLOBAL_DAILY_WEB_SEARCHES", "TUTOR_WEB_MAX_USES", "TUTOR_WEB_DOMAINS"):
        monkeypatch.delenv(name, raising=False)


def web_reply(text="Hooke's law says $F = kx$.", searches=1, queries=("hooke's law spring constant",),
              sources=HOOKE, stop="end_turn", usage=Usage(900, 120), blocks=None):
    return LLMReply(text, [], usage, blocks or [], stop_reason=stop, web_searches=searches,
                    web_queries=list(queries), web_sources=list(sources))


def request_web_call():
    block = {"type": "tool_use", "id": "toolu_w", "name": "request_web_search", "input": {"reason": "notes lack Hooke's law"}}
    return LLMReply("", [ToolCall("toolu_w", "request_web_search", {"reason": "notes lack Hooke's law"})],
                    Usage(60, 10), [block], stop_reason="tool_use")


WEAK = "zzzz qqqq"        # matches nothing in the notes, so the notes "don't cover" it
STRONG = "combined resistance of resistors in parallel"


def ask(db, student, thread_id, llm, allow_web=True, message="Explain this more please"):
    return ask_tutor(db, student, thread_id, message, llm=llm, kb=KB, allow_web=allow_web, now=NOW)


def web_rows(db):
    return len(db.scalars(select(TutorUsage).where(TutorUsage.kind == "web")).all())


# ================================================================== when is the web tool even available?
def test_web_is_off_by_default(db, student, thread, monkeypatch):
    monkeypatch.delenv("TUTOR_WEB_SEARCH")
    llm = FakeLLM(replies=[search_call(WEAK), text_reply("I can't find that in the notes.")])
    ask(db, student, thread[0], llm)
    assert all(c["web_search"] is None and c["tools"] == [SEARCH_TOOL] for c in llm.converse_calls)   # not even offered
    assert "NOTE: The study notes do not cover this well and web search is not available" in llm.converse_calls[1]["messages"][-1]["content"][0]["content"]


def test_the_student_must_consent(db, student, thread):
    llm = FakeLLM(replies=[search_call(WEAK), text_reply("Not in the notes.")])
    ask(db, student, thread[0], llm, allow_web=False)
    assert all(c["web_search"] is None and REQUEST_WEB_TOOL not in c["tools"] for c in llm.converse_calls)


def test_the_web_stays_locked_when_the_notes_cover_the_question(db, student, thread):
    llm = FakeLLM(replies=[search_call(STRONG), text_reply("From your notes: 2 ohm.")])
    answer = ask(db, student, thread[0], llm)
    assert all(c["web_search"] is None for c in llm.converse_calls)       # never switched on
    assert "Looked up on the web" not in answer and web_rows(db) == 0
    assert REQUEST_WEB_TOOL in llm.converse_calls[0]["tools"]            # the model COULD have asked, but didn't


# ================================================================== unlocking
def test_weak_notes_unlock_the_web_and_the_answer_is_labelled(db, student, thread):
    llm = FakeLLM(replies=[search_call(WEAK), web_reply()])
    answer = ask(db, student, thread[0], llm)

    first, second = llm.converse_calls
    assert first["web_search"] is None                                         # locked at first
    assert second["web_search"].max_uses == 2 and second["web_search"].allowed_domains == web_domains()
    hint = second["messages"][-1]["content"][0]["content"]
    assert "web_search tool is now available" in hint and "outside the teacher's notes" in hint

    assert "Hooke's law says" in answer and "🌐 **Looked up on the web.**" in answer
    assert "[Hooke's law](https://phys.libretexts.org/hooke)" in answer and "Searched for: “hooke's law spring constant”" in answer
    saved = get_thread(db, student, thread[0]).messages[-1].content
    assert saved == answer                                                    # the label is saved with the message
    assert web_rows(db) == 1
    usage = db.scalars(select(TutorUsage).where(TutorUsage.kind == "chat")).one()
    assert usage.input_tokens == 70 + 900                                     # search pages are billed as input tokens


def test_the_model_can_ask_for_the_web_itself(db, student, thread):
    llm = FakeLLM(replies=[request_web_call(), web_reply()])
    answer = ask(db, student, thread[0], llm)
    assert llm.converse_calls[0]["web_search"] is None and llm.converse_calls[1]["web_search"].max_uses == 2
    assert "now available" in llm.converse_calls[1]["messages"][-1]["content"][0]["content"]
    assert "Looked up on the web" in answer


def test_tool_results_travel_alone_and_assistant_blocks_unchanged(db, student, thread):
    raw = [{"type": "tool_use", "id": "toolu_1", "name": "search_notes", "input": {"query": WEAK}},
           {"type": "server_tool_use", "id": "srvtoolu_9", "name": "web_search", "input": {"query": "x"}}]
    mixed = LLMReply("", [ToolCall("toolu_1", "search_notes", {"query": WEAK})], Usage(50, 10), raw, stop_reason="tool_use")
    llm = FakeLLM(replies=[mixed, web_reply()])
    ask(db, student, thread[0], llm)
    sent = llm.converse_calls[1]["messages"]
    assert sent[-2] == {"role": "assistant", "content": raw}                 # exactly as received
    assert sent[-1]["role"] == "user" and [b["type"] for b in sent[-1]["content"]] == ["tool_result"]   # nothing else


def test_a_paused_turn_is_sent_back_unchanged_and_continued(db, student, thread):
    paused_blocks = [{"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": "q"}}]
    llm = FakeLLM(replies=[search_call(WEAK),
                           web_reply("", searches=0, stop="pause_turn", blocks=paused_blocks),
                           web_reply("Final answer.")])
    answer = ask(db, student, thread[0], llm)
    sent = llm.converse_calls[2]["messages"]
    assert sent[-1] == {"role": "assistant", "content": paused_blocks}        # nothing added after it
    assert llm.converse_calls[2]["web_search"] is not None                     # the tool stays defined
    assert answer.startswith("Final answer.")


def test_a_turn_that_keeps_pausing_is_cut_off(db, student, thread):
    pausing = [web_reply("", searches=0, stop="pause_turn", blocks=[{"type": "server_tool_use"}]) for _ in range(MAX_PAUSES + 1)]
    llm = FakeLLM(replies=[search_call(WEAK)] + pausing)
    with pytest.raises(TutorError, match="couldn't finish"):
        ask(db, student, thread[0], llm)
    assert get_thread(db, student, thread[0]).messages == [] and web_rows(db) == 0


def test_a_failing_web_tool_falls_back_to_the_notes(db, student, thread):
    llm = FakeLLM(replies=[search_call(WEAK), text_reply("My notes don't cover that. Please ask your teacher.")],
                  web_error=LLMWebSearchUnavailable("web search is not enabled"))
    answer = ask(db, student, thread[0], llm)
    assert llm.converse_calls[1]["web_search"] is not None and llm.converse_calls[2]["web_search"] is None   # tried, then dropped
    assert REQUEST_WEB_TOOL not in llm.converse_calls[2]["tools"]
    assert "Looked up on the web" not in answer and web_rows(db) == 0


# ================================================================== budgets
def spend(db, user, n, kind="web"):
    for _ in range(n):
        db.add(TutorUsage(user_id=user.id, kind=kind, input_tokens=0, output_tokens=0, created_at=NOW))
    db.commit()


def test_the_students_daily_allowance_caps_each_message(db, student, thread, monkeypatch):
    monkeypatch.setenv("TUTOR_DAILY_WEB_SEARCHES", "3")
    spend(db, student, 2)                                                      # one search left today
    llm = FakeLLM(replies=[search_call(WEAK), web_reply()])
    ask(db, student, thread[0], llm)
    assert llm.converse_calls[1]["web_search"].max_uses == 1                   # min(per message 2, left today 1)


def test_nothing_is_offered_once_the_allowance_is_used_up(db, student, thread, monkeypatch):
    monkeypatch.setenv("TUTOR_DAILY_WEB_SEARCHES", "2")
    spend(db, student, 2)
    llm = FakeLLM(replies=[search_call(WEAK), text_reply("Not in my notes; ask your teacher.")])
    ask(db, student, thread[0], llm)
    assert all(c["web_search"] is None and REQUEST_WEB_TOOL not in c["tools"] for c in llm.converse_calls)
    assert "web search is not available" in llm.converse_calls[1]["messages"][-1]["content"][0]["content"]


def test_the_app_wide_budget_applies_to_everyone(db, student, other, thread, monkeypatch):
    monkeypatch.setenv("TUTOR_GLOBAL_DAILY_WEB_SEARCHES", "1")
    spend(db, other, 1)                                                        # someone else used the whole budget
    llm = FakeLLM(replies=[search_call(WEAK), text_reply("Ask your teacher.")])
    ask(db, student, thread[0], llm)
    assert all(c["web_search"] is None for c in llm.converse_calls)


def test_the_allowance_rolls_over_after_24_hours(db, student, thread, monkeypatch):
    monkeypatch.setenv("TUTOR_DAILY_WEB_SEARCHES", "1")
    spend(db, student, 1)
    llm = FakeLLM(replies=[search_call(WEAK), web_reply()])
    ask_tutor(db, student, thread[0], "Explain this more please", llm=llm, kb=KB, allow_web=True, now=NOW + timedelta(hours=25))
    assert llm.converse_calls[1]["web_search"] is not None


def test_billed_searches_are_logged_one_row_each(db, student, thread):
    ask(db, student, thread[0], FakeLLM(replies=[search_call(WEAK), web_reply(searches=2)]))
    assert web_rows(db) == 2
    summary = usage_summary(db, NOW)
    assert summary.web_searches == 2 and summary.calls == 1                    # searches are not counted as AI calls


def test_web_status_for_the_interface(db, student, monkeypatch):
    llm = FakeLLM()
    assert web_search_status(db, student, llm, NOW) == WebStatus(True, 5)
    spend(db, student, 2)
    assert web_search_status(db, student, llm, NOW).remaining == 3
    llm.supports_web_search = False
    assert web_search_status(db, student, llm, NOW) == WebStatus(False, 0)
    llm.supports_web_search = True
    monkeypatch.delenv("TUTOR_WEB_SEARCH")
    assert web_search_status(db, student, llm, NOW) == WebStatus(False, 0)


# ================================================================== safety of what the student sees
def test_source_links_are_checked_before_display():
    domains = ("phys.libretexts.org", "bbc.co.uk/bitesize")
    sources = [
        WebSource("Good", "https://phys.libretexts.org/a"),
        WebSource("Duplicate", "https://phys.libretexts.org/a"),
        WebSource("Subdomain is fine", "https://en.phys.libretexts.org/b"),
        WebSource("Off the allow-list", "https://evil.example/phish"),
        WebSource("Script", "javascript:alert(1)"),
        WebSource("Look-alike host", "https://phys.libretexts.org.evil.example/x"),
        WebSource("](https://evil.example) [click", "https://bbc.co.uk/bitesize/c(1)"),
        WebSource("Fifth", "https://phys.libretexts.org/e"),
        WebSource("Sixth", "https://phys.libretexts.org/f"),
    ]
    footer = _web_footer(["q"], sources, domains)
    assert "evil.example" not in footer and "javascript" not in footer
    assert footer.count("](https://") == 4                                      # at most 4 links
    assert "[Good](https://phys.libretexts.org/a)" in footer and "Duplicate" not in footer
    assert "c%281%29" in footer                                                  # brackets in a URL cannot break the link
    assert "] [" not in footer and "](https://evil" not in footer               # a hostile title cannot forge a link
    assert "[click](https://bbc.co.uk/bitesize/c%281%29)" in footer             # what is left of it is harmless text


def test_the_label_appears_even_when_no_source_was_returned(db, student, thread):
    answer = ask(db, student, thread[0], FakeLLM(replies=[search_call(WEAK), web_reply(sources=[])]))
    assert "🌐 **Looked up on the web.**" in answer and "Sources:" not in answer


def test_web_pages_are_declared_untrusted_in_the_system_prompt(db, student, thread):
    llm = FakeLLM(replies=[text_reply("ok")])
    ask(db, student, thread[0], llm)
    system = llm.converse_calls[0]["system"]
    assert "untrusted DATA" in system and "never put personal details in a search query" in system
    assert "Ada" not in system and "ada@example.com" not in system


# ================================================================== the explanation nudge
def test_an_explanation_can_say_the_notes_dont_cover_it(db, bank, student):
    sid = finished_test(db, student, {1: "A"})
    out = explain_answer(db, student, sid, 1, llm=FakeLLM(structured=good_explanation(notes_sufficient=False)), kb=KB, now=NOW)
    assert out.notes_sufficient is False


def test_explanations_default_to_sufficient_notes(db, bank, student):
    sid = finished_test(db, student, {1: "A"})
    assert explain_answer(db, student, sid, 1, llm=FakeLLM(), kb=KB, now=NOW).notes_sufficient is True
