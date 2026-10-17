"""
scripts/check_ai.py
-------------------
Checks the AI tutor's setup, one step at a time, and says what to fix.

    python -m scripts.check_ai          # notes, search, and one tiny Claude call
    python -m scripts.check_ai --web    # ALSO tries one real web search (costs about 1 cent plus tokens)

Step 3 makes ONE tiny real API call (a few hundred tokens: a fraction of a cent).
It never prints your API key.
"""
import sys

from core.config import (
    BASE_DIR, DEFAULT_MODEL, get_anthropic_key, get_anthropic_model, tutor_limits, web_domains, web_search_enabled,
)
from core.knowledge import KnowledgeBase, load_chunks
from core.llm import AnthropicClient, LLMError, WebSearchConfig
from core.tutor import ExplanationOut


def main() -> None:
    print("1. Study notes")
    chunks = load_chunks(BASE_DIR / "data" / "knowledge")
    print(f"   {len(chunks)} notes loaded.")

    print("2. Search")
    kb = KnowledgeBase(chunks)
    print(f"   mode: {kb.mode}" + (f"   (fallback reason: {kb.fallback_reason})" if kb.fallback_reason else ""))
    if kb.mode == "chroma":
        print("   (the first run may download a small embedding model; later runs are fast)")
    for hit in kb.search("combined resistance of two resistors in parallel", topic="Current Electricity", k=2):
        print(f"   found: {hit.chunk.id}  (score {hit.score:.2f})")

    print("3. Claude")
    key = get_anthropic_key()
    if not key:
        print("   No ANTHROPIC_API_KEY found. Add it to your .env file and run this again.")
        return
    print(f"   model: {get_anthropic_model()}" + ("  (default)" if get_anthropic_model() == DEFAULT_MODEL else "  (from ANTHROPIC_MODEL)"))
    client = AnthropicClient(key, get_anthropic_model())
    try:
        data, usage = client.structured(
            system="You are a physics tutor. Be brief.",
            user="Explain in two sentences why the combined resistance of 6 ohm and 3 ohm in parallel is 2 ohm. "
                 "Use an empty sources list.",
            schema=ExplanationOut, tool_name="submit_explanation", max_tokens=300)
    except LLMError as e:
        print(f"   FAILED: {e}")
        return
    out = ExplanationOut.model_validate(data)
    print(f"   SUCCESS. Tokens used: {usage.input_tokens} in, {usage.output_tokens} out.")
    print(f"   Reply: {out.explanation[:200]}")
    print("4. Limits:", tutor_limits())

    print("5. Web search")
    print(f"   switch (TUTOR_WEB_SEARCH): {'ON' if web_search_enabled() else 'off'} | allowed sites: {len(web_domains())}")
    if "--web" not in sys.argv:
        print("   (run with --web to test one real search)")
        return
    try:
        reply = client.converse(
            system="You are a physics tutor. Use web_search exactly once, then answer in one short sentence.",
            messages=[{"role": "user", "content": "What is the current SI definition of the kilogram?"}],
            tools=[], max_tokens=600, web_search=WebSearchConfig(1, web_domains()))
    except LLMError as e:
        print(f"   FAILED: {e}")
        print("   In the Anthropic Console, check Settings > Capabilities: web search must be enabled for your organization.")
        return
    print(f"   searches billed: {reply.web_searches} | stop reason: {reply.stop_reason} | tokens: "
          f"{reply.usage.input_tokens} in, {reply.usage.output_tokens} out")
    print(f"   queries: {reply.web_queries}")
    for source in reply.web_sources:
        print(f"   source: {source.title} ({source.url})")
    print(f"   reply: {reply.text[:200]}")
    if reply.web_searches == 0:
        print("   The model answered without searching (allowed). Try again or ask something less well known.")


if __name__ == "__main__":
    main()
