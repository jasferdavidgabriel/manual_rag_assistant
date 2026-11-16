"""Tests for the tool calling agent. A scripted fake model and a fake
retriever stand in for the real ones, so no API key or index is needed."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from manual_rag import agent as agent_mod  # noqa: E402
from manual_rag.agent import Agent  # noqa: E402
from manual_rag.retrieval import Hit  # noqa: E402


def hit(page, section, source="mr_coffee_nl4.pdf", text="Fill with vinegar and brew."):
    return Hit(text=text, page=page, section=section, start_line=10, end_line=14,
               similarity=0.6, source=source, manual_title="Mr. Coffee 4-cup coffee maker")


class FakeCfg:
    def __init__(self, agent=None):
        self.sections = {
            "llm": {"model": "main-model", "fallback_models": ["backup-model"], "temperature": 0},
            "agent": agent or {},
        }

    def section(self, name):
        return self.sections.get(name, {})


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def retrieve(self, query, top_k=5, manual=None):
        self.calls.append((query, top_k, manual))
        if "warranty" in query:
            return [hit(9, "Warranty", text="One year limited warranty.")]
        return [hit(5, "Cleaning"), hit(6, "Descaling")]


def tool_call(name, arguments, call_id="call_1"):
    args = arguments if isinstance(arguments, str) else json.dumps(arguments)
    return SimpleNamespace(id=call_id, function=SimpleNamespace(name=name, arguments=args))


def reply(content=None, calls=None):
    msg = SimpleNamespace(content=content, tool_calls=calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


class ScriptedClient:
    """Returns the scripted replies in order and records every request."""

    def __init__(self, replies, fail_models=()):
        self.replies = list(replies)
        self.fail_models = set(fail_models)
        self.requests = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        if kwargs["model"] in self.fail_models:
            raise TimeoutError("overloaded")
        return self.replies.pop(0)


def make_agent(replies, cfg=None, **kw):
    client = ScriptedClient(replies, **kw)
    retriever = FakeRetriever()
    a = Agent(cfg or FakeCfg(), retriever=retriever, client=client,
              manuals_fn=lambda: {"mr_coffee_nl4.pdf": "Mr. Coffee 4-cup coffee maker",
                                  "ge_microwave_jvm3160.pdf": "GE JVM3160 microwave oven"})
    return a, client, retriever


def test_tools_are_described_for_the_model():
    names = [t["function"]["name"] for t in agent_mod.TOOLS]
    assert names == ["search_manual", "list_manuals"]
    assert agent_mod.TOOLS[0]["function"]["parameters"]["required"] == ["query"]


def test_search_then_answer():
    a, client, retriever = make_agent([
        reply(calls=[tool_call("search_manual", {"query": "descale"})]),
        reply(content="Run vinegar through it (mr_coffee_nl4.pdf, page 6, Descaling)."),
    ])
    result = a.run("How do I descale it?", manual="mr_coffee_nl4.pdf")

    assert result.answer.startswith("Run vinegar")
    assert result.model == "main-model"
    assert result.model_calls == 2
    assert [(s.tool, s.arguments) for s in result.steps] == [("search_manual", {"query": "descale"})]
    assert retriever.calls == [("descale", 5, "mr_coffee_nl4.pdf")]  # default manual used
    assert [h.page for h in result.hits] == [5, 6]

    second = client.requests[1]["messages"]
    assert second[-2]["role"] == "assistant" and second[-2]["tool_calls"][0]["id"] == "call_1"
    assert second[-1]["role"] == "tool" and "page 6 | section: Descaling" in second[-1]["content"]


def test_several_tool_calls_and_searches():
    a, client, retriever = make_agent([
        reply(calls=[tool_call("list_manuals", {}, "c1"),
                     tool_call("search_manual", {"query": "warranty", "manual": "mr_coffee_nl4.pdf"}, "c2")]),
        reply(calls=[tool_call("search_manual", {"query": "warranty",
                                                 "manual": "ge_microwave_jvm3160.pdf"}, "c3")]),
        reply(content="Both have one year warranties."),
    ])
    result = a.run("Which has the longer warranty?")
    assert [s.tool for s in result.steps] == ["list_manuals", "search_manual", "search_manual"]
    assert [c[2] for c in retriever.calls] == ["mr_coffee_nl4.pdf", "ge_microwave_jvm3160.pdf"]
    tool_messages = [m for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert "GE JVM3160 microwave oven" in tool_messages[0]["content"]
    assert len(result.hits) == 1  # the same warranty section twice counts once


def test_bad_tool_input_is_reported_back_to_the_model():
    a, client, _ = make_agent([
        reply(calls=[tool_call("search_manual", "{not json", "c1"),
                     tool_call("delete_everything", {}, "c2"),
                     tool_call("search_manual", {"query": "  "}, "c3")]),
        reply(content="Sorry, I couldn't search."),
    ])
    result = a.run("Hi")
    contents = [m["content"] for m in client.requests[1]["messages"] if m["role"] == "tool"]
    assert contents[0].startswith("Error: could not read the tool arguments")
    assert contents[1].startswith("Error: unknown tool delete_everything")
    assert contents[2] == "Error: search_manual needs a query."
    assert result.answer == "Sorry, I couldn't search."


def test_stops_after_max_steps_and_forces_an_answer():
    looping = [reply(calls=[tool_call("search_manual", {"query": f"try {i}"}, f"c{i}")])
               for i in range(2)]
    a, client, _ = make_agent(looping + [reply(content="Best answer from what I found.")],
                              cfg=FakeCfg({"max_steps": 2}))
    result = a.run("Something hard")
    assert len(result.steps) == 2
    assert result.answer == "Best answer from what I found."
    assert result.model_calls == 3
    assert "tools" not in client.requests[-1]  # the last call offers no tools


def test_falls_back_to_the_next_model():
    a, client, _ = make_agent([reply(calls=[tool_call("search_manual", {"query": "hi"})]),
                               reply(content="Answered by the backup.")],
                              fail_models={"main-model"})
    result = a.run("Hi")
    assert result.model == "backup-model"
    assert [r["model"] for r in client.requests][:2] == ["main-model", "backup-model"]


def test_all_models_failing_raises():
    a, _, _ = make_agent([], fail_models={"main-model", "backup-model"})
    with pytest.raises(RuntimeError, match="All models failed"):
        a.run("Hi")


def test_format_hits():
    assert agent_mod.format_hits([]) == "No matching sections found."
    text = agent_mod.format_hits([hit(6, "Descaling", text="x" * 900)], preview_chars=100)
    assert text.startswith("[1] manual: mr_coffee_nl4.pdf (Mr. Coffee 4-cup coffee maker) | page 6")
    assert text.endswith("x" * 100 + "...")


def test_lists_available_manuals_in_the_first_message():
    a, client, _ = make_agent([reply(calls=[tool_call("search_manual", {"query": "x"})]),
                               reply(content="Done.")])
    a.run("Which has the longer warranty?")
    first = client.requests[0]["messages"][1]["content"]
    assert "Available manuals:" in first
    assert "- ge_microwave_jvm3160.pdf: GE JVM3160 microwave oven" in first


def test_nudges_a_model_that_doesnt_search():
    a, client, retriever = make_agent([
        reply(content="I don't know which manuals exist. Let me list them first."),
        reply(calls=[tool_call("search_manual", {"query": "warranty"})]),
        reply(content="One year (mr_coffee_nl4.pdf, page 9, Warranty)."),
    ])
    result = a.run("How long is the warranty?")
    assert result.answer.startswith("One year")
    assert len(result.steps) == 1 and retriever.calls[0][0] == "warranty"
    assert client.requests[1]["messages"][-1] == {"role": "user", "content": agent_mod.NUDGE}


def test_nudges_only_once():
    a, client, _ = make_agent([reply(content="I won't search."), reply(content="Still no.")])
    result = a.run("Hi")
    assert result.answer == "Still no."
    assert len(client.requests) == 2
