"""Unit tests for prompt building and response parsing (no API calls)."""

import types

import pytest

from manual_rag.generation import LLMClient, build_prompt, split_reasoning
from manual_rag.retrieval import Hit


def _hit():
    return Hit("Remove the screw.", 17, "Cooktop Light/Nite Light", 5, 13, 0.6, "734301.pdf")


def test_prompt_contains_location_and_question():
    prompt = build_prompt("How do I change the bulb?", [_hit()])
    for expected in ("Manual: 734301.pdf", "Page: 17", "Section: Cooktop Light/Nite Light",
                     "Lines: 5-13", "Remove the screw.", "Question: How do I change the bulb?"):
        assert expected in prompt


def test_prompt_numbers_chunks_in_rank_order():
    second = Hit("Other text.", 4, "Safety", 1, 3, 0.4, "734301.pdf")
    prompt = build_prompt("q", [_hit(), second])
    assert prompt.index("[Chunk 1 |") < prompt.index("[Chunk 2 |")


def test_prompt_has_refusal_rule():
    assert "The manual does not cover this." in build_prompt("q", [_hit()])


def test_split_reasoning_inline_tags():
    answer, reasoning = split_reasoning("<think>check page 17</think>\n**Answer:** Replace it.")
    assert answer == "**Answer:** Replace it."
    assert reasoning == "check page 17"


def test_split_reasoning_separate_field():
    assert split_reasoning("**Answer:** Done.", "thinking...") == ("**Answer:** Done.", "thinking...")


def test_split_reasoning_handles_empty_content():
    assert split_reasoning(None) == ("", None)


# --- Fallback behaviour, using a fake client (no network) ------------------ #
class APITimeoutError(Exception):  # same name as the openai error
    pass


class AuthenticationError(Exception):
    pass


def _response(text):
    message = types.SimpleNamespace(content=text, reasoning_content="thought about it")
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message, finish_reason="stop")])


def _client(outcomes, fallbacks=("backup-model",)):
    """Build an LLMClient whose fake API returns or raises per model name."""
    calls = []

    def create(model, **kwargs):
        calls.append(model)
        result = outcomes[model]
        if isinstance(result, Exception):
            raise result
        return _response(result)

    client = LLMClient.__new__(LLMClient)  # skip __init__ (no openai / key needed)
    client.cfg = {"model": "main-model", "fallback_models": list(fallbacks)}
    client.client = types.SimpleNamespace(chat=types.SimpleNamespace(
        completions=types.SimpleNamespace(create=create)))
    return client, calls


def test_primary_model_used_when_healthy():
    client, calls = _client({"main-model": "**Answer:** ok"})
    answer = client.generate("q", [_hit()])
    assert answer.model == "main-model" and calls == ["main-model"]
    assert answer.reasoning == "thought about it"


def test_falls_back_when_primary_times_out():
    client, calls = _client({"main-model": APITimeoutError(), "backup-model": "**Answer:** ok"})
    answer = client.generate("q", [_hit()])
    assert answer.model == "backup-model"
    assert calls == ["main-model", "backup-model"]


def test_auth_error_is_not_retried():
    client, calls = _client({"main-model": AuthenticationError(), "backup-model": "x"})
    with pytest.raises(AuthenticationError):
        client.generate("q", [_hit()])
    assert calls == ["main-model"]


def test_all_models_failing_raises_clear_error():
    client, _ = _client({"main-model": APITimeoutError(), "backup-model": APITimeoutError()})
    with pytest.raises(RuntimeError, match="All models failed"):
        client.generate("q", [_hit()])
