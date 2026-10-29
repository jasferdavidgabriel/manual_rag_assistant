"""Unit tests for prompt building and response parsing (no API calls)."""

import types

import pytest

from manual_rag.generation import (
    EMPTY_ANSWER_MESSAGE,
    LLMClient,
    build_prompt,
    clean_title,
    split_reasoning,
)
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
    prompt = build_prompt("q", [_hit()])
    assert '"The manual does not cover this." and nothing else' in prompt


def test_prompt_forbids_invented_steps():
    # Regression: the model once added made up "remedy" steps to an answer
    # about arcing causes because the template always showed a Steps section.
    prompt = build_prompt("q", [_hit()])
    assert "Never invent steps" in prompt
    assert "only if the manual gives steps" in prompt


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


def _client(outcomes, fallbacks=("backup-model",), extra_cfg=None):
    """Build an LLMClient whose fake API returns or raises per model name."""
    calls = []

    def create(model, **kwargs):
        calls.append(model)
        create.last_kwargs = kwargs
        result = outcomes[model]
        if isinstance(result, Exception):
            raise result
        return _response(result)

    client = LLMClient.__new__(LLMClient)  # skip __init__ (no openai / key needed)
    client.cfg = {"model": "main-model", "fallback_models": list(fallbacks), **(extra_cfg or {})}
    client.create = create
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


def test_prompt_asks_for_separate_answers_per_manual():
    # Regression: an ambiguous warranty question matched two manuals and the
    # model spent its whole budget deliberating over how to combine them.
    prompt = build_prompt("q", [_hit()])
    assert "separate answer for each manual" in prompt
    assert "Never merge facts from different manuals" in prompt


def test_empty_answer_is_replaced_with_clear_message():
    client, _ = _client({"main-model": ""})
    answer = client.generate("q", [_hit()])
    assert answer.answer == EMPTY_ANSWER_MESSAGE


# --- Manual titles for contextual headers ---------------------------------- #
def test_clean_title():
    assert clean_title('"Mr. Coffee NL4 4-cup coffeemaker"') == "Mr. Coffee NL4 4-cup coffeemaker"
    assert clean_title("Product name: GE JVM3160 microwave.\nExtra text") == "GE JVM3160 microwave"
    assert clean_title("") is None
    assert len(clean_title("word " * 30).split()) == 12


def test_describe_manual_uses_llm_and_cleans_output():
    client, calls = _client({"main-model": '"Mr. Coffee NL4 4-cup coffeemaker"'})
    title = client.describe_manual([{"page": 1, "text": "MR. COFFEE 4-CUP COFFEEMAKER"}])
    assert title == "Mr. Coffee NL4 4-cup coffeemaker"
    assert calls == ["main-model"]


def test_prompt_shows_manual_title():
    hit = Hit("Text.", 9, "Warranty", 5, 11, 0.5, "mr_coffee_nl4.pdf", "Mr. Coffee 4-cup coffeemaker")
    assert "Manual: mr_coffee_nl4.pdf (Mr. Coffee 4-cup coffeemaker)" in build_prompt("q", [hit])


# --- Speed: model specific switches such as turning reasoning off ---------- #
def test_chat_template_kwargs_are_sent_to_the_model():
    client, _ = _client({"main-model": "**Answer:** ok"},
                        extra_cfg={"chat_template_kwargs": {"enable_thinking": False}})
    client.generate("q", [_hit()])
    assert client.create.last_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    client.describe_manual([{"page": 1, "text": "x"}])  # title requests are fast too
    assert client.create.last_kwargs["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_no_extra_body_when_not_configured():
    client, _ = _client({"main-model": "**Answer:** ok"})
    client.generate("q", [_hit()])
    assert "extra_body" not in client.create.last_kwargs


def test_title_prompt_examples_are_not_the_test_manuals():
    # Regression: using the indexed manuals as examples let the model copy them.
    from manual_rag.generation import DESCRIBE_PROMPT
    assert "Mr. Coffee" not in DESCRIBE_PROMPT and "JVM3160" not in DESCRIBE_PROMPT
