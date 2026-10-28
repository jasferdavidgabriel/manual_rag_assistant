"""Unit tests for prompt building and response parsing (no API calls)."""

from manual_rag.generation import build_prompt, split_reasoning
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
