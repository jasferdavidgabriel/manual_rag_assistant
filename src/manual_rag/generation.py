"""Prompt construction and answer generation with an OpenAI compatible LLM."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Sequence

from .retrieval import Hit

logger = logging.getLogger(__name__)

PROMPT_TEMPLATE = """You are an expert technical manual assistant.
Your job is to help users find exact instructions from their manuals.

Rules:
1. Answer ONLY from the Manual Context below. Do not use outside knowledge.
2. Always cite the Manual, exact Page number, Section, and Line range where the answer was found.
   The context may be split into groups, one per manual ("=== Manual N: ... ==="). If more than one
   group answers the question, write a separate block for each, starting with "### " and that
   manual's product name, using ONLY that group's chunks and its own Location.
   Never mention one manual's facts inside another manual's block.
3. Include a Steps section ONLY when the manual itself gives steps. Copy them as closely as possible.
   Never invent steps, tips, warnings or advice that the manual does not state.
   If the manual lists causes, conditions or facts rather than steps, put them in the Answer and leave out the Steps section.
4. If the information is not in the provided context, reply with exactly "The manual does not cover this." and nothing else.

Format your response as:
**Answer:** [Your answer here]

**Location in Manual:**
- Manual: [manual name]
- Page: [X]
- Section: [Section name]
- Lines: [start_line - end_line]

**Steps:** (only if the manual gives steps)
1. Step one
2. Step two

Manual Context:
{context}

Question: {question}"""

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)

DESCRIBE_PROMPT = """Below are the first pages of a product manual.
In at most 10 words, name the product it is for: brand, model and product type.
Use ONLY names and numbers that appear in the excerpt. The excerpt may be messy OCR text;
if you cannot read the model number or size with confidence, leave it out rather than guess.
Reply with ONLY the product name, nothing else.
Examples: "Bosch SHX878 built-in dishwasher", "Netgear R7000 Wi-Fi router"

Manual excerpt:
{excerpt}"""


def clean_title(text: str | None, max_words: int = 12) -> str | None:
    """Tidy an LLM product title: first line, no quotes or labels, limited length."""
    if not text:
        return None
    line = text.strip().splitlines()[0].strip()
    line = re.sub(r"^(product( name)?|answer)\s*:\s*", "", line, flags=re.IGNORECASE)
    line = line.strip(" \"'*`.")
    words = line.split()
    return " ".join(words[:max_words]) or None


# Errors where trying the next model makes sense: overloaded, retired, rate
# limited, or a server side failure. Auth errors are NOT here, because a bad
# key fails the same way on every model. Matched by name so tests don't need openai.
RETRYABLE_ERRORS = {
    "APITimeoutError",
    "APIConnectionError",
    "NotFoundError",
    "RateLimitError",
    "InternalServerError",
}


EMPTY_ANSWER_MESSAGE = (
    "The model used its whole token budget reasoning and returned no answer. "
    "Try rephrasing the question, naming the product, or raising llm.max_tokens in the config."
)


@dataclass
class Answer:
    """Final model output, split into the reasoning trace and the answer."""

    answer: str
    reasoning: str | None
    hits: list[Hit]
    model: str = ""  # which model produced the answer


def _chunk_block(i: int, h: Hit) -> str:
    title = f" ({h.manual_title})" if h.manual_title else ""
    return (f"[Chunk {i} | Manual: {h.source}{title} | Page: {h.page} | Section: {h.section} | "
            f"Lines: {h.start_line}-{h.end_line}]\n{h.text}")


def group_by_manual(hits: Sequence[Hit]) -> list[tuple[str, list[Hit]]]:
    """Group hits by manual, keeping the order each manual first appears in
    (so the best matching manual comes first) and the rank order inside it."""
    groups: dict[str, list[Hit]] = {}
    for h in hits:
        groups.setdefault(h.source, []).append(h)
    return list(groups.items())


def build_context(hits: Sequence[Hit]) -> str:
    """Format retrieved chunks with their location headers for the prompt.

    With one manual (the web app's case) chunks are listed as before. With
    several, the code groups them per manual so the model never has to sort a
    mixed pile itself, which is where it mixed up citations without reasoning.
    """
    groups = group_by_manual(hits)
    if len(groups) <= 1:
        return "\n\n".join(_chunk_block(i, h) for i, h in enumerate(hits, start=1))

    parts, i = [], 1
    for n, (source, group) in enumerate(groups, start=1):
        title = f" ({group[0].manual_title})" if group[0].manual_title else ""
        blocks = []
        for h in group:
            blocks.append(_chunk_block(i, h))
            i += 1
        parts.append(f"=== Manual {n}: {source}{title} ===\n\n" + "\n\n".join(blocks))
    return "\n\n".join(parts)


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    """Fill the prompt template with the question and retrieved context."""
    return PROMPT_TEMPLATE.format(context=build_context(hits), question=question)


def split_reasoning(content: str | None, reasoning: str | None = None) -> tuple[str, str | None]:
    """Separate a reasoning trace from the final answer.

    Some providers return reasoning in its own field, others inline it in
    <think>...</think> tags. Both are handled.
    """
    content = content or ""
    match = _THINK_BLOCK.search(content)
    if match:
        content = _THINK_BLOCK.sub("", content)
        reasoning = reasoning or match.group(1)
    return content.strip(), (reasoning.strip() if reasoning else None)


class LLMClient:
    """OpenAI compatible chat client (NVIDIA NIM by default) with fallbacks.

    The primary model is tried first. If it times out, is retired, or is
    overloaded, each model in fallback_models is tried in order.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        from openai import OpenAI

        key_name = cfg.get("api_key_env", "NVIDIA_API_KEY")
        api_key = os.environ.get(key_name)
        if not api_key:
            raise EnvironmentError(f"Set {key_name} in your environment or .env file.")
        self.cfg = cfg
        self.client = OpenAI(
            base_url=cfg["base_url"],
            api_key=api_key,
            timeout=cfg.get("timeout", 90),
            max_retries=cfg.get("max_retries", 0),
        )

    @property
    def models(self) -> list[str]:
        """Primary model followed by fallbacks, without duplicates."""
        ordered = [self.cfg["model"], *self.cfg.get("fallback_models", [])]
        return list(dict.fromkeys(ordered))

    def _request_kwargs(self, question: str, hits: Sequence[Hit]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": [{"role": "user", "content": build_prompt(question, hits)}],
            "temperature": self.cfg.get("temperature", 0.3),
            "top_p": self.cfg.get("top_p", 0.95),
            "max_tokens": self.cfg.get("max_tokens", 4096),
        }
        return self._with_template_kwargs(kwargs)

    def _with_template_kwargs(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Add model specific switches from llm.chat_template_kwargs in the config,
        e.g. {"enable_thinking": False} turns off reasoning on Nemotron models."""
        template = self.cfg.get("chat_template_kwargs")
        if template:
            kwargs["extra_body"] = {"chat_template_kwargs": dict(template)}
        return kwargs

    def _complete(self, kwargs: dict[str, Any]) -> tuple[str, Any]:
        """Send one chat request, falling back through the model list.

        Returns:
            (model name, the response's first choice)
        """
        failures: list[str] = []
        for model in self.models:
            try:
                response = self.client.chat.completions.create(model=model, **kwargs)
            except Exception as exc:
                if type(exc).__name__ not in RETRYABLE_ERRORS:
                    raise
                logger.warning("%s failed (%s), trying next model", model, type(exc).__name__)
                failures.append(f"{model}: {type(exc).__name__}")
                continue
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                logger.warning("Answer was cut off at max_tokens. Raise llm.max_tokens in the config.")
            return model, choice
        raise RuntimeError("All models failed: " + "; ".join(failures))

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer:
        """Call the LLM (with fallbacks) and return a parsed answer."""
        model, choice = self._complete(self._request_kwargs(question, hits))
        message = choice.message
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        answer, reasoning = split_reasoning(message.content, reasoning)
        if not answer:
            # A reasoning model can spend its whole budget thinking. Never show a blank.
            answer = EMPTY_ANSWER_MESSAGE
        return Answer(answer=answer, reasoning=reasoning, hits=list(hits), model=model)

    def describe_manual(self, pages: Sequence[dict[str, Any]], max_chars: int = 4000) -> str | None:
        """Ask the LLM for a short product title from a manual's first pages.

        Called once per manual at ingest time to build contextual chunk headers.
        Deliberately ignores llm.chat_template_kwargs, so reasoning stays ON for
        this one call: the title goes onto every chunk and citation, and without
        reasoning the model guessed a wrong product from a messy scanned cover
        ("12-Cup Programmable" for a 4-cup NL4). It runs once per upload, so the
        extra seconds don't affect answer speed.
        """
        excerpt = "\n".join((p.get("text") or "") for p in pages)[:max_chars]
        kwargs = {
            "messages": [{"role": "user", "content": DESCRIBE_PROMPT.format(excerpt=excerpt)}],
            "temperature": 0.0,
            "max_tokens": self.cfg.get("max_tokens", 4096),
        }
        _, choice = self._complete(kwargs)
        answer, _ = split_reasoning(choice.message.content)
        return clean_title(answer)
