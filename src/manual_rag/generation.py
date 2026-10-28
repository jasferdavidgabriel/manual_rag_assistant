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
   If chunks come from different manuals, only combine them when they are clearly about the same product.
3. If the answer involves steps, list them clearly and numbered.
4. If the information is not in the provided context, say "The manual does not cover this." and stop.

Format your response as:
**Answer:** [Your answer here]

**Location in Manual:**
- Manual: [manual name]
- Page: [X]
- Section: [Section name]
- Lines: [start_line - end_line]

**Steps (if applicable):**
1. Step one
2. Step two

Manual Context:
{context}

Question: {question}"""

_THINK_BLOCK = re.compile(r"<think>(.*?)</think>", re.DOTALL)


@dataclass
class Answer:
    """Final model output, split into the reasoning trace and the answer."""

    answer: str
    reasoning: str | None
    hits: list[Hit]


def build_context(hits: Sequence[Hit]) -> str:
    """Format retrieved chunks with their location headers for the prompt."""
    return "\n\n".join(
        f"[Chunk {i} | Manual: {h.source} | Page: {h.page} | Section: {h.section} | "
        f"Lines: {h.start_line}-{h.end_line}]\n{h.text}"
        for i, h in enumerate(hits, start=1)
    )


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
    """OpenAI compatible chat client (NVIDIA NIM by default)."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        from openai import OpenAI

        key_name = cfg.get("api_key_env", "NVIDIA_API_KEY")
        api_key = os.environ.get(key_name)
        if not api_key:
            raise EnvironmentError(f"Set {key_name} in your environment or .env file.")
        self.cfg = cfg
        self.client = OpenAI(base_url=cfg["base_url"], api_key=api_key)

    def generate(self, question: str, hits: Sequence[Hit]) -> Answer:
        """Call the LLM with the retrieved context and return a parsed answer."""
        extra_body = None
        if self.cfg.get("enable_thinking"):
            extra_body = {
                "chat_template_kwargs": {
                    "thinking": True,
                    "reasoning_effort": self.cfg.get("reasoning_effort", "high"),
                }
            }

        response = self.client.chat.completions.create(
            model=self.cfg["model"],
            messages=[{"role": "user", "content": build_prompt(question, hits)}],
            temperature=self.cfg.get("temperature", 0.3),
            top_p=self.cfg.get("top_p", 0.95),
            max_tokens=self.cfg.get("max_tokens", 4096),
            extra_body=extra_body,
        )
        choice = response.choices[0]
        if getattr(choice, "finish_reason", None) == "length":
            logger.warning("Answer was cut off at max_tokens. Raise llm.max_tokens in the config.")

        message = choice.message
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        answer, reasoning = split_reasoning(message.content, reasoning)
        return Answer(answer=answer, reasoning=reasoning, hits=list(hits))
