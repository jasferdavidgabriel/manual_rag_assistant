"""An agent that answers questions by deciding for itself when to search.

The regular pipeline always does the same thing: search once, then answer.
Here the language model is given tools and works in a loop: it can search,
look at what came back, search again with different words or in another
manual, and answer only when it has enough. Your code runs the tools; the
model only asks for them.

Tools:
    search_manual(query, manual=None)  the top sections for a query
    list_manuals()                     which manuals are indexed
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You answer questions about appliance instruction manuals.

Rules:
- Always use search_manual before answering. Never answer from memory.
- If the results don't answer the question, search again with different words.
- For questions about more than one product, search each manual separately.
- Use list_manuals if you need to know which manuals exist.
- Answer only from the search results. If they don't cover the question, say
  the manual doesn't cover it.
- Cite every fact as (manual, page N, section).
- Keep the answer short and practical; use numbered steps for procedures."""

NUDGE = ("Don't describe what you will do. Call search_manual now: answers must come "
         "from the manuals, not from memory.")

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_manual",
            "description": "Search the indexed manuals and return the most relevant sections, "
                           "each with its manual, page, section and line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "What to search for."},
                    "manual": {"type": "string",
                               "description": "Optional manual file name to search in, "
                                              "e.g. ge_microwave_jvm3160.pdf."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_manuals",
            "description": "List the indexed manuals with their product names.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


@dataclass
class Step:
    """One tool call the agent made."""

    tool: str
    arguments: dict
    result_preview: str


@dataclass
class AgentResult:
    answer: str
    model: str
    steps: list[Step] = field(default_factory=list)
    hits: list = field(default_factory=list)  # every section the agent saw, for citations


def format_hits(hits, preview_chars: int = 800) -> str:
    """Search results as text the model can read and cite."""
    if not hits:
        return "No matching sections found."
    blocks = []
    for i, h in enumerate(hits, start=1):
        title = f" ({h.manual_title})" if h.manual_title else ""
        text = h.text if len(h.text) <= preview_chars else h.text[:preview_chars] + "..."
        blocks.append(f"[{i}] manual: {h.source}{title} | page {h.page} | section: {h.section} | "
                      f"lines {h.start_line}-{h.end_line}\n{text}")
    return "\n\n".join(blocks)


class Agent:
    """Tool calling loop over the existing retriever and LLM settings."""

    def __init__(self, cfg, retriever=None, client=None,
                 manuals_fn: Callable[[], dict[str, str]] | None = None):
        self.cfg = cfg
        self.llm = cfg.section("llm") or {}
        settings = cfg.section("agent") or {}
        self.max_steps = int(settings.get("max_steps", 6))
        self.top_k = int(settings.get("top_k", 5))
        self._retriever = retriever
        self._client = client
        self._manuals_fn = manuals_fn

    # ---------- dependencies, created only when first needed ----------

    @property
    def retriever(self):
        if self._retriever is None:
            from . import pipeline
            self._retriever = pipeline.build_retriever(self.cfg)
        return self._retriever

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            key_env = self.llm.get("api_key_env", "NVIDIA_API_KEY")
            if not os.environ.get(key_env):
                raise RuntimeError(f"Set {key_env} to use the agent.")
            self._client = OpenAI(base_url=self.llm.get("base_url"), api_key=os.environ[key_env],
                                  timeout=self.llm.get("timeout", 90),
                                  max_retries=self.llm.get("max_retries", 0))
        return self._client

    def manuals(self) -> dict[str, str]:
        if self._manuals_fn is not None:
            return self._manuals_fn()
        from . import pipeline
        return {name: pipeline.manual_title(self.cfg, name) or name
                for name in pipeline.manuals(self.cfg)}

    # ---------- tools ----------

    def run_tool(self, name: str, arguments: dict, default_manual: str | None,
                 seen: list) -> str:
        if name == "search_manual":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return "Error: search_manual needs a query."
            manual = arguments.get("manual") or default_manual
            hits = self.retriever.retrieve(query, top_k=self.top_k, manual=manual)
            seen.extend(hits)
            return format_hits(hits)
        if name == "list_manuals":
            items = self.manuals()
            return "\n".join(f"{n}: {t}" for n, t in items.items()) or "No manuals are indexed."
        return f"Error: unknown tool {name}. Use search_manual or list_manuals."

    # ---------- the loop ----------

    def _complete(self, messages: list, tools: list | None):
        """One model call, trying the fallback models in order if one fails."""
        models = [self.llm.get("model"), *self.llm.get("fallback_models", [])]
        extra = ({"chat_template_kwargs": self.llm["chat_template_kwargs"]}
                 if self.llm.get("chat_template_kwargs") else {})
        last_error = None
        for model in [m for m in models if m]:
            try:
                kwargs = dict(model=model, messages=messages, temperature=self.llm.get("temperature", 0.3),
                              max_tokens=self.llm.get("max_tokens", 4096), extra_body=extra)
                if tools is not None:
                    kwargs.update(tools=tools, tool_choice="auto")
                return self.client.chat.completions.create(**kwargs), model
            except Exception as exc:  # try the next model
                logger.warning("Model %s failed: %s", model, exc)
                last_error = exc
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def run(self, question: str, manual: str | None = None) -> AgentResult:
        """Answer a question, letting the model decide which tools to use.

        manual, if given, is the default manual for searches that don't name one.
        """
        intro = question if not manual else f"{question}\n\n(The user's current manual is {manual}.)"
        try:  # tell the model up front which manuals exist, saving a tool round
            available = self.manuals()
        except Exception as exc:
            logger.warning("Could not list manuals: %s", exc)
            available = {}
        if available:
            intro += "\n\nAvailable manuals:\n" + "\n".join(f"- {n}: {t}" for n, t in available.items())
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT},
                                          {"role": "user", "content": intro}]
        steps: list[Step] = []
        seen: list = []
        model = ""
        nudged = False

        for _ in range(self.max_steps):
            response, model = self._complete(messages, TOOLS)
            msg = response.choices[0].message
            calls = msg.tool_calls or []
            if not calls and not steps and not nudged:
                # It answered (or said what it would do) without searching: push it once
                messages.append({"role": "assistant", "content": msg.content or ""})
                messages.append({"role": "user", "content": NUDGE})
                nudged = True
                continue
            if not calls:
                return AgentResult(answer=(msg.content or "").strip(), model=model,
                                   steps=steps, hits=_unique(seen))

            messages.append({
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [{"id": c.id, "type": "function",
                                "function": {"name": c.function.name,
                                             "arguments": c.function.arguments or "{}"}}
                               for c in calls],
            })
            for call in calls:
                try:
                    arguments = json.loads(call.function.arguments or "{}")
                    if not isinstance(arguments, dict):
                        raise ValueError("arguments must be a JSON object")
                    result = self.run_tool(call.function.name, arguments, manual, seen)
                except (ValueError, json.JSONDecodeError) as exc:
                    arguments, result = {}, f"Error: could not read the tool arguments ({exc})."
                logger.info("Tool %s(%s)", call.function.name, arguments)
                steps.append(Step(call.function.name, arguments, result[:200]))
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

        # Out of steps: ask for a final answer from what it has, with no more tools
        messages.append({"role": "user", "content": "Answer now using only the results above."})
        response, model = self._complete(messages, None)
        answer = (response.choices[0].message.content or "").strip()
        return AgentResult(answer=answer, model=model, steps=steps, hits=_unique(seen))


def _unique(hits: list) -> list:
    """Drop repeated sections, keeping the first time each was seen."""
    out, keys = [], set()
    for h in hits:
        key = (h.source, h.page, h.section, h.start_line)
        if key not in keys:
            keys.add(key)
            out.append(h)
    return out
