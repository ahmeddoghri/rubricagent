"""Rubrics: weighted criteria for scoring agent behaviour."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

_WORD = re.compile(r"[a-z0-9]+")


def keywords(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@dataclass
class Criterion:
    """One thing we grade for.

    ``keywords`` are the evidence tokens a heuristic judge looks for in a trace;
    an LLM judge uses ``description`` directly and can ignore them.
    """

    name: str
    description: str
    weight: float = 1.0
    keywords: list[str] = field(default_factory=list)


@dataclass
class Rubric:
    criteria: list[Criterion]

    def normalized_weights(self) -> dict[str, float]:
        total = sum(max(0.0, c.weight) for c in self.criteria) or 1.0
        return {c.name: max(0.0, c.weight) / total for c in self.criteria}

    def names(self) -> list[str]:
        return [c.name for c in self.criteria]

    def to_markdown(self) -> str:
        w = self.normalized_weights()
        rows = [f"| {c.name} | {w[c.name]:.0%} | {c.description} |" for c in self.criteria]
        return "| Criterion | Weight | Description |\n|---|---|---|\n" + "\n".join(rows)


def starter_rubric(task: str) -> Rubric:
    """A reasonable default rubric, lightly specialised to the task text.

    In production you'd generate this with an LLM; offline we seed sensible,
    general skill-use criteria and attach task keywords to the relevance one.
    """
    task_kw = [k for k in keywords(task) if len(k) > 3][:8]
    return Rubric(
        [
            Criterion("relevance", "Response addresses the actual task asked.",
                      1.0, task_kw),
            Criterion("correctness", "The answer is factually/logically correct.",
                      1.0, ["because", "therefore", "correct", "answer", "result"]),
            Criterion("completeness", "All parts of the task are covered.",
                      1.0, ["first", "second", "then", "finally", "step", "also"]),
            Criterion("grounding", "Claims are supported by evidence or tools.",
                      1.0, ["source", "cite", "according", "tool", "search", "found"]),
            Criterion("clarity", "Response is clear and well-structured.",
                      1.0, ["summary", "overall", "in short"]),
        ]
    )
