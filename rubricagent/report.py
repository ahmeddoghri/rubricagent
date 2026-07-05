"""Human-readable scorecards from judgments."""
from __future__ import annotations

from .judge import Judgment


def scorecard(judgment: Judgment) -> str:
    lines = [
        f"### Scorecard {judgment.trace_id or ''}".rstrip(),
        f"**Total: {judgment.total:.0%}**",
        "",
        "| Criterion | Score | Rationale |",
        "|---|---|---|",
    ]
    for s in judgment.scores:
        bar = "█" * round(s.score * 10) + "░" * (10 - round(s.score * 10))
        lines.append(f"| {s.name} | `{bar}` {s.score:.0%} | {s.rationale} |")
    return "\n".join(lines)
