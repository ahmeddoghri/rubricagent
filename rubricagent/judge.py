"""Scoring agent traces against a rubric (LLM-as-judge, with an offline default)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Protocol

from .rubric import Criterion, Rubric, keywords


@dataclass
class Trace:
    """A unit of work to grade: the task and what the agent produced."""

    task: str
    response: str
    id: Optional[str] = None


@dataclass
class CriterionScore:
    name: str
    score: float          # 0..1
    rationale: str


@dataclass
class Judgment:
    trace_id: Optional[str]
    scores: list[CriterionScore]
    total: float          # weighted aggregate, 0..1

    def by_name(self) -> dict[str, float]:
        return {s.name: s.score for s in self.scores}


class Grader(Protocol):
    def grade(self, criterion: Criterion, trace: Trace) -> CriterionScore:
        ...


class HeuristicGrader:
    """Deterministic, offline grader based on evidence-keyword coverage.

    Score for a criterion = fraction of its keywords present in the response,
    with a small floor so structure still registers. Good enough to demonstrate
    the evolution loop; swap in an LLM grader for real judging.
    """

    def grade(self, criterion: Criterion, trace: Trace) -> CriterionScore:
        toks = set(keywords(trace.response))
        kws = criterion.keywords or keywords(criterion.description)
        if not kws:
            return CriterionScore(criterion.name, 0.5, "no evidence keywords")
        hits = [k for k in kws if k in toks]
        score = len(hits) / len(kws)
        return CriterionScore(
            criterion.name,
            round(score, 4),
            f"matched {len(hits)}/{len(kws)} evidence terms: {hits[:5]}",
        )


class Judge:
    """Applies a rubric to a trace and produces a weighted judgment."""

    def __init__(self, rubric: Rubric, grader: Optional[Grader] = None) -> None:
        self.rubric = rubric
        self.grader = grader or HeuristicGrader()

    def judge(self, trace: Trace) -> Judgment:
        w = self.rubric.normalized_weights()
        scores = [self.grader.grade(c, trace) for c in self.rubric.criteria]
        total = sum(s.score * w[s.name] for s in scores)
        return Judgment(trace.id, scores, round(total, 4))

    def judge_many(self, traces: list[Trace]) -> list[Judgment]:
        return [self.judge(t) for t in traces]
