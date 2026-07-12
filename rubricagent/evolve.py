"""Self-evolving rubrics.

Given a batch of graded traces with outcome labels (did the agent actually
succeed?), the evolver figures out *which criteria matter* and rewrites the
rubric to match reality:

1. **Reweight** each criterion by how well its score correlates with success.
2. **Prune** criteria that carry no signal (correlation below a floor).
3. **Discover** a new criterion from evidence terms that are common in
   successful traces but rare in failed ones.

The improvement is measured with a proxy metric: the AUC with which the
aggregate rubric score separates good runs from bad. A better rubric is a
better *proxy* for capability, so you can grade cheaply at scale.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .judge import Judge, Trace
from .metrics import auc, pearson
from .rubric import Criterion, Rubric, keywords


@dataclass
class EvolutionReport:
    auc_before: float
    auc_after: float
    correlations: dict[str, float]
    pruned: list[str]
    discovered: Optional[str]

    @property
    def improvement(self) -> float:
        return self.auc_after - self.auc_before


class RubricEvolver:
    def __init__(self, prune_below: float = 0.05, discover: bool = True) -> None:
        self.prune_below = prune_below
        self.discover = discover

    def evolve(
        self, rubric: Rubric, traces: list[Trace], labels: list[int]
    ) -> tuple[Rubric, EvolutionReport]:
        judge = Judge(rubric)
        judgments = judge.judge_many(traces)
        auc_before = auc([j.total for j in judgments], labels)

        # 1. correlation of each criterion's score with success
        corrs: dict[str, float] = {}
        for c in rubric.criteria:
            series = [j.by_name()[c.name] for j in judgments]
            corrs[c.name] = round(pearson(series, [float(x) for x in labels]), 4)

        # 2. reweight (positive correlation only) and 3. prune the dead weight
        new_criteria: list[Criterion] = []
        pruned: list[str] = []
        for c in rubric.criteria:
            corr = corrs[c.name]
            if corr < self.prune_below:
                pruned.append(c.name)
                continue
            new_criteria.append(Criterion(c.name, c.description, weight=corr,
                                          keywords=c.keywords))

        # 4. discover a criterion from discriminative evidence terms
        discovered = None
        if self.discover:
            disc_kw = self._discriminative_terms(traces, labels)
            if disc_kw:
                discovered = "discovered_signal"
                new_criteria.append(
                    Criterion(
                        discovered,
                        "Evidence terms empirically predictive of success: "
                        + ", ".join(disc_kw),
                        weight=max(corrs.values() or [0.5]),
                        keywords=disc_kw,
                    )
                )

        if not new_criteria:  # never return an empty rubric
            new_criteria = list(rubric.criteria)
            pruned = []

        new_rubric = Rubric(new_criteria)
        after_judgments = Judge(new_rubric).judge_many(traces)
        auc_after = auc([j.total for j in after_judgments], labels)

        return new_rubric, EvolutionReport(
            auc_before=round(auc_before, 4),
            auc_after=round(auc_after, 4),
            correlations=corrs,
            pruned=pruned,
            discovered=discovered,
        )

    @staticmethod
    def _discriminative_terms(traces: list[Trace], labels: list[int],
                              top: int = 6) -> list[str]:
        pos, neg = Counter(), Counter()
        npos = sum(1 for y in labels if y == 1) or 1
        nneg = sum(1 for y in labels if y == 0) or 1
        for t, y in zip(traces, labels):
            seen = set(k for k in keywords(t.response) if len(k) > 3)
            (pos if y == 1 else neg).update(seen)
        scored = []
        for term, pc in pos.items():
            p = pc / npos
            n = neg[term] / nneg
            if p - n > 0.25:  # clearly more common in successes
                scored.append((p - n, term))
        scored.sort(reverse=True)
        return [term for _, term in scored[:top]]
