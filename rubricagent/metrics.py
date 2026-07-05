"""Small, dependency-free statistics used by the evolver."""
from __future__ import annotations

import math
from typing import Sequence


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Pearson correlation; 0.0 if a series has no variance."""
    n = len(xs)
    if n == 0:
        return 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def auc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """ROC-AUC via the Mann-Whitney rank-sum identity. 0.5 = chance.

    Measures how well a score separates positive (label=1) from negative traces.
    """
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return 0.5
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n:
                wins += 1.0
            elif p == n:
                wins += 0.5
    return wins / (len(pos) * len(neg))
