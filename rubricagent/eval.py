"""PACE-style evaluation: does an evolved rubric become a better capability proxy?

We synthesize agent traces whose *true* success depends on reasoning and
grounding, then bury that signal under a misleading "clarity" criterion that is
actually more common in failures. A flat, equal-weight rubric is a mediocre
proxy. After one evolution pass the rubric should separate good from bad runs
markedly better (higher AUC).

    python -m rubricagent.eval --n 200
"""
from __future__ import annotations

import argparse
import random

from .evolve import RubricEvolver
from .judge import Trace
from .rubric import starter_rubric

TASK = "Research the launch date and summarize the key risks."

_CORRECT = "because therefore the answer is correct and the result holds"
_GROUND = "according to the source the tool search found evidence"
_COMPLETE = "first then finally every step is covered also"
_CLARITY = "in summary overall this is clear concise and structured"
_FILLER = "okay sure interesting noted thanks"


def generate_dataset(n: int = 200, seed: int = 0):
    """True success is driven by *grounding*; everything else is noise.

    ``correctness`` and ``completeness`` markers appear at random regardless of
    outcome (distractors), ``clarity`` is mildly anti-correlated, and
    ``relevance`` is constant. A flat equal-weight rubric therefore buries the
    one real signal under four useless criteria — exactly the situation the
    evolver is meant to fix.
    """
    rng = random.Random(seed)
    traces, labels = [], []
    for i in range(n):
        label = 1 if rng.random() < 0.5 else 0
        parts = [TASK]                       # relevance: constant, zero signal
        if label == 1:
            parts.append(_GROUND)            # the only true predictor of success
        if rng.random() < 0.5:               # correctness: pure noise
            parts.append(_CORRECT)
        if rng.random() < 0.5:               # completeness: pure noise
            parts.append(_COMPLETE)
        if rng.random() < (0.6 if label == 0 else 0.35):  # clarity: mild anti-signal
            parts.append(_CLARITY)
        if rng.random() < 0.5:
            parts.append(_FILLER)
        rng.shuffle(parts)
        traces.append(Trace(TASK, " ".join(parts), id=f"t{i}"))
        labels.append(label)
    return traces, labels


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    traces, labels = generate_dataset(args.n, args.seed)
    rubric = starter_rubric(TASK)
    evolved, report = RubricEvolver().evolve(rubric, traces, labels)

    print(f"traces={args.n}")
    print(f"AUC (flat rubric)    = {report.auc_before:.3f}")
    print(f"AUC (evolved rubric) = {report.auc_after:.3f}")
    print(f"improvement          = {report.improvement:+.3f}")
    print("\ncriterion correlations with success:")
    for name, corr in sorted(report.correlations.items(), key=lambda kv: -kv[1]):
        print(f"  {name:14s} {corr:+.3f}")
    print("pruned :", report.pruned or "none")
    print("discovered:", report.discovered or "none")


if __name__ == "__main__":
    main()
