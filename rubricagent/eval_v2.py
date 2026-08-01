"""Does the grader work on traces it wasn't built to detect?

The bundled benchmark's synthetic traces are built by concatenating marker
strings, and each marker string contains most of the literal keywords its
matching criterion searches for. The 0.774 -> 1.000 AUC climb measures
whether keyword-overlap scoring can detect data built from keyword overlap.

This module runs the same rubric-evolution pipeline against
:mod:`rubricagent.adversarial`, traces written in ordinary language with
genuine grounding or the lack of it, none built by reading the grader's
keyword lists. It reports the ``grounding`` criterion's correlation with the
true label directly, because that number is the honest measure of whether
the grader can see what it claims to grade, and the AUC after evolution
alone can hide a blind starting criterion behind the separate,
grader-independent "discover new criterion" mechanism.

    python -m rubricagent.eval_v2
"""
from __future__ import annotations

import argparse
import json
from typing import Dict, List, Optional, Sequence

from .adversarial import ADVERSARIAL_TRACES, HOLDOUT_TRACES, LabeledTrace
from .evolve import RubricEvolver
from .judge import Grader, Trace
from .judge_v2 import HeuristicGraderV2
from .rubric import starter_rubric


def _to_traces_and_labels(corpus: Sequence[LabeledTrace]):
    traces = [Trace(t.task, t.response, id=str(i)) for i, t in enumerate(corpus)]
    labels = [1 if t.grounded else 0 for t in corpus]
    return traces, labels


def run_corpus(corpus: Sequence[LabeledTrace], grader: Optional[Grader] = None) -> Dict:
    """Evolve a fresh starter rubric against one labeled corpus."""
    traces, labels = _to_traces_and_labels(corpus)
    rubric = starter_rubric(corpus[0].task)
    evolver = RubricEvolver(grader=grader)
    _, report = evolver.evolve(rubric, traces, labels)
    return {
        "auc_before": report.auc_before,
        "auc_after": report.auc_after,
        "improvement": round(report.improvement, 4),
        "grounding_correlation": report.correlations.get("grounding", 0.0),
        "pruned": report.pruned,
        "discovered": report.discovered,
    }


def build_report() -> Dict:
    return {
        "adversarial": {
            "v1": run_corpus(ADVERSARIAL_TRACES, None),
            "v2": run_corpus(ADVERSARIAL_TRACES, HeuristicGraderV2()),
        },
        "holdout": {
            "v1": run_corpus(HOLDOUT_TRACES, None),
            "v2": run_corpus(HOLDOUT_TRACES, HeuristicGraderV2()),
        },
    }


def format_report(report: Dict) -> str:
    lines = [
        "Does the grounding criterion actually see grounding?",
        "=" * 70,
        f"{'corpus / grader':<20}{'AUC before':>12}{'AUC after':>12}{'grounding corr':>16}",
        "-" * 70,
    ]
    for corpus_name in ("adversarial", "holdout"):
        for grader_name in ("v1", "v2"):
            row = report[corpus_name][grader_name]
            label = f"{corpus_name} / {grader_name}"
            lines.append(
                f"{label:<20}{row['auc_before']:>12.3f}{row['auc_after']:>12.3f}"
                f"{row['grounding_correlation']:>16.3f}"
            )
        lines.append("")
    lines.append(
        "grounding corr = the starter rubric's own grounding criterion's"
    )
    lines.append(
        "correlation with the true label, before evolution touches anything."
    )
    lines.append(
        "AUC after can look fine even when this is 0, because 'discover new"
    )
    lines.append(
        "criterion' mines raw trace text directly and does not depend on"
    )
    lines.append("the grader at all.")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", default=None)
    args = parser.parse_args(argv)

    report = build_report()
    print(format_report(report))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
        print(f"\nWrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
