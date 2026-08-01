"""A heuristic grader with keyword lists sized for real transcripts.

``HeuristicGrader``'s per-criterion keyword lists are 3-6 words each, and
those words were chosen to match the benchmark's own synthetic marker
strings rather than how a real grounded, correct, or complete response
actually reads. Real grounded text says "the SEC filing shows" or "per the
vendor's press release"; none of that shares a token with
``["source", "cite", "according", "tool", "search", "found"]`` unless the
writer happened to use "according" specifically. Run the original grader
against six genuinely grounded and six genuinely ungrounded traces, written
in ordinary language, and it scores every single one 0.00 on grounding: the
keyword list is too narrow to fire at all.

``HeuristicGraderV2`` widens each criterion's evidence vocabulary to the
kind of phrasing an evidence-citing, complete, correct, clear answer
actually uses, written from general knowledge of how those four qualities
show up in text, before looking at the adversarial corpus this module is
tested against. Basic stemming closes the gap between "cited" and "cite",
the same fix this pattern has needed in every keyword-based grader in this
series so far.
"""

from __future__ import annotations

import re

from .judge import CriterionScore, Trace
from .rubric import Criterion

_TOKEN = re.compile(r"[a-z0-9]+")


def stem(word: str) -> str:
    """Crude suffix stripping, consistent with the fix used elsewhere in this
    portfolio for the same class of exact-match brittleness."""
    for suffix in ("ically", "ing", "ies", "ers", "er", "ed", "es", "s"):
        if len(word) > len(suffix) + 2 and word.endswith(suffix):
            word = word[: -len(suffix)]
            break
    if len(word) > 3 and word.endswith("e"):
        word = word[:-1]
    return word


def _stemmed_tokens(text: str) -> set:
    return {stem(t) for t in _TOKEN.findall(text.lower())}


# Wider evidence vocabulary per criterion, written from general knowledge of
# how grounding, correctness, completeness, and clarity actually show up in
# text, not from reading the benchmark's synthetic markers.
EXPANDED_KEYWORDS = {
    "grounding": [
        "source", "cite", "cited", "citing", "according", "tool", "search",
        "found", "filing", "disclosure", "report", "transcript", "record",
        "document", "data", "evidence", "reference", "referencing", "per",
        "confirmed", "confirms", "states", "shows", "recording", "manifest",
        "press", "release", "prospectus", "regulator", "regulatory",
    ],
    "correctness": [
        "because", "therefore", "correct", "answer", "result", "accurate",
        "verified", "checked", "confirmed", "precise", "exact", "matches",
        "consistent", "validated",
    ],
    "completeness": [
        "first", "second", "then", "finally", "step", "also", "additionally",
        "furthermore", "moreover", "both", "covers", "covering", "includes",
        "including", "overall", "entire", "full", "comprehensive",
    ],
    "clarity": [
        "summary", "overall", "clear", "concise", "structured", "organized",
        "explains", "explanation", "straightforward", "simply", "briefly",
    ],
}


class HeuristicGraderV2:
    """Keyword-overlap grading with a wider, stemmed vocabulary."""

    def grade(self, criterion: Criterion, trace: Trace) -> CriterionScore:
        response_tokens = _stemmed_tokens(trace.response)

        keywords = EXPANDED_KEYWORDS.get(criterion.name)
        if keywords is None:
            # Criteria without a curated list (relevance, a discovered
            # criterion) fall back to whatever the criterion itself carries,
            # stemmed for consistency.
            raw = criterion.keywords or _TOKEN.findall(criterion.description.lower())
            keywords = list(raw)

        stemmed_keywords = {stem(k) for k in keywords}
        if not stemmed_keywords:
            return CriterionScore(criterion.name, 0.5, "no evidence keywords")

        hits = stemmed_keywords & response_tokens
        score = len(hits) / len(stemmed_keywords)
        return CriterionScore(
            criterion.name,
            round(min(1.0, score), 4),
            f"matched {len(hits)}/{len(stemmed_keywords)} evidence terms: "
            f"{sorted(hits)[:5]}",
        )
