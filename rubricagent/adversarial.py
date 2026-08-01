"""Traces that don't hand the grader its own keyword list.

The bundled benchmark synthesizes traces by literally concatenating marker
strings, and every marker string was written to contain most of the exact
keywords its matching criterion searches for:

    grounding keywords: ["source", "cite", "according", "tool", "search", "found"]
    _GROUND marker:     "according to the source the tool search found evidence"

Five of six keywords appear verbatim. The benchmark's 0.774 -> 1.000 AUC
climb measures whether keyword-overlap scoring can detect data built from
keyword overlap. It says nothing about whether the grader can read a real
transcript, where a grounded answer says "the SEC 10-K filing lists a March
launch" rather than announcing its own groundedness with the word "source".

Run the original ``HeuristicGrader`` against genuinely grounded and
genuinely ungrounded traces, phrased the way a transcript actually reads,
and it scores every single one at 0.00 on the grounding criterion. Not
because the traces are ambiguous; a person reading them can tell in one
pass which ones cite something and which ones are a shrug. The grader's
keyword list is six words long and none of the six happen to appear.

Each trace here is labeled on two axes: whether it is genuinely grounded
(cites something checkable) and whether the underlying task was actually
completed correctly. Neither label was chosen by reading the grader's
keyword lists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LabeledTrace:
    task: str
    response: str
    grounded: bool
    note: str = ""


TASK = "Research the product launch date and summarize the key risks."

# --- genuinely grounded: cites something checkable, in ordinary language ---
GROUNDED_TRACES: List[LabeledTrace] = [
    LabeledTrace(
        TASK,
        "The SEC 10-K filing lists a March 2026 launch, and analysts flagged "
        "supply chain delays as the top risk.",
        True,
    ),
    LabeledTrace(
        TASK,
        "I checked the vendor's press release: the product ships in Q2, and "
        "the main risk is a chip shortage per their own disclosure.",
        True,
    ),
    LabeledTrace(
        TASK,
        "Per the company's investor call transcript, launch is confirmed for "
        "spring; risks include regulatory approval delays.",
        True,
    ),
    LabeledTrace(
        TASK,
        "Their engineering blog post from last week states a Q3 release window, "
        "citing a pending certification as the main open risk.",
        True,
    ),
    LabeledTrace(
        TASK,
        "The prospectus filed with regulators names June as the target date "
        "and lists component shortages as the primary risk factor.",
        True,
    ),
    LabeledTrace(
        TASK,
        "Based on the earnings call recording, management confirmed an autumn "
        "launch, with the CFO noting currency exposure as the key risk.",
        True,
    ),
]

# --- genuinely ungrounded: a guess, in ordinary language --------------------
UNGROUNDED_TRACES: List[LabeledTrace] = [
    LabeledTrace(
        TASK,
        "I think it probably launches sometime next year, and the risks are "
        "probably fine honestly.",
        False,
    ),
    LabeledTrace(TASK, "My best guess is early next year, no major risks come to mind.", False),
    LabeledTrace(TASK, "Launch is likely soon. Risks are probably manageable.", False),
    LabeledTrace(
        TASK,
        "Hard to say exactly when, but it feels like it should ship eventually. "
        "Risk-wise, nothing jumps out.",
        False,
    ),
    LabeledTrace(TASK, "Sometime this year seems reasonable. Shouldn't be too risky.", False),
    LabeledTrace(
        TASK,
        "I'd guess mid-year. I don't have a strong sense of what could go wrong.",
        False,
    ),
]

ADVERSARIAL_TRACES: List[LabeledTrace] = GROUNDED_TRACES + UNGROUNDED_TRACES


# Written after HeuristicGraderV2's keyword lists were frozen against the
# corpus above. Evaluated exactly once.
HOLDOUT_TRACES: List[LabeledTrace] = [
    LabeledTrace(
        TASK,
        "The company's 8-K disclosure names a February launch and flags a "
        "pending patent dispute as the leading risk.",
        True,
    ),
    LabeledTrace(
        TASK,
        "According to the shipping manifest data I pulled, units start moving "
        "in April; the noted risk is a customs delay at the port.",
        True,
    ),
    LabeledTrace(
        TASK,
        "Referencing the analyst report from last quarter, release is set for "
        "late summer, with talent retention cited as the main concern.",
        True,
    ),
    LabeledTrace(TASK, "Whenever it's ready, I suppose. Can't think of any real risks.", False),
    LabeledTrace(TASK, "Probably next quarter-ish. Nothing especially risky that I can tell.", False),
    LabeledTrace(TASK, "Not totally sure on timing. Risk feels low to me overall.", False),
]
