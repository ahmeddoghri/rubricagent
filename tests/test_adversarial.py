"""Tests for the circular benchmark, the blind grounding grader, and the fixes."""

from __future__ import annotations

import random

import pytest

from rubricagent.adversarial import (
    ADVERSARIAL_TRACES,
    GROUNDED_TRACES,
    HOLDOUT_TRACES,
    UNGROUNDED_TRACES,
)
from rubricagent.eval import generate_dataset
from rubricagent.eval_v2 import build_report, run_corpus
from rubricagent.evolve import RubricEvolver
from rubricagent.judge import HeuristicGrader, Trace
from rubricagent.judge_v2 import HeuristicGraderV2, stem
from rubricagent.rubric import starter_rubric


# --- the finding: the grounding grader is blind to real text ---------------

def test_original_grader_scores_zero_on_every_grounded_trace():
    """Not one of six genuinely grounded traces registers at all."""
    grader = HeuristicGrader()
    rubric = starter_rubric(GROUNDED_TRACES[0].task)
    grounding = next(c for c in rubric.criteria if c.name == "grounding")
    for trace in GROUNDED_TRACES:
        score = grader.grade(grounding, Trace(trace.task, trace.response))
        assert score.score == 0.0, trace.response


def test_fixed_grader_distinguishes_grounded_from_ungrounded():
    grader = HeuristicGraderV2()
    rubric = starter_rubric(GROUNDED_TRACES[0].task)
    grounding = next(c for c in rubric.criteria if c.name == "grounding")
    grounded_scores = [
        grader.grade(grounding, Trace(t.task, t.response)).score for t in GROUNDED_TRACES
    ]
    ungrounded_scores = [
        grader.grade(grounding, Trace(t.task, t.response)).score for t in UNGROUNDED_TRACES
    ]
    assert min(grounded_scores) > 0
    assert max(ungrounded_scores) == 0


def test_stem_unifies_grounding_word_forms():
    assert stem("cited") == stem("cite")
    assert stem("citing") == stem("cite")
    assert stem("confirmed") == stem("confirms")


# --- the starter rubric's own grounding criterion, before evolution --------

def test_original_starter_rubric_is_blind_to_grounding_on_realistic_text():
    """AUC after evolution can look fine; this is the number that shows the
    starting criterion never measured what it claims to."""
    result = run_corpus(ADVERSARIAL_TRACES, None)
    assert result["grounding_correlation"] == 0.0


def test_fixed_grader_correlates_grounding_with_the_true_label():
    result = run_corpus(ADVERSARIAL_TRACES, HeuristicGraderV2())
    assert result["grounding_correlation"] > 0.5


def test_fixed_grader_beats_original_on_starting_auc():
    """Before any evolution runs, does the flat rubric already work?"""
    v1 = run_corpus(ADVERSARIAL_TRACES, None)
    v2 = run_corpus(ADVERSARIAL_TRACES, HeuristicGraderV2())
    assert v2["auc_before"] > v1["auc_before"]


# --- held out, evaluated once ------------------------------------------------

def test_holdout_is_disjoint_from_the_tuning_corpus():
    adversarial_texts = {t.response for t in ADVERSARIAL_TRACES}
    holdout_texts = {t.response for t in HOLDOUT_TRACES}
    assert not (adversarial_texts & holdout_texts)


def test_holdout_grounding_correlation_improves():
    v1 = run_corpus(HOLDOUT_TRACES, None)
    v2 = run_corpus(HOLDOUT_TRACES, HeuristicGraderV2())
    assert v2["grounding_correlation"] > v1["grounding_correlation"]
    assert v2["grounding_correlation"] > 0.7


def test_holdout_final_auc_reaches_the_ceiling_with_the_fixed_grader():
    result = run_corpus(HOLDOUT_TRACES, HeuristicGraderV2())
    assert result["auc_after"] == 1.0


# --- evolve() ignoring the grader entirely (a separate bug) -----------------

def test_evolver_uses_the_grader_it_was_given():
    """Regression test: evolve() used to hardcode Judge(rubric), so a custom
    grader passed to RubricEvolver had no effect on evolution at all."""
    corpus = ADVERSARIAL_TRACES
    traces = [Trace(t.task, t.response, id=str(i)) for i, t in enumerate(corpus)]
    labels = [1 if t.grounded else 0 for t in corpus]
    rubric = starter_rubric(corpus[0].task)

    default_evolver = RubricEvolver()
    v2_evolver = RubricEvolver(grader=HeuristicGraderV2())

    _, default_report = default_evolver.evolve(rubric, traces, labels)
    _, v2_report = v2_evolver.evolve(rubric, traces, labels)

    assert default_report.correlations["grounding"] != v2_report.correlations["grounding"]


# --- the false-discovery fix -------------------------------------------------

def test_discovery_does_not_hallucinate_signal_from_pure_noise():
    """With labels independent of text, 'discovered_signal' should not fire."""
    rng = random.Random(7)
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf",
             "hotel", "india", "juliet", "kilo", "lima", "mike", "november",
             "oscar", "papa", "quebec", "romeo", "sierra", "tango"]
    false_discoveries = 0
    trials = 40
    for _ in range(trials):
        traces, labels = [], []
        for i in range(120):
            label = rng.randint(0, 1)
            text = " ".join(rng.sample(words, 6))
            traces.append(Trace("dummy task", text, id=f"t{i}"))
            labels.append(label)
        rubric = starter_rubric("dummy task")
        _, report = RubricEvolver().evolve(rubric, traces, labels)
        if report.discovered:
            false_discoveries += 1
    # A handful in dozens of trials is expected variance; near-zero, not the
    # roughly-1-in-25 rate measured before the fix.
    assert false_discoveries <= 2


def test_discovery_still_finds_a_real_signal():
    """The fix must not make discovery unable to find anything real."""
    traces, labels = generate_dataset(200, seed=0)
    rubric = starter_rubric("Research the launch date and summarize the key risks.")
    _, report = RubricEvolver().evolve(rubric, traces, labels)
    assert report.discovered is not None
    assert report.auc_after > 0.95


# --- the original benchmark is unaffected -----------------------------------

def test_original_benchmark_numbers_still_reproduce():
    traces, labels = generate_dataset(200, seed=0)
    rubric = starter_rubric("Research the launch date and summarize the key risks.")
    _, report = RubricEvolver().evolve(rubric, traces, labels)
    assert report.auc_before == pytest.approx(0.774, abs=0.01)
    assert report.auc_after == pytest.approx(1.0, abs=0.01)


# --- the full report ---------------------------------------------------------

def test_report_is_reproducible():
    assert build_report() == build_report()


def test_report_shows_v1_blind_and_v2_seeing():
    report = build_report()
    assert report["adversarial"]["v1"]["grounding_correlation"] == 0.0
    assert report["adversarial"]["v2"]["grounding_correlation"] > 0.5
