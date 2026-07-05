from rubricagent import Trace, Judge, starter_rubric, RubricEvolver, scorecard
from rubricagent.eval import generate_dataset
from rubricagent.metrics import auc, pearson


def test_judge_returns_bounded_total():
    r = starter_rubric("summarize the launch risks")
    t = Trace("summarize the launch risks",
              "According to the source the result is correct; therefore, first then finally done.")
    j = Judge(r).judge(t)
    assert 0.0 <= j.total <= 1.0
    assert len(j.scores) == len(r.criteria)


def test_strong_trace_beats_weak_trace():
    r = starter_rubric("what is the answer")
    strong = Trace("q", "According to the source and tool I found the result; "
                        "therefore the answer is correct. First, then, finally.")
    weak = Trace("q", "okay sure thanks")
    assert Judge(r).judge(strong).total > Judge(r).judge(weak).total


def test_metrics_sane():
    assert pearson([1, 2, 3], [1, 2, 3]) > 0.99
    assert pearson([1, 2, 3], [3, 2, 1]) < -0.99
    assert auc([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 1.0
    assert auc([0.1, 0.2], [0, 1]) == 1.0


def test_evolution_improves_separation():
    traces, labels = generate_dataset(200, seed=0)
    rubric = starter_rubric("research and summarize")
    evolved, report = RubricEvolver().evolve(rubric, traces, labels)
    # evolving the rubric makes it a strictly better capability proxy
    assert report.auc_after > report.auc_before
    assert report.auc_after > 0.9
    # the one true signal survives; the misleading clarity criterion is pruned
    assert report.correlations["grounding"] > 0.5
    assert "grounding" not in report.pruned
    assert "clarity" in report.pruned


def test_evolution_never_empties_rubric():
    traces = [Trace("q", "nothing useful here") for _ in range(10)]
    labels = [0, 1] * 5
    rubric = starter_rubric("q")
    evolved, _ = RubricEvolver().evolve(rubric, traces, labels)
    assert len(evolved.criteria) >= 1


def test_scorecard_renders():
    r = starter_rubric("q")
    j = Judge(r).judge(Trace("q", "according to the source, therefore correct"))
    md = scorecard(j)
    assert "Scorecard" in md and "Total" in md
