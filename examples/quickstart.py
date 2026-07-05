"""Grade a trace, then evolve the rubric on a labeled batch.
Run: python examples/quickstart.py
"""
from rubricagent import Trace, Judge, starter_rubric, RubricEvolver, scorecard
from rubricagent.eval import generate_dataset

# 1. Grade a single agent trace against a starter rubric
rubric = starter_rubric("Research the launch date and summarize the risks.")
trace = Trace(
    task="Research the launch date and summarize the risks.",
    response=("According to the source and the tool, I found the launch is Friday. "
              "Because the vendor is late, therefore the main risk is slippage. "
              "First, then, finally: mitigation is staged rollout."),
)
print(scorecard(Judge(rubric).judge(trace)))

# 2. Evolve the rubric from a labeled batch (did the run actually succeed?)
traces, labels = generate_dataset(200, seed=0)
evolved, report = RubricEvolver().evolve(rubric, traces, labels)
print(f"\nRubric quality as a capability proxy (AUC): "
      f"{report.auc_before:.3f} -> {report.auc_after:.3f} ({report.improvement:+.3f})")
print("kept criteria:", evolved.names())
print("pruned:", report.pruned)
