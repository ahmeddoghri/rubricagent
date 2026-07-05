"""RubricAgent — self-evolving rubrics for evaluating LLM-agent skill-use.

>>> from rubricagent import Trace, Judge, starter_rubric
>>> r = starter_rubric("summarize the risks")
>>> j = Judge(r).judge(Trace("summarize the risks",
...     "According to the source, the main risk is delay; therefore plan early."))
>>> 0 <= j.total <= 1
True
"""
from .rubric import Criterion, Rubric, starter_rubric
from .judge import Trace, CriterionScore, Judgment, Judge, HeuristicGrader
from .evolve import RubricEvolver, EvolutionReport
from .report import scorecard
from .metrics import auc, pearson

__all__ = [
    "Criterion", "Rubric", "starter_rubric",
    "Trace", "CriterionScore", "Judgment", "Judge", "HeuristicGrader",
    "RubricEvolver", "EvolutionReport",
    "scorecard", "auc", "pearson",
]
__version__ = "0.1.0"
