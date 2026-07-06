"""RubricAgent — self-evolving rubrics for evaluating LLM-agent skill-use.

>>> from rubricagent import Trace, Judge, starter_rubric
>>> r = starter_rubric("summarize the risks")
>>> j = Judge(r).judge(Trace("summarize the risks",
...     "According to the source, the main risk is delay; therefore plan early."))
>>> 0 <= j.total <= 1
True
"""
from .evolve import EvolutionReport, RubricEvolver
from .judge import CriterionScore, HeuristicGrader, Judge, Judgment, Trace
from .metrics import auc, pearson
from .report import scorecard
from .rubric import Criterion, Rubric, starter_rubric

__all__ = [
    "Criterion", "Rubric", "starter_rubric",
    "Trace", "CriterionScore", "Judgment", "Judge", "HeuristicGrader",
    "RubricEvolver", "EvolutionReport",
    "scorecard", "auc", "pearson",
]
__version__ = "0.1.0"
