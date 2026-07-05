# 📊 RubricAgent

**Self-evolving rubrics for evaluating LLM-agent skill-use.**

![tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

Everyone writes an LLM-as-judge rubric by hand and then trusts it forever. But a
hand-written rubric is just a *guess* about what good agent behaviour looks like —
and most of its criteria turn out to carry no signal, while some are actively
misleading. RubricAgent treats the rubric as something to **learn**: score your
agent traces, correlate each criterion against real outcomes, then **reweight,
prune, and grow** the rubric so it becomes a better proxy for capability.

Runs with **zero dependencies and zero API keys** (deterministic heuristic
grader + pure-stdlib stats). Swap in an LLM grader for production judging.

> **Inspired by the mid-2026 agent-evaluation literature:**
> - *SkillCoach: Self-Evolving Rubrics for Evaluating and Enhancing Agentic Skill-Use* (2026).
> - *PACE: A Proxy for Agentic Capability Evaluation* (2026) — cheap proxies that track true agent capability.

---

## The result in one number

Give a flat, equal-weight rubric where only one of five criteria actually
predicts success (the rest are noise or misleading). One evolution pass:

```bash
python -m rubricagent.eval --n 200
```
```
AUC (flat rubric)    = 0.774
AUC (evolved rubric) = 1.000
improvement          = +0.226

criterion correlations with success:
  grounding      +1.000     ← the real signal, found and up-weighted
  completeness   +0.100
  correctness    +0.036
  relevance      +0.000
  clarity        -0.269     ← misleading, pruned
pruned : ['relevance', 'correctness', 'clarity']
discovered: discovered_signal
```

AUC here is a **PACE-style proxy**: how well the aggregate rubric score separates
runs that truly succeeded from those that failed. A better rubric is a better
proxy, so you can grade cheaply and at scale.

## Install

```bash
git clone https://github.com/ahmeddoghri/rubricagent
cd rubricagent && pip install -e .
```

## Grade a trace

```python
from rubricagent import Trace, Judge, starter_rubric, scorecard

rubric = starter_rubric("Research the launch date and summarize the risks.")
trace = Trace(task="...", response="According to the source I found ... therefore ...")

print(scorecard(Judge(rubric).judge(trace)))
```
```
### Scorecard
**Total: 78%**
| Criterion | Score | Rationale |
|---|---|---|
| grounding | `████████░░` 80% | matched 4/5 evidence terms: [source, according, tool, found] |
| correctness | `██████░░░░` 60% | matched 3/5 evidence terms: [because, therefore, correct] |
...
```

## Evolve the rubric from feedback

```python
from rubricagent import RubricEvolver

# traces you've already run, with a 0/1 label = did the agent actually succeed?
evolved, report = RubricEvolver().evolve(rubric, traces, labels)

print(report.auc_before, "->", report.auc_after)   # proxy quality went up
print(report.pruned)                                 # criteria with no signal
print(evolved.names())                               # the rubric that survived
```

**What one pass does**

1. **Reweight** — every criterion's weight becomes its correlation with success.
2. **Prune** — criteria below a signal floor are dropped (incl. anti-signals).
3. **Discover** — a new criterion is minted from evidence terms that are common
   in wins and rare in losses.

## Bring your own judge

`HeuristicGrader` is offline and deterministic. Any object with
`grade(criterion, trace) -> CriterionScore` works — wrap your LLM there:

```python
class LLMGrader:
    def grade(self, criterion, trace): ...   # ask a model, return CriterionScore

Judge(rubric, grader=LLMGrader())
```

## Tests

```bash
pip install pytest && pytest -q      # 6 passing
```

## License

MIT © Ahmed Doghri
