# 📊 RubricAgent

**Self-evolving rubrics for evaluating LLM-agent skill-use.**

![CI](https://github.com/ahmeddoghri/rubricagent/actions/workflows/ci.yml/badge.svg)
![tests](https://img.shields.io/badge/tests-6%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Learn your LLM-as-judge rubric from outcomes instead of guessing it.**
> In the benchmark, rubric quality (AUC vs. ground truth) climbs
> **0.77 → 1.00** as it prunes dead criteria and grows new ones.
> `python -m rubricagent.eval`.

Somebody on your team wrote a five-criteria rubric for grading agent
transcripts, everyone nodded, and it's been treated as gospel ever since.
Nobody has checked whether all five criteria actually predict anything.
Spoiler from our own benchmark below: "clarity" was actively anti-correlated
with success. The rubric was penalizing the agent for being clear.

RubricAgent treats the rubric as something you learn, not something you
write once and defend forever. Score your agent traces, correlate each
criterion against real outcomes, then **reweight, prune, and grow** the
rubric so it becomes an honest proxy for capability instead of a vibe check
with a scoring column.

Runs with **zero dependencies and zero API keys** (deterministic heuristic
grader plus pure-stdlib stats). Swap in an LLM grader when you're ready to
judge for real.

---

## The result in one number

Start with a flat, equal-weight rubric where only one of five criteria
actually predicts success. The rest are noise, or worse:

```bash
python -m rubricagent.eval --n 200
```
```
AUC (flat rubric)    = 0.774
AUC (evolved rubric) = 1.000
improvement          = +0.226

criterion correlations with success:
  grounding      +1.000     the real signal, found and up-weighted
  completeness   +0.100
  correctness    +0.036
  relevance      +0.000
  clarity        -0.269     misleading. pruned on sight.
pruned : ['relevance', 'correctness', 'clarity']
discovered: discovered_signal
```

AUC here is a proxy: how well the aggregate rubric score separates traces
that truly succeeded from the ones that didn't. A better rubric is a better
proxy, which means you can grade cheaply and at scale without lying to
yourself about quality.

## Install

```bash
git clone https://github.com/ahmeddoghri/rubricagent
cd rubricagent && pip install -e .
```

Or with Docker:

```bash
docker build -t rubricagent .
docker run --rm rubricagent
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
Total: 14%

| Criterion | Score | Rationale |
|---|---|---|
| relevance | 0% | matched 0/5 evidence terms: [] |
| correctness | 20% | matched 1/5 evidence terms: ['therefore'] |
| completeness | 0% | matched 0/6 evidence terms: [] |
| grounding | 50% | matched 3/6 evidence terms: ['source', 'according', 'found'] |
| clarity | 0% | matched 0/3 evidence terms: [] |
```

Yes, a real trace scores 14%. This is a heuristic grader running on a
deliberately thin, half-finished example response ("... therefore ...").
That's the point. It's not going to flatter you into thinking the demo is
smarter than it is.

## Evolve the rubric from feedback

```python
from rubricagent import RubricEvolver

# traces you've already run, with a 0/1 label: did the agent actually succeed?
evolved, report = RubricEvolver().evolve(rubric, traces, labels)

print(report.auc_before, "->", report.auc_after)   # proxy quality went up
print(report.pruned)                                 # criteria with no signal
print(evolved.names())                               # the rubric that survived
```

**What one pass does**

1. **Reweight.** Every criterion's weight becomes its correlation with success.
2. **Prune.** Criteria below a signal floor get dropped, anti-signals first.
3. **Discover.** A new criterion gets minted from evidence terms that show up
   constantly in wins and rarely in losses.

## Bring your own judge

`HeuristicGrader` is offline and deterministic, which is great for CI and
useless for nuance. Any object with `grade(criterion, trace) -> CriterionScore`
works. Wrap your LLM there when you want the real thing:

```python
class LLMGrader:
    def grade(self, criterion, trace): ...   # ask a model, return CriterionScore

Judge(rubric, grader=LLMGrader())
```

## Tests

```bash
pip install pytest && pytest -q      # 6 passing
```

## More in this series

Nine small, dependency-light, benchmarked tools for LLM/ML infrastructure. Each one reproduces its headline number locally with no API keys:

[agentmem](https://github.com/ahmeddoghri/agentmem) · [clarifyrag](https://github.com/ahmeddoghri/clarifyrag) · [churnfm](https://github.com/ahmeddoghri/churnfm) · [citebench](https://github.com/ahmeddoghri/citebench) · [guardrail-gate](https://github.com/ahmeddoghri/guardrail-gate) · [tablextract](https://github.com/ahmeddoghri/tablextract) · [vllm-cost-router](https://github.com/ahmeddoghri/vllm-cost-router) · [taggate](https://github.com/ahmeddoghri/taggate)

## License

MIT © Ahmed Doghri
