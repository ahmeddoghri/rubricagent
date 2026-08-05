# 📊 RubricAgent

**Self-evolving rubrics for evaluating LLM-agent skill-use.**

![tests](https://img.shields.io/badge/tests-21%20passing-brightgreen)
![python](https://img.shields.io/badge/python-3.9%2B-blue)
![deps](https://img.shields.io/badge/runtime%20deps-none-success)
![license](https://img.shields.io/badge/license-MIT-black)

> **Learn your LLM-as-judge rubric from outcomes instead of guessing it.**
> In the benchmark, rubric quality (AUC vs. ground truth) climbs
> **0.77 → 1.00** as it prunes dead criteria and grows new ones.
> `python -m rubricagent.eval`.
>
> The benchmark's own traces were built from the grader's keyword list:
> the grounding marker string contains 5 of the 6 words the grounding
> criterion searches for. On ordinary language, that criterion's
> correlation with the true label is **0.000**, not weak, blind.
> `python -m rubricagent.eval_v2` is the benchmark that found it, and a
> wider-vocabulary grader that gets it to 0.88.

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

## The dataset was built from the grader's own keyword list

Look at how the benchmark generates its "grounding" examples versus what the
grounding criterion searches for:

```python
# rubric.py
Criterion("grounding", ..., ["source", "cite", "according", "tool", "search", "found"])

# eval.py
_GROUND = "according to the source the tool search found evidence"
```

Five of six keywords, verbatim, in the marker string that decides the label.
The 0.774 -> 1.000 climb measures whether keyword-overlap scoring can detect
data built from keyword overlap. It says nothing about whether the grader
can read a transcript that doesn't announce its own groundedness in those
exact words, which is every transcript that has ever existed.

```bash
python -m rubricagent.eval_v2
```
```
Does the grounding criterion actually see grounding?
corpus / grader       AUC before   AUC after  grounding corr
adversarial / v1           0.639       0.944           0.000
adversarial / v2           0.944       1.000           0.883
holdout / v1                0.389       0.667           0.447
holdout / v2                0.667       1.000           0.834
```

`grounding corr` is the number that matters here: the starter rubric's own
grounding criterion's correlation with the true label, before evolution
touches anything. On ordinary language ("The SEC 10-K filing lists a March
launch" vs. "I think it probably launches next year"), the original
criterion correlates at **0.000**. It is not weakly grounded-detecting; it
is blind. AUC after evolution still climbs to 0.944, which would look fine
in isolation, because the separate "discover new criterion" mechanism mines
raw trace text directly and does not depend on the grader at all. The
criterion the rubric ships with never worked.

`HeuristicGraderV2` widens each criterion's evidence vocabulary (grounding
goes from 6 words to about 30, covering "filing", "disclosure",
"transcript", "confirmed", "per", and the rest of how citation actually
reads) and adds basic stemming, so "cited" and "citing" match "cite". Held
out and evaluated once, after the vocabulary was frozen against the corpus
above: correlation 0.447 to 0.834.

`Judge`'s default stays `HeuristicGrader`, unchanged, so the numbers at the
top of this README keep reproducing exactly. Pass `HeuristicGraderV2()`
explicitly, to `Judge` or to `RubricEvolver(grader=...)`, to grade on
vocabulary that generalizes past this project's own synthetic catalog.

### A second bug this surfaced: evolve() ignored any grader you gave it

`RubricEvolver.evolve()` constructed `Judge(rubric)` with no grader argument,
twice, regardless of what was passed to `RubricEvolver(grader=...)`. There
was no way to actually evolve a rubric against `HeuristicGraderV2`, or a real
LLM grader, only to call `judge()` once with it. Fixed: the grader you
configure is now the one evolution actually uses.

### A third: the "discover new criterion" step could hallucinate

Signal is judged by a bare proportion gap (`p - n > 0.25`) with no minimum
sample size. On text where the label is a pure coin flip, unrelated to any
word in it, that produced a confidently-reported "discovered_signal" backed
by nothing in about 1 run in 25. Requiring the term to actually appear in a
real share of the successful traces, not just clear a small proportion gap
that a handful of coincidental hits satisfies, brought that to 0 in 200
trials without weakening the real discovery this project's own benchmark
depends on (still fires, same AUC).

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
