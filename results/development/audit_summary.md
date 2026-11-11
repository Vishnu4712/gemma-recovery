# Day 5 evaluation audit — development-only

Audited against the uploaded `gemma_day5_backup.zip`, using `audit_day5.py`.

## Corrected metrics

| Metric | C0 | C1 | C2 |
|---|---:|---:|---:|
| One-correction success among 31 **visible-failing** starts | 6/31 | 6/31 | 7/31 |
| One-correction success among 29 **officially incorrect** starts | 4/29 (13.8%) | 4/29 (13.8%) | 5/29 (17.2%) |
| End-to-end 3-attempt **ever passed** official scorer (30 tasks) | 16 | 17 | 17 |
| End-to-end 3-attempt **final returned code** passed (30 tasks) | 16 | 16 | 16 |
| Completion tokens / **final solved** problem | 380.4 | 330.0 | 384.2 |
| Mean test executions / task | 1.50 | 1.47 | 1.53 |

C2 minus C1 fixed-state recovery on genuinely incorrect starts: **+3.45 percentage points**, paired 95% percentile-bootstrap interval **[0, 10.34] percentage points** (29 tasks; one discordant success).

## Feedback diagnostic

- 110 development problems were scored: 30 Day 4 + 80 new Day 5.
- 79 initial solutions passed original MBPP visible tests, but 17 of these failed official MBPP+ scoring.
- 31 initially failed original visible tests, but **2 of those passed official MBPP+ scoring** (task IDs **139 and 276**). These are visibly failing but officially correct starting states; exclude them only as a **post-hoc analysis** for true repair. Their official outcomes **must never influence model prompts or task selection**.
- C2 detailed feedback vs vague retry: **5/24 vs 3/24**, exploratory paired interval [0, 20.8] percentage points.
- HumanEval+ exploratory single-attempt transfer: **11/20 C0, 11/20 C1, 12/20 C2**.
- C1 and C2 match on final returned-code correctness (16/30); the notebook's `ever passed` metric is an oracle diagnostic, **not** what a feedback-driven agent can know online.

## Interpreting the result

The **one extra C2 recovery** does not establish general improvement. We have one training seed, one training epoch, 55 examples/condition and only 7 model self-recoveries included in C2. The audit also identifies a practical failure mode: **visible-test feedback may disagree with the stricter official scorer**, and subsequent corrections can damage previously correct code.

## Final holdout

A distinct **70-task MBPP+ holdout** exists in `day5_partition_manifest.json` and was not evaluated in the provided Day 5 ZIP. The separate `06_final_holdout.ipynb` evaluates it once, without changing the frozen IDs or training adapters. **The draft paper currently contains development results only.** Do not claim holdout conclusions before that notebook is executed and scored.
