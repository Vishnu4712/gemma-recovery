# Learning to Recover: A Controlled Study of Execution-Feedback Training for Small Coding Models

**Subtitle:** A matched-budget QLoRA comparison on function-level Python repair  
**Author:** [Your Name] — [Institution]  
**Research track:** Kaggle Gemma 4 Developer Agent Paper Track, November 2026

## Abstract

Can a small coding model learn to recover from failed code more effectively through failure–execution-feedback–correction training than through successful solutions alone? We compare Gemma 4 E4B-it in three conditions: unmodified base (C0), success-only QLoRA (C1), and recovery-oriented QLoRA (C2), with 55 training items and 5,963 supervised target tokens per adapter. On an audited development cohort of 29 officially incorrect states, C2 repairs 5/29 versus 4/29 for C1. **On the frozen final 70-task MBPP+ holdout, this advantage disappears: C1 and C2 both repair 2/24 officially incorrect, visibly failing starts, versus 3/24 for C0.** All three return correct final code for 16/30 tasks in a three-attempt development study. Visible-test feedback sometimes conflicts with the official evaluator, creating needless or harmful revisions. We present a reproducible negative result and discuss the limits of tiny offline recovery datasets and the importance of feedback quality.

# Introduction

Agentic coding systems need to do more than generate a first answer: they must interpret failed tests, correct their code, and stop before further revisions cause regressions. Tool-augmented recovery is especially consequential for small models with limited compute budgets. Offline supervised fine-tuning on successful solutions is straightforward, but its targets contain no explicit failed attempt or execution feedback. Recovery-oriented training instead conditions a corrected solution on the model’s preceding error and the observable test result.

We study whether that additional supervision produces a measurable benefit *under a small, matched training budget*. This is motivated by prior work finding that offline correction traces need not translate into self-correction behavior [3], while execution-feedback RL can improve iterative synthesis [4]. We restrict scope to standalone Python function synthesis and revision, rather than repository-level bug localization, file editing, or SWE-bench. The contribution is a transparent, reproducible controlled pilot, not a new optimization algorithm.

Our analysis distinguishes (a) visible-test failure, which triggers the agent’s retry; (b) official MBPP+ correctness, used only for offline evaluation; and (c) final returned-code correctness, which can be lower than the oracle score of any attempt encountered. We report both development and independently frozen holdout results, including null findings.

# Research Questions

**RQ1:** Does recovery-aware QLoRA outperform success-only QLoRA on identical failed starting states? **RQ2:** Are benefits concentrated in wrong-output, syntax, or runtime failures? **RQ3:** Does the recovery condition avoid repeated failed attempts or reduce tokens and test executions? **RQ4:** Does a benefit persist under a small retry budget and on held-out tasks or a separate programming benchmark?

# Methods

## Problem pool and trajectory collection

MBPP provides natural-language function specifications, reference implementations, and original visible tests [1]; the EvalPlus framework supplies more extensive scoring for MBPP+ and HumanEval+ [2]. We exclude 108 overlapping MBPP+ IDs from an initial 374-task MBPP training partition, retaining 266 eligible training problems. We sample 40 problems, generate two independent C0 trajectories each, and permit at most three attempts using the original public tests. Across 80 trajectories, 53 first attempts pass visible tests; nine of 27 initially unsuccessful trajectories subsequently pass. The initially failing attempts comprise 22 wrong-output, four syntax, and one runtime failure. The dataset also includes 17 verified-reference corrections. All such verification is based on original visible tests and is not proof of extended-test correctness.

## Training conditions

**C0:** The original instruction-tuned Gemma 4 E4B-it checkpoint [9]. **C1:** QLoRA trained on 50 successful model solutions and five reference solutions. **C2:** QLoRA trained on 27 successful model solutions, four standalone references, seven actual model self-recoveries, and 17 reference-based corrections. In a recovery item, the prompt includes the task, earlier unsuccessful code, and observable feedback; the target is corrected Python code. The prompt is masked from the loss. C1/C2 independently start from the same frozen C0; C2 is not further tuned from C1.

We use NF4 quantization, FP16, LoRA rank 8, scaling factor 16, `q_proj`/`v_proj` language-model targets, learning rate $10^{-4}$, 1,024-token maximum sequence length, batch size one, four gradient-accumulation steps, and one epoch (14 optimizer updates per condition). The training seed is 42. Each condition contains 55 items and exactly 5,963 supervised target tokens, but input prompt length and source mixture differ. This controls target-token supervision, *not* total FLOPs or informational content. Training took 54.4 s (C1) and 73.9 s (C2) on one NVIDIA T4, excluding model setup and evaluation [7,8].

## Evaluation and metrics

We partition 180 disjoint MBPP+ IDs into 110 development tasks (30 pilot plus 80 additional) and 70 final tasks. The final 70 IDs are recorded in the Day 5 manifest before their outcomes are inspected. The base model generates the *initial* code for each problem. States failing local visible tests form a paired *fixed-state* cohort: C0, C1, and C2 independently receive exactly the same starting code and feedback, then produce one correction each. The original visible tests are the only execution feedback used for generation. Official EvalPlus tests are evaluated in a separate stage, *after all completions are frozen*, and never provided to the generator.

The operational correction cohort contains all *visible-failing* starting states. For our audited metric we additionally restrict *post hoc* to the subset that fails official MBPP+ scoring initially, avoiding counting an already correct program as a repaired one. This restriction changes only the analysis denominator, not the model’s input, action, or evaluated task IDs. We define repair success as the resulting code passing official MBPP+ tests. Differences between C2 and C1 are measured per paired problem; confidence intervals use percentile bootstrap resampling of tasks (20,000 draws). A second, exploratory 30-task development experiment measures up to three sequential attempts and counts both *ever-correct* and *final returned-code* outcomes. All models use a 384-token generation cap per attempt.

# Results

## RQ1: Paired recovery and independent holdout

On development, 31 of 110 initial solutions fail visible tests. Two (IDs 139 and 276) nevertheless pass the official evaluator, leaving 29 audited incorrect states. C2 repairs five of these, while C1 and C0 each repair four. The C2–C1 difference is $+3.45$ percentage points (95% task-paired bootstrap interval $[0,10.34]$ points); one task accounts for the gain.

On the frozen 70-task holdout, 35 C0 initial answers (50.0%) pass official MBPP+. Twenty-seven first attempts fail the local visible checks, but three (IDs 87, 233, 312) already pass official tests, leaving 24 genuinely incorrect fixed starting states. The **primary holdout comparison is a tie**: C1 and C2 each repair 2/24 (8.3%), while C0 repairs 3/24 (12.5%). C1 and C2 succeed on exactly the same two problems; their empirical paired bootstrap difference and interval are both zero. This degenerate interval describes this sample, *not proof of population equivalence*. The final notebook scored 151 generated solutions (70 initials and $27\times3$ corrections). No model parameters or prompt choices were updated from the frozen holdout results.



| Split               |  C0  |  C1  |  C2  |
|:--------------------|:----:|:----:|:----:|
| Development (29)    | 4/29 | 4/29 | 5/29 |
| Frozen holdout (24) | 3/24 | 2/24 | 2/24 |

Official MBPP+ one-step repair rates on the same C0 failed states. Officially correct initial code is excluded only in the audited analysis.



**Figure (PDF):** Paired fixed-state repair, audited against official MBPP+ scoring. The apparent one-problem development advantage of C2 does not replicate on the frozen holdout.

## RQ2: Failure categories

Development failed states are concentrated in wrong-output cases: 29/31 visible failures are classified as wrong output and only two as syntax errors; none are runtime failures in this subset. C2 repairs 7/29 wrong-output cases when counting all visible-failing states, compared with 6/29 for C0 and C1. No condition repairs either of the two syntax cases. These descriptive counts are too small to attribute effects reliably to failure categories. The classification is derived from the local runner’s observed feedback, not a hand-labeled semantic bug taxonomy.

## RQ3–RQ4: Efficiency, retries and transfer

In the 30-task development end-to-end study, C0, C1, and C2 each return officially correct *final* code on 16/30 tasks. Counting any officially passing attempt instead yields 16/30, 17/30, and 17/30: both fine-tuned conditions sometimes revise a correct intermediate answer into an incorrect final one. Total output tokens are 6,086 (C0), 5,280 (C1), and 6,147 (C2). Normalized by *final* solved count, this is 380.4, 330.0, and 384.2 generated tokens per solved problem. Mean visible test executions per task are 1.50, 1.47, and 1.53. Hence the data do not show a C2 efficiency advantage. Exact repeated unsuccessful corrections in a separate fixed-state development comparison number 4, 3, and 3 for C0/C1/C2.

For exploratory transfer, first-attempt HumanEval+ correctness on 20 tasks is 11/20 (C0), 11/20 (C1), and 12/20 (C2). This is *not* a transfer-of-recovery benchmark, because no correction step is evaluated, and the sample is too small for a robust generalization claim. The frozen final experiment tests one fixed-state correction, not multi-attempt sequential recovery or cross-benchmark transfer.

## Feedback specificity and reliability

On 24 newly added development failure states, C2 repairs 5 with detailed local test feedback versus 3 with only a generic retry instruction, a paired $+8.3$ percentage-point difference (95% bootstrap interval $[0,20.8]$). C1 was not evaluated under this same ablation, so this does not isolate an effect of recovery training.

Local and official tests disagree in both splits (Table 2). On development, 17 initially visible-passing solutions fail the official tests and two initially visible-failing solutions pass them; on the frozen holdout, the corresponding counts are 11 and three. A separate reference-compatibility probe flags one reference implementation (holdout ID 123) with a local runtime/resource-limit error; we keep all 70 IDs rather than silently excluding it. These cases suggest that test coverage, equality semantics, and runner constraints deserve explicit checking. In particular, a false local failure signal can trigger a harmful edit, as seen in the end-to-end regression on development task 436.



| Split         | Local result | Official pass | Official fail |
|:--------------|:-------------|--------------:|--------------:|
| Development   | Pass         |            62 |            17 |
| Development   | Fail         |             2 |            29 |
| Final holdout | Pass         |            32 |            11 |
| Final holdout | Fail         |             3 |            24 |

C0 first-attempt visible-test agreement with official MBPP+ scoring. Rows are the local test outcome.



# Discussion and Limitations

The simplest interpretation is *no detected benefit* from recovery-aware SFT with this small dataset, rather than a claim that learning recovery is impossible. Only seven C2 items are genuine model self-recoveries; the remaining recovery targets come from reference solutions. C2 also replaces some plain successes with correction-conditioned cases relative to C1. Such mixture changes, combined with 14 optimizer updates, a single seed, and prompt-length differences, limit causal attribution. The repaired-state cohorts (29 development and 24 final) are small; a one-case difference shifts the observed rate by 3.4–4.2 percentage points. Because starting states are sampled from the same base model, this experiment does not measure recovery from each fine-tuned model’s *own* error distribution, an important distribution-shift caveat discussed by SCoRe .

The paper does not claim repository-scale software engineering capability. Original MBPP tests are imperfect proxies for correctness, and the subprocess runner has CPU/address-space limits but is *not* a security sandbox. We cannot exclude pretraining contamination by task-ID split alone. Moreover, the degenerate holdout paired CI arises from identical observed successes and offers limited uncertainty characterization outside this sample. We did not try extra epochs, change the dataset after holdout inspection, or train on holdout outcomes.

Promising future work includes collecting hundreds of independently verified *self*-recoveries, checking candidate corrections against stronger tests before supervised training, balancing correction and success mixtures, explicitly supervising diagnosis or root-cause localization, and comparing additional training epochs with multiple seeds. Such experiments should be tuned on new development data and evaluated on a *new* untouched holdout, not reused to claim fresh independent confirmation from these 70 tasks. Recent negative-trajectory training research motivates better error attribution rather than treating all failed examples as equally informative [6].

# Conclusion

A matched-budget comparison of success-only and recovery-aware QLoRA for Gemma 4 E4B-it found a one-case development improvement but **no improvement in the independently frozen holdout**. Both fine-tuned conditions repaired 2/24 officially incorrect initial states, and all three conditions returned the same number of correct final programs in the exploratory multi-turn development study. The most actionable lessons are that offline recovery data must be diverse and validated, paired evaluations must control the failed starting state, and a coding agent should be evaluated on its *final* code under reliable feedback. The accompanying notebooks and scripts reproduce the low-resource study and explicitly document these negative results.

# Artifact availability

Source, notebooks, manifests, audited summaries, and figure scripts are supplied in the accompanying project repository; the public GitHub URL and Kaggle versioned artifact links must be filled in by the author before submission. Raw model-generated code and adapters should be published separately, if permitted, to enable complete independent replay.


# Related Work and References

Our evaluation complements rather than reproduces repository-level coding-agent methods. SCoRe [3] studies multi-turn RL self-correction; RLEF [4] improves use of execution feedback; Reflexion [5] uses verbal feedback at inference time. Negative-trajectory work [6] motivates structured error attribution. LoRA [7] and QLoRA [8] underpin the efficient training strategy.

1. Austin et al. (2021), [Program Synthesis with Large Language Models](https://arxiv.org/abs/2108.07732) (MBPP).
2. Liu et al. (2023), [Is Your Code Generated by ChatGPT Really Correct?](https://proceedings.neurips.cc/paper_files/paper/2023/hash/43e9d647ccd3e4b7b5baab53f0368686-Abstract.html) (EvalPlus).
3. Kumar et al. (2025), [Training Language Models to Self-Correct via Reinforcement Learning](https://proceedings.iclr.cc/paper_files/paper/2025/hash/871ac99fdc5282d0301934d23945ebaa-Abstract-Conference.html) (SCoRe).
4. Gehring et al. (2025), [RLEF: Grounding Code LLMs in Execution Feedback with Reinforcement Learning](https://proceedings.mlr.press/v267/gehring25a.html).
5. Shinn et al. (2023), [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366).
6. Kim (2026), [How to Learn from Negative Trajectories in Coding Agents](https://microsoft.github.io/debug-gym/blog/2026/08/negative-pi/).
7. Hu et al. (2022), [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685).
8. Dettmers et al. (2023), [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314).
9. Google DeepMind (2026), [Gemma 4 Model Card](https://ai.google.dev/gemma/docs/core/model_card_4).

**Public artifacts:** [Insert public GitHub repository URL] · [Insert public Kaggle notebook/version URL] · [Insert public PDF Project Link].

**Provenance note:** The final holdout summary was extracted from the executed Kaggle notebook. To independently recompute holdout rows, download `gemma_final_holdout_backup.zip` from Kaggle and run `scripts/audit_day6.py` from the accompanying repository.
