# Learning to Recover: Execution-Feedback Training for Small Coding Models

**Kaggle Gemma 4 Developer Agent Competition - Paper Track (November 2026)**  

A small-model empirical study of whether **recovery-aware supervised fine-tuning** improves Gemma 4 E4B-it's ability to repair its own function-level Python errors. We compare three controlled conditions: **C0**, the original model; **C1**, a success-only QLoRA adapter; and **C2**, a mixed success + execution-feedback-correction QLoRA adapter. The study is *not* repository-level SWE-bench or multi-file debugging.

## Main results (official EvalPlus MBPP+)

| Paired fixed-state one-correction repair | C0 base | C1 success-only | C2 recovery-aware |
|---|---:|---:|---:|
| **Development** — 29 officially incorrect visible-failing starts | 4/29 (13.8%) | 4/29 (13.8%) | **5/29 (17.2%)** |
| **Frozen final holdout** — 24 officially incorrect visible-failing starts | 3/24 (12.5%) | **2/24 (8.3%)** | **2/24 (8.3%)** |
| End-to-end development, final correct program (30 tasks, up to 3 tries) | 16/30 | 16/30 | 16/30 |

The held-out recovery comparison **does not support a C2 improvement**: C1 and C2 repaired exactly the same two cases. The development advantage was one additional correction, with paired-bootstrap CI touching zero. Initial C0 accuracy on the 70 held-out tasks was 35/70 (50%). For the 27 holdout starts failing the local visible tests, three were already officially correct, leaving 24 genuinely incorrect states. The completed Day 6 notebook reports 151 official scoring requests (70 first attempts + 81 corrections).

For an additional development feedback diagnostic, C2 repaired 5/24 states with detailed failure feedback versus 3/24 with a generic retry prompt; this was **not** also tested for C1 and does not isolate the effect of fine-tuning.

**Paper:** [`paper/paper.pdf`](paper/paper.pdf) (compiled LaTeX); [`paper/paper.tex`](paper/paper.tex) (editable source); [`paper/kaggle_writeup.md`](paper/kaggle_writeup.md) (Kaggle Writeup text). The paper explicitly discusses negative results and the limitations of a tiny training dataset (55 examples each, seven genuine C2 self-recoveries, one epoch, one seed).

## Reproducibility and provenance

```text
notebooks/                  Six day-by-day, output-stripped Kaggle notebooks
src/                        Code runner, task/trajectory generation, QLoRA, evaluation
configs/                    Frozen split, training, development and holdout manifests
results/development/        Independently re-audited Day 5 results
results/final_holdout/      Day 6 summary extracted from executed final notebook
scripts/audit_day5.py       CPU-only Day 5 raw ZIP audit
scripts/audit_day6.py       CPU-only Day 6 raw ZIP audit
scripts/make_figures.py     Development figure generation
tests/                      Sanity tests for the holdout audit
paper/                      LaTeX, PDF, figures, Kaggle Writeup

docs/REPRODUCING.md        Step-by-step run and scoring provenance
docs/SUBMISSION_CHECKLIST.md
```

The successful executed notebook is also archived at `results/final_holdout/executed_day6_notebook.ipynb` for output provenance. The holdout summary saved in `results/final_holdout/final_holdout_summary.json` was **extracted from the successful execution output** of `final-holdout-day-6-2.ipynb`. The corresponding `gemma_final_holdout_backup.zip` (with individual generated code and official scoring JSONL rows) was **not attached to this release**. To independently recompute the entire table from rows, first download that ZIP from the saved Kaggle run and execute:

```bash
python scripts/audit_day6.py /path/to/gemma_final_holdout_backup.zip \
  --out results/final_holdout/day6_recomputed.json
```

This release contains complete experiment **source code** and saved **summary statistics**, but not the full raw generation outputs or large adapter checkpoints. Link to a public Kaggle version or separately share appropriate outputs for complete model-level replay. Do not call the extracted summary an independent re-audit.

## Running the notebooks

1. Create a Kaggle GPU notebook and obtain the approved `google/gemma-4-E4B-it` model checkpoint following its terms. Keep Hugging Face credentials in Kaggle Secrets.
2. Run `notebooks/01_baseline.ipynb` through `notebooks/03_finetune_gemma.ipynb` or attach their completed versioned outputs.
3. Run `notebooks/04_evaluate_recovery.ipynb` and `notebooks/05_failure_efficiency.ipynb`. Preserve the frozen Day 5 holdout manifest unchanged.
4. Attach saved Day 2, Day 3 and Day 5 outputs and run `notebooks/06_final_holdout.ipynb`. This is the **corrected and completed** final holdout code; it preserves all 70 frozen IDs when the local reference-test runner disagrees.
5. Run separate official EvalPlus scoring after generation; do not feed official scores back to the model. Download the generated ZIP backup from Kaggle.

See [full reproduction instructions](docs/REPRODUCING.md). The original project ran on a single T4 GPU for inference and QLoRA; training C1/C2 used rank-8 adapters and one epoch. Version installation can change the environment; preserve the executed Kaggle runs for exact version details.

## Important caveats

- Function-level Python repair, not repository navigation, localization, multi-file patch generation, or contest prediction-track submission.
- The official scoring uses test suites withheld from model feedback, but initial problems and original visible tests are public.
- Local subprocess execution is **not a hardened sandbox**. Do not run generated code with production credentials or access to private files.
- A task-ID split cannot rule out the base model's pretraining exposure to benchmark problems.
- Single training seed, 55 supervised examples per adapter, only seven genuine self-recoveries in C2, unmatched prompt lengths.
- Observed test disagreement means "initially fails visible tests" is not synonymous with "incorrect under official scoring".
- Reusing the 70 held-out problems to select new epochs or tune prompts would make any subsequent score exploratory.

## Publishing and citation

Add your public GitHub and Kaggle version URLs before submitting the final Kaggle Writeup. Keep raw generated-code logs, adapters, access tokens and copied benchmark data outside the repository unless rights and privacy have been checked. The package intentionally **does not include a LICENSE**; select an appropriate license before making it public if you wish to grant reuse permissions, and preserve upstream licenses.

The competition's **Paper Track** requires a **submitted Kaggle Writeup**; a GitHub repository or PDF alone is not the submission. Paper Track deadline: **November 12, 2026, 11:59 PM UTC** (6:59 PM Eastern Standard Time), subject to official organizer updates.
