# Reproducing Days 1–6

This repository contains the source code of the executed Kaggle experiments, with notebooks stripped of execution outputs for GitHub. Model weights and raw generated code are not committed. The printed final-holdout statistics come from the uploaded executed `final-holdout-day-6-2.ipynb`; full independent verification needs its saved Kaggle ZIP.

## Environment and safety

- Tested in Kaggle with NVIDIA Tesla T4; CUDA, 4-bit quantization, Transformers 5.x / PEFT / BitsAndBytes and EvalPlus. The notebooks install the runtime dependencies themselves.
- Model checkpoint: `google/gemma-4-E4B-it`. Follow the model terms and use Kaggle Secrets for any credentials.
- Generated programs run in a restricted Python subprocess with resource and wall-clock limits. **This does not constitute an adversarial security sandbox**; run only on disposable machines without secrets or permissive network/filesystem access.
- Do not expect local reproduction to exactly match GPU/model decoding results if package versions and hardware differ. Preserve original executed Kaggle versions.

## Pipeline

1. `01_baseline.ipynb` creates decontaminated MBPP training split and tests the base model.
2. `02_recovery_data.ipynb` collects 80 trajectories (40 problems × 2) with up to 3 attempts and produces direct success, self-correction and reference-correction data. All acceptance checks use original visible tests.
3. `03_finetune_gemma.ipynb` builds matched target-token plans (55 samples / 5,963 supervised target tokens each), trains independent C1 and C2 rank-8 QLoRA adapters from frozen C0, and saves the adapter folders.
4. `04_evaluate_recovery.ipynb` scores an earlier 30-task development sample using fixed failed C0 starts and an offline MBPP+ scorer.
5. `05_failure_efficiency.ipynb` extends development to 110 unique MBPP+ tasks, adds an end-to-end 30-task analysis, feedback-ablation and HumanEval+ pilot; it freezes a separate 70-task final manifest.
6. `06_final_holdout.ipynb` loads **only saved Day 2 / Day 3 / Day 5 outputs**, checks all 70 frozen IDs, generates 70 initial C0 solutions, selects 27 local visible failures, generates 81 one-step corrections, and submits **151 samples** to the offline EvalPlus scorer. It intentionally does **not** drop a reference whose code fails a runner compatibility check.

Kaggle output directories under `/kaggle/input/notebooks/...` are read-only; use **Add Input → Notebooks** for the saved preceding versions. A new Save & Run All executes in a separate session; do not assume interactive `/kaggle/working` files survive between sessions. Download `gemma_final_holdout_backup.zip` from the completed version output.

## Audit scoring without GPU

```bash
python scripts/audit_day5.py /path/to/gemma_day5_backup.zip \
  --out results/development/day5_recomputed.json
python scripts/audit_day6.py /path/to/gemma_final_holdout_backup.zip \
  --out results/final_holdout/day6_recomputed.json
python -m unittest discover -s tests -v
```

The Day 6 audit validates the frozen manifest, uniqueness of 70 initial records, presence of all three corrections for each local visible failure, and exact official score keys. It then recomputes audited incorrect-state repair and paired bootstrap confidence intervals. It uses no model inference.

### Stats provenance

- `results/development/day5_audit.json` was calculated from the Day 5 raw backup ZIP and can be recomputed.
- `results/final_holdout/final_holdout_summary.json` is a **record of the executed Kaggle notebook's Section L output** (with the notebook SHA256); the raw ZIP itself is not in the release. It can be independently checked only when the completed Kaggle ZIP is added.
- The 70 final task IDs come from `configs/day5_partition_manifest.json`, frozen before Day 6; do not retune on this set if treating it as an independent final evaluation.

### Metrics

A correction is eligible in the **operational** cohort when C0's initial code fails the local original MBPP tests. The **audited** cohort additionally removes inputs whose original solution was already officially correct, *only in the analysis*. Models never receive the extended-test results in their feedback. For multi-attempt evaluation, *final returned-code success* is different from the *ever-correct* diagnostic; a later failed attempt may overwrite an earlier correct program.

