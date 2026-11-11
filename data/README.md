# Training and evaluation data

Data in this study comes from **MBPP**, **MBPP+** (EvalPlus) and a smaller **HumanEval+** transfer sample. The training examples were produced by the C0 model and checked using original visible tests, with a subset of failed states paired with verified reference solutions. Exact selection and sample provenance are described by the notebooks and `configs/`.

We deliberately do not redistribute the raw programming problems, generated code, or full model checkpoints in this GitHub package. The saved Kaggle outputs from Days 2–5 contain the recorded example JSONL files and adapter artifacts. Link those versions in the repository after confirming public sharing is permitted under upstream licenses/terms and after checking that no credentials or private data are present.

For final MBPP+ evaluation, use `configs/day5_partition_manifest.json`; do not select holdout problems based on outcomes or edit the split after seeing them.
