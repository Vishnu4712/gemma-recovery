# Day 6 frozen holdout — results provenance

- Source: executed Kaggle notebook `final-holdout-day-6-2.ipynb`, Section L.
- Frozen manifest: `../../configs/day5_partition_manifest.json` (70 disjoint IDs).
- Recorded summary: `final_holdout_summary.json` extracted directly from the notebook's actual saved execution output.
- Official EvalPlus scoring requests reported by notebook: 151 = 70 initial + (27 visible failures × 3 corrections).
- Scoring for the 24 officially incorrect visible-failing initial states: C0 3/24, C1 2/24, C2 2/24.
- C1/C2 solve the same two problems; the empirical difference and paired-bootstrap interval are zero in this sample, **not evidence of general equivalence**.
- Three false local failure signals (already officially correct): IDs 87, 233, 312.
- One reference local-runner issue: task 123, reported exit=-24. No exclusion.

**Limit:** The uploaded executed notebook did not include the separately downloadable `gemma_final_holdout_backup.zip` as an attachment to this workspace. This repository therefore includes the reported summary, but **not** the row-level official scores / generated codes. For an independent verification, download the ZIP from Kaggle and run `python scripts/audit_day6.py /path/to/gemma_final_holdout_backup.zip`. Do not describe the extracted summary as independently re-audited until that step is complete.
