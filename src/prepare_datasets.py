"""MBPP preparation for Gemma recovery research.

Official task-id splits: train 601..974; validation 511..600; test 11..510.
Keep test split untouched until frozen evaluation settings are chosen.
"""
from __future__ import annotations

import json
from pathlib import Path
from datasets import load_dataset


def load_mbpp():
    ds = load_dataset("google-research-datasets/mbpp", "full")
    splits = {
        "train": [dict(x) for x in ds["train"]],
        "validation": [dict(x) for x in ds["validation"]],
        "test": [dict(x) for x in ds["test"]],
    }
    expected = {"train": (601, 974, 374), "validation": (511, 600, 90), "test": (11, 510, 500)}
    ids_seen = set()
    for name, (lo, hi, count) in expected.items():
        rows = splits[name]
        ids = {int(x["task_id"]) for x in rows}
        assert len(rows) == count == len(ids), f"Unexpected {name} size: {len(rows)}"
        assert all(lo <= i <= hi for i in ids), f"Unexpected {name} task IDs"
        assert ids.isdisjoint(ids_seen), "Task ID overlap detected!"
        ids_seen.update(ids)
    return splits


def exclude_evalplus_overlap(splits):
    """Prevent training/validation exposure of any EvalPlus MBPP+ task.

    MBPP+ includes some task IDs in original MBPP training/validation ranges.
    Load its *IDs only*; never use its reference solutions/hidden tests in prompts.
    """
    evalplus = load_dataset("evalplus/mbppplus", split="test")
    evalplus_ids = {int(x["task_id"]) for x in evalplus}
    clean = {
        "train": [r for r in splits["train"] if int(r["task_id"]) not in evalplus_ids],
        "validation": [r for r in splits["validation"] if int(r["task_id"]) not in evalplus_ids],
    }
    for name, rows in clean.items():
        assert {int(x["task_id"]) for x in rows}.isdisjoint(evalplus_ids)
    return clean, evalplus_ids


def write_manifest(splits, clean=None, evalplus_ids=None, output_path="/kaggle/working/mbpp_split_manifest.json"):
    obj = {name: {"count": len(rows), "task_ids": [int(x["task_id"]) for x in rows]}
           for name, rows in splits.items()}
    if clean is not None:
        obj["mbppplus_decontaminated"] = {
            name: {"count": len(rows), "task_ids": [int(r["task_id"]) for r in rows]}
            for name, rows in clean.items()
        }
    if evalplus_ids is not None:
        obj["evalplus_mbppplus_ids"] = sorted(evalplus_ids)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(obj, indent=2))
    return output_path


if __name__ == "__main__":
    s = load_mbpp()
    for name, rows in s.items():
        print(f"{name}: {len(rows)} rows; first ID {rows[0]['task_id']}")
    clean, evalplus_ids = exclude_evalplus_overlap(s)
    print("MBPP+ benchmark task IDs:", len(evalplus_ids))
    for name, rows in clean.items():
        print(f"Eligible {name}: {len(rows)} tasks after overlap exclusion")
    print(write_manifest(s, clean, evalplus_ids))
