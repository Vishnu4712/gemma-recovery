"""Day 2 trajectory extraction and SFT dataset construction.

Adapted to the user's Day 1 scripts:
  prepare_datasets.py, agent_loop.py, test_runner.py.

Generated programs are executed via Day 1's resource-limited subprocess runner.
THIS IS NOT A SECURITY SANDBOX. Do not run with secrets or sensitive files.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from agent_loop import build_initial_messages, run_agent
from test_runner import run_visible_tests


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Broken JSONL at {path}:{line_no}; restore the last checkpoint") from exc
    return rows


def write_jsonl(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_training_tasks(eligible: list[dict], seed: int, limit: int) -> list[dict]:
    unique = {int(row["task_id"]): row for row in eligible}
    if len(unique) != len(eligible):
        raise ValueError("Duplicate training task IDs")
    ids = sorted(unique)
    random.Random(seed).shuffle(ids)
    return [unique[i] for i in ids[:min(len(ids), limit)]]


def collect_trajectories(
    problems: list[dict],
    generate_fn: Callable,
    checkpoint_path: str | Path,
    model_id: str,
    seed_fn: Callable[[int], None] | None = None,
    seed: int = 42,
    samples_per_task: int = 2,
    max_attempts: int = 3,
    max_new_tokens: int = 384,
) -> list[dict]:
    """Run initial generation + feedback-conditioned attempts; append every trajectory.

    Safe to rerun with same settings: keys already completed are skipped.
    """
    if not 1 <= max_attempts <= 3:
        raise ValueError("Supported retry budget: 1..3 total attempts")
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    prior = read_jsonl(checkpoint_path)
    # Checkpoints with incompatible configurations should not be reused silently.
    for row in prior:
        if (row.get("model") != model_id or row.get("generation_seed_base") != seed or
            row.get("max_attempts") != max_attempts or row.get("max_new_tokens") != max_new_tokens):
            raise ValueError("Checkpoint configuration differs from this run. Use a new output path.")
    completed = {(int(r["task_id"]), int(r["sample_index"])) for r in prior}
    remaining = sum((int(p["task_id"]), i) not in completed for p in problems for i in range(samples_per_task))
    print(f"Existing trajectories: {len(prior)} | New trajectories to generate: {remaining}", flush=True)
    for p in problems:
        task_id = int(p["task_id"])
        for sample_idx in range(samples_per_task):
            key = (task_id, sample_idx)
            if key in completed:
                continue
            trajectory_seed = seed + task_id * 1009 + sample_idx
            if seed_fn is not None:
                seed_fn(trajectory_seed)
            row = run_agent(p, generate_fn, max_attempts=max_attempts, max_new_tokens=max_new_tokens)
            row.update({
                "model": model_id,
                "split": "train",
                "sample_index": sample_idx,
                "trajectory_seed": trajectory_seed,
                "generation_seed_base": seed,
                "max_attempts": max_attempts,
                "max_new_tokens": max_new_tokens,
                "created_utc": datetime.now(timezone.utc).isoformat(),
            })
            with checkpoint_path.open("a", encoding="utf-8") as out:
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                os.fsync(out.fileno())
            prior.append(row)
            completed.add(key)
            statuses = " -> ".join(h["status"] for h in row["history"])
            print(f"task={task_id} sample={sample_idx}: {statuses}; total new tokens="
                  f"{row['total_completion_tokens']}", flush=True)
    return prior


def _retry_text(feedback: str) -> str:
    # Keep *exactly* the user retry instruction from Day 1 agent_loop.py.
    return ("The previous code FAILED the visible tests.\n" + feedback +
            "\nProvide a corrected full Python solution. Do not repeat the same code. Output code only.")


def _new_example(task: dict, target: str, kind: str, source: str, **meta) -> dict:
    prompt = build_initial_messages(task)
    completion = [{"role": "assistant", "content": target}]
    return {
        "task_id": int(task["task_id"]),
        "example_type": kind,
        "source": source,
        "messages": prompt + completion,
        "prompt": prompt,
        "completion": completion,
        "target_code": target,
        **meta,
    }


def _recovery_example(task: dict, failed: dict, correction: str, source: str, sample_index: int) -> dict:
    messages = build_initial_messages(task) + [
        {"role": "assistant", "content": failed["code"]},
        {"role": "user", "content": _retry_text(failed["feedback"])},
        {"role": "assistant", "content": correction},
    ]
    # Conversational prompt-completion format allows Day 3 to train on the
    # CORRECTED assistant reply only, not the failed earlier assistant turn.
    return {
        "task_id": int(task["task_id"]),
        "sample_index": sample_index,
        "example_type": "recovery",
        "source": source,
        "failure_type": failed["status"],
        "failed_code": failed["code"],
        "feedback": failed["feedback"],
        "corrected_code": correction,
        "messages": messages,
        "prompt": messages[:-1],
        "completion": [messages[-1]],
        "target_code": correction,
        "target_turn_only": True,
        "verified_on": "original MBPP visible tests only",
    }


def build_datasets(
    eligible_train: list[dict],
    trajectories: list[dict],
    out_dir: str | Path,
    evalplus_ids: set[int],
    max_success_per_task: int = 2,
    max_recovery_per_task: int = 5,
) -> dict:
    """Verify all selected solutions and export model-vs-reference recovery separately.

    IMPORTANT: Visible-test pass is the only acceptance criterion in Day 2.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tasks = {int(t["task_id"]): t for t in eligible_train}
    if set(tasks) & evalplus_ids:
        raise AssertionError("Training examples overlap MBPP+ evaluation IDs")
    invalid_ids = {int(r["task_id"]) for r in trajectories} - set(tasks)
    if invalid_ids:
        raise AssertionError(f"Unexpected/non-training trajectory IDs: {sorted(invalid_ids)}")

    by_task: dict[int, list[dict]] = {task_id: [] for task_id in tasks}
    for tr in trajectories:
        by_task[int(tr["task_id"])].append(tr)

    c1: list[dict] = []
    self_repairs: list[dict] = []
    ref_repairs: list[dict] = []
    reference_failures = []
    unverified_count = 0

    for task_id, task in sorted(tasks.items()):
        histories = sorted(by_task[task_id], key=lambda tr: int(tr["sample_index"]))
        if not histories:
            continue  # only include tasks actually sampled/generated
        seen_success = set()
        seen_repairs = set()
        first_failure: tuple[dict, int] | None = None

        def check(code: str) -> bool:
            return run_visible_tests(code, task["test_list"], task.get("test_setup_code") or "")["status"] == "passed"

        for trajectory in histories:
            sample_idx = int(trajectory["sample_index"])
            hist = trajectory.get("history", [])
            for step in hist:
                if step.get("status") == "passed":
                    code = step.get("code", "").strip()
                    if code and code not in seen_success and len(seen_success) < max_success_per_task:
                        if check(code):
                            seen_success.add(code)
                            c1.append(_new_example(task, code, "success", "model_success",
                                                   sample_index=sample_idx, attempts_used=step["attempt"],
                                                   verified_on="original MBPP visible tests only"))
                        else:
                            unverified_count += 1
                elif step.get("status") in {"syntax_error", "runtime_error", "wrong_output", "timeout"}:
                    if first_failure is None and step.get("code", "").strip():
                        first_failure = (step, sample_idx)

            for previous, nxt in zip(hist, hist[1:]):
                if previous.get("status") in {"passed", "runner_error"} or nxt.get("status") != "passed":
                    continue
                code = nxt.get("code", "").strip()
                prev_code = previous.get("code", "").strip()
                if not code or not prev_code or code == prev_code:
                    continue
                signature = (prev_code, previous.get("feedback", ""), code)
                if signature in seen_repairs or len(seen_repairs) >= max_recovery_per_task:
                    continue
                if check(code):
                    seen_repairs.add(signature)
                    self_repairs.append(_recovery_example(task, previous, code, "self_recovery", sample_idx))
                else:
                    unverified_count += 1

        reference = (task.get("code") or "").strip()
        reference_ok = bool(reference) and check(reference)
        if not seen_success and reference_ok:
            c1.append(_new_example(task, reference, "success", "verified_reference",
                                   sample_index=None, attempts_used=None,
                                   verified_on="original MBPP visible tests only"))
        if not reference_ok:
            reference_failures.append(task_id)
        # Reference correction is a separate, optional experimental data source.
        if first_failure and reference_ok and reference != first_failure[0]["code"].strip():
            ref_repairs.append(_recovery_example(task, first_failure[0], reference,
                                                 "verified_reference_correction", first_failure[1]))

    recovery = self_repairs + ref_repairs
    # Full C2 mixture is provided as a *candidate*. Match actual assistant-target
    # tokens and effective training steps against C1 in the Day 3 training notebook.
    c2_mixed = c1 + recovery
    for rows in [c1, self_repairs, ref_repairs, recovery, c2_mixed]:
        assert not ({int(r["task_id"]) for r in rows} & evalplus_ids)
        assert {int(r["task_id"]) for r in rows} <= set(tasks)

    write_jsonl(out_dir / "success_only.jsonl", c1)
    write_jsonl(out_dir / "recovery_self.jsonl", self_repairs)
    write_jsonl(out_dir / "recovery_reference.jsonl", ref_repairs)
    write_jsonl(out_dir / "recovery.jsonl", recovery)
    write_jsonl(out_dir / "c2_mixed.jsonl", c2_mixed)

    first_failure_counts = Counter()
    first_success = 0
    solved_at_three = 0
    for tr in trajectories:
        if tr.get("history"):
            first_failure_counts[tr["history"][0]["status"]] += 1
            first_success += tr["history"][0]["status"] == "passed"
            solved_at_three += bool(tr.get("solved_visible"))
    summary = {
        "model": trajectories[0]["model"] if trajectories else None,
        "num_eligible_train_tasks": len(tasks),
        "num_trajectories": len(trajectories),
        "unique_sampled_tasks": len({int(tr["task_id"]) for tr in trajectories}),
        "first_attempt_status_counts": dict(first_failure_counts),
        "first_attempt_visible_success_rate": round(first_success / len(trajectories), 4) if trajectories else None,
        "three_attempt_visible_success_rate": round(solved_at_three / len(trajectories), 4) if trajectories else None,
        "c1_success_examples": len(c1),
        "c1_source_counts": dict(Counter(e["source"] for e in c1)),
        "c2_self_recovery_examples": len(self_repairs),
        "c2_reference_recovery_examples_optional": len(ref_repairs),
        "c2_recovery_combined_examples": len(recovery),
        "c2_mixed_candidate_examples": len(c2_mixed),
        "self_recovery_failure_types": dict(Counter(e["failure_type"] for e in self_repairs)),
        "recovery_failure_types_all": dict(Counter(e["failure_type"] for e in recovery)),
        "reference_not_verified_task_ids": reference_failures,
        "rejected_model_candidates_after_recheck": unverified_count,
        "evaluation_overlap_ids": [],
        "notes": [
            "All acceptance checks use ONLY original MBPP visible tests; not hidden MBPP+ correctness.",
            "Correction provenance is recorded; model-generated and reference-based recovery are separated.",
            "Day 3 must use prompt-completion training and match assistant-target token budgets.",
            "Do not use C2 mixed rows naively without token matching/label masking.",
        ],
    }
    (out_dir / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
