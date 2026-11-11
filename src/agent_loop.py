"""Gemma 4 multi-turn function-repair loop (day-one research prototype)."""
from __future__ import annotations

import re
from typing import Any

from test_runner import run_visible_tests


def extract_python(text: str) -> str:
    """Best effort; keep a record of raw responses for audit."""
    fences = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    if fences:
        return fences[-1].strip()
    text = text.strip().replace("<|im_end|>", "").strip()
    # If Gemma adds a preface, discard it before the first code-like line.
    match = re.search(r"(?m)^(?:from\s+\S+\s+import|import\s+\S+|def\s+\w+|class\s+\w+|@\w+)", text)
    return text[match.start():].strip() if match else text


def build_initial_messages(problem: dict) -> list[dict[str, str]]:
    examples = "\n".join(problem.get("test_list", [])[:3])
    return [
        {"role": "system", "content": "Write correct Python code only. Do not include Markdown or explanations. "
            "Include all necessary imports and define the function exactly as requested."},
        {"role": "user", "content": f"Problem:\n{problem['text']}\n\nVisible tests:\n{examples}\n\nWrite Python code that passes these tests."},
    ]


def run_agent(problem: dict, generate_fn, max_attempts: int = 3, max_new_tokens: int = 384) -> dict:
    """generate_fn(messages, max_new_tokens) -> (response_text, input_tokens, output_tokens)."""
    assert 1 <= max_attempts <= 3
    messages = build_initial_messages(problem)
    history = []
    for attempt in range(1, max_attempts + 1):
        raw, prompt_tokens, completion_tokens = generate_fn(messages, max_new_tokens)
        code = extract_python(raw)
        outcome = run_visible_tests(code, problem["test_list"], problem.get("test_setup_code") or "")
        item = {
            "attempt": attempt, "code": code, "raw_output": raw,
            "status": outcome["status"], "feedback": outcome["feedback"],
            "visible_passed": outcome["passed"], "visible_total": outcome["total"],
            "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
        }
        history.append(item)
        if outcome["status"] == "passed":
            break
        messages.append({"role": "assistant", "content": code})
        messages.append({"role": "user", "content": "The previous code FAILED the visible tests.\n"
                        + outcome["feedback"] + "\nProvide a corrected full Python solution. "
                        "Do not repeat the same code. Output code only."})
    return {
        "task_id": int(problem["task_id"]), "problem": problem["text"],
        "solved_visible": history[-1]["status"] == "passed", "history": history,
        "attempts_used": len(history),
        "total_completion_tokens": sum(h["completion_tokens"] for h in history),
        "total_prompt_tokens": sum(h["prompt_tokens"] for h in history),
        "note": "Visible-test success only; not MBPP+ or hidden-test success.",
    }
