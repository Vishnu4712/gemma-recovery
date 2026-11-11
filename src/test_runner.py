"""Small, resource-limited function-code runner for Kaggle Day 1.

SECURITY: subprocess + resource limits are NOT a strong sandbox. Model-generated
code can still access local files or network unless the external runtime forbids
it. Never run with secrets, private data or elevated access. For a public-facing
benchmark or adversarial model, run in a truly isolated container/service.
"""
from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

# Entire driver runs in a separate Python interpreter; no generated code is exec'd
# in the notebook kernel. Child stdout is reserved for a JSON response.
_CHILD = r'''
import ast, contextlib, io, json, sys, traceback
payload = json.load(sys.stdin)
source = payload["code"]
checks = payload["tests"]
setup = payload.get("setup", "") or ""

def trace(e):
    return "".join(traceback.format_exception_only(type(e), e)).strip()[:1800]

namespace = {"__name__": "__main__"}
result = {"status": "passed", "passed": 0, "total": len(checks), "feedback": "All visible tests passed", "failures": []}

try:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        exec(compile(source, "<candidate>", "exec"), namespace)
        if setup:
            exec(compile(setup, "<setup>", "exec"), namespace)
except BaseException as e:
    result["status"] = "syntax_error" if isinstance(e, SyntaxError) else "runtime_error"
    result["feedback"] = f"Failed before tests: {trace(e)}"
else:
    for test in checks:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                # For simple equality assertions, extract actual/expected values
                # to provide useful execution feedback without reevaluating them.
                tree = ast.parse(test, mode="exec")
                if (len(tree.body) == 1 and isinstance(tree.body[0], ast.Assert) and
                    isinstance(tree.body[0].test, ast.Compare) and
                    len(tree.body[0].test.ops) == 1 and
                    isinstance(tree.body[0].test.ops[0], ast.Eq)):
                    cmp = tree.body[0].test
                    actual = eval(compile(ast.Expression(cmp.left), "<actual>", "eval"), namespace)
                    expected = eval(compile(ast.Expression(cmp.comparators[0]), "<expected>", "eval"), namespace)
                    if actual != expected:
                        raise AssertionError(f"expected {repr(expected)[:220]}, got {repr(actual)[:220]}")
                else:
                    exec(compile(tree, "<test>", "exec"), namespace)
            result["passed"] += 1
        except BaseException as e:
            kind = "wrong_output" if isinstance(e, AssertionError) else "runtime_error"
            result["status"] = kind if result["status"] == "passed" else result["status"]
            result["failures"].append({"test": test[:500], "kind": kind, "message": trace(e)})
            # feedback limited to first failing test to avoid lengthy prompts
            if len(result["failures"]) >= 2:
                break
    if result["failures"]:
        result["feedback"] = "\n".join(
            f"Test: {f['test']}\nError: {f['message']}" for f in result["failures"]
        )[:1600]
print(json.dumps(result, ensure_ascii=False))
'''


def _limits():
    # POSIX-only; Kaggle is Linux. Limits accidental infinite loops and output size.
    resource.setrlimit(resource.RLIMIT_CPU, (3, 4))
    resource.setrlimit(resource.RLIMIT_AS, (1024 * 1024**2, 1024 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def run_visible_tests(code: str, tests: list[str], setup: str = "", timeout_s: int = 6) -> dict:
    """Execute candidate and visible asserts with timeout.

    Avoid passing hidden benchmark tests here: they must not feed back to the model.
    """
    payload = json.dumps({"code": code, "tests": list(tests), "setup": setup})
    # Pass a small environment, with no authentication tokens inherited.
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"),
           "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
    with tempfile.TemporaryDirectory(prefix="gemma_code_") as temp:
        try:
            cp = subprocess.run(
                [sys.executable, "-I", "-S", "-B", "-c", _CHILD],
                input=payload, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, cwd=temp,
                env=env, preexec_fn=_limits, timeout=timeout_s,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "passed": 0, "total": len(tests),
                    "feedback": f"Execution timed out after {timeout_s} seconds", "failures": []}
        if cp.returncode != 0:
            return {"status": "runtime_error", "passed": 0, "total": len(tests),
                    "feedback": f"Runner failed: {cp.stderr[-700:]} (exit={cp.returncode})", "failures": []}
        try:
            output = json.loads(cp.stdout)
            assert output["status"] in {"passed", "wrong_output", "runtime_error", "syntax_error"}
            return output
        except (ValueError, KeyError, AssertionError):
            return {"status": "runner_error", "passed": 0, "total": len(tests),
                    "feedback": f"Malformed runner result: {cp.stdout[-500:]}", "failures": []}


if __name__ == "__main__":
    tests = ["assert find_max([1,5,3]) == 5", "assert find_max([-4, -2]) == -2"]
    wrong = run_visible_tests("def find_max(a):\n    return min(a)\n", tests)
    right = run_visible_tests("def find_max(a):\n    return max(a)\n", tests)
    print("Wrong:", wrong)
    print("Right:", right)
    assert wrong["status"] == "wrong_output"
    assert right["status"] == "passed"
