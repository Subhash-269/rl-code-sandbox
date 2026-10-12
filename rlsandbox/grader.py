"""Grade a solution against hidden tests inside the sandbox.

Steps, in order:

1. Static integrity checks on the solution source and the workspace files.
2. Create a fresh grading directory. Copy the agent's workspace files into it
   (everything except PROMPT.md), then write solution.py.
3. In a separate step, write the hidden cases and the generated test file into
   the grading directory. Fresh differential cases are written the same way.
4. Run pytest in the sandbox and read per-test results from the JUnit XML.
5. reward = hidden tests passed / hidden tests total, zeroed on any hard flag.

The reward is computed from the per-test JUnit records and the known number of
hidden cases, never from pytest's exit code.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from . import integrity as checks
from .sandbox import Sandbox, get_sandbox
from .task import Task

TEST_TEMPLATE = '''\
import copy
import json
import pathlib

import pytest

_CASES = json.loads((pathlib.Path(__file__).parent / "{cases_file}").read_text(encoding="utf-8"))


def _norm(x):
    # JSON round trip: rejects objects with a custom __eq__ and normalizes tuples to lists.
    return json.loads(json.dumps(x, sort_keys=True))


@pytest.mark.parametrize("i", range(len(_CASES)), ids=["{prefix}%d" % i for i in range(len(_CASES))])
def {test_name}(i):
    from solution import {function} as fn
    case = _CASES[i]
    got = fn(*copy.deepcopy(case["args"]))
    assert _norm(got) == _norm(case["expected"])
'''


def _write_tests(grade_dir: Path, task: Task, cases: list[dict], kind: str) -> None:
    cases_file = f"{kind}_cases.json"
    (grade_dir / cases_file).write_text(json.dumps(cases), encoding="utf-8")
    (grade_dir / f"test_{kind}.py").write_text(
        TEST_TEMPLATE.format(cases_file=cases_file, prefix=kind[0], test_name=f"test_{kind}", function=task.function),
        encoding="utf-8",
    )


def parse_junit(path: Path) -> dict[str, str]:
    """Map test name (for example 'test_hidden[h3]') to 'passed', 'failed', 'error' or 'skipped'."""
    results: dict[str, str] = {}
    root = ET.parse(path).getroot()
    for tc in root.iter("testcase"):
        tags = {child.tag for child in tc}
        if "error" in tags:
            outcome = "error"
        elif "failure" in tags:
            outcome = "failed"
        elif "skipped" in tags:
            outcome = "skipped"
        else:
            outcome = "passed"
        results[tc.get("name", "")] = outcome
    return results


def _count(results: dict[str, str], test_name: str, prefix: str, n: int) -> int:
    return sum(results.get(f"{test_name}[{prefix}{i}]") == "passed" for i in range(n))


def grade_source(
    task: Task,
    source: str,
    workspace: Path | None = None,
    sandbox: Sandbox | None = None,
    integrity: bool = True,
    fresh_n: int = 10,
    fresh_seed: int = 0,
    extra_flags: list | None = None,
    keep_dir: bool = False,
) -> dict[str, Any]:
    """Grade ``source`` for ``task``. Returns a JSON-serializable report."""
    sandbox = sandbox or get_sandbox()
    start = time.monotonic()
    flags: list[checks.Flag] = list(extra_flags or [])
    if integrity:
        flags += checks.static_check(task, source, workspace)

    grade_dir = Path(tempfile.mkdtemp(prefix=f"grade_{task.id}_"))
    try:
        # Step 2: agent files only.
        if workspace is not None:
            for p in workspace.rglob("*"):
                rel = p.relative_to(workspace)
                if p.is_file() and rel.as_posix() != "PROMPT.md":
                    dest = grade_dir / rel
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(p, dest)
        (grade_dir / "solution.py").write_text(source, encoding="utf-8")

        # Step 3: hidden and fresh tests, added only now.
        fresh = task.fresh_cases(seed=fresh_seed, n=fresh_n) if fresh_n else []
        _write_tests(grade_dir, task, task.hidden_cases, "hidden")
        if fresh:
            _write_tests(grade_dir, task, fresh, "fresh")

        # Step 4: run.
        result = sandbox.run_pytest(grade_dir, timeout_s=task.timeout_s)
        report_xml = grade_dir / "report.xml"
        results: dict[str, str] = {}
        if result.timed_out:
            status = "timeout"
        elif not report_xml.exists():
            status = "no_report"
        else:
            try:
                results = parse_junit(report_xml)
                status = "ok"
            except ET.ParseError:
                status = "bad_report"
    finally:
        if not keep_dir:
            shutil.rmtree(grade_dir, ignore_errors=True)

    n_hidden = len(task.hidden_cases)
    hidden_passed = _count(results, "test_hidden", "h", n_hidden)
    fresh_passed = _count(results, "test_fresh", "f", len(fresh)) if fresh else None
    raw_reward = hidden_passed / n_hidden if n_hidden else 0.0
    fresh_frac = (fresh_passed / len(fresh)) if fresh else None
    if status == "ok" and hidden_passed < n_hidden and any(v == "error" for v in results.values()):
        status = "errors"

    if integrity:
        flags += checks.differential_flags(raw_reward, fresh_frac, len(fresh))

    hard = checks.has_hard(flags)
    return {
        "task": task.id,
        "reward": 0.0 if hard else raw_reward,
        "raw_reward": raw_reward,
        "hidden_passed": hidden_passed,
        "hidden_total": n_hidden,
        "fresh_passed": fresh_passed,
        "fresh_total": len(fresh),
        "status": status,
        "integrity_checks": integrity,
        "hard_flagged": hard,
        "flags": [f.to_dict() for f in flags],
        "sandbox": sandbox.name,
        "sandbox_isolation": sandbox.isolation,
        "sandbox_result": result.to_dict(),
        "grade_runtime_s": round(time.monotonic() - start, 3),
        "grade_dir": str(grade_dir) if keep_dir else None,
    }

