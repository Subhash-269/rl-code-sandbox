import dataclasses

from rlsandbox.grader import grade_source
from rlsandbox.task import get_task


def test_timeout_gives_zero(sandbox):
    task = dataclasses.replace(get_task("rle_encode"), timeout_s=8)
    src = "def rle_encode(s):\n    while True:\n        pass\n"
    rep = grade_source(task, src, sandbox=sandbox, fresh_n=0)
    assert rep["status"] == "timeout"
    assert rep["reward"] == 0.0


def test_syntax_error_gives_zero(sandbox):
    task = get_task("parse_kv")
    rep = grade_source(task, "def parse_kv(text)\n    return {}\n", sandbox=sandbox, fresh_n=0)
    assert rep["reward"] == 0.0
    assert rep["hidden_passed"] == 0
    assert any(f["code"] == "syntax_error" for f in rep["flags"])


def test_partial_credit(sandbox):
    task = get_task("balanced_brackets")
    # Ignores bracket order, so it gets some cases right and some wrong.
    src = (
        "def is_balanced(s):\n"
        "    return all(s.count(o) == s.count(c) for o, c in ['()', '[]', '{}'])\n"
    )
    rep = grade_source(task, src, sandbox=sandbox)
    assert 0.0 < rep["reward"] < 1.0, rep
    assert rep["reward"] == rep["hidden_passed"] / rep["hidden_total"]


def test_missing_function_gives_zero(sandbox):
    rep = grade_source(get_task("merge_intervals"), "def something_else():\n    pass\n", sandbox=sandbox, fresh_n=0)
    assert rep["reward"] == 0.0
