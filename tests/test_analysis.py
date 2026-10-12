import json

from rlsandbox.analysis import ablate, failures, load_records, summarize
from rlsandbox.run_episode import run_episode

from conftest import GENERATED, HANDWRITTEN


def test_show_tests_prompt_lists_hidden_cases():
    task = HANDWRITTEN[0]
    assert "## Tests" not in task.render_prompt()
    shown = task.render_prompt(show_tests=True)
    assert "## Tests" in shown
    assert json.dumps(task.hidden_cases[0]["expected"]) in shown


def test_show_tests_condition_skips_leak_check(sandbox):
    rep = run_episode(HANDWRITTEN[0], "reference", sandbox=sandbox, condition="show_tests")
    assert rep["reward"] == 1.0, rep["flags"]
    assert not any(f["check"] == "leak" for f in rep["flags"])


def test_reports_save_source_and_can_be_regraded(sandbox, tmp_path):
    out = tmp_path / "runs.jsonl"
    recs = [
        run_episode(HANDWRITTEN[0], "reference", sandbox=sandbox),
        run_episode(GENERATED[0], "reference", sandbox=sandbox),
        run_episode(HANDWRITTEN[0], "cheat_hardcode", sandbox=sandbox, condition="show_tests"),
    ]
    out.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
    records = load_records([out])
    assert all(r["source"] for r in records)

    assert "| neutral | reference |" in summarize(records)
    assert "cheat_hardcode" not in failures(records[:2])

    _, rows = ablate(records, sandbox=sandbox)
    ref, gen, cheat = rows
    assert ref["reward_static_plus_differential"] == 1.0
    assert gen["reward_static_plus_differential"] == 1.0
    assert cheat["reward_no_checks"] == 1.0
    assert cheat["reward_static_only"] == 0.0
    assert cheat["reward_differential_only"] == 0.0
