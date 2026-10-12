"""Summaries and the differential-check ablation over saved episode reports.

Reads JSONL files written by ``run_episode --out``. Nothing here calls a model:
the ablation regrades the saved solution sources, so both arms score exactly
the same solutions.

    python -m rlsandbox.analysis summarize results/llm_neutral.jsonl
    python -m rlsandbox.analysis failures  results/llm_neutral.jsonl --out results/failures.md
    python -m rlsandbox.analysis ablate    results/llm_neutral.jsonl results/llm_show.jsonl --out results/ablation.json
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from . import integrity as checks
from .generator import generate
from .grader import grade_source
from .sandbox import Sandbox, get_sandbox
from .task import Task, get_task


def load_records(paths: Iterable[Path]) -> list[dict]:
    out = []
    for p in paths:
        for line in Path(p).read_text(encoding="utf-8").splitlines():
            if line.strip():
                out.append(json.loads(line))
    return out


def resolve_task(rec: dict) -> Task:
    """Rebuild the task a record was run on (generated tasks from template and seed)."""
    if rec.get("task_source") == "generated":
        params = rec.get("task_params") or {}
        if "template" not in params or "seed" not in params:
            raise ValueError(f"record for {rec['task']} has no template/seed; rerun it with the current code")
        return generate(params["template"], params["seed"])
    return get_task(rec["task"])


def _passed(rec: dict) -> bool:
    return rec["reward"] == 1.0


# ---------------------------------------------------------------------------
# summarize: pass rate per task, flags that fired, runtime
# ---------------------------------------------------------------------------

def summarize(records: list[dict]) -> str:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for r in records:
        groups[(r.get("condition", "neutral"), r["agent"], r["task"])].append(r)

    lines = ["| condition | agent | task | passed | mean reward | mean raw reward | hard flags | soft flags | median runtime (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for (cond, agent, task), rs in sorted(groups.items()):
        hard = Counter(f["code"] for r in rs for f in r["flags"] if f["severity"] == checks.HARD)
        soft = Counter(f["code"] for r in rs for f in r["flags"] if f["severity"] == checks.SOFT)
        fmt = lambda c: ", ".join(f"{k} x{v}" for k, v in sorted(c.items())) or "-"
        lines.append(
            f"| {cond} | {agent} | {task} | {sum(map(_passed, rs))}/{len(rs)} "
            f"| {statistics.mean(r['reward'] for r in rs):.2f} | {statistics.mean(r['raw_reward'] for r in rs):.2f} "
            f"| {fmt(hard)} | {fmt(soft)} | {statistics.median(r['runtime_s'] for r in rs):.1f} |"
        )

    total = len(records)
    errors = sum(1 for r in records if r.get("agent_error"))
    sandboxes = Counter(r.get("sandbox") for r in records)
    lines += [
        "",
        f"episodes: {total}, passed: {sum(map(_passed, records))}, agent errors: {errors}, "
        f"sandbox: {dict(sandboxes)}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# failures: every solution that failed or was flagged, for reading by hand
# ---------------------------------------------------------------------------

def failures(records: list[dict]) -> str:
    out = ["# Failed or flagged episodes", ""]
    for r in records:
        if _passed(r) and not r["flags"]:
            continue
        flags = "; ".join(f"{f['severity']}:{f['code']} ({f['detail']})" for f in r["flags"]) or "none"
        out += [
            f"## {r['task']} - {r.get('condition', 'neutral')} - repeat {r.get('repeat', 0)}",
            "",
            f"- reward {r['reward']}, raw {r['raw_reward']}, hidden {r['hidden']}, fresh {r['fresh']}, status {r['status']}",
            f"- flags: {flags}",
            f"- agent error: {r.get('agent_error')}",
            "",
            "```python",
            (r.get("source") or "<source not saved>").rstrip(),
            "```",
            "",
            "Why it failed: _TODO_",
            "",
        ]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# ablate: same saved solutions, scored with and without the differential check
# ---------------------------------------------------------------------------

def ablate_record(rec: dict, sandbox: Sandbox, fresh_n: int = 10) -> dict:
    """Regrade one saved solution once, then split the result into the ablation arms.

    One sandbox run with integrity on and fresh inputs gives everything needed:
    raw_reward is the score with no checks, the non-differential hard flags are
    the static arm, and adding the differential flags gives the full arm.
    """
    if not rec.get("source"):
        raise ValueError(f"record for {rec['task']} has no saved source; rerun it with --out")
    task = resolve_task(rec)
    rep = grade_source(task, rec["source"], sandbox=sandbox, integrity=True,
                       fresh_n=fresh_n, fresh_seed=rec.get("seed", 0))
    static_hard = sorted({f["code"] for f in rep["flags"]
                          if f["severity"] == checks.HARD and f["check"] != "differential"})
    diff = [f for f in rep["flags"] if f["check"] == "differential"]
    diff_hard = any(f["severity"] == checks.HARD for f in diff)
    raw = rep["raw_reward"]
    return {
        "task": rec["task"],
        "condition": rec.get("condition", "neutral"),
        "repeat": rec.get("repeat", 0),
        "hidden": f"{rep['hidden_passed']}/{rep['hidden_total']}",
        "fresh": f"{rep['fresh_passed']}/{rep['fresh_total']}",
        "reward_no_checks": raw,
        "reward_static_only": 0.0 if static_hard else raw,
        "reward_differential_only": 0.0 if diff_hard else raw,
        "reward_static_plus_differential": 0.0 if (static_hard or diff_hard) else raw,
        "static_hard_flags": static_hard,
        "differential_flags": [f["code"] for f in diff],
    }


def ablate(records: list[dict], sandbox: Sandbox | None = None, fresh_n: int = 10) -> tuple[str, list[dict]]:
    sandbox = sandbox or get_sandbox()
    rows = [ablate_record(r, sandbox, fresh_n) for r in records]

    by_cond: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_cond[row["condition"]].append(row)

    lines = ["| condition | episodes | pass, no checks | pass, static only | pass, differential only "
             "| pass, static + differential | caught only by differential | soft fresh mismatch |",
             "|---|---|---|---|---|---|---|---|"]
    for cond, rs in sorted(by_cond.items()):
        p0 = sum(r["reward_no_checks"] == 1.0 for r in rs)
        p1 = sum(r["reward_static_only"] == 1.0 for r in rs)
        pd = sum(r["reward_differential_only"] == 1.0 for r in rs)
        p2 = sum(r["reward_static_plus_differential"] == 1.0 for r in rs)
        diff_only = sum(r["reward_static_only"] == 1.0 and r["reward_static_plus_differential"] == 0.0 for r in rs)
        soft = sum("fresh_input_mismatch" in r["differential_flags"] for r in rs)
        lines.append(f"| {cond} | {len(rs)} | {p0} | {p1} | {pd} | {p2} | {diff_only} | {soft} |")

    caught = [r for r in rows if r["reward_static_only"] == 1.0 and r["reward_static_plus_differential"] == 0.0]
    if caught:
        lines += ["", "Caught only by the differential check:"]
        lines += [f"- {r['task']} ({r['condition']}, repeat {r['repeat']}): hidden {r['hidden']}, fresh {r['fresh']}"
                  for r in caught]
    return "\n".join(lines), rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["summarize", "failures", "ablate"])
    ap.add_argument("files", nargs="+", type=Path, help="JSONL files from run_episode --out")
    ap.add_argument("--out", type=Path, help="write the output (markdown, or JSON rows for ablate) here")
    ap.add_argument("--fresh-n", type=int, default=10, help="fresh inputs per solution in the ablation")
    ap.add_argument("--sandbox", default=None, choices=["auto", "docker", "subprocess"])
    args = ap.parse_args(argv)

    records = load_records(args.files)
    if args.command == "summarize":
        text = summarize(records)
    elif args.command == "failures":
        text = failures(records)
    else:
        text, rows = ablate(records, sandbox=get_sandbox(args.sandbox), fresh_n=args.fresh_n)
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
            args.out = None  # rows saved; table goes to stdout
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
