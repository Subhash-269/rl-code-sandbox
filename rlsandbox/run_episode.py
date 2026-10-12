"""Run episodes: task -> agent writes a solution in a fresh workspace -> grader -> report.

Examples:

    python -m rlsandbox.run_episode --task all --agent reference
    python -m rlsandbox.run_episode --task rle_encode --agent cheat_conftest --no-integrity
    python -m rlsandbox.run_episode --generated 3 --seed 7 --agent reference
    python -m rlsandbox.run_episode --task all --agent llm --repeats 3 --out results/llm_neutral.jsonl
    python -m rlsandbox.run_episode --task all --agent llm --repeats 3 --condition show_tests --out results/llm_show.jsonl

Prints one JSON report per episode (one per line with --jsonl). Reports written
with --out also carry the solution source and the raw model reply, so a run can
be regraded offline later (see ``python -m rlsandbox.analysis``).

Conditions:

    neutral     the agent sees the prompt and public examples only (default)
    show_tests  the prompt also lists the hidden cases it is graded on, and the
                LLM is told it is scored only on them. The leak check is skipped
                because the exposure is deliberate. Fresh inputs then act as a
                held-out set that measures overfitting to the visible tests.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import integrity as checks
from .agents import AGENT_NAMES, SYSTEMS, TaskView, make_agent
from .generator import generate_many
from .grader import grade_source
from .sandbox import Sandbox, get_sandbox
from .task import Task, create_workspace, load_tasks, remove_dir

CONDITIONS = sorted(SYSTEMS)


def run_episode(task: Task, agent_name: str, sandbox: Sandbox | None = None, integrity: bool = True,
                fresh_n: int = 10, seed: int = 0, condition: str = "neutral", repeat: int = 0) -> dict:
    if condition not in SYSTEMS:
        raise ValueError(f"unknown condition {condition!r}; choose from {CONDITIONS}")
    show_tests = condition == "show_tests"
    sandbox = sandbox or get_sandbox()
    start = time.monotonic()
    workspace = create_workspace(task, show_tests=show_tests)
    try:
        prompt = task.render_prompt(show_tests=show_tests)
        # Leak check runs on the environment's own output, before the agent touches anything.
        # Under show_tests the hidden cases are in the prompt on purpose, so it is skipped.
        pre_flags = checks.leak_check(task, workspace, prompt) if integrity and not show_tests else []
        agent = make_agent(agent_name, task, condition=condition)
        view = TaskView(id=task.id, function=task.function, prompt=prompt, starter_code=task.starter_code)
        error = None
        try:
            source = agent.act(view, workspace)
        except Exception as e:  # an agent failure is a zero-reward episode, not a crash
            source, error = task.starter_code, f"{type(e).__name__}: {e}"
        (workspace / "solution.py").write_text(source, encoding="utf-8")
        report = grade_source(task, source, workspace=workspace, sandbox=sandbox, integrity=integrity,
                              fresh_n=fresh_n, fresh_seed=seed, extra_flags=pre_flags)
    finally:
        remove_dir(workspace)
    return {
        "task": task.id,
        "task_source": task.source,
        "task_params": task.params,
        "condition": condition,
        "repeat": repeat,
        "seed": seed,
        "agent": getattr(agent, "name", agent_name),
        "reward": report["reward"],
        "raw_reward": report["raw_reward"],
        "hidden": f"{report['hidden_passed']}/{report['hidden_total']}",
        "fresh": f"{report['fresh_passed']}/{report['fresh_total']}" if report["fresh_total"] else None,
        "status": report["status"],
        "agent_error": error,
        "flags": report["flags"],
        "integrity_checks": integrity,
        "sandbox": report["sandbox"],
        "grade_runtime_s": report["grade_runtime_s"],
        "runtime_s": round(time.monotonic() - start, 3),
        "source": source,
        "raw_response": getattr(agent, "last_raw", None),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task", default="all", help="task id, comma-separated ids, or 'all' (handwritten tasks)")
    ap.add_argument("--generated", type=int, default=0, help="also run N generated tasks")
    ap.add_argument("--agent", default="reference", choices=AGENT_NAMES)
    ap.add_argument("--sandbox", default=None, choices=["auto", "docker", "subprocess"])
    ap.add_argument("--no-integrity", action="store_true", help="disable integrity checks (for ablations)")
    ap.add_argument("--fresh-n", type=int, default=10, help="fresh inputs for the differential check (0 disables)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--condition", default="neutral", choices=CONDITIONS,
                    help="what the agent sees; show_tests lists the graded cases in the prompt")
    ap.add_argument("--repeats", type=int, default=1, help="episodes per task")
    ap.add_argument("--jsonl", action="store_true", help="one compact JSON object per line")
    ap.add_argument("--out", type=Path, help="append JSONL reports (including solution source) to this file")
    args = ap.parse_args(argv)

    tasks: list[Task] = []
    if args.task and args.task != "none":
        all_tasks = load_tasks()
        if args.task == "all":
            tasks = all_tasks
        else:
            wanted = args.task.split(",")
            by_id = {t.id: t for t in all_tasks}
            missing = [w for w in wanted if w not in by_id]
            if missing:
                ap.error(f"unknown task(s) {missing}; available: {sorted(by_id)}")
            tasks = [by_id[w] for w in wanted]
    tasks += generate_many(args.generated, seed=args.seed)

    sandbox = get_sandbox(args.sandbox)
    if sandbox.name == "subprocess":
        print("warning: Docker not in use; running with the WEAK subprocess fallback", file=sys.stderr)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        for r in range(args.repeats):
            rep = run_episode(task, args.agent, sandbox=sandbox, integrity=not args.no_integrity,
                              fresh_n=args.fresh_n, seed=args.seed, condition=args.condition, repeat=r)
            # Keep the terminal readable; the full record (with source) goes to --out.
            shown = {k: v for k, v in rep.items() if k not in ("source", "raw_response")}
            print(json.dumps(shown) if args.jsonl else json.dumps(shown, indent=2), flush=True)
            if args.out:
                with args.out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rep) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
