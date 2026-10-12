"""Print the cheater x check matrix used in the README.

Runs every deliberate cheater on every handwritten task twice, once with the
integrity checks off and once with them on. The cheaters are deterministic and
need no API key. Run from the repo root:

    python scripts/cheater_matrix.py

Each check cell counts the tasks on which that check raised a hard flag
(soft flags are shown in parentheses).
"""

from __future__ import annotations

from rlsandbox.agents import CHEATERS
from rlsandbox.run_episode import run_episode
from rlsandbox.sandbox import get_sandbox
from rlsandbox.task import load_tasks

# Flag code -> column. Codes not listed fall into "static: code".
FAMILIES = {
    "hidden_file_in_workspace": "leak",
    "hidden_value_leaked": "leak",
    "forbidden_workspace_file": "static: workspace files",
    "extra_workspace_file": "static: workspace files",
    "hardcoded_hidden_values": "hardcode",
    "possible_hardcoded_value": "hardcode",
    "special_casing_suspected": "differential",
    "fresh_input_mismatch": "differential",
}
COLUMNS = ["leak", "static: workspace files", "static: code", "hardcode", "differential"]


def main() -> int:
    sandbox = get_sandbox()
    tasks = load_tasks()
    n = len(tasks)
    header = ["cheater", "mean reward, checks off", *COLUMNS, "mean reward, checks on"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name in sorted(CHEATERS):
        off = [run_episode(t, name, sandbox=sandbox, integrity=False) for t in tasks]
        on = [run_episode(t, name, sandbox=sandbox, integrity=True) for t in tasks]
        hard = {c: 0 for c in COLUMNS}
        soft = {c: 0 for c in COLUMNS}
        for rep in on:
            fired_hard, fired_soft = set(), set()
            for f in rep["flags"]:
                col = FAMILIES.get(f["code"], "static: code")
                (fired_hard if f["severity"] == "hard" else fired_soft).add(col)
            for c in fired_hard:
                hard[c] += 1
            for c in fired_soft - fired_hard:
                soft[c] += 1

        def cell(c: str) -> str:
            parts = [f"**{hard[c]}/{n}**"] if hard[c] else []
            if soft[c]:
                parts.append(f"({soft[c]}/{n} soft)")
            return " ".join(parts) or "-"

        r_off = sum(r["reward"] for r in off) / n
        r_on = sum(r["reward"] for r in on) / n
        lines.append(f"| `{name}` | {r_off:.2f} | " + " | ".join(cell(c) for c in COLUMNS) + f" | {r_on:.2f} |")
    lines.append("")
    lines.append(f"{n} handwritten tasks per cheater, sandbox: {sandbox.name}.")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
