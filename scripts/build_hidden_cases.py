"""Rebuild hidden/cases.json for the handwritten tasks.

Hidden cases are a few hand-picked edge cases plus seeded random inputs, all
labelled by running the reference solution. Run from the repo root:

    python scripts/build_hidden_cases.py

Rerunning with the same SEED gives the same files.
"""

from __future__ import annotations

import importlib.util
import json
import random
from pathlib import Path

SEED = 1234
N_RANDOM = 8
TASKS_DIR = Path(__file__).resolve().parent.parent / "rlsandbox" / "tasks"

EDGE_CASES: dict[str, list[list]] = {
    "rle_encode": [["a"], ["abc"], ["zzzzzzzzzzzz"], ["aabbaa"]],
    "merge_intervals": [[[]], [[[5, 5]]], [[[1, 10], [2, 3], [4, 5]]], [[[6, 8], [1, 2], [3, 4]]]],
    "balanced_brackets": [[")("], ["((("], ["{[()()]}"], ["a(b]c"], ["no brackets"]],
    "lru_cache_ops": [
        [1, [["put", "a", 1], ["put", "b", 2], ["get", "a"], ["get", "b"]]],
        [2, [["put", "a", 1], ["put", "a", 5], ["put", "b", 2], ["put", "c", 3], ["get", "a"], ["get", "b"]]],
        [3, []],
    ],
    "parse_kv": [
        ["a=1\na=2"],
        ["  key  =  value with spaces  "],
        ["url=http://x?a=b"],
        ["=nokey\njust text\n#x=1"],
    ],
}


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    for task_dir in sorted(TASKS_DIR.iterdir()):
        meta_path = task_dir / "task.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ref = getattr(_load(task_dir / "reference.py", f"ref_{task_dir.name}"), meta["function"])
        gen = _load(task_dir / "inputs.py", f"inp_{task_dir.name}").random_inputs
        rng = random.Random(f"{SEED}:{meta['id']}")
        inputs = EDGE_CASES.get(meta["id"], []) + [gen(rng) for _ in range(N_RANDOM)]
        cases = []
        for args in inputs:
            expected = json.loads(json.dumps(ref(*json.loads(json.dumps(args)))))
            cases.append({"args": args, "expected": expected})
        out = task_dir / "hidden" / "cases.json"
        out.parent.mkdir(exist_ok=True)
        out.write_text(json.dumps(cases, indent=1) + "\n", encoding="utf-8")
        print(f"{meta['id']}: {len(cases)} hidden cases")


if __name__ == "__main__":
    main()
