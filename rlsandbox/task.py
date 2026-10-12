"""Task format and loading.

A task is one folder under ``rlsandbox/tasks/<task_id>/`` with:

    task.json        id, function name, prompt, public examples, timeout
    starter.py       starter code the agent sees
    reference.py     reference solution (never shown to the agent)
    inputs.py        ``random_inputs(rng) -> list`` used by the differential check
    hidden/cases.json  hidden test cases: [{"args": [...], "expected": ...}, ...]

Only the prompt, the public examples and the starter code ever reach the
agent's workspace. Hidden cases and the reference stay in the task folder and
are copied into a separate grading directory at grading time.
"""

from __future__ import annotations

import importlib.util
import json
import random
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

TASKS_DIR = Path(__file__).parent / "tasks"

InputGen = Callable[[random.Random], list]


@dataclass
class Task:
    id: str
    function: str
    prompt: str
    starter_code: str
    reference_code: str
    public_examples: list[dict[str, Any]]
    hidden_cases: list[dict[str, Any]]
    input_gen: InputGen
    timeout_s: float = 20.0
    source: str = "handwritten"
    params: dict[str, Any] = field(default_factory=dict)

    def render_prompt(self, show_tests: bool = False) -> str:
        """The full text the agent sees: description plus public examples.

        ``show_tests`` also lists the hidden cases. That is only for the
        visible-tests condition, where the point is to give the agent a chance
        to overfit and measure it on fresh inputs.
        """
        lines = [f"# Task: {self.id}", "", self.prompt.strip(), ""]
        lines.append(f"Implement `{self.function}` in solution.py.")
        if self.public_examples:
            lines += ["", "## Examples", ""]
            for ex in self.public_examples:
                args = ", ".join(json.dumps(a) for a in ex["args"])
                lines.append(f"    {self.function}({args}) == {json.dumps(ex['expected'])}")
        if show_tests:
            lines += ["", "## Tests", "", "Your solution is graded on exactly these cases:", ""]
            for case in self.hidden_cases:
                args = ", ".join(json.dumps(a) for a in case["args"])
                lines.append(f"    {self.function}({args}) == {json.dumps(case['expected'])}")
        return "\n".join(lines) + "\n"

    def run_reference(self, args: list) -> Any:
        """Run the trusted reference in-process and return a JSON-normalized result."""
        ns: dict[str, Any] = {}
        exec(compile(self.reference_code, f"<reference:{self.id}>", "exec"), ns)
        result = ns[self.function](*json.loads(json.dumps(args)))
        return json.loads(json.dumps(result))

    def fresh_cases(self, seed: int, n: int = 10) -> list[dict[str, Any]]:
        """New cases from the input generator, labelled by the reference."""
        rng = random.Random(seed)
        cases = []
        for _ in range(n):
            args = self.input_gen(rng)
            cases.append({"args": args, "expected": self.run_reference(args)})
        return cases


def _load_input_gen(path: Path) -> InputGen:
    spec = importlib.util.spec_from_file_location(f"rlsandbox_inputs_{path.parent.name}", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.random_inputs


def load_task(task_dir: Path) -> Task:
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    return Task(
        id=meta["id"],
        function=meta["function"],
        prompt=meta["prompt"],
        starter_code=(task_dir / "starter.py").read_text(encoding="utf-8"),
        reference_code=(task_dir / "reference.py").read_text(encoding="utf-8"),
        public_examples=meta.get("public_examples", []),
        hidden_cases=json.loads((task_dir / "hidden" / "cases.json").read_text(encoding="utf-8")),
        input_gen=_load_input_gen(task_dir / "inputs.py"),
        timeout_s=float(meta.get("timeout_s", 20.0)),
    )


def load_tasks() -> list[Task]:
    return [load_task(d) for d in sorted(TASKS_DIR.iterdir()) if (d / "task.json").exists()]


def get_task(task_id: str) -> Task:
    for t in load_tasks():
        if t.id == task_id:
            return t
    raise KeyError(f"unknown task {task_id!r}")


def create_workspace(task: Task, root: Path | None = None, show_tests: bool = False) -> Path:
    """Fresh agent workspace: PROMPT.md and solution.py (starter). Nothing hidden unless ``show_tests``."""
    ws = Path(tempfile.mkdtemp(prefix=f"ws_{task.id}_", dir=root))
    (ws / "PROMPT.md").write_text(task.render_prompt(show_tests=show_tests), encoding="utf-8")
    (ws / "solution.py").write_text(task.starter_code, encoding="utf-8")
    return ws


def remove_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
