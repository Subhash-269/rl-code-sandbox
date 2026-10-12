"""Verifiable task generation from parameterized templates.

Each template turns a seeded RNG into a Task: it picks parameters, writes the
prompt and the reference solution with those parameters baked in, and labels
randomized hidden inputs by running the reference. Same seed, same task.

``self_check`` grades the reference against its own hidden tests in the
sandbox, which is the minimum bar for a generated task to be usable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

from .task import Task


@dataclass
class Template:
    name: str
    build: Callable[[random.Random], Task]


def _finish(task: Task, rng: random.Random, n_hidden: int) -> Task:
    for _ in range(n_hidden):
        args = task.input_gen(rng)
        task.hidden_cases.append({"args": args, "expected": task.run_reference(args)})
    return task


def _sum_divisible(rng: random.Random) -> Task:
    k = rng.randint(2, 9)
    fn = f"sum_divisible_by_{k}"

    def gen(r: random.Random) -> list:
        return [[r.randint(-200, 200) for _ in range(r.randint(0, 12))]]

    return Task(
        id=f"gen_sum_divisible_k{k}",
        function=fn,
        prompt=f"Return the sum of the integers in the list that are divisible by {k}. "
        "Negative numbers count too. An empty list sums to 0.",
        starter_code=f"def {fn}(nums: list[int]) -> int:\n    raise NotImplementedError\n",
        reference_code=f"def {fn}(nums):\n    return sum(x for x in nums if x % {k} == 0)\n",
        public_examples=[{"args": [[k, k + 1, 2 * k]], "expected": 3 * k}, {"args": [[]], "expected": 0}],
        hidden_cases=[],
        input_gen=gen,
        source="generated",
        params={"template": "sum_divisible", "k": k},
    )


def _rotate_left(rng: random.Random) -> Task:
    k = rng.randint(1, 5)
    fn = f"rotate_left_{k}"

    def gen(r: random.Random) -> list:
        return [[r.randint(0, 99) for _ in range(r.randint(0, 9))]]

    return Task(
        id=f"gen_rotate_left_k{k}",
        function=fn,
        prompt=f"Rotate the list left by {k} positions and return a new list. If the list is "
        f"shorter than {k}, rotate by {k} modulo its length. An empty list stays empty.",
        starter_code=f"def {fn}(xs: list[int]) -> list[int]:\n    raise NotImplementedError\n",
        reference_code=(
            f"def {fn}(xs):\n"
            "    if not xs:\n"
            "        return []\n"
            f"    s = {k} % len(xs)\n"
            "    return xs[s:] + xs[:s]\n"
        ),
        public_examples=[{"args": [list(range(1, k + 3))], "expected": list(range(k + 1, k + 3)) + list(range(1, k + 1))}],
        hidden_cases=[],
        input_gen=gen,
        source="generated",
        params={"template": "rotate_left", "k": k},
    )


def _count_long_words(rng: random.Random) -> Task:
    n = rng.randint(2, 6)
    fn = f"count_words_longer_than_{n}"
    vocab = ["a", "to", "the", "word", "apple", "banana", "kiwi", "strawberry", "sky", "planet"]

    def gen(r: random.Random) -> list:
        return [" ".join(r.choice(vocab) for _ in range(r.randint(0, 10)))]

    return Task(
        id=f"gen_count_long_words_n{n}",
        function=fn,
        prompt=f"Count the words in the text that have more than {n} characters. Words are "
        "separated by whitespace.",
        starter_code=f"def {fn}(text: str) -> int:\n    raise NotImplementedError\n",
        reference_code=f"def {fn}(text):\n    return sum(1 for w in text.split() if len(w) > {n})\n",
        public_examples=[{"args": [""], "expected": 0}],
        hidden_cases=[],
        input_gen=gen,
        source="generated",
        params={"template": "count_long_words", "n": n},
    )


TEMPLATES: dict[str, Template] = {
    t.name: t
    for t in [
        Template("sum_divisible", _sum_divisible),
        Template("rotate_left", _rotate_left),
        Template("count_long_words", _count_long_words),
    ]
}


def generate(template: str, seed: int, n_hidden: int = 12) -> Task:
    """Build one task from a template. Deterministic for a given (template, seed)."""
    rng = random.Random(f"{template}:{seed}")
    task = TEMPLATES[template].build(rng)
    task.id = f"{task.id}_s{seed}"
    task.params["seed"] = seed
    return _finish(task, rng, n_hidden)


def generate_many(n: int, seed: int = 0, n_hidden: int = 12) -> list[Task]:
    names = sorted(TEMPLATES)
    return [generate(names[i % len(names)], seed + i, n_hidden) for i in range(n)]


def self_check(task: Task, sandbox=None) -> bool:
    """True if the reference solution scores 1.0 on its own hidden tests in the sandbox."""
    from .grader import grade_source

    report = grade_source(task, task.reference_code, sandbox=sandbox, integrity=False, fresh_n=0)
    return report["reward"] == 1.0
