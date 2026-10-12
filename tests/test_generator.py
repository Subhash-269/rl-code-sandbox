import json

import pytest

from rlsandbox.generator import TEMPLATES, generate, self_check


def _snapshot(task):
    return json.dumps({
        "id": task.id, "prompt": task.prompt, "ref": task.reference_code,
        "hidden": task.hidden_cases, "fresh": task.fresh_cases(seed=3, n=5),
    }, sort_keys=True)


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_generator_is_deterministic(template):
    assert _snapshot(generate(template, seed=42)) == _snapshot(generate(template, seed=42))


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_different_seeds_give_different_tests(template):
    a, b = generate(template, seed=1), generate(template, seed=2)
    assert a.hidden_cases != b.hidden_cases


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_reference_labels_its_own_cases(template):
    task = generate(template, seed=5)
    for case in task.hidden_cases:
        assert task.run_reference(case["args"]) == case["expected"]


@pytest.mark.parametrize("template", sorted(TEMPLATES))
def test_reference_passes_own_tests_in_sandbox(template, sandbox):
    assert self_check(generate(template, seed=9), sandbox=sandbox)
