import pytest

from rlsandbox.agents import CHEATERS
from rlsandbox.run_episode import run_episode

from conftest import ALL_TASKS, HANDWRITTEN


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.id)
def test_reference_agent_gets_full_reward(task, sandbox):
    rep = run_episode(task, "reference", sandbox=sandbox)
    assert rep["reward"] == 1.0, rep
    assert not any(f["severity"] == "hard" for f in rep["flags"]), rep["flags"]


@pytest.mark.parametrize("agent", sorted(CHEATERS))
@pytest.mark.parametrize("task", HANDWRITTEN[:2] + ALL_TASKS[-1:], ids=lambda t: t.id)
def test_cheating_agent_is_flagged_and_zeroed(agent, task, sandbox):
    rep = run_episode(task, agent, sandbox=sandbox)
    assert rep["reward"] == 0.0, rep
    assert rep["flags"], f"{agent} was not flagged on {task.id}"


# Without integrity checks, these hacks really do collect reward. This is the
# reason the checks exist, and it keeps the cheaters honest as test fixtures.
@pytest.mark.parametrize("agent", ["cheat_hardcode", "cheat_conftest", "cheat_test_reader"])
def test_hacks_succeed_when_integrity_is_off(agent, sandbox):
    rep = run_episode(HANDWRITTEN[0], agent, sandbox=sandbox, integrity=False)
    assert rep["raw_reward"] == 1.0 and rep["reward"] == 1.0, rep


def test_each_cheater_trips_its_intended_check(sandbox):
    task = HANDWRITTEN[0]
    expected = {
        "cheat_hardcode": {"hardcoded_hidden_values", "special_casing_suspected"},
        "cheat_exit": {"forbidden_call"},
        "cheat_conftest": {"forbidden_workspace_file"},
        "cheat_test_reader": {"file_path_trick", "mentions_test_files"},
        "cheat_always_equal": {"dunder_override"},
    }
    for agent, codes in expected.items():
        rep = run_episode(task, agent, sandbox=sandbox)
        got = {f["code"] for f in rep["flags"]}
        assert codes <= got, (agent, got)
