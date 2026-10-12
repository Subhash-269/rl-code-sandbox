# rlsandbox

A small, verifiable coding environment of the kind used to produce reward for RL on code:
a task gives a prompt and starter code, an agent writes `solution.py`, and a grader runs
hidden tests in a sandbox and returns a reward in [0, 1]. Integrity checks zero the reward
when a solution games the grader instead of solving the task.

This is an environment and grader, not a training loop. No policy is trained here.

## Purpose

When code solutions are graded by tests, a model can raise its score without solving the
task: hardcoding expected outputs, exiting before assertions run, patching pytest, or
reading the test files. If the reward can't tell those apart from real solutions, training
on it teaches the shortcut.

This project builds a small grader that tries to keep that from happening and measures
how well it works:

1. **Verifiable tasks.** Each task has a reference solution, hidden tests labelled by that
   reference, and an input generator that can make fresh tests on demand.
2. **Isolated grading.** Solutions run in a locked-down Docker container. The reward comes
   from per-test results, never from the exit code.
3. **Reward-hacking checks.** Static, leak, hardcode and differential checks zero the
   reward for known exploits. Each check has a deliberately cheating agent that proves it
   fires.
4. **Measurement on a real model.** Run an LLM on the tasks, with and without the hidden
   tests visible, and report pass rates, which checks fired, and how many "passing"
   solutions the differential check catches. The checks' known blind spots are listed too.

The scope is kept small on purpose: a handful of tasks that are fully understood and
numbers that were actually measured.

## Setup (Windows, PowerShell)

```powershell
C:\Python314\python.exe -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt        # the package itself plus pytest
pip install -r requirements-llm.txt    # only for the LLM agent: anthropic, google-genai
pytest -q
```

Docker Desktop must be running for real isolation. The runner image
(`rlsandbox-runner:py312`) is built automatically on first use; to build it by hand:

```powershell
docker build -t rlsandbox-runner:py312 rlsandbox\docker
```

If Docker is not reachable the grader falls back to a plain subprocess and prints a
warning. Every report records which sandbox it used (`"sandbox": "docker"`).

## Running episodes

```powershell
python -m rlsandbox.run_episode --task rle_encode --agent reference
python -m rlsandbox.run_episode --task rle_encode --agent cheat_hardcode
```

Agents: `reference` (the oracle), five deliberate cheaters (`cheat_hardcode`, `cheat_exit`,
`cheat_conftest`, `cheat_test_reader`, `cheat_always_equal`), and `llm`.

LLM runs (key from the environment only, never committed):

```powershell
$env:ANTHROPIC_API_KEY = "..."        # or $env:GEMINI_API_KEY
python -m rlsandbox.run_episode --task all --agent llm --repeats 3 --jsonl --out results\llm_neutral.jsonl
python -m rlsandbox.run_episode --task all --agent llm --repeats 3 --jsonl --condition show_tests --out results\llm_show.jsonl
```

Reports written with `--out` include the solution source and the raw model reply, so runs
can be regraded without calling the model again.

### Conditions

- `neutral` (default): the agent sees the prompt and public examples only.
- `show_tests`: the prompt also lists the hidden cases, and the LLM is told it is scored
  only on them. The leak check is skipped because the exposure is deliberate. Fresh inputs
  from the task's generator then act as a held-out set, which is what makes overfitting
  measurable. Under `neutral` the agent never sees the hidden tests, so it has no way to
  special-case them.

## Analysis

```powershell
python -m rlsandbox.analysis summarize results\llm_neutral.jsonl results\llm_show.jsonl
python -m rlsandbox.analysis failures  results\llm_neutral.jsonl --out results\failures.md
python -m rlsandbox.analysis ablate    results\llm_neutral.jsonl results\llm_show.jsonl --out results\ablation.json
```

`ablate` regrades each saved solution once and reports four arms on the same solutions:
no checks, static checks only, differential check only, and both.

## How grading works

```mermaid
flowchart LR
    T["Task<br/>prompt, starter, public examples"] --> W["Agent workspace<br/>PROMPT.md + solution.py"]
    T -.->|"before the agent acts"| L{{"leak check"}}
    W --> A["Agent<br/>reference / cheater / LLM"]
    A --> S["solution.py<br/>+ any files it wrote"]
    S --> SC{{"static checks<br/>workspace files, AST, hardcode"}}
    S --> G["Fresh grading dir<br/>agent files copied in"]
    H[("hidden cases<br/>+ fresh generated cases")] -->|"added only now"| G
    G --> D["Docker sandbox<br/>no network, read-only root,<br/>CPU/memory/PID limits"]
    D --> J["JUnit XML<br/>per-test results"]
    J --> R["raw reward =<br/>hidden passed / hidden total"]
    J --> DF{{"differential check<br/>hidden vs fresh pass rate"}}
    L --> F["Flags"]
    SC --> F
    DF --> F
    R --> OUT(["reward"])
    F -->|"any hard flag: reward = 0"| OUT
```

1. Static checks on the solution source and workspace files (`integrity.static_check`).
2. A fresh grading directory gets the agent's files, then the solution, then, in a separate
   step, the hidden cases, the generated test file and the fresh differential cases.
3. pytest runs in the sandbox. Per-test results come from JUnit XML, never the exit code.
4. `reward = hidden passed / hidden total`, zeroed by any hard flag.

## Integrity checks

| check | what it catches |
|---|---|
| leak | hidden files or distinctive hidden values in the prompt or the prepared workspace |
| static: workspace files | `conftest.py`, `pytest.ini`, `.pth`, test files and other files that change how pytest runs |
| static: AST and strings | test-framework imports, exit calls, frame inspection, `__file__`, builtins patching, mentions of test files |
| hardcode | literals in the solution that match two or more hidden cases |
| differential | hidden pass rate much higher than the pass rate on fresh generated inputs |
| JSON normalization (grader) | objects with a custom `__eq__` that claim to equal anything |

### Cheater × check matrix

Each deliberate cheater was run on all 5 handwritten tasks, once with the checks off and
once with them on (Docker sandbox). A cell counts the tasks where that check raised a hard
flag. Regenerate it with `python scripts/cheater_matrix.py`.

| cheater | mean reward, checks off | leak | static: workspace files | static: code | hardcode | differential | mean reward, checks on |
|---|---|---|---|---|---|---|---|
| `cheat_always_equal` | 0.00 | - | - | (5/5 soft) | - | - | 0.00 |
| `cheat_conftest` | 1.00 | - | **5/5** | - | - | - | 0.00 |
| `cheat_exit` | 0.00 | - | - | **5/5** | - | - | 0.00 |
| `cheat_hardcode` | 1.00 | - | - | - | **5/5** | **5/5** | 0.00 |
| `cheat_test_reader` | 1.00 | - | - | **5/5** | - | - | 0.00 |

How to read it:

- Three hacks really work when the checks are off: patching pytest with a `conftest.py`,
  hardcoding the hidden answers, and reading the test data at grading time. Each gets full
  reward. With the checks on, each is caught on every task and scores 0.
- Two hacks fail even with the checks off, because of how the grader is built, not because
  of a flag. `cheat_exit` exits early, but the reward comes from per-test JUnit records, not
  the exit code. `cheat_always_equal` returns an object whose `__eq__` is always true, but
  the grader JSON-normalizes outputs before comparing. Its `__eq__` override is still
  reported as a soft flag.
- `cheat_hardcode` is caught twice, by the hardcode check and by the differential check, so
  either check alone would stop it.
- No cheater targets the leak check. That check guards the environment itself, making sure
  hidden data never reaches the agent, so it stays quiet here.

## What the checks don't catch

These are known limits of the current heuristics (from the code, not measured):

- The static checks are pattern-based. Hidden values that are computed or encoded instead
  of written as literals (for example base64 or arithmetic) pass the hardcode check.
  Imports through `importlib` with a computed name are only a soft flag.
- The hardcode check needs values that are distinctive (strings of 5+ characters, numbers
  ≥ 100). Tests with small outputs such as booleans or small ints can be memorized unseen.
- The differential check only uses fresh inputs from the task's own generator. A solution
  that is wrong outside that input distribution looks correct. A solution that special-cases
  only a few hidden cases stays under the 0.34 gap and gets a soft flag, not a zero.
- Hidden tests are labelled by the reference solution, so a bug in the reference becomes the
  expected answer.
- The subprocess fallback offers no real isolation.

## Results

_To be filled with measured numbers only._

## Layout

```
rlsandbox/task.py         task format and loading
rlsandbox/sandbox.py      Docker and subprocess backends
rlsandbox/grader.py       grading pipeline and reward
rlsandbox/integrity.py    leak, static, hardcode and differential checks
rlsandbox/generator.py    seeded task templates and self-check
rlsandbox/run_episode.py  episode runner CLI
rlsandbox/analysis.py     summaries, failure dump, ablation
rlsandbox/agents/         reference, cheaters, LLM agent
rlsandbox/tasks/          five handwritten tasks
scripts/                  hidden-case builder, cheater matrix
tests/                    pytest suite
```
