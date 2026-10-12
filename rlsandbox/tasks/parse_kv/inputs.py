import random

_KEYS = ["host", "port", "user", "mode", "path", "x"]
_VALUES = ["1", "on", "a=b", "/tmp", "", "hello world"]


def random_inputs(rng: random.Random) -> list:
    lines = []
    for _ in range(rng.randint(0, 7)):
        kind = rng.random()
        if kind < 0.6:
            pad = " " * rng.randint(0, 2)
            lines.append(f"{pad}{rng.choice(_KEYS)}{pad}={pad}{rng.choice(_VALUES)}")
        elif kind < 0.75:
            lines.append("# " + rng.choice(_KEYS) + "=ignored")
        elif kind < 0.9:
            lines.append(rng.choice(["", "   ", "no equals here"]))
        else:
            lines.append(" = orphan value")
    return ["\n".join(lines)]
