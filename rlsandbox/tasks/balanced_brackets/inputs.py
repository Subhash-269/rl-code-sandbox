import random


def _balanced(rng: random.Random, depth: int) -> str:
    if depth == 0 or rng.random() < 0.3:
        return rng.choice(["", "a", "x1"])
    o, c = rng.choice(["()", "[]", "{}"])
    return o + _balanced(rng, depth - 1) + c + _balanced(rng, depth - 1)


def random_inputs(rng: random.Random) -> list:
    s = _balanced(rng, 4)
    if rng.random() < 0.5 and s:
        # Corrupt about half the cases so both outcomes are exercised.
        i = rng.randrange(len(s))
        s = s[:i] + rng.choice("()[]{}") + s[i + 1:]
    return [s]
