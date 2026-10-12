import random


def random_inputs(rng: random.Random) -> list:
    runs = rng.randint(0, 6)
    s = "".join(rng.choice("abcxyz") * rng.randint(1, 13) for _ in range(runs))
    return [s]
