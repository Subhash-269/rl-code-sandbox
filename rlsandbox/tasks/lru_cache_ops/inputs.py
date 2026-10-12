import random


def random_inputs(rng: random.Random) -> list:
    capacity = rng.randint(1, 3)
    ops = []
    for _ in range(rng.randint(0, 15)):
        key = rng.choice("abcde")
        if rng.random() < 0.5:
            ops.append(["put", key, rng.randint(0, 99)])
        else:
            ops.append(["get", key])
    return [capacity, ops]
