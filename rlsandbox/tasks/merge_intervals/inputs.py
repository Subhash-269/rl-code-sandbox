import random


def random_inputs(rng: random.Random) -> list:
    intervals = []
    for _ in range(rng.randint(0, 8)):
        start = rng.randint(-10, 40)
        intervals.append([start, start + rng.randint(0, 7)])
    return [intervals]
