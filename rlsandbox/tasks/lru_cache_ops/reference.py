from collections import OrderedDict


def lru_cache_ops(capacity: int, ops: list[list]) -> list[int]:
    cache: OrderedDict = OrderedDict()
    results = []
    for op in ops:
        if op[0] == "put":
            _, key, value = op
            if key in cache:
                cache.move_to_end(key)
            elif len(cache) >= capacity:
                cache.popitem(last=False)
            cache[key] = value
        else:
            key = op[1]
            if key in cache:
                cache.move_to_end(key)
                results.append(cache[key])
            else:
                results.append(-1)
    return results
