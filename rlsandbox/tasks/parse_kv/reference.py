def parse_kv(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in text.split("\n"):
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key:
            result[key] = value.strip()
    return result
