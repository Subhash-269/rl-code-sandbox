import pytest

from rlsandbox.generator import generate_many
from rlsandbox.sandbox import get_sandbox
from rlsandbox.task import load_tasks


def pytest_report_header(config):
    sb = get_sandbox()
    return f"rlsandbox sandbox backend: {sb.name} ({sb.isolation})"


@pytest.fixture(scope="session")
def sandbox():
    return get_sandbox()


HANDWRITTEN = load_tasks()
GENERATED = generate_many(3, seed=11)
ALL_TASKS = HANDWRITTEN + GENERATED
