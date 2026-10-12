from __future__ import annotations

from pathlib import Path

from .base import Agent, TaskView


class ReferenceAgent(Agent):
    """Oracle that returns the reference solution. A sanity check for the pipeline, not a real agent."""
    name = "reference"

    def __init__(self, reference_code: str):
        self._code = reference_code

    def act(self, task: TaskView, workspace: Path) -> str:
        return self._code
