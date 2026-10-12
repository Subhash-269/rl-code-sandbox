from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskView:
    """What an honest agent is allowed to see: no hidden cases, no reference."""
    id: str
    function: str
    prompt: str
    starter_code: str


class Agent(ABC):
    name = "agent"

    @abstractmethod
    def act(self, task: TaskView, workspace: Path) -> str:
        """Return the source of solution.py. The agent may also write other files into ``workspace``."""
