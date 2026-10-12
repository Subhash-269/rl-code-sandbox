from __future__ import annotations

from .base import Agent, TaskView
from .cheating import CHEATERS, AlwaysEqualAgent, ConftestAgent, ExitAgent, HardcodeAgent, TestReaderAgent
from .llm import SYSTEMS, LLMAgent, llm_available
from .reference import ReferenceAgent

AGENT_NAMES = ["reference", *CHEATERS, "llm"]


def make_agent(name: str, task, condition: str = "neutral") -> Agent:
    """Build an agent for one task.

    Only the reference oracle and the hardcode cheater get privileged task data;
    that is what they simulate. Every agent's ``act`` receives a TaskView only.
    """
    if name == "reference":
        return ReferenceAgent(task.reference_code)
    if name == HardcodeAgent.name:
        return HardcodeAgent(task.hidden_cases)
    if name in CHEATERS:
        return CHEATERS[name]()
    if name == "llm":
        return LLMAgent(condition=condition)
    raise KeyError(f"unknown agent {name!r}; choose from {AGENT_NAMES}")


__all__ = [
    "Agent", "TaskView", "ReferenceAgent", "HardcodeAgent", "ExitAgent", "ConftestAgent",
    "TestReaderAgent", "AlwaysEqualAgent", "LLMAgent", "CHEATERS", "AGENT_NAMES",
    "make_agent", "llm_available", "SYSTEMS",
]
