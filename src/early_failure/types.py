"""Small interfaces shared by the model, benchmark, and experiment runner."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from numbers import Real
from typing import Any, Protocol


@dataclass(frozen=True)
class Task:
    env_id: str
    seed: int


@dataclass
class Observation:
    text: str
    terminated: bool = False
    truncated: bool = False
    success: bool = False
    reward: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated


class Agent(Protocol):
    """Generate from the supplied history and seed without retaining branch state."""

    def generate(self, messages: list[dict[str, str]], *, seed: int,
                 temperature: float) -> str: ...


class Environment(Protocol):
    """One episode at a time; return a new observation snapshot on each step."""

    system_prompt: str

    def reset(self, task: Task) -> Observation: ...
    def step(self, action: str) -> Observation: ...
    def close(self) -> None: ...


@dataclass(frozen=True)
class EpisodePrefix:
    """Only information available before the next real action."""

    task: Task
    actions: tuple[str, ...]
    observations: tuple[Observation, ...]
    messages: tuple[dict[str, str], ...]
    remaining_steps: int

    @property
    def step(self) -> int:
        return len(self.actions)


@dataclass(frozen=True)
class Score:
    """A success-oriented score; optional diagnostics stay separate from q."""

    q: float
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if (isinstance(self.q, bool) or not isinstance(self.q, Real)
                or not math.isfinite(self.q) or not 0 <= self.q <= 1):
            raise ValueError("q must be a finite number in [0, 1]")
        object.__setattr__(self, "q", float(self.q))
        if not isinstance(self.details, dict):
            raise ValueError("Score details must be a dictionary")


class PrefixScorer(Protocol):
    """Score the observed prefix without changing the agent or real environment."""

    def score(self, prefix: EpisodePrefix) -> Score: ...


class FeatureExtractor(Protocol):
    """Read model activations from a supplied prefix without generating actions."""

    def extract_features(self, messages: list[dict[str, str]], *,
                         layers: tuple[int, ...]) -> dict[int, list[float]]: ...
