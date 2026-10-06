"""Dependency-free test doubles. These are NOT MiniWoB or model evaluations."""
from __future__ import annotations

import random

from .types import Observation, Task


class SyntheticAgent:
    def generate(self, messages: list[dict[str, str]], *, seed: int,
                 temperature: float) -> str:
        # A fallible policy with no access to the eventual trajectory outcome.
        return "advance" if temperature == 0 or random.Random(seed).random() < 0.94 else "mistake"


class SyntheticEnvironment:
    system_prompt = "Complete the sequence. Use advance to proceed."

    def __init__(self, length: int = 8):
        self.length = length
        self.step_index = 0
        self.broken = False
        self.closed = False

    def _observation(self) -> Observation:
        done = self.step_index >= self.length
        success = done and not self.broken
        return Observation(
            text=f"Sequence progress: {self.step_index}/{self.length}. Mechanism: "
                 + ("damaged" if self.broken else "intact"),
            terminated=done, success=success, reward=float(success),
        )

    def reset(self, task: Task) -> Observation:
        self.step_index, self.broken, self.closed = 0, False, False
        return self._observation()

    def step(self, action: str) -> Observation:
        if self.closed or self.step_index >= self.length:
            raise RuntimeError("Cannot step a closed/terminal synthetic environment")
        self.broken = self.broken or action != "advance"
        self.step_index += 1
        return self._observation()

    def close(self) -> None:
        self.closed = True
