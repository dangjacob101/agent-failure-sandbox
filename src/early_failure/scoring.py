"""Monte Carlo scoring by checked replay of an observed episode prefix."""
from __future__ import annotations

import math
from typing import Callable

from .history import append_turn, derive_seed, replay_signature
from .types import Agent, Environment, EpisodePrefix, Score


class ReplayMismatch(RuntimeError):
    """A branch failed to recreate the observed prefix; its score is invalid."""


class MonteCarloScorer:
    """Estimate success within the remaining budget using independent branches.

    A shorter lookahead counts unfinished branches as censored. Replay and
    infrastructure errors abort scoring instead of becoming failure labels.
    """

    def __init__(
        self, agent: Agent, environment_factory: Callable[[], Environment], *,
        samples: int = 4, max_lookahead: int | None = None,
        temperature: float = 0.7, seed: int = 42,
    ):
        if not isinstance(samples, int) or isinstance(samples, bool) or samples < 1:
            raise ValueError("samples must be a positive integer")
        if max_lookahead is not None and (
            not isinstance(max_lookahead, int) or isinstance(max_lookahead, bool)
            or max_lookahead < 1
        ):
            raise ValueError("max_lookahead must be None or a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.agent = agent
        self.environment_factory = environment_factory
        self.samples = samples
        self.max_lookahead = max_lookahead
        self.temperature = temperature
        self.seed = seed

    def score(self, prefix: EpisodePrefix) -> Score:
        """Replay past actions, then sample only hypothetical future actions."""
        if (prefix.remaining_steps < 1 or len(prefix.observations) != prefix.step + 1
                or prefix.observations[-1].done):
            raise ValueError("MC scoring requires a nonterminal prefix with remaining budget")
        budget = prefix.remaining_steps
        if self.max_lookahead is not None:
            budget = min(budget, self.max_lookahead)
        successes, censored, generated = 0, 0, 0
        for branch in range(self.samples):
            env = self.environment_factory()
            try:
                obs = env.reset(prefix.task)
                if replay_signature(obs) != replay_signature(prefix.observations[0]):
                    raise ReplayMismatch(f"{prefix.task}: reset differs on branch {branch}")
                for index, action in enumerate(prefix.actions):
                    obs = env.step(action)
                    if replay_signature(obs) != replay_signature(prefix.observations[index + 1]):
                        raise ReplayMismatch(f"{prefix.task}: replay differs after action {index + 1}")
                messages = [dict(message) for message in prefix.messages]
                for continuation in range(budget):
                    action = self.agent.generate(
                        [dict(message) for message in messages],
                        seed=derive_seed(self.seed, prefix.task.env_id, prefix.task.seed,
                                         "mc", prefix.step, branch, continuation),
                        temperature=self.temperature,
                    )
                    generated += 1
                    obs = env.step(action)
                    append_turn(messages, action, obs)
                    if obs.done:
                        break
                successes += int(obs.success)
                # A full remaining budget is an observed bounded-horizon failure.
                censored += int(not obs.done and budget < prefix.remaining_steps)
            finally:
                env.close()
        return Score(q=successes / self.samples, details={
            "mc_successes": successes, "mc_samples": self.samples,
            "mc_censored": censored, "lookahead_steps": budget,
            "generated_actions": generated,
        })
