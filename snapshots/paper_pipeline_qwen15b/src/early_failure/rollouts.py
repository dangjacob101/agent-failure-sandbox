"""Run the real episode; ask an interchangeable scorer at pre-action checkpoints."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import time
from typing import Callable

from .config import ExperimentConfig
from .history import append_turn, derive_seed, initial_messages
from .scoring import MonteCarloScorer
from .types import Agent, Environment, EpisodePrefix, PrefixScorer, Score, Task


def default_scorer(agent: Agent, environment_factory: Callable[[], Environment],
                   config: ExperimentConfig) -> MonteCarloScorer:
    return MonteCarloScorer(
        agent, environment_factory, samples=config.mc_samples,
        max_lookahead=config.mc_max_steps, temperature=config.temperature, seed=config.seed,
    )


def collect_episode(
    agent: Agent, environment_factory: Callable[[], Environment], task: Task,
    config: ExperimentConfig, monitor=None, *, scorer: PrefixScorer | None = None,
) -> dict:
    scorer = scorer if scorer is not None else default_scorer(agent, environment_factory, config)
    env = environment_factory()
    episode_id = f"{task.env_id}::seed={task.seed}"
    started = time.monotonic()
    try:
        obs = env.reset(task)
        if obs.done:
            raise ValueError(f"{episode_id}: reset returned a terminal state")
        system_prompt = env.system_prompt
        messages = initial_messages(system_prompt, obs)
        observations, actions, checks, turns = [obs], [], [], []
        for step in range(config.max_steps):
            if step % config.check_every == 0:
                # Scorers see a copy of the past, never the real episode's future.
                prefix = EpisodePrefix(task, tuple(actions), tuple(deepcopy(observations)),
                                       tuple(deepcopy(messages)), config.max_steps - step)
                check_started = time.monotonic()
                score = scorer.score(prefix)
                if not isinstance(score, Score):
                    raise TypeError("PrefixScorer.score must return a Score")
                check = {"step": step, "q": score.q, "details": score.details,
                         "elapsed_seconds": time.monotonic() - check_started}
                checks.append(check)
                if monitor is not None:
                    check["prediction"] = monitor.predict(task.env_id, [c["q"] for c in checks])

            # Main-action seeds do not depend on how many hypothetical branches ran.
            action = agent.generate(
                [dict(message) for message in messages],
                seed=derive_seed(config.seed, task.env_id, task.seed, "main", step),
                temperature=config.temperature,
            )
            obs = env.step(action)
            actions.append(action)
            observations.append(obs)
            turns.append({"step": step, "action": action, "observation": asdict(obs)})
            append_turn(messages, action, obs)
            if obs.done:
                break

        if obs.success:
            reason = "success"
        elif obs.truncated:
            reason = "environment_timeout"
        elif obs.terminated:
            reason = "environment_failure"
        else:
            reason = "step_budget_exhausted"
        return {"episode_id": episode_id, "env_id": task.env_id, "seed": task.seed,
                "system_prompt": system_prompt,
                "success": bool(obs.success), "outcome_reason": reason,
                "num_steps": len(actions), "initial_observation": asdict(observations[0]),
                "turns": turns, "checks": checks,
                "elapsed_seconds": time.monotonic() - started}
    finally:
        env.close()
