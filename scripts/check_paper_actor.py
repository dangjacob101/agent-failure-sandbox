#!/usr/bin/env python3
"""Development-only actor check: fixed tasks/seeds, no Monte Carlo rollouts."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from early_failure.config import ExperimentConfig
from early_failure.history import append_turn, initial_messages
from early_failure.miniwob import MiniWoBEnvironment
from early_failure.models import HuggingFaceAgent
from early_failure.rollouts import collect_episode
from early_failure.scoring import MonteCarloScorer
from early_failure.types import EpisodePrefix, Observation, Score, Task


PROMPT = '''Complete the browser goal using the visible elements.
Output exactly ONE action and nothing else. Use the number in square brackets as ref.
Click the button whose text matches the goal. Never click body or div elements.
Example: goal "Click Go", element [7] <button> text="Go" -> CLICK(ref=7)
For a text goal, type the exact requested text into the input, then click Submit.
Example: goal "Enter hello", element [4] <input> -> TYPE(ref=4, text="hello")
TYPE appends text. If the input already has the requested text, click Submit instead.
Allowed actions: CLICK(ref=<int>) or TYPE(ref=<int>, text="<str>").'''


class ActorOnlyScore:
    def score(self, prefix):
        return Score(0.5, {"actor_only_placeholder": True, "mc_samples": 0})


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def score_saved_enter_text(output, *, samples=8, horizon=None, episode_count=10, prefix="development_mc8"):
    """Replay only observed prefixes from a fixed initial block of development seeds."""
    config = json.loads((output / "config.json").read_text())
    metadata = json.loads((output / "metadata.json").read_text())
    if not (output / "summary.json").is_file() or (output / "error.json").exists():
        raise ValueError("Actor development must finish successfully before scoring")
    if samples < 1 or not 1 <= episode_count <= 20 or Path(prefix).name != prefix:
        raise ValueError("Invalid MC sample count, episode count, or output prefix")
    source_horizon = config["max_steps"]
    horizon = source_horizon if horizon is None else horizon
    if not 1 <= horizon <= source_horizon:
        raise ValueError("MC horizon must be positive and no longer than the recorded actor horizon")
    result_path = output / f"{prefix}.jsonl"
    if result_path.exists():
        raise ValueError(f"Refusing to overwrite {result_path}")
    prompt = config["system_prompt"]
    env_id = "miniwob/enter-text-v1"
    selected = [json.loads(line) for line in (output / "episodes.jsonl").read_text().splitlines()]
    seeds = list(range(40000, 40000 + episode_count))
    selected = [row for row in selected if row["env_id"] == env_id and row["seed"] in seeds]
    if [row["seed"] for row in selected] != seeds:
        raise ValueError("Expected every fixed development seed once, in order")
    for row in selected:
        row["source_actor_success"] = row["success"]
        row["source_actor_num_steps"] = row["num_steps"]
        row["turns"] = row["turns"][:horizon]
        row["num_steps"] = len(row["turns"])
        last = Observation(**row["turns"][-1]["observation"])
        row["success"] = last.success
        row["outcome_reason"] = ("success" if last.success else "environment_timeout" if last.truncated
                                 else "environment_failure" if last.terminated else "step_budget_exhausted")
    agent = HuggingFaceAgent(
        config["model_id"], revision=config["model_revision"], device=config["device"], dtype=config["dtype"],
        max_new_tokens=config["max_new_tokens"], max_context_tokens=config["max_context_tokens"],
    )

    def environment_factory():
        env = MiniWoBEnvironment()
        env.system_prompt = prompt
        return env

    scorer = MonteCarloScorer(agent, environment_factory, samples=samples, max_lookahead=None,
                             temperature=config["temperature"], seed=config["seed"])
    started, scored = time.monotonic(), []
    try:
        if agent.provenance() != metadata["agent"]:
            raise ValueError("Model runtime differs from actor collection")
        write_json(output / f"{prefix}_metadata.json", {
            "development_only": True, "samples": samples, "max_steps": horizon,
            "source_actor_max_steps": source_horizon,
            "temperature": config["temperature"], "master_seed": config["seed"],
            "actor_metadata": metadata, "task_seeds": seeds,
            "prefix_input": "Recorded actions/observations strictly before the checkpoint; no actual future or outcome.",
        })
        with result_path.open("w") as stream:
            for episode in selected:
                episode_started = time.monotonic()
                observations = [Observation(**episode["initial_observation"])]
                messages = initial_messages(prompt, observations[0])
                actions, checks = [], []
                task = Task(episode["env_id"], episode["seed"])
                for step, turn in enumerate(episode["turns"]):
                    state = EpisodePrefix(task, tuple(actions), tuple(observations), tuple(messages),
                                          horizon - step)
                    check_started = time.monotonic()
                    score = scorer.score(state)
                    check = {"step": step, "q": score.q, "details": score.details,
                             "elapsed_seconds": time.monotonic() - check_started}
                    checks.append(check)
                    print(json.dumps({"mc_seed": task.seed, "step": step, "q": score.q,
                                      "elapsed_seconds": check["elapsed_seconds"]}), flush=True)
                    # Reveal the recorded next turn only after this prefix was scored.
                    action, observation = turn["action"], Observation(**turn["observation"])
                    actions.append(action)
                    observations.append(observation)
                    append_turn(messages, action, observation)
                result = {**episode, "checks": checks, "mc_elapsed_seconds": time.monotonic() - episode_started}
                scored.append(result)
                stream.write(json.dumps(result, allow_nan=False) + "\n")
                stream.flush()
                print(json.dumps({"mc_episode_completed": task.seed, "success": episode["success"],
                                  "q": [check["q"] for check in checks],
                                  "elapsed_seconds": result["mc_elapsed_seconds"]}), flush=True)
                if len(scored) == 2:
                    print(json.dumps({"first_two_episode_seconds": sum(row["mc_elapsed_seconds"] for row in scored),
                                      "projected_total_episode_seconds": episode_count / 2 * sum(row["mc_elapsed_seconds"] for row in scored)}), flush=True)
        by_step = {}
        for step in range(horizon):
            groups = {}
            for label in ("success", "failure"):
                qs = [check["q"] for row in scored if row["success"] == (label == "success")
                      for check in row["checks"] if check["step"] == step]
                groups[label] = {"n": len(qs), "q": qs, "zeros": qs.count(0),
                                 "mean": sum(qs) / len(qs) if qs else None,
                                 "histogram": {str(q): n for q, n in sorted(Counter(qs).items())}}
            by_step[str(step)] = groups
        diagnostic = {"development_only": True, "episodes": len(scored), "samples": samples,
                      "max_steps": horizon, "source_actor_max_steps": source_horizon,
                      "successes": sum(row["success"] for row in scored),
                      "failures": sum(not row["success"] for row in scored),
                      "elapsed_seconds": time.monotonic() - started, "by_step": by_step,
                      "generated_actions": sum(check["details"]["generated_actions"] for row in scored for check in row["checks"])}
        write_json(output / f"{prefix}_diagnostics.json", diagnostic)
        print(json.dumps(diagnostic), flush=True)
    except BaseException as error:
        write_json(output / f"{prefix}_error.json", {
            "type": type(error).__name__, "message": str(error),
            "note": "MC development incomplete; actor episodes remain unchanged.",
        })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/paper_actor_development"))
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-0.5B-Instruct")
    parser.add_argument("--revision", help="Hub commit; a missing revision is resolved before downloading weights")
    parser.add_argument("--score-saved-enter-text", type=Path, help="MC8-score the first ten saved development episodes")
    parser.add_argument("--mc-samples", type=int, default=8)
    parser.add_argument("--mc-horizon", type=int)
    parser.add_argument("--mc-episodes", type=int, default=10)
    parser.add_argument("--mc-output-prefix", default="development_mc8")
    args = parser.parse_args()
    if args.score_saved_enter_text:
        score_saved_enter_text(args.score_saved_enter_text, samples=args.mc_samples, horizon=args.mc_horizon,
                              episode_count=args.mc_episodes, prefix=args.mc_output_prefix)
        return
    output = args.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    revision = args.revision
    if revision is None and args.model_id == "Qwen/Qwen2.5-Coder-0.5B-Instruct":
        revision = "ea3f2471cf1b1f0db85067f1ef93848e38e88c25"
    if revision is None:
        from huggingface_hub import HfApi
        revision = HfApi().model_info(args.model_id, token=False).sha
    print(json.dumps({"model_id": args.model_id, "pinned_revision": revision}), flush=True)
    config = ExperimentConfig(
        model_id=args.model_id, model_revision=revision,
        device="mps", dtype="float16", max_steps=4, temperature=0.7,
        max_new_tokens=48, max_context_tokens=8192, seed=42,
        env_ids=("miniwob/enter-text-v1", "miniwob/click-button-v1"),
        calibration_seeds=tuple(range(40000, 40020)), evaluation_seeds=(99999,),
        output_dir=str(output),
    )
    settings = asdict(config)
    settings.update(purpose="development_only_actor_feasibility", system_prompt=PROMPT,
                    development_seeds=list(range(40000, 40020)), monte_carlo_enabled=False,
                    placeholder_scores_are_not_predictions=True)
    settings.pop("calibration_seeds")
    settings.pop("evaluation_seeds")
    write_json(output / "config.json", settings)
    (output / "system_prompt.txt").write_text(PROMPT + "\n")
    prompt_hash = hashlib.sha256(PROMPT.encode()).hexdigest()
    agent = HuggingFaceAgent(
        config.model_id, device=config.device, dtype=config.dtype,
        revision=config.model_revision, max_new_tokens=config.max_new_tokens,
        max_context_tokens=config.max_context_tokens,
    )

    def environment_factory():
        env = MiniWoBEnvironment()
        env.system_prompt = PROMPT
        return env

    started, episodes = time.monotonic(), []
    try:
        write_json(output / "metadata.json", {
            "agent": agent.provenance(), "prompt_sha256": prompt_hash,
            "versions": {name: importlib.metadata.version(name)
                         for name in ("torch", "transformers", "miniwob", "selenium")},
            "development_only": True,
        })
        with (output / "episodes.jsonl").open("w") as stream:
            for env_id in config.env_ids:
                task_episodes = []
                for seed in range(40000, 40020):
                    episode = collect_episode(agent, environment_factory, Task(env_id, seed), config,
                                              scorer=ActorOnlyScore())
                    episode["split"] = "development"
                    task_episodes.append(episode)
                    episodes.append(episode)
                    stream.write(json.dumps(episode, allow_nan=False) + "\n")
                    stream.flush()
                    print(json.dumps({"env_id": env_id, "seed": seed, "success": episode["success"],
                                      "num_steps": episode["num_steps"],
                                      "actions": [turn["action"] for turn in episode["turns"]]}), flush=True)
                    if len(task_episodes) in (10, 20):
                        print(json.dumps({"checkpoint": len(task_episodes), "env_id": env_id,
                                          "successes": sum(row["success"] for row in task_episodes),
                                          "failures": sum(not row["success"] for row in task_episodes)}), flush=True)
        summary = {
            "development_only": True, "monte_carlo_enabled": False,
            "prompt_sha256": prompt_hash, "elapsed_seconds": time.monotonic() - started,
            "by_environment": {env_id: {
                "episodes": sum(row["env_id"] == env_id for row in episodes),
                "successes": sum(row["success"] for row in episodes if row["env_id"] == env_id),
                "failures": sum(not row["success"] for row in episodes if row["env_id"] == env_id),
                "unique_initial_observations": len({row["initial_observation"]["text"]
                                                    for row in episodes if row["env_id"] == env_id}),
                "first_10_outcomes": [row["success"] for row in episodes if row["env_id"] == env_id][:10],
            } for env_id in config.env_ids},
        }
        write_json(output / "summary.json", summary)
        print(json.dumps(summary), flush=True)
    except BaseException as error:
        write_json(output / "error.json", {"type": type(error).__name__, "message": str(error),
                                          "note": "Incomplete development run; infrastructure errors are not task failures."})
        raise


if __name__ == "__main__":
    main()
