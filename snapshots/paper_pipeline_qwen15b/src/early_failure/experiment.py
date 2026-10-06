"""Public parent function: collect calibration, fit, monitor held-out episodes."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import importlib.metadata
import json
from pathlib import Path
import platform
from typing import Callable
import warnings

from .config import ExperimentConfig
from .conformal import TrajectoryConformal
from .metrics import summarize
from .rollouts import collect_episode, default_scorer
from .types import Agent, Environment, PrefixScorer, Task


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _versions() -> dict:
    versions = {"python": platform.python_version()}
    for name in ("early-failure-sandbox", "miniwob", "gymnasium", "selenium", "torch", "transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return versions


def _provenance(agent: Agent, environment_factory: Callable[[], Environment],
                config: ExperimentConfig, scorer: PrefixScorer) -> dict:
    model_info = agent.provenance() if hasattr(agent, "provenance") else {
        "implementation": config.agent_tag or config.agent_backend
    }
    environment_info = {"implementation": config.environment_tag or config.environment}
    env = environment_factory()
    try:
        # Adapters may need reset before reporting their resolved runtime.
        if hasattr(env, "provenance"):
            env.reset(Task(config.env_ids[0], config.calibration_seeds[0]))
            environment_info.update(env.provenance())
    finally:
        env.close()
    scorer_info = scorer.provenance() if hasattr(scorer, "provenance") else {
        "implementation": config.scorer_tag or "monte-carlo-v1"
    }
    return {"versions": _versions(), "agent": model_info,
            "environment": environment_info, "scorer": scorer_info}


def _build_components(config, agent, environment_factory):
    """Resolve defaults here; the episode loop knows only the interfaces."""
    if agent is None:
        if config.agent_backend == "synthetic":
            from .synthetic import SyntheticAgent
            agent = SyntheticAgent()
        else:
            from .models import HuggingFaceAgent
            agent = HuggingFaceAgent(
                config.model_id, device=config.device, dtype=config.dtype,
                max_new_tokens=config.max_new_tokens, max_context_tokens=config.max_context_tokens,
                revision=config.model_revision,
            )
    if environment_factory is None:
        if config.environment == "synthetic":
            from .synthetic import SyntheticEnvironment
            environment_factory = lambda: SyntheticEnvironment(length=config.max_steps)
        else:
            from .miniwob import MiniWoBEnvironment
            environment_factory = lambda: MiniWoBEnvironment(**config.env_kwargs)
    return agent, environment_factory


def _collect_split(path, split, seeds, agent, environment_factory, config, scorer, monitor=None):
    """Flush each completed episode so interrupted runs retain their evidence."""
    episodes = []
    with path.open("w") as stream:
        for env_id in config.env_ids:
            for seed in seeds:
                print(f"{split}: {env_id} seed={seed}", flush=True)
                episode = collect_episode(agent, environment_factory, Task(env_id, seed),
                                          config, monitor, scorer=scorer)
                episodes.append(episode)
                stream.write(json.dumps(episode, allow_nan=False) + "\n")
                stream.flush()
    return episodes


def run_experiment(
    config: ExperimentConfig | None = None, *, agent: Agent | None = None,
    environment_factory: Callable[[], Environment] | None = None,
    scorer: PrefixScorer | None = None, **overrides,
) -> dict:
    """Collect, calibrate, and evaluate. Override settings or inject components independently."""
    config = replace(config or ExperimentConfig(), **overrides)
    if agent is not None and not config.agent_tag:
        raise ValueError("Set agent_tag to a stable implementation/version ID for an injected agent")
    if environment_factory is not None and not config.environment_tag:
        raise ValueError("Set environment_tag for an injected environment factory")
    if scorer is not None and not config.scorer_tag:
        raise ValueError("Set scorer_tag to a stable implementation/version ID for an injected scorer")
    if config.environment == "custom" and environment_factory is None:
        raise ValueError("Custom environments require environment_factory")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Choose a new run directory.")

    # Check reused calibration BEFORE importing model dependencies or downloading weights.
    saved = None
    if config.calibration_path:
        saved = json.loads(Path(config.calibration_path).read_text())
        if saved.get("schema_version") != 1:
            raise ValueError("Unsupported calibration artifact schema")
        if saved.get("protocol_fingerprint") != config.fingerprint():
            raise ValueError("Calibration protocol mismatch: model, environment, horizon, or scoring settings changed")
        expected_keys = {(env, seed) for env in config.env_ids for seed in config.calibration_seeds}
        old_keys = {(x["env_id"], x["seed"]) for x in saved["calibration_tasks"]}
        if old_keys != expected_keys:
            raise ValueError("calibration_seeds must match the saved calibration task manifest")
        if old_keys & {(env, seed) for env in config.env_ids for seed in config.evaluation_seeds}:
            raise ValueError("Evaluation task seeds overlap saved calibration")

    agent, environment_factory = _build_components(config, agent, environment_factory)
    scorer = scorer if scorer is not None else default_scorer(agent, environment_factory, config)
    synthetic = config.environment == "synthetic" or config.agent_backend == "synthetic"
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / "config.json", config.to_dict())
    monitor = TrajectoryConformal(alpha=config.alpha)
    try:
        runtime = _provenance(agent, environment_factory, config, scorer)
        _write_json(output / "metadata.json", {"runtime": runtime, "protocol_fingerprint": config.fingerprint(),
                                             "synthetic": synthetic})
        if saved is not None and saved.get("runtime") != runtime:
            raise ValueError("Calibration runtime mismatch: resolved model revision, device, precision, browser, or dependencies changed")
        local_or_unversioned = "resolved_revision" in runtime["agent"] and (
            Path(config.model_id).exists() or not runtime["agent"]["resolved_revision"]
        )
        if saved is not None and local_or_unversioned and not config.agent_tag:
            raise ValueError("Calibration reuse for an unversioned/local model requires agent_tag tied to checkpoint contents")
        if saved is None:
            calibration = _collect_split(output / "calibration_episodes.jsonl", "calibration",
                                         config.calibration_seeds, agent, environment_factory, config, scorer)
            monitor.fit(calibration)
            saved = {"schema_version": 1, "protocol": config.protocol(), "runtime": runtime,
                     "protocol_fingerprint": config.fingerprint(),
                     "calibration_tasks": [{"env_id": e["env_id"], "seed": e["seed"]} for e in calibration],
                     "monitor": monitor.to_dict()}
        else:
            monitor = TrajectoryConformal.from_dict({**saved["monitor"], "alpha": config.alpha})
            saved = {**saved, "monitor": monitor.to_dict()}
        _write_json(output / "calibration.json", saved)
        diagnostics = monitor.diagnostics()
        _write_json(output / "calibration_diagnostics.json", diagnostics)
        for env_id, diagnostic in diagnostics["environments"].items():
            for label in ("success", "failure"):
                if not diagnostic[label]["supported"]:
                    warnings.warn(f"{env_id}: no {label} calibration episodes; that class cannot be rejected")
                elif not diagnostic[label]["can_reject_at_alpha"]:
                    warnings.warn(f"{env_id}: too few {label} calibration episodes to reject at alpha={config.alpha}")
        if config.scorer_tag is None and config.temperature == 0 and config.mc_samples > 1:
            warnings.warn("Greedy deterministic continuations repeat the same rollout; use temperature > 0 for MC diversity")
        evaluation = _collect_split(output / "evaluation_episodes.jsonl", "evaluation",
                                    config.evaluation_seeds, agent, environment_factory, config, scorer, monitor)
        grouped = defaultdict(list)
        for episode in evaluation:
            grouped[episode["env_id"]].append(episode)
        result = {"synthetic": synthetic, "output_dir": str(output.resolve()),
                  "metrics": summarize(evaluation),
                  "by_environment": {key: summarize(rows) for key, rows in grouped.items()},
                  "calibration_diagnostics": diagnostics}
        _write_json(output / "summary.json", result)
        return result
    except Exception as error:
        _write_json(output / "error.json", {"type": type(error).__name__, "message": str(error),
                                           "note": "Run incomplete; infrastructure errors are not task failure labels."})
        raise
