"""Paper sequence: MC state scores, conformal labels, activations, linear probes."""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time

from .config import ExperimentConfig
from .experiment import _build_components, _collect_split, _provenance, _write_json
from .features import extract_feature_records, label_counts
from .probe_metrics import summarize_probe_outcomes
from .probes import LinearProbeBank
from .rollouts import default_scorer
from .step_labeling import MODES, StepwiseLabeler


@dataclass(frozen=True)
class ProbeConfig:
    training_seeds: tuple[int, ...] = tuple(range(2000, 2050))
    layers: tuple[int, ...] = (6, 12, 18, 24)
    primary_layer: int = 12
    labeling_mode: str = "paper_equations"
    min_per_class: int = 2
    stop_if_untrainable: bool = True
    feature_extractor_tag: str | None = None

    def __post_init__(self):
        for name in ("training_seeds", "layers"):
            values = tuple(getattr(self, name))
            object.__setattr__(self, name, values)
            minimum = 1 if name == "layers" else 0
            if not values or len(set(values)) != len(values) or any(type(x) is not int or x < minimum for x in values):
                raise ValueError(f"{name} must contain unique integers >= {minimum}")
        if type(self.primary_layer) is not int or self.primary_layer not in self.layers:
            raise ValueError("primary_layer must be one of the requested layers")
        if self.labeling_mode not in MODES:
            raise ValueError(f"labeling_mode must be one of {MODES}")
        if type(self.min_per_class) is not int or self.min_per_class < 1:
            raise ValueError("min_per_class must be a positive integer")
        if type(self.stop_if_untrainable) is not bool:
            raise ValueError("stop_if_untrainable must be boolean")


def _outcomes(episodes):
    successes = sum(episode["success"] for episode in episodes)
    return {"episodes": len(episodes), "success": successes, "failure": len(episodes) - successes}


def _viable_steps(labeler, config):
    """Check attainability before spending compute on training/evaluation data."""
    viable = []
    calibration = labeler.to_dict()["calibration_rewards"]
    for env in config.env_ids:
        for step in range(0, config.max_steps, config.check_every):
            # Class rank decisions change only at observed rewards and between ties.
            groups = calibration.get(env, {}).get(str(step), {})
            values = sorted({0.0, 1.0, *(q for qs in groups.values() for q in qs)})
            candidates = values + [(left + right) / 2 for left, right in zip(values, values[1:])]
            targets = {labeler.predict(env, step, q)["target"] for q in candidates}
            if {0, 1}.issubset(targets):
                viable.append({"env_id": env, "step": step})
    return viable


def _write_report(output, result):
    lines = ["Paper-pipeline sandbox", f'Status: {result["status"]}',
             f'Labeling: {result["labeling_mode"]}; alpha={result["alpha"]}',
             "Calibration, probe training, and evaluation use separate task seeds."]
    for split, values in result["outcomes"].items():
        lines.append(f'{split}: {values["success"]} success, {values["failure"]} failure ({values["episodes"]} runs).')
    if result.get("reason"):
        lines.append(result["reason"])
    if "probe_evaluation" in result:
        evaluation = result["probe_evaluation"]
        lines.extend([f'Trained probes: {result["trained_probes"]}',
                      f'Conformal-label accuracy: {evaluation["metrics"]["accuracy"]} '
                      f'on {evaluation["n_evaluated"]} labeled state/layer pairs.',
                      f'Conformal label coverage: {evaluation["label_coverage"]}.',
                      "Repeated layers/checks are correlated; pooled probe accuracy is descriptive."])
        outcomes = result["outcome_evaluation"]
        primary = outcomes["primary"]
        lines.extend([f'Primary layer: {outcomes["primary_layer"]} (fixed before testing).',
                      f'Failures warned about: {primary["detected_failure_episodes"]}/{primary["failed_episodes"]}.',
                      f'Successful runs wrongly warned about: {primary["false_alarm_episodes"]}/{primary["successful_episodes"]}.',
                      f'Runs with a supported probe prediction: {primary["supported_episodes"]}/{primary["episodes"]}.',
                      f'Median warning lead among detected failures: {primary["median_failure_lead_steps"]} actions.',
                      "Per-layer results, baselines, and episode-level confidence intervals are in summary.json."])
    lines.extend(["Probe targets are conformal state labels, not real final task outcomes.",
                  "Final-outcome warning performance is evaluated separately.",
                  "A linear direction is predictive association, not evidence of a causal failure mechanism.",
                  "Paper stepwise guarantees do not imply a bound on any probe alarm across an entire episode.",
                  "This is a small-model/MiniWoB adaptation; the actor is not the paper's fine-tuned Llama agent."])
    _write_json(output / "summary.json", result)
    (output / "report.txt").write_text("\n".join(lines) + "\n")


def run_paper_experiment(config: ExperimentConfig | None = None, *, probe_config: ProbeConfig | None = None,
                         agent=None, environment_factory=None, scorer=None, feature_extractor=None,
                         **overrides):
    """Run three independent data splits, fit probes, and test both kinds of target.

    Model/environment/scoring parameters use ExperimentConfig. ProbeConfig selects
    layers, training seeds, and the explicitly named labeling rule.
    """
    config = replace(config or ExperimentConfig(), **overrides)
    settings = probe_config or ProbeConfig()
    if config.mc_max_steps is not None and config.mc_max_steps < config.max_steps:
        raise ValueError("The paper pipeline requires full remaining-horizon continuations; remove mc_max_steps")
    if config.calibration_path:
        raise ValueError("The paper pipeline collects fresh timestep calibration; legacy calibration_path is incompatible")
    manifests = {"calibration": config.calibration_seeds, "training": settings.training_seeds,
                 "evaluation": config.evaluation_seeds}
    if len(set().union(*map(set, manifests.values()))) != sum(map(len, manifests.values())):
        raise ValueError("calibration, training, and evaluation seeds must be disjoint")
    for component, tag, name in ((agent, config.agent_tag, "agent_tag"),
                                  (environment_factory, config.environment_tag, "environment_tag"),
                                  (scorer, config.scorer_tag, "scorer_tag"),
                                  (feature_extractor, settings.feature_extractor_tag, "feature_extractor_tag")):
        if component is not None and not tag:
            raise ValueError(f"Set {name} for an injected implementation")
    if config.environment == "custom" and environment_factory is None:
        raise ValueError("custom environments require environment_factory")
    output = Path(config.output_dir)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    _write_json(output / "config.json", config.to_dict())
    _write_json(output / "probe_config.json", asdict(settings))
    source_dir = Path(__file__).parent
    _write_json(output / "source_snapshot.json", {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(source_dir.glob("*.py"))})
    try:
        agent, environment_factory = _build_components(config, agent, environment_factory)
        scorer = scorer if scorer is not None else default_scorer(agent, environment_factory, config)
        feature_extractor = feature_extractor if feature_extractor is not None else agent
        if not callable(getattr(feature_extractor, "extract_features", None)):
            raise ValueError("feature_extractor must expose extract_features(messages, *, layers)")
        metadata = {"pipeline": "paper_state_labels_and_linear_probes_v1",
                    "protocol_fingerprint": config.fingerprint(),
                    "probe_config": asdict(settings), "system_prompt_source": "recorded separately in each episode after reset",
                    "task_manifests": {split: [{"env_id": env, "seed": seed} for env in config.env_ids for seed in seeds]
                                       for split, seeds in manifests.items()},
                    "runtime": _provenance(agent, environment_factory, config, scorer),
                    "feature_convention": "HF hidden_states[layer], final nonpadding prompt token; layers 1..N",
                    "evaluation_mode": "offline replay of pre-action prefixes; no outcome or MC branch text in features",
                    "synthetic": config.environment == "synthetic"}
        metadata["runtime"]["versions"]["scikit-learn"] = importlib.metadata.version("scikit-learn")
        _write_json(output / "metadata.json", metadata)
        episodes = {}
        episodes["calibration"] = _collect_split(
            output / "calibration_episodes.jsonl", "calibration", config.calibration_seeds,
            agent, environment_factory, config, scorer)
        labeler = StepwiseLabeler(settings.labeling_mode, config.alpha).fit(episodes["calibration"])
        _write_json(output / "step_calibration.json", labeler.to_dict())
        _write_json(output / "calibration_diagnostics.json", labeler.diagnostics())
        viable = _viable_steps(labeler, config)
        result = {"status": "insufficient_calibration", "labeling_mode": settings.labeling_mode,
                  "alpha": config.alpha, "outcomes": {"calibration": _outcomes(episodes["calibration"])},
                  "viable_steps": viable, "trained_probes": 0}
        if not viable and settings.stop_if_untrainable:
            result.update(reason="No timestep can produce both probe target classes. Training/evaluation were not collected.",
                          elapsed_seconds=time.monotonic() - started)
            _write_report(output, result)
            return result
        episodes["training"] = _collect_split(
            output / "training_episodes.jsonl", "training", settings.training_seeds,
            agent, environment_factory, config, scorer)
        cache = {}
        training = extract_feature_records(episodes["training"], labeler, feature_extractor, layers=settings.layers,
                                           output_path=output / "training_features.jsonl", cache=cache)
        bank = LinearProbeBank(seed=config.seed, min_per_class=settings.min_per_class).fit(training)
        artifact = bank.to_dict()
        _write_json(output / "probes.json", artifact)
        trained = sum(item["status"] == "trained" for item in artifact["probes"])
        result.update(status="insufficient_training_labels", trained_probes=trained,
                      outcomes={split: _outcomes(items) for split, items in episodes.items()},
                      training_labels=label_counts(episodes["training"], labeler))
        if not trained and settings.stop_if_untrainable:
            result.update(reason="No timestep/layer has enough examples of both conformal labels to train a probe.",
                          elapsed_seconds=time.monotonic() - started)
            _write_report(output, result)
            return result
        # The bank is frozen before any test trajectories or activations are collected.
        episodes["evaluation"] = _collect_split(
            output / "evaluation_episodes.jsonl", "evaluation", config.evaluation_seeds,
            agent, environment_factory, config, scorer)
        evaluation = extract_feature_records(episodes["evaluation"], labeler, feature_extractor, layers=settings.layers,
                                             output_path=output / "evaluation_features.jsonl", cache=cache)
        probe_evaluation = bank.evaluate(evaluation)
        predictions = probe_evaluation.pop("predictions")
        with (output / "probe_predictions.jsonl").open("w") as stream:
            for prediction in predictions:
                stream.write(json.dumps(prediction, allow_nan=False) + "\n")
        outcome_evaluation = summarize_probe_outcomes(episodes["evaluation"], predictions,
                                                      layers=settings.layers, primary_layer=settings.primary_layer)
        result.update(status="complete" if trained else "insufficient_training_labels",
                      outcomes={split: _outcomes(items) for split, items in episodes.items()},
                      evaluation_labels=label_counts(episodes["evaluation"], labeler),
                      probe_evaluation=probe_evaluation, outcome_evaluation=outcome_evaluation,
                      elapsed_seconds=time.monotonic() - started)
        _write_report(output, result)
        return result
    except Exception as error:
        _write_json(output / "error.json", {"type": type(error).__name__, "message": str(error)})
        raise
