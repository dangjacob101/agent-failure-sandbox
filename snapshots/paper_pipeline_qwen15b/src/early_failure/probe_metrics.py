"""Episode-level probe warnings against final task outcomes, independent of training."""
from __future__ import annotations

import math
from numbers import Real
from statistics import median
from typing import Any, Iterable


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _wilson(numerator: int, denominator: int) -> dict[str, Any]:
    """A binomial interval over episodes, never over correlated checks."""
    if not denominator:
        return {"numerator": numerator, "denominator": 0, "lower": None, "upper": None}
    z = 1.959963984540054
    rate = numerator / denominator
    scale = 1 + z * z / denominator
    center = (rate + z * z / (2 * denominator)) / scale
    radius = z * math.sqrt(rate * (1 - rate) / denominator
                          + z * z / (4 * denominator * denominator)) / scale
    return {"numerator": numerator, "denominator": denominator,
            "lower": max(0.0, center - radius), "upper": min(1.0, center + radius)}


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (
            f" >= {minimum}" if minimum is not None else ""))
    return value


def summarize_probe_outcomes(
    episodes: Iterable[dict[str, Any]],
    predictions: Iterable[dict[str, Any]],
    *,
    layers: Iterable[int],
    primary_layer: int,
) -> dict[str, Any]:
    """Keep every episode in outcome denominators, including unsupported probes."""
    layer_ids = [_integer(layer, "layer") for layer in layers]
    if not layer_ids or len(set(layer_ids)) != len(layer_ids):
        raise ValueError("layers must be nonempty and unique")
    primary_layer = _integer(primary_layer, "primary_layer")
    if primary_layer not in layer_ids:
        raise ValueError("primary_layer must belong to layers")
    episode_by_id = {}
    for episode in episodes:
        if not isinstance(episode, dict):
            raise ValueError("episodes must be dictionaries")
        for field in ("episode_id", "env_id"):
            if not isinstance(episode.get(field), str) or not episode[field]:
                raise ValueError(f"{field} must be a nonempty string")
        if type(episode.get("success")) is not bool:
            raise ValueError("success must be a boolean final outcome")
        _integer(episode.get("num_steps"), "num_steps", 1)
        identifier = episode["episode_id"]
        if identifier in episode_by_id:
            raise ValueError("duplicate episode_id")
        episode_by_id[identifier] = episode
    by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layer_ids}
    seen = set()
    for prediction in predictions:
        if not isinstance(prediction, dict):
            raise ValueError("predictions must be dictionaries")
        identifier = prediction.get("episode_id")
        if not isinstance(identifier, str) or identifier not in episode_by_id:
            raise ValueError("prediction references an unknown episode_id")
        episode = episode_by_id[identifier]
        if prediction.get("env_id") != episode["env_id"]:
            raise ValueError("prediction environment differs from its episode")
        step = _integer(prediction.get("step"), "prediction step", 0)
        if step >= episode["num_steps"]:
            raise ValueError("prediction step must precede the terminal step")
        layer = _integer(prediction.get("layer"), "prediction layer")
        if layer not in by_layer:
            raise ValueError("prediction layer is not in layers")
        identity = identifier, step, layer
        if identity in seen:
            raise ValueError("duplicate episode-step-layer prediction")
        seen.add(identity)
        if "prediction" not in prediction:
            raise ValueError("prediction label is missing")
        label = prediction["prediction"]
        if label is not None and (type(label) is not int or label not in (0, 1)):
            raise ValueError("prediction must be 0, 1, or None")
        probability = prediction.get("p_success")
        if probability is not None:
            if (isinstance(probability, bool) or not isinstance(probability, Real)
                    or not math.isfinite(probability) or not 0 <= probability <= 1):
                raise ValueError("p_success must be a finite probability")
            if label is None or int(probability >= 0.5) != label:
                raise ValueError("p_success disagrees with prediction")
        by_layer[layer].append(prediction)

    total = len(episode_by_id)
    successes = sum(episode["success"] for episode in episode_by_id.values())
    failures = total - successes

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        supported = [row for row in rows if row["prediction"] is not None]
        supported_ids = {row["episode_id"] for row in supported}
        first_alarms: dict[str, int] = {}
        for row in supported:
            if row["prediction"] == 0:
                identifier = row["episode_id"]
                first_alarms[identifier] = min(first_alarms.get(identifier, row["step"]), row["step"])
        alarms = [
            {"episode_id": identifier, "env_id": episode_by_id[identifier]["env_id"],
             "success": episode_by_id[identifier]["success"], "first_alarm_step": step,
             "lead_steps": episode_by_id[identifier]["num_steps"] - step}
            for identifier, step in sorted(first_alarms.items())
        ]
        detected = [alarm for alarm in alarms if not alarm["success"]]
        false_alarms = len(alarms) - len(detected)
        leads = [alarm["lead_steps"] for alarm in detected]
        lead_counts = {str(lead): sum(value >= lead for value in leads) for lead in (1, 2, 3)}
        intervals = {
            "failure_recall": _wilson(len(detected), failures),
            "episode_false_alarm_rate": _wilson(false_alarms, successes),
            "alarm_precision": _wilson(len(detected), len(alarms)),
            "episode_prediction_coverage": _wilson(len(supported_ids), total),
            "failure_recall_by_minimum_lead_steps": {
                lead: _wilson(count, failures) for lead, count in lead_counts.items()},
        }
        unsupported_ids = sorted(set(episode_by_id) - supported_ids)
        return {
            "episodes": total, "successful_episodes": successes, "failed_episodes": failures,
            "checks": len(rows), "supported_checks": len(supported),
            "unsupported_checks": len(rows) - len(supported),
            "supported_check_fraction": _ratio(len(supported), len(rows)),
            "supported_episodes": len(supported_ids), "unsupported_episodes": len(unsupported_ids),
            "episode_prediction_coverage": _ratio(len(supported_ids), total),
            "has_supported_predictions": bool(supported),
            "unsupported_episode_ids": unsupported_ids,
            "alarm_episodes": len(alarms), "detected_failure_episodes": len(detected),
            "missed_failure_episodes": failures - len(detected),
            "false_alarm_episodes": false_alarms,
            "failure_recall": _ratio(len(detected), failures),
            "episode_false_alarm_rate": _ratio(false_alarms, successes),
            "alarm_precision": _ratio(len(detected), len(alarms)),
            "failure_recall_by_minimum_lead_steps": {
                lead: _ratio(count, failures) for lead, count in lead_counts.items()},
            "detected_failures_by_minimum_lead_steps": lead_counts,
            "median_failure_lead_steps": median(leads) if leads else None,
            "confidence_intervals_95": intervals,
            "alarms": alarms,
        }

    summaries = {str(layer): summarize(by_layer[layer]) for layer in layer_ids}
    baseline_rows = [
        {"episode_id": identifier, "step": 0, "prediction": 0}
        for identifier in episode_by_id
    ]
    always_warn = summarize(baseline_rows)
    never_warn = summarize([{**row, "prediction": 1} for row in baseline_rows])
    return {
        "primary_layer": primary_layer, "primary": summaries[str(primary_layer)],
        "by_layer": summaries,
        "baselines": {"always_warn": always_warn, "never_warn": never_warn},
        "target": "final_episode_outcome",
        "notes": [
            "Probe outcome metrics carry no conformal coverage guarantee.",
            "Unsupported episodes remain in all outcome denominators.",
            "Check coverage uses supplied preterminal prediction records.",
            "Wilson intervals use episode counts; lead-time medians use detected failures only.",
            "The primary layer must be fixed before examining evaluation outcomes.",
        ],
    }
