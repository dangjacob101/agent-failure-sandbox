"""Episode-level conformal calibration for repeated prefix checks.

Calibrate one maximum per completed episode: max(1-q) for success, max(q) for
failure. A prefix maximum cannot exceed the completed maximum, so its p-value
cannot be smaller. This bounds the chance of ever excluding the true outcome
at alpha, without assuming independent turns.

The bound requires exchangeability within environment/outcome and a fixed
policy, score, horizon, and check schedule. q is a success-oriented score;
p-values are compatibility ranks, not probabilities of success or failure.
See the README for the guarantee, missing-class caveats, and limitations.
"""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Mapping, Sequence
from math import isfinite
from numbers import Real
from typing import Any


CLASSES = ("success", "failure")
METHOD = "trajectory_max_class_conditional"
SCHEMA_VERSION = 1


def _sequence(value: Any, name: str) -> Sequence:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _probability(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    number = float(value)
    if not isfinite(number) or not 0.0 <= number <= 1.0:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return number


class TrajectoryConformal:
    """Conformal prediction sets, calibrated by environment and final outcome.

    Callers must supply completed, held-out episodes and keep calibration
    separate from model/threshold selection; this class cannot verify that.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = _probability(alpha, "alpha")
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be strictly between 0 and 1")
        self._scores: dict[str, dict[str, list[float]]] = {}

    def fit(self, episodes: Sequence[dict]) -> TrajectoryConformal:
        """Calibrate using one outcome-specific maximum per episode."""
        episodes = _sequence(episodes, "episodes")
        if not episodes:
            raise ValueError("episodes must contain at least one completed episode")

        # Build locally so a malformed replacement leaves prior calibration intact.
        scores: dict[str, dict[str, list[float]]] = {}
        episode_ids: set[str] = set()
        for index, episode in enumerate(episodes):
            if not isinstance(episode, Mapping):
                raise ValueError(f"episode {index} must be a mapping")
            episode_id = _identifier(episode.get("episode_id"), "episode_id")
            if episode_id in episode_ids:
                raise ValueError(f"duplicate episode_id: {episode_id}")
            episode_ids.add(episode_id)
            env_id = _identifier(episode.get("env_id"), "env_id")
            success = episode.get("success")
            if not isinstance(success, bool):
                raise ValueError(f"episode {episode_id}: success must be a boolean")
            checks = _sequence(episode.get("checks"), f"episode {episode_id}: checks")
            if not checks:
                raise ValueError(f"episode {episode_id}: checks must not be empty")

            previous_step = -1
            maximum = 0.0
            for check in checks:
                if not isinstance(check, Mapping):
                    raise ValueError(f"episode {episode_id}: each check must be a mapping")
                step = check.get("step")
                if isinstance(step, bool) or not isinstance(step, int) or step < 0:
                    raise ValueError(f"episode {episode_id}: step must be a nonnegative integer")
                if step <= previous_step:
                    raise ValueError(f"episode {episode_id}: steps must be strictly increasing")
                previous_step = step
                q = _probability(check.get("q"), f"episode {episode_id}: q")
                maximum = max(maximum, 1.0 - q if success else q)

            by_class = scores.setdefault(env_id, {label: [] for label in CLASSES})
            by_class["success" if success else "failure"].append(maximum)

        for by_class in scores.values():
            for values in by_class.values():
                values.sort()
        self._scores = scores
        return self

    @staticmethod
    def _p_value(calibration: Sequence[float], nonconformity: float) -> float:
        # Including equality makes ties conservative. The +1 includes the
        # test episode in the rank and gives p=1 for an unsupported class.
        at_least = len(calibration) - bisect_left(calibration, nonconformity)
        return (1 + at_least) / (len(calibration) + 1)

    def predict(self, env_id: str, q_history: Sequence[float]) -> dict:
        """Return labels with p > alpha; only singleton failure triggers an alarm.

        Supply all monitored scores on every call; no episode state is stored.
        An empty set is labeled out_of_distribution, but does not prove a shift.
        """
        env_id = _identifier(env_id, "env_id")
        if env_id not in self._scores:
            raise ValueError(f"unknown or uncalibrated environment: {env_id}")
        q_history = _sequence(q_history, "q_history")
        if not q_history:
            raise ValueError("q_history must not be empty")
        qs = [_probability(q, "q_history entry") for q in q_history]
        by_class = self._scores[env_id]
        p_success = self._p_value(by_class["success"], max(1.0 - q for q in qs))
        p_failure = self._p_value(by_class["failure"], max(qs))
        p_values = {"success": p_success, "failure": p_failure}
        prediction_set = [label for label in CLASSES if p_values[label] > self.alpha]
        if len(prediction_set) == 2:
            status = "uncertain"
        elif not prediction_set:
            status = "out_of_distribution"
        else:
            status = prediction_set[0]
        return {
            "p_success": p_success,
            "p_failure": p_failure,
            "prediction_set": prediction_set,
            "status": status,
            "alarm": status == "failure",
        }

    def diagnostics(self) -> dict:
        """Describe calibration size and rank resolution for each class."""
        environments = {}
        for env_id, by_class in sorted(self._scores.items()):
            environments[env_id] = {
                "n_episodes": sum(len(values) for values in by_class.values()),
                **{
                    label: {
                        "n": len(by_class[label]),
                        "min_p": 1.0 / (len(by_class[label]) + 1),
                        "supported": bool(by_class[label]),
                        "can_reject_at_alpha": 1.0 / (len(by_class[label]) + 1) <= self.alpha,
                    }
                    for label in CLASSES
                },
            }
        return {
            "fitted": bool(self._scores),
            "method": METHOD,
            "alpha": self.alpha,
            "calibration_unit": "completed_episode",
            "rejection_rule": "p <= alpha",
            "environments": environments,
        }

    def to_dict(self) -> dict:
        """Return JSON-compatible calibration state (schema version 1)."""
        if not self._scores:
            raise ValueError("cannot serialize an unfitted calibrator")
        return {
            "schema_version": SCHEMA_VERSION,
            "method": METHOD,
            "alpha": self.alpha,
            "calibration_scores": {
                env_id: {label: list(by_class[label]) for label in CLASSES}
                for env_id, by_class in sorted(self._scores.items())
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> TrajectoryConformal:
        """Validate and restore state produced by :meth:`to_dict`."""
        if not isinstance(data, Mapping):
            raise ValueError("serialized calibration must be a mapping")
        if type(data.get("schema_version")) is not int or data["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported conformal schema_version")
        if data.get("method") != METHOD:
            raise ValueError("unsupported conformal method")
        result = cls(alpha=data.get("alpha"))
        calibration = data.get("calibration_scores")
        if not isinstance(calibration, Mapping) or not calibration:
            raise ValueError("calibration_scores must be a nonempty mapping")
        for env_id, by_class in calibration.items():
            env_id = _identifier(env_id, "env_id")
            if not isinstance(by_class, Mapping) or set(by_class) != set(CLASSES):
                raise ValueError("each environment must contain success and failure calibration scores")
            restored = {}
            for label in CLASSES:
                values = _sequence(by_class[label], f"{env_id}/{label} scores")
                restored[label] = sorted(_probability(value, "calibration score") for value in values)
            if not any(restored.values()):
                raise ValueError(f"environment {env_id} must have at least one calibration episode")
            result._scores[env_id] = restored
        return result
