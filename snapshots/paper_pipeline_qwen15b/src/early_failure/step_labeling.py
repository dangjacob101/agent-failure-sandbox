"""Label intermediate MC scores using either paper ranks or released percentiles.

Final outcomes define calibration groups. Each timestep is calibrated separately;
these labels are probe targets, not an episode-level repeated-warning guarantee.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real


MODES = ("paper_equations", "upstream_percentiles")
LABELS = ("success", "failure")
METHOD = "stepwise_final_outcome_calibrated_labels"
SCHEMA_VERSION = 1


def _probability(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a finite number in [0, 1]")
    return float(value)


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _sequence(value, name):
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _percentile(sorted_values, percentile):
    """NumPy's default linear interpolation, without a NumPy dependency."""
    position = (len(sorted_values) - 1) * (percentile / 100)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    left, right = sorted_values[lower], sorted_values[upper]
    difference = right - left
    return right - difference * (1 - fraction) if fraction >= 0.5 else left + difference * fraction


class StepwiseLabeler:
    """Calibrate per environment/timestep, then label each score independently.

    Paper mode retains p >= alpha, including ties. Missing either outcome class
    prevents a training target. Upstream mode reproduces its success-first
    percentile rule and empty-class defaults; it makes no coverage claim.
    """

    def __init__(self, mode="paper_equations", alpha=0.1):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.alpha = _probability(alpha, "alpha")
        if not 0 < self.alpha < 1:
            raise ValueError("alpha must be strictly between 0 and 1")
        self._rewards = {}

    def fit(self, episodes):
        """Use real final outcomes to group nonterminal checks at each timestep."""
        episodes = _sequence(episodes, "episodes")
        if not episodes:
            raise ValueError("episodes must contain completed calibration episodes")
        rewards, seen = {}, set()
        for episode in episodes:
            if not isinstance(episode, Mapping):
                raise ValueError("each episode must be a mapping")
            episode_id = _name(episode.get("episode_id"), "episode_id")
            if episode_id in seen:
                raise ValueError(f"duplicate episode_id: {episode_id}")
            seen.add(episode_id)
            env = _name(episode.get("env_id"), "env_id")
            success = episode.get("success")
            if type(success) is not bool:
                raise ValueError("episode success must be a boolean final outcome")
            num_steps = _integer(episode.get("num_steps"), "num_steps", 1)
            checks = _sequence(episode.get("checks"), "checks")
            if not checks:
                raise ValueError("each calibration episode must have checks")
            previous = -1
            for check in checks:
                if not isinstance(check, Mapping):
                    raise ValueError("each check must be a mapping")
                step = _integer(check.get("step"), "step")
                if step <= previous or step >= num_steps:
                    raise ValueError("check steps must increase and precede the terminal action")
                previous = step
                q = _probability(check.get("q"), "q")
                groups = rewards.setdefault(env, {}).setdefault(step, {label: [] for label in LABELS})
                groups["success" if success else "failure"].append(q)
        self._rewards = {
            env: {step: {label: sorted(values) for label, values in groups.items()}
                  for step, groups in steps.items()} for env, steps in rewards.items()
        }
        return self

    @staticmethod
    def _support(groups):
        return {label: len(groups[label]) for label in LABELS}

    @staticmethod
    def _p_value(values, nonconformity):
        return (1 + sum(value >= nonconformity for value in values)) / (len(values) + 1)

    def _thresholds(self, groups):
        coverage = 1 - self.alpha
        return {
            "t_s": _percentile(groups["success"], coverage * 100) if groups["success"] else 1.0,
            "t_f": _percentile(groups["failure"], (1 - coverage) * 100) if groups["failure"] else 0.0,
        }

    def predict(self, env_id, step, q):
        """Return a binary probe target only when this method assigns one."""
        env_id = _name(env_id, "env_id")
        step = _integer(step, "step")
        q = _probability(q, "q")
        if env_id not in self._rewards:
            raise ValueError(f"unknown or uncalibrated environment: {env_id}")
        groups = self._rewards[env_id].get(step)
        if groups is None:
            return {"status": "unsupported", "target": None, "details": {
                "mode": self.mode, "step": step, "q": q, "support": {label: 0 for label in LABELS},
                "both_classes_supported": False, "reason": "No calibration checks at this timestep.",
            }}
        support = self._support(groups)
        supported = all(support.values())
        details = {"mode": self.mode, "step": step, "q": q, "support": support,
                   "both_classes_supported": supported}
        if self.mode == "paper_equations":
            p_success = self._p_value([1 - value for value in groups["success"]], 1 - q)
            p_failure = self._p_value(groups["failure"], q)
            labels = [label for label, p in (("success", p_success), ("failure", p_failure)) if p >= self.alpha]
            details.update(p_success=p_success, p_failure=p_failure, prediction_set=labels,
                           retention_rule="p >= alpha")
            if not supported:
                details["reason"] = "Both final-outcome classes are required to assign a paper-mode target."
                return {"status": "unsupported", "target": None, "details": details}
            status = labels[0] if len(labels) == 1 else ("uncertain" if labels else "empty")
        else:
            thresholds = self._thresholds(groups)
            details.update(**thresholds, coverage=1 - self.alpha, formal_coverage_claim=False,
                           rule="q >= t_s: success; else q <= t_f: failure; else uncertain",
                           used_empty_class_defaults=not supported)
            status = "success" if q >= thresholds["t_s"] else ("failure" if q <= thresholds["t_f"] else "uncertain")
        target = 1 if status == "success" else (0 if status == "failure" else None)
        return {"status": status, "target": target, "details": details}

    def diagnostics(self):
        environments = {}
        for env, steps in sorted(self._rewards.items()):
            environments[env] = {}
            for step, groups in sorted(steps.items()):
                support = self._support(groups)
                item = {"support": support, "both_classes_supported": all(support.values())}
                if self.mode == "paper_equations":
                    by_class = {}
                    for label, rewards in groups.items():
                        scores = [1 - q if label == "success" else q for q in rewards]
                        minimum_p = self._p_value(scores, 1.0)
                        by_class[label] = {
                            "n": len(scores), "supported": bool(scores),
                            "min_rank_p_without_ties": 1 / (len(scores) + 1),
                            "min_achievable_p_at_nonconformity_1": minimum_p,
                            "maximal_nonconformity_ties": sum(score >= 1 for score in scores),
                            "can_reject_at_alpha": bool(scores) and minimum_p < self.alpha,
                        }
                    reasons = []
                    for label, diagnostic in by_class.items():
                        target = "failure" if label == "success" else "success"
                        if not diagnostic["supported"]:
                            reasons.append(f"No {label} calibration checks; no binary targets are assigned.")
                        else:
                            if diagnostic["min_rank_p_without_ties"] >= self.alpha:
                                reasons.append(f"Too few {label} checks to reject p < alpha; {target} targets are impossible.")
                            if diagnostic["maximal_nonconformity_ties"] and not diagnostic["can_reject_at_alpha"]:
                                reasons.append(f"Ties keep the minimum {label} p-value at or above alpha; {target} targets are impossible.")
                    item.update(classes=by_class, reasons=reasons,
                                failure_target_possible=item["both_classes_supported"] and by_class["success"]["can_reject_at_alpha"],
                                success_target_possible=item["both_classes_supported"] and by_class["failure"]["can_reject_at_alpha"])
                else:
                    item.update(**self._thresholds(groups), used_empty_class_defaults=not item["both_classes_supported"])
                environments[env][str(step)] = item
        return {
            "fitted": bool(self._rewards), "method": METHOD, "mode": self.mode, "alpha": self.alpha,
            "calibration_group_label": "real_final_outcome", "calibration_unit": "one_check_per_episode_at_each_timestep",
            "retention_rule": "p >= alpha" if self.mode == "paper_equations" else "upstream_success_first_percentiles",
            "formal_coverage_claim": "fixed environment/timestep/class under exchangeability; no simultaneous-time guarantee"
            if self.mode == "paper_equations" else False,
            "environments": environments,
        }

    def to_dict(self):
        """Serialize rewards and support counts, preserving ties exactly."""
        if not self._rewards:
            raise ValueError("cannot serialize an unfitted labeler")
        calibration = {
            env: {str(step): {label: list(groups[label]) for label in LABELS} for step, groups in sorted(steps.items())}
            for env, steps in sorted(self._rewards.items())
        }
        return {"schema_version": SCHEMA_VERSION, "method": METHOD, "mode": self.mode, "alpha": self.alpha,
                "calibration_rewards": calibration,
                "support": {env: {step: self._support(groups) for step, groups in steps.items()}
                            for env, steps in calibration.items()}}

    @classmethod
    def from_dict(cls, artifact):
        """Restore a validated artifact without importing the model or environment."""
        if not isinstance(artifact, Mapping):
            raise ValueError("serialized labeler must be a mapping")
        if type(artifact.get("schema_version")) is not int or artifact["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported stepwise labeler schema_version")
        if artifact.get("method") != METHOD:
            raise ValueError("unsupported stepwise labeler method")
        result = cls(mode=artifact.get("mode"), alpha=artifact.get("alpha"))
        calibration = artifact.get("calibration_rewards")
        if not isinstance(calibration, Mapping) or not calibration:
            raise ValueError("calibration_rewards must be a nonempty mapping")
        for env, steps in calibration.items():
            _name(env, "env_id")
            if not isinstance(steps, Mapping) or not steps:
                raise ValueError("each environment must have calibrated timesteps")
            restored = {}
            for key, groups in steps.items():
                if not isinstance(key, str) or not key.isdecimal() or str(int(key)) != key:
                    raise ValueError("serialized timestep keys must be canonical nonnegative integers")
                if not isinstance(groups, Mapping) or set(groups) != set(LABELS):
                    raise ValueError("each timestep must have success and failure reward lists")
                by_class = {label: sorted(_probability(q, "q") for q in _sequence(groups[label], label)) for label in LABELS}
                if not any(by_class.values()):
                    raise ValueError("each calibrated timestep needs at least one reward")
                restored[int(key)] = by_class
            result._rewards[env] = restored
        if artifact.get("support") != result.to_dict()["support"]:
            raise ValueError("serialized support counts do not match rewards")
        return result
