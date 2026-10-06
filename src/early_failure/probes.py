"""Linear probes for conformal state labels, with one model per task/step/layer."""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
from numbers import Real
from typing import Any, Iterable
import warnings


def _integer(value: Any, name: str, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise ValueError(f"{name} must be an integer" + (
            f" >= {minimum}" if minimum is not None else ""))
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _record(record: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        raise ValueError("probe records must be dictionaries")
    for name in ("episode_id", "env_id"):
        if not isinstance(record.get(name), str) or not record[name]:
            raise ValueError(f"{name} must be a nonempty string")
    step = _integer(record.get("step"), "step", 0)
    layer = _integer(record.get("layer"), "layer")
    features = record.get("features")
    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError("features must be a nonempty sequence")
    target = record.get("target")
    if target is not None and (type(target) is not int or target not in (0, 1)):
        raise ValueError("target must be a conformal label 0, 1, or None")
    return {
        "episode_id": record["episode_id"], "env_id": record["env_id"],
        "step": step, "layer": layer, "target": target,
        "features": [_number(value, "feature") for value in features],
    }


def _key(record: dict[str, Any]) -> tuple[str, int, int]:
    return record["env_id"], record["step"], record["layer"]


def _records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [_record(record) for record in records]
    identities = [(row["episode_id"], *_key(row)) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate episode/env/step/layer record")
    return rows


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _metrics(pairs: list[tuple[int, int]]) -> dict[str, Any]:
    counts = Counter(pairs)
    tn, fp = counts[0, 0], counts[0, 1]
    fn, tp = counts[1, 0], counts[1, 1]
    n = len(pairs)
    f1_success = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
    f1_failure = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "n": n, "accuracy": (tp + tn) / n if n else None,
        "f1_success": f1_success if n else None,
        "f1_failure": f1_failure if n else None,
        "macro_f1": (f1_success + f1_failure) / 2 if n else None,
        # A single-class evaluation cannot establish balanced discrimination.
        "balanced_accuracy": ((tp / (tp + fn) + tn / (tn + fp)) / 2
                              if tp + fn and tn + fp else None),
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


class LinearProbeBank:
    """Train on raw features and singleton conformal labels; None labels are omitted."""

    def __init__(self, seed: int = 42, min_per_class: int = 2):
        self.seed = _integer(seed, "seed", 0)
        self.min_per_class = _integer(min_per_class, "min_per_class", 1)
        self._probes: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._train_episode_ids: set[str] = set()
        self._provenance: dict[str, Any] = {}
        self._fitted = False

    def fit(self, records: Iterable[dict[str, Any]]) -> LinearProbeBank:
        rows = _records(records)
        if not rows:
            raise ValueError("at least one training record is required")
        groups: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[_key(row)].append(row)
        probes = {}
        sklearn_version = None
        for key, group in sorted(groups.items()):
            width = len(group[0]["features"])
            if any(len(row["features"]) != width for row in group):
                raise ValueError("inconsistent feature width within a probe")
            labeled = [row for row in group if row["target"] is not None]
            counts = Counter(row["target"] for row in labeled)
            probe = {
                "env_id": key[0], "step": key[1], "layer": key[2],
                "feature_width": width, "n_records": len(group),
                "n_labeled": len(labeled), "n_unlabeled": len(group) - len(labeled),
                "class_counts": {"failure": counts[0], "success": counts[1]},
                "train_episode_ids": sorted({row["episode_id"] for row in group}),
                "majority_target": (int(counts[1] > counts[0]) if labeled else None),
                "status": "unsupported", "reason": "insufficient_classes",
                "coefficients": [], "intercept": None, "n_iter": None,
                "converged": None,
            }
            if min(counts[0], counts[1]) >= self.min_per_class:
                # Keeping this optional leaves replay and synthetic tests lightweight.
                try:
                    import sklearn
                    from sklearn.exceptions import ConvergenceWarning
                    from sklearn.linear_model import LogisticRegression
                except ImportError as error:
                    raise ImportError("Probe training requires scikit-learn") from error
                sklearn_version = sklearn.__version__
                model = LogisticRegression(
                    class_weight="balanced", max_iter=1000, random_state=self.seed)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model.fit([row["features"] for row in labeled],
                              [row["target"] for row in labeled])
                probe.update({
                    "status": "trained", "reason": None,
                    "coefficients": model.coef_[0].tolist(),
                    "intercept": float(model.intercept_[0]),
                    "n_iter": int(model.n_iter_[0]),
                    "converged": not any(issubclass(item.category, ConvergenceWarning)
                                         for item in caught),
                })
            probes[key] = probe
        self._probes = probes
        self._train_episode_ids = {row["episode_id"] for row in rows}
        self._provenance = {
            "estimator": "sklearn.linear_model.LogisticRegression",
            "class_weight": "balanced", "max_iter": 1000,
            "random_state": self.seed, "feature_transform": "none",
            "target_source": "singleton_conformal_state_label",
            "decision_threshold": 0.5, "majority_tie_target": 0,
            "sklearn_version": sklearn_version,
        }
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise ValueError("fit the probe bank first")

    def predict(self, record: dict[str, Any]) -> dict[str, Any]:
        """Predict without needing a label or importing scikit-learn."""
        self._require_fit()
        row = _record(record)
        probe = self._probes.get(_key(row))
        result = {
            "episode_id": row["episode_id"], "env_id": row["env_id"],
            "step": row["step"], "layer": row["layer"], "target": row["target"],
            "prediction": None, "label": None, "p_success": None,
            "majority_prediction": probe["majority_target"] if probe else None,
            "status": "unsupported", "reason": "unknown_probe",
        }
        if probe is None:
            return result
        if len(row["features"]) != probe["feature_width"]:
            raise ValueError("prediction feature width differs from training")
        if probe["status"] != "trained":
            result["reason"] = probe["reason"]
            return result
        logit = math.fsum(weight * value for weight, value in
                          zip(probe["coefficients"], row["features"])) + probe["intercept"]
        probability = _sigmoid(logit)
        prediction = int(probability >= 0.5)
        result.update({
            "prediction": prediction, "label": "success" if prediction else "failure",
            "p_success": probability, "status": "trained", "reason": None,
        })
        return result

    def evaluate(self, records: Iterable[dict[str, Any]]) -> dict[str, Any]:
        """Measure held-out conformal-label prediction; do not substitute final outcomes."""
        self._require_fit()
        rows = _records(records)
        overlap = self._train_episode_ids.intersection(row["episode_id"] for row in rows)
        if overlap:
            raise ValueError(f"training/evaluation episode overlap: {sorted(overlap)}")
        predictions = [self.predict(row) for row in rows]

        def summarize(items: list[dict[str, Any]]) -> dict[str, Any]:
            labeled = [item for item in items if item["target"] is not None]
            scored = [item for item in labeled if item["prediction"] is not None]
            baseline = [item for item in labeled if item["majority_prediction"] is not None]
            return {
                "n_records": len(items), "n_labeled": len(labeled),
                "n_predicted": sum(item["prediction"] is not None for item in items),
                "n_evaluated": len(scored), "n_unsupported": sum(
                    item["status"] == "unsupported" for item in items),
                "label_coverage": len(labeled) / len(items) if items else None,
                "evaluation_coverage": len(scored) / len(labeled) if labeled else None,
                "metrics": _metrics([(item["target"], item["prediction"]) for item in scored]),
                # Its own n exposes when unsupported probes leave different coverage.
                "majority_baseline": _metrics([
                    (item["target"], item["majority_prediction"]) for item in baseline]),
            }

        grouped: dict[tuple[str, int, int], list[dict[str, Any]]] = defaultdict(list)
        for item in predictions:
            grouped[_key(item)].append(item)
        return {
            **summarize(predictions), "target_source": "singleton_conformal_state_label",
            "per_probe": [{"env_id": key[0], "step": key[1], "layer": key[2],
                           **summarize(items)} for key, items in sorted(grouped.items())],
            "predictions": predictions,
        }

    def to_dict(self) -> dict[str, Any]:
        self._require_fit()
        return copy.deepcopy({
            "schema_version": 1, "method": "linear_logistic_probe_bank",
            "seed": self.seed, "min_per_class": self.min_per_class,
            "train_episode_ids": sorted(self._train_episode_ids),
            "provenance": self._provenance,
            "probes": [self._probes[key] for key in sorted(self._probes)],
        })

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> LinearProbeBank:
        if (not isinstance(payload, dict) or type(payload.get("schema_version")) is not int
                or payload["schema_version"] != 1
                or payload.get("method") != "linear_logistic_probe_bank"):
            raise ValueError("unsupported probe-bank serialization")
        bank = cls(seed=payload.get("seed"), min_per_class=payload.get("min_per_class"))
        ids = payload.get("train_episode_ids")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(item, str) or not item for item in ids)
                or len(set(ids)) != len(ids)):
            raise ValueError("invalid training episode IDs")
        probes = payload.get("probes")
        if not isinstance(probes, list) or not probes:
            raise ValueError("serialized probes must be a nonempty list")
        for probe in probes:
            if not isinstance(probe, dict):
                raise ValueError("invalid serialized probe")
            env_id = probe.get("env_id")
            if not isinstance(env_id, str) or not env_id:
                raise ValueError("invalid probe environment")
            key = (env_id, _integer(probe.get("step"), "step", 0),
                   _integer(probe.get("layer"), "layer"))
            width = _integer(probe.get("feature_width"), "feature_width", 1)
            n = _integer(probe.get("n_records"), "n_records", 1)
            n_labeled = _integer(probe.get("n_labeled"), "n_labeled", 0)
            n_unlabeled = _integer(probe.get("n_unlabeled"), "n_unlabeled", 0)
            counts = probe.get("class_counts")
            if not isinstance(counts, dict):
                raise ValueError("invalid class counts")
            failure = _integer(counts.get("failure"), "failure count", 0)
            success = _integer(counts.get("success"), "success count", 0)
            if n_labeled != failure + success or n != n_labeled + n_unlabeled:
                raise ValueError("inconsistent probe counts")
            group_ids = probe.get("train_episode_ids")
            if (not isinstance(group_ids, list) or len(group_ids) != n
                    or any(not isinstance(item, str) for item in group_ids)
                    or len(set(group_ids)) != n or not set(group_ids).issubset(ids)):
                raise ValueError("invalid per-probe training episode IDs")
            expected_majority = int(success > failure) if n_labeled else None
            if (probe.get("majority_target") != expected_majority
                    or isinstance(probe.get("majority_target"), bool)):
                raise ValueError("invalid majority baseline")
            trained = min(failure, success) >= bank.min_per_class
            if probe.get("status") != ("trained" if trained else "unsupported"):
                raise ValueError("probe status disagrees with class counts")
            coefficients = probe.get("coefficients")
            if not isinstance(coefficients, list):
                raise ValueError("invalid probe coefficients")
            if trained:
                if len(coefficients) != width or probe.get("reason") is not None:
                    raise ValueError("invalid trained probe")
                for value in coefficients:
                    _number(value, "coefficient")
                _number(probe.get("intercept"), "intercept")
                _integer(probe.get("n_iter"), "n_iter", 1)
                if type(probe.get("converged")) is not bool:
                    raise ValueError("invalid convergence diagnostic")
            elif (coefficients or probe.get("intercept") is not None
                  or probe.get("reason") != "insufficient_classes"):
                raise ValueError("unsupported probes must not contain a fitted model")
            if key in bank._probes:
                raise ValueError("duplicate serialized probe key")
            bank._probes[key] = copy.deepcopy(probe)
        provenance = payload.get("provenance")
        if (not isinstance(provenance, dict)
                or provenance.get("target_source") != "singleton_conformal_state_label"
                or provenance.get("feature_transform") != "none"):
            raise ValueError("invalid probe provenance")
        if set(ids) != set().union(*(set(probe["train_episode_ids"]) for probe in probes)):
            raise ValueError("training episode IDs disagree with probe records")
        bank._provenance = copy.deepcopy(provenance)
        bank._train_episode_ids = set(ids)
        bank._fitted = True
        return bank
