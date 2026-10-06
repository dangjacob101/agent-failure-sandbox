"""Independent probe keys, abstention, held-out metrics, and portable models."""
import copy
import importlib.util
import json
import math
import unittest

from early_failure.probes import LinearProbeBank


def record(episode_id, feature=-1.0, target=0, env_id="env-a", step=0, layer=0):
    return {"episode_id": episode_id, "env_id": env_id, "step": step,
            "layer": layer, "features": [feature], "target": target}


def training_rows():
    return [record("f1", -2.0), record("f2", -1.0),
            record("s1", 1.0, 1), record("s2", 2.0, 1)]


def portable_bank():
    # A fixed numeric model also exercises inference without scikit-learn.
    payload = LinearProbeBank().fit([record("train-f1"), record("train-f2")]).to_dict()
    probe = payload["probes"][0]
    ids = ["train-f1", "train-f2", "train-s1", "train-s2"]
    payload["train_episode_ids"] = ids
    probe.update({
        "train_episode_ids": ids, "n_records": 4, "n_labeled": 4,
        "class_counts": {"failure": 2, "success": 2}, "status": "trained",
        "reason": None, "coefficients": [2.0], "intercept": 0.0,
        "n_iter": 3, "converged": True,
    })
    return LinearProbeBank.from_dict(payload)


class ProbeTests(unittest.TestCase):
    def test_unsupported_classes_never_make_a_constant_probe(self):
        bank = LinearProbeBank().fit([record("f1"), record("f2"), record("u", target=None)])
        prediction = bank.predict(record("test", feature=100, target=None))
        self.assertEqual(prediction["status"], "unsupported")
        self.assertEqual(prediction["reason"], "insufficient_classes")
        self.assertIsNone(prediction["prediction"])
        self.assertIsNone(prediction["p_success"])
        self.assertEqual(prediction["majority_prediction"], 0)
        diagnostic = bank.to_dict()["probes"][0]
        self.assertEqual(diagnostic["n_labeled"], 2)
        self.assertEqual(diagnostic["n_unlabeled"], 1)
        self.assertEqual(diagnostic["coefficients"], [])

    def test_all_unlabeled_training_and_unknown_keys_are_unsupported(self):
        bank = LinearProbeBank().fit([record("u", target=None)])
        self.assertIsNone(bank.predict(record("x"))["majority_prediction"])
        for change in ({"env_id": "other"}, {"step": 1}, {"layer": 1}):
            row = record("x")
            row.update(change)
            self.assertEqual(bank.predict(row)["reason"], "unknown_probe")

    def test_stable_sigmoid_and_success_threshold(self):
        bank = portable_bank()
        for feature, expected in ((-1000, 0.0), (0, 0.5), (1000, 1.0)):
            result = bank.predict(record("eval", feature, target=None))
            self.assertEqual(result["p_success"], expected)
            self.assertEqual(result["prediction"], int(expected >= 0.5))
            self.assertIsNone(result["target"])

    def test_metrics_and_label_coverage_exclude_unlabeled_targets(self):
        bank = portable_bank()
        report = bank.evaluate([
            record("eval1", -1, 0), record("eval2", 1, 1),
            record("eval3", 1, 0), record("eval4", -1, None),
        ])
        self.assertEqual(report["label_coverage"], 0.75)
        self.assertEqual(report["n_predicted"], 4)
        self.assertEqual(report["n_evaluated"], 3)
        self.assertAlmostEqual(report["metrics"]["accuracy"], 2 / 3)
        self.assertAlmostEqual(report["metrics"]["f1_success"], 2 / 3)
        self.assertEqual(report["metrics"]["balanced_accuracy"], 0.75)
        self.assertAlmostEqual(report["majority_baseline"]["accuracy"], 2 / 3)
        self.assertEqual(report["metrics"]["confusion"], {"tn": 1, "fp": 1, "fn": 0, "tp": 1})

    def test_unsupported_and_empty_evaluation_have_no_fabricated_metrics(self):
        bank = LinearProbeBank().fit([record("train")])
        report = bank.evaluate([record("eval"), record("other", env_id="unknown")])
        self.assertEqual(report["n_unsupported"], 2)
        self.assertEqual(report["evaluation_coverage"], 0)
        self.assertIsNone(report["metrics"]["accuracy"])
        self.assertEqual(report["majority_baseline"]["n"], 1)
        self.assertIsNone(report["majority_baseline"]["balanced_accuracy"])
        self.assertIsNone(bank.evaluate([])["label_coverage"])

    def test_whole_episode_overlap_rejected_even_across_keys_or_unlabeled_rows(self):
        bank = LinearProbeBank().fit([record("train", target=None)])
        for row in (record("train", layer=99), record("train", step=99),
                    record("train", env_id="other", target=None)):
            with self.assertRaisesRegex(ValueError, "episode overlap"):
                bank.evaluate([row])

    def test_json_roundtrip_is_independent_and_preserves_overlap_check(self):
        bank = portable_bank()
        payload = json.loads(json.dumps(bank.to_dict(), allow_nan=False))
        restored = LinearProbeBank.from_dict(payload)
        self.assertEqual(bank.to_dict(), restored.to_dict())
        self.assertEqual(bank.predict(record("eval")), restored.predict(record("eval")))
        payload["probes"][0]["coefficients"][0] = 999
        self.assertEqual(restored.to_dict()["probes"][0]["coefficients"], [2.0])
        with self.assertRaisesRegex(ValueError, "episode overlap"):
            restored.evaluate([record("train-s1")])

    def test_validation_and_failed_refit_preserve_previous_state(self):
        bank = LinearProbeBank().fit([record("train")])
        before = bank.to_dict()
        invalid = [[], [record("dup"), record("dup")],
                   [record("x"), {**record("y"), "features": [1, 2]}]]
        for field, value in (("episode_id", ""), ("env_id", ""), ("step", -1),
                             ("layer", True), ("target", True), ("target", 2),
                             ("features", []), ("features", [math.nan]),
                             ("features", [True])):
            invalid.append([{**record("bad"), field: value}])
        for rows in invalid:
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                bank.fit(rows)
            self.assertEqual(bank.to_dict(), before)
        with self.assertRaisesRegex(ValueError, "feature width"):
            bank.predict({**record("eval"), "features": [1, 2]})
        with self.assertRaisesRegex(ValueError, "duplicate"):
            bank.evaluate([record("eval"), record("eval")])
        for kwargs in ({"seed": True}, {"seed": -1}, {"min_per_class": 0}):
            with self.assertRaises(ValueError):
                LinearProbeBank(**kwargs)

    def test_unfitted_bank_and_corrupt_serialization_rejected(self):
        for operation in (lambda bank: bank.predict(record("eval")),
                          lambda bank: bank.evaluate([]), lambda bank: bank.to_dict()):
            with self.assertRaises(ValueError):
                operation(LinearProbeBank())
        payload = portable_bank().to_dict()
        for field, value in (("coefficients", []), ("coefficients", [math.inf]),
                             ("intercept", math.nan), ("status", "unsupported"),
                             ("n_records", 5), ("train_episode_ids", []),
                             ("majority_target", True)):
            bad = copy.deepcopy(payload)
            bad["probes"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                LinearProbeBank.from_dict(bad)
        for field, value in (("schema_version", True), ("method", "pickle"),
                             ("train_episode_ids", []), ("probes", [])):
            bad = copy.deepcopy(payload)
            bad[field] = value
            with self.assertRaises(ValueError):
                LinearProbeBank.from_dict(bad)


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "scikit-learn is optional")
class ProbeTrainingTests(unittest.TestCase):
    def test_fit_matches_released_logistic_regression_on_raw_features(self):
        from sklearn.linear_model import LogisticRegression
        rows = training_rows() + [record("s3", 4, 1), record("u", -10000, None)]
        bank = LinearProbeBank().fit(rows)
        labeled = [row for row in rows if row["target"] is not None]
        reference = LogisticRegression(class_weight="balanced", max_iter=1000,
                                       random_state=42).fit(
            [row["features"] for row in labeled], [row["target"] for row in labeled])
        probability = bank.predict(record("eval", 0.25))["p_success"]
        self.assertAlmostEqual(probability, reference.predict_proba([[0.25]])[0, 1])
        diagnostic = bank.to_dict()["probes"][0]
        self.assertEqual(diagnostic["class_counts"], {"failure": 2, "success": 3})
        self.assertTrue(diagnostic["converged"])
        self.assertEqual(bank.to_dict()["provenance"]["feature_transform"], "none")

    def test_environment_step_and_layer_train_independent_models(self):
        rows = training_rows()
        for env_id, step, layer in (("env-b", 0, 0), ("env-a", 1, 0), ("env-a", 0, 1)):
            rows += [{**row, "env_id": env_id, "step": step, "layer": layer,
                      "features": [-row["features"][0]]} for row in training_rows()]
        bank = LinearProbeBank().fit(rows)
        self.assertEqual(len(bank.to_dict()["probes"]), 4)
        self.assertEqual(bank.predict(record("eval", 2))["prediction"], 1)
        for env_id, step, layer in (("env-b", 0, 0), ("env-a", 1, 0), ("env-a", 0, 1)):
            self.assertEqual(bank.predict(record("eval", 2, env_id=env_id,
                                                 step=step, layer=layer))["prediction"], 0)

    def test_one_example_per_class_is_explicitly_unsupported_by_default(self):
        rows = [record("f", -2, 0), record("s", 2, 1)]
        self.assertEqual(LinearProbeBank().fit(rows).predict(record("x"))["status"], "unsupported")
        self.assertEqual(LinearProbeBank(min_per_class=1).fit(rows).predict(record("x"))["status"], "trained")


if __name__ == "__main__":
    unittest.main()
