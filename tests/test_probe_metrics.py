"""Outcome recall and false alarms use episodes, including unsupported ones."""
import copy
import math
import unittest

from early_failure.probe_metrics import summarize_probe_outcomes


def episode(identifier, success=False, num_steps=4, env_id="env-a"):
    return {"episode_id": identifier, "env_id": env_id,
            "success": success, "num_steps": num_steps}


def prediction(identifier, step=0, label=0, layer=2, env_id="env-a"):
    return {"episode_id": identifier, "env_id": env_id, "step": step, "layer": layer,
            "prediction": label, "p_success": None if label is None else float(label)}


def summarize(episodes, predictions, layers=(2,), primary_layer=2):
    return summarize_probe_outcomes(episodes, predictions, layers=layers, primary_layer=primary_layer)


class ProbeOutcomeTests(unittest.TestCase):
    def test_recall_false_alarms_precision_and_lead_use_episode_denominators(self):
        episodes = [episode("f1"), episode("f2"), episode("f3"),
                    episode("s1", True), episode("s2", True)]
        predictions = [prediction("f1", 2), prediction("f1", 1), prediction("f1", 3),
                       prediction("f2", 3), prediction("f3", 0, None),
                       prediction("s1", 0), prediction("s2", 0, 1)]
        result = summarize(episodes, predictions)["primary"]
        self.assertEqual(result["detected_failure_episodes"], 2)
        self.assertEqual(result["missed_failure_episodes"], 1)
        self.assertEqual(result["false_alarm_episodes"], 1)
        self.assertAlmostEqual(result["failure_recall"], 2 / 3)
        self.assertEqual(result["episode_false_alarm_rate"], 0.5)
        self.assertAlmostEqual(result["alarm_precision"], 2 / 3)
        self.assertEqual(result["failure_recall_by_minimum_lead_steps"],
                         {"1": 2 / 3, "2": 1 / 3, "3": 1 / 3})
        self.assertEqual(result["median_failure_lead_steps"], 2)
        self.assertEqual(result["unsupported_episode_ids"], ["f3"])
        self.assertEqual(result["episode_prediction_coverage"], 4 / 5)
        self.assertEqual(result["supported_check_fraction"], 6 / 7)
        self.assertEqual(result["confidence_intervals_95"]["failure_recall"]["denominator"], 3)

    def test_primary_is_fixed_and_layers_are_reported_separately(self):
        episodes = [episode("f"), episode("s", True)]
        predictions = [prediction("f", label=1, layer=2), prediction("f", layer=4),
                       prediction("s", label=1, layer=4)]
        report = summarize(episodes, predictions, layers=(2, 4))
        self.assertEqual(report["primary_layer"], 2)
        self.assertEqual(report["primary"]["failure_recall"], 0)
        self.assertEqual(report["by_layer"]["4"]["failure_recall"], 1)
        self.assertEqual(report["primary"]["episode_prediction_coverage"], 0.5)

    def test_always_and_never_warning_baselines_preserve_class_imbalance(self):
        episodes = [episode("f1", num_steps=1), episode("f2", num_steps=2),
                    episode("f3", num_steps=3), episode("s", True)]
        baselines = summarize(episodes, [])["baselines"]
        always = baselines["always_warn"]
        never = baselines["never_warn"]
        self.assertEqual(always["alarm_precision"], 0.75)
        self.assertEqual(always["failure_recall"], 1)
        self.assertEqual(always["episode_false_alarm_rate"], 1)
        self.assertEqual(always["detected_failures_by_minimum_lead_steps"], {"1": 3, "2": 2, "3": 1})
        self.assertEqual(never["failure_recall"], 0)
        self.assertEqual(never["episode_false_alarm_rate"], 0)
        self.assertIsNone(never["alarm_precision"])

    def test_unsupported_episodes_remain_in_recall_denominator(self):
        episodes = [episode("f1"), episode("f2"), episode("s", True)]
        result = summarize(episodes, [prediction("f1", label=None)])["primary"]
        self.assertEqual(result["failure_recall"], 0)
        self.assertEqual(result["episode_false_alarm_rate"], 0)
        self.assertFalse(result["has_supported_predictions"])
        self.assertEqual(result["unsupported_episodes"], 3)
        self.assertEqual(result["episode_prediction_coverage"], 0)
        self.assertEqual(result["supported_check_fraction"], 0)
        self.assertIsNone(result["median_failure_lead_steps"])

    def test_wilson_uses_episode_counts_and_matches_known_interval(self):
        episodes = [episode(f"f{i}", num_steps=20) for i in range(10)]
        rows = [prediction(f"f{i}", step) for i in range(5) for step in range(20)]
        interval = summarize(episodes, rows)["primary"]["confidence_intervals_95"]["failure_recall"]
        self.assertEqual(interval["numerator"], 5)
        self.assertEqual(interval["denominator"], 10)
        self.assertAlmostEqual(interval["lower"], 0.236593090512564, places=12)
        self.assertAlmostEqual(interval["upper"], 0.763406909487436, places=12)

    def test_empty_and_single_class_data_keep_undefined_rates_null(self):
        empty = summarize([], [])["primary"]
        self.assertIsNone(empty["failure_recall"])
        self.assertIsNone(empty["episode_prediction_coverage"])
        self.assertIsNone(empty["supported_check_fraction"])
        self.assertIsNone(empty["confidence_intervals_95"]["failure_recall"]["lower"])
        only_success = summarize([episode("s", True)], [prediction("s")])["primary"]
        self.assertIsNone(only_success["failure_recall"])
        self.assertEqual(only_success["alarm_precision"], 0)

    def test_validation_rejects_bad_outcomes_duplicate_ids_and_steps(self):
        valid = episode("e")
        for field, value in (("episode_id", ""), ("env_id", ""), ("success", 1),
                             ("num_steps", 0), ("num_steps", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize([{**valid, field: value}], [])
        with self.assertRaisesRegex(ValueError, "duplicate episode_id"):
            summarize([valid, valid], [])
        for field, value in (("episode_id", "unknown"), ("env_id", "other"),
                             ("step", -1), ("step", 4), ("step", True),
                             ("layer", 3), ("prediction", True), ("prediction", 2),
                             ("p_success", math.nan), ("p_success", 1.1), ("p_success", 1.0)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize([valid], [{**prediction("e"), field: value}])
        with self.assertRaisesRegex(ValueError, "duplicate episode-step-layer"):
            summarize([valid], [prediction("e"), prediction("e")])
        missing_label = prediction("e")
        del missing_label["prediction"]
        with self.assertRaisesRegex(ValueError, "label is missing"):
            summarize([valid], [missing_label])
        for layers, primary in (((), 2), ((2, 2), 2), ((2,), 4), ((True,), True)):
            with self.assertRaises(ValueError):
                summarize([valid], [], layers=layers, primary_layer=primary)

    def test_inputs_are_not_modified(self):
        episodes, rows = [episode("e")], [prediction("e", 2), prediction("e", 0)]
        before = copy.deepcopy((episodes, rows))
        summarize(episodes, rows)
        self.assertEqual((episodes, rows), before)


if __name__ == "__main__":
    unittest.main()
