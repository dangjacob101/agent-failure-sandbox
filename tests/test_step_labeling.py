"""Intermediate labels must follow the selected method, including abstentions."""
from copy import deepcopy
import json
import unittest

from early_failure.step_labeling import StepwiseLabeler


def episode(name, qs, success=True, env="env"):
    return {"episode_id": name, "env_id": env, "success": success, "num_steps": len(qs),
            "checks": [{"step": step, "q": q} for step, q in enumerate(qs)]}


def calibration(success_qs, failure_qs, count=4, env="env"):
    return ([episode(f"{env}-s{i}", success_qs, env=env) for i in range(count)]
            + [episode(f"{env}-f{i}", failure_qs, False, env) for i in range(count)])


class StepwiseLabelingTests(unittest.TestCase):
    def test_default_is_paper_and_alpha_equality_is_retained(self):
        model = StepwiseLabeler(alpha=0.25).fit(calibration([0.75], [0.25], count=3))
        self.assertEqual(model.mode, "paper_equations")
        result = model.predict("env", 0, 0)
        self.assertEqual(result["details"]["p_success"], 0.25)
        self.assertEqual(result["details"]["prediction_set"], ["success", "failure"])
        self.assertIsNone(result["target"])
        self.assertEqual(result["status"], "uncertain")
        model.fit(calibration([0.75], [0.25], count=4))
        result = model.predict("env", 0, 0)
        self.assertEqual(result["details"]["p_success"], 0.2)
        self.assertEqual(result["target"], 0)

    def test_rank_uses_plus_one_and_counts_equal_scores(self):
        data = [episode("s0", [0.75]), episode("s1", [0.75]), episode("s2", [0.5]),
                episode("f0", [0.25], False), episode("f1", [0.25], False), episode("f2", [0.5], False)]
        model = StepwiseLabeler(alpha=0.3).fit(data)
        self.assertEqual(model.predict("env", 0, 0.75)["details"]["p_success"], 1)
        self.assertEqual(model.predict("env", 0, 0.5)["details"]["p_success"], 0.5)
        self.assertEqual(model.predict("env", 0, 0.25)["details"]["p_failure"], 1)
        self.assertEqual(model.predict("env", 0, 1)["details"]["p_failure"], 0.25)

    def test_success_failure_uncertain_and_empty_keep_correct_targets(self):
        overlapping = StepwiseLabeler(alpha=0.3).fit(calibration([0.25], [0.75]))
        for q, status, target in ((0, "failure", 0), (1, "success", 1), (0.5, "uncertain", None)):
            result = overlapping.predict("env", 0, q)
            self.assertEqual((result["status"], result["target"]), (status, target))
        separated = StepwiseLabeler(alpha=0.3).fit(calibration([0.9], [0.1]))
        result = separated.predict("env", 0, 0.5)
        self.assertEqual(result["status"], "empty")
        self.assertIsNone(result["target"])
        self.assertEqual(result["details"]["prediction_set"], [])

    def test_timestep_and_environment_groups_are_independent(self):
        data = calibration([0.8, 0.2], [0.2, 0.8]) + calibration([0.2], [0.8], env="other")
        model = StepwiseLabeler(alpha=0.3).fit(data)
        self.assertEqual(model.predict("env", 0, 0.5)["status"], "empty")
        self.assertEqual(model.predict("env", 1, 0.5)["status"], "uncertain")
        self.assertEqual(model.predict("other", 0, 0.5)["status"], "uncertain")
        # An earlier extreme is not carried into a later prediction.
        model.predict("env", 0, 0)
        self.assertEqual(model.predict("env", 1, 1)["status"], "success")
        self.assertEqual(model.diagnostics()["environments"]["env"]["1"]["support"], {"success": 4, "failure": 4})

    def test_real_final_outcome_groups_ignore_any_existing_check_labels(self):
        data = calibration([0.2], [0.8])
        for row in data:
            row["checks"][0]["prediction"] = {"status": "failure" if row["success"] else "success"}
        model = StepwiseLabeler().fit(data)
        saved = model.to_dict()["calibration_rewards"]["env"]["0"]
        self.assertEqual(saved["success"], [0.2] * 4)
        self.assertEqual(saved["failure"], [0.8] * 4)

    def test_missing_class_in_paper_mode_never_creates_a_target(self):
        model = StepwiseLabeler(alpha=0.5).fit([episode(f"s{i}", [1]) for i in range(3)])
        result = model.predict("env", 0, 0)
        self.assertEqual(result["details"]["prediction_set"], ["failure"])
        self.assertEqual(result["status"], "unsupported")
        self.assertIsNone(result["target"])
        self.assertEqual(result["details"]["support"], {"success": 3, "failure": 0})
        self.assertFalse(model.diagnostics()["environments"]["env"]["0"]["failure_target_possible"])

    def test_later_timesteps_count_only_episodes_that_reach_them(self):
        data = [episode("s", [0.7]), episode("f", [0.2, 0.4], False)]
        model = StepwiseLabeler().fit(data)
        later = model.diagnostics()["environments"]["env"]["1"]
        self.assertEqual(later["support"], {"success": 0, "failure": 1})
        self.assertIsNone(model.predict("env", 1, 0)["target"])
        for mode in ("paper_equations", "upstream_percentiles"):
            result = StepwiseLabeler(mode).fit(data).predict("env", 5, 1)
            self.assertEqual(result["status"], "unsupported")
            self.assertIsNone(result["target"])

    def test_diagnostics_show_strict_rank_limit_and_zero_score_ties(self):
        small = StepwiseLabeler().fit(calibration([1], [0], count=9))
        diagnostic = small.diagnostics()["environments"]["env"]["0"]
        self.assertEqual(diagnostic["classes"]["success"]["min_rank_p_without_ties"], 0.1)
        self.assertFalse(diagnostic["failure_target_possible"])
        self.assertIn("Too few", diagnostic["reasons"][0])
        enough = StepwiseLabeler().fit(calibration([1], [0], count=10))
        self.assertTrue(enough.diagnostics()["environments"]["env"]["0"]["failure_target_possible"])
        tied = StepwiseLabeler().fit(calibration([0], [0], count=20))
        diagnostic = tied.diagnostics()["environments"]["env"]["0"]
        self.assertEqual(diagnostic["classes"]["success"]["maximal_nonconformity_ties"], 20)
        self.assertEqual(diagnostic["classes"]["success"]["min_achievable_p_at_nonconformity_1"], 1)
        self.assertFalse(diagnostic["failure_target_possible"])
        self.assertIn("Ties", diagnostic["reasons"][0])

    def test_upstream_uses_released_percentile_directions_and_linear_interpolation(self):
        data = [episode(f"s{i}", [q]) for i, q in enumerate((0, 0.1, 0.3, 1))]
        data += [episode(f"f{i}", [q], False) for i, q in enumerate((0, 0.2, 0.5, 0.7))]
        model = StepwiseLabeler("upstream_percentiles", alpha=0.1).fit(data)
        details = model.predict("env", 0, 0.5)["details"]
        self.assertAlmostEqual(details["t_s"], 0.79)
        self.assertAlmostEqual(details["t_f"], 0.06)
        self.assertEqual(details["coverage"], 0.9)
        self.assertFalse(details["formal_coverage_claim"])
        self.assertEqual(model.predict("env", 0, details["t_s"])["target"], 1)
        self.assertEqual(model.predict("env", 0, details["t_f"])["target"], 0)
        self.assertIsNone(model.predict("env", 0, 0.5)["target"])

    def test_upstream_prefers_success_when_thresholds_overlap(self):
        model = StepwiseLabeler("upstream_percentiles").fit(calibration([0.2], [0.8]))
        self.assertEqual(model.predict("env", 0, 0.5)["target"], 1)
        self.assertEqual(model.predict("env", 0, 0.2)["target"], 1)
        self.assertEqual(model.predict("env", 0, 0.1)["target"], 0)

    def test_upstream_empty_class_defaults_are_explicit(self):
        only_failures = StepwiseLabeler("upstream_percentiles").fit([episode("f", [0.2], False)])
        result = only_failures.predict("env", 0, 1)
        self.assertEqual(result["target"], 1)
        self.assertEqual(result["details"]["t_s"], 1)
        self.assertFalse(result["details"]["both_classes_supported"])
        self.assertTrue(result["details"]["used_empty_class_defaults"])
        only_successes = StepwiseLabeler("upstream_percentiles").fit([episode("s", [0.8])])
        result = only_successes.predict("env", 0, 0)
        self.assertEqual(result["target"], 0)
        self.assertEqual(result["details"]["t_f"], 0)

    def test_both_modes_roundtrip_json_without_sharing_mutable_rewards(self):
        for mode in ("paper_equations", "upstream_percentiles"):
            with self.subTest(mode=mode):
                model = StepwiseLabeler(mode, alpha=0.3).fit(calibration([0.8, 0.3], [0.2, 0.7]))
                artifact = json.loads(json.dumps(model.to_dict()))
                restored = StepwiseLabeler.from_dict(artifact)
                self.assertEqual(model.diagnostics(), restored.diagnostics())
                self.assertEqual(model.predict("env", 1, 0.5), restored.predict("env", 1, 0.5))
                artifact["calibration_rewards"]["env"]["0"]["success"].append(0)
                self.assertEqual(restored.to_dict()["support"]["env"]["0"]["success"], 4)

    def test_malformed_calibration_and_invalid_queries_are_rejected(self):
        valid = episode("s", [0.5])
        invalid = [[], [None], [valid, deepcopy(valid)]]
        for key, value in (("success", 1), ("env_id", ""), ("episode_id", ""), ("checks", []),
                           ("num_steps", 0), ("num_steps", True)):
            bad = deepcopy(valid)
            bad[key] = value
            invalid.append([bad])
        for value in (True, "0.5", -1, 1.1, float("nan"), float("inf")):
            invalid.append([episode("s", [value])])
        for steps in ([0, 0], [1, 0], [0, 2]):
            bad = episode("s", [0.5, 0.5])
            for check, step in zip(bad["checks"], steps):
                check["step"] = step
            invalid.append([bad])
        model = StepwiseLabeler().fit([valid])
        original = model.to_dict()
        for data in invalid:
            with self.subTest(data=data), self.assertRaises(ValueError):
                model.fit(data)
            self.assertEqual(model.to_dict(), original)
        for args in (("other", 0, 0.5), ("env", True, 0.5), ("env", -1, 0.5), ("env", 0, True)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                model.predict(*args)

    def test_bad_settings_and_serialized_artifacts_are_rejected(self):
        for kwargs in ({"mode": "other"}, {"alpha": 0}, {"alpha": 1}, {"alpha": True}, {"alpha": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                StepwiseLabeler(**kwargs)
        with self.assertRaises(ValueError):
            StepwiseLabeler().to_dict()
        artifact = StepwiseLabeler().fit(calibration([0.8], [0.2])).to_dict()
        bad_artifacts = [None, {}]
        for key, value in (("schema_version", True), ("method", "other"), ("calibration_rewards", {}),
                           ("support", {}), ("alpha", 0)):
            bad = deepcopy(artifact)
            bad[key] = value
            bad_artifacts.append(bad)
        bad = deepcopy(artifact)
        bad["calibration_rewards"]["env"]["00"] = bad["calibration_rewards"]["env"].pop("0")
        bad_artifacts.append(bad)
        for bad in bad_artifacts:
            with self.subTest(artifact=bad), self.assertRaises(ValueError):
                StepwiseLabeler.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
