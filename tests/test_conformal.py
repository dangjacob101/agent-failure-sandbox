"""Behavior and finite-sample checks for episode-level conformal sets."""

import copy
import json
import unittest

from early_failure.conformal import TrajectoryConformal


def episode(episode_id, qs, success=True, env_id="env-a"):
    return {
        "episode_id": episode_id,
        "env_id": env_id,
        "success": success,
        "checks": [{"step": step, "q": q} for step, q in enumerate(qs)],
    }


class TrajectoryConformalTests(unittest.TestCase):
    def test_plus_one_and_conservative_ties(self):
        model = TrajectoryConformal(alpha=0.25).fit([
            episode("s1", [0.75]), episode("s2", [0.75]), episode("s3", [0.5]),
            episode("f1", [0.25], False), episode("f2", [0.25], False),
            episode("f3", [0.5], False),
        ])
        self.assertEqual(model.predict("env-a", [0.75])["p_success"], 1.0)
        self.assertEqual(model.predict("env-a", [0.5])["p_success"], 0.5)
        self.assertEqual(model.predict("env-a", [0.25])["p_failure"], 1.0)
        # Minimal p is 1/(n+1), never zero. Equality with alpha is rejected.
        result = model.predict("env-a", [0.0])
        self.assertEqual(result["p_success"], 0.25)
        self.assertEqual(result["prediction_set"], ["failure"])
        self.assertTrue(result["alarm"])

    def test_four_statuses_and_alarm_only_on_singleton_failure(self):
        model = TrajectoryConformal(alpha=0.25).fit([
            episode(f"s{i}", [0.75]) for i in range(3)
        ] + [episode(f"f{i}", [0.25], False) for i in range(3)])
        self.assertEqual(model.predict("env-a", [1.0])["status"], "success")
        self.assertFalse(model.predict("env-a", [1.0])["alarm"])
        self.assertEqual(model.predict("env-a", [0.0])["status"], "failure")
        self.assertEqual(model.predict("env-a", [0.5])["status"], "out_of_distribution")
        self.assertFalse(model.predict("env-a", [0.5])["alarm"])
        permissive = TrajectoryConformal(alpha=0.1).fit([
            episode("s", [0.75]), episode("f", [0.25], False)
        ])
        self.assertEqual(permissive.predict("env-a", [0.5])["status"], "uncertain")
        self.assertFalse(permissive.predict("env-a", [0.5])["alarm"])

    def test_missing_class_is_never_rejected(self):
        successes = TrajectoryConformal(alpha=0.5).fit([episode("s", [0.9])])
        for qs in ([0.0], [1.0], [0.0, 1.0]):
            result = successes.predict("env-a", qs)
            self.assertEqual(result["p_failure"], 1.0)
            self.assertIn("failure", result["prediction_set"])
        failures = TrajectoryConformal(alpha=0.5).fit([episode("f", [0.1], False)])
        result = failures.predict("env-a", [0.0])
        self.assertEqual(result["p_success"], 1.0)
        self.assertFalse(result["alarm"])
        diagnostics = failures.diagnostics()["environments"]["env-a"]
        self.assertEqual(diagnostics["n_episodes"], 1)
        self.assertEqual(diagnostics["success"]["n"], 0)
        self.assertEqual(diagnostics["success"]["min_p"], 1.0)
        self.assertFalse(diagnostics["success"]["supported"])
        self.assertFalse(diagnostics["success"]["can_reject_at_alpha"])

    def test_environments_are_calibrated_separately(self):
        model = TrajectoryConformal(alpha=0.5).fit([
            episode("a", [0.9], env_id="a"), episode("b", [0.1], env_id="b")
        ])
        self.assertEqual(model.predict("a", [0.5])["p_success"], 0.5)
        self.assertEqual(model.predict("b", [0.5])["p_success"], 1.0)
        with self.assertRaisesRegex(ValueError, "unknown"):
            model.predict("new-environment", [0.5])

    def test_running_max_retains_earlier_extremes(self):
        model = TrajectoryConformal(alpha=0.25).fit([
            episode(f"s{i}", [0.5, 0.75]) for i in range(3)
        ] + [episode(f"f{i}", [0.5, 0.25], False) for i in range(3)])
        first = model.predict("env-a", [0.0])
        second = model.predict("env-a", [0.0, 0.25])
        self.assertEqual(first, second)
        self.assertEqual(second["p_success"], 0.25)
        extremes = model.predict("env-a", [0.0, 1.0])
        self.assertEqual(extremes["status"], "out_of_distribution")
        self.assertEqual(extremes, model.predict("env-a", [0.0, 1.0] + [0.5] * 150))

    def test_one_calibration_score_per_episode_not_per_check(self):
        model = TrajectoryConformal().fit([
            episode("s", [0.9] * 150), episode("f", [0.1] * 120, False)
        ])
        diagnostic = model.diagnostics()["environments"]["env-a"]
        self.assertEqual(diagnostic["n_episodes"], 2)
        self.assertEqual(diagnostic["success"]["n"], 1)
        self.assertEqual(diagnostic["failure"]["n"], 1)

    def test_repeated_check_coverage_by_exhaustive_exchangeable_holdout(self):
        # For a uniformly chosen held-out episode among 10 exchangeable
        # episodes, no more than alpha=0.2 may ever reject its true class.
        # Enumerating the held-out choice avoids a probabilistic/flaky test.
        # Each episode has 150 strongly dependent checks and late extremes.
        for success in (True, False):
            label = "success" if success else "failure"
            episodes = []
            for index in range(10):
                maximum = (index + 1) / 16.0
                scores = [maximum / 2] * 99 + [maximum] + [maximum / 4] * 50
                qs = [1.0 - score if success else score for score in scores]
                episodes.append(episode(str(index), qs, success))
            ever_rejected = 0
            for heldout_index, heldout in enumerate(episodes):
                calibration = [item for i, item in enumerate(episodes) if i != heldout_index]
                model = TrajectoryConformal(alpha=0.2).fit(calibration)
                qs = [check["q"] for check in heldout["checks"]]
                terminal_p = model.predict("env-a", qs)[f"p_{label}"]
                reject_any = False
                previous_p = 1.0
                for stop in range(1, len(qs) + 1):
                    result = model.predict("env-a", qs[:stop])
                    p = result[f"p_{label}"]
                    self.assertGreaterEqual(p, terminal_p)
                    self.assertLessEqual(p, previous_p)
                    previous_p = p
                    reject_any |= label not in result["prediction_set"]
                ever_rejected += reject_any
            self.assertEqual(ever_rejected, 2)

    def test_serialization_roundtrip_and_copies(self):
        model = TrajectoryConformal(alpha=0.3).fit([
            episode("s", [0.75, 0.5]), episode("f", [0.25], False, "other")
        ])
        payload = json.loads(json.dumps(model.to_dict()))
        restored = TrajectoryConformal.from_dict(payload)
        self.assertEqual(model.to_dict(), restored.to_dict())
        self.assertEqual(model.diagnostics(), restored.diagnostics())
        self.assertEqual(model.predict("env-a", [0.4]), restored.predict("env-a", [0.4]))
        payload["calibration_scores"]["env-a"]["success"].append(1.0)
        self.assertEqual(len(restored.to_dict()["calibration_scores"]["env-a"]["success"]), 1)
        with self.assertRaises(ValueError):
            TrajectoryConformal().to_dict()

    def test_rejects_malformed_calibration_and_prediction(self):
        valid = episode("valid", [0.5])
        invalid_episodes = [[], [None], [valid, copy.deepcopy(valid)]]
        for field, value in (("episode_id", ""), ("env_id", ""), ("success", 1), ("checks", [])):
            bad = copy.deepcopy(valid)
            bad[field] = value
            invalid_episodes.append([bad])
        for q in (float("nan"), float("inf"), -0.1, 1.1, True, "0.5", None):
            invalid_episodes.append([episode("bad", [q])])
        for steps in ([0, 0], [1, 0], [-1], [True], [1.0]):
            bad = copy.deepcopy(valid)
            bad["checks"] = [{"step": step, "q": 0.5} for step in steps]
            invalid_episodes.append([bad])
        for episodes in invalid_episodes:
            with self.subTest(episodes=episodes), self.assertRaises(ValueError):
                TrajectoryConformal().fit(episodes)
        for alpha in (0, 1, -0.1, 1.1, True, float("nan"), "0.1"):
            with self.subTest(alpha=alpha), self.assertRaises(ValueError):
                TrajectoryConformal(alpha)
        model = TrajectoryConformal().fit([valid])
        for history in ([], "0.5", [None], [float("nan")], [-0.1], [1.1], [True]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                model.predict("env-a", history)

    def test_invalid_refit_preserves_previous_calibration(self):
        model = TrajectoryConformal().fit([episode("s", [0.75])])
        before = model.to_dict()
        with self.assertRaises(ValueError):
            model.fit([episode("s", [0.25]), episode("s", [0.5])])
        self.assertEqual(model.to_dict(), before)

    def test_rejects_corrupt_serialized_state(self):
        payload = TrajectoryConformal().fit([episode("s", [0.5])]).to_dict()
        for key, value in (("schema_version", 2), ("schema_version", True),
                           ("method", "other"), ("alpha", 0), ("calibration_scores", {})):
            bad = copy.deepcopy(payload)
            bad[key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                TrajectoryConformal.from_dict(bad)
        for scores in ({"success": [], "failure": []}, {"success": [0.5]},
                       {"success": [float("nan")], "failure": []}):
            bad = copy.deepcopy(payload)
            bad["calibration_scores"]["env-a"] = scores
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                TrajectoryConformal.from_dict(bad)


if __name__ == "__main__":
    unittest.main()
