from __future__ import annotations

from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from early_failure import ExperimentConfig
from early_failure.conformal import TrajectoryConformal
from early_failure.metrics import summarize


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/summarize_outcome_pilot.py"
SPEC = importlib.util.spec_from_file_location("pilot_report", SCRIPT)
pilot_report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot_report)


class PilotReportTests(unittest.TestCase):
    def fixture(self, directory):
        path = Path(directory)
        env = "synthetic/pilot"
        config = ExperimentConfig(environment="synthetic", agent_backend="synthetic", model_id="fixture",
                                  env_ids=(env,), calibration_seeds=(0, 1), evaluation_seeds=(10, 11, 12, 13),
                                  max_steps=3, mc_samples=4, alpha=0.5, output_dir=str(path))

        def episode(seed, success, scores):
            return {"episode_id": f"{env}::seed={seed}", "env_id": env, "seed": seed,
                    "success": success, "num_steps": len(scores), "elapsed_seconds": 10.0,
                    "outcome_reason": "success" if success else "step_budget_exhausted",
                    "turns": [{"step": step, "action": "CLICK(ref=1)"} for step in range(len(scores))],
                    "checks": [{"step": step, "q": q, "elapsed_seconds": 1.0} for step, q in enumerate(scores)]}

        calibration = [episode(0, True, [0.25]), episode(1, False, [0.75])]
        evaluation = [episode(10, True, [1.0]), episode(11, False, [0.0]),
                      episode(12, False, [0.5, 0.5]), episode(13, False, [1.0, 0.0])]
        monitor = TrajectoryConformal(alpha=config.alpha).fit(calibration)
        for row in evaluation:
            for index, check in enumerate(row["checks"]):
                check["prediction"] = monitor.predict(env, [previous["q"] for previous in row["checks"][:index + 1]])
        runtime = {"agent": {"model_id": "fixture"}, "environment": {}, "versions": {}}
        artifacts = {
            "config": config.to_dict(),
            "metadata": {"runtime": runtime, "protocol_fingerprint": config.fingerprint(), "synthetic": True},
            "calibration": {"schema_version": 1, "protocol": config.protocol(),
                            "protocol_fingerprint": config.fingerprint(), "runtime": runtime,
                            "calibration_tasks": [{"env_id": env, "seed": row["seed"]} for row in calibration],
                            "monitor": monitor.to_dict()},
            "summary": {"synthetic": True, "metrics": summarize(evaluation)},
        }
        for name, value in artifacts.items():
            (path / f"{name}.json").write_text(json.dumps(value))
        for name, values in (("calibration", calibration), ("evaluation", evaluation)):
            self.write_episodes(path, name, values)
        return path, calibration, evaluation

    def write_episodes(self, path, split, values):
        (path / f"{split}_episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in values))

    def test_reports_correctness_abstention_and_episode_denominators(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory)
            report, rows = pilot_report.analyze_run(path)
            result = report["evaluation"]
            self.assertEqual(result["check_counts"], {"success_only": 2, "failure_only": 1,
                                                      "uncertain": 2, "empty": 1, "total": 6})
            self.assertEqual(result["singleton_accuracy"], {"numerator": 2, "denominator": 3, "value": 2 / 3})
            self.assertEqual(result["singleton_decision_fraction"]["value"], 0.5)
            self.assertEqual(result["conformal"]["failure_recall"]["value"], 1 / 3)
            self.assertEqual(result["conformal"]["false_alarm_rate"]["denominator"], 1)
            self.assertEqual(result["conformal"]["alarm_precision"]["value"], 1)
            self.assertEqual(result["conformal"]["failure_recall_by_minimum_lead_actions"]["2"]["numerator"], 0)
            self.assertEqual(result["baselines"]["first_q_below_0_5"]["failure_recall"]["numerator"], 2)
            self.assertEqual(result["baselines"]["always_failure_before_first_action"]["false_alarm_rate"]["value"], 1)
            self.assertIsNone(rows[2]["alarm_correct"])
            self.assertEqual(report["elapsed_seconds"]["all_episodes"], 60.0)

    def test_wilson_intervals_and_empty_denominator(self):
        metric = pilot_report.proportion(5, 10)
        self.assertAlmostEqual(metric["wilson_95_percent_interval"][0], 0.2365930905)
        self.assertAlmostEqual(metric["wilson_95_percent_interval"][1], 0.7634069095)
        self.assertIsNone(pilot_report.proportion(0, 0)["value"])
        self.assertIsNone(pilot_report.proportion(0, 0)["wilson_95_percent_interval"])
        self.assertNotIn("wilson_95_percent_interval", pilot_report.proportion(1, 2, interval=False))

    def test_outputs_include_one_csv_row_per_evaluation_episode(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory)
            text = pilot_report.write_report(path)
            self.assertIn("caught failures 1/3", text)
            self.assertIn("2 uncertain, 1 empty", text)
            with (path / "analysis/episode_predictions.csv").open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 4)
            self.assertEqual(json.loads(rows[-1]["prediction_statuses"]), ["success", "out_of_distribution"])
            self.assertEqual((path / "analysis/report.txt").read_text(), text)
            self.assertTrue((path / "analysis/report.json").is_file())

    def test_saved_protocol_is_preferred_and_invalid_copy_does_not_fall_back(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory)
            report, _ = pilot_report.analyze_run(path)
            self.assertEqual(Path(report["comparison_rules_declared_in"]).name,
                             "miniwob_outcome_pilot_protocol.json")
            saved = path / "protocol.json"
            saved.write_text(json.dumps({"fixed_descriptive_baselines": list(pilot_report.BASELINE_NAMES)}))
            report, _ = pilot_report.analyze_run(path)
            self.assertEqual(report["comparison_rules_declared_in"], str(saved.resolve()))
            saved.write_text(json.dumps({"fixed_descriptive_baselines": ["a changed rule"]}))
            with self.assertRaisesRegex(ValueError, "baseline declarations changed"):
                pilot_report.analyze_run(path)

    def test_missing_summary_or_error_prevents_writes(self):
        for mode in ("missing", "error"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, _, _ = self.fixture(directory)
                if mode == "missing":
                    (path / "summary.json").unlink()
                else:
                    (path / "error.json").write_text('{"message": "interrupted"}')
                with self.assertRaises((FileNotFoundError, ValueError)):
                    pilot_report.write_report(path)
                self.assertFalse((path / "analysis").exists())

    def test_missing_duplicate_or_unexpected_episode_is_rejected(self):
        for mode in ("missing", "duplicate", "unexpected"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, _, evaluation = self.fixture(directory)
                if mode == "missing":
                    evaluation.pop()
                elif mode == "duplicate":
                    evaluation.append(deepcopy(evaluation[0]))
                else:
                    evaluation[0]["seed"] = 999
                self.write_episodes(path, "evaluation", evaluation)
                with self.assertRaises(ValueError):
                    pilot_report.analyze_run(path)

    def test_overlapping_splits_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path, _, _ = self.fixture(directory)
            config = json.loads((path / "config.json").read_text())
            config["evaluation_seeds"][0] = config["calibration_seeds"][0]
            (path / "config.json").write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "overlap"):
                pilot_report.analyze_run(path)

    def test_wrong_types_terminal_checks_and_schedule_are_rejected(self):
        for mode in ("boolean_seed", "numeric_success", "terminal_check", "skipped_check", "nan_score"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, _, evaluation = self.fixture(directory)
                if mode == "boolean_seed":
                    evaluation[0]["seed"] = True
                elif mode == "numeric_success":
                    evaluation[0]["success"] = 1
                elif mode == "terminal_check":
                    evaluation[0]["checks"][0]["step"] = evaluation[0]["num_steps"]
                elif mode == "skipped_check":
                    evaluation[-1]["checks"].pop(0)
                else:
                    evaluation[0]["checks"][0]["q"] = float("nan")
                self.write_episodes(path, "evaluation", evaluation)
                with self.assertRaises(ValueError):
                    pilot_report.analyze_run(path)

    def test_stale_predictions_calibration_and_summary_are_rejected(self):
        for mode in ("prediction", "calibration", "summary"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                path, _, evaluation = self.fixture(directory)
                if mode == "prediction":
                    evaluation[0]["checks"][0]["prediction"]["p_success"] = 0.9
                    self.write_episodes(path, "evaluation", evaluation)
                else:
                    file = path / f"{mode}.json"
                    artifact = json.loads(file.read_text())
                    if mode == "calibration":
                        artifact["monitor"]["calibration_scores"]["synthetic/pilot"]["success"] = [0.0]
                    else:
                        artifact["metrics"]["successful_episodes"] = 3
                    file.write_text(json.dumps(artifact))
                with self.assertRaises(ValueError):
                    pilot_report.analyze_run(path)

    def test_impossible_alarm_diagnostics_distinguish_support_and_ties(self):
        cases = (([], "No successful"), ([0] * 2, "Too few"), ([1] * 20, "Ties"), ([0] * 20, None))
        for scores, reason in cases:
            with self.subTest(scores=scores):
                result = pilot_report.score_diagnostics([], [], {"task": {"success": scores, "failure": []}}, 0.1)["task"]
                self.assertAlmostEqual(result["minimum_achievable_p_success_at_nonconformity_1"],
                                       (1 + sum(value == 1 for value in scores)) / (len(scores) + 1))
                if reason:
                    self.assertIn(reason, result["failure_alarms_impossible_reasons"][0])
                    self.assertFalse(result["success_can_be_rejected"])
                else:
                    self.assertEqual(result["failure_alarms_impossible_reasons"], [])
                    self.assertTrue(result["success_can_be_rejected"])
        both = pilot_report.score_diagnostics(
            [], [], {"task": {"success": [0.75, 1, 1], "failure": []}}, 0.1)["task"]
        self.assertEqual(len(both["failure_alarms_impossible_reasons"]), 2)
        self.assertEqual(both["minimum_achievable_p_success_at_nonconformity_1"], 0.75)


if __name__ == "__main__":
    unittest.main()
