from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from early_failure import ExperimentConfig, Score, run_experiment
from early_failure.metrics import summarize
from early_failure.rollouts import collect_episode
from early_failure.scoring import ReplayMismatch
from early_failure.synthetic import SyntheticAgent, SyntheticEnvironment
from early_failure.types import Task


class PipelineTests(unittest.TestCase):
    def config(self, **kwargs):
        return ExperimentConfig(environment="synthetic", agent_backend="synthetic", model_id="synthetic-policy",
                                env_ids=("synthetic/sequence",), calibration_seeds=(0, 1, 2),
                                evaluation_seeds=(10, 11), max_steps=4, mc_samples=2, **kwargs)

    def test_complete_run_saved_calibration_and_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            config = self.config(output_dir=str(path / "first"))
            result = run_experiment(config)
            self.assertEqual(result["metrics"]["episodes"], 2)
            self.assertTrue(result["synthetic"])
            episodes = [json.loads(line) for line in (path / "first/evaluation_episodes.jsonl").read_text().splitlines()]
            for episode in episodes:
                self.assertEqual(episode["num_steps"], 4)
                self.assertEqual([c["step"] for c in episode["checks"]], [0, 1, 2, 3])
                self.assertTrue(all("prediction" in c for c in episode["checks"]))
            again = replace(config, output_dir=str(path / "second"),
                            calibration_path=str(path / "first/calibration.json"))
            second = run_experiment(again)
            self.assertEqual(result["metrics"]["alarms"], second["metrics"]["alarms"])
            self.assertFalse((path / "second/calibration_episodes.jsonl").exists())
            with patch("early_failure.experiment._versions", return_value={"python": "changed-runtime"}):
                with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                    run_experiment(replace(again, output_dir=str(path / "changed-runtime")))
            with self.assertRaises(FileExistsError):
                run_experiment(config)
            for altered in ({"max_steps": 5}, {"mc_samples": 3}, {"model_id": "another"},
                            {"calibration_seeds": (4, 5)}):
                with self.assertRaises(ValueError):
                    run_experiment(replace(again, output_dir=str(path / "bad"), **altered))

    def test_overlap_and_invalid_config(self):
        with self.assertRaises(ValueError):
            ExperimentConfig(calibration_seeds=(0,), evaluation_seeds=(0,))
        for kwargs in ({"mc_samples": 0}, {"check_every": 0}, {"alpha": float("nan")},
                       {"temperature": -1}, {"mc_max_steps": 0}, {"env_ids": ()}):
            with self.assertRaises(ValueError):
                self.config(**kwargs) if not (set(kwargs) & {"mc_samples", "env_ids"}) else ExperimentConfig(**kwargs)

    def test_local_checkpoint_cannot_reuse_an_inherited_hub_revision_without_tag(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "weights").mkdir()
            config = replace(self.config(), model_id=str(path / "weights"), output_dir=str(path / "first"))
            runtime = {"agent": {"resolved_revision": "copied-upstream-commit"}, "versions": {}, "environment": {}}
            with patch("early_failure.experiment._provenance", return_value=runtime):
                run_experiment(config)
                with self.assertRaisesRegex(ValueError, "requires agent_tag"):
                    run_experiment(replace(config, output_dir=str(path / "second"),
                                           calibration_path=str(path / "first/calibration.json")))

    def test_main_actions_do_not_change_with_mc_count_or_schedule(self):
        base = self.config()
        trajectories = []
        for config in (base, replace(base, mc_samples=5), replace(base, check_every=3)):
            episode = collect_episode(SyntheticAgent(), lambda: SyntheticEnvironment(4),
                                      Task("synthetic/sequence", 10), config)
            trajectories.append([t["action"] for t in episode["turns"]])
        self.assertEqual(trajectories[0], trajectories[1])
        self.assertEqual(trajectories[0], trajectories[2])

    def test_replay_error_is_not_failure_and_closes_every_environment(self):
        created = []

        class DriftingEnvironment(SyntheticEnvironment):
            def reset(self, task):
                obs = super().reset(task)
                obs.text += f" session={len(created)}"
                return obs

        def factory():
            env = DriftingEnvironment()
            created.append(env)
            return env

        with self.assertRaises(ReplayMismatch):
            collect_episode(SyntheticAgent(), factory, Task("synthetic/sequence", 3), self.config())
        self.assertTrue(all(env.closed for env in created))

    def test_score_uses_prefix_not_future_outcome_and_censoring(self):
        config = replace(self.config(), max_steps=2, temperature=0)
        full = collect_episode(SyntheticAgent(), lambda: SyntheticEnvironment(5), Task("synthetic/sequence", 1), config)
        self.assertEqual(full["outcome_reason"], "step_budget_exhausted")
        self.assertTrue(all(c["q"] == 0 for c in full["checks"]))
        self.assertTrue(all(c["details"]["mc_censored"] == 0 for c in full["checks"]))
        short = collect_episode(SyntheticAgent(), lambda: SyntheticEnvironment(5), Task("synthetic/sequence", 1),
                                replace(config, mc_max_steps=1))
        self.assertEqual(short["checks"][0]["details"]["mc_censored"], config.mc_samples)
        self.assertEqual(short["checks"][1]["details"]["mc_censored"], 0)

    def test_custom_environment_uses_builtin_model_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config(), environment="custom", agent_backend="hf",
                             model_id="example-model", environment_tag="test-env-v1", output_dir=directory)
            with patch("early_failure.models.HuggingFaceAgent", return_value=SyntheticAgent()) as constructor:
                result = run_experiment(config, environment_factory=lambda: SyntheticEnvironment(4))
            self.assertEqual(constructor.call_args.args, ("example-model",))
            self.assertEqual(result["metrics"]["episodes"], 2)

    def test_custom_agent_uses_builtin_environment_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config(), environment="miniwob", agent_backend="hf",
                             agent_tag="test-agent-v1", output_dir=directory)
            with patch("early_failure.miniwob.MiniWoBEnvironment", side_effect=lambda **kw: SyntheticEnvironment(4)) as constructor:
                with patch("early_failure.models.HuggingFaceAgent") as model_constructor:
                    run_experiment(config, agent=SyntheticAgent())
            self.assertTrue(constructor.called)
            model_constructor.assert_not_called()

    def test_custom_scorer_needs_no_replay_and_cannot_mutate_real_history(self):
        resets = []
        checked_steps = []

        class EnvironmentWithoutReplay(SyntheticEnvironment):
            def reset(self, task):
                self.assert_new_seed(task)
                return super().reset(task)

            def assert_new_seed(self, task):
                if task.seed in resets:
                    raise RuntimeError("This backend cannot replay")
                resets.append(task.seed)

        class CheapScorer:
            def score(self, prefix):
                checked_steps.append(prefix.step)
                prefix.observations[0].text = "scorer changed its private copy"
                return Score(0.75)

        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config(), environment="custom", environment_tag="no-replay-v1",
                             scorer_tag="cheap-scorer-v1", output_dir=directory)
            result = run_experiment(config, environment_factory=lambda: EnvironmentWithoutReplay(4), scorer=CheapScorer())
            self.assertEqual(len(resets), 5)
            self.assertEqual(checked_steps, [0, 1, 2, 3] * 5)
            self.assertEqual(result["metrics"]["mc_generated_actions"], 0)
            self.assertEqual(result["metrics"]["mc_elapsed_seconds"], 0)
            episodes = [json.loads(x) for x in Path(directory, "evaluation_episodes.jsonl").read_text().splitlines()]
            self.assertNotIn("private copy", episodes[0]["initial_observation"]["text"])
            self.assertEqual(episodes[0]["checks"][0]["details"], {})

    def test_custom_environment_runtime_changes_reject_saved_calibration(self):
        version = "v1"

        class VersionedEnvironment(SyntheticEnvironment):
            def provenance(self):
                return {"runtime_version": version}

        with tempfile.TemporaryDirectory() as directory:
            config = replace(self.config(), environment="custom", environment_tag="test-env-v1",
                             output_dir=str(Path(directory, "first")))
            run_experiment(config, environment_factory=lambda: VersionedEnvironment(4))
            version = "v2"
            with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                run_experiment(replace(config, output_dir=str(Path(directory, "second")),
                                       calibration_path=str(Path(directory, "first/calibration.json"))),
                               environment_factory=lambda: VersionedEnvironment(4))

    def test_horizon_over_100_turns_and_pre_action_checks(self):
        config = replace(self.config(), max_steps=120, mc_samples=1, check_every=10, temperature=0)
        episode = collect_episode(SyntheticAgent(), lambda: SyntheticEnvironment(120),
                                  Task("synthetic/sequence", 0), config)
        self.assertTrue(episode["success"])
        self.assertEqual(episode["num_steps"], 120)
        self.assertEqual([c["step"] for c in episode["checks"]], list(range(0, 120, 10)))

    def test_metrics_early_lead_and_class_denominators(self):
        def episode(identifier, success, alarm_step):
            label = "failure" if alarm_step is not None else ("success" if success else "failure")
            return {"episode_id": identifier, "success": success, "num_steps": 120,
                    "checks": [{"step": 10, "generated_actions": 2, "mc_censored": 0,
                                "elapsed_seconds": 1.0, "prediction": {"prediction_set": [label],
                                                                       "alarm": alarm_step is not None}}]}
        result = summarize([episode("failure", False, 10), episode("success", True, None)])
        self.assertEqual(result["episode_false_alarm_rate"], 0)
        self.assertEqual(result["median_failure_lead_steps"], 110)
        self.assertEqual(result["failure_recall_by_minimum_lead_steps"]["100"], 1)
        self.assertIsNone(summarize([episode("failure", False, 10)])["episode_false_alarm_rate"])


if __name__ == "__main__":
    unittest.main()
