"""Three-stage integration with real linear fits and fake model/environment I/O."""

from contextlib import redirect_stdout
import copy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from early_failure import ExperimentConfig, ProbeConfig, run_paper_experiment
from early_failure.features import extract_feature_records
from early_failure.types import Observation, Score


class FixtureAgent:
    def __init__(self):
        self.inputs = []

    def generate(self, messages, *, seed, temperature):
        self.inputs.append(copy.deepcopy(messages))
        return "NEXT_ACTION"


class FixtureEnvironment:
    system_prompt = "Act using the visible signal."

    def __init__(self, resets, output):
        self.resets = resets
        self.output = output

    def reset(self, task):
        self.task = task
        self.step_number = 0
        self.resets.append(task.seed)
        if task.seed >= 200:
            assert (self.output / "probes.json").exists(), "evaluation started before probes were saved"
        return Observation(f"visible_signal={1 if task.seed % 2 else -1}")

    def step(self, action):
        self.step_number += 1
        if self.step_number == 1:
            return Observation(f"observed_first_action; visible_signal={1 if self.task.seed % 2 else -1}")
        # Training/test outcomes intentionally disagree with state labels.
        success = bool(self.task.seed % 2) if self.task.seed < 100 else not bool(self.task.seed % 2)
        return Observation("TERMINAL_OUTCOME_SECRET", terminated=True, success=success)

    def close(self):
        pass


class FixtureScorer:
    def score(self, prefix):
        return Score(0.9 if prefix.task.seed % 2 else 0.1,
                     details={"hypothetical_only": "MC_BRANCH_SECRET"})


class ResetPromptEnvironment(FixtureEnvironment):
    @property
    def system_prompt(self):
        assert hasattr(self, "task"), "prompt accessed before reset"
        return f"Act for {self.task.env_id}, seed={self.task.seed}."


class FixtureExtractor:
    def __init__(self):
        self.inputs = []

    def extract_features(self, messages, *, layers):
        self.inputs.append(copy.deepcopy(messages))
        signal = 1.0 if messages[1]["content"].endswith("=1") else -1.0
        return {layer: [signal * 3, float(layer)] for layer in layers}


class FixtureLabeler:
    def predict(self, env_id, step, q):
        return {"status": "success" if q > 0.5 else "failure", "target": int(q > 0.5)}


def json_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


class PaperPipelineTests(unittest.TestCase):
    def config(self, output, **changes):
        return replace(ExperimentConfig(
            environment="custom", env_ids=("fixture-task",), max_steps=2,
            calibration_seeds=tuple(range(40)), evaluation_seeds=tuple(range(200, 206)),
            alpha=0.1, output_dir=str(output), agent_tag="fixture-agent-v1",
            environment_tag="fixture-environment-v1", scorer_tag="fixture-score-v1",
        ), **changes)

    def settings(self, **changes):
        return replace(ProbeConfig(
            training_seeds=tuple(range(100, 108)), layers=(1, 2), primary_layer=1,
            min_per_class=2, feature_extractor_tag="fixture-features-v1",
        ), **changes)

    def run_fixture(self, output, *, config=None, settings=None):
        resets, extractor = [], FixtureExtractor()
        with redirect_stdout(io.StringIO()):
            result = run_paper_experiment(
                config or self.config(output), probe_config=settings or self.settings(),
                agent=FixtureAgent(), environment_factory=lambda: FixtureEnvironment(resets, output),
                scorer=FixtureScorer(), feature_extractor=extractor,
            )
        return result, resets, extractor

    def test_three_stages_train_real_probes_without_replacing_targets_with_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            result, resets, extractor = self.run_fixture(output)
            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["trained_probes"], 4)
            self.assertEqual(result["outcomes"], {
                "calibration": {"episodes": 40, "success": 20, "failure": 20},
                "training": {"episodes": 8, "success": 4, "failure": 4},
                "evaluation": {"episodes": 6, "success": 3, "failure": 3},
            })
            self.assertEqual(resets, list(range(40)) + list(range(100, 108)) + list(range(200, 206)))
            artifact = json.loads((output / "probes.json").read_text())
            self.assertEqual(artifact["provenance"]["estimator"], "sklearn.linear_model.LogisticRegression")
            self.assertTrue(artifact["provenance"]["sklearn_version"])
            self.assertTrue(all(probe["coefficients"] and probe["converged"] for probe in artifact["probes"]))
            evaluation = result["probe_evaluation"]
            self.assertEqual(evaluation["n_evaluated"], 24)
            self.assertEqual(evaluation["metrics"]["accuracy"], 1.0)
            self.assertEqual(evaluation["label_coverage"], 1.0)
            outcomes = result["outcome_evaluation"]["primary"]
            self.assertEqual(outcomes["failure_recall"], 0.0)
            self.assertEqual(outcomes["episode_false_alarm_rate"], 1.0)
            for split in ("training", "evaluation"):
                real = {row["episode_id"]: row["success"] for row in json_lines(output / f"{split}_episodes.jsonl")}
                records = json_lines(output / f"{split}_features.jsonl")
                self.assertTrue(all(row["target"] == int(row["q"] > 0.5) for row in records))
                self.assertTrue(all(row["target"] != int(real[row["episode_id"]]) for row in records))
                self.assertTrue(all("success" not in row and "outcome_reason" not in row for row in records))
            flattened = json.dumps(extractor.inputs)
            self.assertNotIn("TERMINAL_OUTCOME_SECRET", flattened)
            self.assertNotIn("MC_BRANCH_SECRET", flattened)
            self.assertNotIn("final_outcome", flattened)
            self.assertEqual({len(history) for history in extractor.inputs}, {2, 4})
            metadata = json.loads((output / "metadata.json").read_text())
            manifests = [{(row["env_id"], row["seed"]) for row in values}
                         for values in metadata["task_manifests"].values()]
            self.assertEqual(len(set.union(*manifests)), sum(map(len, manifests)))
            self.assertEqual(json.loads((output / "summary.json").read_text()), result)
            self.assertFalse((output / "error.json").exists())

    def test_missing_calibration_class_stops_before_training_or_features(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            config = self.config(output, calibration_seeds=tuple(range(0, 40, 2)))
            result, resets, extractor = self.run_fixture(output, config=config)
            self.assertEqual(result["status"], "insufficient_calibration")
            self.assertEqual(result["trained_probes"], 0)
            self.assertEqual(resets, list(config.calibration_seeds))
            self.assertEqual(extractor.inputs, [])
            for name in ("training_episodes.jsonl", "evaluation_episodes.jsonl", "probes.json"):
                self.assertFalse((output / name).exists())
            self.assertTrue((output / "calibration_diagnostics.json").exists())

    def test_missing_training_class_stops_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            settings = self.settings(training_seeds=tuple(range(100, 108, 2)))
            result, resets, _ = self.run_fixture(output, settings=settings)
            self.assertEqual(result["status"], "insufficient_training_labels")
            self.assertEqual(result["trained_probes"], 0)
            self.assertEqual(resets, list(range(40)) + list(settings.training_seeds))
            self.assertFalse((output / "evaluation_episodes.jsonl").exists())

    def test_task_dependent_prompts_are_recorded_and_replayed_after_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            actor, extractor, resets = FixtureAgent(), FixtureExtractor(), []
            with redirect_stdout(io.StringIO()):
                result = run_paper_experiment(
                    self.config(output, mc_max_steps=2), probe_config=self.settings(),
                    agent=actor, environment_factory=lambda: ResetPromptEnvironment(resets, output),
                    scorer=FixtureScorer(), feature_extractor=extractor,
                )
            self.assertEqual(result["status"], "complete")
            feature_prompts = set()
            for split in ("calibration", "training", "evaluation"):
                for episode in json_lines(output / f"{split}_episodes.jsonl"):
                    actual = f'Act for {episode["env_id"]}, seed={episode["seed"]}.'
                    self.assertEqual(episode["system_prompt"], actual)
                    if split != "calibration":
                        feature_prompts.add(actual)
            self.assertEqual({history[0]["content"] for history in extractor.inputs}, feature_prompts)
            self.assertTrue(all(history in actor.inputs for history in extractor.inputs))
            self.assertEqual(len(extractor.inputs), 2 * (8 + 6))

    def test_short_mc_lookahead_is_rejected_before_environment_or_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            actor, extractor, factory = FixtureAgent(), FixtureExtractor(), Mock()
            with patch("early_failure.paper_pipeline._build_components") as build:
                with self.assertRaisesRegex(ValueError, "full remaining-horizon"):
                    run_paper_experiment(
                        self.config(directory, mc_max_steps=1), probe_config=self.settings(),
                        agent=actor, environment_factory=factory, scorer=FixtureScorer(),
                        feature_extractor=extractor,
                    )
                build.assert_not_called()
            factory.assert_not_called()
            self.assertEqual(actor.inputs, [])
            self.assertEqual(extractor.inputs, [])
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_training_overlap_is_rejected_before_building_components(self):
        for seeds in ((0, 100), (200, 100)):
            with self.subTest(seeds=seeds), tempfile.TemporaryDirectory() as directory:
                with patch("early_failure.paper_pipeline._build_components") as build:
                    with self.assertRaisesRegex(ValueError, "disjoint"):
                        run_paper_experiment(self.config(directory), probe_config=self.settings(training_seeds=seeds))
                    build.assert_not_called()
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_custom_components_require_tags_before_building(self):
        for missing in ("agent_tag", "environment_tag", "scorer_tag", "feature_extractor_tag"):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                config, settings = self.config(directory), self.settings()
                if missing == "feature_extractor_tag":
                    settings = replace(settings, feature_extractor_tag=None)
                else:
                    config = replace(config, **{missing: None})
                with patch("early_failure.paper_pipeline._build_components") as build:
                    with self.assertRaisesRegex(ValueError, missing):
                        run_paper_experiment(config, probe_config=settings, agent=FixtureAgent(),
                                             environment_factory=lambda: None, scorer=FixtureScorer(),
                                             feature_extractor=FixtureExtractor())
                    build.assert_not_called()

    def test_nonempty_output_is_refused_without_loading_or_overwriting(self):
        with tempfile.TemporaryDirectory() as directory:
            sentinel = Path(directory) / "keep.txt"
            sentinel.write_text("existing result")
            with patch("early_failure.paper_pipeline._build_components") as build:
                with self.assertRaises(FileExistsError):
                    run_paper_experiment(self.config(directory), probe_config=self.settings(),
                                         environment_factory=lambda: None)
                build.assert_not_called()
            self.assertEqual(sentinel.read_text(), "existing result")
            self.assertEqual(list(Path(directory).iterdir()), [sentinel])


class FeaturePrefixTests(unittest.TestCase):
    def episode(self):
        return {
            "episode_id": "fixture-task::seed=5", "env_id": "fixture-task", "seed": 5,
            "success": False, "num_steps": 2,
            "initial_observation": asdict(Observation("visible_signal=1")),
            "turns": [
                {"step": 0, "action": "FIRST_ACTION", "observation": asdict(Observation("observed first action"))},
                {"step": 1, "action": "FUTURE_FINAL_ACTION", "observation": asdict(Observation("TERMINAL_OUTCOME_SECRET", terminated=True))},
            ],
            "checks": [{"step": 0, "q": 0.9}, {"step": 1, "q": 0.1}],
        }

    def extract(self, episode, extractor=None, **kwargs):
        return extract_feature_records([episode], FixtureLabeler(), extractor or FixtureExtractor(),
                                       layers=(1, 2), system_prompt="SYSTEM", **kwargs)

    def test_reconstructs_only_past_actions_and_preserves_state_targets(self):
        episode, extractor = self.episode(), FixtureExtractor()
        before = copy.deepcopy(episode)
        records = self.extract(episode, extractor)
        self.assertEqual(extractor.inputs, [
            [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "visible_signal=1"}],
            [{"role": "system", "content": "SYSTEM"}, {"role": "user", "content": "visible_signal=1"},
             {"role": "assistant", "content": "FIRST_ACTION"}, {"role": "user", "content": "observed first action"}],
        ])
        self.assertEqual([row["target"] for row in records], [1, 1, 0, 0])
        self.assertEqual(episode, before)

    def test_identical_prefixes_share_features_but_not_labels(self):
        episode, extractor, cache = self.episode(), FixtureExtractor(), {}
        first = self.extract(episode, extractor, cache=cache)
        altered = copy.deepcopy(episode)
        altered["episode_id"] = "other-task::seed=6"
        altered["success"] = True
        altered["checks"][0]["q"] = 0.1
        second = self.extract(altered, extractor, cache=cache)
        self.assertEqual(len(extractor.inputs), 2)
        self.assertEqual(first[0]["features"], second[0]["features"])
        self.assertNotEqual(first[0]["target"], second[0]["target"])

    def test_saved_episode_prompt_overrides_fallback_and_changes_cache_key(self):
        episode, extractor, cache = self.episode(), FixtureExtractor(), {}
        episode["system_prompt"] = "ACTUAL_EPISODE_PROMPT"
        first = self.extract(episode, extractor, cache=cache)
        second_episode = copy.deepcopy(episode)
        second_episode["episode_id"] = "fixture-task::seed=6"
        second_episode["system_prompt"] = "OTHER_EPISODE_PROMPT"
        second = self.extract(second_episode, extractor, cache=cache)
        self.assertEqual([history[0]["content"] for history in extractor.inputs],
                         ["ACTUAL_EPISODE_PROMPT"] * 2 + ["OTHER_EPISODE_PROMPT"] * 2)
        self.assertNotEqual(first[0]["prompt_sha256"], second[0]["prompt_sha256"])

    def test_rejects_features_after_terminal_observation(self):
        episode = self.episode()
        episode["turns"][0]["observation"]["terminated"] = True
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.extract(episode)

    def test_rejects_terminal_initial_observation(self):
        episode = self.episode()
        episode["initial_observation"]["terminated"] = True
        with self.assertRaisesRegex(ValueError, "terminal"):
            self.extract(episode)

    def test_rejects_duplicate_or_postcompletion_checkpoints(self):
        for steps in ((0, 0), (1, 0), (0, 2)):
            episode = self.episode()
            episode["checks"] = [{"step": step, "q": 0.5} for step in steps]
            with self.subTest(steps=steps), self.assertRaisesRegex(ValueError, "checkpoint"):
                self.extract(episode)

    def test_rejects_wrong_turn_count_or_invalid_activation_vectors(self):
        episode = self.episode()
        episode["num_steps"] = 3
        with self.assertRaisesRegex(ValueError, "turn count"):
            self.extract(episode)
        for result in ({1: [1.0]}, {1: [float("nan")], 2: [2.0]}, {1: [], 2: [2.0]}, {1: [True], 2: [2.0]}):
            extractor = FixtureExtractor()
            extractor.extract_features = lambda messages, *, layers: result
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.extract(self.episode(), extractor)


if __name__ == "__main__":
    unittest.main()
