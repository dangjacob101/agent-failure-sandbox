from __future__ import annotations

from dataclasses import replace
import unittest

from early_failure.history import append_turn, initial_messages
from early_failure.scoring import MonteCarloScorer, ReplayMismatch
from early_failure.synthetic import SyntheticEnvironment
from early_failure.types import EpisodePrefix, Task


class AdvancingAgent:
    def __init__(self):
        self.seeds = []

    def generate(self, messages, *, seed, temperature):
        self.seeds.append(seed)
        return "advance"


class ScoringTests(unittest.TestCase):
    def prefix(self, *, length=3, actions=(), remaining_steps=None):
        task = Task("synthetic/sequence", 0)
        env = SyntheticEnvironment(length)
        obs = env.reset(task)
        messages, observations = initial_messages(env.system_prompt, obs), [obs]
        for action in actions:
            obs = env.step(action)
            append_turn(messages, action, obs)
            observations.append(obs)
        env.close()
        return EpisodePrefix(task=task, actions=tuple(actions),
                             observations=tuple(observations), messages=tuple(messages),
                             remaining_steps=length - len(actions)
                             if remaining_steps is None else remaining_steps)

    def test_successful_branches_replay_prefix_and_close(self):
        created = []

        def factory():
            env = SyntheticEnvironment(3)
            created.append(env)
            return env

        agent = AdvancingAgent()
        prefix = self.prefix(actions=("advance",))
        scorer = MonteCarloScorer(agent, factory, samples=3)
        first = scorer.score(prefix)
        self.assertEqual(first.q, 1.0)
        self.assertEqual(first.details["generated_actions"], 6)
        self.assertEqual(first.details["mc_censored"], 0)
        self.assertEqual(len(set(agent.seeds)), 6)
        self.assertEqual(scorer.score(prefix), first)
        self.assertEqual(agent.seeds[:6], agent.seeds[6:])
        self.assertTrue(all(env.closed for env in created))

    def test_full_budget_failure_and_short_lookahead_censoring(self):
        prefix = self.prefix(length=5, remaining_steps=2)
        factory = lambda: SyntheticEnvironment(5)
        full = MonteCarloScorer(AdvancingAgent(), factory, samples=2).score(prefix)
        short = MonteCarloScorer(AdvancingAgent(), factory, samples=2,
                                max_lookahead=1).score(prefix)
        self.assertEqual(full.q, 0.0)
        self.assertEqual(full.details["mc_censored"], 0)
        self.assertEqual(short.q, 0.0)
        self.assertEqual(short.details["mc_censored"], 2)
        self.assertEqual(short.details["lookahead_steps"], 1)

    def test_terminal_failure_is_not_censored(self):
        prefix = self.prefix(length=2, actions=("mistake",), remaining_steps=4)
        score = MonteCarloScorer(AdvancingAgent(), lambda: SyntheticEnvironment(2),
                                max_lookahead=1).score(prefix)
        self.assertEqual(score.q, 0.0)
        self.assertEqual(score.details["mc_censored"], 0)

    def test_replay_mismatch_after_action_closes_environment(self):
        env = SyntheticEnvironment(3)
        prefix = self.prefix(actions=("advance",))
        altered = replace(prefix.observations[-1], text="different state")
        prefix = replace(prefix, observations=(prefix.observations[0], altered))
        with self.assertRaisesRegex(ReplayMismatch, "after action 1"):
            MonteCarloScorer(AdvancingAgent(), lambda: env).score(prefix)
        self.assertTrue(env.closed)

    def test_generation_error_propagates_and_closes_environment(self):
        class BrokenAgent:
            def generate(self, messages, *, seed, temperature):
                raise RuntimeError("generation failed")

        env = SyntheticEnvironment(3)
        with self.assertRaisesRegex(RuntimeError, "generation failed"):
            MonteCarloScorer(BrokenAgent(), lambda: env).score(self.prefix())
        self.assertTrue(env.closed)

    def test_invalid_prefix_fails_before_branch_creation(self):
        def no_branches():
            self.fail("invalid prefixes must not create environments")

        scorer = MonteCarloScorer(AdvancingAgent(), no_branches)
        prefix = self.prefix()
        for invalid in (replace(prefix, remaining_steps=0),
                        replace(prefix, observations=()),
                        self.prefix(length=1, actions=("advance",), remaining_steps=1)):
            with self.subTest(prefix=invalid):
                with self.assertRaisesRegex(ValueError, "nonterminal prefix"):
                    scorer.score(invalid)

    def test_invalid_scoring_settings(self):
        for settings in ({"samples": 0}, {"samples": True}, {"max_lookahead": 0},
                         {"max_lookahead": False}, {"temperature": float("nan")},
                         {"seed": -1}):
            with self.subTest(settings=settings):
                with self.assertRaises(ValueError):
                    MonteCarloScorer(AdvancingAgent(), SyntheticEnvironment, **settings)


if __name__ == "__main__":
    unittest.main()
