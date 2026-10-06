"""Opt-in tests against local Chrome and the actual MiniWoB++ HTML.

Run after installing the miniwob extra and Chrome/Chromium:
    EARLY_FAILURE_BROWSER_TESTS=1 python -m unittest discover -s tests \
        -p test_browser_integration.py -v

The delayed-click test intentionally takes over ten seconds to check that model
latency cannot turn a valid MiniWoB action into a wall-clock timeout.
"""

import json
import os
import re
import time
import unittest

from early_failure.miniwob import MiniWoBEnvironment
from early_failure.history import replay_signature
from early_failure.types import Task


def element_ref(observation, *, text=None, element_id=None):
    for element in observation.info["replay_signature"]["dom_elements"]:
        if (text is not None and element.get("text") == text
                or element_id is not None and element.get("id") == element_id):
            return element["ref"]
    raise AssertionError(f"No DOM element for text={text!r}, id={element_id!r}")


@unittest.skipUnless(os.environ.get("EARLY_FAILURE_BROWSER_TESTS") == "1",
                     "Set EARLY_FAILURE_BROWSER_TESTS=1 to open local test browsers")
class BrowserIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.env = MiniWoBEnvironment()
        self.addCleanup(self.env.close)

    def test_click_success_and_failure_with_seed_zero(self):
        for button, expected_success in (("ONE", True), ("TWO", False)):
            with self.subTest(button=button):
                obs = self.env.reset(Task("miniwob/click-test-2-v1", 0))
                self.assertFalse(obs.done)
                ref = element_ref(obs, text=button)
                terminal = self.env.step(f"CLICK(ref={ref})")
                self.assertTrue(terminal.terminated)
                self.assertFalse(terminal.truncated)
                self.assertEqual(terminal.success, expected_success)
                self.assertEqual(terminal.info["raw_reward"], 1.0 if expected_success else -1.0)

    def test_enter_text_and_replay_signatures_at_each_prefix(self):
        task = Task("miniwob/enter-text-v1", 0)
        initial = self.env.reset(task)
        target = re.search(r'Enter "([^"]+)"', initial.text).group(1)
        input_ref = element_ref(initial, element_id="tt")
        submit_ref = element_ref(initial, element_id="subbtn")
        type_action = f"TYPE(ref={input_ref}, text={json.dumps(target)})"
        typed = self.env.step(type_action)
        self.assertFalse(typed.done)
        dom = typed.info["replay_signature"]["dom_elements"]
        self.assertEqual(next(item["value"] for item in dom if item["ref"] == input_ref), target)
        terminal = self.env.step(f"CLICK(ref={submit_ref})")
        self.assertTrue(terminal.success)

        # A fresh browser and seeded task must recreate the same observable state
        # at both the empty prefix and after the already-taken typing action.
        initial_replay = self.env.reset(task)
        self.assertEqual(replay_signature(initial_replay), replay_signature(initial))
        typed_replay = self.env.step(type_action)
        self.assertEqual(replay_signature(typed_replay), replay_signature(typed))
        terminal_replay = self.env.step(f"CLICK(ref={submit_ref})")
        self.assertEqual(replay_signature(terminal_replay), replay_signature(terminal))

    def test_success_after_original_ten_second_wall_clock_limit(self):
        initial = self.env.reset(Task("miniwob/click-test-2-v1", 0))
        ref = element_ref(initial, text="ONE")
        time.sleep(10.5)
        terminal = self.env.step(f"CLICK(ref={ref})")
        self.assertTrue(terminal.success)
        self.assertEqual(terminal.reward, 1.0)
        self.assertEqual(terminal.info["raw_reward"], 1.0)


if __name__ == "__main__":
    unittest.main()
