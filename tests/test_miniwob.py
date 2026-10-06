"""Contract tests for real adapter semantics without opening a browser."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from early_failure.miniwob import MiniWoBEnvironment, _load_backend, parse_action
from early_failure.types import Task


def raw_observation():
    return {"utterance": "Enter a value", "fields": (("target", "hello"),),
            "dom_elements": ({"ref": 7, "tag": "INPUT_text", "text": "",
                              "value": "", "flags": [1, 0, 0, 1], "left": [10]},)}


class FakeEnv:
    def __init__(self):
        self.unwrapped = self
        self.scripts = []
        self.instance = SimpleNamespace(driver=SimpleNamespace(execute_script=self.script))
        self.action_space_config = SimpleNamespace(
            allowed_keys=("<Enter>", "<Tab>", "C-a"), text_max_len=64,
            screen_width=160, screen_height=210, scroll_amount=50)
        self.actions = []
        self.closed = False
        self.result = (raw_observation(), 0.0, False, False,
                       {"raw_reward": 0.0, "done": False, "elapsed": 100})

    def script(self, script):
        self.scripts.append(script)
        return True

    def reset(self, seed, options=None):
        self.seed = seed
        self.reset_options = options
        return raw_observation(), {"done": False, "raw_reward": 0.0}

    def create_action(self, action_type, **kwargs):
        return {"action_type": action_type, **kwargs}

    def step(self, action):
        self.actions.append(action)
        return self.result

    def close(self):
        self.closed = True


class MiniWoBTests(unittest.TestCase):
    def setUp(self):
        self.native = FakeEnv()
        self.settings = None

        def make(env_id, **kwargs):
            self.settings = kwargs
            return self.native

        self.backend = patch("early_failure.miniwob._load_backend", return_value=(make, object()))
        self.backend.start()
        self.addCleanup(self.backend.stop)
        self.env = MiniWoBEnvironment()
        self.addCleanup(self.env.close)

    def reset(self):
        return self.env.reset(Task(env_id="miniwob/enter-text-v1", seed=0))

    def test_reset_seed_zero_headless_and_clock(self):
        self.reset()
        self.assertEqual(self.native.seed, 0)
        self.assertEqual(self.native.reset_options, {"record_screenshots": False})
        self.assertIsNone(self.settings["render_mode"])
        self.assertEqual(self.settings["refresh_freq"], 0)
        self.assertIn("window.clearTimeout(core.EP_TIMER)", self.native.scripts[0])
        self.assertNotIn("core.EP_TIMER = null", self.native.scripts[0])

    def test_type_focuses_requested_element_and_press_uses_key_index(self):
        self.reset()
        self.env.step('TYPE(ref=7, text="hello")')
        self.assertEqual(self.native.actions[-1],
                         {"action_type": "FOCUS_ELEMENT_AND_TYPE_TEXT", "ref": 7, "text": "hello"})
        self.env.step('PRESS(key="Enter")')
        self.assertEqual(self.native.actions[-1], {"action_type": "PRESS_KEY", "key": 0})

    def test_partial_terminal_reward_is_failure_and_full_reward_is_success(self):
        self.reset()
        self.native.result = (raw_observation(), -1.0, True, False, {"raw_reward": 0.5})
        self.assertFalse(self.env.step("CLICK(7)").success)
        self.reset()
        self.native.result = (raw_observation(), 1.0, True, False, {"raw_reward": 1.0})
        self.assertTrue(self.env.step("CLICK(7)").success)

    def test_nonterminal_reward_does_not_label_success(self):
        self.reset()
        self.native.result = (raw_observation(), 1.0, False, False, {"raw_reward": 1.0})
        self.assertFalse(self.env.step("CLICK(7)").success)

    def test_truncation_is_not_success(self):
        self.reset()
        self.native.result = (raw_observation(), 1.0, True, True, {"raw_reward": 1.0})
        self.assertFalse(self.env.step("CLICK(7)").success)

    def test_scroll_uses_wheel_direction_distance_and_valid_viewport_center(self):
        self.reset()
        numpy_stub = SimpleNamespace(array=lambda values, dtype: values, float32=float)
        with patch.dict("sys.modules", {"numpy": numpy_stub}):
            self.env.step("SCROLL(dx=0, dy=-100)")
        self.assertEqual(self.native.actions[-1],
                         {"action_type": "SCROLL_UP_COORDS", "coords": [80.0, 105.0]})
        self.assertEqual(self.native.action_space_config.scroll_amount, 100)

    def test_terminal_reward_missing_fails_closed(self):
        self.reset()
        self.native.result = (raw_observation(), 1.0, True, False, {})
        with self.assertRaisesRegex(RuntimeError, "missing raw_reward"):
            self.env.step("CLICK(7)")

    def test_invalid_action_consumes_noop_and_provides_feedback(self):
        self.reset()
        obs = self.env.step("CLICK(7); import os")
        self.assertEqual(self.native.actions[-1], {"action_type": "NONE"})
        self.assertIn("Action error:", obs.text)
        self.assertIn("action_error", obs.info)

    def test_replay_signature_retains_dom_geometry_but_omits_elapsed(self):
        obs = self.reset()
        self.assertEqual(obs.info["replay_signature"]["dom_elements"][0]["left"], [10])
        self.assertNotIn("elapsed", self.env.step("CLICK(7)").info)

    def test_dynamic_task_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "not been reviewed"):
            self.env.reset(Task(env_id="miniwob/chase-circle-v1", seed=1))

    def test_provenance_records_resolved_browser_versions(self):
        with self.assertRaisesRegex(RuntimeError, "reset"):
            self.env.provenance()
        self.reset()
        self.native.instance.driver.capabilities = {
            "browserName": "chrome", "browserVersion": "123.0.0.0",
            "chrome": {"chromedriverVersion": "123.0.0.0 (revision)"},
        }
        self.assertEqual(self.env.provenance(), {
            "browser": "chrome", "browser_version": "123.0.0.0",
            "chromedriver_version": "123.0.0.0", "clock_mode": "action_count",
        })

    def test_failed_clock_install_closes_browser(self):
        self.native.instance.driver.execute_script = lambda script: False
        with self.assertRaisesRegex(RuntimeError, "episode clock"):
            self.reset()
        self.assertTrue(self.native.closed)

    def test_parser_handles_escaped_text_and_rejects_execution(self):
        self.assertEqual(parse_action('TYPE(7, "a\\\"b")'), ("TYPE", {"ref": 7, "text": 'a"b'}))
        for action in ('TYPE(7, __import__("os"))', 'CLICK(ref=True)',
                       'CLICK(ref=1, ref=2)', 'SCROLL(dx=3, dy=100)', 'CLICK(-1)'):
            with self.subTest(action=action), self.assertRaises(ValueError):
                parse_action(action)


class BackendLoadingTests(unittest.TestCase):
    def test_packaged_html_uri_escapes_spaces_and_preserves_explicit_base_url(self):
        calls = []
        fake_gym = SimpleNamespace(register_envs=lambda module: None,
                                   make=lambda env_id, **kwargs: calls.append(kwargs))
        fake_miniwob = SimpleNamespace(__file__="/private/tmp/Mini WoB/miniwob/__init__.py")
        fake_reward = SimpleNamespace(get_binary_reward=lambda metadata: 1.0)
        with patch.dict("sys.modules", {"gymnasium": fake_gym, "miniwob": fake_miniwob,
                                        "miniwob.reward": fake_reward}):
            make, _ = _load_backend()
            make("miniwob/click-test-2-v1", base_url=None)
            make("miniwob/click-test-2-v1", base_url="http://127.0.0.1:8080/miniwob/")
        self.assertEqual(calls[0]["base_url"], "file:///private/tmp/Mini%20WoB/miniwob/html/miniwob/")
        self.assertEqual(calls[1]["base_url"], "http://127.0.0.1:8080/miniwob/")


if __name__ == "__main__":
    unittest.main()
