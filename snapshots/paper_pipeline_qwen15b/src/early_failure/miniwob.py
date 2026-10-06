"""MiniWoB++ DOM adapter for small, replayable browser experiments.

The prompt/action vocabulary follows skarnati20/interp-sandbox. Browser actions
are built through MiniWoB's public API, rather than hard-coded action indices.
Only reviewed static tasks are enabled: replay is not browser-state cloning.
"""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path
from typing import Any

from .types import Observation, Task


STATIC_TASKS = frozenset(
    f"miniwob/{name}-v1"
    for name in (
        "click-test-2", "click-button", "click-button-sequence",
        "click-checkboxes", "click-dialog", "enter-text", "login-user",
    )
)

SYSTEM_PROMPT = '''You are an autonomous web agent completing browser tasks.
Given the goal and visible DOM elements, output ONLY one next action:
  CLICK(ref=<int>)
  TYPE(ref=<int>, text="<str>")
  PRESS(key="<str>") (for example "Enter", "Tab", or "Backspace")
  SCROLL(dx=0, dy=<int>) (positive scrolls down, negative scrolls up)
TYPE focuses the specified element and appends text; it does not clear it.
SCROLL is vertical at the center of the task viewport. Respond with one action.'''

# MiniWoB exposes no Python timeout/pause setting. Install before reset to avoid
# the normal ten-second episode timeout during slow model calls/MC continuations.
# Keep EP_TIMER non-null: core.endEpisode uses it as the active-episode guard.
# This does NOT pause arbitrary task timers; restrict use to STATIC_TASKS.
_DISABLE_EPISODE_CLOCK = """
if (typeof core === 'undefined' || typeof core.startEpisodeReal !== 'function'
    || typeof core.endEpisode !== 'function' || typeof core.clearTimer !== 'function'
    || !('EP_TIMER' in core)) {
  throw new Error('Unsupported MiniWoB core: cannot disable episode clock');
}
if (!core.__earlyFailureClockDisabled) {
  const originalStart = core.startEpisodeReal;
  const originalEnd = core.endEpisode;
  if (!originalStart.toString().includes('core.EP_TIMER = setTimeout')
      || !originalEnd.toString().includes('core.EP_TIMER !== null')) {
    throw new Error('Unsupported MiniWoB timer implementation; inspect adapter hook');
  }
  core.startEpisodeReal = function() {
    const result = originalStart.apply(this, arguments);
    window.clearTimeout(core.EP_TIMER);
    core.clearTimer();
    return result;
  };
  core.endEpisode = function(reward, timeProportional, reason) {
    return originalEnd.call(this, reward, false, reason);
  };
  core.__earlyFailureClockDisabled = true;
}
return core.__earlyFailureClockDisabled;
"""


def _load_backend():
    """Keep statistical tests and synthetic runs independent of browser packages."""
    try:
        import gymnasium
        import miniwob
        from miniwob.reward import get_binary_reward
    except ImportError as exc:
        raise ImportError(
            "MiniWoB runs require the miniwob extra and Chrome/Chromium with a "
            "matching ChromeDriver. Install this package with [miniwob]."
        ) from exc
    gymnasium.register_envs(miniwob)

    def make(env_id, **kwargs):
        if kwargs.get("base_url") is None:
            # Escape spaces in bundled HTML paths so Chrome can navigate to them.
            kwargs["base_url"] = (Path(miniwob.__file__).parent / "html/miniwob").resolve().as_uri() + "/"
        return gymnasium.make(env_id, **kwargs)

    return make, get_binary_reward


def _plain(value: Any) -> Any:
    """Convert NumPy-like values without importing NumPy in the core package."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    return value


def format_dom(elements) -> str:
    """Render visible elements with the references needed by model actions."""
    lines = []
    for element in elements:
        if not element.get("visible", True):
            continue
        attrs = []
        for key in ("text", "value", "id", "classes"):
            value = str(element.get(key, "") or "")
            if value:
                attrs.append(f"{key}={json.dumps(value, ensure_ascii=False)}")
        # Focus and tampered state affect the result of keyboard actions.
        if "flags" in element:
            attrs.append(f"flags={json.dumps(_plain(element['flags']))}")
        tag = element.get("tag", "div")
        lines.append(f"[{element.get('ref', '')}] <{tag}> {' '.join(attrs)} </{tag}>")
    return "\n".join(lines) if lines else "(No visible DOM elements)"


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
    """Parse a single action without executing model-generated Python."""
    text = text.strip()
    if text.isdigit():
        text = f"CLICK({text})"
    if text.upper() in ("SCROLL_DOWN", "SCROLL_UP"):
        return "SCROLL", {"dx": 0, "dy": 50 if text.upper() == "SCROLL_DOWN" else -50}
    try:
        node = ast.parse(text, mode="eval").body
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            raise ValueError("Expected a single action command")
        name = node.func.id.upper()
        parameters = {"CLICK": ("ref",), "TYPE": ("ref", "text"),
                      "PRESS": ("key",), "SCROLL": ("dx", "dy")}
        if name not in parameters:
            raise ValueError(f"Unsupported action: {name}")
        names = parameters[name]
        if len(node.args) > len(names):
            raise ValueError("Too many positional action arguments")
        values = {key: ast.literal_eval(arg) for key, arg in zip(names, node.args)}
        for keyword in node.keywords:
            if keyword.arg not in names or keyword.arg in values:
                raise ValueError("Unknown or repeated action argument")
            values[keyword.arg] = ast.literal_eval(keyword.value)
        if set(values) != set(names):
            raise ValueError(f"Expected arguments: {', '.join(names)}")
        for key in ("ref", "dx", "dy"):
            if key in values and type(values[key]) is not int:
                raise ValueError(f"{key} must be an integer")
        if "ref" in values and values["ref"] <= 0:
            raise ValueError("An interactive element ref must be positive")
        for key in ("text", "key"):
            if key in values and not isinstance(values[key], str):
                raise ValueError(f"{key} must be a string")
        if name == "SCROLL" and values["dx"] != 0:
            raise ValueError("MiniWoB adapter supports vertical scrolling only (dx=0)")
        return name, values
    except (SyntaxError, TypeError) as exc:
        raise ValueError("Invalid action command") from exc


class MiniWoBEnvironment:
    """A fresh browser per reset, using an action budget instead of a wall clock.

    Report this protocol change with results. The rollout layer verifies replay.
    """

    system_prompt = SYSTEM_PROMPT

    def __init__(self, *, headless: bool = True, base_url: str | None = None,
                 wait_ms: float = 150.0, system_prompt: str | None = None):
        if not math.isfinite(wait_ms) or wait_ms < 0:
            raise ValueError("wait_ms must be finite and nonnegative")
        self.headless = headless
        self.base_url = base_url
        self.wait_ms = wait_ms
        if system_prompt is not None:
            if not isinstance(system_prompt, str) or not system_prompt.strip():
                raise ValueError("system_prompt must be a nonempty string")
            self.system_prompt = system_prompt
        self._env = None
        self._done = False

    def reset(self, task: Task) -> Observation:
        self.close()
        if task.env_id not in STATIC_TASKS:
            raise ValueError(
                f"{task.env_id!r} has not been reviewed for static DOM replay. "
                f"Supported tasks: {', '.join(sorted(STATIC_TASKS))}. "
                "Timed, dynamic, canvas and hidden-state tasks need another adapter."
            )
        make, binary_reward = _load_backend()
        self._env = make(
            task.env_id, render_mode=None if self.headless else "human",
            base_url=self.base_url, wait_ms=self.wait_ms,
            reward_processor=binary_reward, refresh_freq=0,
            action_space_config="all_supported",
        )
        try:
            driver = self._env.unwrapped.instance.driver
            if driver.execute_script(_DISABLE_EPISODE_CLOCK) is not True:
                raise RuntimeError("Could not disable MiniWoB's episode clock")
            raw, info = self._env.reset(seed=task.seed, options={"record_screenshots": False})
            self._done = bool(info.get("done", False))
            return self._observation(raw, info, 0.0, self._done, False)
        except BaseException:
            self.close()
            raise

    def step(self, action: str) -> Observation:
        if self._env is None or self._done:
            raise RuntimeError("Call reset() before stepping an inactive/finished episode")
        action_error = None
        try:
            name, values = parse_action(action)
            native = self._create_action(name, values)
        except ValueError as exc:
            # Invalid output consumes a no-op turn; feedback is replayed too.
            action_error = str(exc)
            native = self._env.unwrapped.create_action("NONE")
        raw, reward, terminated, truncated, info = self._env.step(native)
        self._done = bool(terminated or truncated)
        observation = self._observation(raw, info, reward, terminated, truncated)
        if action_error:
            observation.text += f"\nAction error: {action_error}"
            observation.info["action_error"] = action_error
        return observation

    def _create_action(self, name: str, values: dict) -> dict:
        native = self._env.unwrapped
        config = native.action_space_config
        if name == "CLICK":
            return native.create_action("CLICK_ELEMENT", **values)
        if name == "TYPE":
            if len(values["text"]) > config.text_max_len:
                raise ValueError(f"Text exceeds MiniWoB limit ({config.text_max_len} characters)")
            return native.create_action("FOCUS_ELEMENT_AND_TYPE_TEXT", **values)
        if name == "PRESS":
            key = values["key"]
            if key not in config.allowed_keys:
                key = f"<{key}>"
            if key not in config.allowed_keys:
                raise ValueError(f"Unsupported key: {values['key']!r}")
            return native.create_action("PRESS_KEY", key=config.allowed_keys.index(key))
        if values["dy"] == 0:
            return native.create_action("NONE")
        # Wheel distance belongs to the config; coordinates only locate the event.
        config.scroll_amount = abs(values["dy"])
        action_type = "SCROLL_DOWN_COORDS" if values["dy"] > 0 else "SCROLL_UP_COORDS"
        import numpy as np  # Included by MiniWoB; lazy to preserve dependency-free core.
        coords = np.array([config.screen_width / 2, config.screen_height / 2], dtype=np.float32)
        return native.create_action(action_type, coords=coords)

    @staticmethod
    def _observation(raw, info, reward, terminated, truncated) -> Observation:
        if terminated and "raw_reward" not in info:
            raise RuntimeError("Terminal MiniWoB info is missing raw_reward; refusing to infer success")
        success = bool(terminated and not truncated and info.get("raw_reward") == 1.0)
        text = f"Goal: {raw.get('utterance', '')}\n\nVisible DOM Elements:\n{format_dom(raw.get('dom_elements', ()))}"
        # Check the full DOM, excluding wall-clock and object identity fields.
        # Matching these observations cannot establish equality of hidden state.
        state = _plain({key: raw.get(key, ()) for key in ("utterance", "fields", "dom_elements")})
        metadata = {"replay_signature": state, "clock_mode": "action_count"}
        for key in ("raw_reward", "reason"):
            if key in info:
                metadata[key] = _plain(info[key])
        return Observation(text=text, reward=float(reward), terminated=bool(terminated),
                           truncated=bool(truncated), success=success, info=metadata)

    def close(self) -> None:
        env, self._env = self._env, None
        self._done = False
        if env is not None:
            env.close()

    def provenance(self) -> dict[str, Any]:
        """Record the browser runtime used for calibration compatibility checks."""
        if self._env is None:
            raise RuntimeError("Call reset() before requesting browser provenance")
        capabilities = getattr(self._env.unwrapped.instance.driver, "capabilities", {})
        driver_version = capabilities.get("chrome", {}).get("chromedriverVersion")
        return {
            "browser": capabilities.get("browserName"),
            "browser_version": capabilities.get("browserVersion"),
            "chromedriver_version": driver_version.split(" ", 1)[0] if driver_version else None,
            "clock_mode": "action_count",
        }
