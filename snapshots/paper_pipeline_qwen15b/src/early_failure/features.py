"""Build probe examples from observed prefixes, keeping labels outside model inputs."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math

from .history import append_turn, initial_messages
from .types import Observation


def extract_feature_records(episodes, labeler, extractor, *, layers, system_prompt=None,
                            output_path=None, cache=None):
    """Extract only pre-action states; the final outcome never enters the prompt."""
    cache = {} if cache is None else cache
    records = []
    stream = open(output_path, "w") if output_path is not None else None
    try:
        for episode in episodes:
            if len(episode["turns"]) != episode["num_steps"]:
                raise ValueError("saved turn count does not match the episode")
            initial = Observation(**episode["initial_observation"])
            if initial.done:
                raise ValueError("cannot extract from a terminal initial observation")
            prompt = episode.get("system_prompt", system_prompt)
            if not isinstance(prompt, str) or not prompt:
                raise ValueError("the actual rollout system prompt is required")
            messages = initial_messages(prompt, initial)
            cursor = 0
            previous_step = -1
            for check in episode["checks"]:
                step = check["step"]
                if type(step) is not int or not previous_step < step < episode["num_steps"]:
                    raise ValueError("feature checkpoints must increase and precede completion")
                previous_step = step
                while cursor < step:
                    turn = episode["turns"][cursor]
                    obs = Observation(**turn["observation"])
                    if obs.done:
                        raise ValueError("cannot extract a feature after a terminal observation")
                    append_turn(messages, turn["action"], obs)
                    cursor += 1
                prompt_hash = hashlib.sha256(json.dumps(messages, sort_keys=True).encode()).hexdigest()
                key = (prompt_hash, tuple(layers))
                if key not in cache:
                    vectors = extractor.extract_features(deepcopy(messages), layers=tuple(layers))
                    if set(vectors) != set(layers):
                        raise ValueError("extractor did not return exactly the requested layers")
                    vectors = {layer: list(vector) for layer, vector in vectors.items()}
                    for vector in vectors.values():
                        if not vector or any(isinstance(x, bool) or not math.isfinite(x) for x in vector):
                            raise ValueError("activation vectors must contain finite numbers")
                    cache[key] = vectors
                label = labeler.predict(episode["env_id"], step, check["q"])
                for layer in layers:
                    record = {"episode_id": episode["episode_id"], "env_id": episode["env_id"],
                              "step": step, "layer": layer, "features": list(cache[key][layer]),
                              "target": label["target"], "state_label": label,
                              "q": check["q"], "prompt_sha256": prompt_hash}
                    records.append(record)
                    if stream is not None:
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                if stream is not None:
                    stream.flush()
    finally:
        if stream is not None:
            stream.close()
    return records


def label_counts(episodes, labeler):
    """Count each state once, independently of how many layers are extracted."""
    counts, by_step = Counter(), {}
    for episode in episodes:
        for check in episode["checks"]:
            label = labeler.predict(episode["env_id"], check["step"], check["q"])
            counts[label["status"]] += 1
            key = f'{episode["env_id"]}::step={check["step"]}'
            by_step.setdefault(key, Counter())[label["status"]] += 1
    return {"all_states": dict(counts), "by_environment_step": {k: dict(v) for k, v in by_step.items()}}
