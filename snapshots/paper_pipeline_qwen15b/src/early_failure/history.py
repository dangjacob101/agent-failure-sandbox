"""Shared conversation updates and reproducible rollout seeds."""
from __future__ import annotations

import hashlib

from .types import Observation


def derive_seed(*parts: object) -> int:
    """Give each main action and hypothetical branch its own stable seed."""
    digest = hashlib.sha256(repr(parts).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def initial_messages(system_prompt: str, obs: Observation) -> list[dict[str, str]]:
    return [{"role": "system", "content": system_prompt},
            {"role": "user", "content": obs.text}]


def append_turn(messages: list[dict[str, str]], action: str, obs: Observation) -> None:
    messages.extend([{"role": "assistant", "content": action},
                     {"role": "user", "content": obs.text}])


def replay_signature(obs: Observation) -> tuple:
    # Adapters can include state omitted from their shorter prompt text.
    return (obs.text, obs.terminated, obs.truncated, obs.success,
            obs.info.get("replay_signature"))
