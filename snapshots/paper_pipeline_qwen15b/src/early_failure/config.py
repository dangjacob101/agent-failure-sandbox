"""Serializable experiment settings; calibration is tied to the scoring protocol."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    # Model and environment backends can be selected independently.
    agent_backend: str = "hf"
    model_id: str = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
    model_revision: str | None = None
    device: str = "auto"
    dtype: str = "auto"
    env_ids: tuple[str, ...] = ("miniwob/click-test-2-v1", "miniwob/enter-text-v1")
    environment: str = "miniwob"
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    calibration_seeds: tuple[int, ...] = tuple(range(20))
    evaluation_seeds: tuple[int, ...] = tuple(range(1000, 1010))
    max_steps: int = 8
    mc_samples: int = 4
    # None means continue all the way to the episode's remaining step budget.
    mc_max_steps: int | None = None
    check_every: int = 1
    temperature: float = 0.7
    max_new_tokens: int = 128
    max_context_tokens: int = 8192
    alpha: float = 0.1
    seed: int = 42
    output_dir: str = "runs/miniwob_conformal"
    calibration_path: str | None = None
    # Version tags prevent accidental calibration reuse after custom code changes.
    agent_tag: str | None = None
    environment_tag: str | None = None
    scorer_tag: str | None = None

    def __post_init__(self) -> None:
        for name in ("env_ids", "calibration_seeds", "evaluation_seeds"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        for name in ("max_steps", "mc_samples", "check_every", "max_new_tokens", "max_context_tokens"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.mc_max_steps is not None and (
            not isinstance(self.mc_max_steps, int) or isinstance(self.mc_max_steps, bool)
            or self.mc_max_steps < 1
        ):
            raise ValueError("mc_max_steps must be None or a positive integer")
        if not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError("alpha must be between 0 and 1")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        if self.max_context_tokens <= self.max_new_tokens:
            raise ValueError("max_context_tokens must exceed max_new_tokens")
        if not self.env_ids or len(set(self.env_ids)) != len(self.env_ids):
            raise ValueError("env_ids must be nonempty and unique")
        if any(not isinstance(x, str) or not x for x in self.env_ids):
            raise ValueError("env_ids must be nonempty strings")
        for name in ("calibration_seeds", "evaluation_seeds"):
            values = getattr(self, name)
            if not values or len(set(values)) != len(values):
                raise ValueError(f"{name} must be nonempty and unique")
            if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in values):
                raise ValueError(f"{name} must contain nonnegative integers")
        if set(self.calibration_seeds) & set(self.evaluation_seeds):
            raise ValueError("Calibration and evaluation task seeds must be disjoint")
        if self.environment not in ("miniwob", "synthetic", "custom"):
            raise ValueError("environment must be miniwob, synthetic, or custom")
        if self.agent_backend not in ("hf", "synthetic"):
            raise ValueError("agent_backend must be hf or synthetic; custom agents are injected")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, path: str | Path) -> ExperimentConfig:
        return cls(**json.loads(Path(path).read_text()))

    def protocol(self) -> dict[str, Any]:
        """Anything changing the score/outcome distribution invalidates calibration."""
        values = self.to_dict()
        for key in ("calibration_seeds", "evaluation_seeds", "output_dir", "calibration_path", "alpha"):
            values.pop(key)
        # JSON round trip normalizes tuples and provides the same representation on load.
        return json.loads(json.dumps({"version": 2, **values}, sort_keys=True))

    def fingerprint(self) -> str:
        return hashlib.sha256(json.dumps(self.protocol(), sort_keys=True).encode()).hexdigest()
