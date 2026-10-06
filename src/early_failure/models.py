"""Causal language-model adapter; weights and optional imports load on demand."""

from __future__ import annotations

from contextlib import contextmanager
import math
import random
import threading
from typing import Callable


DEFAULT_MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

# PyTorch RNGs are process-global. Serialize our calls; sampling outside this
# adapter must also avoid concurrency if strict RNG isolation is required.
_GENERATION_LOCK = threading.RLock()


def trim_messages(
    messages: list[dict[str, str]],
    *,
    max_prompt_tokens: int,
    count_tokens: Callable[[list[dict[str, str]]], int],
) -> list[dict[str, str]]:
    """Discard oldest complete action/observation exchanges until a prompt fits.

    Keep system messages, the initial task, and the latest action/observation.
    Only adjacent assistant/user pairs are removed; content is never sliced
    and the caller's history is unchanged. Fail if the protected content cannot fit.
    """
    if max_prompt_tokens <= 0:
        raise ValueError("The context window must leave room for prompt tokens.")
    if not messages:
        raise ValueError("messages must contain a user task.")
    history = []
    for message in messages:
        if not isinstance(message, dict) or not isinstance(message.get("role"), str):
            raise ValueError("Each message must have string role and content fields.")
        if not isinstance(message.get("content"), str):
            raise ValueError("Each message must have string role and content fields.")
        history.append(dict(message))
    if not any(message["role"] == "user" for message in history):
        raise ValueError("messages must contain a user task.")

    while True:
        prompt_tokens = count_tokens(history)
        if prompt_tokens <= max_prompt_tokens:
            return history
        user_indices = [i for i, message in enumerate(history) if message["role"] == "user"]
        protected_users = {user_indices[0], user_indices[-1]}
        removable = next(
            (
                i
                for i in range(len(history) - 1)
                if history[i]["role"] == "assistant"
                and history[i + 1]["role"] == "user"
                and i + 1 not in protected_users
            ),
            None,
        )
        if removable is None:
            raise ValueError(
                f"Irreducible prompt requires {prompt_tokens} tokens, but only "
                f"{max_prompt_tokens} remain after reserving max_new_tokens. "
                "Shorten the task/observation, lower max_new_tokens, or increase "
                "max_context_tokens within the model's actual context limit."
            )
        del history[removable : removable + 2]


@contextmanager
def _preserved_rng(torch, device):
    """Restore caller RNG state, including when inference raises."""
    python_state = random.getstate()
    cpu_state = torch.random.get_rng_state()
    device_state = None
    if device.type == "cuda":
        device_state = torch.cuda.get_rng_state(device)
    elif device.type == "mps":
        device_state = torch.mps.get_rng_state()
    try:
        yield
    finally:
        random.setstate(python_state)
        torch.random.set_rng_state(cpu_state)
        if device.type == "cuda":
            torch.cuda.set_rng_state(device_state, device)
        elif device.type == "mps":
            torch.mps.set_rng_state(device_state)


@contextmanager
def _isolated_rng(torch, device, seed: int):
    """Seed CPU and the selected accelerator, then restore even after an error."""
    with _preserved_rng(torch, device):
        random.seed(seed)
        # Avoid seeding accelerators whose states were not saved.
        torch.random.default_generator.manual_seed(seed)
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        elif device.type == "mps":
            torch.mps.manual_seed(seed)
        yield


class HuggingFaceAgent:
    """A single-device causal LM with seeded generation and bounded chat history.

    ``device='auto'`` prefers CUDA, then MPS, then CPU. Reproducibility assumes
    unchanged hardware and library versions.
    """

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        device: str = "auto",
        dtype: str = "auto",
        max_new_tokens: int = 128,
        max_context_tokens: int = 8192,
        revision: str | None = None,
    ):
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("model_id must be a nonempty model name or local path.")
        if not isinstance(device, str) or not device:
            raise ValueError("device must be auto, cpu, mps, or cuda[:index].")
        if dtype not in {"auto", "float32", "fp32", "float16", "fp16", "bfloat16", "bf16"}:
            raise ValueError("dtype must be auto, float32/fp32, float16/fp16, or bfloat16/bf16.")
        for name, value in (("max_new_tokens", max_new_tokens), ("max_context_tokens", max_context_tokens)):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer.")
        if max_new_tokens >= max_context_tokens:
            raise ValueError("max_context_tokens must exceed max_new_tokens.")
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self.max_context_tokens = max_context_tokens
        self.revision = revision
        self._torch = None
        self._model = None
        self._tokenizer = None
        self._resolved_device = None
        self._resolved_dtype = None
        self._effective_context_tokens = max_context_tokens

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "HuggingFaceAgent needs the optional model dependencies; install "
                "the failure_sandbox package with its [hf] extra."
            ) from exc

        mps_available = bool(
            getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
        )
        selected = self.device
        if selected == "auto":
            selected = "cuda" if torch.cuda.is_available() else ("mps" if mps_available else "cpu")
        device = torch.device(selected)
        if device.type not in {"cpu", "cuda", "mps"}:
            raise ValueError("Supported devices are cpu, cuda[:index], and mps.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "cuda" and device.index is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}")
        if device.type == "mps" and not mps_available:
            raise RuntimeError("MPS was requested but is not available.")

        dtype_name = {"fp32": "float32", "fp16": "float16", "bf16": "bfloat16"}.get(self.dtype, self.dtype)
        if dtype_name == "auto":
            dtype_name = "float32"
            if device.type == "mps":
                dtype_name = "float16"
            elif device.type == "cuda":
                with torch.cuda.device(device):
                    dtype_name = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
        pretrained_kwargs = {"trust_remote_code": False}
        if self.revision is not None:
            pretrained_kwargs["revision"] = self.revision
        # Loading checkpoints can consume RNG state too; isolate initialization.
        with _isolated_rng(torch, device, 0):
            tokenizer = AutoTokenizer.from_pretrained(self.model_id, **pretrained_kwargs)
            model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                torch_dtype=getattr(torch, dtype_name),
                **pretrained_kwargs,
            )
            model.to(device)
            model.eval()

        limits = [self.max_context_tokens]
        configs = [model.config, getattr(model.config, "text_config", None)]
        for config in configs:
            for name in ("max_position_embeddings", "n_positions", "max_sequence_length", "seq_length"):
                limit = getattr(config, name, None)
                if isinstance(limit, int) and 0 < limit < 1_000_000_000:
                    limits.append(limit)
        # HF uses very large sentinel values when no tokenizer limit is known.
        limit = getattr(tokenizer, "model_max_length", None)
        if isinstance(limit, int) and 0 < limit < 1_000_000_000:
            limits.append(limit)
        effective_context = min(limits)
        if self.max_new_tokens >= effective_context:
            raise ValueError(
                f"max_new_tokens={self.max_new_tokens} leaves no prompt room in "
                f"the effective model context of {effective_context} tokens."
            )
        self._torch = torch
        self._model = model
        self._tokenizer = tokenizer
        self._resolved_device = device
        self._resolved_dtype = dtype_name
        self._effective_context_tokens = effective_context

    def provenance(self) -> dict:
        """Load and return checkpoint/runtime identity for calibration reuse.

        Local files need a separate version tag when no Hub commit is available.
        """
        with _GENERATION_LOCK:
            self._load()
            revision = getattr(self._model.config, "_commit_hash", None)
            if not revision:
                revision = getattr(self._tokenizer, "init_kwargs", {}).get("_commit_hash")
            device = self._resolved_device.type
            if self._resolved_device.index is not None:
                device += f":{self._resolved_device.index}"
            return {
                "model_id": self.model_id,
                "requested_revision": self.revision,
                "resolved_revision": str(revision) if revision else None,
                "device": device,
                "dtype": self._resolved_dtype,
            }

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        seed: int,
        temperature: float,
    ) -> str:
        """Return only completion text; MC branches cannot advance our RNG state."""
        if not isinstance(seed, int) or isinstance(seed, bool) or not -(2**63) <= seed < 2**64:
            raise ValueError("seed must be an integer in PyTorch's [-2**63, 2**64) range.")
        if not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative.")
        with _GENERATION_LOCK:
            self._load()
            inputs = self._encode_messages(messages)
            prompt_tokens = inputs["input_ids"].shape[-1]
            kwargs = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": temperature > 0,
                "return_dict_in_generate": False,
                "output_hidden_states": False,
                "output_attentions": False,
                "output_scores": False,
            }
            if temperature > 0:
                kwargs["temperature"] = temperature
            pad_token_id = self._tokenizer.pad_token_id
            if pad_token_id is None:
                pad_token_id = self._tokenizer.eos_token_id
            if pad_token_id is not None:
                kwargs["pad_token_id"] = pad_token_id
            with _isolated_rng(self._torch, self._resolved_device, seed), self._torch.inference_mode():
                output = self._model.generate(**inputs, **kwargs)
            completion = output[0, prompt_tokens:]
            return self._tokenizer.decode(completion, skip_special_tokens=True)

    def _encode_messages(self, messages: list[dict[str, str]]) -> dict:
        """Use the same prompt and completion reserve for generation and features."""
        encoded = None

        def count_tokens(history):
            nonlocal encoded
            text = self._tokenizer.apply_chat_template(
                history, tokenize=False, add_generation_prompt=True
            )
            encoded = self._tokenizer(
                text, add_special_tokens=False, return_tensors="pt",
                truncation=False, return_token_type_ids=False,
            )
            return encoded["input_ids"].shape[-1]

        trim_messages(
            messages,
            max_prompt_tokens=self._effective_context_tokens - self.max_new_tokens,
            count_tokens=count_tokens,
        )
        return {key: value.to(self._resolved_device) for key, value in encoded.items()}

    def extract_features(
        self, messages: list[dict[str, str]], *, layers: tuple[int, ...],
    ) -> dict[int, list[float]]:
        """Return HF hidden_states[layer] at the last nonpadding prompt token.

        Layers are numbered 1..N; 0 is the excluded embedding state. HF may
        apply final normalization to state N (Qwen does). Only supplied messages
        are encoded; no action or hypothetical continuation is generated.
        """
        if (not isinstance(layers, tuple) or not layers
                or any(not isinstance(layer, int) or isinstance(layer, bool) or layer < 1
                       for layer in layers) or len(set(layers)) != len(layers)):
            raise ValueError("layers must be a nonempty tuple of unique positive block numbers")
        with _GENERATION_LOCK:
            self._load()
            config = getattr(self._model.config, "text_config", None) or self._model.config
            num_layers = getattr(config, "num_hidden_layers", None)
            if isinstance(num_layers, int) and max(layers) > num_layers:
                raise ValueError(f"layers must be within transformer blocks 1..{num_layers}")
            with _preserved_rng(self._torch, self._resolved_device), self._torch.inference_mode():
                self._model.eval()
                inputs = self._encode_messages(messages)
                last_token = inputs["input_ids"].shape[-1] - 1
                if "attention_mask" in inputs:
                    positions = inputs["attention_mask"][0].nonzero(as_tuple=False)
                    if positions.numel() == 0:
                        raise ValueError("The prompt contains no nonpadding tokens")
                    last_token = int(positions[-1].item())
                output = self._model(
                    **inputs, output_hidden_states=True, output_attentions=False,
                    return_dict=True, use_cache=False,
                )
                states = getattr(output, "hidden_states", None)
                if states is None or len(states) < 2:
                    raise RuntimeError("The model did not return transformer hidden states")
                if max(layers) >= len(states):
                    raise ValueError(f"layers must be within transformer blocks 1..{len(states) - 1}")
                # Copy only selected vectors to Python; retain no device tensors.
                return {layer: states[layer][0, last_token].detach().float().cpu().tolist()
                        for layer in layers}
