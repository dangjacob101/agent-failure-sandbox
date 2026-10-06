"""Model adapter tests run without torch, transformers, weights, or a GPU."""

from contextlib import contextmanager, nullcontext
import copy
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from early_failure.models import DEFAULT_MODEL_ID, HuggingFaceAgent, trim_messages


def message(role, content):
    return {"role": role, "content": content}


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, device):
        return self

    def __getitem__(self, index):
        row, selection = index
        assert row == 0
        return self.values[selection]


class FakeTokenizer:
    model_max_length = 10**30  # HF's unknown-limit sentinel.
    pad_token_id = None
    eos_token_id = 99

    def __init__(self):
        self.histories = []
        self.encodings = []

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        self.histories.append(copy.deepcopy(messages))
        return " ".join(item["content"] for item in messages)

    def __call__(self, text, **kwargs):
        assert kwargs == {
            "add_special_tokens": False, "return_tensors": "pt",
            "truncation": False, "return_token_type_ids": False,
        }
        self.encodings.append(text)
        return {"input_ids": FakeTensor(list(range(len(text.split()) + 1)))}

    def decode(self, tokens, **kwargs):
        assert kwargs == {"skip_special_tokens": True}
        return ",".join(map(str, tokens))


class FakeTorch:
    def __init__(self, *, cuda=False, mps=False, bf16=True):
        self.cpu_state = 101
        self.cuda_states = {0: 202, 1: 303}
        self.mps_state = 404
        self.current_cuda = 0
        self.float32, self.float16, self.bfloat16 = "fp32", "fp16", "bf16"
        self.random = SimpleNamespace(
            get_rng_state=lambda: self.cpu_state,
            set_rng_state=lambda state: setattr(self, "cpu_state", state),
            default_generator=SimpleNamespace(manual_seed=lambda seed: setattr(self, "cpu_state", seed)),
        )
        self.cuda = SimpleNamespace(
            is_available=lambda: cuda,
            current_device=lambda: self.current_cuda,
            is_bf16_supported=lambda: bf16,
            get_rng_state=lambda device: self.cuda_states[device.index or 0],
            set_rng_state=lambda state, device: self.cuda_states.__setitem__(device.index or 0, state),
            manual_seed=lambda seed: self.cuda_states.__setitem__(self.current_cuda, seed),
            device=self.cuda_device,
        )
        self.mps = SimpleNamespace(
            get_rng_state=lambda: self.mps_state,
            set_rng_state=lambda state: setattr(self, "mps_state", state),
            manual_seed=lambda seed: setattr(self, "mps_state", seed),
        )
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
        self.inference_mode = nullcontext

    @staticmethod
    def device(value):
        parts = value.split(":")
        return SimpleNamespace(type=parts[0], index=int(parts[1]) if len(parts) > 1 else None)

    @contextmanager
    def cuda_device(self, device):
        original = self.current_cuda
        self.current_cuda = device.index or 0
        try:
            yield
        finally:
            self.current_cuda = original


class FakeModel:
    def __init__(self, torch, context=512):
        self.torch = torch
        self.config = SimpleNamespace(max_position_embeddings=context)
        self.calls = []
        self.fail_generation = False
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.eval_called = True

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        random.random()
        self.torch.cpu_state += 1
        sample = self.torch.cpu_state
        if self.device.type == "mps":
            self.torch.mps_state += 1
            sample = self.torch.mps_state
        elif self.device.type == "cuda":
            index = self.device.index or 0
            self.torch.cuda_states[index] += 1
            sample = self.torch.cuda_states[index]
        if self.fail_generation:
            raise RuntimeError("simulated generation failure")
        return FakeTensor(kwargs["input_ids"].values + [sample])


@contextmanager
def fake_dependencies(*, cuda=False, mps=False, bf16=True, context=512):
    torch = FakeTorch(cuda=cuda, mps=mps, bf16=bf16)
    tokenizer = FakeTokenizer()
    model = FakeModel(torch, context=context)
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
        yield torch, tokenizer, model, transformers


class TrimMessagesTests(unittest.TestCase):
    def test_drops_oldest_complete_pairs_without_mutating_history(self):
        history = [
            message("system", "instructions"), message("user", "task"),
            message("assistant", "action1"), message("user", "obs1"),
            message("assistant", "action2"), message("user", "obs2"),
            message("assistant", "action3"), message("user", "latest"),
        ]
        original = copy.deepcopy(history)
        trimmed = trim_messages(history, max_prompt_tokens=6, count_tokens=len)
        self.assertEqual(trimmed, history[:2] + history[4:])
        self.assertEqual(history, original)
        trimmed[0]["content"] = "edited"
        self.assertEqual(history, original)

    def test_preserves_all_system_messages_and_latest_pair(self):
        history = [
            message("system", "instructions"), message("user", "task"),
            message("assistant", "old"), message("user", "old"),
            message("system", "additional rule"),
            message("assistant", "recent"), message("user", "current"),
        ]
        trimmed = trim_messages(history, max_prompt_tokens=5, count_tokens=len)
        self.assertEqual(trimmed, history[:2] + history[4:])

    def test_oversized_protected_content_is_not_silently_truncated(self):
        history = [message("system", "rule"), message("user", "oversized task")]
        with self.assertRaisesRegex(ValueError, "Irreducible prompt requires 2 tokens"):
            trim_messages(history, max_prompt_tokens=1, count_tokens=len)

    def test_rejects_missing_task_and_invalid_content(self):
        for history in ([], [message("system", "rule")], [message("user", 7)]):
            with self.subTest(history=history), self.assertRaises(ValueError):
                trim_messages(history, max_prompt_tokens=30, count_tokens=len)


class HuggingFaceAgentTests(unittest.TestCase):
    history = [message("system", "rule"), message("user", "task")]

    def test_construction_is_lazy_and_defaults_to_small_reference_model(self):
        with fake_dependencies() as (_, _, _, transformers):
            agent = HuggingFaceAgent()
            self.assertEqual(agent.model_id, DEFAULT_MODEL_ID)
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
            self.assertIsNone(agent._model)

    def test_returns_only_completion_without_hidden_states(self):
        with fake_dependencies() as (_, _, model, transformers):
            agent = HuggingFaceAgent(max_new_tokens=4)
            self.assertEqual(agent.generate(self.history, seed=12, temperature=0), "13")
            call = model.calls[-1]
            self.assertFalse(call["output_hidden_states"])
            self.assertFalse(call["return_dict_in_generate"])
            self.assertFalse(call["do_sample"])
            self.assertNotIn("temperature", call)
            self.assertEqual(call["pad_token_id"], 99)
            self.assertTrue(model.eval_called)
            self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], "fp32")
            agent.generate(self.history, seed=1, temperature=0.7)
            self.assertTrue(model.calls[-1]["do_sample"])
            self.assertEqual(model.calls[-1]["temperature"], 0.7)
            transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_all_supported_devices_restore_rng_even_if_generation_raises(self):
        for kwargs in ({}, {"cuda": True}, {"mps": True}):
            with self.subTest(device=kwargs), fake_dependencies(**kwargs) as (torch, _, model, _):
                agent = HuggingFaceAgent(max_new_tokens=4)
                before = (torch.cpu_state, dict(torch.cuda_states), torch.mps_state, random.getstate())
                main = agent.generate(self.history, seed=17, temperature=1)
                agent.generate(self.history, seed=321, temperature=1)
                self.assertEqual(agent.generate(self.history, seed=17, temperature=1), main)
                self.assertEqual((torch.cpu_state, torch.cuda_states, torch.mps_state, random.getstate()), before)
                model.fail_generation = True
                with self.assertRaisesRegex(RuntimeError, "simulated"):
                    agent.generate(self.history, seed=51, temperature=1)
                self.assertEqual((torch.cpu_state, torch.cuda_states, torch.mps_state, random.getstate()), before)

    def test_initial_weight_loading_also_preserves_rng(self):
        with fake_dependencies() as (torch, _, model, transformers):
            def load_weights(*args, **kwargs):
                random.random()
                torch.cpu_state += 777
                return model

            transformers.AutoModelForCausalLM.from_pretrained.side_effect = load_weights
            before = (torch.cpu_state, random.getstate())
            HuggingFaceAgent(max_new_tokens=4).generate(self.history, seed=7, temperature=1)
            self.assertEqual((torch.cpu_state, random.getstate()), before)

    def test_accelerator_precision_and_explicit_cuda_index(self):
        cases = [
            ({"cuda": True}, "auto", "cuda", "bf16"),
            ({"cuda": True, "bf16": False}, "cuda:1", "cuda", "fp16"),
            ({"mps": True}, "auto", "mps", "fp16"),
        ]
        for availability, device, expected_device, expected_dtype in cases:
            with self.subTest(device=device), fake_dependencies(**availability) as (torch, _, model, transformers):
                agent = HuggingFaceAgent(device=device, max_new_tokens=4)
                agent.generate(self.history, seed=7, temperature=1)
                self.assertEqual(model.device.type, expected_device)
                self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], expected_dtype)
                self.assertEqual(torch.cuda_states, {0: 202, 1: 303})
                self.assertEqual(torch.current_cuda, 0)

    def test_revision_and_explicit_dtype_reach_both_pretrained_loaders(self):
        with fake_dependencies() as (_, _, _, transformers):
            agent = HuggingFaceAgent(model_id="local-model", dtype="bf16", revision="pinned", max_new_tokens=4)
            agent.generate(self.history, seed=7, temperature=0)
            for factory in (transformers.AutoModelForCausalLM, transformers.AutoTokenizer):
                args = factory.from_pretrained.call_args
                self.assertEqual(args.args, ("local-model",))
                self.assertEqual(args.kwargs["revision"], "pinned")
                self.assertFalse(args.kwargs["trust_remote_code"])
            self.assertEqual(transformers.AutoModelForCausalLM.from_pretrained.call_args.kwargs["torch_dtype"], "bf16")

    def test_provenance_loads_once_and_exposes_resolved_checkpoint(self):
        with fake_dependencies() as (torch, tokenizer, model, transformers):
            model.config._commit_hash = "immutable-model-commit"
            tokenizer.init_kwargs = {"_commit_hash": "tokenizer-fallback"}
            agent = HuggingFaceAgent(revision="main", max_new_tokens=4)
            before = (torch.cpu_state, random.getstate())
            self.assertEqual(agent.provenance(), {
                "model_id": DEFAULT_MODEL_ID,
                "requested_revision": "main",
                "resolved_revision": "immutable-model-commit",
                "device": "cpu",
                "dtype": "float32",
            })
            self.assertEqual((torch.cpu_state, random.getstate()), before)
            self.assertEqual(agent.provenance()["resolved_revision"], "immutable-model-commit")
            transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()
            model.config._commit_hash = None
            self.assertEqual(agent.provenance()["resolved_revision"], "tokenizer-fallback")
            tokenizer.init_kwargs = {}
            self.assertIsNone(agent.provenance()["resolved_revision"])

    def test_effective_context_uses_actual_model_limit_and_reserves_completion(self):
        with fake_dependencies(context=9) as (_, tokenizer, model, _):
            history = self.history + [
                message("assistant", "old action"), message("user", "old observation"),
                message("assistant", "recent"), message("user", "current"),
            ]
            agent = HuggingFaceAgent(max_new_tokens=4, max_context_tokens=100)
            agent.generate(history, seed=7, temperature=0)
            self.assertEqual(tokenizer.histories[-1], self.history + history[-2:])
            self.assertEqual(model.calls[-1]["input_ids"].shape[-1] + 4, 9)

    def test_tokenizer_limit_is_also_respected(self):
        with fake_dependencies(context=100) as (_, tokenizer, _, _):
            tokenizer.model_max_length = 6
            agent = HuggingFaceAgent(max_new_tokens=4)
            with self.assertRaisesRegex(ValueError, "Irreducible prompt"):
                agent.generate(self.history, seed=7, temperature=0)

    def test_no_room_for_completion_or_task_fails_clearly(self):
        with fake_dependencies(context=4):
            with self.assertRaisesRegex(ValueError, "effective model context of 4"):
                HuggingFaceAgent(max_new_tokens=4).generate(self.history, seed=1, temperature=0)

    def test_invalid_settings_and_unavailable_devices(self):
        for kwargs in ({"dtype": "int8"}, {"max_new_tokens": 0}, {"max_context_tokens": 128}, {"max_new_tokens": True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                HuggingFaceAgent(**kwargs)
        agent = HuggingFaceAgent()
        for seed, temperature in ((True, 0), (2**64, 0), (0, -1), (0, float("nan"))):
            with self.subTest(seed=seed, temperature=temperature), self.assertRaises(ValueError):
                agent.generate(self.history, seed=seed, temperature=temperature)
        for device in ("cuda", "mps"):
            with self.subTest(device=device), fake_dependencies(), self.assertRaises(RuntimeError):
                HuggingFaceAgent(device=device).generate(self.history, seed=0, temperature=0)


if __name__ == "__main__":
    unittest.main()
