"""Activation extraction tests use fake tensors and never download a model."""

from contextlib import contextmanager
import copy
import gc
import random
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import weakref

from early_failure.models import HuggingFaceAgent
from test_models import FakeModel, FakeTensor, FakeTokenizer, FakeTorch, message


class FeatureTensor:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, index):
        values = self.values
        for part in index if isinstance(index, tuple) else (index,):
            values = values[part]
        return FeatureTensor(values)

    def to(self, device):
        return self

    def nonzero(self, *, as_tuple):
        assert not as_tuple
        return FeatureTensor([[i] for i, value in enumerate(self.values) if value])

    def numel(self):
        return len(self.values)

    def item(self):
        return self.values[0]

    def detach(self):
        return self

    def float(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return copy.deepcopy(self.values)


class FeatureTokenizer(FakeTokenizer):
    def __init__(self, mask=None):
        super().__init__()
        self.mask = mask

    def __call__(self, text, **kwargs):
        result = super().__call__(text, **kwargs)
        if self.mask is not None:
            result["input_ids"] = FakeTensor(list(range(len(self.mask))))
            result["attention_mask"] = FeatureTensor([self.mask])
        return result


class FeatureModel(FakeModel):
    def __init__(self, torch, context):
        super().__init__(torch, context=context)
        self.config.num_hidden_layers = 3
        self.forward_calls = []
        self.state_refs = []
        self.fail_forward = False
        self.return_states = True

    def __call__(self, **kwargs):
        assert self.eval_called and self.torch.in_inference
        self.forward_calls.append(kwargs)
        random.random()
        self.torch.cpu_state += 1
        if self.device.type == "mps":
            self.torch.mps_state += 1
        elif self.device.type == "cuda":
            self.torch.cuda_states[self.device.index] += 1
        if self.fail_forward:
            raise RuntimeError("simulated feature failure")
        length = kwargs["input_ids"].shape[-1]
        states = tuple(FeatureTensor([[[float(layer * 100 + token), layer * 100 + token + 0.5]
                                       for token in range(length)]]) for layer in range(4))
        self.state_refs = [weakref.ref(state) for state in states]
        return SimpleNamespace(hidden_states=states if self.return_states else None)


@contextmanager
def feature_dependencies(*, mask=None, context=512, cuda=False, mps=False):
    torch = FakeTorch(cuda=cuda, mps=mps)
    torch.in_inference = False

    @contextmanager
    def inference_mode():
        torch.in_inference = True
        try:
            yield
        finally:
            torch.in_inference = False

    torch.inference_mode = inference_mode
    tokenizer = FeatureTokenizer(mask)
    model = FeatureModel(torch, context)
    transformers = SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=Mock(return_value=tokenizer)),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=Mock(return_value=model)),
    )
    with patch.dict("sys.modules", {"torch": torch, "transformers": transformers}):
        yield torch, tokenizer, model, transformers


class ActivationFeaturesTests(unittest.TestCase):
    history = [message("system", "rule"), message("user", "task")]

    def test_selected_hf_states_are_plain_vectors_without_generating_actions(self):
        with feature_dependencies() as (_, tokenizer, model, transformers):
            agent = HuggingFaceAgent(max_new_tokens=4)
            transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()
            original = copy.deepcopy(self.history)
            features = agent.extract_features(self.history, layers=(3, 1))
            self.assertEqual(features, {3: [302.0, 302.5], 1: [102.0, 102.5]})
            self.assertEqual(self.history, original)
            self.assertEqual(tokenizer.histories[-1], original)
            self.assertEqual(model.calls, [])
            self.assertEqual({k: v for k, v in model.forward_calls[0].items() if k != "input_ids"}, {
                "output_hidden_states": True, "output_attentions": False,
                "return_dict": True, "use_cache": False,
            })
            gc.collect()
            self.assertTrue(all(ref() is None for ref in model.state_refs))
            agent.extract_features(self.history, layers=(2,))
            transformers.AutoModelForCausalLM.from_pretrained.assert_called_once()

    def test_last_nonpadding_token_for_left_right_and_internal_padding(self):
        for mask, last in (([1, 1, 1, 0, 0], 2), ([0, 0, 1, 1, 1], 4), ([1, 0, 1, 0, 0], 2)):
            with self.subTest(mask=mask), feature_dependencies(mask=mask):
                features = HuggingFaceAgent(max_new_tokens=4).extract_features(self.history, layers=(1,))
                self.assertEqual(features[1], [float(100 + last), 100 + last + 0.5])

    def test_generation_and_extraction_use_identical_trimmed_prompt(self):
        history = self.history + [
            message("assistant", "old action"), message("user", "old observation"),
            message("assistant", "recent"), message("user", "current"),
        ]
        with feature_dependencies(context=9) as (_, tokenizer, model, _):
            agent = HuggingFaceAgent(max_new_tokens=4, max_context_tokens=100)
            completion = agent.generate(history, seed=7, temperature=0)
            generated_prompt = copy.deepcopy(tokenizer.histories[-1])
            agent.extract_features(history, layers=(1,))
            self.assertEqual(tokenizer.histories[-1], generated_prompt)
            self.assertEqual(generated_prompt, self.history + history[-2:])
            self.assertEqual(model.forward_calls[-1]["input_ids"].shape[-1] + 4, 9)
            self.assertEqual(agent.generate(history, seed=7, temperature=0), completion)

    def test_rng_is_preserved_on_all_devices_and_forward_failure(self):
        for availability in ({}, {"mps": True}, {"cuda": True}):
            with self.subTest(availability=availability), feature_dependencies(**availability) as (torch, _, model, _):
                agent = HuggingFaceAgent(max_new_tokens=4)
                before = (torch.cpu_state, torch.mps_state, dict(torch.cuda_states), random.getstate())
                agent.extract_features(self.history, layers=(1,))
                self.assertEqual((torch.cpu_state, torch.mps_state, torch.cuda_states, random.getstate()), before)
                torch.random.default_generator.manual_seed = Mock()
                model.fail_forward = True
                with self.assertRaisesRegex(RuntimeError, "simulated feature failure"):
                    agent.extract_features(self.history, layers=(1,))
                torch.random.default_generator.manual_seed.assert_not_called()
                self.assertEqual((torch.cpu_state, torch.mps_state, torch.cuda_states, random.getstate()), before)

    def test_invalid_layer_numbers_are_rejected_before_loading(self):
        for layers in ((), (0,), (-1,), (True,), (1.5,), (1, 1), [1], "1"):
            with self.subTest(layers=layers), feature_dependencies() as (_, _, _, transformers):
                with self.assertRaisesRegex(ValueError, "layers"):
                    HuggingFaceAgent().extract_features(self.history, layers=layers)
                transformers.AutoModelForCausalLM.from_pretrained.assert_not_called()

    def test_out_of_range_layer_rejected_before_forward(self):
        with feature_dependencies() as (_, _, model, _):
            with self.assertRaisesRegex(ValueError, r"1\.\.3"):
                HuggingFaceAgent(max_new_tokens=4).extract_features(self.history, layers=(4,))
            self.assertEqual(model.forward_calls, [])

    def test_returned_layer_count_is_checked_when_config_omits_it(self):
        with feature_dependencies() as (_, _, model, _):
            del model.config.num_hidden_layers
            with self.assertRaisesRegex(ValueError, r"1\.\.3"):
                HuggingFaceAgent(max_new_tokens=4).extract_features(self.history, layers=(4,))

    def test_missing_states_empty_mask_and_oversized_prompt_fail_clearly(self):
        with feature_dependencies() as (_, _, model, _):
            model.return_states = False
            with self.assertRaisesRegex(RuntimeError, "hidden states"):
                HuggingFaceAgent(max_new_tokens=4).extract_features(self.history, layers=(1,))
        with feature_dependencies(mask=[0, 0]) as (_, _, model, _):
            with self.assertRaisesRegex(ValueError, "nonpadding"):
                HuggingFaceAgent(max_new_tokens=4).extract_features(self.history, layers=(1,))
            self.assertEqual(model.forward_calls, [])
        with feature_dependencies(context=6):
            with self.assertRaisesRegex(ValueError, "Irreducible prompt"):
                HuggingFaceAgent(max_new_tokens=4).extract_features(self.history, layers=(1,))


if __name__ == "__main__":
    unittest.main()
