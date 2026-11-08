"""Small source-framework cache semantics; no original checkpoint execution."""
from __future__ import annotations

from hashlib import sha256
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from cache_source import (TensorCacheLogits, TensorPrefillLogits, _AppendTensorCache, empty_cache,
                          phase_positions_and_mask, validate_phase, validate_prefill_phase)


def raw_hash(value):
    return sha256(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def state_inventory(model):
    return {name: (str(value.dtype), tuple(value.shape), raw_hash(value))
            for name, value in list(model.named_parameters()) + list(model.named_buffers())}


def tiny_source(**config_overrides):
    """Random synthetic BF16 weights, with the original attention replay policy."""
    import algos
    import load_quantized
    import ref_attention
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(21)
    settings = dict(vocab_size=32, hidden_size=32, intermediate_size=48,
                    num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                    max_position_embeddings=64, attn_implementation="eager")
    settings.update(config_overrides)
    config = LlamaConfig(**settings)
    model = LlamaForCausalLM(config).bfloat16().eval()
    policy = {"sites": {f"model.layers.{i}.self_attn": {
        "status": "quantized", "kind": "attention", "operands": {
            "q": {"source": "reference"}, "k": {"source": "reference"},
            "v": {"source": "reference"},
            "p": {"mode": "static", "exp": -8, "label": "static"}}}
        for i in range(config.num_hidden_layers)}}
    replay = load_quantized.PolicyAlgo(policy, {})
    with algos.active(replay):
        ref_attention.patch_attention(model)
    return model


class TensorCacheSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def assert_same_words(self, lhs, rhs):
        self.assertEqual(lhs.dtype, rhs.dtype)
        self.assertEqual(tuple(lhs.shape), tuple(rhs.shape))
        self.assertEqual(raw_hash(lhs), raw_hash(rhs))

    def test_prefill_and_two_decodes_match_original_cache_framework(self):
        from transformers import DynamicCache
        from raw_capture import prohibit_weight_conversion

        model = tiny_source()
        original_state = state_inventory(model)
        wrapped = TensorCacheLogits(model)
        tensors = empty_cache(model)
        conventional = DynamicCache(config=model.config)
        phases = (torch.tensor([[2, 3, 5]]), torch.tensor([[7]]), torch.tensor([[11]]))
        with prohibit_weight_conversion(), torch.no_grad():
            for phase_index, ids in enumerate(phases):
                length = tensors[0].shape[-2]
                positions, mask = phase_positions_and_mask(ids, length)
                validate_phase(ids, positions, mask, tensors)
                frozen_inputs = [raw_hash(value) for value in tensors]
                actual = wrapped(ids, positions, mask, *tensors)
                expected = model(ids, position_ids=positions, attention_mask=mask,
                                 past_key_values=conventional, use_cache=True)
                self.assert_same_words(actual[0], expected.logits)
                for i, layer in enumerate(conventional.layers):
                    self.assert_same_words(actual[1 + 2 * i], layer.keys)
                    self.assert_same_words(actual[2 + 2 * i], layer.values)
                    self.assertEqual(actual[1 + 2 * i].shape[1], model.config.num_key_value_heads)
                self.assertEqual([raw_hash(value) for value in tensors], frozen_inputs)
                self.assertEqual(actual[1].shape[-2], 3 + phase_index)
                tensors = actual[1:]
        self.assertEqual(state_inventory(model), original_state)

    def test_cache_contains_post_rope_unexpanded_keys_and_full_prefix_casts(self):
        import ref_attention
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

        model = tiny_source()
        wrapped = TensorCacheLogits(model)
        ids = torch.tensor([[2, 3, 5]])
        positions, mask = phase_positions_and_mask(ids, 0)
        with torch.no_grad():
            result = wrapped(ids, positions, mask, *empty_cache(model))
            hidden = model.model.layers[0].input_layernorm(model.model.embed_tokens(ids))
            attention = model.model.layers[0].self_attn
            shape = (1, 3, -1, attention.head_dim)
            query = attention.q_proj(hidden).view(shape).transpose(1, 2)
            key = attention.k_proj(hidden).view(shape).transpose(1, 2)
            value = attention.v_proj(hidden).view(shape).transpose(1, 2)
            cosine, sine = model.model.rotary_emb(hidden, positions)
            _, rotated_key = apply_rotary_pos_emb(query, key, cosine, sine)
            self.assert_same_words(result[1], rotated_key)
            self.assert_same_words(result[2], value)
            self.assertNotEqual(raw_hash(result[1]), raw_hash(key))
            casts = []
            original_cast = ref_attention._cast
            def observe(algo, site, operand, tensor, dims):
                casts.append((site, operand, tuple(tensor.shape), dims))
                return original_cast(algo, site, operand, tensor, dims)
            token = torch.tensor([[7]])
            positions, mask = phase_positions_and_mask(token, 3)
            with patch.object(ref_attention, "_cast", side_effect=observe):
                wrapped(token, positions, mask, *result[1:])
            for _, operand, shape, dims in casts:
                if operand in ("k", "v"):
                    self.assertEqual(shape, (1, 4, 4, 8))
                    self.assertEqual(dims, (-2, -1))
            self.assertEqual(sum(operand in ("k", "v") for _, operand, _, _ in casts), 4)

    def test_large_new_key_recodes_old_activation_prefix(self):
        import ref_numerics

        old = torch.ones((1, 2, 2, 8), dtype=torch.bfloat16)
        new = torch.full((1, 2, 1, 8), 128, dtype=torch.bfloat16)
        cache = _AppendTensorCache(((old, old),))
        before, before_scale = ref_numerics.quantize(old, (-2, -1))
        key, _ = cache.update(new, new, 0)
        after, after_scale = ref_numerics.quantize(key, (-2, -1))
        self.assertFalse(torch.equal(before_scale, after_scale))
        self.assertFalse(torch.equal(before, after[:, :, :2]))
        self.assert_same_words(key[:, :, :2], old)

    def test_wrong_positions_mask_or_cache_encoding_are_refused(self):
        model = tiny_source()
        wrapped = TensorCacheLogits(model)
        ids = torch.tensor([[2, 3]])
        cache = empty_cache(model)
        positions, mask = phase_positions_and_mask(ids, 0)
        with self.assertRaisesRegex(ValueError, "positions"):
            validate_phase(ids, positions + 1, mask, cache)
        with self.assertRaisesRegex(ValueError, "mask"):
            validate_phase(ids, positions, torch.zeros_like(mask), cache)
        with self.assertRaisesRegex(ValueError, "BF16"):
            wrapped(ids, positions, mask, *(value.to(torch.uint8) for value in cache))
        expanded = tuple(value.repeat_interleave(2, dim=1) for value in cache)
        with self.assertRaisesRegex(ValueError, "geometry"):
            wrapped(ids, positions, mask, *expanded)

    def test_update_requires_all_layers_exactly_once(self):
        tensor = torch.ones((1, 2, 1, 8), dtype=torch.bfloat16)
        empty = tensor[:, :, :0]
        cache = _AppendTensorCache(((empty, empty), (empty, empty)))
        cache.update(tensor, tensor, 0)
        with self.assertRaisesRegex(ValueError, "exactly once"):
            cache.update(tensor, tensor, 0)
        with self.assertRaisesRegex(ValueError, "omitted"):
            cache.tensors()

    def test_fixed_prefill_and_decode_export_have_tensor_only_outputs(self):
        model = tiny_source()
        wrapped = TensorCacheLogits(model)
        tensors = empty_cache(model)
        with torch.no_grad():
            for ids in (torch.tensor([[2, 3, 5]]), torch.tensor([[7]])):
                positions, mask = phase_positions_and_mask(ids, tensors[0].shape[-2])
                args = (ids, positions, mask, *tensors)
                expected = wrapped(*args)
                program = torch.export.export(wrapped, args, strict=False)
                actual = program.module()(*args)
                self.assertEqual(len(actual), 5)
                for lhs, rhs in zip(actual, expected):
                    self.assert_same_words(lhs, rhs)
                self.assertTrue(all(isinstance(value, torch.Tensor) for value in actual))
                tensors = expected[1:]

    def test_native_uint8_decode_keeps_full_prefix_attention_quantizers(self):
        import fx_export

        model = tiny_source()
        wrapped = TensorCacheLogits(model)
        frozen = state_inventory(model)
        replay = model.model.layers[0].self_attn._atlas_ref_attn["algo"]
        with torch.no_grad():
            prefill = torch.tensor([[2, 3, 5]])
            positions, mask = phase_positions_and_mask(prefill, 0)
            prefix = wrapped(prefill, positions, mask, *empty_cache(model))[1:]
            token = torch.tensor([[7]])
            positions, mask = phase_positions_and_mask(token, 3)
            args = (token, positions, mask, *prefix)
            expected = wrapped(*args)
            with fx_export._native_sites(wrapped, replay.policy, "uint8") as plan:
                self.assertIsNotNone(plan["attention"])
                program = torch.export.export(wrapped, args, strict=False)
            actual = program.module()(*args)
            for lhs, rhs in zip(actual, expected):
                self.assert_same_words(lhs, rhs)
            nodes = [node for node in program.graph.nodes if node.op == "call_function"]
            self.assertEqual(sum(str(node.target) == "atlas.qk_matmul.default" for node in nodes), 2)
            self.assertEqual(sum(str(node.target) == "atlas.pv_matmul.default" for node in nodes), 2)
            cache_casts = [node for node in nodes if str(node.target) == "atlas.quantize_act.default"
                           and node.args[6].endswith((":k", ":v"))]
            self.assertEqual(len(cache_casts), 4)
            for node in cache_casts:
                self.assertEqual(node.args[1], "dynamic")
                self.assertEqual(node.args[5], "uint8")
                self.assertEqual(tuple(node.args[0].meta["val"].shape), (1, 4, 4, 8))
        self.assertEqual(state_inventory(model), frozen)

    def test_three_input_prefill_and_two_decodes_match_original_and_legacy(self):
        from transformers import DynamicCache
        from raw_capture import prohibit_weight_conversion

        model = tiny_source()
        frozen = state_inventory(model)
        prefill, decode = TensorPrefillLogits(model).eval(), TensorCacheLogits(model).eval()
        original = DynamicCache(config=model.config)
        self.assertIs(prefill.model, model)
        self.assertIn("model.model.layers.0", dict(prefill.named_modules()))
        tensors = None
        with prohibit_weight_conversion(), torch.no_grad():
            for ids in (torch.tensor([[2, 3, 5, 7]]), torch.tensor([[11]]), torch.tensor([[13]])):
                past = 0 if tensors is None else tensors[0].shape[-2]
                positions, mask = phase_positions_and_mask(ids, past)
                if tensors is None:
                    validate_prefill_phase(ids, positions, mask)
                    actual = prefill(ids, positions, mask)
                    legacy = decode(ids, positions, mask, *empty_cache(model))
                    for lhs, rhs in zip(actual, legacy):
                        self.assert_same_words(lhs, rhs)
                else:
                    validate_phase(ids, positions, mask, tensors)
                    actual = decode(ids, positions, mask, *tensors)
                expected = model(ids, position_ids=positions, attention_mask=mask,
                                 past_key_values=original, use_cache=True)
                self.assert_same_words(actual[0], expected.logits)
                for index, layer in enumerate(original.layers):
                    self.assert_same_words(actual[1 + 2 * index], layer.keys)
                    self.assert_same_words(actual[2 + 2 * index], layer.values)
                    self.assertGreater(actual[1 + 2 * index].shape[-2], 0)
                tensors = actual[1:]
        self.assertEqual(state_inventory(model), frozen)

    def test_native_three_input_prefill_graph_has_only_positive_tensor_shapes(self):
        import fx_export
        from cache_roles import derive_cache_phase_abi
        from torch.utils import _pytree

        model = tiny_source()
        boundary = TensorPrefillLogits(model).eval()
        ids = torch.tensor([[2, 3, 5, 7]])
        positions, mask = phase_positions_and_mask(ids, 0)
        replay = model.model.layers[0].self_attn._atlas_ref_attn["algo"]
        with torch.no_grad():
            expected = boundary(ids, positions, mask)
            with fx_export._native_sites(boundary, replay.policy, "uint8"):
                program = torch.export.export(boundary, (ids, positions, mask), strict=True)
            actual = program.module()(ids, positions, mask)
        for lhs, rhs in zip(actual, expected):
            self.assert_same_words(lhs, rhs)
        abi = derive_cache_phase_abi(program, layer_count=2, past_length=0, query_length=4,
                                    kv_heads=2, head_dim=8, batch_size=1)
        self.assertTrue(abi["initial_cache_empty"])
        self.assertEqual(len(abi["inputs"]), 3)
        self.assertEqual(len(abi["outputs"]), 5)
        for node in program.graph.nodes:
            for value in _pytree.tree_leaves(node.meta.get("val")):
                if isinstance(value, torch.Tensor):
                    self.assertTrue(all(extent > 0 for extent in value.shape), node.name)
        native = [node for node in program.graph.nodes if str(node.target) == "atlas.qk_matmul.default"]
        self.assertEqual(len(native), 2)
        self.assertTrue(any(path == "model.model.layers.0"
            for node in native for path, _ in node.meta.get("nn_module_stack", {}).values()))

    def test_absent_initial_cache_update_returns_source_tensors_without_concat(self):
        key = torch.ones((1, 2, 4, 8), dtype=torch.bfloat16)
        value = torch.full_like(key, 2)
        cache = _AppendTensorCache(((None, None),))
        self.assertEqual(cache.get_seq_length(), 0)
        with patch.object(torch, "cat", side_effect=AssertionError("empty-prefix concat called")):
            actual_key, actual_value = cache.update(key, value, 0)
        self.assertIs(actual_key, key)
        self.assertIs(actual_value, value)
        self.assertEqual(cache.get_seq_length(), 4)

    def test_three_input_prefill_admission_refuses_nonzero_positions_and_wrong_mask(self):
        ids = torch.tensor([[2, 3, 5, 7]])
        positions, mask = phase_positions_and_mask(ids, 0)
        with self.assertRaisesRegex(ValueError, "begin at zero"):
            validate_prefill_phase(ids, positions + 1, mask)
        with self.assertRaisesRegex(ValueError, "causal mask"):
            validate_prefill_phase(ids, positions, torch.zeros_like(mask))

    def test_prefill_post_rope_cache_views_serialize_contiguously_inside_source_layer(self):
        import fx_export
        from cache_roles import derive_cache_phase_abi
        from safetensors.torch import load_file, save_file
        from transformers import DynamicCache

        model = tiny_source(hidden_size=64, num_attention_heads=8, num_key_value_heads=4)
        before = state_inventory(model)
        boundary = TensorPrefillLogits(model).eval()
        ids = torch.tensor([[2, 3, 5, 7]])
        positions, mask = phase_positions_and_mask(ids, 0)
        original_cache = DynamicCache(config=model.config)
        replay = model.model.layers[0].self_attn._atlas_ref_attn["algo"]
        observed = []
        original_update = original_cache.update
        def observe_update(key, value, layer, *args, **kwargs):
            observed.append((layer, key.is_contiguous(), raw_hash(key), raw_hash(value)))
            return original_update(key, value, layer, *args, **kwargs)
        with torch.no_grad():
            with patch.object(original_cache, "update", side_effect=observe_update):
                original = model(ids, position_ids=positions, attention_mask=mask,
                                 past_key_values=original_cache, use_cache=True)
            self.assertFalse(observed[0][1])
            actual = boundary(ids, positions, mask)
            expected = (original.logits, *(value for layer in original_cache.layers
                                           for value in (layer.keys, layer.values)))
            for lhs, rhs in zip(actual, expected):
                self.assert_same_words(lhs, rhs)
            self.assertTrue(all(value.is_contiguous() for value in actual[1:]))
            with tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="cache-layout-") as directory:
                path = Path(directory) / "outputs.safetensors"
                save_file({f"output_{i}": value for i, value in enumerate(actual)}, str(path))
                restored = load_file(str(path))
                for i, value in enumerate(actual):
                    self.assert_same_words(value, restored[f"output_{i}"])
            with fx_export._native_sites(boundary, replay.policy, "uint8"):
                program = torch.export.export(boundary, (ids, positions, mask), strict=True)
        abi = derive_cache_phase_abi(program, layer_count=2, past_length=0, query_length=4,
                                    kv_heads=4, head_dim=8, batch_size=1)
        nodes = {node.name: node for node in program.graph.nodes}
        for output in abi["outputs"][1:]:
            node = nodes[output["source_node"]]
            self.assertTrue(any(path == f"model.model.layers.{output['layer']}"
                for path, _ in node.meta.get("nn_module_stack", {}).values()), node.name)
        self.assertEqual(state_inventory(model), before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
