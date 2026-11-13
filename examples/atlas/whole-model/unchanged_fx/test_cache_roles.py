"""Small exported graphs establish cache source identities without model intake."""
from __future__ import annotations

import copy
import json
import unittest

import torch

from cache_roles import bind_region_cache_roles, derive_cache_phase_abi, prioritize_stage_output


class TinyCachePhase(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("source_bias", torch.zeros((1, 1, 3), dtype=torch.bfloat16))

    def forward(self, unusual_ids, absolute, additive, *past):
        query = unusual_ids.shape[1]
        current = additive[:, :, :, :1].expand(1, 2, query, 3)
        logits = current[:, 0] + self.source_bias
        return (logits, *(torch.cat((value, current), dim=2) for value in past))


def tiny_export(past=4, query=1, layers=2):
    args = (torch.ones((1, query), dtype=torch.int64),
            torch.arange(past, past + query)[None],
            torch.zeros((1, 1, query, past + query), dtype=torch.bfloat16),
            *(torch.zeros((1, 2, past, 3), dtype=torch.bfloat16) for _ in range(2 * layers)))
    return torch.export.export(TinyCachePhase(), args, strict=True)


class TinyPrefillPhase(TinyCachePhase):
    def forward(self, unusual_ids, absolute, additive):
        current = additive[:, :, :, :1].expand(1, 2, unusual_ids.shape[1], 3)
        logits = current[:, 0] + self.source_bias
        return (logits, *(current + index for index in range(4)))


def tiny_prefill_export(query=4):
    args = (torch.ones((1, query), dtype=torch.int64), torch.arange(query)[None],
            torch.zeros((1, 1, query, query), dtype=torch.bfloat16))
    return torch.export.export(TinyPrefillPhase(), args, strict=True)


def abi(program, past=4, query=1, **kwargs):
    return derive_cache_phase_abi(program, layer_count=2, past_length=past,
                                  query_length=query, **kwargs)


class CacheRoleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.program = tiny_export()

    def test_graph_signature_excludes_lifted_state_and_preserves_actual_names(self):
        result = abi(self.program, kv_heads=2, head_dim=3, batch_size=1)
        names = list(self.program.graph_signature.user_inputs)
        self.assertEqual([entry["source_node"] for entry in result["inputs"]], names)
        self.assertEqual([entry["source_node"] for entry in result["outputs"]],
                         list(self.program.graph_signature.user_outputs))
        self.assertEqual(result["inputs"][0]["role"], "input_ids")
        self.assertEqual(result["inputs"][3], dict(index=3, role="past_key", layer=0,
                         source_node=names[3], shape=[1, 2, 4, 3], dtype="torch.bfloat16"))
        self.assertNotIn("b_source_bias", [entry["source_node"] for entry in result["inputs"]])
        self.assertEqual(result["outputs"][-1]["shape"], [1, 2, 5, 3])
        json.dumps(result)

    def test_all_three_fixed_phase_lengths(self):
        for past, query in ((0, 4), (4, 1), (5, 1)):
            with self.subTest(past=past, query=query):
                result = abi(tiny_export(past, query), past, query)
                self.assertEqual(result["past_length"], past)
                self.assertEqual(result["total_length"], past + query)

    def test_implicit_empty_prefill_has_three_inputs_and_positive_outputs(self):
        result = abi(tiny_prefill_export(), 0, 4)
        self.assertIs(result["initial_cache_empty"], True)
        self.assertEqual([entry["role"] for entry in result["inputs"]],
                         ["input_ids", "position_ids", "attention_mask"])
        self.assertEqual((result["kv_heads"], result["head_dim"]), (2, 3))
        self.assertEqual(len(result["outputs"]), 5)
        self.assertTrue(all(entry["shape"] == [1, 2, 4, 3] for entry in result["outputs"][1:]))
        record = dict(kind="layer1", source_boundaries=["hidden_in"],
                      source_outputs=[result["outputs"][3]["source_node"], "hidden_out"])
        bound = bind_region_cache_roles(result, record)
        self.assertEqual(bound["outputs"][0]["role"], "updated_key")
        self.assertEqual(bound["outputs"][0]["layer"], 1)
        self.assertEqual(bound["outputs"][1]["role"], "hidden_state")

    def test_three_input_prefill_refuses_positive_past_or_incompatible_geometry(self):
        program = tiny_prefill_export()
        for options in (dict(past=1, query=4), dict(past=0, query=3),
                        dict(past=0, query=4, kv_heads=4), dict(past=0, query=4, head_dim=8)):
            with self.subTest(options=options), self.assertRaises(ValueError):
                abi(program, **options)
        for shape, dtype in (((1, 2, 0, 3), torch.bfloat16), ((1, 4, 4, 3), torch.bfloat16),
                             ((1, 2, 4, 3), torch.float32)):
            changed = copy.deepcopy(program)
            name = abi(program, 0, 4)["outputs"][-1]["source_node"]
            node = next(node for node in changed.graph.nodes if node.name == name)
            node.meta["val"] = torch.empty(shape, dtype=dtype)
            with self.subTest(shape=shape, dtype=dtype), self.assertRaisesRegex(ValueError, "dtype/geometry"):
                abi(changed, 0, 4)

    def test_legacy_47_input_decoder_and_explicit_empty_abi_are_unchanged(self):
        result = derive_cache_phase_abi(tiny_export(layers=22), layer_count=22, past_length=4, query_length=1)
        self.assertEqual((len(result["inputs"]), len(result["outputs"])), (47, 45))
        self.assertNotIn("initial_cache_empty", result)
        self.assertEqual(result["inputs"][-1]["index"], 46)
        self.assertEqual(result["inputs"][-1]["role"], "past_value")
        self.assertEqual(result["outputs"][-1]["index"], 44)
        self.assertEqual(result["outputs"][-1]["role"], "updated_value")
        self.assertNotIn("initial_cache_empty", abi(tiny_export(0, 4), 0, 4))

    def test_expected_layer_count_and_phase_geometry_refuse(self):
        for options in (dict(layer_count=3), dict(past_length=3), dict(query_length=2),
                        dict(kv_heads=4), dict(head_dim=8), dict(batch_size=2)):
            values = dict(layer_count=2, past_length=4, query_length=1)
            values.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                derive_cache_phase_abi(self.program, **values)

    def test_every_cache_pair_dtype_and_shape_is_checked(self):
        result = abi(self.program)
        for entry in (*result["inputs"][3:], *result["outputs"][1:]):
            changed = copy.deepcopy(self.program)
            node = next(node for node in changed.graph.nodes if node.name == entry["source_node"])
            node.meta["val"] = torch.empty(entry["shape"], dtype=torch.float32)
            with self.subTest(name=node.name), self.assertRaisesRegex(ValueError, "dtype/geometry"):
                abi(changed)
        changed = copy.deepcopy(self.program)
        node = next(node for node in changed.graph.nodes
                    if node.name == result["outputs"][-1]["source_node"])
        node.meta["val"] = torch.empty((1, 4, 5, 3), dtype=torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "dtype/geometry"):
            abi(changed)

    def test_source_graph_output_identity_must_match_signature(self):
        changed = copy.deepcopy(self.program)
        node = next(node for node in changed.graph.nodes if node.op == "output")
        values = list(node.args[0])
        values[1], values[2] = values[2], values[1]
        node.args = (tuple(values),)
        with self.assertRaisesRegex(ValueError, "identities"):
            abi(changed)

    def test_value_admission_is_explicitly_separate(self):
        result = abi(self.program)
        self.assertEqual(result["value_admission"], "cache_source.validate_phase_required")

    def test_extracted_cache_roles_follow_saved_indices(self):
        result = abi(self.program)
        past_key = result["inputs"][5]["source_node"]
        updated_key = result["outputs"][3]["source_node"]
        updated_value = result["outputs"][4]["source_node"]
        record = dict(kind="layer1", source_boundaries=["hidden_in", past_key],
                      source_outputs=[updated_key, updated_value, "hidden_out"])
        bound = bind_region_cache_roles(result, record)
        self.assertEqual(bound["inputs"][1]["global_index"], 5)
        self.assertEqual(bound["inputs"][1]["role"], "past_key")
        self.assertEqual(bound["outputs"][0]["index"], 0)
        self.assertEqual(bound["outputs"][0]["role"], "updated_key")
        self.assertEqual(bound["outputs"][2]["role"], "hidden_state")
        self.assertEqual(record["source_outputs"], [updated_key, updated_value, "hidden_out"])

    def test_output_priority_preserves_nodes_and_cache_order(self):
        result = abi(self.program)
        graph = torch.fx.Graph()
        key = graph.placeholder(result["outputs"][1]["source_node"])
        value = graph.placeholder(result["outputs"][2]["source_node"])
        hidden = graph.placeholder("hidden")
        self.assertEqual(prioritize_stage_output((key, value, hidden), result), (hidden, key, value))
        other = graph.placeholder("other")
        self.assertEqual(prioritize_stage_output((key, other, hidden), result), (key, other, hidden))

    def test_region_duplicate_identities_refuse(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            bind_region_cache_roles(abi(self.program), dict(kind="layer0",
                                    source_boundaries=["x", "x"], source_outputs=[]))


if __name__ == "__main__":
    unittest.main()
