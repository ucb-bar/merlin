"""Fixture argument order must follow the positional export ABI."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import load_file

from cache_roles import derive_cache_phase_abi
from extract_capsules import (Capture, file_hash, fixture_inputs, load_cache_phase,
                              regional_capture_metadata, save_region, sequential_region_manifest, validate_phase_inputs)


class FixtureInputTests(unittest.TestCase):
    def test_large_cache_abi_uses_numeric_positions(self):
        fixtures = {f'case_0.input_{i}': i for i in reversed(range(47))}
        fixtures['case_0.output_0'] = 'output'
        fixtures['case_1.input_0'] = 'other case'
        self.assertEqual(fixture_inputs(fixtures), tuple(range(47)))

    def test_small_abi_is_unchanged(self):
        self.assertEqual(fixture_inputs({'case_0.input_0': 'ids'}), ('ids',))

    def test_other_case_can_be_selected(self):
        self.assertEqual(fixture_inputs({'case_1.input_0': 'ids'}, 1), ('ids',))

    def test_missing_or_empty_indices_refuse(self):
        for values in ({}, {'case_0.input_1': 1}, {'case_0.input_0': 0, 'case_0.input_2': 2}):
            with self.subTest(values=values), self.assertRaisesRegex(ValueError, 'contiguous'):
                fixture_inputs(values)

    def test_noncanonical_indices_refuse(self):
        for suffix in ('01', '-1', 'x', '', '٠'):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(ValueError, 'canonical'):
                fixture_inputs({f'case_0.input_{suffix}': 0})


class RegionalIdentityTests(unittest.TestCase):
    def test_metadata_starts_new_namespace_and_preserves_parent_receipt(self):
        metadata = {"_m2m_node_id": "parent:17", "_m2m_origin_ids": ["ancestor:9"],
                    "nn_module_stack": {"layer": ("model.layer", "Layer")},
                    "custom": {"m2m_node_id": "parent:17", "m2m_lineage": ["ancestor:9", "parent:17"],
                               "m2m_deferred_constant_users": [{"lineage": ["ancestor:9"]}],
                               "unrelated": "preserved"}}
        before = copy.deepcopy(metadata)
        fresh, parent = regional_capture_metadata(metadata)
        self.assertEqual(metadata, before)
        self.assertEqual(fresh, {"nn_module_stack": metadata["nn_module_stack"],
                                 "custom": {"unrelated": "preserved"}})
        self.assertEqual(parent, {key: value for key, value in metadata.items() if key.startswith("_m2m_")} |
                         {"custom": {key: value for key, value in metadata["custom"].items() if key.startswith("m2m_")}})
        parent["custom"]["m2m_lineage"].append("changed:receipt")
        self.assertEqual(metadata, before)

    def test_new_region_export_preserves_values_and_archives_parent_namespace(self):
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("offset", torch.ones(3))

            def forward(self, value):
                return (value + self.offset).relu()

        inputs = (torch.tensor([-2., 0., 2.]),)
        program = torch.export.export(Model(), inputs)
        for index, node in enumerate(program.graph.nodes):
            identity = f"g:original:root:n{index + 100}"
            node.meta.update(_m2m_node_id=identity, _m2m_origin_ids=[f"parent:{index}"])
            node.meta["custom"] = {"m2m_node_id": identity,
                                   "m2m_lineage": [f"parent:{index}", identity], "unrelated": "preserved"}
        capture = Capture(program.module())
        capture.source_program = program
        capture.run(*inputs)
        selected = {node for node in capture.module.graph.nodes if node.op == "call_function"}
        output, = next(node for node in capture.module.graph.nodes if node.op == "output").args[0]
        identities = {node.name: copy.deepcopy(node.meta.get("custom")) for node in capture.module.graph.nodes}
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with self.assertRaisesRegex(ValueError, "authenticated source model SHA-256"):
                save_region(capture, folder, "unauthenticated", (output,), selected)
            torch.export.save(program, folder / "parent.pt2")
            capture.source_model_sha256 = file_hash(folder / "parent.pt2")
            record = save_region(capture, folder, "region", (output,), selected)
            self.assertEqual(record["parent_capture"]["source_model_sha256"], capture.source_model_sha256)
            mapping = record["parent_capture"]["node_map"]
            self.assertTrue(mapping)
            for row in mapping:
                self.assertEqual(row["identity_metadata"]["custom"]["m2m_node_id"],
                                 identities[row["source_node"]]["m2m_node_id"])
            exported = torch.export.load(folder / "region" / "model.pt2")
            for node in exported.graph.nodes:
                self.assertNotIn("_m2m_node_id", node.meta)
                self.assertNotIn("_m2m_origin_ids", node.meta)
                self.assertNotIn("m2m_node_id", node.meta.get("custom", {}))
                self.assertNotIn("m2m_lineage", node.meta.get("custom", {}))
            tensors = load_file(folder / "region" / "tensors.safetensors")
            self.assertTrue(torch.equal(exported.module()(tensors["input_0"])[0],
                                        tensors["framework_output_0"]))
            self.assertEqual({node.name: node.meta.get("custom") for node in capture.module.graph.nodes}, identities)



class TinyCache(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("bias", torch.zeros((1, 1, 3), dtype=torch.bfloat16))

    def forward(self, ids, positions, mask, *past):
        current = mask[:, :, :, :1].expand(1, 2, ids.shape[1], 3)
        hidden = current[:, 0] + self.bias
        return (hidden * 2, *(torch.cat((value, current), dim=2) for value in past))


class TinyPrefill(TinyCache):
    def forward(self, ids, positions, mask):
        current = mask[:, :, :, :1].expand(1, 2, ids.shape[1], 3)
        hidden = current[:, 0] + self.bias
        return (hidden * 2, *(current + index for index in range(4)))


class CacheExtractionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.args = (torch.ones((1, 1), dtype=torch.int64), torch.tensor([[4]]),
                    torch.zeros((1, 1, 1, 5), dtype=torch.bfloat16),
                    *(torch.zeros((1, 2, 4, 3), dtype=torch.bfloat16) for _ in range(4)))
        cls.program = torch.export.export(TinyCache(), cls.args, strict=True)
        cls.abi = derive_cache_phase_abi(cls.program, layer_count=2, past_length=4, query_length=1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.bundle = Path(self.temporary.name)
        torch.export.save(self.program, self.bundle / "model.pt2")
        for name in ("verification.json", "fresh_process_verification.json"):
            (self.bundle / name).write_text('{}\n')
        self.manifest = dict(files={"model.pt2": dict(sha256=file_hash(self.bundle / "model.pt2"))},
            provenance=dict(phase="decode5", boundary="causal_lm_tensor_bf16_cache", raw_capture={"source": "fixed64"},
                            past_length=4, seqlen=1, total_length=5))
        self.write_json("fx_manifest.json", self.manifest)
        self.binding = dict(schema="atlas.source_cache_phase_binding.v1", abi=copy.deepcopy(self.abi),
            model_pt2_sha256=file_hash(self.bundle / "model.pt2"),
            source_fx_manifest_sha256=file_hash(self.bundle / "fx_manifest.json"))
        self.write_json("cache_phase_abi.json", self.binding)
        self.receipt = dict(schema="atlas.source_cache_capture.v1", status="SOURCE_CACHE_CAPTURED",
                            raw_identity={"source": "fixed64"}, phases={"decode5": dict(status="SOURCE_CAPTURED",
                                input_count=7, output_count=5)})
        self.rebind()

    def write_json(self, name, value):
        (self.bundle / name).write_text(json.dumps(value) + '\n')

    def rebind(self):
        phase = self.receipt["phases"][self.manifest["provenance"]["phase"]]
        for key, name in (("model_pt2_sha256", "model.pt2"), ("manifest_sha256", "fx_manifest.json"),
                          ("cache_phase_abi_sha256", "cache_phase_abi.json"),
                          ("verification_sha256", "verification.json"),
                          ("fresh_process_verification_sha256", "fresh_process_verification.json")):
            phase[key] = file_hash(self.bundle / name)
        self.write_json("cache_capture.json", self.receipt)

    def load_phase(self):
        return load_cache_phase(self.program, self.bundle, self.manifest,
                               self.bundle / "cache_phase_abi.json", self.bundle / "cache_capture.json")

    def test_authenticated_cache_binding_and_no_implicit_intake(self):
        phase = self.load_phase()
        self.assertEqual(phase["binding"]["abi"], self.abi)
        with self.assertRaisesRegex(ValueError, "requires"):
            load_cache_phase(self.program, self.bundle, self.manifest, None, None)
        with self.assertRaisesRegex(ValueError, "both"):
            load_cache_phase(self.program, self.bundle, self.manifest, self.bundle / "cache_phase_abi.json", None)

    def test_stale_sidecar_refuses_even_when_root_hash_is_updated(self):
        for field, value in (("role", "past_value"), ("source_node", "wrong_source"), ("shape", [1, 2, 99, 3])):
            changed = copy.deepcopy(self.binding)
            changed["abi"]["inputs"][3][field] = value
            self.write_json("cache_phase_abi.json", changed)
            self.rebind()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "actual exported"):
                self.load_phase()

    def test_corrupt_sidecar_or_source_binding_refuses(self):
        self.write_json("cache_phase_abi.json", {**self.binding, "model_pt2_sha256": "bad"})
        with self.assertRaisesRegex(ValueError, "file binding"):
            self.load_phase()
        self.rebind()
        with self.assertRaisesRegex(ValueError, "sidecar source"):
            self.load_phase()

    def test_incomplete_capture_receipts_refuse(self):
        for status in ("RUNNING", "SOURCE_CACHE_PARITY", "SOURCE_CACHE_PARITY_ONLY", "FAILED"):
            self.receipt["status"] = status
            self.write_json("cache_capture.json", self.receipt)
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "completed source"):
                self.load_phase()

    def test_region_reorders_hidden_before_cache_and_preserves_operations(self):
        capture = Capture(self.program.module())
        capture.source_program = self.program
        capture.run(*self.args)
        nodes = {node.name: node for node in capture.module.graph.nodes}
        key, value = (nodes[entry["source_node"]] for entry in self.abi["outputs"][1:3])
        hidden = next(node for node in nodes.values() if str(node.target) == "aten.add.Tensor")
        selected = {key, value, hidden}
        record = save_region(capture, self.bundle, "layer0", (key, value, hidden), selected, self.abi)
        self.assertEqual(record["source_outputs"], [hidden.name, key.name, value.name])
        self.assertEqual([entry["role"] for entry in record["cache_roles"]["outputs"]],
                         ["hidden_state", "updated_key", "updated_value"])
        saved = torch.export.load(self.bundle / "layer0" / "model.pt2")
        self.assertCountEqual([str(node.target) for node in saved.graph.nodes if node.op == "call_function"],
                              [str(node.target) for node in selected])
        tensors = load_file(self.bundle / "layer0" / "tensors.safetensors")
        self.assertTrue(torch.equal(tensors["framework_output_0"], capture.env[hidden]))
        self.assertTrue(torch.equal(tensors["framework_output_1"], capture.env[key]))

    def test_temporal_cache_inputs_are_separate_from_fixture_allowlist(self):
        regions = [dict(kind="layer0", source_boundaries=["embedding", *[x["source_node"] for x in self.abi["inputs"][3:5]]],
                        source_outputs=["hidden0", *[x["source_node"] for x in self.abi["outputs"][1:3]]]),
                   dict(kind="layer1", source_boundaries=["hidden0", *[x["source_node"] for x in self.abi["inputs"][5:]]],
                        source_outputs=["hidden1", *[x["source_node"] for x in self.abi["outputs"][3:]]])]
        graph = torch.fx.Graph()
        for name in {name for region in regions for field in ("source_boundaries", "source_outputs") for name in region[field]}:
            graph.placeholder(name)
        result = sequential_region_manifest(regions, graph.nodes, self.load_phase())
        self.assertEqual(result["schema"], "atlas.sequential_cache_phase_fx.v1")
        self.assertEqual(result["fixture_boundary_allowlist"], ["embedding"])
        self.assertEqual(len(result["external_cache_inputs"]), 4)
        self.assertEqual(result["links"][1]["inputs"][0]["source_node"], "hidden0")
        with self.assertRaisesRegex(ValueError, "unresolved generated"):
            sequential_region_manifest(regions, graph.nodes)
        incomplete = copy.deepcopy(regions)
        incomplete[1]["source_boundaries"].pop()
        with self.assertRaisesRegex(ValueError, "omitted a declared past"):
            sequential_region_manifest(incomplete, graph.nodes, self.load_phase())
        incomplete = copy.deepcopy(regions)
        incomplete[1]["source_outputs"].pop()
        with self.assertRaisesRegex(ValueError, "omitted a declared updated"):
            sequential_region_manifest(incomplete, graph.nodes, self.load_phase())
        regions[1]["source_boundaries"].append("undeclared")
        graph.placeholder("undeclared")
        with self.assertRaisesRegex(ValueError, "unresolved generated"):
            sequential_region_manifest(regions, graph.nodes, self.load_phase())

    def test_no_cache_intake_and_manifest_stay_unchanged(self):
        class NoCache(torch.nn.Module):
            def forward(self, ids):
                return ids + 1
        program = torch.export.export(NoCache(), (torch.ones(1),))
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(load_cache_phase(program, Path(directory), {}, None, None))
        graph = torch.fx.Graph()
        for name in ("embedding", "hidden0", "hidden1"):
            graph.placeholder(name)
        regions = [dict(kind="layer0", source_boundaries=["embedding"], source_outputs=["hidden0"]),
                   dict(kind="layer1", source_boundaries=["hidden0"], source_outputs=["hidden1"])]
        result = sequential_region_manifest(regions, graph.nodes)
        self.assertEqual(result["schema"], "atlas.sequential_unchanged_fx.v1")
        self.assertNotIn("cache_phase", result)
        self.assertNotIn("external_cache_inputs", result)

    def test_three_input_prefill_sidecar_and_region_roles(self):
        from cache_source import phase_positions_and_mask
        ids = torch.ones((1, 4), dtype=torch.int64)
        args = (ids, *phase_positions_and_mask(ids, 0))
        self.program = torch.export.export(TinyPrefill(), args, strict=True)
        self.abi = derive_cache_phase_abi(self.program, layer_count=2, past_length=0, query_length=4)
        torch.export.save(self.program, self.bundle / "model.pt2")
        self.manifest["files"]["model.pt2"]["sha256"] = file_hash(self.bundle / "model.pt2")
        self.manifest["provenance"].update(phase="prefill4", past_length=0, seqlen=4, total_length=4)
        self.write_json("fx_manifest.json", self.manifest)
        self.binding.update(abi=self.abi, model_pt2_sha256=file_hash(self.bundle / "model.pt2"),
                            source_fx_manifest_sha256=file_hash(self.bundle / "fx_manifest.json"))
        self.write_json("cache_phase_abi.json", self.binding)
        phase = self.receipt["phases"].pop("decode5")
        phase["input_count"] = 3
        self.receipt["phases"]["prefill4"] = phase
        self.rebind()
        self.assertIs(self.load_phase()["binding"]["abi"]["initial_cache_empty"], True)
        validate_phase_inputs(args, self.abi)
        with self.assertRaises(ValueError):
            validate_phase_inputs((ids, args[1] + 1, args[2]), self.abi)
        with self.assertRaises(ValueError):
            validate_phase_inputs((*args, torch.empty(0)), self.abi)
        capture = Capture(self.program.module())
        capture.source_program = self.program
        capture.run(*args)
        nodes = {node.name: node for node in capture.module.graph.nodes}
        key, value = (nodes[entry["source_node"]] for entry in self.abi["outputs"][1:3])
        hidden = next(node for node in nodes.values() if str(node.target) == "aten.add.Tensor")
        record = save_region(capture, self.bundle, "layer0", (key, value, hidden), {key, value, hidden}, self.abi)
        self.assertEqual([entry["role"] for entry in record["cache_roles"]["outputs"]],
                         ["hidden_state", "updated_key", "updated_value"])
        self.assertTrue(all(entry["role"] not in {"past_key", "past_value"}
                            for entry in record["cache_roles"]["inputs"]))

    def test_implicit_prefill_has_no_external_cache_inputs_and_requires_all_outputs(self):
        phase = self.load_phase()
        original = phase["binding"]["abi"]
        original["inputs"] = original["inputs"][:3]
        original.update(past_length=0, query_length=4, total_length=4, initial_cache_empty=True)
        regions = [dict(kind="layer0", source_boundaries=["embedding"],
                        source_outputs=["hidden0", *[entry["source_node"] for entry in original["outputs"][1:3]]]),
                   dict(kind="layer1", source_boundaries=["hidden0"],
                        source_outputs=["hidden1", *[entry["source_node"] for entry in original["outputs"][3:]]])]
        graph = torch.fx.Graph()
        for name in {name for region in regions for field in ("source_boundaries", "source_outputs") for name in region[field]}:
            graph.placeholder(name)
        result = sequential_region_manifest(regions, graph.nodes, phase)
        self.assertEqual(result["external_cache_inputs"], [])
        self.assertEqual(result["fixture_boundary_allowlist"], ["embedding"])
        for mode in ("missing_marker", "positive_past", "missing_output", "undeclared_input"):
            changed_phase, changed_regions = copy.deepcopy(phase), copy.deepcopy(regions)
            if mode == "missing_marker":
                del changed_phase["binding"]["abi"]["initial_cache_empty"]
            elif mode == "positive_past":
                changed_phase["binding"]["abi"]["past_length"] = 1
            elif mode == "missing_output":
                changed_regions[-1]["source_outputs"].pop()
            else:
                changed_regions[-1]["source_boundaries"].append("undeclared_cache")
                graph.placeholder("undeclared_cache")
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                sequential_region_manifest(changed_regions, graph.nodes, changed_phase)


if __name__ == '__main__':
    unittest.main()
