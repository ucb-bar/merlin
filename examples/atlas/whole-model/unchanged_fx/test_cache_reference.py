"""Small cache-oracle protocol tests, including optional frozen-Interpreter replay.

Numerical tests require explicit CACHE_REF_* paths to the existing independent
helper and qualifications. No original checkpoint or generated program is run.
"""
from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from safetensors.torch import save_file

import cache_reference as reference


def tensor_record(value):
    raw = value.detach().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(value.dtype), shape=list(value.shape), bytes=len(raw), sha256=sha256(raw).hexdigest())


class Layer(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(2, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, past_value, hidden, past_key):
        hidden = hidden * self.gain
        update = hidden.reshape(1, 1, hidden.shape[1], 2)
        key = torch.cat((past_key, update), dim=2)
        value = torch.cat((past_value, update), dim=2)
        # Deliberately differ from the global K/V order.
        return hidden.clone(), value, key


class PrefillLayer(Layer):
    def forward(self, hidden):
        hidden = hidden * self.gain
        update = hidden.reshape(1, 1, hidden.shape[1], 2)
        return hidden.clone(), update.clone(), update.clone()


class Identity(torch.nn.Module):
    def forward(self, hidden):
        return (hidden.clone(),)


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(2, dtype=torch.bfloat16), requires_grad=False)

    def forward(self, hidden):
        return (torch.nn.functional.linear(hidden, self.weight),)


def make_phase(root, past, query, implicit=False):
    root.mkdir()
    hidden = torch.full((1, query, 2), past + 1, dtype=torch.bfloat16)
    # These deliberately wrong framework cache values must never be consumed.
    poisoned = torch.full((1, 1, past, 2), 100, dtype=torch.bfloat16)
    stages = {
        "layer0": (Layer(), (poisoned.clone(), hidden, poisoned.clone()),
                   ["past_value0", "embedding", "past_key0"], ["hidden0", "updated_value0", "updated_key0"]),
        "norm": (Identity(), (hidden,), ["hidden0"], ["hidden_norm"]),
        "head": (Head(), (hidden,), ["hidden_norm"], ["logits"]),
    }
    if implicit:
        if past != 0:
            raise ValueError('implicit test prefill must start empty')
        stages['layer0'] = (PrefillLayer(), (hidden,), ['embedding'],
                            ['hidden0', 'updated_value0', 'updated_key0'])
    source_records = []
    for name, (module, args, boundaries, outputs) in stages.items():
        folder = root / name
        folder.mkdir()
        program = torch.export.export(module.eval(), args, strict=True)
        torch.export.save(program, folder / "model.pt2")
        arrays = {f"input_{i}": value.contiguous() for i, value in enumerate(args)}
        arrays["framework_output_0"] = torch.zeros_like(hidden)
        save_file(arrays, str(folder / "tensors.safetensors"))
        inputs = [dict(index=i, source_node=node, role="source_tensor") for i, node in enumerate(boundaries)]
        out_roles = [dict(index=i, source_node=node, role="source_tensor") for i, node in enumerate(outputs)]
        if name == "layer0":
            if not implicit:
                inputs[0].update(role="past_value", layer=0, global_index=4)
                inputs[2].update(role="past_key", layer=0, global_index=3)
            out_roles[0].update(role="hidden_state", layer=0)
            out_roles[1].update(role="updated_value", layer=0, global_index=2)
            out_roles[2].update(role="updated_key", layer=0, global_index=1)
        if name == "head":
            out_roles[0].update(role="logits", global_index=0)
        source = dict(kind=name, input_keys=[f"input_{i}" for i in range(len(args))],
            source_boundaries=boundaries, source_outputs=outputs,
            tensors={key: tensor_record(value) for key, value in arrays.items()},
            parameters={key: tensor_record(value) for key, value in program.state_dict.items()},
            model_sha256=reference.file_hash(folder / "model.pt2"),
            cache_roles=dict(inputs=inputs, outputs=out_roles))
        reference.write_json(folder / "source.json", source)
        source_records.append(source)
    shape_in, shape_out = [1, 1, past, 2], [1, 1, past + query, 2]
    abi = dict(layer_count=1, batch_size=1, kv_heads=1, head_dim=2, past_length=past,
        query_length=query, total_length=past + query,
        inputs=[dict(index=0, role="input_ids"), dict(index=1, role="position_ids"), dict(index=2, role="attention_mask"),
            dict(index=3, role="past_key", layer=0, source_node="past_key0", shape=shape_in, dtype="torch.bfloat16"),
            dict(index=4, role="past_value", layer=0, source_node="past_value0", shape=shape_in, dtype="torch.bfloat16")],
        outputs=[dict(index=0, role="logits", source_node="logits"),
            dict(index=1, role="updated_key", layer=0, source_node="updated_key0", shape=shape_out, dtype="torch.bfloat16"),
            dict(index=2, role="updated_value", layer=0, source_node="updated_value0", shape=shape_out, dtype="torch.bfloat16")])
    external = [dict(region="layer0", input_index=index, boundary=node, source_node=node,
        global_input_index=global_index, role=role, layer=0, shape=shape_in, dtype="torch.bfloat16")
        for index, node, global_index, role in ((0, "past_value0", 4, "past_value"), (2, "past_key0", 3, "past_key"))]
    if implicit:
        abi["inputs"] = abi["inputs"][:3]
        abi["initial_cache_empty"] = True
        external = []
    whole = dict(schema="atlas.sequential_cache_phase_fx.v1", regions=list(stages),
        links=[dict(kind="layer0", inputs=[]), dict(kind="norm", inputs=[dict(
            input_index=0, boundary="hidden0", source_node="hidden0", region="layer0", output_index=0)]),
            dict(kind="head", inputs=[dict(input_index=0, boundary="hidden_norm", source_node="hidden_norm", region="norm", output_index=0)])],
        fixture_boundary_allowlist=["embedding"], external_cache_inputs=external,
        cache_phase=dict(phase=f"past{past}query{query}", cache_capture_sha256="shared-capture",
                         binding=dict(schema="atlas.source_cache_phase_binding.v1", abi=abi,
                                      model_pt2_sha256=f"different-phase-graph-{past}-{query}")))
    manifest = dict(source_model_sha256=f"different-phase-graph-{past}-{query}",
                    source_provenance=dict(raw_capture={"export_manifest_sha256": "shared-raw-export"}),
                    regions=source_records, cache_phase=whole["cache_phase"], whole_model=whole)
    reference.write_json(root / "manifest.json", manifest)
    return reference.read_phase(root)


class FakeTensor:
    def __init__(self, raw, dtype):
        self.raw, self.dtype = raw, dtype


class CacheReferenceProtocolTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="cache-reference-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.phase = make_phase(self.root / "phase", 0, 4)

    def test_three_input_prefill_has_no_external_empty_tensor_roles(self):
        phase = make_phase(self.root / 'implicit', 0, 4, implicit=True)
        self.assertTrue(phase['abi']['initial_cache_empty'])
        self.assertEqual(phase['whole']['external_cache_inputs'], [])
        self.assertEqual(set(reference.cache_output_locations(phase)), {(0, 'key'), (0, 'value')})

    def test_exact_role_mapping_uses_cache_output_indices_not_zero(self):
        outputs = reference.cache_output_locations(self.phase)
        self.assertEqual(outputs[(0, "key")]["output_index"], 2)
        self.assertEqual(outputs[(0, "value")]["output_index"], 1)
        self.assertEqual(outputs[(0, "key")]["global_output_index"], 1)

    def test_cache_role_duplication_refuses(self):
        broken = deepcopy(self.phase)
        broken["sources"]["norm"]["cache_roles"]["outputs"][0] = deepcopy(
            broken["sources"]["layer0"]["cache_roles"]["outputs"][2])
        with self.assertRaisesRegex(ValueError, "unique exact"):
            reference.cache_output_locations(broken)

    def test_empty_prefill_seeds_are_independent_of_captured_cache_answers(self):
        consumed = []
        def fixture(key):
            consumed.append(key)
            if key != "input_1":
                raise AssertionError("captured cache fixture was read")
            return torch.ones((1, 4, 2), dtype=torch.bfloat16)
        helper = SimpleNamespace(Tensor=FakeTensor, raw_tensor=lambda value: FakeTensor(
            value.view(torch.uint16).numpy().copy(), "bf16"))
        values, origins = reference.stage_inputs(helper, self.phase, "layer0", {}, {}, fixture)
        self.assertEqual(consumed, ["input_1"])
        self.assertEqual(values[0].raw.shape, (1, 1, 0, 2))
        self.assertTrue(origins[0]["independent_empty_cache"])
        self.assertTrue(origins[2]["independent_empty_cache"])

    def test_nonempty_cache_without_both_parent_paths_refuses(self):
        phase = make_phase(self.root / "decode", 4, 1)
        for parent, capsules in ((None, None), (self.root, None), (None, self.root)):
            with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, "explicit parent"):
                reference.parent_caches(phase, parent, capsules, {})

    def test_no_framework_cache_fallback_when_independent_parent_missing(self):
        phase = make_phase(self.root / "decode", 4, 1)
        helper = SimpleNamespace(Tensor=FakeTensor)
        with self.assertRaisesRegex(ValueError, "fallback is forbidden"):
            reference.stage_inputs(helper, phase, "layer0", {}, {}, lambda key: self.fail("fixture fallback read"))

    def test_current_hidden_state_requires_prior_reference_output(self):
        with self.assertRaisesRegex(ValueError, "hidden predecessor"):
            reference.stage_inputs(None, self.phase, "norm", {}, {}, lambda key: self.fail("hidden fixture read"))

    def test_duplicate_external_cache_input_refuses(self):
        manifest = deepcopy(self.phase["manifest"])
        manifest["whole_model"]["external_cache_inputs"][1] = deepcopy(
            manifest["whole_model"]["external_cache_inputs"][0])
        reference.write_json(self.phase["root"] / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "global phase ABI"):
            reference.read_phase(self.phase["root"])

    def test_source_json_drift_from_embedded_manifest_record_refuses(self):
        path = self.phase["root"] / "layer0/source.json"
        source = reference.read_json(path)
        source["device_comparison"] = "modified after extraction"
        reference.write_json(path, source)
        with self.assertRaisesRegex(ValueError, "embedded source metadata"):
            reference.read_phase(self.phase["root"])

    def test_missing_embedded_region_refuses(self):
        manifest = deepcopy(self.phase["manifest"])
        manifest["regions"] = manifest["regions"][:-1]
        reference.write_json(self.phase["root"] / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "embedded source metadata"):
            reference.read_phase(self.phase["root"])

    def test_duplicate_embedded_region_refuses(self):
        manifest = deepcopy(self.phase["manifest"])
        manifest["regions"].append(deepcopy(manifest["regions"][0]))
        reference.write_json(self.phase["root"] / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "duplicate embedded source"):
            reference.read_phase(self.phase["root"])

    def test_top_level_cache_phase_drift_refuses(self):
        manifest = deepcopy(self.phase["manifest"])
        manifest["cache_phase"] = deepcopy(manifest["cache_phase"])
        manifest["cache_phase"]["phase"] = "different phase"
        reference.write_json(self.phase["root"] / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "top-level cache phase"):
            reference.read_phase(self.phase["root"])

    def test_global_source_archive_binding_drift_refuses(self):
        manifest = deepcopy(self.phase["manifest"])
        for phase in (manifest["cache_phase"], manifest["whole_model"]["cache_phase"]):
            phase["binding"]["model_pt2_sha256"] = "different source archive"
        reference.write_json(self.phase["root"] / "manifest.json", manifest)
        with self.assertRaisesRegex(ValueError, "source archive binding"):
            reference.read_phase(self.phase["root"])


@unittest.skipUnless(os.environ.get("CACHE_REF_DRIVER"), "explicit frozen reference qualification paths required")
class FrozenInterpreterCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        args = SimpleNamespace(model_root=Path(os.environ["CACHE_REF_MODEL_ROOT"]),
            reference_root=Path(os.environ["CACHE_REF_ROOT"]), helper_binding=Path(os.environ["CACHE_REF_BINDING"]),
            validation=Path(os.environ["CACHE_REF_VALIDATION"]), policy=Path(os.environ["CACHE_REF_POLICY"]),
            mask_validation=Path(os.environ["CACHE_REF_MASK_VALIDATION"]),
            bf16_linear_validation=Path(os.environ["CACHE_REF_LINEAR_VALIDATION"]), engine="numpy", native_qualification=None)
        cls.args = args
        cls.helper, binding, sources = reference.load_reference_driver(Path(os.environ["CACHE_REF_DRIVER"]),
            args.helper_binding, args.model_root, args.reference_root)
        cls.binding, cls.sources = binding, sources
        cls.policy, cls.proof = reference.qualify(cls.helper, binding, sources, args)
        cls.proof["model_root"] = str(args.model_root)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ["TMPDIR"], prefix="frozen-cache-reference-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def chain(self):
        phases, results, outputs = [], [], []
        for index, (past, query) in enumerate(((0, 4), (4, 1), (5, 1))):
            phase = make_phase(self.root / f"phase{index}", past, query)
            out = self.root / f"reference{index}"
            result = reference.run_phase(self.helper, phase["root"], out, self.policy, self.proof,
                parent_reference=None if not index else outputs[-1],
                parent_capsules=None if not index else phases[-1]["root"], retain=("*",))
            phases.append(phase)
            results.append(result)
            outputs.append(out)
        return phases, results, outputs

    def test_positive_prefill_then_nonempty_decode_through_frozen_interpreter(self):
        prefill = make_phase(self.root / 'implicit-prefill', 0, 4, implicit=True)
        out = self.root / 'implicit-reference'
        reference.run_phase(self.helper, prefill['root'], out, self.policy, self.proof)
        decode = make_phase(self.root / 'positive-decode', 4, 1)
        final = self.root / 'positive-reference'
        result = reference.run_phase(self.helper, decode['root'], final, self.policy, self.proof,
            parent_reference=out, parent_capsules=prefill['root'])
        self.assertEqual(result['status'], 'EXPECTED')
        with np.load(final / 'layer0/runtime_inputs.npz', allow_pickle=False) as consumed:
            self.assertTrue(np.all(consumed['input_0'] == np.uint16(0x3f80)))

    def test_prefill_two_decodes_use_only_expected_parent_words_and_keep_all_outputs(self):
        phases, results, outputs = self.chain()
        expected_values = [1, 1, 1, 1, 5, 6]
        expected = torch.tensor(expected_values, dtype=torch.bfloat16).view(torch.uint16).numpy()
        with np.load(outputs[-1] / "layer0/expected.npz", allow_pickle=False) as arrays:
            self.assertTrue(np.array_equal(arrays["output_2"][0, 0, :, 0], expected))
            self.assertTrue(np.array_equal(arrays["output_1"][0, 0, :, 0], expected))
            self.assertGreater(len(arrays.files), 3)
        for index, result in enumerate(results):
            self.assertEqual(result["status"], "EXPECTED")
            self.assertFalse(result["ordinary_framework_kernels_called"])
            self.assertFalse(result["compiler_or_emitted_program_inputs"])
            self.assertEqual(len(result["stages"]), 3)
            for stage in result["stages"]:
                self.assertEqual(reference.read_json(outputs[index] / stage["stage"] / "reference.json"), stage)
            if index:
                self.assertEqual(result["parent_reference_receipt_sha256"], reference.file_hash(outputs[index - 1] / "reference.json"))
                self.assertNotEqual(result["source_model_sha256"], results[index - 1]["source_model_sha256"])
                incoming = reference.read_json(outputs[index] / "layer0/inputs.json")
                self.assertTrue(incoming["input_0"]["origin"]["independent_parent_cache"])
                self.assertEqual(incoming["input_0"]["origin"]["output_index"], 1)
                self.assertEqual(incoming["input_2"]["origin"]["output_index"], 2)

    def test_corrupt_parent_cache_word_refuses(self):
        parent = make_phase(self.root / "parent", 0, 4)
        out = self.root / "parent-reference"
        reference.run_phase(self.helper, parent["root"], out, self.policy, self.proof)
        path = out / "layer0/expected.npz"
        with np.load(path, allow_pickle=False) as source:
            arrays = {key: source[key] for key in source.files}
        arrays["output_2"][0, 0, 0, 0] ^= np.uint16(1)
        np.savez(path, **arrays)
        current = make_phase(self.root / "decode", 4, 1)
        with self.assertRaisesRegex(ValueError, "cache words differ"):
            reference.parent_caches(current, out, parent["root"], self.proof)

    def test_generated_receipt_cannot_supply_parent_cache(self):
        parent = make_phase(self.root / "parent", 0, 4)
        out = self.root / "generated"
        out.mkdir()
        reference.write_json(out / "reference.json", {"status": "PASS", "oracle_mode": "generated_execution"})
        current = make_phase(self.root / "decode", 4, 1)
        with self.assertRaisesRegex(ValueError, "completed independent"):
            reference.parent_caches(current, out, parent["root"], self.proof)

    def parent_and_current(self):
        parent = make_phase(self.root / "parent", 0, 4)
        out = self.root / "parent-reference"
        reference.run_phase(self.helper, parent["root"], out, self.policy, self.proof)
        current = make_phase(self.root / "decode", 4, 1)
        return parent, out, current

    def test_changed_shared_parameter_inventory_refuses(self):
        parent, out, current = self.parent_and_current()
        current["parameters"]["gain"]["sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "unchanged source parameters"):
            reference.parent_caches(current, out, parent["root"], self.proof)

    def test_parent_arithmetic_binding_mismatch_refuses(self):
        parent, out, current = self.parent_and_current()
        record = reference.read_json(out / "reference.json")
        record["validation_sha256"] = "different arithmetic gate"
        reference.write_json(out / "reference.json", record)
        with self.assertRaisesRegex(ValueError, "arithmetic binding differs"):
            reference.parent_caches(current, out, parent["root"], self.proof)

    def test_parent_stage_json_must_equal_completed_root_record(self):
        parent, out, current = self.parent_and_current()
        stage = reference.read_json(out / "layer0/reference.json")
        stage["outputs"]["output_2"]["sha256"] = "changed publication"
        reference.write_json(out / "layer0/reference.json", stage)
        with self.assertRaisesRegex(ValueError, "publication differs"):
            reference.parent_caches(current, out, parent["root"], self.proof)

    def test_wrong_prefix_length_refuses(self):
        parent, out, _ = self.parent_and_current()
        wrong = make_phase(self.root / "wrong-length", 5, 1)
        with self.assertRaisesRegex(ValueError, "complete prefix"):
            reference.parent_caches(wrong, out, parent["root"], self.proof)

    def test_wrong_frozen_driver_binding_refuses_before_import(self):
        binding = deepcopy(self.binding)
        binding["adapter_sha256"] = "0" * 64
        path = self.root / "stale-helper-binding.json"
        reference.write_json(path, binding)
        with self.assertRaisesRegex(ValueError, "matching helper binding"):
            reference.load_reference_driver(Path(os.environ["CACHE_REF_DRIVER"]), path,
                                            self.args.model_root, self.args.reference_root)

    def test_different_policy_receipt_cannot_reuse_qualification(self):
        binding = deepcopy(self.binding)
        binding["selected_policy_sha256"] = "different selected policy"
        with self.assertRaisesRegex(ValueError, "policy receipts differ"):
            reference.qualify(self.helper, binding, self.sources, self.args)

    def test_archive_drift_after_load_refuses_before_interpreter(self):
        phase = make_phase(self.root / "phase", 0, 4)
        original_load = torch.export.load
        def changed_after_load(path):
            result = original_load(path)
            Path(path).write_bytes(b"modified after deserialization")
            return result
        with patch.object(torch.export, "load", side_effect=changed_after_load), patch.object(
                self.helper, "Interpreter") as interpreter:
            with self.assertRaisesRegex(ValueError, "changed before interpretation"):
                reference.run_phase(self.helper, phase["root"], self.root / "out", self.policy, self.proof)
            interpreter.assert_not_called()
        self.assertEqual(reference.read_json(self.root / "out/reference.json")["status"], "REFUSED")


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main(verbosity=2)
