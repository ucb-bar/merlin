"""State routing and real LLVM frontend checks over a diagnostic two-stage session."""
from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest

from merlin.common.paths import repo_root


def get_model_and_inputs(*, model2mlir_root):
    """Explicit local CLI fixture; this diagnostic contains no model checkpoint."""
    import torch
    from verify_session import load_protocol
    fixture = SessionTests()
    fixture.torch, fixture.protocol = torch, load_protocol(Path(model2mlir_root))
    session = fixture.session()
    return SimpleNamespace(external_runtime_session=lambda: session), ()


class SessionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        root = os.environ.get("TVM_SESSION_MODEL2MLIR")
        if not root:
            raise unittest.SkipTest("Set TVM_SESSION_MODEL2MLIR to the existing model2MLIR protocol checkout")
        sys.path.insert(0, root)
        path = repo_root() / "examples/gemmini/comparisons/tvm/verify_session.py"
        sys.path.insert(0, str(path.parent))
        spec = importlib.util.spec_from_file_location("tvm_session_verifier", path)
        cls.verifier = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.verifier)
        cls.protocol = cls.verifier.load_protocol(Path(root))

    def session(self):
        torch, protocol = self.torch, self.protocol

        class Prefix(torch.nn.Module):
            def forward(self, image):
                return image + 1

        class Decode(torch.nn.Module):
            def forward(self, token, state):
                return state + token * 2, state + token

        tokens = torch.tensor([[1., -2.], [-3., 4.], [2., 1.]], dtype=torch.float32)
        spec = {"steps": 3, "streams": [{"name": "token", "input_index": 0, "values": tokens}],
                "states": [{"name": "state", "input_index": 1, "output_index": 1}], "quality": {"output_index": 0}}
        return protocol.make_external_runtime_session(
            version=2,
            programs=(protocol.ExternalRuntimeProgram("prefill", Prefix().eval(), (torch.tensor([0., 1.]),), 1),
                      protocol.ExternalRuntimeProgram("decode", Decode().eval(), (tokens[0], torch.zeros(2)), 3, spec)),
            metadata={"kind": "diagnostic_recurrence", "paper_ready": False, "stages": ["prefill", "decode"],
                      "stage_schedule": [{"name": "prefill", "steps": 1}, {"name": "decode", "steps": 3}],
                      "quality_program": "decode", "bindings": [{"name": "initial_state", "from": {"program": "prefill", "output_index": 0},
                                                                 "to": {"program": "decode", "input_index": 1}}],
                      "provenance": {"checkpoint": "randomless_toy", "full_checkpoint": False, "synthetic_inputs": True}})

    def reference_compiler(self, program, inputs, output_count):
        def run(values):
            return program.module(*(self.verifier.torch_array(value, self.np, self.torch) for value in values))
        return run

    def adapter_fixture(self, session=None, compiler=None):
        trace = {}
        result = self.verifier.verify_session(session or self.session(), compiler or self.reference_compiler, self.np, self.torch, adapter_trace=trace)
        records, stages, seen = [], {}, set()
        for call in trace["calls"]:
            signature = self.verifier.static_signature(call["program"], call["inputs"])
            if signature not in seen:
                seen.add(signature)
                name = "stage_" + str(len(records))
                records.append({"program": call["program"], "inputs": call["inputs"]})
                stages[name] = {"inputs": call["inputs"], "outputs": call["outputs"]}
        package = {"stages": stages, "constants_bytes": 64, "duplicated_stage_constants_bytes": 128, "serial_workspace_bytes": 128}
        return result, trace, records, package

    def test_automatic_adapter_routes_and_source_fixture_are_independent(self):
        def perturbed(program, inputs, output_count):
            run = self.reference_compiler(program, inputs, output_count)
            def invoke(values):
                result = self.verifier.outputs(run(values), self.np, self.torch)
                result[0] += self.np.float32(1e-5)
                return tuple(self.verifier.torch_array(value, self.np, self.torch) for value in result)
            return invoke
        result, trace, records, package = self.adapter_fixture(compiler=perturbed)
        route, order = self.verifier.session_adapter_plan(trace, records, package)
        self.assertTrue(result["reset_verified"])
        self.assertEqual(route, {"version": 1, "calls": [{"stage": "stage_0", "inputs": [{"fixture": 0}]},
                                                        {"stage": "stage_1", "inputs": [{"fixture": 1}, {"output": [0, 0]}]},
                                                        {"stage": "stage_1", "inputs": [{"fixture": 2}, {"output": [1, 1]}]},
                                                        {"stage": "stage_1", "inputs": [{"fixture": 3}, {"output": [2, 1]}]}]})
        expected = [[1., 2.], [3., -2.], [2., 0.], [-4., 8.], [-1., 4.], [3., 6.], [1., 5.]]
        self.assertEqual([(value["invocation"], value["output_index"]) for value in order], [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)])
        with tempfile.TemporaryDirectory(prefix="tvm-session-adapter-") as temp:
            source = self.verifier.file_identity(__file__, self.verifier.safe_path)
            report = self.verifier.write_session_adapter(Path(temp) / "session", trace, records, package, self.np, reference_sources=[source])
            with self.np.load(report["fixture"]["path"], allow_pickle=False) as fixture:
                self.assertEqual(set(fixture.files), {*("input_" + str(index) for index in range(4)), *("output_" + str(index) for index in range(7))})
                for index, value in enumerate(expected):
                    self.np.testing.assert_array_equal(fixture["output_" + str(index)], self.np.array([value], dtype="float32"))
            provenance = json.loads(Path(report["fixture_provenance"]["path"]).read_text())
            self.assertEqual(provenance["reference_sources"], [source])
            self.assertEqual(provenance["fixture_sha256"], report["fixture"]["sha256"])
            self.assertEqual(provenance["observation_outputs"], [[1, 0], [2, 0], [3, 0]])
            self.assertEqual(provenance["final_output"], [3, 0])
            self.assertFalse(report["device_admission_verified"])

    def test_automatic_adapter_admission_refuses_state_and_abi_changes(self):
        import copy
        _, trace, records, package = self.adapter_fixture()
        cases = []
        invalid = copy.deepcopy(trace); invalid["host_verified"] = False
        cases.append((invalid, records, package, "host gates"))
        invalid = copy.deepcopy(trace); invalid["source_reset_stable"] = False
        cases.append((invalid, records, package, "source resets"))
        invalid = copy.deepcopy(trace); invalid["calls"][1]["routes"][1] = {"output": [1, 0]}
        cases.append((invalid, records, package, "earlier invocation"))
        invalid = copy.deepcopy(trace); invalid["calls"][0]["routes"][0] = {"fixture": True}
        cases.append((invalid, records, package, "plain integer"))
        invalid = copy.deepcopy(trace); invalid["calls"][1]["routes"][1] = {"cast": [0, 0]}
        cases.append((invalid, records, package, "No transforms"))
        invalid = copy.deepcopy(trace); invalid["calls"][1]["routes"][1] = {"output": [0, 1]}
        cases.append((invalid, records, package, "earlier invocation"))
        invalid = copy.deepcopy(trace); invalid["calls"][0]["values"][0][0] += 1
        cases.append((invalid, records, package, "output bytes changed"))
        invalid = copy.deepcopy(trace); invalid["calls"][0]["state_after"] = [[1, 0]]
        cases.append((invalid, records, package, "future or absent"))
        invalid_package = copy.deepcopy(package); invalid_package["stages"]["stage_1"]["outputs"][0] = {**invalid_package["stages"]["stage_1"]["outputs"][0], "dtype": "float64"}
        cases.append((trace, records, invalid_package, "source ABI"))
        for invalid, inventory, exported, error in cases:
            with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                self.verifier.session_adapter_plan(invalid, inventory, exported)
        with tempfile.TemporaryDirectory(prefix="tvm-session-rejected-adapter-") as temp:
            destination = Path(temp) / "session"
            with self.assertRaisesRegex(ValueError, "identities"):
                self.verifier.write_session_adapter(destination, trace, records, package, self.np, reference_sources=[])
            self.assertFalse(destination.exists())
            with self.assertRaisesRegex(ValueError, "explicit export budget"):
                self.verifier.write_session_adapter(destination, trace, records, package, self.np,
                                                    reference_sources=[self.verifier.file_identity(__file__, self.verifier.safe_path)], memory_limit_bytes=767)
            self.assertFalse(destination.exists())

    def test_session_memory_counts_simultaneous_state_and_trace_separately(self):
        _, trace, records, package = self.adapter_fixture()
        self.verifier.session_adapter_plan(trace, records, package)
        memory = self.verifier.session_memory_plan(trace, package)
        self.assertEqual(memory["staged_external_inputs_bytes"], 256)
        self.assertEqual(memory["peak_planned_tensor_bytes"], 768)
        self.assertEqual(memory["invocations"][1]["routed_prior_input_bytes"], 64)
        self.assertEqual(memory["invocations"][1]["current_output_bytes"], 128)
        self.assertEqual(memory["verification_trace"]["single_trace_tensor_bytes"], 896)
        self.assertGreater(memory["verification_trace"]["single_trace_tensor_bytes"], memory["peak_planned_tensor_bytes"])
        self.assertFalse(memory["production_fit_established"])
        self.assertIsNone(memory["complete_production_peak_bytes"])
        self.assertIn("stack", memory["unknown_overheads"])
        for value in (True, 1., -1):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "plain nonnegative"):
                self.verifier.session_memory_plan(trace, {**package, "constants_bytes": value})

    def test_automatic_adapter_growing_state_uses_actual_static_signatures(self):
        _, trace, records, package = self.adapter_fixture(self.growing_session())
        route, order = self.verifier.session_adapter_plan(trace, records, package)
        self.assertEqual([call["stage"] for call in route["calls"]], ["stage_0", "stage_1", "stage_2"])
        self.assertEqual([record["inputs"][1]["shape"] for record in records], [[1], [2], [3]])
        self.assertEqual([call["inputs"][1] for call in route["calls"]], [{"fixture": 1}, {"output": [0, 1]}, {"output": [1, 1]}])
        self.assertEqual(len(order), 6)

    def test_automatic_adapter_bf16_preserves_raw_payload(self):
        torch, np, protocol = self.torch, self.np, self.protocol
        class Identity(torch.nn.Module):
            def forward(self, value):
                return value.clone()
        words = np.array([0x0000, 0x8000, 0x0001, 0x3f80, 0xbf80], dtype=np.uint16)
        values = torch.from_numpy(words.copy()).view(torch.bfloat16)
        program = protocol.ExternalRuntimeProgram("raw_bf16", Identity().eval(), (values,), 1, {"steps": 1, "quality": {"output_index": 0}})
        session = protocol.make_external_runtime_session(version=1, programs=(program,), metadata={})
        _, trace, records, package = self.adapter_fixture(session)
        with tempfile.TemporaryDirectory(prefix="tvm-session-bf16-adapter-") as temp:
            report = self.verifier.write_session_adapter(Path(temp) / "session", trace, records, package, np, reference_sources=[self.verifier.file_identity(__file__, self.verifier.safe_path)])
            with np.load(report["fixture"]["path"], allow_pickle=False) as fixture:
                self.assertEqual(fixture["output_0"].dtype, np.uint16)
                np.testing.assert_array_equal(fixture["input_0"], words[None])
                np.testing.assert_array_equal(fixture["output_0"], words[None])

    def test_adapter_trace_cannot_bypass_host_failure(self):
        import copy
        import onnx
        from unittest.mock import Mock, patch
        trace = {}
        def wrong(program, inputs, output_count):
            run = self.reference_compiler(program, inputs, output_count)
            return lambda values: tuple(self.verifier.torch_array(value + 10, self.np, self.torch) for value in self.verifier.outputs(run(values), self.np, self.torch))
        with self.assertRaises(self.verifier.SessionComparisonError):
            self.verifier.verify_session(self.session(), wrong, self.np, self.torch, adapter_trace=trace)
        self.assertFalse(trace["host_verified"])
        with self.assertRaisesRegex(ValueError, "host gates"):
            self.verifier.session_adapter_plan(trace, [], {})
        _, valid, records, _ = self.adapter_fixture()
        source = self.verifier.file_identity(__file__, self.verifier.safe_path)
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None)
        compiler.records, compiler.imported_modules = records, [object()] * len(records)
        unstable = copy.deepcopy(valid); unstable["source_reset_stable"] = False
        bounded = copy.deepcopy(valid); bounded["capture_error"] = "bounded adapter trace"
        for refused, sources, error in ((trace, [source], "host gates"), (unstable, [source], "source resets"), (bounded, [source], "bounded adapter trace"), (valid, [], "identities")):
            exporter = SimpleNamespace(export_graphs=Mock())
            with self.subTest(error=error), patch.object(self.verifier, "load_module", return_value=exporter) as loader, self.assertRaisesRegex(ValueError, error):
                compiler.export_stages(Path("unused-export"), memory_limit_bytes=1 << 20, adapter_trace=refused, reference_sources=sources)
            loader.assert_not_called()
            exporter.export_graphs.assert_not_called()

    def test_adapter_source_reset_instability_and_trace_bound_preserve_host_result(self):
        torch, protocol = self.torch, self.protocol
        class Stateful(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = 0
            def forward(self, value):
                self.calls += 1
                return value + self.calls * 1e-6
        program = protocol.ExternalRuntimeProgram("unstable", Stateful().eval(), (torch.ones(2),), 2, {"steps": 2, "quality": {"output_index": 0}})
        session = protocol.make_external_runtime_session(version=1, programs=(program,), metadata={})
        trace = {}
        result = self.verifier.verify_session(session, lambda program, inputs, count: lambda values: torch.from_numpy(values[0].copy()), self.np, torch, adapter_trace=trace)
        self.assertTrue(result["reset_verified"])
        self.assertTrue(trace["host_verified"])
        self.assertFalse(trace["source_reset_stable"])
        with self.assertRaisesRegex(ValueError, "source resets"):
            self.verifier.session_adapter_plan(trace, [], {})
        class Identity(torch.nn.Module):
            def forward(self, value):
                return value.clone()
        program = protocol.ExternalRuntimeProgram("bounded", Identity().eval(), (torch.ones(2),), 65, {"steps": 65, "quality": {"output_index": 0}})
        session = protocol.make_external_runtime_session(version=1, programs=(program,), metadata={})
        result, trace, records, package = self.adapter_fixture(session)
        self.assertTrue(result["reset_verified"])
        self.assertEqual(len(result["checks"]), 130)
        self.assertEqual(len(trace["calls"]), 64)
        with self.assertRaisesRegex(ValueError, "bounded adapter trace"):
            self.verifier.session_adapter_plan(trace, records, package)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit modern exporter and LLVM host build")
    def test_integer_session_automatic_adapter_with_complete_k_oracle(self):
        from contextlib import nullcontext
        import onnx
        import tvm
        import tinyllama_integer
        torch, np, protocol = self.torch, self.np, self.protocol
        weight = (torch.arange(51, dtype=torch.int32).reshape(17, 3) % 7 - 3).to(torch.int8)
        image = (torch.arange(34, dtype=torch.int32).reshape(2, 17) % 9 - 4).to(torch.int8)
        tokens = torch.stack((image + 1, -image, image - 2))
        class Prefix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", weight)
            def forward(self, image):
                value = torch._int_mm(image, self.weight)
                return value, value + 3
        class Decode(Prefix):
            def forward(self, token, state):
                value = torch._int_mm(token, self.weight) + state
                return value, value + 3
        spec = {"steps": 3, "streams": [{"name": "token", "input_index": 0, "values": tokens}],
                "states": [{"name": "state", "input_index": 1, "output_index": 1}], "quality": {"output_index": 0}}
        session = protocol.make_external_runtime_session(version=2,
            programs=(protocol.ExternalRuntimeProgram("prefill", Prefix().eval(), (image,), 1),
                      protocol.ExternalRuntimeProgram("decode", Decode().eval(), (tokens[0], torch.zeros((2, 3), dtype=torch.int32)), 3, spec)),
            metadata={"kind": "synthetic_integer_adapter_probe", "paper_ready": False, "quality_program": "decode",
                      "stages": ["prefill", "decode"], "stage_schedule": [{"name": "prefill", "steps": 1}, {"name": "decode", "steps": 3}],
                      "bindings": [{"name": "initial_state", "from": {"program": "prefill", "output_index": 1}, "to": {"program": "decode", "input_index": 1}}],
                      "provenance": {"checkpoint": "none", "synthetic_inputs": True, "full_checkpoint": False}})
        def dot(left):
            return np.array([[sum(int(left[row, k]) * int(weight[k, column]) for k in range(17)) for column in range(3)] for row in range(2)], dtype=np.int32)
        oracle = []
        value = dot(image)
        oracle.extend((value, value + 3))
        state = value + 3
        for token in tokens:
            value = dot(token) + state
            oracle.extend((value, value + 3))
            state = value + 3
        destination = os.environ.get("TVM_SESSION_INTEGER_PROBE_DIR")
        context = nullcontext(destination) if destination else tempfile.TemporaryDirectory(prefix="tvm-session-integer-adapter-")
        with context as folder:
            out = self.verifier.safe_path(Path(folder))
            if destination:
                out.mkdir(parents=True, exist_ok=False)
            compiler = self.verifier.StageCompiler(out, "optimized", np, torch, onnx, tvm, exporter="dynamo", export_options=tinyllama_integer.export_options(torch))
            trace = {}
            result = self.verifier.verify_session(session, compiler, np, torch, adapter_trace=trace)
            references = [self.verifier.file_identity(path, self.verifier.safe_path) for path in (__file__, self.verifier.__file__, protocol.__file__, tinyllama_integer.__file__)]
            exported = compiler.export_stages(out / "baremetal", memory_limit_bytes=1 << 20, adapter_trace=trace, reference_sources=references)
            adapter = exported["automatic_session_adapter"]
            self.assertEqual(len(result["checks"]), 8)
            self.assertTrue(result["reset_verified"])
            self.assertEqual(adapter["invocations"], 4)
            self.assertEqual(adapter["retained_invocation_outputs"], 8)
            with np.load(adapter["fixture"]["path"], allow_pickle=False) as fixture:
                for index, expected in enumerate(oracle):
                    np.testing.assert_array_equal(fixture["output_" + str(index)], expected[None])
            for record in compiler.records:
                graph = onnx.load(record["onnx"]["path"])
                self.assertEqual(sum(node.op_type == "MatMulInteger" for node in graph.graph.node), 1)
            (out / "host.json").write_text(json.dumps({"host": result, "baremetal_export": exported, "source_identities": references,
                                                       "oracle": "Complete-K Python integer sum over signed source operands; exact all eight invocation outputs; no TVM evaluation", "model_qualified": False}, indent=2, sort_keys=True) + "\n")

    def test_exporter_and_schema_selection_are_explicit(self):
        import onnx
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None)
        self.assertEqual(compiler.opset, 17)
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, exporter="dynamo")
        self.assertEqual(compiler.opset, 18)
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, opset=22)
        self.assertEqual(compiler.opset, 22)
        for value in (0, True, "22", onnx.defs.onnx_opset_version() + 1):
            with self.subTest(opset=value), self.assertRaisesRegex(ValueError, "opset"):
                self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, opset=value)
        with self.assertRaisesRegex(ValueError, "exporter"):
            self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, exporter="unknown")
        for options in ({}, {"exporter": "dynamo", "opset": 17}):
            with self.assertRaisesRegex(ValueError, "Fused BF16 export"):
                self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, bf16_fused_export=True, **options)
        self.assertEqual(compiler.export_options, {})
        self.assertIsNone(compiler.translation_source)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_factory_translation_options_preserve_verifier_controls(self):
        import onnx
        options = {"custom_translation_table": {self.torch.ops.aten._int_mm.default: lambda a, b: (a, b)}, "optimize": False}
        compiler = self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, exporter="dynamo", export_options=options)
        self.assertFalse(compiler.bf16_fused_export)
        self.assertIn(self.torch.ops.aten.scalar_tensor.default, compiler.export_options["custom_translation_table"])
        self.assertIs(compiler.export_options["custom_translation_table"][self.torch.ops.aten._int_mm.default], options["custom_translation_table"][self.torch.ops.aten._int_mm.default])
        self.assertFalse(compiler.export_options["optimize"])
        options["custom_translation_table"].clear()
        self.assertEqual(len(compiler.export_options["custom_translation_table"]), 2)
        for invalid in ({"dynamo": True}, {"opset_version": 22}, {"optimize": 1}, {"custom_translation_table": []}, {"custom_translation_table": {"op": 1}}):
            with self.subTest(options=invalid), self.assertRaises(ValueError):
                self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, exporter="dynamo", export_options=invalid)
        with self.assertRaisesRegex(ValueError, "explicit dynamo"):
            self.verifier.StageCompiler(Path("unused"), "optimized", self.np, self.torch, onnx, None, export_options={"custom_translation_table": {"op": lambda: None}})

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_dynamo_export_preserves_typed_scalar_values(self):
        import math
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class Scalars(torch.nn.Module):
            def forward(self, double_input, float_input):
                return (double_input * torch.scalar_tensor(0.004, dtype=torch.float64),
                        double_input + torch.scalar_tensor(math.pi, dtype=torch.float64),
                        float_input * torch.scalar_tensor(0.004, dtype=torch.float32),
                        torch.scalar_tensor(2**53 + 1, dtype=torch.int64),
                        torch.scalar_tensor(-(2**53 + 1), dtype=torch.int64),
                        torch.scalar_tensor(True, dtype=torch.bool),
                        torch.scalar_tensor(-0.0, dtype=torch.float64))

        inputs = (np.array([0.25, -1.0], dtype=np.float64), np.array([0.25, -1.0], dtype=np.float32))
        expected = (inputs[0] * np.float64(0.004), inputs[0] + np.float64(math.pi), inputs[1] * np.float32(0.004),
                    np.array(2**53 + 1, dtype=np.int64), np.array(-(2**53 + 1), dtype=np.int64), np.array(True), np.array(-0.0, dtype=np.float64))
        program = SimpleNamespace(name="typed_scalars", module=Scalars().eval())
        for mode in ("baseline", "optimized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="tvm-session-scalars-") as temp:
                compiler = self.verifier.StageCompiler(Path(temp), mode, np, torch, onnx, tvm, exporter="dynamo", opset=22)
                run = compiler(program, inputs, len(expected))
                for _ in range(2):
                    actual = self.verifier.outputs(run(inputs), np, torch)
                    for result, reference in zip(actual, expected):
                        self.assertEqual((result.dtype, result.shape), (reference.dtype, reference.shape))
                        self.assertEqual(result.tobytes(), reference.tobytes())
                record = compiler.records[0]
                self.assertEqual(record["translation_source"], compiler.translation_source)
                self.assertIn("scalar_tensor_export", record)
                self.assertNotIn("bf16_fused_arithmetic", record)
                graph = onnx.load(str(Path(temp) / "stage_0/model.onnx"))
                constants = [onnx.numpy_helper.to_array(tensor) for tensor in graph.graph.initializer]
                constants += [onnx.numpy_helper.to_array(attr.t) for node in graph.graph.node if node.op_type == "Constant" for attr in node.attribute if attr.name == "value"]
                double_words = {value.tobytes() for value in constants if value.dtype == np.float64 and value.size == 1}
                self.assertIn(np.array(0.004, dtype=np.float64).tobytes(), double_words)
                self.assertIn(np.array(math.pi, dtype=np.float64).tobytes(), double_words)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_export_preserves_bias_boundary_and_attention_dtypes(self):
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class Fused(torch.nn.Module):
            def __init__(self):
                super().__init__()
                weight = torch.zeros((3, 17), dtype=torch.bfloat16)
                weight[:, :2] = 1
                self.register_buffer("weight", weight)
                self.register_buffer("bias", torch.full((3,), 1 / 256, dtype=torch.bfloat16))

            def forward(self, image, q, k, v):
                return torch.nn.functional.linear(image, self.weight, self.bias), torch.nn.functional.scaled_dot_product_attention(q, k, v)

        image = torch.zeros((1, 2, 17), dtype=torch.bfloat16)
        image[:, :, 0], image[:, :, 1] = 1, 1 / 256
        q, k = torch.zeros((1, 2, 17, 8), dtype=torch.bfloat16), torch.zeros((1, 2, 17, 8), dtype=torch.bfloat16)
        v = torch.arange(17, dtype=torch.bfloat16).reshape(1, 1, 17, 1).expand(1, 2, 17, 8).contiguous()
        values, module = (image, q, k, v), Fused().eval()
        with torch.no_grad():
            expected = module(*values)
            split = (image @ module.weight.t()) + module.bias
        self.assertFalse(torch.equal(split, expected[0]))
        inputs = tuple(self.verifier.array(value, np, torch) for value in values)
        program = SimpleNamespace(name="fused_bf16", module=module)
        original_weights = self.verifier.parameter_digest(module, torch)
        with tempfile.TemporaryDirectory(prefix="tvm-session-fused-bf16-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
            actual = self.verifier.outputs(compiler(program, inputs, 2)(inputs), np, torch)
            for value, reference in zip(actual, expected):
                self.verifier.compare(value, self.verifier.array(reference, np, torch), np, 0, 0)
            graph = onnx.shape_inference.infer_shapes(onnx.load(str(Path(temp) / "stage_0/model.onnx")))
            types = {value.name: value.type.tensor_type.elem_type for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
            for node in graph.graph.node:
                if node.op_type in ("MatMul", "Add", "Mul", "Softmax"):
                    self.assertTrue(all(types[name] == onnx.TensorProto.FLOAT for name in node.output))
            self.assertTrue(all(value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16 for value in graph.graph.input))
            self.assertTrue(all(value.type.tensor_type.elem_type == onnx.TensorProto.BFLOAT16 for value in graph.graph.output))
            parameters = {value.name: value for value in graph.graph.initializer if value.name in ("weight", "bias")}
            self.assertEqual(set(parameters), {"weight", "bias"})
            self.assertTrue(all(value.data_type == onnx.TensorProto.BFLOAT16 for value in parameters.values()))
            self.assertEqual(compiler.records[0]["translation_source"], compiler.translation_source)
            self.assertEqual(compiler.records[0]["translated_aten_operations"], {"aten.scalar_tensor.default": 0, "aten.linear.default": 1, "aten.scaled_dot_product_attention.default": 1, "aten.layer_norm.default": 0, "aten.silu.default": 0})
            self.assertEqual(self.verifier.parameter_digest(module, torch), original_weights)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_attention_masks_causal_and_scale(self):
        import math
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class MaskedAttention(torch.nn.Module):
            def __init__(self, mask=None, causal=False, scale=None):
                super().__init__()
                self.register_buffer("mask", mask)
                self.causal, self.scale = causal, scale

            def forward(self, q, k, v):
                return torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=self.mask, is_causal=self.causal, scale=self.scale)

        mask = torch.zeros((3, 5), dtype=torch.bool)
        mask[1:, -1] = True
        additive = torch.where(mask, 0., -torch.inf)
        q, k = torch.zeros((1, 1, 3, 1), dtype=torch.bfloat16), torch.zeros((1, 1, 5, 1), dtype=torch.bfloat16)
        v = torch.arange(0, 10, 2, dtype=torch.bfloat16).reshape(1, 1, 5, 1)
        masked_output = torch.tensor([0., 8., 8.], dtype=torch.bfloat16).reshape(1, 1, 3, 1)
        causal_output = torch.tensor([0., 1., 2.], dtype=torch.bfloat16).reshape(1, 1, 3, 1)
        scaled_values = (torch.ones((1, 1, 3, 1), dtype=torch.bfloat16), torch.tensor([-1., 1.], dtype=torch.bfloat16).reshape(1, 1, 2, 1), torch.tensor([0., 4.], dtype=torch.bfloat16).reshape(1, 1, 2, 1))
        cases = [("bool", MaskedAttention(mask), (q, k, v), masked_output), ("bf16-mask", MaskedAttention(additive.bfloat16()), (q, k, v), masked_output), ("fp32-mask", MaskedAttention(additive), (q, k, v), masked_output), ("rectangular-causal", MaskedAttention(causal=True), (q, k, v), causal_output), ("explicit-scale", MaskedAttention(scale=math.log(3) / 2), scaled_values, torch.full_like(q, 3))]
        for name, module, values, known_output in cases:
            with self.subTest(case=name), tempfile.TemporaryDirectory(prefix="tvm-session-bf16-mask-") as temp:
                with torch.no_grad():
                    reference = module(*values)
                self.assertTrue(torch.equal(reference, known_output))
                inputs = tuple(self.verifier.array(value, np, torch) for value in values)
                compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
                actual = self.verifier.array(compiler(SimpleNamespace(name=name, module=module), inputs, 1)(inputs), np, torch)
                self.verifier.compare(actual, self.verifier.array(reference, np, torch), np, 0, 0)
        malformed = MaskedAttention(torch.full((3, 5), torch.nan))
        with tempfile.TemporaryDirectory(prefix="tvm-session-bf16-nan-mask-") as temp:
            inputs = tuple(self.verifier.array(value, np, torch) for value in (q, k, v))
            compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
            result = compiler(SimpleNamespace(name="nan-mask", module=malformed), inputs, 1)(inputs)
            actual = self.verifier.array(result, np, torch, require_finite=False)
            self.assertTrue(np.isnan(actual).all())
            with self.assertRaisesRegex(ValueError, "finite"):
                self.verifier.array(result, np, torch)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_importer_literal_and_cast_like_refusals(self):
        import tvm
        from tvm.relax.frontend.onnx.onnx_frontend import CastLike, Constant
        for attributes in ({}, {"value_int": 1, "value_float": 1.}, {"value_string": b"unsupported"}):
            with self.subTest(attributes=attributes), self.assertRaisesRegex(ValueError, "exactly one supported"):
                Constant._impl_v13(None, [], attributes, None)
        target = tvm.relax.Var("target", tvm.relax.TensorStructInfo([1], dtype=""))
        with self.assertRaisesRegex(ValueError, "known target dtype"):
            CastLike._impl_v15(None, [None, target], {}, None)
        with self.assertRaisesRegex(ValueError, "attributes"):
            CastLike._impl_v15(None, [None, target], {"round_mode": b"unknown"}, None)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_layer_norm_preserves_original_affine_and_output_boundary(self):
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class Norm(torch.nn.Module):
            def __init__(self, shape, weight, bias):
                super().__init__()
                self.shape = shape
                self.register_buffer("weight", weight)
                self.register_buffer("bias", bias)

            def forward(self, value):
                return torch.nn.functional.layer_norm(value, self.shape, self.weight, self.bias, 1e-6)

        data = torch.tensor([[1., 2., 4.], [-2., 1., 3.]], dtype=torch.bfloat16)
        weight = torch.tensor([0.5, 1.25, -0.75], dtype=torch.bfloat16)
        bias = torch.tensor([0.125, -0.25, 0.375], dtype=torch.bfloat16)
        with torch.no_grad():
            original = torch.nn.functional.layer_norm(data, (3,), weight, bias, 1e-6)
            split = torch.nn.functional.layer_norm(data.float(), (3,), eps=1e-6).bfloat16() * weight + bias
        self.assertFalse(torch.equal(original, split))
        cases = [("affine", (3,), data, weight, bias), ("no-bias", (3,), data, weight, None), ("no-weight", (3,), data, None, bias), ("no-affine", (3,), data, None, None), ("two-axes", (2, 3), data.unsqueeze(0), weight.expand(2, 3).contiguous(), bias.expand(2, 3).contiguous())]
        for name, shape, value, scale, shift in cases:
            module = Norm(shape, scale, shift).eval()
            parameter_digest = self.verifier.parameter_digest(module, torch)
            with torch.no_grad():
                expected = self.verifier.array(module(value), np, torch)
            inputs = (self.verifier.array(value, np, torch),)
            input_identity = self.verifier.tensor_identity(inputs[0])
            first = None
            for mode in ("baseline", "optimized"):
                with self.subTest(case=name, mode=mode), tempfile.TemporaryDirectory(prefix="tvm-session-bf16-norm-") as temp:
                    compiler = self.verifier.StageCompiler(Path(temp), mode, np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
                    run = compiler(SimpleNamespace(name=name, module=module), inputs, 1)
                    for repeat in range(2):
                        actual = self.verifier.array(run(inputs), np, torch)
                        self.verifier.compare(actual, expected, np, 1e-4, 1e-4)
                        self.assertEqual(actual.tobytes(), expected.tobytes())
                        if first is None:
                            first = actual.copy()
                        self.assertEqual(actual.tobytes(), first.tobytes())
                        self.assertEqual(self.verifier.tensor_identity(inputs[0]), input_identity)
                    graph = onnx.shape_inference.infer_shapes(onnx.load(str(Path(temp) / "stage_0/model.onnx")))
                    types = {entry.name: entry.type.tensor_type.elem_type for entry in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
                    self.assertFalse(any(node.op_type == "LayerNormalization" for node in graph.graph.node))
                    self.assertEqual(sum(node.op_type == "ReduceMean" for node in graph.graph.node), 2)
                    for node in graph.graph.node:
                        if node.op_type in ("ReduceMean", "Sub", "Mul", "Add", "Sqrt", "Div"):
                            self.assertTrue(all(types[entry] == onnx.TensorProto.FLOAT for entry in node.output))
                    self.assertEqual(graph.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.BFLOAT16)
                    self.assertEqual(graph.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.BFLOAT16)
                    self.assertTrue(all(entry.data_type == onnx.TensorProto.BFLOAT16 for entry in graph.graph.initializer if entry.name in ("weight", "bias")))
                    self.assertEqual(compiler.records[0]["translated_aten_operations"]["aten.layer_norm.default"], 1)
                    self.assertEqual(self.verifier.parameter_digest(module, torch), parameter_digest)
        helper = self.verifier.load_module("fused_bf16_dynamic_norm_helper", repo_root() / "examples/gemmini/comparisons/tvm/bf16_export.py")
        dynamic = torch.onnx.export(Norm((3,), weight, bias).eval(), (data,), opset_version=22, dynamo=True, optimize=False, custom_translation_table=helper.translations(torch), dynamic_shapes={"value": {0: torch.export.Dim("rows", min=1, max=8)}})
        dimensions = dynamic.model_proto.graph.input[0].type.tensor_type.shape.dim
        self.assertTrue(dimensions[0].dim_param)
        self.assertEqual(dimensions[1].dim_value, 3)
        self.assertEqual(dynamic.model_proto.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.BFLOAT16)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_export_keeps_fp32_builtin_semantics(self):
        import onnx
        import tvm
        torch, np = self.torch, self.np

        class FloatOperators(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(8, 8)
                self.norm = torch.nn.LayerNorm(8)

            def forward(self, q, k, v):
                return self.linear(q), torch.nn.functional.scaled_dot_product_attention(q, k, v), self.norm(q)

        with torch.random.fork_rng():
            torch.manual_seed(151)
            module = FloatOperators().eval()
            values = tuple(torch.randn((1, 2, 17, 8)) for _ in range(3))
        with torch.no_grad():
            expected = module(*values)
        inputs = tuple(self.verifier.array(value, np, torch) for value in values)
        with tempfile.TemporaryDirectory(prefix="tvm-session-fused-fp32-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "baseline", np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True)
            actual = self.verifier.outputs(compiler(SimpleNamespace(name="fp32", module=module), inputs, 3)(inputs), np, torch)
            for value, reference in zip(actual, expected):
                self.verifier.compare(value, self.verifier.array(reference, np, torch), np, 1e-4, 1e-4)
            graph = onnx.load(str(Path(temp) / "stage_0/model.onnx"))
            default = torch.onnx.export(module, values, opset_version=22, dynamo=True, optimize=False, input_names=["input_0", "input_1", "input_2"], output_names=["output_0", "output_1", "output_2"])
            self.assertEqual([node.op_type for node in graph.graph.node], [node.op_type for node in default.model_proto.graph.node])
            self.assertEqual(sum(node.op_type == "LayerNormalization" for node in graph.graph.node), 1)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_fused_bf16_export_refuses_unverified_contracts(self):
        from onnxscript import ir
        helper = self.verifier.load_module("fused_bf16_test_helper", repo_root() / "examples/gemmini/comparisons/tvm/bf16_export.py")
        translations = helper.translations(self.torch)
        bf16 = SimpleNamespace(dtype=ir.DataType.BFLOAT16, shape=(1, 2, 17, 8))
        fp32 = SimpleNamespace(dtype=ir.DataType.FLOAT, shape=(1, 2, 17, 8))
        linear = translations[self.torch.ops.aten.linear.default]
        attention = translations[self.torch.ops.aten.scaled_dot_product_attention.default]
        layer_norm = translations[self.torch.ops.aten.layer_norm.default]
        with self.assertRaisesRegex(ValueError, "matching operand"):
            linear(bf16, fp32)
        for options in ({"dropout_p": 0.1}, {"enable_gqa": True}, {"is_causal": True, "attn_mask": bf16}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                attention(bf16, bf16, bf16, **options)
        with self.assertRaisesRegex(ValueError, "matching Q/K/V"):
            attention(bf16, fp32, bf16)
        self.assertNotIn(self.torch.ops.aten.native_layer_norm.default, translations)
        with self.assertRaisesRegex(ValueError, "BF16 dispatcher"):
            layer_norm(fp32, [8])
        for shape in ([], [0], [-1], [8.0], [1, 1, 1, 1, 8]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "positive static"):
                layer_norm(bf16, shape)
        for value, shape in ((bf16, [7]), (SimpleNamespace(dtype=ir.DataType.BFLOAT16, shape=(1, None)), [8]), (SimpleNamespace(dtype=ir.DataType.BFLOAT16, shape=(1, 8)), [2, 8])):
            with self.subTest(input_shape=value.shape, normalized_shape=shape), self.assertRaisesRegex(ValueError, "static trailing"):
                layer_norm(value, shape)
        for epsilon in (-1., float("nan"), float("inf")):
            with self.subTest(epsilon=epsilon), self.assertRaisesRegex(ValueError, "epsilon"):
                layer_norm(bf16, [8], eps=epsilon)
        for affine in ({"weight": fp32}, {"bias": fp32}):
            with self.subTest(affine=affine), self.assertRaisesRegex(ValueError, "original BF16"):
                layer_norm(bf16, [8], **affine)
        for name, shape in (("weight", (7,)), ("bias", (1, 8)), ("weight", None)):
            with self.subTest(affine=name, shape=shape), self.assertRaisesRegex(ValueError, "affine shapes"):
                layer_norm(bf16, [8], **{name: SimpleNamespace(dtype=ir.DataType.BFLOAT16, shape=shape)})

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX"), "Requires explicit opt-in and onnxscript dependency")
    def test_modern_export_preserves_source_cache_across_repeats(self):
        import onnx
        import tvm
        torch = self.torch

        class CachedPrefix(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.cache = SimpleNamespace(length=torch.zeros((), dtype=torch.int64))

            def forward(self, value):
                self.cache.length = torch.zeros_like(self.cache.length)
                self.cache.length += value.numel()
                return value + self.cache.length.to(value.dtype)

        session = self.session()
        prefix = CachedPrefix().eval()
        session = replace(session, programs=(replace(session.programs[0], module=prefix), session.programs[1]))
        with tempfile.TemporaryDirectory(prefix="tvm-session-modern-cache-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, torch, onnx, tvm, exporter="dynamo", opset=22)
            result = self.verifier.verify_session(session, compiler, self.np, torch)
            self.assertTrue(result["reset_verified"])
            self.assertEqual(len(result["checks"]), 8)
            self.assertEqual(type(prefix.cache.length), torch.Tensor)
            self.assertEqual(prefix.cache.length.numpy().item(), 2)
            self.assertTrue(all(record["opset"] == 22 and record["exporter"] == "dynamo" for record in compiler.records))

    def growing_session(self):
        torch, protocol = self.torch, self.protocol

        class Grow(torch.nn.Module):
            def forward(self, token, cache):
                updated = torch.cat((cache, token))
                return updated.sum().reshape(1), updated

        tokens = torch.tensor([[1.], [-2.], [3.]], dtype=torch.float32)
        spec = {"steps": 3, "streams": [{"name": "token", "input_index": 0, "values": tokens}],
                "states": [{"name": "cache", "input_index": 1, "output_index": 1}], "quality": {"output_index": 0}}
        return protocol.make_external_runtime_session(
            version=1, programs=(protocol.ExternalRuntimeProgram("grow", Grow().eval(), (tokens[0], torch.tensor([10.])), 3, spec),),
            metadata={"kind": "diagnostic_growing_state", "paper_ready": False})

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_changing_state_extents_have_distinct_static_compilations(self):
        import onnx
        import tvm
        with tempfile.TemporaryDirectory(prefix="tvm-session-grow-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            result = self.verifier.verify_session(self.growing_session(), compiler, self.np, self.torch)
            self.assertEqual(result["compilation_signatures"], 3)
            self.assertEqual([record["inputs"][1]["shape"] for record in compiler.records], [[1], [2], [3]])
            self.assertEqual(len(result["checks"]), 6)
            self.assertTrue(result["reset_verified"])
            for row, wanted in zip(result["checks"][:3], (11., 9., 12.)):
                expected = self.np.array([wanted], dtype="float32")
                self.assertEqual(row["outputs"][0]["actual"], self.verifier.tensor_identity(expected))

    def test_streams_routes_reset_and_repeated_outputs(self):
        result = self.verifier.verify_session(self.session(), self.reference_compiler, self.np, self.torch)
        self.assertTrue(result["reset_verified"])
        self.assertEqual(result["compilation_signatures"], 2)
        self.assertEqual(len(result["checks"]), 8)
        rows = [row for row in result["checks"] if row["program"] == "decode" and row["repeat"] == 0]
        wanted = ([3., -2.], [-4., 8.], [3., 6.])
        for row, values in zip(rows, wanted):
            expected = self.np.array(values, dtype="float32")
            self.assertEqual(row["outputs"][0]["actual"], self.verifier.tensor_identity(expected))
            self.assertEqual(len(row["routed_states"]), 1)
        self.assertFalse(result["session"]["metadata"]["paper_ready"])

    def test_wrong_output_and_wrong_route_fail(self):
        def wrong(program, inputs, output_count):
            run = self.reference_compiler(program, inputs, output_count)
            return lambda values: tuple(value + 1 for value in run(values)) if program.name == "decode" else run(values)
        with self.assertRaisesRegex(AssertionError, "numerical comparison"):
            self.verifier.verify_session(self.session(), wrong, self.np, self.torch)
        session = self.session()
        route = session.routes[-1]
        corrupted = replace(route, source=replace(route.source, output_index=99))
        session = replace(session, routes=(*session.routes[:-1], corrupted))
        with self.assertRaisesRegex(ValueError, "absent output"):
            self.verifier.verify_session(session, self.reference_compiler, self.np, self.torch)

    def test_biased_stage_has_structured_failure_and_failed_receipt(self):
        from unittest.mock import patch
        def biased(program, inputs, output_count):
            run = self.reference_compiler(program, inputs, output_count)
            return lambda values: tuple(value + 0.5 for value in run(values)) if program.name == "decode" else run(values)
        with self.assertRaisesRegex(AssertionError, "numerical comparison") as caught:
            self.verifier.verify_session(self.session(), biased, self.np, self.torch)
        failure = caught.exception.diagnostics
        self.assertEqual({key: failure[key] for key in ("stage", "program", "repeat", "observation", "tensor", "output_index")},
                         {"stage": "per_observation", "program": "decode", "repeat": 0, "observation": 0, "tensor": "output", "output_index": 0})
        self.assertEqual(failure["max_absolute_error"], 0.5)
        self.assertEqual(failure["mismatch_count"], 2)
        self.assertEqual((failure["rtol"], failure["atol"]), (1e-4, 1e-4))
        self.assertEqual(failure["actual"]["shape"], [2])
        self.assertEqual(failure["actual"]["dtype"], "float32")
        self.assertIn("sha256", failure["actual"]["identity"])
        json.dumps(failure, allow_nan=False)
        with tempfile.TemporaryDirectory(prefix="tvm-session-failure-") as temp:
            folder = Path(temp) / "receipt"
            argv = ["verify_session.py", "--output-dir", str(folder)]
            for name in ("model2mlir-root", "tvm-source", "tvm-build", "factory"):
                argv.extend(("--" + name, str(Path(temp) / name)))
            identify = self.verifier.file_identity
            calls = 0
            def fail_verification(path, check):
                nonlocal calls
                calls += 1
                if calls == 1:
                    raise caught.exception
                return identify(path, check)
            with patch.object(sys, "argv", argv), patch.object(self.verifier, "file_identity", side_effect=fail_verification):
                self.assertEqual(self.verifier.main(), 1)
            receipt = json.loads((folder / "results.json").read_text())
            self.assertEqual(receipt["status"], "failed")
            self.assertFalse(receipt["host_session_verified"])
            self.assertEqual(receipt["failure_diagnostics"], failure)
            self.assertIn("SessionComparisonError", receipt["traceback"])
            retention = receipt["failure_diagnostics"]["fixture_retention"]
            self.assertEqual(retention["status"], "retained")
            self.assertEqual(retention["artifact"], identify(folder / "failed_comparison.npz", self.verifier.safe_path))
            with self.np.load(retention["artifact"]["path"], allow_pickle=False) as fixture:
                for name in ("actual", "reference"):
                    self.assertEqual(self.verifier.tensor_identity(fixture[name]), failure[name]["identity"])
                self.np.testing.assert_array_equal(fixture["actual"] - fixture["reference"], self.np.full(2, 0.5, dtype="float32"))
            self.assertEqual({path.name for path in folder.iterdir()}, {"results.json", "failed_comparison.npz"})

    def test_failure_fixture_budget_snapshot_and_failed_write_cleanup(self):
        from unittest.mock import patch
        np = self.np
        actual, reference = np.array([1, -2], dtype="int64"), np.array([2, -2], dtype="int64")
        with patch.object(self.verifier, "FAILED_FIXTURE_LIMIT_BYTES", 1), self.assertRaises(AssertionError) as caught:
            self.verifier.compare(actual, reference, np, 0, 0)
        self.assertIsNone(caught.exception.tensors)
        self.assertEqual(caught.exception.diagnostics["fixture_retention"]["status"], "skipped")
        with self.assertRaises(AssertionError) as caught:
            self.verifier.compare(actual, reference, np, 0, 0)
        error = caught.exception
        actual[:] = 0
        self.assertEqual(self.verifier.tensor_identity(error.tensors[0]), error.diagnostics["actual"]["identity"])
        def broken_save(stream, **arrays):
            stream.write(b"partial fixture")
            raise OSError("fixture write failed")
        with tempfile.TemporaryDirectory(prefix="tvm-session-failure-write-") as temp:
            folder = Path(temp)
            with patch.object(np, "savez", side_effect=broken_save):
                self.verifier.write_failure_fixture(error, folder)
            self.assertEqual(error.diagnostics["fixture_retention"]["status"], "failed")
            self.assertEqual(error.diagnostics["fixture_retention"]["error"], "fixture write failed")
            self.assertEqual(list(folder.iterdir()), [])
            self.verifier.write_failure_fixture(error, folder)
            with np.load(folder / "failed_comparison.npz", allow_pickle=False) as fixture:
                self.assertEqual(fixture["actual"].dtype, np.dtype("int64"))
                self.assertEqual(self.verifier.tensor_identity(fixture["actual"]), error.diagnostics["actual"]["identity"])
            before = (folder / "failed_comparison.npz").read_bytes()
            self.verifier.write_failure_fixture(error, folder)
            self.assertEqual(error.diagnostics["fixture_retention"]["status"], "failed")
            self.assertEqual((folder / "failed_comparison.npz").read_bytes(), before)

    def test_invalid_tensor_diagnostics_do_not_claim_numerical_metrics(self):
        np = self.np
        reference = np.ones(2, dtype="float32")
        for actual in (np.ones(3, dtype="float32"), np.ones(2, dtype="float64"), np.array([np.nan, np.inf], dtype="float32")):
            with self.subTest(dtype=actual.dtype, shape=actual.shape), self.assertRaisesRegex(AssertionError, "shape/dtype/finite") as caught:
                self.verifier.compare(actual, reference, np, 1e-4, 1e-4)
            failure = caught.exception.diagnostics
            self.assertIsNone(failure["max_absolute_error"])
            self.assertIsNone(failure["mismatch_count"])
            self.assertIsNone(caught.exception.tensors)
            json.dumps(failure, allow_nan=False)
        self.assertIsNone(failure["actual"]["identity"])
        def nonfinite(program, inputs, output_count):
            return lambda values: self.torch.full((2,), float("nan"))
        with self.assertRaisesRegex(AssertionError, "finite") as caught:
            self.verifier.verify_session(self.session(), nonfinite, np, self.torch)
        self.assertEqual(caught.exception.diagnostics["program"], "prefill")
        self.assertEqual(caught.exception.diagnostics["output_index"], 0)
        self.assertFalse(caught.exception.diagnostics["actual"]["finite"])
        json.dumps(caught.exception.diagnostics, allow_nan=False)

    def test_difference_metrics_preserve_large_integers_and_handle_overflow(self):
        np = self.np
        for actual, expected, delta in ((np.array([2 ** 63 - 1], dtype="int64"), np.array([-2 ** 63], dtype="int64"), 2 ** 64 - 1),
                                        (np.array([np.finfo("float64").max]), np.array([-np.finfo("float64").max]), None)):
            with self.assertRaisesRegex(AssertionError, "numerical comparison") as caught:
                self.verifier.compare(actual, expected, np, 0, 0)
            failure = caught.exception.diagnostics
            self.assertEqual(failure["mismatch_count"], 1)
            self.assertEqual(failure["max_absolute_error"], delta)
            json.dumps(failure, allow_nan=False)

    def test_unknown_cadence_route_update_and_dtype_are_rejected(self):
        session = self.session()
        for invocation in (replace(session.execution_schedule[0], cadence="unknown"),):
            malformed = replace(session, execution_schedule=(invocation, session.execution_schedule[1]))
            with self.assertRaisesRegex(ValueError, "cadence"):
                self.verifier.verify_session(malformed, self.reference_compiler, self.np, self.torch)
        malformed = replace(session, routes=(replace(session.routes[0], update="after_session"), *session.routes[1:]))
        with self.assertRaisesRegex(ValueError, "state route"):
            self.verifier.verify_session(malformed, self.reference_compiler, self.np, self.torch)
        malformed = replace(session, execution_schedule=(replace(session.execution_schedule[0], cadence="per_observation", repeats=3),
                            replace(session.execution_schedule[1], cadence="once_after_observations")))
        with self.assertRaisesRegex(ValueError, "per-observation stage"):
            self.verifier.verify_session(malformed, self.reference_compiler, self.np, self.torch)
        with self.assertRaisesRegex(AssertionError, "dtype"):
            self.verifier.compare(self.np.ones(2, dtype="int32"), self.np.ones(2, dtype="int64"), self.np, 1e-4, 1e-4)
        with self.assertRaisesRegex(ValueError, "Unsupported session dtype"):
            self.verifier.array(self.torch.ones(2, dtype=self.torch.complex64), self.np, self.torch)

    @unittest.skipUnless(importlib.util.find_spec("ml_dtypes"), "Original BF16 verification needs ml_dtypes")
    def test_bfloat16_torch_numpy_tvm_round_trip_preserves_bits(self):
        words = self.np.array([0, 0x8000, 0x3F81, 0xBFC1, 0x0080, 0x7F7F], dtype="uint16")
        tensor = self.torch.from_numpy(words.copy()).view(self.torch.bfloat16)
        value = self.verifier.array(tensor, self.np, self.torch)
        self.assertEqual(value.dtype.name, "bfloat16")
        self.np.testing.assert_array_equal(value.view(self.np.uint16), words)
        restored = self.verifier.torch_array(value, self.np, self.torch)
        self.assertTrue(self.torch.equal(restored.view(self.torch.uint16), tensor.view(self.torch.uint16)))
        self.assertEqual(self.verifier.tensor_identity(value)["bytes"], 12)
        with self.assertRaises(AssertionError) as caught:
            self.verifier.compare(value, value + self.np.array(1, dtype=value.dtype), self.np, 0, 0)
        with tempfile.TemporaryDirectory(prefix="tvm-session-failure-bf16-") as temp:
            self.verifier.write_failure_fixture(caught.exception, Path(temp))
            retention = caught.exception.diagnostics["fixture_retention"]
            self.assertEqual(retention["encodings"]["actual"], "raw_uint16_bfloat16")
            with self.np.load(retention["artifact"]["path"], allow_pickle=False) as fixture:
                self.np.testing.assert_array_equal(fixture["actual"], words)
        if os.environ.get("TVM_LIBRARY_PATH"):
            import tvm
            candidate = self.verifier.outputs(tvm.nd.array(value), self.np, self.torch)[0]
            self.np.testing.assert_array_equal(candidate.view(self.np.uint16), words)
        with self.assertRaisesRegex(ValueError, "finite"):
            self.verifier.array(self.torch.tensor([float("nan")], dtype=self.torch.bfloat16), self.np, self.torch)

    @unittest.skipUnless(importlib.util.find_spec("ml_dtypes") and os.environ.get("TVM_LIBRARY_PATH"), "Needs BF16 dependency and TVM")
    def test_original_bfloat16_session_compiles_and_routes_without_casts(self):
        import onnx
        import tvm
        session = self.session()
        bf16 = self.torch.bfloat16
        session = replace(session,
            programs=tuple(replace(program, inputs=tuple(value.to(bf16) for value in program.inputs)) for program in session.programs),
            input_bindings=tuple(replace(binding, initial=binding.initial.to(bf16)) for binding in session.input_bindings),
            streams=tuple(replace(stream, values=stream.values.to(bf16)) for stream in session.streams))
        with tempfile.TemporaryDirectory(prefix="tvm-session-bfloat16-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            result = self.verifier.verify_session(session, compiler, self.np, self.torch, rtol=0, atol=0)
        self.assertEqual(len(result["checks"]), 8)
        self.assertTrue(result["reset_verified"])
        self.assertTrue(all(row["outputs"][0]["actual"]["dtype"] == "bfloat16" for row in result["checks"]))

    def test_network_and_signature_budget_are_enforced(self):
        with self.verifier.offline(), self.assertRaisesRegex(RuntimeError, "network"):
            socket.create_connection(("127.0.0.1", 9))
        with self.assertRaisesRegex(ValueError, "signature budget"):
            self.verifier.verify_session(self.session(), self.reference_compiler, self.np, self.torch, max_signatures=1)

    def external_graph(self, folder):
        import onnx
        value = self.np.array([[1., -2.], [3., 4.]], dtype="float32")
        weight = onnx.numpy_helper.from_array(value, name="weight")
        graph = onnx.helper.make_graph([onnx.helper.make_node("Add", ["input", "weight"], ["output"])], "external",
            [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [2, 2])],
            [onnx.helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [2, 2])], [weight])
        model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 17)])
        path = folder / "model.onnx"
        onnx.save_model(model, str(path), save_as_external_data=True, all_tensors_to_one_file=True, location="weights.bin", size_threshold=0)
        return path, value

    def test_external_weights_are_loaded_and_content_bound(self):
        import onnx
        with tempfile.TemporaryDirectory(prefix="tvm-session-external-") as temp:
            path, value = self.external_graph(Path(temp))
            graph, identities = self.verifier.load_onnx_graph(path, onnx)
            self.assertEqual(len(identities), 2)
            self.assertEqual({Path(item["path"]).name for item in identities}, {"model.onnx", "weights.bin"})
            self.np.testing.assert_array_equal(onnx.numpy_helper.to_array(graph.graph.initializer[0]), value)
            self.assertEqual(graph.graph.initializer[0].data_location, onnx.TensorProto.DEFAULT)

    def test_external_paths_metadata_and_ranges_are_checked_before_loading(self):
        import onnx
        with tempfile.TemporaryDirectory(prefix="tvm-session-external-") as temp:
            folder = Path(temp)
            path, _ = self.external_graph(folder)
            original = onnx.load(str(path), load_external_data=False)
            for location in ("../outside.bin", "/outside.bin"):
                malformed = onnx.ModelProto()
                malformed.CopyFrom(original)
                malformed.graph.initializer[0].external_data[0].value = location
                path.write_bytes(malformed.SerializeToString())
                with self.assertRaisesRegex(ValueError, "stage directory"):
                    self.verifier.load_onnx_graph(path, onnx)
            for key, value, expected in (("offset", "-1", "nonnegative"), ("length", "17", "range exceeds"), ("basepath", "elsewhere", "metadata")):
                malformed = onnx.ModelProto()
                malformed.CopyFrom(original)
                entries = malformed.graph.initializer[0].external_data
                matches = [entry for entry in entries if entry.key == key]
                entry = matches[0] if matches else entries.add()
                entry.key, entry.value = key, value
                path.write_bytes(malformed.SerializeToString())
                with self.assertRaisesRegex(ValueError, expected):
                    self.verifier.load_onnx_graph(path, onnx)
            malformed = onnx.ModelProto()
            malformed.CopyFrom(original)
            entry = malformed.graph.initializer[0].external_data.add()
            entry.key, entry.value = "location", "weights.bin"
            path.write_bytes(malformed.SerializeToString())
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.verifier.load_onnx_graph(path, onnx)

    def test_external_weight_changes_during_check_are_rejected(self):
        import onnx
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix="tvm-session-external-") as temp:
            folder = Path(temp)
            path, _ = self.external_graph(folder)
            check = onnx.checker.check_model
            def mutate(*args, **kwargs):
                check(*args, **kwargs)
                weights = folder / "weights.bin"
                weights.write_bytes(bytes([weights.read_bytes()[0] ^ 1]) + weights.read_bytes()[1:])
            with patch.object(onnx.checker, "check_model", mutate), self.assertRaisesRegex(AssertionError, "changed"):
                self.verifier.load_onnx_graph(path, onnx)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_checked_session_exports_static_stages_with_shared_constants(self):
        import onnx
        import tvm
        with tempfile.TemporaryDirectory(prefix="tvm-session-package-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            trace = {}
            result = self.verifier.verify_session(self.session(), compiler, self.np, self.torch, adapter_trace=trace)
            exported = compiler.export_stages(Path(temp) / "baremetal", memory_limit_bytes=1 << 20, target="llvm")
            self.assertTrue(result["reset_verified"])
            self.assertFalse(exported["session_device_execution"])
            self.assertEqual(exported["state_routing"], "caller_owned")
            self.assertEqual({stage["program"] for stage in exported["stages"].values()}, {"prefill", "decode"})
            manifest = json.loads(Path(exported["artifact"]["path"]).read_text())
            self.assertEqual(len(manifest["stages"]), 2)
            self.assertLessEqual(manifest["constants_bytes"], manifest["duplicated_stage_constants_bytes"])
            self.assertTrue(all(stage["entrypoint"].endswith("_model_run") for stage in manifest["stages"].values()))
            exported = compiler.export_stages(Path(temp) / "automatic", memory_limit_bytes=1 << 20, target="llvm", adapter_trace=trace,
                                              reference_sources=[self.verifier.file_identity(__file__, self.verifier.safe_path)])
            adapter = exported["automatic_session_adapter"]
            self.assertEqual(adapter["invocations"], 4)
            self.assertEqual(adapter["retained_invocation_outputs"], 7)
            self.assertTrue(adapter["static_route_generated"])
            self.assertFalse(adapter["device_admission_verified"])
            self.assertFalse(adapter["production_memory"]["production_fit_established"])
            with self.np.load(adapter["fixture"]["path"], allow_pickle=False) as fixture:
                self.np.testing.assert_array_equal(fixture["output_5"], self.np.array([[3., 6.]], dtype="float32"))

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_real_onnx_relax_pipeline_verifies_every_stage(self):
        import onnx
        import tvm
        with tempfile.TemporaryDirectory(prefix="tvm-session-") as temp:
            compiler = self.verifier.StageCompiler(Path(temp), "optimized", self.np, self.torch, onnx, tvm)
            result = self.verifier.verify_session(self.session(), compiler, self.np, self.torch)
            self.assertTrue(result["reset_verified"])
            self.assertEqual(len(compiler.records), 2)
            self.assertEqual({record["program"] for record in compiler.records}, {"prefill", "decode"})
            self.assertEqual(len(result["checks"]), 8)
            for record in compiler.records:
                self.assertEqual(record["pipelines"], ["zero", "default_build"])
                self.assertIn("tir", record["vm"]["inventory"]["function_counts"])


if __name__ == "__main__":
    unittest.main()
