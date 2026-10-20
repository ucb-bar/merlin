"""SmolVLA artifacts, session routing and native-backend diagnostics."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from merlin.common.paths import repo_root


SOURCE = repo_root() / "examples/gemmini/comparisons/tvm/smolvla_session.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("smolvla_session_tested", SOURCE)
FACTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FACTORY)
PREP_SPEC = importlib.util.spec_from_file_location("smolvla_observation_tested", SOURCE.with_name("prepare_smolvla_observation.py"))
PREPARATION = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREPARATION)
NUMERIC_SPEC = importlib.util.spec_from_file_location("smolvla_bf16_diagnostic_tested", SOURCE.with_name("bf16_export.py"))
NUMERICS = importlib.util.module_from_spec(NUMERIC_SPEC)
NUMERIC_SPEC.loader.exec_module(NUMERICS)
DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location("smolvla_backends_tested", SOURCE.with_name("diagnose_smolvla_backends.py"))
DIAGNOSTIC = importlib.util.module_from_spec(DIAGNOSTIC_SPEC)
DIAGNOSTIC_SPEC.loader.exec_module(DIAGNOSTIC)


class SmolVLAArtifactTests(unittest.TestCase):
    def config(self):
        return {"type": "smolvla", **FACTORY.GEOMETRY,
                "input_features": {"observation.state": {"type": "STATE", "shape": [6]},
                    **{f"observation.images.camera{index}": {"type": "VISUAL", "shape": [3, 256, 256]}
                       for index in range(1, 4)}},
                "output_features": {"action": {"type": "ACTION", "shape": [6]}}}

    def test_absent_checkpoint_fails_before_framework_loading(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            FACTORY.checkpoint_artifacts(directory)

    def test_nonofficial_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "config.json").write_text(json.dumps(self.config()))
            (root / "model.safetensors").write_text("fixture only; never a model")
            with self.assertRaisesRegex(ValueError, "pinned official full"):
                FACTORY.checkpoint_artifacts(root)

    def test_geometry_truncation_and_precision_changes_are_rejected(self):
        for change in ({"num_steps": 1}, {"num_vlm_layers": 2}, {"load_vlm_weights": False}):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / "config.json").write_text(json.dumps({**self.config(), **change}))
                (root / "model.safetensors").write_text("fixture only; never a model")
                identity = {"path": str(root / "model.safetensors"), "sha256": FACTORY.WEIGHT_SHA256}
                with mock.patch.object(FACTORY, "file_identity", return_value=identity), self.assertRaisesRegex(ValueError, "complete pinned"):
                    FACTORY.checkpoint_artifacts(root)

    def test_unsupported_precision_and_unattributed_dataset_fail_early(self):
        base = {"checkpoint_dir": "/missing", "backbone_dir": "/missing", "model2mlir_root": "/missing"}
        for change in ({"precision": "float32"}, {"cpu_threads": 32}, {"fixture_kind": "dataset"}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**base, **change)

    def test_restricted_component_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Restricted path component"):
            FACTORY.directory_artifacts("/tmp/" + "vl" + "si" + "-fixture")


class SmolVLATensorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        torch.set_num_threads(1)
        cls.config = SimpleNamespace(resize_imgs_with_padding=(2, 2), tokenizer_max_length=3,
                                    max_state_dim=4, max_action_dim=4, chunk_size=5,
                                    robot_state_feature=SimpleNamespace(shape=[2]))

    def test_synthetic_fixture_is_repeatable_and_pads_only_physical_state(self):
        first = FACTORY.fixture_inputs(self.config, 3, seed=42)
        second = FACTORY.fixture_inputs(self.config, 3, seed=42)
        self.assertEqual(first[0].shape, (1, 3, 3, 2, 2))
        self.assertTrue(all(self.torch.equal(a, b) for a, b in zip(first, second)))
        self.assertEqual(self.torch.count_nonzero(first[4][:, 2:]).item(), 0)

    def test_npz_precision_is_never_coerced(self):
        import numpy as np
        values = FACTORY.fixture_inputs(self.config, 3, seed=42)
        names = ("images", "image_masks", "language_tokens", "language_mask", "state", "noise")
        arrays = {name: value.numpy() for name, value in zip(names, values)}
        arrays["images"] = arrays["images"].astype(np.float16)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.npz"
            np.savez(path, **arrays)
            with self.assertRaisesRegex(ValueError, "dtype"):
                FACTORY.fixture_inputs(self.config, 3, input_npz=path)

    def test_complete_schedule_exact_time_stream_and_physical_decode(self):
        root = os.environ.get("MODEL2MLIR_ROOT")
        if not root:
            self.skipTest("MODEL2MLIR_ROOT selects the actual loader/protocol for this contract test")
        from verify_session import load_module, load_protocol, verify_session
        import numpy as np
        torch = self.torch
        load_protocol(Path(root))
        loader = load_module("smolvla_contract_loader", Path(root) / "workloads/smolvla/loader.py")

        class Backbone(torch.nn.Module):
            def forward(self, **kwargs):
                embeds = kwargs["inputs_embeds"][0]
                cache = {index: {"key_states": embeds, "value_states": embeds}
                         for index in range(16)}
                return (embeds, None), cache

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vlm_with_expert = Backbone()
                self.config = SimpleNamespace(num_vlm_layers=16, num_steps=10, chunk_size=5,
                                              action_feature=SimpleNamespace(shape=[2]))

            def embed_prefix(self, images, masks, tokens, language_mask, state):
                embeds = state[:, None, :] + sum(image.mean() for image in images)
                return embeds, torch.ones(1, 1, dtype=torch.bool), torch.zeros(1, 1)

            def denoise_step(self, prefix_masks, cache, flow, timestep):
                return flow * 0.2 + timestep[:, None, None] + cache[0]["key_states"].mean()

            def sample_actions(self, images, masks, tokens, language_mask, state, noise):
                embeds, _, _ = self.embed_prefix(images, masks, tokens, language_mask, state)
                value = noise
                for index in range(self.config.num_steps):
                    time = torch.tensor(1.0 + index * (-1.0 / self.config.num_steps)).expand(1)
                    value = value - (1.0 / self.config.num_steps) * (
                        value * 0.2 + time[:, None, None] + embeds.mean())
                return value

        model = Model().eval()
        inputs = FACTORY.fixture_inputs(self.config, 3, seed=42)
        provenance = {}
        session = FACTORY.make_policy_session(model, inputs, loader, provenance)
        self.assertTrue(provenance["upstream_sample_actions_equivalence"]["exact"])
        self.assertEqual([program.steps for program in session.programs], [1, 10, 1])
        self.assertEqual(session.streams[0].values.shape, (10, 1))
        self.assertEqual(session.programs[-1].module.action_dim, 2)
        self.assertFalse(any(route.name == "timestep" for route in session.routes))

        def compile_eager(program, _inputs, _outputs):
            def run(values):
                with torch.no_grad():
                    return program.module(*(torch.from_numpy(value) for value in values))
            return run

        result = verify_session(session, compile_eager, np, torch)
        counts = {name: sum(row["program"] == name for row in result["checks"])
                  for name in ("prefix_encode", "flow_denoise", "action_decode")}
        self.assertEqual(counts, {"prefix_encode": 2, "flow_denoise": 20, "action_decode": 2})
        self.assertTrue(result["reset_verified"])


class SmolVLAObservationTests(unittest.TestCase):
    def schema(self):
        return {"codebase_version": "v3.0", "robot_type": "so100", "fps": 30,
                "total_episodes": 50, "total_frames": 19631, "total_tasks": 1,
                "features": {**{name: {"dtype": "video", "shape": [480, 640, 3],
                                       "names": ["height", "width", "channels"]}
                                for name in PREPARATION.CAMERA_MAP},
                             **{name: {"dtype": "float32", "shape": [6],
                                       "names": PREPARATION.STATE_NAMES}
                                for name in ("observation.state", "action")}}}

    def test_official_camera_coordinate_schema_and_frozen_episode_split(self):
        PREPARATION.validate_schema(self.schema())
        counts = {split: sum(PREPARATION.episode_split(index) == split for index in range(50))
                  for split in ("train", "development", "heldout")}
        self.assertEqual(counts, {"train": 30, "development": 10, "heldout": 10})
        for index in (-1, 50, True):
            with self.subTest(index=index), self.assertRaises(ValueError):
                PREPARATION.episode_split(index)
        changed = self.schema()
        changed["features"]["observation.state"]["names"] = list(reversed(PREPARATION.STATE_NAMES))
        with self.assertRaisesRegex(ValueError, "coordinate schema"):
            PREPARATION.validate_schema(changed)
        changed = self.schema()
        changed["features"]["observation.images.fake"] = changed["features"]["observation.images.top"]
        with self.assertRaisesRegex(ValueError, "exactly its recorded"):
            PREPARATION.validate_schema(changed)

    def test_recorded_preparation_uses_original_processor_without_inventing_camera(self):
        import torch
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        config = SimpleNamespace(image_features={name: {} for name in (
            "observation.images.camera1", "observation.images.camera2", "observation.images.camera3")},
            empty_cameras=0, resize_imgs_with_padding=(2, 2), max_state_dim=32,
            max_action_dim=32, chunk_size=50)
        policy = SimpleNamespace(config=config)
        policy.prepare_images = lambda batch: SmolVLAPolicy.prepare_images(policy, batch)
        policy.prepare_state = lambda batch: SmolVLAPolicy.prepare_state(policy, batch)
        observation = {"task": "Pick up the cube", "observation.state": torch.arange(6, dtype=torch.float32),
                       "action": torch.full((6,), 99.0),
                       "observation.images.top": torch.ones(3, 480, 640) * 0.25,
                       "observation.images.wrist": torch.ones(3, 480, 640) * 0.75}
        passed = []

        def processor(batch):
            passed.append(batch)
            return {**{key: value.unsqueeze(0) for key, value in batch.items() if torch.is_tensor(value)},
                    "observation.state": (batch["observation.state"] - 2.0).unsqueeze(0),
                    "observation.language.tokens": torch.zeros(1, 48, dtype=torch.int64),
                    "observation.language.attention_mask": torch.ones(1, 48, dtype=torch.bool)}

        rng = torch.random.get_rng_state().clone()
        first = PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)
        second = PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)
        self.assertEqual(first[0].shape, (1, 2, 3, 2, 2))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(first, second)))
        self.assertTrue(torch.equal(first[4][0, :6], observation["observation.state"] - 2))
        self.assertEqual(first[4][0, 6:].count_nonzero().item(), 0)
        self.assertTrue(first[1].all())
        self.assertNotIn("action", passed[0])
        self.assertNotIn("observation.images.camera3", passed[0])
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertTrue(torch.equal(observation["observation.state"], torch.arange(6, dtype=torch.float32)))
        observation["observation.images.top"] = observation["observation.images.top"].to(torch.float16)
        with self.assertRaisesRegex(ValueError, "FP32 CHW"):
            PREPARATION.prepare_tensors(policy, processor, observation, noise_seed=7, torch=torch)

    def test_observation_manifest_checks_npz_and_camera_route_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = root / "observation.npz"
            fixture.write_bytes(b"identity control only")
            manifest = {
                "version": 1, "fixture_kind": "dataset", "dataset_id": PREPARATION.DATASET_ID,
                "dataset_revision": PREPARATION.DATASET_REVISION, "checkpoint_revision": FACTORY.MODEL_REVISION,
                "precision": "original_lerobot", "schema_verified": True,
                "selected_cameras": list(PREPARATION.CAMERA_MAP.values()), "camera_mapping": PREPARATION.CAMERA_MAP,
                "state_names": PREPARATION.STATE_NAMES, "active_gate_changed": False,
                "model_quality_qualified": False, "episode_index": 0, "split": "train",
                "input_fixture": FACTORY.file_identity(fixture, FACTORY.safe_path),
                "bindings": [{"path": str(root / relative), "sha256": digest}
                             for relative, digest in PREPARATION.PINNED_FILES.items()]}
            path = root / "observation.json"
            path.write_text(json.dumps(manifest))
            identities = {item["path"]: item for item in manifest["bindings"] + [manifest["input_fixture"]]}
            with mock.patch.object(FACTORY, "file_identity", side_effect=lambda name, _: identities[str(name)]):
                self.assertEqual(FACTORY.observation_manifest(fixture, path)["selected_cameras"],
                                 ["observation.images.camera1", "observation.images.camera2"])
                manifest["selected_cameras"].reverse()
                path.write_text(json.dumps(manifest))
                with self.assertRaisesRegex(ValueError, "pinned preparation"):
                    FACTORY.observation_manifest(fixture, path)
            manifest["selected_cameras"].reverse()
            path.write_text(json.dumps(manifest))
            fixture.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "NPZ identity"):
                FACTORY.observation_manifest(fixture, path)


class SmolVLASourceScalarTests(unittest.TestCase):
    def test_source_scalar_defaults_and_fail_closed_sites(self):
        import copy
        import math
        import torch

        sentinel = object()
        from verify_session import StageCompiler
        for options, exporter in (({"source_scalar_policy": "unknown"}, "dynamo"), ({"source_scalar_policy": "torch_cpu_language_scale960_v1"}, "legacy")):
            with self.subTest(options=options, exporter=exporter), self.assertRaisesRegex(ValueError, "source scalar|Source scalar"):
                StageCompiler(Path("unused"), "baseline", None, torch, SimpleNamespace(), None, exporter=exporter, export_options=options)

        class Language(torch.nn.Module):
            def __init__(self, duplicate=False):
                super().__init__()
                self.register_buffer("weight", torch.zeros(49280, 960, dtype=torch.bfloat16))
                self.duplicate = duplicate

            def forward(self, tokens):
                value = torch.nn.functional.embedding(tokens, self.weight)
                if self.duplicate:
                    return value * math.sqrt(960), value * math.sqrt(960)
                return (value * math.sqrt(960)).float()

        class BadProducer(torch.nn.Module):
            def forward(self, value):
                return value * math.sqrt(960)

        policy = "torch_cpu_language_scale960_v1"
        tokens = torch.zeros(1, 48, dtype=torch.int64)
        with mock.patch.object(NUMERICS, "source_scalar_contract", side_effect=lambda *args: {"name": policy}):
            with self.assertRaisesRegex(ValueError, "ExportedProgram"):
                NUMERICS.prepare_source_scalar_export(torch, sentinel, policy)
            ambiguous = torch.export.export(Language(duplicate=True), (tokens,))
            with self.assertRaisesRegex(ValueError, "Ambiguous"):
                NUMERICS.prepare_source_scalar_export(torch, ambiguous, policy)
            bad = torch.export.export(BadProducer(), (torch.ones(1, 48, 960, dtype=torch.bfloat16),))
            with self.assertRaisesRegex(ValueError, "original embedding producer"):
                NUMERICS.prepare_source_scalar_export(torch, bad, policy)
            source = torch.export.export(Language(), (tokens,))
            for dtype, shape in ((torch.float32, (1, 48, 960)), (torch.bfloat16, (1, 47, 960))):
                changed = copy.deepcopy(source)
                node = next(node for node in changed.graph.nodes if node.target == torch.ops.aten.mul.Tensor)
                with node.meta["val"].fake_mode:
                    node.meta["val"] = torch.empty(shape, dtype=dtype)
                with self.subTest(dtype=dtype, shape=shape), self.assertRaisesRegex(ValueError, "result metadata"):
                    NUMERICS.prepare_source_scalar_export(torch, changed, policy)
            missing_mode = copy.deepcopy(source)
            node = next(node for node in missing_mode.graph.nodes if node.target == torch.ops.aten.mul.Tensor)
            node.meta["val"] = torch.empty(1, 48, 960, dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "FakeTensor metadata"):
                NUMERICS.prepare_source_scalar_export(torch, missing_mode, policy)

    def test_source_scalar_no_site_preserves_original_EP_and_explicit_casts(self):
        import math
        import torch

        class Controls(torch.nn.Module):
            def __init__(self, case):
                super().__init__()
                self.case = case

            def forward(self, value, other):
                if self.case == "Tensor_other":
                    return (value * other).float()
                if self.case == "integer":
                    return (value * 2).float()
                if self.case == "nearby_float":
                    return (value * (math.sqrt(960) + 1e-9)).float()
                if self.case == "explicit_cast_Tensor_other":
                    return (value.float().bfloat16() * other).float()
                return (value * math.sqrt(960)).float()

        policy = "torch_cpu_language_scale960_v1"
        cases = [("Tensor_other", torch.bfloat16, (1, 48, 960)), ("FP32", torch.float32, (1, 48, 960)),
                 ("integer", torch.bfloat16, (1, 48, 960)), ("nearby_float", torch.bfloat16, (1, 48, 960)),
                 ("explicit_cast_Tensor_other", torch.bfloat16, (1, 48, 960)), ("wrong_shape", torch.bfloat16, (1, 47, 960))]
        for case, dtype, shape in cases:
            inputs = (torch.ones(shape, dtype=dtype), torch.full(shape, 31, dtype=dtype))
            source = torch.export.export(Controls(case), inputs)
            original_graph = str(source.graph)
            original_signature = str(source.graph_signature)
            with self.subTest(case=case), mock.patch.object(NUMERICS, "source_scalar_contract", side_effect=lambda *args: {"name": policy}):
                prepared, record = NUMERICS.prepare_source_scalar_export(torch, source, policy)
            self.assertIs(prepared, source)
            self.assertEqual(record["site_count"], 0)
            self.assertEqual(record["sites"], [])
            self.assertEqual(str(source.graph), original_graph)
            self.assertEqual(str(source.graph_signature), original_signature)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "requires opt-in pinned TVM/PyTorch CPU build")
    def test_source_scalar_real_public_export_and_strict_LLVM_grid(self):
        import gc
        import hashlib
        import math
        import re
        import numpy as np
        import onnx
        import torch
        if torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0" or torch.backends.cpu.get_cpu_capability() != "AVX512" or torch.get_num_threads() != 1:
            self.skipTest("requires pinned AVX512 PyTorch CPU build and one native thread")
        import tvm
        from tvm import relax
        from tvm.relax.frontend.onnx import from_onnx
        from verify_resnet import prepare_graph

        class Language(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("weight", torch.zeros(49280, 960, dtype=torch.bfloat16))

            def forward(self, tokens):
                return (torch.nn.functional.embedding(tokens, self.weight) * math.sqrt(960)).float()

        # Finite independent grid includes signed zero and tiny BF16 operands.
        grid = np.array([0, 0x8000, 1, 0x8001, 0x0080, 0x8080, 0x3f00, 0xbf00, 0x3f80, 0xbf80, 0x3f81, 0xbf81, 0x4001, 0xc001, 0x3f90, 0xbf90], dtype=np.uint16)
        model = Language().eval()
        row_words = np.resize(grid, 960).copy()
        model.weight[0].copy_(torch.from_numpy(row_words).view(torch.bfloat16))
        tokens = torch.zeros(1, 48, dtype=torch.int64)
        source_value = model(tokens).numpy().copy()
        self.assertEqual(source_value.dtype, np.float32)
        self.assertEqual(source_value.shape, (1, 48, 960))
        gathered = torch.nn.functional.embedding(tokens, model.weight)
        rounded_scalar = torch.tensor(math.sqrt(960), dtype=torch.bfloat16)
        rounded_control = (gathered * rounded_scalar).float().numpy().copy()
        self.assertEqual(int(rounded_scalar.view(torch.uint16)), 0x41f8)
        self.assertTrue(np.any(source_value.view(np.uint32) != rounded_control.view(np.uint32)))
        # ±1.125 distinguish native original-scalar opmath from narrowing to31.
        for word in (0x3f90, 0xbf90):
            column = int(np.flatnonzero(row_words == word)[0])
            self.assertNotEqual(source_value[0, 0, column], rounded_control[0, 0, column])
        tokens_before = tokens.numpy().tobytes()
        weight_before = hashlib.sha256(model.weight.view(torch.uint16).numpy().tobytes()).hexdigest()
        source = torch.export.export(model, (tokens,))
        original_graph = str(source.graph)
        original_signature = str(source.graph_signature)
        prepared, record = NUMERICS.prepare_source_scalar_export(torch, source, "torch_cpu_language_scale960_v1")
        self.assertIsNot(prepared, source)
        self.assertEqual(record["site_count"], 1)
        self.assertEqual(len(record["sites"]), 1)
        self.assertEqual(str(source.graph), original_graph)
        self.assertEqual(str(source.graph_signature), original_signature)
        self.assertEqual(str(prepared.graph_signature), original_signature)
        for exported in (source, prepared):
            self.assertEqual(hashlib.sha256(exported.state_dict["weight"].view(torch.uint16).numpy().tobytes()).hexdigest(), weight_before)
        graph = torch.onnx.export(prepared, dynamo=True, opset_version=18, optimize=False).model_proto
        weight = graph.graph.initializer[0]
        self.assertEqual(weight.data_type, onnx.TensorProto.BFLOAT16)
        self.assertEqual(list(weight.dims), [49280, 960])
        self.assertEqual(hashlib.sha256(weight.raw_data).hexdigest(), weight_before)
        graph_before = hashlib.sha256(graph.SerializeToString()).hexdigest()
        casts = [next(attr.i for attr in node.attribute if attr.name == "to") for node in graph.graph.node if node.op_type == "Cast"]
        self.assertEqual(casts, [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT, onnx.TensorProto.BFLOAT16, onnx.TensorProto.FLOAT])
        scalar = onnx.numpy_helper.to_array(next(attr.t for node in graph.graph.node if node.op_type == "Constant" for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR))
        self.assertEqual(scalar.dtype, np.float32)
        self.assertEqual(scalar.shape, ())
        self.assertEqual(int(scalar.view(np.uint32)), 0x41f7def6)
        types = {value.name: value.type.tensor_type.elem_type for value in list(graph.graph.value_info) + list(graph.graph.input) + list(graph.graph.output)}
        product = next(node for node in graph.graph.node if node.op_type == "Mul")
        self.assertEqual([types[name] for name in product.input], [onnx.TensorProto.FLOAT, onnx.TensorProto.FLOAT])
        self.assertEqual(types[product.output[0]], onnx.TensorProto.FLOAT)
        imported = from_onnx(graph, keep_params_in_input=False)
        for mode in ("baseline", "optimized"):
            _, lowered = prepare_graph(imported, mode, tvm)
            executable = relax.build(lowered, "llvm", pipeline=None)
            pending = [executable.mod]
            modules = []
            seen = set()
            while pending:
                module = pending.pop()
                if module.handle.value in seen:
                    continue
                seen.add(module.handle.value)
                if module.type_key == "llvm":
                    modules.append(module)
                pending.extend(module.imported_modules)
            llvm = "\n".join(module.get_source("ll") for module in modules)
            operations = [line for line in llvm.splitlines() if re.search(r"\b(?:fadd|fmul|fsub|fdiv|fcmp|fneg|fptrunc|fpext)\b", line)]
            self.assertTrue(operations)
            self.assertRegex(llvm, r"\bfmul\b")
            self.assertNotIn("llvm.fma", llvm)
            self.assertNotIn("llvm.fmuladd", llvm)
            self.assertFalse(any(re.search(r"\b(?:fast|reassoc|contract|arcp|afn|nsz|nnan|ninf|double)\b", line) for line in operations))
            for fresh in range(2):
                vm = relax.VirtualMachine(executable, tvm.cpu())
                supplied = tvm.nd.array(tokens.numpy().copy())
                for repeat in range(2):
                    with self.subTest(mode=mode, fresh=fresh, repeat=repeat):
                        actual = vm["main"](supplied).numpy()
                        self.assertEqual(actual.dtype, np.float32)
                        self.assertEqual(actual.shape, (1, 48, 960))
                        self.assertEqual(actual.tobytes(), source_value.tobytes())
                        self.assertEqual(supplied.numpy().tobytes(), tokens.numpy().tobytes())
                del vm, supplied
            del executable, modules, lowered
            gc.collect()
        self.assertEqual(str(source.graph), original_graph)
        self.assertEqual(str(source.graph_signature), original_signature)
        self.assertEqual(str(prepared.graph_signature), original_signature)
        self.assertEqual(tokens.numpy().tobytes(), tokens_before)
        for exported in (source, prepared):
            self.assertEqual(hashlib.sha256(exported.state_dict["weight"].view(torch.uint16).numpy().tobytes()).hexdigest(), weight_before)
        self.assertEqual(hashlib.sha256(model.weight.view(torch.uint16).numpy().tobytes()).hexdigest(), weight_before)
        self.assertEqual(hashlib.sha256(weight.raw_data).hexdigest(), weight_before)
        self.assertEqual(hashlib.sha256(graph.SerializeToString()).hexdigest(), graph_before)
        scalar_after = onnx.numpy_helper.to_array(next(attr.t for node in graph.graph.node if node.op_type == "Constant" for attr in node.attribute if attr.type == onnx.AttributeProto.TENSOR))
        self.assertEqual(scalar_after.dtype, np.float32)
        self.assertEqual(int(scalar_after.view(np.uint32)), 0x41f7def6)
        del graph, imported, prepared, source, weight
        gc.collect()

        # Exercise actual factory selection and StageCompiler, including the default control.
        from verify_session import StageCompiler, array
        policy = NUMERICS.source_scalar_contract(torch, "torch_cpu_language_scale960_v1")
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_scalar_policy": policy}}))
        options = FACTORY.get_onnx_export_options(torch, marker)
        self.assertEqual(options["source_scalar_policy"], policy["name"])
        self.assertNotIn("source_scalar_policy", FACTORY.get_onnx_export_options(torch))
        inputs = (tokens.numpy().copy(),)
        for mode, selected in (("baseline", True), ("optimized", True), ("baseline", False)):
            with self.subTest(mode=mode, selected=selected), tempfile.TemporaryDirectory() as directory:
                compiler = StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", opset=18, export_options=options if selected else {"optimize": False})
                run = compiler(SimpleNamespace(name="language_scale960", module=model), inputs, 1)
                expected = source_value if selected else rounded_control
                for repeat in range(2):
                    self.assertEqual(array(run(inputs), np, torch).tobytes(), expected.tobytes())
                record = compiler.records[0]
                if selected:
                    self.assertEqual(record["source_scalar_export"]["name"], policy["name"])
                    self.assertEqual(record["source_scalar_export"]["site_count"], 1)
                    self.assertTrue(record["source_scalar_export"]["bindings_preserved"])
                else:
                    self.assertNotIn("source_scalar_export", record)
                saved = onnx.load(record["onnx_artifacts"][0]["path"])
                self.assertEqual(hashlib.sha256(saved.graph.initializer[0].raw_data).hexdigest(), weight_before)
                self.assertEqual(tokens.numpy().tobytes(), tokens_before)
                self.assertEqual(hashlib.sha256(model.weight.view(torch.uint16).numpy().tobytes()).hexdigest(), weight_before)
                del run, compiler, saved
            gc.collect()

class SmolVLASourceRotaryTests(unittest.TestCase):
    policy = "torch_cpu_denoise_rotary64_table_v1"

    @staticmethod
    def fixture(torch, dtype=None):
        dtype = torch.float32 if dtype is None else dtype

        class Rotary(torch.nn.Module):
            def forward(self, mask):
                base = mask.sum(-1).unsqueeze(1) + torch.cat([torch.ones([1, 50], dtype=torch.bool, device="cpu")], 1).cumsum(1) - 1
                outputs = []
                for index in range(24):
                    timescale = 10000 ** (0.03125 * torch.arange(32, dtype=dtype, device="cpu"))
                    positions = base - base.min(1, keepdim=True).values if index % 3 == 0 else base
                    radians = positions.unsqueeze(2).to(dtype) / timescale.unsqueeze(0).unsqueeze(1).to(dtype)
                    angles = radians.unsqueeze(2)
                    outputs.extend((angles.sin(), angles.cos()))
                return tuple(outputs)

        return Rotary().eval()

    def test_source_rotary_defaults_factory_and_export_admission(self):
        import torch
        from verify_session import StageCompiler

        for policy, exporter in (("unknown", "dynamo"), (self.policy, "legacy")):
            with self.subTest(policy=policy), self.assertRaisesRegex(ValueError, "source rotary|Source rotary"):
                StageCompiler(Path("unused"), "baseline", None, torch, SimpleNamespace(), None, exporter=exporter, export_options={"source_rotary_policy": policy})
        self.assertNotIn("source_rotary_policy", FACTORY.get_onnx_export_options(torch))
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_rotary_policy": {"name": self.policy}}}))
        self.assertEqual(FACTORY.get_onnx_export_options(torch, marker)["source_rotary_policy"], self.policy)
        with mock.patch.object(torch, "get_num_threads", return_value=2), self.assertRaisesRegex(ValueError, "one native thread"):
            NUMERICS.source_rotary_contract(torch, self.policy)

    def test_source_rotary_no_site_strict_DAG_and_table_authentication(self):
        import copy
        import torch

        mask = torch.ones(1, 177, dtype=torch.bool)
        with mock.patch.object(NUMERICS, "source_rotary_contract", side_effect=lambda *args: {"name": self.policy}):
            with self.assertRaisesRegex(ValueError, "ExportedProgram"):
                NUMERICS.prepare_source_rotary_export(torch, object(), self.policy)
            for model, inputs in ((torch.nn.Identity(), (mask,)), (self.fixture(torch, torch.float64), (mask,))):
                source = torch.export.export(model, inputs)
                graph, signature = str(source.graph), str(source.graph_signature)
                prepared, record = NUMERICS.prepare_source_rotary_export(torch, source, self.policy)
                self.assertIs(prepared, source)
                self.assertEqual(record["site_count"], 0)
                self.assertEqual(record["added_constant_names"], [])
                self.assertEqual((str(source.graph), str(source.graph_signature)), (graph, signature))
            source = torch.export.export(self.fixture(torch), (mask,))
            mutations = [(torch.ops.aten.pow.Scalar, "wavelength", lambda n: (10001, n.args[1])),
                         (torch.ops.aten.mul.Tensor, "exponent", lambda n: (n.args[0], 0.0312501)),
                         (torch.ops.aten.arange.default, "frequency range", lambda n: (31,)),
                         (torch.ops.aten.sum.dim_IntList, "integer-position", lambda n: (n.args[0], [-2]))]
            for target, error, change in mutations:
                changed = copy.deepcopy(source)
                node = next(node for node in changed.graph.nodes if node.target == target)
                node.args = change(node)
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    NUMERICS.prepare_source_rotary_export(torch, changed, self.policy)
            for replacement, error in ((torch.ops.aten.neg.default, "inventory"), (torch.ops.aten.sin.default, "sin/cos pairs")):
                changed = copy.deepcopy(source)
                next(node for node in changed.graph.nodes if node.target == torch.ops.aten.cos.default).target = replacement
                with self.subTest(error=error), self.assertRaisesRegex(ValueError, error):
                    NUMERICS.prepare_source_rotary_export(torch, changed, self.policy)
        with mock.patch.dict(NUMERICS.SOURCE_ROTARY_TABLE_WORDS, {"sine": "0" * 64}), self.assertRaisesRegex(ValueError, "table words"):
            NUMERICS._source_rotary_tables(torch)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "requires opt-in pinned TVM/PyTorch CPU build")
    def test_source_rotary_public_factory_compiler_preserves_native_words(self):
        import gc
        import hashlib
        import numpy as np
        import onnx
        import torch
        import tvm
        from verify_session import StageCompiler, array, onnx_tensors
        if torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0" or torch.backends.cpu.get_cpu_capability() != "AVX512" or torch.get_num_threads() != 1:
            self.skipTest("requires pinned AVX512 PyTorch CPU build and one native thread")

        contract = NUMERICS.source_rotary_contract(torch, self.policy)
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_rotary_policy": contract}}))
        options = FACTORY.get_onnx_export_options(torch, marker)
        model = self.fixture(torch)
        mask = torch.arange(177).reshape(1, 177) < 129
        source = torch.export.export(model, (mask,))
        graph, signature = str(source.graph), str(source.graph_signature)
        bindings = StageCompiler(Path("unused"), "baseline", np, torch, onnx, tvm)._export_bindings(source)
        prepared, record = NUMERICS.prepare_source_rotary_export(torch, source, self.policy)
        self.assertEqual(record["site_count"], 48)
        self.assertEqual(len(record["added_constant_names"]), 2)
        self.assertTrue(record["original_user_and_output_specs_preserved"])
        self.assertEqual(StageCompiler._export_abi(prepared), StageCompiler._export_abi(source))
        tables_before = {name: hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() for name, value in prepared.constants.items()}
        for count in (0, 129, 177):
            control = torch.arange(177).reshape(1, 177) < count
            for native, actual in zip(model(control), prepared.module()(control)):
                self.assertEqual(native.view(torch.uint8).numpy().tobytes(), actual.view(torch.uint8).numpy().tobytes())
        inputs = (mask.numpy().copy(),)
        input_before = inputs[0].tobytes()
        expected = tuple(value.numpy().copy() for value in model(mask))
        self.assertEqual(len(expected), 48)
        for mode in ("baseline", "optimized"):
            for fresh in range(2):
                with self.subTest(mode=mode, fresh=fresh), tempfile.TemporaryDirectory() as directory:
                    compiler = StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", opset=18, export_options=options)
                    run = compiler(SimpleNamespace(name="denoise_rotary64", module=model), inputs, 48)
                    for repeat in range(2):
                        actual = run(inputs)
                        self.assertEqual(len(actual), 48)
                        for value, native in zip(actual, expected):
                            decoded = array(value, np, torch)
                            self.assertEqual(decoded.dtype, np.float32)
                            self.assertEqual(decoded.shape, (1, 50, 1, 32))
                            self.assertEqual(decoded.tobytes(), native.tobytes())
                    receipt = compiler.records[0]["source_rotary_export"]
                    self.assertEqual(receipt["site_count"], 48)
                    self.assertTrue(receipt["bindings_preserved"])
                    saved = onnx.load(compiler.records[0]["onnx_artifacts"][0]["path"])
                    self.assertEqual([value.name for value in saved.graph.input], ["input_0"])
                    self.assertEqual(len(saved.graph.output), 48)
                    self.assertFalse(any(node.op_type in ("Sin", "Cos") for node in saved.graph.node))
                    tables = [value for value in onnx_tensors(saved, onnx) if value.data_type == onnx.TensorProto.FLOAT and tuple(value.dims) == (228, 32)]
                    self.assertEqual({hashlib.sha256(value.raw_data).hexdigest() for value in tables}, {table["sha256"] for table in contract["table_bindings"]})
                    self.assertEqual(inputs[0].tobytes(), input_before)
                    del run, compiler, saved
                gc.collect()
        self.assertEqual((str(source.graph), str(source.graph_signature)), (graph, signature))
        self.assertEqual(StageCompiler(Path("unused"), "baseline", np, torch, onnx, tvm)._export_bindings(source), bindings)
        self.assertEqual({name: hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest() for name, value in prepared.constants.items()}, tables_before)


class SmolVLASourceFlashTests(unittest.TestCase):
    policy = "torch_cpu_flash_bf16_vision1024_v1"

    @staticmethod
    def inputs(torch, length=1024):
        shape = (1, 12, length, 64)
        rows = torch.arange(length).reshape(1, 1, length, 1)
        columns = torch.arange(64).reshape(1, 1, 1, 64)
        heads = torch.arange(12).reshape(1, 12, 1, 1)
        return tuple(torch.empty_strided(shape, (12 * length * 64, 64, 768, 1), dtype=torch.bfloat16).copy_(((rows * 3 + columns * 5 + heads * 7 + index) % 29 - 14).float() / 32) for index in range(3))

    @staticmethod
    def fixture(torch, mask_kind="zeros", length=1024, normalize=True, scalar=1.0):
        class ToyNativeVisionBF16QKV(torch.nn.Module):
            def forward(self, query, key, value, supplied_mask=None):
                if normalize:
                    query, key, value = (tensor.transpose(1, 2).contiguous().transpose(1, 2) for tensor in (query, key, value))
                shape = (1, 1, length, length)
                if mask_kind == "original":
                    ones = torch.ones([1, 32, 32], dtype=torch.float32, device="cpu").to(device="cpu", dtype=torch.bool)
                    ones = ones.view(1, 1024).unsqueeze(1).unsqueeze(2).expand(shape).to(torch.bfloat16)
                    inverse = torch.tensor(scalar, dtype=torch.bfloat16, device="cpu") - ones
                    mask = inverse.masked_fill(inverse.to(torch.bool), torch.finfo(torch.bfloat16).min)
                elif mask_kind == "dynamic":
                    mask = supplied_mask
                elif mask_kind == "nonzero":
                    mask = torch.ones(shape, dtype=torch.bfloat16, device="cpu")
                else:
                    mask = torch.zeros(shape, dtype=torch.bfloat16, device="cpu")
                    if mask_kind == "negativezero":
                        mask = mask.masked_fill(torch.ones(shape, dtype=torch.bool, device="cpu"), -0.0)
                if mask_kind == "unrelated_mutation":
                    positions = torch.zeros(1, 48, dtype=torch.int64)
                    positions[torch.ones(1, 48, dtype=torch.bool)] = 1
                result = torch.nn.functional.scaled_dot_product_attention(query, key, value, mask, dropout_p=0.0, is_causal=False, scale=0.125)
                return (result, positions) if mask_kind == "unrelated_mutation" else result
        return ToyNativeVisionBF16QKV().eval()

    def test_source_flash_defaults_and_factory_export_refusals(self):
        import torch
        from verify_session import StageCompiler

        for policy, exporter, fused in (("unknown", "dynamo", True), (self.policy, "legacy", False), (self.policy, "dynamo", False)):
            with self.subTest(policy=policy, exporter=exporter, fused=fused), self.assertRaisesRegex(ValueError, "[Ff]lash"):
                StageCompiler(Path("unused"), "baseline", None, torch, SimpleNamespace(), None, exporter=exporter, bf16_fused_export=fused, export_options={"source_flash_policy": policy})
        self.assertNotIn("source_flash_policy", FACTORY.get_onnx_export_options(torch))
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_flash_policy": {"name": self.policy}}}))
        self.assertEqual(FACTORY.get_onnx_export_options(torch, marker)["source_flash_policy"], self.policy)
        with mock.patch.object(torch, "get_num_threads", return_value=2), self.assertRaisesRegex(ValueError, "one native thread"):
            NUMERICS.source_flash_contract(torch, self.policy)

    def test_source_flash_requires_original_zero_mask_DAG_layout_and_flags(self):
        import copy
        import torch

        inputs = self.inputs(torch)
        with mock.patch.object(NUMERICS, "source_flash_contract", side_effect=lambda *args: {"name": self.policy}):
            for kind in ("zeros", "original", "unrelated_mutation"):
                source = torch.export.export(self.fixture(torch, kind), inputs)
                graph, signature = str(source.graph), str(source.graph_signature)
                constants = {name: value.clone() for name, value in source.constants.items()}
                prepared, record = NUMERICS.prepare_source_flash_export(torch, source, self.policy, expected_count=1)
                self.assertIs(prepared, source)
                self.assertEqual(record["site_count"], 1)
                self.assertEqual((str(source.graph), str(source.graph_signature)), (graph, signature))
                self.assertTrue(all(torch.equal(source.constants[name], original) for name, original in constants.items()))
            for kind in ("dynamic", "nonzero", "negativezero"):
                arguments = inputs + (torch.zeros(1, 1, 1024, 1024, dtype=torch.bfloat16),) if kind == "dynamic" else inputs
                changed = torch.export.export(self.fixture(torch, kind), arguments)
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "[Ff]lash"):
                    NUMERICS.prepare_source_flash_export(torch, changed, self.policy)
            changed = torch.export.export(self.fixture(torch, "original", scalar=0.0), inputs)
            with self.assertRaisesRegex(ValueError, "scalar-one"):
                NUMERICS.prepare_source_flash_export(torch, changed, self.policy)
            changed = torch.export.export(self.fixture(torch, normalize=False), tuple(value.contiguous() for value in inputs))
            with self.assertRaisesRegex(ValueError, "strides"):
                NUMERICS.prepare_source_flash_export(torch, changed, self.policy)
            source = torch.export.export(self.fixture(torch), inputs)
            mutated = copy.deepcopy(source)
            attention = next(node for node in mutated.graph.nodes if node.target == torch.ops.aten.scaled_dot_product_attention.default)
            mask = attention.args[3]
            with mutated.graph.inserting_before(attention):
                alias = mutated.graph.call_function(torch.ops.aten.alias.default, (mask,))
                writer = mutated.graph.call_function(torch.ops.aten.fill_.Scalar, (alias, 1.0))
            alias.meta["val"] = writer.meta["val"] = mask.meta["val"]
            mutated.graph.lint()
            with self.assertRaisesRegex(ValueError, "[Ff]lash"):
                NUMERICS.prepare_source_flash_export(torch, mutated, self.policy)
            for flags in ({"scale": 0.25}, {"dropout_p": 0.1}, {"is_causal": True}, {"enable_gqa": True}):
                changed = copy.deepcopy(source)
                node = next(node for node in changed.graph.nodes if node.target == torch.ops.aten.scaled_dot_product_attention.default)
                node.kwargs = {**node.kwargs, **flags}
                with self.subTest(flags=flags), self.assertRaisesRegex(ValueError, "[Ff]lash"):
                    NUMERICS.prepare_source_flash_export(torch, changed, self.policy)
            changed = copy.deepcopy(source)
            node = next(node for node in changed.graph.nodes if node.target == torch.ops.aten.scaled_dot_product_attention.default)
            key = node.args[1]
            with key.meta["val"].fake_mode:
                key.meta["val"] = torch.empty_strided((1, 12, 1024, 32), (393216, 32, 384, 1), dtype=torch.bfloat16)
            with self.assertRaisesRegex(ValueError, "geometry"):
                NUMERICS.prepare_source_flash_export(torch, changed, self.policy)
            small = torch.export.export(self.fixture(torch, length=4), self.inputs(torch, 4))
            prepared, record = NUMERICS.prepare_source_flash_export(torch, small, self.policy)
            self.assertIs(prepared, small)
            self.assertEqual(record["site_count"], 0)
            with self.assertRaisesRegex(ValueError, "inventory"):
                NUMERICS.prepare_source_flash_export(torch, small, self.policy, expected_count=1)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "requires opt-in pinned TVM/PyTorch CPU build")
    def test_source_flash_public_factory_compiler_and_ordinary_fallback(self):
        import hashlib
        import numpy as np
        import onnx
        import torch
        import tvm
        from verify_session import StageCompiler, array
        if torch.version.git_version != "449b1768410104d3ed79d3bcfe4ba1d65c7f22c0" or torch.backends.cpu.get_cpu_capability() != "AVX512" or torch.get_num_threads() != 1:
            self.skipTest("requires pinned AVX512 PyTorch CPU build and one native thread")

        contract = NUMERICS.source_flash_contract(torch, self.policy)
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_flash_policy": contract}}))
        options = FACTORY.get_onnx_export_options(torch, marker)
        bindings = contract["bindings"]
        for mode, length, selected in (("baseline", 1024, True), ("optimized", 1024, True), ("baseline", 4, True), ("baseline", 4, False)):
            model, tensors = self.fixture(torch, length=length), self.inputs(torch, length)
            if length == 4:
                tensors[0].zero_()
                tensors[1].zero_()
                tensors[2].fill_(1)
            inputs = tuple(array(value, np, torch) for value in tensors)
            before = [value.tobytes() for value in inputs]
            expected = model(*tensors).contiguous().view(torch.uint16).numpy().copy()
            source = torch.export.export(model, tensors)
            graph, signature = str(source.graph), str(source.graph_signature)
            prepared, admitted = NUMERICS.prepare_source_flash_export(torch, source, self.policy)
            self.assertIs(prepared, source)
            self.assertEqual(admitted["site_count"], int(length == 1024))
            with self.subTest(mode=mode, length=length, selected=selected), tempfile.TemporaryDirectory() as directory:
                compiler = StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", opset=22, bf16_fused_export=True, export_options=options if selected else {"optimize": False})
                run = compiler(SimpleNamespace(name="native_vision_flash", module=model), inputs, 1)
                for repeat in range(2):
                    actual = array(run(inputs), np, torch)
                    self.assertEqual(actual.dtype.name, "bfloat16")
                    self.assertEqual(actual.view(np.uint16).tobytes(), expected.tobytes())
                record = compiler.records[0]
                if selected:
                    self.assertEqual(record["source_flash_export"]["site_count"], int(length == 1024))
                    self.assertTrue(record["source_flash_export"]["bindings_preserved"])
                else:
                    self.assertNotIn("source_flash_export", record)
                saved = onnx.load(record["onnx_artifacts"][0]["path"])
                custom = [node for node in saved.graph.node if node.domain == "merlin.source_preserving" and node.op_type == "TorchFlashBf16Vision1024"]
                self.assertEqual([(len(node.input), len(node.output), len(node.attribute)) for node in custom], [(4, 1, 0)] if length == 1024 else [])
                self.assertEqual(saved.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.BFLOAT16)
                self.assertEqual([value.name for value in saved.graph.input], ["input_0", "input_1", "input_2"])
                self.assertEqual([value.tobytes() for value in inputs], before)
                self.assertEqual((str(source.graph), str(source.graph_signature)), (graph, signature))
        for binding in bindings:
            self.assertEqual(hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest(), binding["sha256"])


class SmolVLASourcePolicyTests(unittest.TestCase):
    policies = {
        "mean": ("Source mean", ("torch_cpu_mean720_v1", "torch_cpu_mean720_960_v1")),
        "softmax": ("Source softmax", ("torch_cpu_softmax177_v1", "torch_cpu_softmax177_227_v1", "torch_cpu_softmax177_227_prefix_v1")),
        "layernorm": ("Source LayerNorm", ("torch_cpu_layernorm768_v1",)),
        "conv": ("Source convolution", ("torch_cpu_conv3x16_v1",)),
        "sgemm": ("Source SGEMM", ("torch_cpu_sgemm177_v1", "torch_cpu_attention177_227_v1", "torch_cpu_attention177_227_prefix_v1")),
        "scalar": ("Source scalar", ("torch_cpu_language_scale960_v1",)),
        "rotary": ("Source rotary", ("torch_cpu_denoise_rotary64_table_v1",)),
        "flash": ("[Ff]lash", ("torch_cpu_flash_bf16_vision1024_v1",)),
        "silu": ("[Ss]ource SiLU", ("torch_cpu_silu_fp32_v1",)),
    }

    def test_native_source_policies_need_no_bindings_and_unknown_policies_fail(self):
        sentinel = object()
        for kind, (error, _) in self.policies.items():
            with self.subTest(kind=kind):
                contract = getattr(NUMERICS, "source_" + kind + "_contract")
                self.assertIsNone(contract(None))
                if kind in ("scalar", "rotary", "flash", "silu"):
                    self.assertEqual(getattr(NUMERICS, "prepare_source_" + kind + "_export")(None, sentinel), (sentinel, None))
                else:
                    self.assertEqual(getattr(NUMERICS, "source_" + kind + "_translations")(None), {})
                unknown_error = "Unknown source SiLU" if kind == "silu" else error.replace("Source ", "Unknown source ")
                with self.assertRaisesRegex(ValueError, unknown_error):
                    contract(None, "unknown")

    def test_reduction_contracts_refuse_unproved_runtime_and_modified_binary(self):
        runtime = SimpleNamespace(version=SimpleNamespace(git_version="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"), backends=SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX512")), get_num_threads=lambda: 1, __file__="/tmp/pinned-torch/__init__.py")
        for kind in ("mean", "softmax", "layernorm"):
            contract = getattr(NUMERICS, "source_" + kind + "_contract")
            for policy in self.policies[kind][1]:
                with self.subTest(kind=kind, policy=policy), mock.patch("verify_resnet.file_identity", return_value={"sha256": "changed"}), self.assertRaisesRegex(ValueError, "independently proved"):
                    contract(runtime, policy)
                for field, value in (("get_num_threads", lambda: 2), ("version", SimpleNamespace(git_version="different")), ("backends", SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX2")))):
                    with self.subTest(kind=kind, policy=policy, field=field), mock.patch.object(runtime, field, value), self.assertRaisesRegex(ValueError, "pinned PyTorch"):
                        contract(runtime, policy)

    def test_source_closure_includes_rotary_and_every_active_policy_binding(self):
        import bf16_export
        torch = object()
        policies = {kind: choices[0] for kind, (_, choices) in self.policies.items()}
        records = {kind: {"name": policy, "bindings": [{"path": "/tmp/" + kind + ".bin", "sha256": kind}]} for kind, policy in policies.items()}
        contracts = {"source_" + kind + "_contract": mock.Mock(return_value=records[kind]) for kind in policies}
        implementation = {"path": bf16_export.__file__, "sha256": "implementation"}
        with mock.patch.multiple(bf16_export, **contracts), mock.patch.object(FACTORY, "file_identity", return_value=implementation) as identify:
            self.assertEqual(FACTORY.source_policy_records(torch, dict.fromkeys(policies, "native")), {})
            self.assertEqual(FACTORY.source_policy_artifacts({}), [])
            identify.assert_not_called()
            for contract in contracts.values():
                contract.assert_not_called()
            for selected in ({"rotary": policies["rotary"]}, policies):
                with self.subTest(policies=tuple(selected)):
                    identify.reset_mock()
                    for contract in contracts.values():
                        contract.reset_mock()
                    actual = FACTORY.source_policy_records(torch, selected)
                    self.assertEqual(actual, {"source_" + kind + "_policy": records[kind] for kind in selected})
                    self.assertEqual(FACTORY.source_policy_artifacts(actual), [binding for kind in selected for binding in records[kind]["bindings"]] + [implementation])
                    identify.assert_called_once_with(bf16_export.__file__, FACTORY.safe_path)
                    for kind, policy in policies.items():
                        contract = contracts["source_" + kind + "_contract"]
                        if kind in selected:
                            contract.assert_called_once_with(torch, policy)
                        else:
                            contract.assert_not_called()

    def test_source_policies_require_integer_session_and_one_thread_before_checkpoint_access(self):
        base = {"checkpoint_dir": "/missing", "backbone_dir": "/missing", "model2mlir_root": "/missing"}
        for kind, (error, policies) in self.policies.items():
            key = "source_" + kind + "_policy"
            changes = [{key: "unknown"}]
            for policy in policies:
                changes.extend(({key: policy}, {key: policy, "quantized": True, "integer_diagnostic_dir": "/missing", "cpu_threads": 2}))
            for change in changes:
                with self.subTest(kind=kind, change=change), mock.patch.object(FACTORY, "checkpoint_artifacts", side_effect=AssertionError("Admission must precede checkpoint access")), self.assertRaisesRegex(ValueError, error):
                    FACTORY.get_model_and_inputs(**base, **change)

    def test_session_hooks_select_only_declared_policies_and_preserve_default_table(self):
        import torch
        import bf16_export
        import tinyllama_integer
        original = tinyllama_integer.export_options(torch)["custom_translation_table"]
        operators = {"mean": torch.ops.aten.mean.dim, "softmax": torch.ops.aten.softmax.int,
                     "layernorm": torch.ops.aten.layer_norm.default, "conv": torch.ops.aten.convolution.default,
                     "sgemm": torch.ops.aten.matmul.default}
        for kind, operator in operators.items():
            with self.subTest(kind=kind), mock.patch.object(tinyllama_integer, "export_options", side_effect=lambda _: {"custom_translation_table": dict(original)}), mock.patch.object(bf16_export, "source_" + kind + "_translations", return_value={operator: "selected"}) as select:
                self.assertEqual(FACTORY.get_onnx_export_options(torch)["custom_translation_table"], original)
                model = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {}}))
                self.assertEqual(FACTORY.get_onnx_export_options(torch, model)["custom_translation_table"], original)
                select.assert_not_called()
                for policy in self.policies[kind][1]:
                    with self.subTest(policy=policy):
                        select.reset_mock()
                        model.session.metadata["provenance"]["source_" + kind + "_policy"] = {"name": policy}
                        self.assertEqual(FACTORY.get_onnx_export_options(torch, model)["custom_translation_table"], {**original, operator: "selected"})
                        select.assert_called_once_with(torch, policy)
                        self.assertNotIn(operator, original)


class SmolVLASourceMeanTests(unittest.TestCase):
    def test_export_selects_only_bound_fp32_last_axis_keepdim_geometry(self):
        import torch
        class Means(torch.nn.Module):
            def forward(self, value, other):
                return value.mean(-1, keepdim=True), value.mean(-1), value.mean(1, keepdim=True), other.mean(-1, keepdim=True)
        for dtype in (torch.float32, torch.bfloat16):
            inputs = (torch.arange(36000, dtype=torch.float32).reshape(1, 50, 720).to(dtype), torch.ones(1, 2, 720, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            with mock.patch.object(NUMERICS, "source_mean_contract", return_value={"name": "torch_cpu_mean720_v1"}):
                table = NUMERICS.source_mean_translations(torch, "torch_cpu_mean720_v1")
            graph = torch.onnx.export(Means().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchMean720Fp32", 1, 1, 0)] if dtype == torch.float32 else [])
            self.assertEqual(sum(node.op_type == "ReduceMean" for node in graph.graph.node), 3 if dtype == torch.float32 else 4)
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))

    def test_real_export_extends_only_declared_mean960_geometry_and_preserves_720(self):
        import onnx
        import torch
        class Means(torch.nn.Module):
            def forward(self, short, wide, other):
                return (short.mean(-1, keepdim=True), wide.mean(-1, keepdim=True), wide.mean(2, keepdim=True),
                        wide.mean(-1), wide.mean(1, keepdim=True), other.mean(-1, keepdim=True),
                        wide.mean(-1, keepdim=True, dtype=torch.float64))

        for dtype in (torch.float32, torch.bfloat16):
            inputs = tuple(torch.linspace(-1, 1, rows * width).reshape(1, rows, width).to(dtype) for rows, width in ((50, 720), (177, 960), (176, 960)))
            before = tuple(value.clone() for value in inputs)
            for policy in ("native", "torch_cpu_mean720_v1", "torch_cpu_mean720_960_v1"):
                with self.subTest(dtype=dtype, policy=policy):
                    if policy == "native":
                        table = NUMERICS.source_mean_translations(torch, policy)
                    else:
                        with mock.patch.object(NUMERICS, "source_mean_contract", return_value={"name": policy}):
                            table = NUMERICS.source_mean_translations(torch, policy)
                    graph = torch.onnx.export(Means().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
                    selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
                    expected = [] if dtype != torch.float32 or policy == "native" else ["TorchMean720Fp32"]
                    if dtype == torch.float32 and policy == "torch_cpu_mean720_960_v1":
                        expected += ["TorchMean960Fp32"] * 2
                    self.assertEqual([node.op_type for node in selected], expected)
                    self.assertEqual([(len(node.input), len(node.output), len(node.attribute)) for node in selected], [(1, 1, 0)] * len(expected))
                    self.assertEqual(sum(node.op_type == "ReduceMean" for node in graph.graph.node), 7 - len(expected))
                    infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
                    for node in selected:
                        for name in (*node.input, *node.output):
                            self.assertEqual(infos[name].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                        shape = [dimension.dim_value for dimension in infos[node.output[0]].type.tensor_type.shape.dim]
                        self.assertEqual(shape, [1, 50 if node.op_type == "TorchMean720Fp32" else 177, 1])
                    self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))


class SmolVLASourceSiluTests(unittest.TestCase):
    policy = "torch_cpu_silu_fp32_v1"

    def test_source_silu_defaults_contract_and_factory_refusals(self):
        import torch
        from verify_session import StageCompiler
        for policy, exporter, fused in (("unknown", "dynamo", True), (self.policy, "legacy", False), (self.policy, "dynamo", False)):
            with self.subTest(policy=policy, exporter=exporter, fused=fused), self.assertRaisesRegex(ValueError, "[Ss]ource SiLU"):
                StageCompiler(Path("unused"), "baseline", None, torch, SimpleNamespace(), None, exporter=exporter, bf16_fused_export=fused, export_options={"source_silu_policy": policy})
        self.assertNotIn("source_silu_policy", FACTORY.get_onnx_export_options(torch))
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_silu_policy": {"name": self.policy}}}))
        self.assertEqual(FACTORY.get_onnx_export_options(torch, marker)["source_silu_policy"], self.policy)
        with mock.patch.object(torch, "get_num_threads", return_value=2), self.assertRaisesRegex(ValueError, "one native thread"):
            NUMERICS.source_silu_contract(torch, self.policy)
        with mock.patch.dict(os.environ, {"ATEN_CPU_CAPABILITY": "avx512"}), self.assertRaisesRegex(ValueError, "ISA override"):
            NUMERICS.source_silu_contract(torch, self.policy)
        with mock.patch.object(NUMERICS, "_source_cpu_bindings", side_effect=ValueError("independently proved native library")), self.assertRaisesRegex(ValueError, "independently proved"):
            NUMERICS.source_silu_contract(torch, self.policy)

    def test_source_silu_EP_admission_preserves_BF16_and_refuses_unproved_routes(self):
        import copy
        import torch
        class Activation(torch.nn.Module):
            def forward(self, value):
                return torch.ops.aten.silu.default(value)
        class Identity(torch.nn.Module):
            def forward(self, value):
                return value
        with mock.patch.object(NUMERICS, "source_silu_contract", side_effect=lambda *args: {"name": self.policy}):
            with self.assertRaisesRegex(ValueError, "ExportedProgram"):
                NUMERICS.prepare_source_silu_export(torch, object(), self.policy)
            for model, value, expected in ((Identity(), torch.ones(32), 0), (Activation(), torch.ones(32, dtype=torch.bfloat16), 0), (Activation(), torch.ones(1, 50, 720), 1), (Activation(), torch.ones(640), 1)):
                source = torch.export.export(model, (value,))
                graph, signature = str(source.graph), str(source.graph_signature)
                prepared, record = NUMERICS.prepare_source_silu_export(torch, source, self.policy)
                self.assertIs(prepared, source)
                self.assertEqual(record["site_count"], expected)
                self.assertFalse(record["graph_mutated"])
                self.assertEqual((str(source.graph), str(source.graph_signature)), (graph, signature))
            for value in (torch.ones(31), torch.ones(32, dtype=torch.float64), torch.ones(32, dtype=torch.float16), torch.ones(4, 8).T, torch.ones(33)[1:]):
                source = torch.export.export(Activation(), (value,))
                with self.subTest(shape=tuple(value.shape), dtype=value.dtype), self.assertRaisesRegex(ValueError, "Source SiLU"):
                    NUMERICS.prepare_source_silu_export(torch, source, self.policy)
            changed = copy.deepcopy(torch.export.export(Activation(), (torch.ones(32),)))
            node = next(node for node in changed.graph.nodes if node.target == torch.ops.aten.silu.default)
            with node.meta["val"].fake_mode:
                node.meta["val"] = torch.ones(64)
            with self.assertRaisesRegex(ValueError, "result metadata"):
                NUMERICS.prepare_source_silu_export(torch, changed, self.policy)

    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "requires opt-in pinned TVM/PyTorch CPU build")
    def test_source_silu_public_factory_compiler_exact_native_and_default_dtype_routes(self):
        import hashlib
        import numpy as np
        import onnx
        import torch
        import tvm
        from verify_session import StageCompiler, array
        class Activation(torch.nn.Module):
            def forward(self, value):
                return torch.ops.aten.silu.default(value)
        contract = NUMERICS.source_silu_contract(torch, self.policy)
        marker = SimpleNamespace(session=SimpleNamespace(metadata={"provenance": {"source_silu_policy": contract}}))
        options = FACTORY.get_onnx_export_options(torch, marker)
        value = torch.linspace(-8, 8, 36000).reshape(1, 50, 720)
        inputs = (array(value, np, torch),)
        before = inputs[0].tobytes()
        expected = Activation()(value).numpy().copy()
        source = torch.export.export(Activation(), (value,))
        graph, signature = str(source.graph), str(source.graph_signature)
        for mode in ("baseline", "optimized"):
            for fresh in range(2):
                with self.subTest(mode=mode, fresh=fresh), tempfile.TemporaryDirectory() as directory:
                    compiler = StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", opset=18, bf16_fused_export=True, export_options=options)
                    run = compiler(SimpleNamespace(name="native_time_silu", module=Activation()), inputs, 1)
                    for repeat in range(2):
                        actual = array(run(inputs), np, torch)
                        self.assertEqual(actual.dtype, np.float32)
                        self.assertEqual(actual.tobytes(), expected.tobytes())
                    record = compiler.records[0]
                    self.assertEqual(record["source_silu_export"]["site_count"], 1)
                    self.assertTrue(record["source_silu_export"]["bindings_preserved"])
                    saved = onnx.load(record["onnx_artifacts"][0]["path"])
                    custom = [node for node in saved.graph.node if node.op_type == "TorchSiluAvx512F32" and node.domain == "merlin.source_preserving"]
                    self.assertEqual([(len(node.input), len(node.output), len(node.attribute)) for node in custom], [(1, 1, 0)])
                    self.assertEqual([v.name for v in saved.graph.input], ["input_0"])
                    self.assertEqual(saved.graph.output[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
        for dtype in (torch.float32, torch.bfloat16):
            tensor = torch.ones(32, dtype=dtype)
            default = torch.onnx.export(Activation(), (tensor,), dynamo=True, opset_version=18, optimize=False, custom_translation_table=NUMERICS.translations(torch)).model_proto
            self.assertFalse(any(node.op_type == "TorchSiluAvx512F32" for node in default.graph.node))
            active = torch.onnx.export(Activation(), (tensor,), dynamo=True, opset_version=18, optimize=False, custom_translation_table=NUMERICS.translations(torch, source_silu_policy=self.policy)).model_proto
            self.assertEqual(sum(node.op_type == "TorchSiluAvx512F32" for node in active.graph.node), int(dtype == torch.float32))
            if dtype == torch.bfloat16:
                self.assertEqual([node.op_type for node in active.graph.node], [node.op_type for node in default.graph.node])
        self.assertEqual(inputs[0].tobytes(), before)
        self.assertEqual((str(source.graph), str(source.graph_signature)), (graph, signature))
        for binding in contract["bindings"]:
            self.assertEqual(hashlib.sha256(Path(binding["path"]).read_bytes()).hexdigest(), binding["sha256"])


class SmolVLASourceSoftmaxTests(unittest.TestCase):
    def test_export_selects_only_bound_fp32_shape_and_axis(self):
        import torch
        class Softmax(torch.nn.Module):
            def forward(self, value, other):
                return value.softmax(-1), torch.ops.aten._softmax.default(value, -1, False), value.softmax(2), other.softmax(-1)
        for dtype in (torch.float32, torch.bfloat16):
            inputs = (torch.linspace(-2, 2, 132750).reshape(1, 15, 50, 177).to(dtype), torch.ones(1, 15, 50, 176, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            with mock.patch.object(NUMERICS, "source_softmax_contract", return_value={"name": "torch_cpu_softmax177_v1"}):
                table = NUMERICS.source_softmax_translations(torch, "torch_cpu_softmax177_v1")
            graph = torch.onnx.export(Softmax().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchSoftmaxAvx512F32LastDim177", 1, 1, 0)] * 2 if dtype == torch.float32 else [])
            self.assertEqual(sum(node.op_type == "Softmax" for node in graph.graph.node), 2 if dtype == torch.float32 else 4)
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))


    def test_real_export_keeps_177_policy_and_admits_227_only_when_selected(self):
        import torch
        class Softmax(torch.nn.Module):
            def forward(self, short, long):
                return short.softmax(-1), long.softmax(-1), long.softmax(2)
        for dtype in (torch.float32, torch.bfloat16):
            inputs = tuple(torch.linspace(-2, 2, 750 * width).reshape(1, 15, 50, width).to(dtype) for width in (177, 227))
            for policy in ("torch_cpu_softmax177_v1", "torch_cpu_softmax177_227_v1", "torch_cpu_softmax177_227_prefix_v1"):
                with mock.patch.object(NUMERICS, "source_softmax_contract", return_value={"name": policy}):
                    table = NUMERICS.source_softmax_translations(torch, policy)
                graph = torch.onnx.export(Softmax().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
                selected = [node.op_type for node in graph.graph.node if node.domain == "merlin.source_preserving"]
                expected = ["TorchSoftmaxAvx512F32LastDim177"] + (["TorchSoftmaxAvx512F32LastDim227"] if policy != "torch_cpu_softmax177_v1" else [])
                self.assertEqual(selected, expected if dtype == torch.float32 else [])
                self.assertEqual(sum(node.op_type == "Softmax" for node in graph.graph.node), 3 - len(selected))


    def test_prefix_policy_preserves_old_scopes_and_both_callback_fallbacks(self):
        import onnx
        import torch
        class Softmax(torch.nn.Module):
            def forward(self, prefix, wrong):
                return prefix.softmax(-1), torch.ops.aten._softmax.default(prefix, -1, False), prefix.softmax(2), wrong.softmax(-1)
        policy = "torch_cpu_softmax177_227_prefix_v1"
        with mock.patch.object(NUMERICS, "_source_cpu_bindings", return_value={"fixture": "bound"}):
            record = NUMERICS.source_softmax_contract(torch, policy)
        self.assertEqual(record["export_shapes"], [[1, 15, 50, 177], [1, 15, 50, 227], [1, 15, 177, 177]])
        self.assertIn("unqualified", record["qualification"])
        self.assertEqual(record["prefix_native_strides"], [469935, 31329, 177, 1])
        for dtype in (torch.float32, torch.bfloat16):
            inputs = (torch.linspace(-2, 2, 469935).reshape(1, 15, 177, 177).to(dtype), torch.ones(1, 15, 176, 177, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            for selected_policy in ("native", "torch_cpu_softmax177_v1", "torch_cpu_softmax177_227_v1", policy):
                with mock.patch.object(NUMERICS, "source_softmax_contract", return_value=None if selected_policy == "native" else {"name": selected_policy}):
                    table = NUMERICS.source_softmax_translations(torch, selected_policy)
                graph = torch.onnx.export(Softmax().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
                selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
                expected = 2 if selected_policy == policy and dtype == torch.float32 else 0
                self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchSoftmaxAvx512F32LastDim177", 1, 1, 0)] * expected)
                self.assertEqual(sum(node.op_type == "Softmax" for node in graph.graph.node), 4 - expected)
                infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
                for node in selected:
                    self.assertEqual([infos[name].type.tensor_type.elem_type for name in (*node.input, *node.output)], [onnx.TensorProto.FLOAT] * 2)
                    self.assertEqual([dim.dim_value for dim in infos[node.output[0]].type.tensor_type.shape.dim], [1, 15, 177, 177])
                self.assertTrue(all(torch.equal(value, original) for value, original in zip(inputs, before)))



class SmolVLASourceLayerNormTests(unittest.TestCase):
    def test_export_selects_original_bf16_affines_shape_and_epsilon(self):
        import torch
        class LayerNorm(torch.nn.Module):
            def forward(self, value, weight, bias):
                return torch.nn.functional.layer_norm(value, (768,), weight, bias, 1e-6), torch.nn.functional.layer_norm(value, (768,), weight, bias, 1e-5), torch.nn.functional.layer_norm(value, (768,), None, None, 1e-6)
        for dtype in (torch.bfloat16, torch.float32):
            inputs = (torch.linspace(-2, 2, 786432).reshape(1, 1024, 768).to(dtype), torch.ones(768, dtype=dtype), torch.zeros(768, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            with mock.patch.object(NUMERICS, "source_layernorm_contract", return_value={"name": "torch_cpu_layernorm768_v1"}):
                table = NUMERICS.source_layernorm_translations(torch, "torch_cpu_layernorm768_v1")
            graph = torch.onnx.export(LayerNorm().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchLayerNormAvx2Bf16Width768", 4, 1, 0)] if dtype == torch.bfloat16 else [])
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))
            if selected:
                from onnx import numpy_helper
                constants = {node.output[0]: numpy_helper.to_array(node.attribute[0].t) for node in graph.graph.node if node.op_type == "Constant" and node.attribute[0].type == 4}
                self.assertEqual(int(constants[selected[0].input[3]].view("uint32")), 0x358637bd)


class SmolVLASourceConvTests(unittest.TestCase):
    def test_contract_refuses_other_dispatch_and_modified_binary(self):
        import torch
        with mock.patch.object(NUMERICS, "_source_cpu_bindings", return_value=[]) as bindings:
            record = NUMERICS.source_conv_contract(torch, "torch_cpu_conv3x16_v1")
            self.assertEqual(record["operator"], "TorchConv2dAvx512Bf16IC3Kernel16Stride16")
            bindings.assert_called_once_with(torch, NUMERICS.SOURCE_CONV_ARTIFACTS, "Source convolution")
            for feature in ("_is_avx512_bf16_supported", "_is_amx_tile_supported"):
                with mock.patch.object(torch.cpu, feature, return_value=True), self.assertRaisesRegex(ValueError, "software-BF16"):
                    NUMERICS.source_conv_contract(torch, "torch_cpu_conv3x16_v1")
            with mock.patch.dict(os.environ, {"ONEDNN_MAX_CPU_ISA": "AVX2"}), self.assertRaisesRegex(ValueError, "software-BF16"):
                NUMERICS.source_conv_contract(torch, "torch_cpu_conv3x16_v1")
            with mock.patch.object(torch.backends, "mkldnn", SimpleNamespace(is_available=lambda: True, enabled=False)), self.assertRaisesRegex(ValueError, "software-BF16"):
                NUMERICS.source_conv_contract(torch, "torch_cpu_conv3x16_v1")
        with mock.patch.object(torch, "get_num_threads", return_value=1), mock.patch("verify_resnet.file_identity", return_value={"sha256": "changed"}), self.assertRaisesRegex(ValueError, "independently proved"):
            NUMERICS.source_conv_contract(torch, "torch_cpu_conv3x16_v1")

    def test_real_export_selects_only_bound_bf16_patch_geometry(self):
        import torch
        class Conv(torch.nn.Module):
            def forward(self, value, weight, bias):
                return torch.ops.aten.conv2d.default(value, weight, bias, [16, 16], [0, 0], [1, 1], 1), torch.ops.aten.convolution.default(value, weight, bias, [16, 16], [0, 0], [1, 1], False, [0, 0], 1), torch.nn.functional.conv2d(value, weight, None, stride=16), torch.nn.functional.conv2d(value, weight, bias, stride=8)
        for dtype in (torch.bfloat16, torch.float32):
            inputs = (torch.zeros(1, 3, 512, 512, dtype=dtype), torch.ones(768, 3, 16, 16, dtype=dtype), torch.zeros(768, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            with mock.patch.object(NUMERICS, "source_conv_contract", return_value={"name": "torch_cpu_conv3x16_v1"}):
                table = NUMERICS.source_conv_translations(torch, "torch_cpu_conv3x16_v1")
            graph = torch.onnx.export(Conv().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchConv2dAvx512Bf16IC3Kernel16Stride16", 3, 1, 0)] * 2 if dtype == torch.bfloat16 else [])
            self.assertEqual(sum(node.op_type == "Conv" for node in graph.graph.node), 2 if dtype == torch.bfloat16 else 4)
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))


class SmolVLASourceSgemmTests(unittest.TestCase):
    def test_contract_refuses_other_precision_dispatch_and_modified_binary(self):
        runtime = SimpleNamespace(version=SimpleNamespace(git_version="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"), backends=SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX512"), mkl=SimpleNamespace(is_available=lambda: True)), get_num_threads=lambda: 1, get_float32_matmul_precision=lambda: "highest", __file__="/tmp/pinned-torch/__init__.py")
        with mock.patch.object(NUMERICS, "_source_cpu_bindings", return_value=[]):
            record = NUMERICS.source_sgemm_contract(runtime, "torch_cpu_sgemm177_v1")
            self.assertEqual(record["operators"], ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177"])
            with mock.patch.object(runtime, "get_float32_matmul_precision", return_value="high"), self.assertRaisesRegex(ValueError, "highest-precision"):
                NUMERICS.source_sgemm_contract(runtime, "torch_cpu_sgemm177_v1")
            with mock.patch.dict(os.environ, {"MKL_CBWR": "COMPATIBLE"}), self.assertRaisesRegex(ValueError, "highest-precision"):
                NUMERICS.source_sgemm_contract(runtime, "torch_cpu_sgemm177_v1")
            with mock.patch.object(runtime.backends.mkl, "is_available", return_value=False), self.assertRaisesRegex(ValueError, "highest-precision"):
                NUMERICS.source_sgemm_contract(runtime, "torch_cpu_sgemm177_v1")
        with mock.patch("verify_resnet.file_identity", return_value={"sha256": "changed"}), self.assertRaisesRegex(ValueError, "independently proved"):
            NUMERICS.source_sgemm_contract(runtime, "torch_cpu_sgemm177_v1")

    def test_real_export_selects_only_bound_fp32_qk_and_pv(self):
        import torch
        class Attention(torch.nn.Module):
            def forward(self, query, key, probability, value):
                return query @ key, probability @ value, query[..., :49, :] @ key
        for dtype in (torch.float32, torch.bfloat16):
            inputs = tuple(torch.ones(shape, dtype=dtype) for shape in ((1, 15, 50, 64), (1, 15, 64, 177), (1, 15, 50, 177), (1, 15, 177, 64)))
            before = tuple(value.clone() for value in inputs)
            with mock.patch.object(NUMERICS, "source_sgemm_contract", return_value={"name": "torch_cpu_sgemm177_v1", "operators": ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177"]}):
                table = NUMERICS.source_sgemm_translations(torch, "torch_cpu_sgemm177_v1")
            graph = torch.onnx.export(Attention().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [("TorchMklSgemmF32QK177", 2, 1, 0), ("TorchMklSgemmF32PV177", 2, 1, 0)] if dtype == torch.float32 else [])
            self.assertEqual(sum(node.op_type == "MatMul" for node in graph.graph.node), 1 if dtype == torch.float32 else 3)
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))


    def test_extended_contract_refuses_other_bf16_dispatch(self):
        import torch
        policy = "torch_cpu_attention177_227_v1"
        with mock.patch.object(NUMERICS, "_source_cpu_bindings", return_value=[]) as bindings:
            record = NUMERICS.source_sgemm_contract(torch, policy)
            self.assertEqual(record["operators"], ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177", "TorchMklSgemmF32QK227", "TorchOnednnGemmBf16PV227"])
            self.assertIn("include/oneapi/dnnl/dnnl_version.h", bindings.call_args.args[1])
            for overrides in ({"DNNL_MAX_CPU_ISA": "AVX2"}, {"ONEDNN_DEFAULT_FPMATH_MODE": "BF16"}):
                with mock.patch.dict(os.environ, overrides), self.assertRaisesRegex(ValueError, "software-BF16"):
                    NUMERICS.source_sgemm_contract(torch, policy)
            with mock.patch.object(torch.cpu, "_is_avx512_bf16_supported", return_value=True), self.assertRaisesRegex(ValueError, "software-BF16"):
                NUMERICS.source_sgemm_contract(torch, policy)

    def test_real_export_keeps_177_policy_and_admits_only_proved_227_dtypes(self):
        import torch
        class Attention(torch.nn.Module):
            def forward(self, query, key, probability, value):
                return query @ key, probability @ value, query.to(torch.bfloat16) @ key.to(torch.bfloat16), probability.float() @ value.float(), query[..., :49, :] @ key
        inputs = (torch.ones((1, 15, 50, 64)), torch.ones((1, 15, 64, 227)), torch.ones((1, 15, 50, 227), dtype=torch.bfloat16), torch.ones((1, 15, 227, 64), dtype=torch.bfloat16))
        before = tuple(value.clone() for value in inputs)
        for policy in ("torch_cpu_sgemm177_v1", "torch_cpu_attention177_227_v1", "torch_cpu_attention177_227_prefix_v1"):
            operators = ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177"] + (["TorchMklSgemmF32QK227", "TorchOnednnGemmBf16PV227"] if policy != "torch_cpu_sgemm177_v1" else [])
            if policy == "torch_cpu_attention177_227_prefix_v1":
                operators.append("TorchOnednnGemmBf16PV177")
            with mock.patch.object(NUMERICS, "source_sgemm_contract", return_value={"name": policy, "operators": operators}):
                table = NUMERICS.source_sgemm_translations(torch, policy)
            graph = torch.onnx.export(Attention().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
            selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
            self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [(name, 2, 1, 0) for name in operators[2:4]])
            self.assertEqual(sum(node.op_type == "MatMul" for node in graph.graph.node), 5 - len(selected))
            self.assertTrue(all(torch.equal(original, value) for original, value in zip(before, inputs)))


    def test_prefix_bf16_pv177_is_distinct_from_old_fp32_pv177_and_defaults(self):
        import onnx
        import torch
        class Attention(torch.nn.Module):
            def forward(self, probability, value, expert):
                return probability @ value, probability[..., :176, :] @ value, expert.float() @ value.float()
        policy = "torch_cpu_attention177_227_prefix_v1"
        with mock.patch.object(NUMERICS, "_source_cpu_bindings", return_value=[]):
            record = NUMERICS.source_sgemm_contract(torch, policy)
            self.assertEqual(record["operators"], ["TorchMklSgemmF32QK177", "TorchMklSgemmF32PV177", "TorchMklSgemmF32QK227", "TorchOnednnGemmBf16PV227", "TorchOnednnGemmBf16PV177"])
            self.assertEqual(record["prefix_pv_native_strides"], [[469935, 31329, 177, 1], [169920, 64, 960, 1]])
            self.assertIn("unobserved", record["prefix_pv_evidence"])
            for overrides in ({"MKL_CBWR": "COMPATIBLE"}, {"DNNL_MAX_CPU_ISA": "AVX2"}, {"ONEDNN_DEFAULT_FPMATH_MODE": "BF16"}):
                with mock.patch.dict(os.environ, overrides), self.assertRaises(ValueError):
                    NUMERICS.source_sgemm_contract(torch, policy)
            with mock.patch.object(torch.cpu, "_is_avx512_bf16_supported", return_value=True), self.assertRaisesRegex(ValueError, "software-BF16"):
                NUMERICS.source_sgemm_contract(torch, policy)
        for dtype in (torch.bfloat16, torch.float32):
            inputs = (torch.ones(1, 15, 177, 177, dtype=dtype), torch.ones(1, 15, 177, 64, dtype=dtype), torch.ones(1, 15, 50, 177, dtype=dtype))
            before = tuple(value.clone() for value in inputs)
            for selected_policy in ("native", "torch_cpu_sgemm177_v1", "torch_cpu_attention177_227_v1", policy):
                with mock.patch.object(NUMERICS, "_source_cpu_bindings", return_value=[]):
                    table = NUMERICS.source_sgemm_translations(torch, selected_policy)
                graph = torch.onnx.export(Attention().eval(), inputs, dynamo=True, opset_version=18, optimize=False, custom_translation_table=table).model_proto
                selected = [node for node in graph.graph.node if node.domain == "merlin.source_preserving"]
                expected = (["TorchOnednnGemmBf16PV177"] if selected_policy == policy and dtype == torch.bfloat16 else []) + (["TorchMklSgemmF32PV177"] if selected_policy != "native" else [])
                self.assertEqual([(node.op_type, len(node.input), len(node.output), len(node.attribute)) for node in selected], [(name, 2, 1, 0) for name in expected])
                self.assertEqual(sum(node.op_type == "MatMul" for node in graph.graph.node), 3 - len(expected))
                infos = {value.name: value for value in (*graph.graph.input, *graph.graph.value_info, *graph.graph.output)}
                for node in selected:
                    wanted_dtype = onnx.TensorProto.BFLOAT16 if node.op_type == "TorchOnednnGemmBf16PV177" else onnx.TensorProto.FLOAT
                    self.assertEqual([infos[name].type.tensor_type.elem_type for name in (*node.input, *node.output)], [wanted_dtype] * 3)
                self.assertTrue(all(torch.equal(value, original) for value, original in zip(inputs, before)))



class SmolVLAArithmeticDiagnosticTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit LLVM-enabled TVM host build")
    def test_fused_bf16_silu_preserves_one_output_cast_and_fp32_default(self):
        from decimal import Decimal, localcontext
        import numpy as np
        import onnx
        import torch
        import tvm
        from verify_session import StageCompiler, array

        class Silu(torch.nn.Module):
            def forward(self, value):
                return torch.nn.functional.silu(value)

        values = torch.tensor([0.0, -0.0, 0.01171875, 0.01953125, 1.015625, 2.015625, -1.0078125, -2.015625, -15.75, -14.75], dtype=torch.bfloat16)
        source = Silu()(values)
        self.assertTrue(torch.equal(source.view(torch.uint16), Silu()(values.float()).bfloat16().view(torch.uint16)))
        staged = values * values.sigmoid()
        self.assertTrue(torch.any(staged[values > 0].view(torch.uint16) != source[values > 0].view(torch.uint16)))
        self.assertTrue(torch.any(staged[values < 0].view(torch.uint16) != source[values < 0].view(torch.uint16)))
        for value, output in zip(values[2:], source[2:], strict=True):
            word = int(output.view(torch.uint16))
            with localcontext() as context:
                context.prec = 80
                exact_input = Decimal.from_float(float(value))
                ideal = exact_input / (1 + (-exact_input).exp())
                distances = [(abs(ideal - Decimal.from_float(float(torch.tensor(candidate, dtype=torch.uint16).view(torch.bfloat16)))), candidate) for candidate in range(word - 2, word + 3)]
                ordered = sorted(distances, key=lambda item: (item[0], item[1] & 1))
                self.assertEqual(ordered[0][1], word)
                self.assertGreater(ordered[1][0] - ordered[0][0], Decimal("1e-65"))
        inputs = (array(values, np, torch),)
        for mode in ("baseline", "optimized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                compiler = StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", opset=18, bf16_fused_export=True)
                run = compiler(SimpleNamespace(name="bf16_silu", module=Silu()), inputs, 1)
                actual = array(run(inputs), np, torch)
                self.assertEqual(actual.tobytes(), array(source, np, torch).tobytes())
                self.assertEqual(array(run(inputs), np, torch).tobytes(), actual.tobytes())
                graph = onnx.load(compiler.records[0]["onnx_artifacts"][0]["path"])
                self.assertEqual([node.op_type for node in graph.graph.node], ["Cast", "Sigmoid", "Mul", "Cast"])
        # FP32 dispatcher and default BF16 export remain their built-in graphs.
        for dtype, fused, expected in ((torch.float32, False, ["Sigmoid", "Mul"]), (torch.float32, True, ["Sigmoid", "Mul"]), (torch.bfloat16, False, ["Sigmoid", "Mul"])):
            graph = torch.onnx.export(Silu(), (values.to(dtype),), dynamo=True, opset_version=18, optimize=False, custom_translation_table=NUMERICS.translations(torch) if fused else None).model_proto
            self.assertEqual([node.op_type for node in graph.graph.node], expected)
        self.assertEqual(int(source[0].view(torch.uint16)), 0)
        self.assertEqual(int(source[1].view(torch.uint16)), 0x8000)

    def test_proposal_refuses_heldout_unbound_or_nonfinite_metadata_and_never_accepts(self):
        fixture = {"split": "train", "schema_verified": True, "input_fixture": {"path": "/chosen/observation.npz", "sha256": "0" * 64}, "state_names": ["first", "second"]}
        proposal = DIAGNOSTIC.numerical_contract_proposal(fixture, {}, {}, evidence=[{"status": "passed"}])
        self.assertEqual(proposal["qualification_verdict"], "NO_ACCEPTANCE")
        self.assertFalse(proposal["thresholds_adopted"])
        self.assertFalse(proposal["acceptance_contract_selected"])
        self.assertFalse(proposal["source"]["fixture_binding_verified_by_producer"])
        self.assertFalse(proposal["source"]["source_closure_verified_by_producer"])
        self.assertEqual(proposal["existing_gate"]["rtol"], 1e-4)
        self.assertIsNone(proposal["action_quality"]["policy_coordinates"]["limits"]["rms_error"])
        self.assertFalse(proposal["claim_scoped_gaps"]["functional_compiler_checks"]["training_camera_or_backbone_identity_required"])
        proposal["source"]["fixture"]["state_names"].append("third")
        self.assertEqual(fixture["state_names"], ["first", "second"])
        for change in ({"split": "heldout"}, {"split": "synthetic"}, {"schema_verified": False}, {"input_fixture": None}, {"input_fixture": {"path": "/chosen/observation.npz", "sha256": "not a hash"}}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                DIAGNOSTIC.numerical_contract_proposal({**fixture, **change}, {}, {})
        with self.assertRaises(ValueError):
            DIAGNOSTIC.numerical_contract_proposal(fixture, {"epsilon": float("nan")}, {})

    def test_action_statistics_preserve_raw_signed_zero_encoding_distinction(self):
        import torch
        result = DIAGNOSTIC.statistics(torch.tensor([0.0]), torch.tensor([-0.0]), torch)
        self.assertEqual(result["exactly_unequal_count"], 0)
        self.assertEqual(result["raw_encoding_unequal_count"], 1)
        self.assertEqual(result["outside_existing_pointwise_comparison_count"], 0)

    def test_original_processor_provenance_binds_selected_config_state_and_classes(self):
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor import PolicyProcessorPipeline
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)), "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
        before = NormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
        after = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats={})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = DIAGNOSTIC.original_processor_diagnostic(root, root, torch, physical_dim=2)
            self.assertFalse(result["available"])
            (root / "stats.safetensors").write_bytes(b"selected bytes; mocked processor loading")
            for name in ("policy_preprocessor.json", "policy_postprocessor.json"):
                (root / name).write_text(json.dumps({"steps": [{"state_file": "stats.safetensors"}]}))
            with mock.patch.object(PolicyProcessorPipeline.__origin__, "from_pretrained", side_effect=[SimpleNamespace(steps=[before]), SimpleNamespace(steps=[after])]) as load:
                result = DIAGNOSTIC.original_processor_diagnostic(root, root, torch, physical_dim=2)
            self.assertEqual(load.call_count, 2)
            self.assertTrue(result["available"])
            self.assertTrue(result["bindings_unchanged"])
            bound = {Path(item["path"]).name for item in result["bindings"]}
            self.assertTrue({"policy_preprocessor.json", "policy_postprocessor.json", "stats.safetensors", "normalize_processor.py", "pipeline.py", "processor_smolvla.py"} <= bound)
            self.assertFalse(result["normalization_qualified"])
            self.assertTrue(result["action_probe_transform_is_identity"])

    def test_original_processor_namespaced_statistics_are_not_silently_remapped(self):
        import torch
        from lerobot.configs.types import FeatureType, NormalizationMode, PolicyFeature
        from lerobot.processor.normalize_processor import NormalizerProcessorStep, UnnormalizerProcessorStep
        features = {"observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)), "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))}
        norm_map = {FeatureType.STATE: NormalizationMode.MEAN_STD, FeatureType.ACTION: NormalizationMode.MEAN_STD}
        stats = {"so100.buffer.action": {"mean": [3.0, 7.0], "std": [2.0, 4.0]}}
        before = NormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        after = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats=stats)
        result = DIAGNOSTIC.processor_transform_diagnostic(SimpleNamespace(steps=[before]), SimpleNamespace(steps=[after]), torch, physical_dim=2)
        self.assertTrue(result["state_probe_transform_is_identity"])
        self.assertTrue(result["action_probe_transform_is_identity"])
        self.assertFalse(result["plain_action_statistics_present"])
        self.assertFalse(result["statistics_remapped"])
        self.assertEqual(sorted(after._tensor_stats), ["so100.buffer.action"])
        inventory = result["feature_statistics_inventory"]["postprocessor"]
        self.assertIsNone(inventory["features"]["action"]["effective_statistics_key"])
        self.assertEqual(inventory["statistics"]["so100.buffer.action"]["mean"]["shape"], [2])
        self.assertEqual(inventory["statistics"]["so100.buffer.action"]["mean"]["dtype"], "torch.float32")
        after = UnnormalizerProcessorStep(features=features, norm_map=norm_map, stats={"action": stats["so100.buffer.action"]})
        result = DIAGNOSTIC.processor_transform_diagnostic(SimpleNamespace(steps=[before]), SimpleNamespace(steps=[after]), torch, physical_dim=2)
        self.assertTrue(result["plain_action_statistics_present"])
        self.assertFalse(result["action_probe_transform_is_identity"])
        self.assertFalse(result["physical_action_units_qualified"])

    def test_action_diagnostic_separates_policy_coordinates_from_explicit_physical_units(self):
        import torch
        source = torch.zeros(1, 3, 2)
        candidate = source + torch.tensor([0.25, -0.5])
        result = DIAGNOSTIC.action_quality_diagnostic(source, candidate, torch)
        self.assertIsNone(result["physical"])
        self.assertEqual(len(result["policy_coordinates_by_coordinate"]), 2)
        self.assertEqual(len(result["policy_coordinates_by_horizon"]), 3)
        self.assertFalse(result["normalization_qualified"])
        self.assertFalse(result["acceptance_contract_selected"])
        self.assertFalse(result["model_quality_qualified"])
        self.assertFalse(result["robot_success_qualified"])
        result = DIAGNOSTIC.action_quality_diagnostic(source, candidate, torch, physical_transform=lambda value: value * torch.tensor([2.0, 10.0]) + 7.0, physical_units=["first coordinate units", "second coordinate units"])
        self.assertEqual(result["physical"]["by_coordinate"][0]["max_absolute_error"], 0.5)
        self.assertEqual(result["physical"]["by_coordinate"][1]["max_absolute_error"], 5.0)
        self.assertIn("mixed BF16/FP32", result["reference_forward_precision"])

    def test_action_diagnostic_rejects_missing_units_nonfinite_and_mutating_transforms(self):
        import torch
        source = torch.zeros(1, 2, 2)
        for kwargs in ({"physical_transform": lambda value: value}, {"physical_units": ["units", "units"]}, {"physical_transform": lambda value: value, "physical_units": ["units", " "]}):
            with self.subTest(kwargs=list(kwargs)), self.assertRaises(ValueError):
                DIAGNOSTIC.action_quality_diagnostic(source, source, torch, **kwargs)
        with self.assertRaises(ValueError):
            DIAGNOSTIC.action_quality_diagnostic(source, source + float("nan"), torch)
        rng = torch.random.get_rng_state().clone()
        def mutate(value):
            value.add_(1)
            torch.rand(1)
            return value
        with self.assertRaisesRegex(AssertionError, "mutated its inputs"):
            DIAGNOSTIC.action_quality_diagnostic(source, source, torch, physical_transform=mutate, physical_units=["units", "units"])
        self.assertEqual(source.count_nonzero().item(), 0)
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))

    def test_declared_integer_projection_bias_cast_and_source_preservation(self):
        import copy
        import torch
        source = torch.nn.Sequential(torch.nn.Linear(3, 2, bias=True, dtype=torch.bfloat16))
        with torch.no_grad():
            source[0].weight.copy_(torch.tensor([[0.25, -0.75, 0.5], [1.0, 0.125, -0.5]]))
            source[0].bias.copy_(torch.tensor([0.125, -0.25]))
        before = {name: value.clone() for name, value in source.state_dict().items()}
        target = copy.deepcopy(source)
        records, projections, recipe = FACTORY.install_integer_projections(target, torch)
        value = torch.tensor([[0.25, 1.5, -0.5], [0.0, 0.0, 0.0]], dtype=torch.bfloat16)
        output = target(value)
        check = FACTORY.check_integer_projection(projections["0"], value, output, torch)
        self.assertTrue(check["float_bias_and_output_cast_exact"])
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(projections["0"].integer.weight_kn.dtype, torch.int8)
        self.assertEqual(projections["0"].weight.dtype, torch.bfloat16)
        self.assertEqual(projections["0"].weight.numel(), 0)
        self.assertTrue(all(torch.equal(before[name], value) for name, value in source.state_dict().items()))
        self.assertEqual(records[0]["float_interface_dtype"], "torch.bfloat16")
        self.assertEqual(recipe["accumulator"], "int32")
        self.assertNotEqual(recipe["attention_and_state"], "float32_cpu")
        with self.assertRaisesRegex(ValueError, "original floating interface"):
            target(value.float())

    def test_linear_bound_handles_cancellation_and_one_declared_cast(self):
        import torch
        left = torch.tensor([[1.0, -1.0, 0.5], [0.25, 0.5, -0.125]], dtype=torch.bfloat16)
        weight = torch.tensor([[1.0, 1.0, 0.0], [0.5, -0.75, 0.25]], dtype=torch.bfloat16)
        bias = torch.tensor([0.0, 0.125], dtype=torch.bfloat16)
        result = NUMERICS.bf16_linear_error_diagnostic(left, weight, bias, torch)
        actual = (left.float() @ weight.float().T + bias.float()).bfloat16().double()
        self.assertTrue(torch.all((actual - result["higher_precision_value"]).abs()
                                  <= result["absolute_error_bound_to_fp64_diagnostic"]))
        self.assertEqual(result["higher_precision_value"][0, 0].item(), 0.0)
        self.assertGreater(result["fp32_arithmetic_error_bound"][0, 0].item(), 0.0)
        self.assertEqual(result["operation_bound"], 7)
        self.assertTrue(result["diagnostic_only"])
        self.assertFalse(result["acceptance_contract_selected"])

    def test_linear_diagnostic_rejects_dtype_changes_and_unbounded_overflow(self):
        import torch
        for left, weight in ((torch.ones(1, 2), torch.ones(1, 2, dtype=torch.bfloat16)),
                             (torch.full((1, 2), 1e30, dtype=torch.bfloat16),
                              torch.full((1, 2), 1e30, dtype=torch.bfloat16))):
            with self.subTest(left=left.dtype), self.assertRaises(ValueError):
                NUMERICS.bf16_linear_error_diagnostic(left, weight, None, torch)


class SmolVLABackendArgumentTests(unittest.TestCase):
    def test_invalid_seeds_attribution_and_existing_outputs_fail_early(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["--checkpoint-dir", str(root / "policy"), "--backbone-dir", str(root / "backbone"), "--output-dir", str(root / "new")]
            for seeds in (("0", "0"), ("-1",), (str(2**63),)):
                with self.subTest(seeds=seeds), self.assertRaises(SystemExit):
                    DIAGNOSTIC.parse_args(base + ["--seeds", *seeds])
            with self.assertRaises(SystemExit):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0", "--fixture-kind", "dataset"])
            with self.assertRaises(SystemExit):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0", "1", "--input-npz", str(root / "fixture.npz")])
            for attributes in ([], ["--fixture-kind", "synthetic"], ["--fixture-kind", "synthetic", "--input-source", " "]):
                with self.subTest(attributes=attributes), self.assertRaises(SystemExit):
                    DIAGNOSTIC.parse_args(base + ["--seeds", "0", "--input-npz", str(root / "fixture.npz"), *attributes])
            (root / "new").mkdir()
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root), self.assertRaises(FileExistsError):
                DIAGNOSTIC.parse_args(base + ["--seeds", "0"])
            alias = root / "alias"
            alias.symlink_to(root / "new", target_is_directory=True)
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root), self.assertRaises(FileExistsError):
                DIAGNOSTIC.parse_args(base[:4] + ["--output-dir", str(alias), "--seeds", "0"])

    def test_supplied_fixtures_require_truthful_explicit_attribution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = ["--checkpoint-dir", str(root / "policy"), "--backbone-dir", str(root / "backbone"), "--output-dir", str(root / "new"), "--seeds", "0", "--input-npz", str(root / "fixture.npz")]
            for kind in ("dataset", "synthetic"):
                with self.subTest(kind=kind), mock.patch("merlin.common.paths.artifacts_dir", return_value=root):
                    args = DIAGNOSTIC.parse_args(base + ["--fixture-kind", kind, "--input-source", "Attributed supplied development fixture"])
                self.assertEqual(args.fixture_kind, kind)
                self.assertEqual(args.input_source, "Attributed supplied development fixture")


class SmolVLABackendCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import numpy as np
        import torch
        cls.np, cls.torch = np, torch
        cls.original_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        cls.torch.set_num_threads(cls.original_threads)

    def fixture(self, *, mutate_cache=False, integer_linears=False):
        torch = self.torch

        class Backbone(torch.nn.Module):
            def forward(self, **kwargs):
                embeddings = kwargs["inputs_embeds"][0]
                cache = {index: {"key_states": embeddings.clone(), "value_states": embeddings.clone()} for index in range(2)}
                return (embeddings, None), cache

        class Source(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.config = SimpleNamespace(num_steps=10, num_vlm_layers=2, chunk_size=50, action_feature=SimpleNamespace(shape=[2]))
                self.vlm_with_expert = Backbone()
                self.register_buffer("constant", torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4) / 16)
                if integer_linears:
                    self.prefix_projection = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
                    self.flow_projection = torch.nn.Linear(4, 4, dtype=torch.float32)

            def embed_prefix(self, images, masks, tokens, language_mask, state):
                outputs = [torch.nn.functional.scaled_dot_product_attention(self.constant + image.mean(), self.constant, self.constant) for image in images]
                embedding = sum(outputs).reshape(1, 4, 4).bfloat16()
                if integer_linears:
                    embedding = self.prefix_projection(embedding)
                return embedding, torch.ones(1, 4, dtype=torch.bool), torch.zeros(1, 4, dtype=torch.bool)

            def denoise_step(self, *, x_t, prefix_pad_masks, past_key_values, timestep):
                if mutate_cache:
                    past_key_values[0]["key_states"].add_(1)
                flow = self.flow_projection(x_t) if integer_linears else x_t
                return flow * 0.2 + timestep[:, None, None] + past_key_values[0]["value_states"].float().mean()

            def sample_actions(self, images, masks, tokens, language_mask, state, noise):
                embedding, padding, _ = self.embed_prefix(images, masks, tokens, language_mask, state)
                _, cache = self.vlm_with_expert.forward(inputs_embeds=[embedding, None], fill_kv_cache=True)
                value = noise
                for index in range(10):
                    timestep = torch.tensor(1.0 + index * (-1.0 / 10)).expand(noise.shape[0])
                    velocity = self.denoise_step(x_t=value, prefix_pad_masks=padding, past_key_values=cache, timestep=timestep)
                    value = value + (-1.0 / 10) * velocity
                return value

        config = SimpleNamespace(resize_imgs_with_padding=(2, 2), tokenizer_max_length=3, max_state_dim=4, max_action_dim=4, chunk_size=50, robot_state_feature=SimpleNamespace(shape=[2]))
        return Source().eval(), DIAGNOSTIC.factory.fixture_inputs(config, 3, seed=7)

    def test_complete_capture_dispatch_noise_cache_and_repeatability(self):
        torch, np = self.torch, self.np
        model, inputs = self.fixture()
        state_before = DIAGNOSTIC.state_identity(model, torch)
        flags_before = DIAGNOSTIC.backend_flags(torch)
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory:
            runs = {}
            for backend in ("default", "math"):
                first, metadata = DIAGNOSTIC.run_source_policy(model, inputs, backend, 7, 0, directory, np, torch, profile=True)
                second, _ = DIAGNOSTIC.run_source_policy(model, inputs, backend, 7, 1, directory, np, torch)
                self.assertTrue(all(torch.equal(first[name], second[name]) for name in first))
                self.assertEqual(first["final_actions"].shape, (1, 50, 2))
                self.assertEqual(len(metadata["prefix_cache_checks"]), 10)
                self.assertTrue(all(item["before"] == item["after"] for item in metadata["prefix_cache_checks"]))
                self.assertEqual(metadata["captures"]["step_00_input_state"], metadata["input_bindings"]["noise"])
                self.assertEqual(metadata["attention_profile"]["native_sdpa_calls_by_stage"], dict(zip(DIAGNOSTIC.SCOPES, (3, 0, 0))))
                self.assertTrue(metadata["temporary_wrappers_restored"])
                with np.load(metadata["output_file"]["path"], allow_pickle=False) as saved:
                    self.assertEqual(saved["prefix_key_cache"].dtype, np.uint16)
                    for index in range(10):
                        np.testing.assert_array_equal(saved[f"step_{index:02d}_output_state"], saved[f"step_{index:02d}_input_state"] + (-1.0 / 10) * saved[f"step_{index:02d}_velocity"])
                runs[backend] = metadata
            self.assertIn("aten::_scaled_dot_product_flash_attention_for_cpu", runs["default"]["attention_profile"]["operators"])
            self.assertIn("aten::_scaled_dot_product_attention_math", runs["math"]["attention_profile"]["operators"])
        self.assertEqual(DIAGNOSTIC.state_identity(model, torch), state_before)
        self.assertEqual(DIAGNOSTIC.backend_flags(torch), flags_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))

    def test_cache_failure_restores_context_hooks_and_rng(self):
        torch = self.torch
        model, inputs = self.fixture(mutate_cache=True)
        flags_before = DIAGNOSTIC.backend_flags(torch)
        rng_before = torch.random.get_rng_state().clone()
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(AssertionError, "mutated the prefix cache"):
            DIAGNOSTIC.run_source_policy(model, inputs, "math", 7, 0, directory, self.np, torch)
        self.assertEqual(DIAGNOSTIC.backend_flags(torch), flags_before)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), rng_before))
        self.assertNotIn("embed_prefix", model.__dict__)
        self.assertNotIn("denoise_step", model.__dict__)
        self.assertNotIn("forward", model.vlm_with_expert.__dict__)

    def test_velocity_replay_separates_prefix_from_state_feedback(self):
        torch = self.torch
        model, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            captures, _ = DIAGNOSTIC.run_source_policy(model, inputs, "default", 7, 0,
                                                       directory, self.np, torch)
        original = {name: DIAGNOSTIC.tensor_identity(value, torch) for name, value in captures.items()}
        observed_states = []

        def candidate(**kwargs):
            observed_states.append(kwargs["x_t"].clone())
            return model.denoise_step(**kwargs) + 0.125

        _, shared = DIAGNOSTIC.compare_identical_state_velocities(
            model.denoise_step, candidate, captures, torch, expected_steps=10)
        self.assertTrue(shared["identical_prefix"])
        self.assertEqual(shared["steps"], 10)
        for index, value in enumerate(observed_states):
            self.assertTrue(torch.equal(value, captures[f"step_{index:02d}_input_state"]))
        alternate = (captures["prefix_pad_mask"], captures["prefix_key_cache"],
                     captures["prefix_value_cache"] + 0.25)
        _, conditioned = DIAGNOSTIC.compare_identical_state_velocities(
            model.denoise_step, model.denoise_step, captures, torch,
            expected_steps=10, candidate_prefix=alternate)
        self.assertFalse(conditioned["identical_prefix"])
        self.assertTrue(all(row["comparison"]["max_absolute_error"] > 0.2
                            for row in conditioned["comparisons"]))
        self.assertEqual(original, {name: DIAGNOSTIC.tensor_identity(value, torch)
                                    for name, value in captures.items()})
        self.assertFalse(shared["acceptance_contract_selected"])

    def test_velocity_replay_refuses_incomplete_or_mutating_candidates(self):
        torch = self.torch
        model, inputs = self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            captures, _ = DIAGNOSTIC.run_source_policy(model, inputs, "default", 7, 0,
                                                       directory, self.np, torch)
        incomplete = dict(captures)
        del incomplete["step_09_input_state"]
        with self.assertRaisesRegex(ValueError, "complete contiguous"):
            DIAGNOSTIC.compare_identical_state_velocities(
                model.denoise_step, model.denoise_step, incomplete, torch, expected_steps=10)
        rng_before = torch.random.get_rng_state().clone()

        def mutation(**kwargs):
            kwargs["past_key_values"][0]["key_states"].add_(1)
            torch.rand(1)
            return model.denoise_step(**kwargs)

        with self.assertRaisesRegex(AssertionError, "mutated state/mask/cache"):
            DIAGNOSTIC.compare_identical_state_velocities(
                model.denoise_step, mutation, captures, torch, expected_steps=10)
        self.assertTrue(torch.equal(rng_before, torch.random.get_rng_state()))
        altered_mask = (torch.zeros_like(captures["prefix_pad_mask"]),
                        captures["prefix_key_cache"], captures["prefix_value_cache"])
        with self.assertRaisesRegex(ValueError, "routing mask"):
            DIAGNOSTIC.compare_identical_state_velocities(
                model.denoise_step, model.denoise_step, captures, torch,
                expected_steps=10, candidate_prefix=altered_mask)

    def test_complete_integer_policy_keeps_source_and_tracks_stage_placement(self):
        import copy
        torch = self.torch
        source, inputs = self.fixture(integer_linears=True)
        before = DIAGNOSTIC.state_identity(source, torch)
        candidate = copy.deepcopy(source)
        records, projections, recipe = DIAGNOSTIC.factory.install_integer_projections(candidate, torch)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("merlin.common.paths.artifacts_dir", return_value=root):
                result, receipt = DIAGNOSTIC.factory.run_integer_policy_diagnostic(
                    source, candidate, inputs, records, projections, root / "integer",
                    seed=7, fixture_kind="synthetic", recipe=recipe, torch=torch)
            self.assertEqual(result["denoising_steps"], 10)
            self.assertEqual(result["executed_projection_count"], 2)
            by_name = {record["name"]: record for record in result["projection_inventory"]}
            self.assertEqual(by_name["prefix_projection"]["calls_by_stage"], {"prefix_encode": 1})
            self.assertEqual(by_name["flow_projection"]["calls_by_stage"], {"flow_denoise": 10})
            self.assertTrue(all(record["integer_operator_reference"]["passed"]
                                for record in result["projection_inventory"]))
            self.assertTrue(result["source_mask_routing_exact"])
            for replay in result["identical_state_velocity_replay"].values():
                self.assertEqual(replay["steps"], 10)
                self.assertTrue(replay["reference_replay_exact"])
                self.assertTrue(replay["inputs_cache_rng_unchanged"])
            self.assertFalse(result["model_quality_qualified"])
            self.assertFalse(result["active_gate_changed"])
            self.assertFalse(result["device_execution"])
            self.assertTrue(Path(receipt["path"]).is_file())
            memory = result["memory"]
            self.assertEqual(memory["host_two_model_diagnostic_constants_bytes"],
                             memory["source_constants_bytes"] + memory["target_constants_bytes"])
        self.assertEqual(before, DIAGNOSTIC.state_identity(source, torch))

    def test_stride_zero_hash_and_signed_bf16_distances(self):
        torch = self.torch
        singleton = torch.tensor(0.5).expand(1)
        self.assertEqual(DIAGNOSTIC.tensor_identity(singleton, torch), DIAGNOSTIC.tensor_identity(torch.tensor([0.5]), torch))
        default = torch.tensor([0.0, 1.0], dtype=torch.bfloat16)
        alternate = torch.tensor([-0.0, 1.0078125], dtype=torch.bfloat16)
        result = DIAGNOSTIC.statistics(default, alternate, torch)
        self.assertEqual(result["bf16_representable_step_distance"]["max"], 1)
        self.assertEqual(result["exactly_unequal_count"], 1)
        expanded_default = torch.tensor(1.0, dtype=torch.bfloat16).expand(1)
        expanded_alternate = torch.tensor(1.0078125, dtype=torch.bfloat16).expand(1)
        expanded_result = DIAGNOSTIC.statistics(expanded_default, expanded_alternate, torch)
        self.assertEqual(expanded_result["bf16_representable_step_distance"]["max"], 1)
        self.assertEqual(DIAGNOSTIC.dense_flat_copy(expanded_default, torch).stride(), (1,))
        self.assertEqual((DIAGNOSTIC.RTOL, DIAGNOSTIC.ATOL), (1e-4, 1e-4))


if __name__ == "__main__":
    unittest.main()
