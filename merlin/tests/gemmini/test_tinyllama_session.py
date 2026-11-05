"""Fail-closed artifact and workload gates; no checkpoint substitute is executed."""

import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

from merlin.common.paths import repo_root


SOURCE = repo_root() / "examples/gemmini/comparisons/tvm/tinyllama_session.py"
sys.path.insert(0, str(SOURCE.parent))
SPEC = importlib.util.spec_from_file_location("tinyllama_session_tested", SOURCE)
FACTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FACTORY)


class TinyLlamaArtifactTests(unittest.TestCase):
    def write_artifacts(self, root, config=None):
        for name in ("model.safetensors", "tokenizer.json", "tokenizer_config.json",
                     "special_tokens_map.json"):
            (root / name).write_text("fixture only; never a model")
        (root / "config.json").write_text(json.dumps(
            config or {**FACTORY.MODEL_SEMANTICS, "torch_dtype": "bfloat16"}))

    def pinned_identity(self, path, validator):
        path = validator(path)
        return {"path": str(path), "sha256": FACTORY.WEIGHT_SHA256}

    def test_absent_checkpoint_fails_before_framework_imports(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(FileNotFoundError):
            FACTORY.checkpoint_artifacts(directory)

    def test_nonofficial_weights_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root)
            with self.assertRaisesRegex(ValueError, "pinned official full checkpoint"):
                FACTORY.checkpoint_artifacts(root)

    def test_truncated_geometry_is_rejected_after_identity_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_artifacts(root, {**FACTORY.MODEL_SEMANTICS, "num_hidden_layers": 2,
                                        "torch_dtype": "bfloat16"})
            with mock.patch.object(FACTORY, "file_identity", self.pinned_identity), self.assertRaisesRegex(ValueError, "complete pinned model geometry"):
                FACTORY.checkpoint_artifacts(root)

    def test_precision_and_recurrence_require_explicit_valid_choices(self):
        base = {"checkpoint_dir": "/missing", "model2mlir_root": "/missing",
                "corpus_path": "/missing"}
        for changes in ({"precision": "float16"}, {"prefill_tokens": 0},
                        {"decode_tokens": 1}, {"cpu_threads": 32},
                        {"prefill_tokens": 2048, "decode_tokens": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**{**base, **changes})

    def test_quantization_and_language_workload_fail_closed(self):
        base = {"checkpoint_dir": "/missing", "model2mlir_root": "/missing",
                "corpus_path": "/missing"}
        for changes in ({"quantized": 1}, {"quantized": True, "precision": "bfloat16"},
                        {"quality_corpus_path": "/missing"}, {"workload": "other"},
                        {"workload": "development_language"},
                        {"workload": "synthetic_feasibility", "prefill_tokens": 9},
                        {"activation_scale": "other"}, {"arithmetic_policy": "other"},
                        {"arithmetic_policy": "fixed_tree_fp32_v1"},
                        {"arithmetic_policy": "fixed_tree_fp32_v1", "quantized": True, "precision": "bfloat16"},
                        {"arithmetic_policy": "fixed_tree_fp32_v2_pv"},
                        {"arithmetic_policy": "fixed_tree_fp32_v2_pv", "quantized": True, "precision": "bfloat16"},
                        {"arithmetic_policy": "fixed_tree_fp32_v3_softmax_reduce"},
                        {"rotary_policy": "unknown"}, {"rotary_policy": "source_table_v1", "precision": "bfloat16"},
                        {"silu_export_policy": "unknown"}, {"silu_export_policy": "torch_avx512_sleef_v1"},
                        {"silu_export_policy": "torch_avx512_sleef_v1", "quantized": True, "cpu_threads": 1},
                        {"silu_export_policy": "torch_avx512_sleef_v1", "quantized": True, "precision": "bfloat16"},
                        {"rotary_policy": "source_table_v1", "prefill_tokens": 159, "decode_tokens": 2},
                        {"arithmetic_policy": "fixed_tree_fp32_v2_pv", "quantized": True, "prefill_tokens": 159, "decode_tokens": 2}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                FACTORY.get_model_and_inputs(**{**base, **changes})

    def test_restricted_artifact_path_is_rejected_before_read(self):
        with self.assertRaisesRegex(ValueError, "Restricted path component"):
            FACTORY.checkpoint_artifacts("/tmp/" + "ham" + "mer" + "-fixture")


try:
    import torch
    import numpy as np
    import tinyllama_integer as INTEGER
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "Optional PyTorch environment is absent")
class TinyLlamaIntegerTests(unittest.TestCase):
    def test_source_silu_export_contract_refuses_unbound_native_dispatch(self):
        from types import SimpleNamespace
        fake = SimpleNamespace(version=SimpleNamespace(git_version="449b1768410104d3ed79d3bcfe4ba1d65c7f22c0"), backends=SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: "AVX512")), get_num_threads=lambda: 2)
        self.assertIsNone(INTEGER.source_silu_contract(fake))
        artifacts = {name: {"path": "/tmp/fixture", "sha256": sha} for name, sha in INTEGER.SOURCE_SILU_ARTIFACT_HASHES.items()}
        with mock.patch.object(INTEGER, "_source_silu_artifacts", return_value=artifacts):
            self.assertEqual(INTEGER.source_silu_contract(fake, "torch_avx512_sleef_v1")["native_loop"], "paired_16_word_vectors;admitted_contiguous_chunks_divisible_by32;scalar_std_exp_tail_not_admitted")
        with mock.patch.object(INTEGER, "_source_silu_artifacts", return_value={name: {**record, "sha256": "unproved"} for name, record in artifacts.items()}), self.assertRaisesRegex(ValueError, "independently proved"):
            INTEGER.source_silu_contract(fake, "torch_avx512_sleef_v1")
        for field, value in (("git", "other"), ("cpu", "AVX2"), ("threads", 1)):
            changed = SimpleNamespace(version=SimpleNamespace(git_version=value if field == "git" else fake.version.git_version), backends=SimpleNamespace(cpu=SimpleNamespace(get_cpu_capability=lambda: value if field == "cpu" else "AVX512")), get_num_threads=lambda: value if field == "threads" else 2)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "pinned PyTorch"):
                INTEGER.source_silu_contract(changed, "torch_avx512_sleef_v1")

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX") and os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit modern exporter and LLVM host build")
    def test_source_silu_opt_in_export_preserves_fullshape_native_in_both_modes(self):
        import onnx
        import tvm
        import verify_session
        threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            class Stage(torch.nn.Module):
                def forward(self, value):
                    return torch.nn.functional.silu(value)
            model = Stage().eval()
            value = torch.linspace(-80, 80, 5632, dtype=torch.float32).reshape(1, 1, 5632)
            expected = model(value).detach().numpy()
            state = {name: tensor.clone() for name, tensor in model.state_dict().items()}
            with tempfile.TemporaryDirectory() as directory:
                for mode in ("baseline", "optimized"):
                    options = INTEGER.export_options(torch, silu_export_policy="torch_avx512_sleef_v1")
                    (Path(directory) / mode).mkdir()
                    compiler = verify_session.StageCompiler(Path(directory) / mode, mode, np, torch, onnx, tvm, exporter="dynamo", export_options=options)
                    inputs = (value.numpy(),)
                    run = compiler(type("Stage", (), {"name": "source_silu", "module": model})(), inputs, 1)
                    actual = verify_session.outputs(run(inputs), np, torch)[0]
                    graph = onnx.load(str(Path(directory) / mode / "stage_0/model.onnx"))
                    self.assertEqual([(node.domain, node.op_type) for node in graph.graph.node], [("merlin.source_preserving", "TorchSiluAvx512F32")])
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
            self.assertEqual(set(model.state_dict()), set(state))
            self.assertNotIn(torch.ops.aten.silu.default, INTEGER.export_options(torch)["custom_translation_table"])
        finally:
            torch.set_num_threads(threads)

    def test_v3_softmax_independent_tree_mask_tail_and_backend_exp_contract(self):
        import decimal
        logits = torch.tensor([[[[0., -1., -80., -100.], [1., 1., 1., 1.], [-0., 0., 2., 3.]]]], dtype=torch.float32)
        mask = torch.tensor([[[[0., 0., 0., 0.], [0., 0., torch.finfo(torch.float32).min, torch.finfo(torch.float32).min], [0., torch.finfo(torch.float32).min, torch.finfo(torch.float32).min, torch.finfo(torch.float32).min]]]])
        active = mask.numpy() == 0
        shifted = np.where(active, logits.numpy() - np.max(np.where(active, logits.numpy(), np.finfo(np.float32).min), axis=-1, keepdims=True), np.float32(0))
        exponent = np.where(active, torch.exp(torch.from_numpy(shifted)).numpy(), np.float32(0))
        leaves = np.pad(exponent, ((0, 0), (0, 0), (0, 0), (0, 252)))
        while leaves.shape[-1] > 1: leaves = (leaves[..., 0::2] + leaves[..., 1::2]).astype(np.float32)
        expected = exponent / leaves
        actual = INTEGER.fixed_tree_softmax(logits, mask).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        self.assertTrue(np.all(actual[~active].view(np.uint32) == 0))
        self.assertEqual(float(actual[0, 0, 2, 0]), 1.)
        changed = logits.clone(); changed[torch.from_numpy(~active)] = 1000.
        self.assertTrue(torch.equal(INTEGER.fixed_tree_softmax(changed, mask), torch.from_numpy(actual)))
        for value in (0., -1., -80., -100.):
            with decimal.localcontext() as context:
                context.prec = 80
                exact = np.float32(float(decimal.Decimal.from_float(value).exp()))
            candidate = torch.exp(torch.tensor(value, dtype=torch.float32)).numpy()
            self.assertLessEqual(abs(int(candidate.view(np.uint32)) - int(exact.view(np.uint32))), 1)
        contract = INTEGER.arithmetic_contract("fixed_tree_fp32_v3_softmax_reduce")
        self.assertNotEqual(contract["softmax_native_exp"], contract["softmax_compiled_exp"])
        self.assertIn("not_bit_deterministic", contract["softmax_exp_equivalence"])
        self.assertEqual(INTEGER.arithmetic_contract("fixed_tree_fp32_v2_pv"), INTEGER.FIXED_TREE_PV_POLICY)

    def test_v3_full_last_static_softmax_rows_and_invalid_admission(self):
        logits = torch.linspace(-8, 0, 128).reshape(1, 1, 1, 128).expand(1, 2, 3, 128).clone()
        mask = torch.zeros((1, 1, 3, 128), dtype=torch.float32)
        mask[..., 0, 64:] = torch.finfo(torch.float32).min
        full = INTEGER.fixed_tree_softmax(logits, mask)
        last = INTEGER.fixed_tree_softmax(logits[..., -1:, :], mask[..., -1:, :])
        static = INTEGER.fixed_tree_softmax(torch.nn.functional.pad(logits, (0, 32)), torch.nn.functional.pad(mask, (0, 32), value=torch.finfo(torch.float32).min))
        self.assertTrue(torch.equal(full[..., -1:, :].view(torch.int32), last.view(torch.int32)))
        self.assertTrue(torch.equal(full.view(torch.int32), static[..., :128].contiguous().view(torch.int32)))
        self.assertFalse(bool(static[..., 128:].view(torch.int32).any()))
        for bad in (torch.full_like(mask, torch.finfo(torch.float32).min), torch.ones_like(mask), torch.full_like(mask, float("nan")), torch.full_like(mask, float("-inf"))):
            with self.assertRaises(ValueError): INTEGER.fixed_tree_softmax(logits, bad)
        for bad in (logits.double(), torch.full_like(logits, float("nan")), torch.ones((1, 2, 1, 161))):
            with self.assertRaises(ValueError): INTEGER.fixed_tree_softmax(bad, mask)

    def test_v3_invalid_mask_precedes_cache_update_and_projections(self):
        from types import SimpleNamespace
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
        config = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        candidate = INTEGER.FixedTreeAttention(LlamaAttention(config, 0).eval(), "fixed_tree_fp32_v3_softmax_reduce").eval()
        cache = SimpleNamespace(layers=[SimpleNamespace(max_cache_len=160, keys=torch.zeros((1, 2, 160, 16)), values=torch.zeros((1, 2, 160, 16)))], update=mock.Mock())
        for invalid in (torch.full((1, 1, 1, 160), torch.finfo(torch.float32).min), torch.ones((1, 1, 1, 160))):
            with mock.patch.object(candidate.q_proj, "forward", side_effect=AssertionError("projection must not run")), self.assertRaisesRegex(ValueError, "zero/minimum masks"):
                candidate(torch.ones((1, 1, 64)), (torch.ones((1, 1, 16)), torch.zeros((1, 1, 16))), invalid, past_key_values=cache)
        cache.update.assert_not_called()

    def test_source_rotary_table_preserves_source_words_and_rejects_positions(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        original = LlamaRotaryEmbedding(LlamaConfig(hidden_size=64, num_attention_heads=4)).eval()
        table = INTEGER.SourceRotaryTable(original).eval()
        parameters = {name: value.clone() for name, value in original.named_buffers()}
        for selected in (torch.arange(8), torch.arange(128), torch.arange(160), torch.tensor([0, 127, 159])):
            positions = selected.reshape(1, -1); value = torch.zeros((1, len(selected), 64))
            for got, wanted in zip(table(value, positions), original(value, positions)):
                self.assertTrue(torch.equal(got.view(torch.int32), wanted.view(torch.int32)))
        for name, value in original.named_buffers(): self.assertTrue(torch.equal(value, parameters[name]))
        for selected in (torch.tensor([[-1]]), torch.tensor([[160]]), torch.tensor([[1.]])):
            with self.assertRaises(ValueError): table(torch.zeros((1, 1, 64)), selected)
        original.config.rope_parameters["rope_theta"] = 5000.
        with self.assertRaises(ValueError): INTEGER.SourceRotaryTable(original)

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODERN_ONNX") and os.environ.get("TVM_LIBRARY_PATH"), "Requires explicit modern exporter and LLVM host build")
    def test_v3_softmax_and_source_rotary_independent_and_combined_export(self):
        import onnx, tvm, verify_session
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        table = INTEGER.SourceRotaryTable(LlamaRotaryEmbedding(LlamaConfig(hidden_size=64, num_attention_heads=4)).eval()).eval()
        class Candidate(torch.nn.Module):
            def __init__(self, kind):
                super().__init__(); self.kind = kind; self.rope = table
            def forward(self, logits, mask, positions):
                if self.kind == "softmax": return INTEGER.fixed_tree_softmax(logits, mask)
                cos, sin = self.rope(logits, positions)
                if self.kind == "rotary": return cos, sin
                return INTEGER.fixed_tree_softmax(logits, mask), cos, sin
        logits = torch.linspace(-3., 3., 24).reshape(1, 2, 3, 4); mask = torch.zeros((1, 1, 3, 4)); mask[..., 0, 1:] = torch.finfo(torch.float32).min
        positions = torch.tensor([[0, 127, 159]], dtype=torch.int64)
        for kind in ("softmax", "rotary", "both"):
            module = Candidate(kind).eval(); expected = verify_session.outputs(module(logits, mask, positions), np, torch)
            for mode in ("baseline", "optimized"):
                with tempfile.TemporaryDirectory() as directory:
                    compiler = verify_session.StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", export_options=INTEGER.export_options(torch, "fixed_tree_fp32_v3_softmax_reduce"))
                    inputs = tuple(value.numpy() for value in (logits, mask, positions)); run = compiler(type("Stage", (), {"name": kind, "module": module})(), inputs, len(expected))
                    actual = verify_session.outputs(run(inputs), np, torch)
                    for got, wanted in zip(actual, expected): verify_session.compare(got, wanted, np, 1e-4, 1e-4)
                    if kind != "softmax":
                        for got, wanted in zip(actual[-2:], expected[-2:]): np.testing.assert_array_equal(got.view(np.uint32), wanted.view(np.uint32))
                    graph = onnx.load(str(Path(directory) / "stage_0/model.onnx"))
                    self.assertFalse(any(node.op_type in ("Cos", "Sin") for node in graph.graph.node))
                    if kind != "rotary": self.assertEqual(sum(node.op_type == "FixedTreeDot" for node in graph.graph.node), 1)

    @staticmethod
    def scalar_tree_dot(left, right):
        output = np.empty((*left.shape[:2], left.shape[-2], right.shape[-2]), dtype=np.float32)
        for batch, head, row, column in np.ndindex(output.shape):
            terms = [np.float32(np.float32(a) * np.float32(b)) for a, b in zip(left[batch, head, row], right[batch, head, column])]
            while len(terms) > 1:
                terms = [np.float32(terms[index] + terms[index + 1]) for index in range(0, len(terms), 2)]
            output[batch, head, row, column] = terms[0]
        return output

    def test_fixed_tree_native_has_independent_order_and_no_fma_oracle(self):
        generator = np.random.default_rng(194)
        left = generator.uniform(-4, 4, (1, 2, 7, 64)).astype(np.float32)
        right = generator.uniform(-4, 4, (1, 2, 9, 64)).astype(np.float32)
        expected = self.scalar_tree_dot(left, right)
        actual = INTEGER.fixed_tree_dot(torch.from_numpy(left), torch.from_numpy(right)).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(INTEGER.fixed_tree_dot(torch.from_numpy(left[..., -1:, :]), torch.from_numpy(right)).numpy(), actual[..., -1:, :])
        padded = np.pad(right, ((0, 0), (0, 0), (0, 3), (0, 0)))
        np.testing.assert_array_equal(INTEGER.fixed_tree_dot(torch.from_numpy(left), torch.from_numpy(padded)).numpy()[..., :9], actual)
        ordered_left = torch.tensor([1e8, 1., -1e8, 1.]).reshape(1, 1, 1, 4)
        ordered_right = torch.ones_like(ordered_left)
        self.assertEqual(float(INTEGER.fixed_tree_dot(ordered_left, ordered_right)), 0.)
        ascending = np.float32(0)
        for value in ordered_left.numpy().flat:
            ascending = np.float32(ascending + value)
        self.assertEqual(float(ascending), 1.)
        epsilon = np.float32(2**-23)
        no_fma_left = np.array([np.float32(1 + epsilon), -1], dtype=np.float32).reshape(1, 1, 1, 2)
        no_fma_right = np.array([np.float32(1 - epsilon), 1], dtype=np.float32).reshape(1, 1, 1, 2)
        self.assertEqual(float(INTEGER.fixed_tree_dot(torch.from_numpy(no_fma_left), torch.from_numpy(no_fma_right))), 0.)
        self.assertNotEqual(float(np.float32(np.sum(no_fma_left.astype(np.float64) * no_fma_right.astype(np.float64)))), 0.)
        for shape in ((2048,), (1, 3, 2048)):
            value = generator.uniform(-1, 1, shape).astype(np.float32)
            expected_mean = self.scalar_tree_dot(value.reshape(1, 1, -1, 2048), value.reshape(1, 1, -1, 2048))
            expected_mean = np.diagonal(expected_mean[0, 0]).reshape(*shape[:-1], 1) * np.float32(1 / 2048)
            actual_mean = INTEGER.fixed_tree_mean_squares(torch.from_numpy(value)).numpy()
            np.testing.assert_array_equal(actual_mean.view(np.uint32), expected_mean.view(np.uint32))

    def test_fixed_tree_rejects_invalid_operands_without_mutation(self):
        value = torch.ones((1, 1, 2, 8))
        before = value.clone()
        for bad in (value.double(), value[..., :3], torch.empty((1, 1, 0, 8)), value + float("inf"), torch.full_like(value, torch.finfo(torch.float32).max)):
            with self.subTest(shape=tuple(bad.shape), dtype=bad.dtype), self.assertRaises(ValueError):
                INTEGER.fixed_tree_mean_squares(bad)
        with self.assertRaisesRegex(ValueError, "matching"):
            INTEGER.fixed_tree_dot(value, torch.ones((1, 2, 2, 8)))
        self.assertTrue(torch.equal(value, before))
        self.assertEqual(INTEGER.arithmetic_contract()["name"], "native")
        self.assertNotIn("arithmetic_policy", INTEGER.recipe("power_of_two"))
        self.assertEqual(INTEGER.recipe("power_of_two", "fixed_tree_fp32_v1")["arithmetic_policy"]["status"], "opt_in_candidate_not_adopted")

    def test_fixed_tree_installer_preserves_norm_parameters_and_source_interface(self):
        from types import SimpleNamespace
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention, eager_attention_forward
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        lm = torch.nn.Module()
        lm.model = torch.nn.Module()
        lm.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(22)])
        config = LlamaConfig(hidden_size=2048, intermediate_size=5632, num_hidden_layers=22, num_attention_heads=32, num_key_value_heads=4)
        config._attn_implementation = "eager"
        for index, layer in enumerate(lm.model.layers):
            layer.input_layernorm = LlamaRMSNorm(2048, 1e-5)
            layer.post_attention_layernorm = LlamaRMSNorm(2048, 1e-5)
            with torch.device("meta"):
                layer.self_attn = LlamaAttention(config, index)
        lm.model.norm = LlamaRMSNorm(2048, 1e-5)
        lm.config = config
        lm.eval()
        parameters = {name: value for name, value in lm.named_parameters()}
        metadata = INTEGER.install_fixed_tree_arithmetic(lm)
        self.assertEqual(metadata["rms_norms"], 45)
        self.assertEqual(lm.config._attn_implementation, "eager")
        self.assertTrue(all(isinstance(layer.self_attn, INTEGER.FixedTreeAttention) for layer in lm.model.layers))
        self.assertTrue(all(not module.training for module in lm.modules()))
        self.assertIs(ALL_ATTENTION_FUNCTIONS.get_interface("eager", eager_attention_forward), eager_attention_forward)
        for name, value in lm.named_parameters():
            self.assertIs(value, parameters[name])

    def test_fixed_tree_missing_mask_fails_before_caller_state_changes(self):
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
        config = LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2)
        original = LlamaAttention(config, 0).eval()
        candidate = INTEGER.FixedTreeAttention(original).eval()
        value = torch.ones((1, 1, 64))
        before = value.clone()
        cache = mock.Mock()
        parameters = {name: parameter.detach().clone() for name, parameter in candidate.named_parameters()}
        with self.assertRaisesRegex(ValueError, "explicit causal attention mask"):
            candidate(value, (torch.ones((1, 1, 16)), torch.zeros((1, 1, 16))), past_key_values=cache)
        cache.update.assert_not_called()
        self.assertTrue(torch.equal(value, before))
        for name, parameter in candidate.named_parameters():
            self.assertTrue(torch.equal(parameter, parameters[name]))

    def test_fixed_tree_import_is_lazy_and_alias_registration_is_idempotent(self):
        import subprocess
        script = """import importlib.util, sys, torch
from pathlib import Path
path=Path(sys.argv[1]);modules=[]
assert not hasattr(torch.ops.merlin_fixed_order,'dot')
for name in ('fixed_tree_import_one','fixed_tree_import_alias'):
 spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);modules.append(module)
assert not hasattr(torch.ops.merlin_fixed_order,'dot')
assert not hasattr(torch.ops.merlin_fixed_order,'mean_squares')
assert modules[0].recipe()['name']=='tinyllama_w8a8_dynamic_token_v1'
value=torch.ones((1,1,1,8));expected=torch.tensor([[[[8.]]]])
for module in modules:
 assert torch.equal(module.fixed_tree_dot(value,value),expected)
 assert torch.equal(module.fixed_tree_mean_squares(value),torch.ones_like(value[...,:1]))
assert modules[0]._FIXED_TREE_OPS==modules[1]._FIXED_TREE_OPS
"""
        result = subprocess.run([sys.executable, "-c", script, str(Path(INTEGER.__file__))], capture_output=True, text=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_v2_pv_common_capacity_full_last_cached_has_independent_raw_oracle(self):
        generator = np.random.default_rng(20261012)
        probabilities = generator.uniform(0, 1, (1, 2, 5, 139)).astype(np.float32)
        values = generator.uniform(-1, 1, (1, 2, 139, 4)).astype(np.float32)
        padded_p = np.pad(probabilities, [(0, 0)] * 3 + [(0, 117)])
        padded_v = np.pad(values.transpose(0, 1, 3, 2), [(0, 0)] * 3 + [(0, 117)])
        expected = self.scalar_tree_dot(padded_p, padded_v)
        source_p, source_v = torch.from_numpy(probabilities), torch.from_numpy(values)
        before_p, before_v = source_p.clone(), source_v.clone()
        actual = INTEGER.fixed_tree_pv(source_p, source_v).numpy()
        last = INTEGER.fixed_tree_pv(source_p[..., -1:, :], source_v).numpy()
        cached = INTEGER.fixed_tree_pv(torch.nn.functional.pad(source_p[..., -1:, :], (0, 21)), torch.nn.functional.pad(source_v, (0, 0, 0, 21))).numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        np.testing.assert_array_equal(last.view(np.uint32), actual[..., -1:, :].view(np.uint32))
        np.testing.assert_array_equal(cached.view(np.uint32), last.view(np.uint32))
        self.assertTrue(torch.equal(source_p, before_p) and torch.equal(source_v, before_v))
        wrong = expected.copy(); wrong.view(np.uint32).flat[0] ^= np.uint32(1)
        with self.assertRaises(AssertionError):
            np.testing.assert_array_equal(actual.view(np.uint32), wrong.view(np.uint32))
        zero = INTEGER.fixed_tree_pv(torch.full((1, 1, 1, 2), -0.0), torch.ones((1, 1, 2, 1)))
        self.assertEqual(int(zero.numpy().view(np.uint32).item()), 0)

    def test_v2_pv_capacity_shape_nonfinite_controls_preserve_inputs(self):
        p = torch.ones((1, 1, 2, 11)); v = torch.ones((1, 1, 11, 4))
        before_p, before_v = p.clone(), v.clone()
        for bad_p, bad_v in ((p.double(), v), (p, v.double()), (p, v[..., :10, :]),
                             (torch.ones((1, 1, 161, 11)), v), (torch.ones((1, 1, 1, 161)), torch.ones((1, 1, 161, 4))),
                             (p + float("nan"), v), (p, v + float("inf"))):
            with self.subTest(p_shape=bad_p.shape, v_shape=bad_v.shape), self.assertRaises(ValueError):
                INTEGER.fixed_tree_pv(bad_p, bad_v)
        self.assertTrue(torch.equal(p, before_p) and torch.equal(v, before_v))
        contract = INTEGER.arithmetic_contract("fixed_tree_fp32_v2_pv")
        self.assertEqual((contract["pv_capacity"], contract["pv_tree_k"]), (160, 256))
        self.assertEqual(contract["version"], 1)
        self.assertEqual(INTEGER.arithmetic_contract("fixed_tree_fp32_v1")["attention_scope"], "qk_only_scaling_mask_softmax_and_pv_unchanged")
        self.assertNotIn("arithmetic_policy", INTEGER.recipe())
        self.assertEqual(INTEGER.recipe("power_of_two", "fixed_tree_fp32_v2_pv")["arithmetic_policy"]["pv_capacity"], 160)

    def test_v2_capacity_rejection_precedes_cache_and_parameter_mutation(self):
        from types import SimpleNamespace
        from transformers import LlamaConfig
        from transformers.models.llama.modeling_llama import LlamaAttention
        original = LlamaAttention(LlamaConfig(hidden_size=64, num_attention_heads=4, num_key_value_heads=2), 0).eval()
        candidate = INTEGER.FixedTreeAttention(original, "fixed_tree_fp32_v2_pv").eval()
        hidden = torch.ones((1, 1, 64)); before = hidden.clone()
        params = {name: value.detach().clone() for name, value in candidate.named_parameters()}
        position = (torch.ones((1, 1, 16)), torch.zeros((1, 1, 16)))
        for capacity, key_length, mask_length in ((161, 161, 160), (160, 161, 160), (160, 160, 161), (160, 160, 159)):
            cache = mock.Mock(layers=[SimpleNamespace(max_cache_len=capacity, keys=torch.zeros((1, 2, key_length, 16)), values=torch.zeros((1, 2, key_length, 16)))])
            with self.subTest(capacity=capacity, keys=key_length, mask=mask_length), self.assertRaisesRegex(ValueError, "capacity160"):
                candidate(hidden, position, torch.zeros((1, 1, 1, mask_length)), past_key_values=cache)
            cache.update.assert_not_called()
        cache = mock.Mock(layers=[])
        with self.assertRaisesRegex(ValueError, "initialized static cache"):
            candidate(hidden, position, torch.zeros((1, 1, 1, 160)), past_key_values=cache)
        cache.update.assert_not_called()
        for values in (torch.zeros((1, 2, 161, 16)), torch.zeros((1, 2, 160, 16), dtype=torch.bfloat16)):
            cache = mock.Mock(layers=[SimpleNamespace(max_cache_len=160, keys=torch.zeros((1, 2, 160, 16)), values=values)])
            with self.assertRaisesRegex(ValueError, "initialized static cache"):
                candidate(hidden, position, torch.zeros((1, 1, 1, 160)), past_key_values=cache)
            cache.update.assert_not_called()
        for shape in ((2, 2, 160, 16), (1, 3, 160, 16), (1, 2, 160, 17)):
            cache = mock.Mock(layers=[SimpleNamespace(max_cache_len=160, keys=torch.zeros(shape), values=torch.zeros(shape))])
            with self.assertRaisesRegex(ValueError, "initialized static cache"):
                candidate(hidden, position, torch.zeros((1, 1, 1, 160)), past_key_values=cache)
            cache.update.assert_not_called()
        cache = mock.Mock(layers=[])
        with self.assertRaisesRegex(ValueError, "capacity160"):
            candidate(torch.ones((1, 161, 64)), position, torch.zeros((1, 1, 161, 160)), past_key_values=cache)
        cache.update.assert_not_called()
        with self.assertRaisesRegex(ValueError, "capacity160"):
            candidate(hidden, position, torch.zeros((1, 1, 2, 160)), past_key_values=cache)
        cache.update.assert_not_called()
        with self.assertRaisesRegex(ValueError, "matching full-sequence mask"):
            candidate(hidden, position, torch.zeros((1, 1, 1, 2)))
        self.assertTrue(torch.equal(hidden, before))
        for name, value in candidate.named_parameters(): self.assertTrue(torch.equal(value, params[name]))

    def test_v1_attention_matmul_unchanged_after_v2_registration(self):
        from types import SimpleNamespace
        generator = torch.Generator().manual_seed(971)
        q = torch.rand((1, 2, 3, 64), generator=generator)
        k = torch.rand((1, 2, 7, 64), generator=generator)
        v = torch.rand((1, 2, 7, 4), generator=generator)
        mask = torch.zeros((1, 1, 3, 7))
        v1 = SimpleNamespace(num_key_value_groups=1, training=False, arithmetic_policy="fixed_tree_fp32_v1")
        v2 = SimpleNamespace(num_key_value_groups=1, training=False, arithmetic_policy="fixed_tree_fp32_v2_pv")
        expected_weights = torch.softmax(INTEGER.fixed_tree_dot(q, k) * .125 + mask, dim=-1)
        expected_output = torch.matmul(expected_weights, v).transpose(1, 2).contiguous()
        INTEGER.fixed_tree_attention(v2, q, k, v, mask, .125)
        actual, weights = INTEGER.fixed_tree_attention(v1, q, k, v, mask, .125)
        np.testing.assert_array_equal(actual.numpy().view(np.uint32), expected_output.numpy().view(np.uint32))
        np.testing.assert_array_equal(weights.numpy().view(np.uint32), expected_weights.numpy().view(np.uint32))
        self.assertEqual(set(INTEGER.export_options(torch)["custom_translation_table"]), {torch.ops.aten._int_mm.default})

    def test_encoder_oracle_rejects_wrong_codes_scales_dtype_and_nonfinite(self):
        value = np.asarray([[127., .5, 1.5, -2.5], [0., 0., 0., 0.]], dtype=np.float32)
        codes, scales = INTEGER.independent_quantize_rows(value)
        result = INTEGER.check_operand_encoding(value, codes, scales)
        self.assertEqual(result["checked_codes"], 8)
        self.assertEqual(result["checked_scales"], 2)
        wrong_codes = codes.copy()
        wrong_codes[0, 1] = 1
        for wrong in (wrong_codes, np.full_like(codes, -128)):
            with self.assertRaisesRegex(AssertionError, "rounding/clipping"):
                INTEGER.check_operand_encoding(value, wrong, scales)
        with self.assertRaisesRegex(AssertionError, "scale encoding"):
            INTEGER.check_operand_encoding(value, codes, scales * np.float32(2))
        for wrong_codes, wrong_scales in ((codes.astype(np.int32), scales), (codes, scales.astype(np.float64)), (codes[:, :2], scales), (codes, scales[:, 0])):
            with self.assertRaises(ValueError):
                INTEGER.check_operand_encoding(value, wrong_codes, wrong_scales)
        for wrong in (value.astype(np.float64), np.asarray([[np.inf]], dtype=np.float32), np.zeros((0, 3), dtype=np.float32)):
            with self.assertRaises(ValueError):
                INTEGER.independent_quantize_rows(wrong)

    def test_oracle_detects_shared_emitted_code_bug_before_integer_dot(self):
        projection = INTEGER.IntegerProjection(torch.ones(2, 4))
        value = torch.tensor([[127., .5, 1.5, -2.5]])
        for rows, columns in ((0, 7), (3, 0), (True, 7)):
            with self.assertRaisesRegex(ValueError, "positive integer"):
                INTEGER.check_projection(projection, value, rows=rows, columns=columns)
        codes, scales = INTEGER.quantize_rows(value)
        codes[0, 1] = 1
        with mock.patch.object(INTEGER, "quantize_rows", return_value=(codes, scales)), self.assertRaisesRegex(AssertionError, "rounding/clipping"):
            INTEGER.check_projection(projection, value)

    def test_bounded_pair_isolates_boundary_crossing_from_local_encoding(self):
        source = np.asarray([[.9921875, .0664062425494194]], dtype=np.float32)
        actual = np.asarray([[.9921875, .0664062574505806]], dtype=np.float32)
        codes, scales = INTEGER.independent_quantize_rows(actual, "power_of_two")
        result = INTEGER.diagnose_activation_pair(source, actual, actual_codes=codes, actual_scales=scales)
        first = result["first_code_difference"]
        self.assertTrue(result["local_actual_encoding"]["passed"])
        self.assertEqual(first["index"], [0, 1])
        self.assertEqual((first["source_code"], first["actual_code"]), (8, 9))
        self.assertTrue(first["same_scale_crosses_rounding_boundary"])
        self.assertLess(first["source_ratio"], 8.5)
        self.assertGreater(first["actual_ratio"], 8.5)
        with self.assertRaisesRegex(ValueError, "element budget"):
            INTEGER.diagnose_activation_pair(source, actual, maximum_elements=1)
        with self.assertRaisesRegex(ValueError, "supplied together"):
            INTEGER.diagnose_activation_pair(source, actual, actual_codes=codes)
        codes[0, 1] = 8
        with self.assertRaisesRegex(AssertionError, "rounding/clipping"):
            INTEGER.diagnose_activation_pair(source, actual, actual_codes=codes, actual_scales=scales)

    def test_trace_reports_first_float_and_later_first_code_difference(self):
        lm = torch.nn.Module()
        for name in ("first", "second"):
            lm.add_module(name, INTEGER.IntegerProjection(torch.ones(2, 2), "power_of_two"))
        source = {"first": torch.tensor([[.9921875, .0625]]), "second": torch.tensor([[.9921875, .0664062425494194]])}
        actual = {"first": torch.tensor([[.9921875, np.nextafter(np.float32(.0625), np.float32(1))]]), "second": torch.tensor([[.9921875, .0664062574505806]])}
        result = FACTORY.compare_projection_traces(actual, source, lm)
        self.assertEqual(result["first_float_input_difference"], "first")
        self.assertEqual(result["first_quantized_input_difference"]["projection"], "second")
        self.assertEqual(result["checks"][0]["code_mismatches"], 0)
        with self.assertRaisesRegex(AssertionError, "invocation order"):
            FACTORY.compare_projection_traces(dict(reversed(list(actual.items()))), source, lm)

    def test_capture_hooks_remove_on_failure_and_attention_checks_refuse_wrong_state(self):
        projection = INTEGER.IntegerProjection(torch.ones(2, 3), "power_of_two")
        with self.assertRaisesRegex(RuntimeError, "intentional"), FACTORY.capture_last_token_inputs(projection):
            projection(torch.ones(1, 3))
            raise RuntimeError("intentional")
        self.assertEqual(len(projection._forward_pre_hooks), 0)
        source = {"mask": torch.zeros(1, 1, 1, 3), "rope": (torch.ones(1, 1, 2), torch.zeros(1, 1, 2))}
        actual = {"mask": torch.tensor([[[[0., 0., 0., -torch.inf]]]]), "rope": source["rope"]}
        self.assertTrue(FACTORY.compare_cache_attention(actual, source, 3, torch)["active_mask_exact"])
        actual["mask"][..., -1] = 0
        with self.assertRaisesRegex(AssertionError, "unused cache slots"):
            FACTORY.compare_cache_attention(actual, source, 3, torch)
        actual["mask"][..., -1] = -.1
        with self.assertRaisesRegex(AssertionError, "unused cache slots"):
            FACTORY.compare_cache_attention(actual, source, 3, torch)
        actual["mask"][..., -1] = -torch.inf
        actual["rope"] = source["rope"][:1]
        with self.assertRaisesRegex(AssertionError, "RoPE components"):
            FACTORY.compare_cache_attention(actual, source, 3, torch)
        actual["rope"] = (torch.zeros(1, 1, 2), torch.zeros(1, 1, 2))
        with self.assertRaises(AssertionError):
            FACTORY.compare_cache_attention(actual, source, 3, torch)

    def test_actual_integer_projection_matches_independent_numpy(self):
        generator = np.random.default_rng(194)
        weight = generator.uniform(-2, 2, (19, 33)).astype(np.float32)
        value = generator.uniform(-4, 4, (2, 3, 33)).astype(np.float32)
        projection = INTEGER.IntegerProjection(torch.from_numpy(weight))
        actual = projection(torch.from_numpy(value)).numpy()
        q_weight, scale_weight = INTEGER.independent_quantize_rows(weight)
        q_value, scale_value = INTEGER.independent_quantize_rows(value)
        accumulator = q_value.astype(np.int64) @ q_weight.T.astype(np.int64)
        expected = accumulator.astype(np.float32) * scale_value * scale_weight.T
        np.testing.assert_array_equal(projection.weight_kn.numpy(), q_weight.T)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(INTEGER.check_projection(projection, torch.from_numpy(value))["passed"])
        self.assertEqual(projection.weight_kn.dtype, torch.int8)
        self.assertEqual(torch._int_mm(torch.from_numpy(q_value.reshape(-1, 33)),
                                      projection.weight_kn).dtype, torch.int32)

    def test_zero_subnormal_and_ties_have_explicit_semantics(self):
        value = torch.tensor([[0., 0., 0., 0.], [127., .5, 1.5, -2.5],
                              [1e-44, -1e-44, 0., 0.]])
        quantized, scale = INTEGER.quantize_rows(value)
        np_quantized, np_scale = INTEGER.independent_quantize_rows(value.numpy())
        np.testing.assert_array_equal(quantized.numpy(), np_quantized)
        np.testing.assert_array_equal(scale.numpy(), np_scale)
        self.assertEqual(scale[0].item(), 1.)
        self.assertEqual(quantized[1].tolist(), [127, 0, 2, -2])
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertEqual(quantized[2].tolist(), [0, 0, 0, 0])

    def test_power_of_two_scales_match_exact_frexp_oracle_at_dyadic_boundaries(self):
        values = []
        for exponent in (-100, -20, 0, 20, 100):
            exact = np.float32(2. ** exponent)
            for raw in (np.nextafter(exact, np.float32(0)), exact,
                        np.nextafter(exact, np.float32(np.inf))):
                maximum = np.float32(raw * np.float32(127))
                values.append([maximum, maximum * np.float32(.5), -maximum * np.float32(.5)])
        values.extend([[0., 0., 0.], [1e-44, -1e-44, 0.], [254., 1., 3.]])
        value = torch.from_numpy(np.asarray(values, dtype=np.float32))
        quantized, scale = INTEGER.quantize_rows(value, "power_of_two")
        expected_q, expected_scale = INTEGER.independent_quantize_rows(value.numpy(), "power_of_two")
        np.testing.assert_array_equal(scale.numpy(), expected_scale)
        np.testing.assert_array_equal(quantized.numpy(), expected_q)
        self.assertEqual(quantized[-1].tolist(), [127, 0, 2])
        self.assertEqual(scale[-1].item(), 2.)
        self.assertEqual(scale[-3].item(), 1.)
        self.assertTrue(bool(torch.isfinite(scale).all()))
        self.assertEqual(INTEGER.recipe()["name"], "tinyllama_w8a8_dynamic_token_v1")
        self.assertNotEqual(INTEGER.recipe("power_of_two")["name"], INTEGER.recipe()["name"])

    def test_power_of_two_projection_matches_complete_integer_and_float_oracles(self):
        generator = np.random.default_rng(771)
        weight = generator.uniform(-2, 2, (19, 33)).astype(np.float32)
        value = generator.uniform(-4, 4, (2, 3, 33)).astype(np.float32)
        projection = INTEGER.IntegerProjection(torch.from_numpy(weight), activation_scale="power_of_two")
        q_weight, scale_weight = INTEGER.independent_quantize_rows(weight)
        q_value, scale_value = INTEGER.independent_quantize_rows(value, "power_of_two")
        expected = (q_value.astype(np.int64) @ q_weight.T.astype(np.int64)).astype(np.float32) * scale_value * scale_weight.T
        np.testing.assert_array_equal(projection(torch.from_numpy(value)).numpy(), expected)
        self.assertTrue(INTEGER.check_projection(projection, torch.from_numpy(value))["passed"])

    def test_vector_projection_is_preserved(self):
        projection = INTEGER.IntegerProjection(torch.tensor([[1., -2., 0.], [-1., 0., 3.]]))
        output = projection(torch.tensor([1., 2., -3.]))
        self.assertEqual(tuple(output.shape), (2,))

    def test_unsupported_and_nonfinite_inputs_are_rejected(self):
        for weight in (torch.ones(2, 3, dtype=torch.bfloat16), torch.ones(2, 131072),
                       torch.full((2, 3), float("nan"))):
            with self.subTest(shape=weight.shape, dtype=weight.dtype), self.assertRaises(ValueError):
                INTEGER.IntegerProjection(weight)
        projection = INTEGER.IntegerProjection(torch.ones(2, 3))
        for value in (torch.ones(1, 3, dtype=torch.bfloat16), torch.ones(1, 4),
                      torch.full((1, 3), float("inf"))):
            with self.subTest(shape=value.shape, dtype=value.dtype), self.assertRaises(ValueError):
                projection(value)
        with self.assertRaisesRegex(ValueError, "exactly 155"):
            INTEGER.install_projections(torch.nn.Linear(3, 2, bias=False))

    def test_operation_and_memory_plan_exposes_pending_target_terms(self):
        projection = INTEGER.IntegerProjection(torch.ones(2, 3))
        record = {"packed_weight": INTEGER.identity(projection.weight_kn),
                  "weight_scale": INTEGER.identity(projection.weight_scale)}
        plan = INTEGER.memory_plan(projection, [record], 128, 32)
        self.assertEqual(plan["one_kv_state_bytes"], 2 * 22 * 4 * 160 * 64 * 4)
        self.assertEqual(plan["io_bytes"]["simultaneous_kv_input_output"], 2 * plan["one_kv_state_bytes"])
        self.assertEqual(plan["constants_duplicated_two_stage_bytes"], 2 * plan["constants_shared_once_bytes"])
        self.assertFalse(plan["stage_constant_sharing_implemented"])
        self.assertIn("graph_workspace", plan["missing_peak_terms"])
        placement = INTEGER.operation_placement()
        self.assertEqual(placement["learned_projections"]["count"], 155)
        for name in ("qk_contractions", "pv_contractions", "rope", "softmax"):
            self.assertEqual(placement[name]["placement"], "cpu")

    def test_language_corpus_minimum_unique_tokens_and_frozen_contract(self):
        class Tokenizer:
            def __call__(self, text, **_kwargs):
                number = int(text)
                return {"input_ids": torch.arange(128).reshape(1, -1) + number}
        documents = [{"id": str(index), "text": str(index), "source": "unit-test-fixture"}
                     for index in range(8)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "corpus.json"
            payload = {"source": "unit-test-fixture-never-quality-evidence", "selection": "all",
                       "documents": documents}
            path.write_text(json.dumps(payload))
            tokens, contract = INTEGER.language_inputs(path, Tokenizer(), FACTORY.safe_path, FACTORY.file_identity)
            self.assertEqual(len(tokens), 8)
            self.assertEqual(contract["minimum_documents"], 8)
            self.assertEqual(len(contract["sha256"]), 64)
            contract["documents"][0]["id"] = "modified"
            with self.assertRaisesRegex(ValueError, "contract changed"):
                INTEGER.language_quality(None, tokens, [{}] * 8, contract)
            documents[-1]["text"] = documents[0]["text"]
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "distinct selected token"):
                INTEGER.language_inputs(path, Tokenizer(), FACTORY.safe_path, FACTORY.file_identity)

    def test_wikitext_preparation_rejects_nonpinned_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            artifact = Path(directory) / "source.parquet"
            artifact.write_text("fixture; never a dataset")
            with self.assertRaisesRegex(ValueError, "pinned source bytes"):
                INTEGER.prepare_wikitext_corpus(artifact, "/missing", Path(directory) / "prepared")
            self.assertFalse((Path(directory) / "prepared").exists())

    def test_frozen_language_thresholds_reject_likelihood_or_token_loss(self):
        documents = [{"id": str(index)} for index in range(8)]
        contract = {**INTEGER.QUALITY_CONTRACT, "documents": documents}
        import hashlib
        contract["sha256"] = hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()
        tokens = [torch.zeros((1, 128), dtype=torch.int64) for _ in range(8)]
        reference = [{"nll": torch.ones(127), "top1": torch.zeros((1, 127), dtype=torch.int64)}
                     for _ in range(8)]
        for increase, disagree, passed in ((.1, 0, True), (.3, 0, False), (.1, 40, False)):
            candidate = []
            for _ in range(8):
                top1 = torch.zeros((1, 127), dtype=torch.int64)
                top1[:, :disagree] = 1
                candidate.append({"nll": torch.full((127,), 1. + increase), "top1": top1})
            with self.subTest(increase=increase, disagree=disagree), mock.patch.object(
                    INTEGER, "language_reference", return_value=candidate):
                result = INTEGER.language_quality(None, tokens, reference, contract)
                self.assertEqual(result["passed"], passed)
                self.assertEqual(result["predicted_tokens"], 8 * 127)
                self.assertFalse(result["paper_language_quality"])

    @unittest.skipUnless(os.environ.get("TVM_TINYLLAMA_INTEGER_ONNX") == "1", "Optional ONNX/TVM integration disabled")
    def test_compiled_emitted_activation_codes_and_scales_match_independent_oracle(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        class Probe(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = INTEGER.IntegerProjection(torch.tensor([[1., -2., 3., 4.], [-1., 0., 2., 3.]]), "power_of_two")
            def forward(self, value):
                codes, scales = self.projection.integer_inputs(value)
                return self.projection(value), codes, scales
        values = np.asarray([[254., 1., 3., -5.], [0., 0., 0., 0.], [.9921875, .0664062425494194, .0664062574505806, -.0664062574505806]], dtype=np.float32)
        probe = Probe().eval()
        with tempfile.TemporaryDirectory() as directory:
            compiler = verify_session.StageCompiler(Path(directory), "optimized", np, torch, onnx, tvm,
                                                     exporter="dynamo", export_options=INTEGER.export_options(torch))
            executable = compiler(SimpleNamespace(name="emitted_operands", module=probe), [values], 3)
            _result, codes, scales = verify_session.outputs(executable([values]), np, torch)
            checked = INTEGER.check_operand_encoding(values, codes, scales, "power_of_two")
            self.assertEqual(checked["checked_codes"], values.size)
            self.assertEqual(codes[2].tolist(), [127, 8, 9, -9])

    def hf_cache_fixture(self, arithmetic_policy=None):
        from transformers import LlamaConfig, LlamaForCausalLM
        from transformers.models.llama.modeling_llama import LlamaRMSNorm, LlamaAttention
        from types import SimpleNamespace
        path = FACTORY.safe_path(Path(os.environ["TVM_SESSION_MODEL2MLIR"]) / "m2m/capture/causal_session.py")
        spec = importlib.util.spec_from_file_location("tinyllama_test_causal_session", path)
        causal = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(causal)
        torch.manual_seed(194)
        config = LlamaConfig(hidden_size=64, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, vocab_size=128)
        config._attn_implementation = "eager"
        lm = LlamaForCausalLM(config).eval().requires_grad_(False)
        provenance = {}
        if arithmetic_policy:
            provenance["arithmetic_policy"] = INTEGER.arithmetic_contract(arithmetic_policy)
            attention_options = {} if arithmetic_policy == "fixed_tree_fp32_v1" else {"arithmetic_policy": arithmetic_policy}
            for name, module in list(lm.named_modules()):
                if isinstance(module, (LlamaRMSNorm, LlamaAttention)):
                    parent, _, leaf = name.rpartition(".")
                    replacement = INTEGER.FixedTreeRMSNorm(module) if isinstance(module, LlamaRMSNorm) else INTEGER.FixedTreeAttention(module, **attention_options)
                    setattr(lm.get_submodule(parent), leaf, replacement)
        capture = SimpleNamespace(lm=lm, tokens=torch.arange(12).reshape(1, -1), prefill_tokens=4, decode_tokens=8, provenance=provenance)
        return causal, capture

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODEL2MLIR"), "Optional pinned cache-wrapper integration disabled")
    def test_real_hf_fixed_tree_cache_dispatch_and_unchanged_gate(self):
        for policy in ("fixed_tree_fp32_v1", "fixed_tree_fp32_v2_pv"):
            with self.subTest(policy=policy):
                causal, capture = self.hf_cache_fixture(policy)
                with mock.patch.object(INTEGER, "fixed_tree_dot", wraps=INTEGER.fixed_tree_dot) as observed:
                    result = FACTORY.validate_reference_stages(capture, causal, torch)
                if policy == "fixed_tree_fp32_v1":
                    self.assertEqual(observed.call_count, 36)
                else:
                    pv_calls = [call for call in observed.call_args_list if call.args[0].shape[-1] == 256]
                    self.assertTrue(pv_calls)
                    self.assertTrue(all(call.args[0].shape[-1] == call.args[1].shape[-1] == 256 for call in pv_calls))
                    self.assertTrue({1, 4, 12}.issubset({call.args[0].shape[-2] for call in pv_calls}))
                self.assertEqual(result["reference"], "candidate_full_sequence_hf_use_cache_false")
                self.assertEqual((result["rtol"], result["atol"]), (1e-4, 1e-4))
                self.assertEqual(len(result["checks"]), 9)
                self.assertTrue(all(check["attention"]["active_mask_exact"] and check["attention"]["last_token_rope_exact"] for check in result["checks"]))

    @unittest.skipUnless(os.environ.get("TVM_SESSION_MODEL2MLIR"), "Optional pinned cache-wrapper integration disabled")
    def test_real_hf_fixed_cache_matches_positions_masks_and_full_sequence(self):
        causal, capture = self.hf_cache_fixture()
        lm = capture.lm
        result = FACTORY.validate_reference_stages(capture, causal, torch)
        self.assertTrue(result["passed"])
        self.assertEqual(len(result["checks"]), 9)
        self.assertTrue(all(check["attention"]["active_mask_exact"] and check["attention"]["last_token_rope_exact"] for check in result["checks"]))
        self.assertEqual(len(lm.model.layers[0].self_attn._forward_pre_hooks), 0)
        original_decode = causal.FixedCacheDecodeStep.forward
        def wrong_decode(stage, *args):
            logits, cache, position = original_decode(stage, *args)
            return logits + .01, cache, position
        with mock.patch.object(causal.FixedCacheDecodeStep, "forward", wrong_decode), self.assertRaisesRegex(AssertionError, "cache gate failed at 5"):
            FACTORY.validate_reference_stages(capture, causal, torch)
        diagnostic = capture.provenance["functional_cache_reference"]
        self.assertFalse(diagnostic["passed"])
        self.assertEqual((diagnostic["rtol"], diagnostic["atol"]), (1e-4, 1e-4))
        self.assertTrue(diagnostic["attention"]["active_mask_exact"])
        self.assertEqual(len(lm.model.layers[0].self_attn._forward_pre_hooks), 0)

    @unittest.skipUnless(os.environ.get("TVM_TINYLLAMA_INTEGER_ONNX") == "1", "Optional ONNX/TVM integration disabled")
    def test_real_integer_export_import_and_llvm_execution(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        generator = np.random.default_rng(21)
        weight = generator.uniform(-2, 2, (19, 33)).astype(np.float32)
        value = generator.uniform(-4, 4, (2, 3, 33)).astype(np.float32)
        value[0, 0] = 0
        value[0, 1, :3] = [254., 1., 3.]
        for mode in ("maxabs", "power_of_two"):
            projection = INTEGER.IntegerProjection(torch.from_numpy(weight), activation_scale=mode).eval()
            with self.subTest(activation_scale=mode), tempfile.TemporaryDirectory() as directory:
                compiler = verify_session.StageCompiler(Path(directory), "optimized", np, torch, onnx, tvm,
                                                         exporter="dynamo", export_options=INTEGER.export_options(torch))
                executable = compiler(SimpleNamespace(name="projection", module=projection), [value], 1)
                actual = verify_session.outputs(executable([value]), np, torch)[0]
                np.testing.assert_allclose(actual, projection(torch.from_numpy(value)).numpy(), rtol=1e-4, atol=1e-4)
                graph = onnx.load(str(Path(directory) / "stage_0/model.onnx"))
                contractions = [node for node in graph.graph.node if node.op_type == "MatMulInteger"]
                self.assertEqual(len(contractions), 1)
                weights = {tensor.name: tensor for tensor in graph.graph.initializer}
                self.assertEqual(weights[contractions[0].input[1]].data_type, onnx.TensorProto.INT8)

    @unittest.skipUnless(os.environ.get("TVM_TINYLLAMA_FIXED_TREE_ONNX") == "1", "Optional ordered FP32 ONNX/TVM integration disabled")
    def test_real_fixed_tree_export_import_and_llvm_execution(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        class Ordered(torch.nn.Module):
            def forward(self, left, right, value):
                return INTEGER.fixed_tree_dot(left, right), INTEGER.fixed_tree_mean_squares(value)
        generator = np.random.default_rng(981)
        left = generator.uniform(-2, 2, (1, 2, 5, 64)).astype(np.float32)
        right = generator.uniform(-2, 2, (1, 2, 7, 64)).astype(np.float32)
        value = generator.uniform(-1, 1, (1, 3, 2048)).astype(np.float32)
        candidate = SimpleNamespace(provenance={"arithmetic_policy": INTEGER.arithmetic_contract("fixed_tree_fp32_v1")})
        options = verify_session.factory_export_options(FACTORY, torch, candidate)
        self.assertEqual(len(options["custom_translation_table"]), 3)
        # A prior candidate must not opt a later native model into ordered export.
        native = SimpleNamespace(provenance={})
        native_options = verify_session.factory_export_options(FACTORY, torch, native)
        self.assertEqual(set(native_options["custom_translation_table"]), {torch.ops.aten._int_mm.default})
        self.assertEqual(set(INTEGER.export_options(torch)["custom_translation_table"]), {torch.ops.aten._int_mm.default})
        legacy = SimpleNamespace(get_onnx_export_options=lambda selected: {"custom_translation_table": {}})
        self.assertEqual(verify_session.factory_export_options(legacy, torch, candidate), {"custom_translation_table": {}})
        self.assertIsNone(verify_session.factory_export_options(SimpleNamespace(), torch, candidate))
        expected_dot = self.scalar_tree_dot(left, right)
        expected_mean = self.scalar_tree_dot(value.reshape(1, 1, 3, 2048), value.reshape(1, 1, 3, 2048))[0, 0].diagonal().reshape(1, 3, 1) * np.float32(1 / 2048)
        for mode in ("baseline", "optimized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                compiler = verify_session.StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", export_options=options)
                executable = compiler(SimpleNamespace(name="ordered", module=Ordered().eval()), [left, right, value], 1)
                actual_dot, actual_mean = verify_session.outputs(executable([left, right, value]), np, torch)
                np.testing.assert_array_equal(actual_dot.view(np.uint32), expected_dot.view(np.uint32))
                np.testing.assert_array_equal(actual_mean.view(np.uint32), expected_mean.view(np.uint32))
                graph = onnx.load(str(Path(directory) / "stage_0/model.onnx"))
                self.assertEqual([(node.domain, node.op_type) for node in graph.graph.node], [("merlin.fixed_order", "FixedTreeDot"), ("merlin.fixed_order", "FixedTreeMeanSquares")])
                self.assertEqual(next(item.version for item in graph.opset_import if item.domain == "merlin.fixed_order"), 1)

    @unittest.skipUnless(os.environ.get("TVM_TINYLLAMA_FIXED_TREE_ONNX") == "1", "Optional ordered FP32 ONNX/TVM integration disabled")
    def test_v2_pv_real_export_import_preserves_declared_tree_and_raw_oracle(self):
        import onnx
        import tvm
        import verify_session
        from types import SimpleNamespace
        class OrderedPV(torch.nn.Module):
            def forward(self, probabilities, values):
                return INTEGER.fixed_tree_pv(probabilities, values)
        generator = np.random.default_rng(261012)
        p = generator.uniform(0, 1, (1, 2, 3, 11)).astype(np.float32)
        v = generator.uniform(-1, 1, (1, 2, 11, 4)).astype(np.float32)
        expected = self.scalar_tree_dot(np.pad(p, [(0, 0)] * 3 + [(0, 245)]), np.pad(v.transpose(0, 1, 3, 2), [(0, 0)] * 3 + [(0, 245)]))
        candidate = SimpleNamespace(provenance={"arithmetic_policy": INTEGER.arithmetic_contract("fixed_tree_fp32_v2_pv")})
        options = verify_session.factory_export_options(FACTORY, torch, candidate)
        self.assertEqual(len(options["custom_translation_table"]), 3)
        for mode in ("baseline", "optimized"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                compiler = verify_session.StageCompiler(Path(directory), mode, np, torch, onnx, tvm, exporter="dynamo", export_options=options)
                executable = compiler(SimpleNamespace(name="ordered_pv", module=OrderedPV().eval()), [p, v], 1)
                for repeat in range(2):
                    actual = verify_session.outputs(executable([p, v]), np, torch)[0]
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                graph = onnx.load(str(Path(directory) / "stage_0/model.onnx"))
                self.assertEqual(sum(node.op_type == "Pad" for node in graph.graph.node), 2)
                self.assertEqual([(node.domain, node.op_type) for node in graph.graph.node if node.domain], [("merlin.fixed_order", "FixedTreeDot")])
                self.assertEqual(next(item.version for item in graph.opset_import if item.domain == "merlin.fixed_order"), 1)


if __name__ == "__main__":
    unittest.main()
