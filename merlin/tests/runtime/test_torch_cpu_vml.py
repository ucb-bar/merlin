"""Source-gated host parity for the captured TorchAO rotary trig kernels."""

from __future__ import annotations

import numpy as np
import pytest

from merlin.common import mlir_query as mq
from merlin.runtime.torch_cpu_vml import TORCH_COMMIT, TorchVmlUnavailable, classify_kernel, evaluate, load_attested_vml
from merlin.xdsl_dialects._common import text as mlir_text
from merlin.xdsl_dialects.lowering.outline import outline_dispatches


def _module(kind: str, *, provenance: str | None = None, input_map: str = "(d0, d1) -> (d0, d1)"):
    from xdsl.dialects.builtin import StringAttr

    aten = provenance if provenance is not None else f"aten.{kind}.default"
    module = mq.parse(
        f"""builtin.module {{
  func.func @forward(%x: tensor<2x17xf32>) -> tensor<2x17xf32> {{
    %empty = tensor.empty() : tensor<2x17xf32>
    %result = linalg.generic {{indexing_maps = [affine_map<{input_map}>,
                                               affine_map<(d0, d1) -> (d0, d1)>],
                               iterator_types = ["parallel", "parallel"]}}
      ins(%x : tensor<2x17xf32>) outs(%empty : tensor<2x17xf32>) {{
      ^bb0(%value: f32, %old: f32):
        %v = math.{kind} %value : f32
        linalg.yield %v : f32
    }} -> tensor<2x17xf32>
    func.return %result : tensor<2x17xf32>
  }}
}}"""
    )
    module.attributes["prov.quantization"] = StringAttr("int8_dyn_act_int8_weight")
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    generic.attributes["prov.op"] = StringAttr(kind)
    generic.attributes["prov.aten"] = StringAttr(aten)
    return module


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_classifier_requires_exact_aten_provenance_and_identity_layout(kind):
    def route(module):
        kfn = next(op for op in module.walk() if op.name == "func.func")
        return classify_kernel(kfn)

    assert route(_module(kind)) == {"input": 0, "kind": kind}
    assert route(_module(kind, provenance="aten.tanh.default")) is None
    assert route(_module(kind, input_map="(d0, d1) -> (d1, d0)")) is None


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_attested_vml_matches_torch_on_independent_values(kind):
    try:
        lib, _ = load_attested_vml()
    except TorchVmlUnavailable as error:
        pytest.skip(str(error))
    import torch

    generator = np.random.default_rng(20261106)
    cases = [
        np.arange(17, dtype=np.float32)[:, None] * np.float32(1 / 17),
        generator.uniform(-100, 100, 1024).astype(np.float32),
        generator.normal(size=257).astype(np.float32),
        np.array([0, 1, 2, 3, 0x007FFFFF, 0x00800000, 0x7F7FFFFF, 0x7F800000, 0x80000001], np.uint32).view(np.float32),
    ]
    for values in cases:
        actual = evaluate(lib, kind, values)
        expected = getattr(torch.from_numpy(values.copy()), kind)().numpy()
        np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))


def test_vml_host_switch_leaves_rv64_lowering_input_unchanged(monkeypatch):
    from merlin.llvmlower.passes_xdsl import preprocess_text

    source = mlir_text(_module("cos"))
    baseline = preprocess_text(source)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_VML_TRIG", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")
    selected = preprocess_text(source)
    assert selected == baseline
    assert "vmsCos" not in selected[0]


def test_vml_host_switch_leaves_rv64_object_unchanged(tmp_path, monkeypatch):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model

    install = toolchain.llvm_install() / "bin"
    tools_present = (
        toolchain.clang().is_file() and (install / "mlir-opt").is_file() and (install / "mlir-translate").is_file()
    )
    if not tools_present:
        pytest.skip("RV64 toolchain unavailable")
    source = mlir_text(_module("cos"))
    baseline = lower_model(source, tmp_path / "baseline", targets=("riscv",))
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_VML_TRIG", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")
    selected = lower_model(source, tmp_path / "selected", targets=("riscv",))
    assert baseline.ll_path.read_bytes() == selected.ll_path.read_bytes()
    assert baseline.riscv_obj.read_bytes() == selected.riscv_obj.read_bytes()


def test_vml_refuses_unattested_library_hash(monkeypatch):
    from merlin.runtime import torch_cpu_vml

    try:
        load_attested_vml()
    except TorchVmlUnavailable as error:
        pytest.skip(str(error))
    monkeypatch.setattr(torch_cpu_vml, "LIBTORCH_CPU_SHA256", "0" * 64)
    with pytest.raises(TorchVmlUnavailable, match="bytes differ"):
        load_attested_vml()


@pytest.mark.parametrize("kind", ["sin", "cos"])
def test_host_dispatch_routes_trig_and_records_library(kind, tmp_path, monkeypatch):
    try:
        _, digest = load_attested_vml()
    except TorchVmlUnavailable as error:
        pytest.skip(str(error))
    import torch

    from merlin.runtime.dispatch_runtime import DispatchRuntimeError, execute

    module = _module(kind)
    outlined = outline_dispatches(module)
    values = np.arange(34, dtype=np.float32).reshape(2, 17) * np.float32(0.137)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_VML_TRIG", "1")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX2")
    counters = {}
    actual = execute(outlined, [values], tmp_path / "host", counters=counters)[0]
    expected = getattr(torch.from_numpy(values.copy()), kind)().numpy()
    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    assert counters["torch_cpu_vml_trig_routed"] == counters["torch_cpu_vml_trig_ran"] == 1
    assert counters["dispatch_ledger"][0]["lane"] == "torch_cpu_vml"
    assert counters["dispatch_ledger"][0]["libtorch_cpu_sha256"] == digest
    assert counters["dispatch_ledger"][0]["target_executed"] is False

    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", "wrong-source")
    with pytest.raises(DispatchRuntimeError, match="exact source"):
        execute(outlined, [values], tmp_path / "refused")
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_SOURCE_COMMIT", TORCH_COMMIT)
    monkeypatch.setenv("MERLIN_TORCHAO_CPU_CAPABILITY", "AVX512")
    with pytest.raises(DispatchRuntimeError, match="AVX2"):
        execute(outlined, [values], tmp_path / "refused_cpu")
