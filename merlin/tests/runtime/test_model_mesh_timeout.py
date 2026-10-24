"""The requested model timeout must reach real mesh dispatch, not only tile checks."""

from types import SimpleNamespace

import numpy as np
import pytest

from merlin import compile_cli
from merlin.compile import mesh
from merlin.mining import registry
from merlin.runtime import dispatch_runtime
from merlin.targetgen import corpus_spec, oracle_policy


@pytest.fixture
def mesh_model(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(
        """builtin.module {
  func.func @forward(%a: tensor<1x1xi32>, %b: tensor<1x1xi32>,
                     %c: tensor<1x1xi32>) -> tensor<1x1xi32> {
    %r = linalg.matmul ins(%a, %b : tensor<1x1xi32>, tensor<1x1xi32>)
         outs(%c : tensor<1x1xi32>) -> tensor<1x1xi32>
    func.return %r : tensor<1x1xi32>
  }
}
"""
    )
    np.save(bundle / "golden.npy", np.array([[6]], dtype=np.float32))
    arrays = [np.array([[2]], dtype=np.int32), np.array([[3]], dtype=np.int32), np.zeros((1, 1), dtype=np.int32)]
    monkeypatch.setattr(dispatch_runtime, "resolve_forward_args", lambda _: arrays)
    binding = SimpleNamespace(
        integer=True,
        operand_dtype="i32",
        accum_dtype="i32",
        tile_dim=1,
        subnormal_operand_flush=False,
        mlir_dtype=lambda token: token,
        cap_dtype=lambda token: token,
    )
    monkeypatch.setattr(dispatch_runtime, "mesh_datapath", lambda *_a, **_k: binding)
    monkeypatch.setattr(mesh, "_mesh_tile_binding", lambda *_a, **_k: binding)
    monkeypatch.setattr(oracle_policy, "selected_endpoint", lambda _: (None, None))
    monkeypatch.setattr(oracle_policy, "selected_sim_via", lambda _: None)
    monkeypatch.setattr(corpus_spec, "build", lambda *_: (None, "lowered interface"))
    monkeypatch.setattr(registry, "load_rvv_package", lambda _: SimpleNamespace(is_int8=False))
    seen = []

    def oracle(_target, _interface, a, b, **options):
        seen.append(options["timeout"])
        return (np.asarray(a) @ np.asarray(b)).tolist()

    monkeypatch.setattr(mesh, "_matmul_via_oot_cert", oracle)
    return bundle, seen


@pytest.mark.parametrize("timeout", [37, 1800])
def test_model_timeout_reaches_actual_operand_oracle(mesh_model, timeout):
    bundle, seen = mesh_model
    result = compile_cli.compile_rvv(
        "fixture",
        "fp32",
        run="host",
        verify=False,
        package="selected-host-package",
        auto_capture=False,
        timeout=timeout,
        bundle_path=bundle,
        kernel_backend="mesh",
        mesh_target="example",
        mesh_package="selected-accelerator-package",
    )
    assert seen == [timeout]
    assert result["mesh_execution"]["matmul_layers_on_mesh"] == 1
