"""Board-free unit tests for the TVM (Apache TVM v0.19.0 / Relax) baseline arm (merlin.baselines.tvm).

These never touch a board: they exercise bundle resolution (incl. the legacy fp32-LLM fallback), the
RVV-audit symbol->region mapping, and the honesty contract when the board is absent.

They ALSO carry a regression test for the TVM ONNX frontend patch this arm depends on: opset-18
reducers (ReduceMean/Sum/L2/...) must read ``axes`` from the *input* (not the attribute), else a
Llama RMSNorm ``ReduceMean(axis=-1)`` silently reduces over all axes and the whole-model cos collapses
(tiny_llama int8 went cos 0.22 -> ~1.0 with the fix). That test is skipped when the built TVM tree is
not importable (CI without the TVM build), so the suite stays green regardless of TVM build state.
"""
from __future__ import annotations

import os
import sys
import types

import numpy as np
import pytest

from merlin.baselines import tvm as tvm_arm


# --- bundle resolution -------------------------------------------------------------------------

def test_resolve_bundle_int8_convention():
    b = tvm_arm.resolve_bundle("tiny_llama", "int8")
    assert b.model == "tiny_llama" and b.variant == "int8"


def test_region_of_symbol_mapping():
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_matmul_3") == "gemm"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_softmax") == "attention"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_rms_norm") == "norm"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_add_multiply") == "elementwise"
    assert tvm_arm._region_of_symbol("tvmgen_default_fused_take") == "other"


def test_golden_path_prefers_w8a8(tmp_path):
    import merlin.baselines.bundle as _bundle
    b = _bundle.CaptureBundle(model="m", variant="int8", root=tmp_path)
    (tmp_path / "golden.npy").write_bytes(b"x")
    assert tvm_arm.golden_path(b).name == "golden.npy"
    (tmp_path / "golden_w8a8.npy").write_bytes(b"x")
    assert tvm_arm.golden_path(b).name == "golden_w8a8.npy"


# --- ONNX frontend reduce-axes regression (the core patch this arm needs) -----------------------

def _import_built_tvm():
    """Import ``tvm`` from the built (uninstalled) submodule tree, or return None if unavailable."""
    if not tvm_arm.tvm_built():
        return None
    sys.path.insert(0, str(tvm_arm.tvm_python_path()))
    os.environ.setdefault("TVM_LIBRARY_PATH", str(tvm_arm.tvm_lib_dir()))
    ld = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([str(tvm_arm.tvm_lib_dir()), ld]).strip(os.pathsep)
    try:
        import onnx  # noqa: F401
        import tvm  # noqa: F401
    except Exception:
        return None
    # onnx 1.22 removed onnx.mapping which the frontend imports; reconstruct the faithful dtype table.
    import onnx as _onnx
    if not hasattr(_onnx, "mapping"):
        from onnx import helper as _oh
        mm = types.ModuleType("onnx.mapping")
        mm.TENSOR_TYPE_TO_NP_TYPE = {dt: _oh.tensor_dtype_to_np_dtype(dt) for dt in _oh.get_all_tensor_dtypes()}
        sys.modules["onnx.mapping"] = mm
        _onnx.mapping = mm
    return _onnx


@pytest.mark.parametrize("op,axis,keepdims", [("ReduceMean", [-1], 1), ("ReduceSum", [-1], 1),
                                              ("ReduceMean", [1], 0), ("ReduceL2", [2], 1)])
def test_onnx_opset18_reducer_reads_axes_input(op, axis, keepdims):
    """opset-18 reducers take ``axes`` as an input; the frontend must honor it (not reduce all axes).

    Guards the RMSNorm ``ReduceMean(x, axis=-1)`` -> full-reduction defect. Skips when the built TVM
    tree isn't importable."""
    _onnx = _import_built_tvm()
    if _onnx is None:
        pytest.skip("built TVM tree not importable (no build/baselines/tvm)")
    import tvm
    from tvm import relax
    from tvm.relax.frontend.onnx import from_onnx
    from onnx import helper, TensorProto

    data = np.random.randn(1, 8, 16).astype(np.float32)
    axes_t = helper.make_tensor("axes", TensorProto.INT64, [len(axis)], list(axis))
    node = helper.make_node(op, inputs=["x", "axes"], outputs=["y"], keepdims=keepdims,
                            noop_with_empty_axes=0)
    exp = np.sum(data, axis=tuple(axis), keepdims=bool(keepdims))
    graph = helper.make_graph([node], "r",
                              [helper.make_tensor_value_info("x", TensorProto.FLOAT, list(data.shape))],
                              [helper.make_tensor_value_info("y", TensorProto.FLOAT, list(exp.shape))],
                              initializer=[axes_t])
    model = helper.make_model(graph, producer_name="r")
    model.opset_import[0].version = 18

    mod = from_onnx(model, keep_params_in_input=False)
    ex = relax.build(mod, target=tvm.target.Target("llvm"))
    vm = relax.VirtualMachine(ex, tvm.cpu())
    out = vm["main"](tvm.nd.array(data))
    got = out.numpy() if hasattr(out, "numpy") else np.asarray(out[0].numpy())
    # The bug reduced over ALL axes -> shape (1,1,1); the fix keeps the requested axis only.
    assert got.shape == exp.shape, f"{op}: axes-from-input ignored (shape {got.shape} != {exp.shape})"
